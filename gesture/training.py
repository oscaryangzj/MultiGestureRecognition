import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gesture.config import project_path
from gesture.data import load_state_split
from gesture.features import FEATURE_SIZE
from gesture.models import StateMLP


def confusion_matrix(targets, predictions, num_classes):
    result = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        result[int(target), int(prediction)] += 1
    return result


def metrics_from_predictions(targets, predictions, classes):
    matrix = confusion_matrix(targets, predictions, len(classes))
    per_class = {}
    for index, name in enumerate(classes):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        per_class[name] = {
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / support if support else 0.0,
            "support": support,
        }
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
        "per_class": per_class,
    }


def evaluate_state_model(model, features, labels, batch_size, classes):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)),
        batch_size=batch_size,
        shuffle=False,
    )
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            logits = model(batch_features)
            total_loss += float(loss_fn(logits, batch_labels).item()) * len(batch_labels)
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    result = metrics_from_predictions(labels, np.asarray(predictions), classes)
    result["loss"] = total_loss / len(labels) if len(labels) else 0.0
    return result


def train_state(config):
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    classes = list(config["labels"]["state_classes"])
    train_x, train_y, train_sessions = load_state_split(config, "train")
    val_x, val_y, val_sessions = load_state_split(config, "val")
    for split_name, labels in (("train", train_y), ("val", val_y)):
        missing = [name for index, name in enumerate(classes) if index not in labels]
        if missing:
            raise ValueError(f"{split_name} split is missing State classes: {missing}")
    hidden_size = int(config["models"]["state_hidden_size"])
    batch_size = int(config["training"]["batch_size"])

    model = StateMLP(FEATURE_SIZE, hidden_size, len(classes))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )

    output_dir = project_path("runs/state")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    best_val_loss = float("inf")
    history = []

    for epoch in range(int(config["training"]["max_epochs"])):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_labels)

        val_metrics = evaluate_state_model(model, val_x, val_y, batch_size, classes)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_y),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "input_size": FEATURE_SIZE,
                    "hidden_size": hidden_size,
                },
                checkpoint_path,
            )
        print(
            f"epoch {epoch + 1}: train_loss={history[-1]['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.3f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    val_metrics = evaluate_state_model(model, val_x, val_y, batch_size, classes)
    result = {
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "history": history,
        "validation": val_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return checkpoint_path, result
