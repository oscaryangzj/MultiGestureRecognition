import json
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from gesture.action_data import (ActionWindows, action_pulse_frames, collate_action_windows,
                                 load_action_split, mirror_action_targets)
from gesture.action_evaluation import action_event_metrics, action_failure_cases, aggregate_action_metrics
from gesture.action_features import action_feature_size, action_mirror_motion_indices
from gesture.action_inference import action_output_directory, predict_action_session, save_action_predictions
from gesture.action_decoding import events_on_source_frames, write_action_events, write_final_events
from gesture.models import ActionCNNLSTM
from gesture.waving_fsm import trace_waving_session


def epoch_loss(model, loader, device, positive_weight=None, optimizer=None, gradient_clip=None):
    model.train(optimizer is not None)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=positive_weight)
    total_loss, total_labels = 0.0, 0
    for features, targets, mask in loader:
        features, targets, mask = features.to(device), targets.to(device), mask.to(device)
        count = int(mask.sum().item())
        if not count:
            continue
        with torch.set_grad_enabled(optimizer is not None):
            loss = (loss_fn(model(features), targets) * mask).sum() / count
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        total_loss += float(loss.item()) * count
        total_labels += count
    if not total_labels:
        raise ValueError("Action windows have no supervised targets")
    return total_loss / total_labels


def action_normalization(sessions, config):
    features = np.concatenate([session["features"][session["valid"] & session["mask"].any(axis=1)]
                               for session in sessions])
    mean = features.mean(axis=0, dtype=np.float64)
    variance = features.var(axis=0, dtype=np.float64)
    probability = float(config["training"].get("action_mirror_probability", 0.0))
    if probability:
        indices = action_mirror_motion_indices(config)
        second_moment = variance[indices] + mean[indices] ** 2
        mean[indices] *= 1 - 2 * probability
        variance[indices] = second_moment - mean[indices] ** 2
    std = np.maximum(np.sqrt(variance), float(config["features"]["normalization_min_std"]))
    return mean.astype(np.float32), std.astype(np.float32)


def action_data_summary(sessions, config):
    classes = config["labels"]["action_classes"]
    counts = {name: 0 for name in classes}
    duration = 0.0
    for session in sessions:
        supervised = session["valid"] & session["mask"].any(axis=1)
        duration += float(np.diff(session["times"])[supervised[:-1] & supervised[1:]].sum())
        for interval in session["intervals"]:
            channel = classes.index(interval["label"])
            pulse = action_pulse_frames(interval, session["times"], config)
            support = session["mask"][pulse, channel]
            if support.any():
                counts[interval["label"]] += 1
    positive = sum((session["targets"] * session["mask"]).sum(axis=0) for session in sessions)
    background = sum(((1 - session["targets"]) * session["mask"]).sum(axis=0) for session in sessions)
    return {
        "sessions": {kind: sum(session.get("sample_type", "positive") == kind for session in sessions)
                     for kind in ("positive", "negative")},
        "supervised_seconds": duration, "events": counts,
        "positive_target_frames": dict(zip(classes, map(int, positive))),
        "background_target_frames": dict(zip(classes, map(int, background))),
    }


def checkpoint_rank(summary, val_loss):
    selection = summary.get("decoded_events") or summary
    overall = selection["overall"]
    delay = overall["mean_absolute_delay_seconds"]
    return (selection["macro_f1"], -overall["false_positives_per_minute"],
            -delay if delay is not None else -float("inf"), -val_loss)


def train_action(config):
    config = copy.deepcopy(config)
    mode = config["inference"].get("action_model_mode", "one_shot")
    if mode == "one_shot":
        config["labels"]["action_classes"] = list(config["inference"]["action_event_groups"]["one_shot"])
    elif mode != "combined":
        raise ValueError("inference.action_model_mode must be one_shot or combined")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config["training"]["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The configured macOS MPS GPU is unavailable")
    if device.type not in ("mps", "cpu"):
        raise ValueError("Action training device must be mps or explicitly configured cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(config["training"]["action_cpu_threads"]))
    train_sessions = load_action_split(config, "train", resample=True)
    val_sessions = load_action_split(config, "val", resample=True)
    mirror_sessions = []
    if config["training"].get("action_mirror_probability", 0.0):
        for session in val_sessions:
            mirrored = mirror_action_targets(session, config)
            mirrored.update(id=session["id"] + "_mirror", features=session["features"].copy())
            mirrored["features"][:, action_mirror_motion_indices(config)] *= -1
            mirror_sessions.append(mirrored)
    classes = config["labels"]["action_classes"]
    for name, sessions in (("train", train_sessions), ("val", val_sessions)):
        positives = sum((session["targets"] * session["mask"]).sum(axis=0) for session in sessions)
        absent = [label for channel, label in enumerate(classes) if not positives[channel]]
        if absent:
            raise ValueError(f"{name} is missing supervised Action events: {absent}")
    mean, std = action_normalization(train_sessions, config)
    train_data = ActionWindows(train_sessions, mean, std, config, augment=True)
    val_data = ActionWindows(val_sessions, mean, std, config)
    for name, dataset in (("train", train_data), ("val", val_data)):
        retained = sum((targets * mask).sum(dim=0) for _features, targets, mask in dataset)
        absent = [label for channel, label in enumerate(classes) if not retained[channel]]
        if absent:
            raise ValueError(f"{name}: cropping removed all positive targets for {absent}; check action starts and gaps")
    batch_size = int(config["training"]["action_batch_size"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_action_windows)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_action_windows)
    positive = sum((session["targets"] * session["mask"]).sum(axis=0) for session in train_sessions)
    negative = sum(((1 - session["targets"]) * session["mask"]).sum(axis=0) for session in train_sessions)
    weights = np.clip(negative / positive, 1.0, float(config["training"]["action_max_positive_weight"])).astype(np.float32)
    positive_weight = torch.from_numpy(weights).to(device)
    model_config = {name: config["models"]["action_" + name] for name in (
        "hidden_size", "num_layers", "conv_channels", "conv_kernel_size", "dropout")}
    input_size = action_feature_size(config)
    model = ActionCNNLSTM(input_size, num_classes=len(classes), **model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    warmup = int(config["training"]["action_warmup_epochs"])
    start_factor = float(config["training"]["action_warmup_start_factor"])
    decay_every = int(config["training"]["action_lr_decay_every_epochs"])
    decay_factor = float(config["training"]["action_lr_decay_factor"])

    def learning_rate_factor(epoch):
        factor = min(1.0, start_factor + (1.0 - start_factor) * epoch / warmup) if warmup else 1.0
        return factor * decay_factor ** (epoch // decay_every)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    output = action_output_directory(config)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.pt"
    print(f"Training Action on {device}: {len(train_data)} train / {len(val_data)} validation windows")
    print(f"Features: {input_size}; classes: {classes}; positive weights: {weights.tolist()}")
    print(f"Meta-style Conv1d + LSTM: {model_config}; resampling={config['data']['action_sample_rate_hz']:g} Hz")
    print(f"Training motion mirror probability: {config['training'].get('action_mirror_probability', 0.0):g}")
    train_summary = action_data_summary(train_sessions, config)
    val_summary = action_data_summary(val_sessions, config)
    print(f"Train data: {json.dumps(train_summary)}")
    print(f"Validation data: {json.dumps(val_summary)}")
    checkpoint = {
        "classes": classes, "input_size": input_size,
        "model_mode": mode,
        "model_config": model_config, "output_activation": "sigmoid",
        "feature_mean": torch.from_numpy(mean), "feature_std": torch.from_numpy(std),
        "feature_config": config["features"], "no_hand_reset_frames": config["data"]["no_hand_reset_frames"],
        "window_seconds": config["data"]["action_window_seconds"],
        "sample_rate_hz": config["data"]["action_sample_rate_hz"],
        "negative_context": "task",
        "background_visible_frames": config["data"]["action_background_visible_frames"],
        "target_config": {key: config["labels"][key] for key in (
            "action_positive_frames", "action_one_shot_direction_background",
            "action_positive_delay_seconds", "action_positive_duration_seconds",
            "direction_positive_delay_seconds", "direction_positive_duration_seconds")
            if key in config["labels"]},
    }
    history, best_rank, best_evaluations, best_summary, best_mirror_summary = [], None, None, None, None
    for epoch in range(int(config["training"]["max_epochs"])):
        train_loss = epoch_loss(model, train_loader, device, positive_weight, optimizer, float(config["training"]["action_gradient_clip"]))
        val_loss = epoch_loss(model, val_loader, device)
        predictions = {session["id"]: predict_action_session(model, session, checkpoint, batch_size)
                       for session in val_sessions}
        evaluations = {session["id"]: action_event_metrics(session, predictions[session["id"]], config)
                       for session in val_sessions}
        summary = aggregate_action_metrics(evaluations)
        mirror_evaluations = {
            session["id"]: action_event_metrics(session, predict_action_session(model, session, checkpoint, batch_size), config)
            for session in mirror_sessions}
        mirror_summary = aggregate_action_metrics(mirror_evaluations) if mirror_evaluations else None
        selection_summary = aggregate_action_metrics({**evaluations, **mirror_evaluations})
        history.append({"epoch": epoch + 1, "train_weighted_loss": train_loss,
                        "val_loss": val_loss, "validation_events": summary,
                        "mirror_validation_events": mirror_summary,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        rank = checkpoint_rank(selection_summary, val_loss)
        if best_rank is None or rank > best_rank:
            best_rank, best_evaluations, best_summary, best_mirror_summary = rank, evaluations, summary, mirror_summary
            checkpoint.update(state_dict=model.state_dict(), best_epoch=epoch + 1,
                              selection_metric="validation_decoded_event_macro_f1_with_mirror" if mirror_sessions else "validation_decoded_event_macro_f1",
                              validation_summary=summary, mirror_validation_summary=mirror_summary,
                              val_loss=val_loss)
            torch.save(checkpoint, path)
            for session in val_sessions:
                prediction = predictions[session["id"]]
                trace = trace_waving_session(session, prediction, classes, config)
                folder = output / "evaluation" / session["id"]
                write_action_events(folder / "events.csv", events_on_source_frames(session, trace["atomic_events"]))
                write_final_events(folder / "final_events.csv", events_on_source_frames(session, trace["final_events"]))
                save_action_predictions(output / "evaluation" / session["id"] / "predictions.csv",
                                        session, prediction, classes, trace)
                (folder / "failure_cases.json").write_text(
                    json.dumps(action_failure_cases(session, evaluations[session["id"]], config), indent=2),
                    encoding="utf-8")
        decoded = summary["decoded_events"]
        print(f"epoch {epoch + 1}: train_weighted_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"decoded_macro_f1={decoded['macro_f1']:.4f} "
              f"decoded_fp/min={decoded['overall']['false_positives_per_minute']:.3f}" +
              (f" mirror_decoded_macro_f1={mirror_summary['decoded_events']['macro_f1']:.4f} "
               f"mirror_decoded_fp/min={mirror_summary['decoded_events']['overall']['false_positives_per_minute']:.3f}"
               if mirror_summary else ""))
        scheduler.step()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metrics = {
        "device": str(device), "classes": classes, "input_size": input_size,
        "include_acceleration": config["features"].get("action_include_acceleration", True),
        "base_learning_rate": float(config["training"]["learning_rate"]),
        "model_config": model_config, "sample_rate_hz": checkpoint["sample_rate_hz"],
        "negative_context": checkpoint["negative_context"],
        "mirror_probability": config["training"].get("action_mirror_probability", 0.0),
        "train_sessions": [session["id"] for session in train_sessions],
        "val_sessions": [session["id"] for session in val_sessions],
        "train_windows": len(train_data), "val_windows": len(val_data),
        "stride_frames": config["data"]["action_stride_frames"],
        "positive_weights": weights.tolist(), "best_epoch": checkpoint["best_epoch"],
        "selection_metric": checkpoint["selection_metric"],
        "train_data_summary": train_summary, "val_data_summary": val_summary,
        "history": history, "validation": best_evaluations, "validation_summary": best_summary,
        "mirror_validation_summary": best_mirror_summary,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path, metrics
