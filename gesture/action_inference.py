import copy
import csv
from collections import deque

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from gesture.action_features import ACTION_FEATURE_SIZE, ActionFeatureExtractor
from gesture.action_resampling import ActionResampler, scores_on_source_frames
from gesture.config import project_path
from gesture.models import ActionCNNLSTM, ActionLSTM


def action_output_directory(config):
    output_key = ("action_finger_angles_output_dir"
                  if config["features"].get("action_include_finger_angles", False) else "action_output_dir")
    return project_path(config["inference"][output_key])


def load_action_model(config, device=None):
    path = action_output_directory(config) / "model.pt"
    if not path.is_file() and config["features"].get("action_include_finger_angles", False):
        # Let users continue to run the previous model until the angle model is trained.
        path = project_path(config["inference"]["action_output_dir"]) / "model.pt"
    if not path.is_file():
        raise FileNotFoundError("Train Action first: python -m scripts.train_action")
    device = torch.device(device or config["training"]["device"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_config" in checkpoint:
        model = ActionCNNLSTM(checkpoint["input_size"], num_classes=len(checkpoint["classes"]), **checkpoint["model_config"])
    else:
        model = ActionLSTM(checkpoint.get("input_size", ACTION_FEATURE_SIZE), checkpoint["hidden_size"], checkpoint["num_layers"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def preprocessing_config(config, checkpoint):
    result = copy.deepcopy(config)
    result["features"] = checkpoint["feature_config"]
    result["data"]["no_hand_reset_frames"] = checkpoint["no_hand_reset_frames"]
    result["data"]["action_window_seconds"] = checkpoint["window_seconds"]
    result["labels"]["action_classes"] = checkpoint["classes"]
    result["labels"].update(checkpoint["target_config"])
    result["data"]["action_background_visible_frames"] = checkpoint["background_visible_frames"]
    if "sample_rate_hz" in checkpoint:
        result["data"]["action_sample_rate_hz"] = checkpoint["sample_rate_hz"]
    return result


class OnlineActionPredictor:
    """Predict the newest frame using the same rolling context as evaluation."""

    def __init__(self, model, checkpoint, config):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.extractor = ActionFeatureExtractor(preprocessing_config(config, checkpoint))
        self.include_world_landmarks = self.extractor.config["features"].get("action_include_finger_angles", False)
        self.mean = checkpoint["feature_mean"].numpy()
        self.std = checkpoint["feature_std"].numpy()
        self.seconds = float(checkpoint["window_seconds"])
        self.history = deque()
        self.resampler = ActionResampler(checkpoint["sample_rate_hz"]) if "sample_rate_hz" in checkpoint else None
        self.last_scores = None
        self.emissions = []

    @torch.inference_mode()
    def update(self, points, state_scores, elapsed, image_size=(1, 1), world_points=None):
        if self.resampler is not None:
            samples = (self.resampler.update_with_world(points, state_scores, elapsed, world_points)
                       if self.include_world_landmarks else self.resampler.update(points, state_scores, elapsed))
            if not self.include_world_landmarks:
                samples = [(*sample, None) for sample in samples]
        else:
            samples = [(elapsed, points, state_scores, world_points)]
        self.emissions = []
        any_reset = False
        for observed, sample_points, sample_scores, sample_world_points in samples:
            scores, reset = self._predict_sample(sample_points, sample_scores, observed, image_size,
                                                 sample_world_points)
            self.last_scores = scores
            self.emissions.append((scores, reset))
            any_reset |= reset
        if points is None or state_scores is None:
            return None, any_reset
        return self.last_scores, any_reset

    def _predict_sample(self, points, state_scores, elapsed, image_size, world_points=None):
        feature, reset = self.extractor.update(points, state_scores, elapsed, image_size, world_points)
        if reset:
            self.history.clear()
        while self.history and self.history[0][0] <= elapsed - self.seconds:
            self.history.popleft()
        if feature is None:
            return None, reset
        self.history.append((elapsed, (feature - self.mean) / self.std))
        inputs = torch.from_numpy(np.stack([item[1] for item in self.history])).unsqueeze(0).to(self.device)
        scores = torch.sigmoid(self.model(inputs)[0, -1]).cpu().numpy()
        return scores, reset


class ActionThresholdDecoder:
    """Meta-style threshold crossings and debounce for live inspection."""

    def __init__(self, classes, config):
        self.classes = classes
        self.threshold = float(config["inference"]["action_threshold"])
        self.debounce = float(config["inference"]["action_debounce_seconds"])
        self.reset()

    def reset(self):
        self.previous_above = np.zeros(len(self.classes), bool)
        self.last_event_time = None

    def update(self, scores, elapsed):
        if scores is None:
            return []
        above = np.asarray(scores) >= self.threshold
        crossings = np.flatnonzero(above & ~self.previous_above)
        self.previous_above = above
        events = []
        for channel in crossings:
            if self.last_event_time is not None and elapsed - self.last_event_time < self.debounce:
                continue
            events.append(int(channel))
            self.last_event_time = elapsed
        return events


def rolling_contexts(session, seconds):
    history = []
    previous_segment = None
    for frame in np.flatnonzero(session["valid"]):
        segment = session["segments"][frame]
        if segment != previous_segment:
            history = []
        previous_segment = segment
        history.append(int(frame))
        cutoff = session["times"][frame] - seconds
        history = [index for index in history if session["times"][index] > cutoff]
        yield int(frame), np.asarray(history, dtype=np.int64)


def predict_action_session(model, session, checkpoint, batch_size):
    device = next(model.parameters()).device
    mean, std = checkpoint["feature_mean"].numpy(), checkpoint["feature_std"].numpy()
    features = (session["features"] - mean) / std
    predictions = np.full(session["targets"].shape, np.nan, np.float32)
    batch = []

    def evaluate_batch():
        inputs = pad_sequence([torch.from_numpy(features[indices]) for _frame, indices in batch], batch_first=True).to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model(inputs)).cpu().numpy()
        for index, (frame, indices) in enumerate(batch):
            predictions[frame] = scores[index, len(indices) - 1]
        batch.clear()

    for item in rolling_contexts(session, checkpoint["window_seconds"]):
        batch.append(item)
        if len(batch) >= batch_size:
            evaluate_batch()
    if batch:
        evaluate_batch()
    return predictions


def save_action_predictions(path, session, predictions, classes):
    if "source_session" in session:
        session, predictions = scores_on_source_frames(session, predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["frame_index", "elapsed_seconds", "valid"]
    columns += [f"{name}_{suffix}" for name in classes for suffix in ("score", "target", "supervised")]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame, elapsed in enumerate(session["times"]):
            row = {"frame_index": frame, "elapsed_seconds": float(elapsed), "valid": int(session["valid"][frame])}
            for channel, name in enumerate(classes):
                row[f"{name}_score"] = float(predictions[frame, channel]) if session["valid"][frame] else ""
                row[f"{name}_target"] = float(session["targets"][frame, channel])
                row[f"{name}_supervised"] = int(session["mask"][frame, channel])
            writer.writerow(row)
