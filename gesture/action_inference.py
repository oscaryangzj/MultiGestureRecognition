import copy
import csv
from collections import deque

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from gesture.action_decoding import ActionThresholdDecoder
from gesture.action_features import ACTION_FEATURE_SIZE, ActionFeatureExtractor
from gesture.action_resampling import ActionResampler, scores_on_source_frames
from gesture.config import project_path
from gesture.models import ActionCNNLSTM, ActionLSTM


def action_output_directory(config):
    mode = config["inference"].get("action_model_mode", "one_shot")
    if mode not in ("one_shot", "combined"):
        raise ValueError("inference.action_model_mode must be one_shot or combined")
    output_key = "action_combined_output_dir" if mode == "combined" else "action_one_shot_output_dir"
    return project_path(config["inference"].get(output_key, config["inference"]["action_output_dir"]))


def load_action_model(config, device=None):
    path = action_output_directory(config) / "model.pt"
    if not path.is_file() and config["inference"].get("action_model_mode", "one_shot") == "one_shot":
        # Keep the existing one-shot checkpoint available until the dedicated output exists.
        legacy_key = ("action_finger_angles_output_dir"
                      if config["features"].get("action_include_finger_angles", False) else "action_output_dir")
        path = project_path(config["inference"].get(legacy_key, config["inference"]["action_output_dir"])) / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"No Action checkpoint for mode={config['inference'].get('action_model_mode', 'one_shot')}; "
            "train the selected model with python -m scripts.train_action"
        )
    device = torch.device(device or config["training"]["device"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    configured_mode = config["inference"].get("action_model_mode", "one_shot")
    if checkpoint.get("model_mode", configured_mode) != configured_mode:
        raise ValueError(f"Checkpoint mode {checkpoint.get('model_mode')} does not match configured mode {configured_mode}")
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
    for key in tuple(result["labels"]):
        if key.startswith(("action_positive_", "direction_positive_")) or key == "action_one_shot_direction_background":
            del result["labels"][key]
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
        self.timed_emissions = []

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
        self.timed_emissions = []
        any_reset = False
        for observed, sample_points, sample_scores, sample_world_points in samples:
            scores, reset = self._predict_sample(sample_points, sample_scores, observed, image_size,
                                                 sample_world_points)
            self.last_scores = scores
            self.emissions.append((scores, reset))
            self.timed_emissions.append((scores, reset, observed, float(elapsed)))
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


def save_action_predictions(path, session, predictions, classes, waving_trace=None):
    if waving_trace is not None:
        from gesture.waving_fsm import source_waving_trace
        waving_trace = source_waving_trace(session, waving_trace)
    if "source_session" in session:
        session, predictions = scores_on_source_frames(session, predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["frame_index", "elapsed_seconds", "valid"]
    columns += [f"{name}_{suffix}" for name in classes for suffix in ("score", "target", "supervised")]
    if waving_trace is not None:
        columns += ["waving_state", "waving_active", "waving_expected_direction"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame, elapsed in enumerate(session["times"]):
            row = {"frame_index": frame, "elapsed_seconds": float(elapsed), "valid": int(session["valid"][frame])}
            for channel, name in enumerate(classes):
                row[f"{name}_score"] = float(predictions[frame, channel]) if session["valid"][frame] else ""
                row[f"{name}_target"] = float(session["targets"][frame, channel])
                row[f"{name}_supervised"] = int(session["mask"][frame, channel])
            if waving_trace is not None:
                row["waving_state"] = waving_trace["state"][frame]
                row["waving_active"] = int(waving_trace["active"][frame])
                row["waving_expected_direction"] = waving_trace["expected_direction"][frame]
            writer.writerow(row)
