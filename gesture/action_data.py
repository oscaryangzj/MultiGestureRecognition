import csv
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from gesture.action_features import action_mirror_motion_indices, make_action_features
from gesture.action_candidates import action_excluded_frames, action_opportunities
from gesture.config import project_path
from gesture.data import load_frame_times, load_landmarks_csv, load_session, load_world_landmarks_csv


def validate_action_intervals(intervals, frame_count, classes):
    previous_end = -1
    for interval in sorted(intervals, key=lambda item: int(item["start_frame"])):
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        if interval["label"] not in classes or not 0 <= start <= end < frame_count:
            raise ValueError(f"Invalid Action interval: {interval}")
        if start <= previous_end:
            raise ValueError(f"Overlapping Action intervals: {interval}")
        previous_end = end


def action_pulse_frames(interval, times, config):
    end = int(interval["end_frame"])
    labels = config["labels"]
    if "action_positive_start_offset_frames" in labels:
        # Used only when inspecting a retained checkpoint from the earlier baseline.
        return np.arange(end + int(labels["action_positive_start_offset_frames"]),
                         min(len(times), end + int(labels["action_positive_end_offset_frames"]) + 1))
    completion = interval.get("end_time", times[end])
    delay = float(labels["action_positive_delay_seconds"])
    duration = float(labels["action_positive_duration_seconds"])
    relative = times - completion
    return np.flatnonzero((relative >= delay - 1e-9) & (relative < delay + duration - 1e-9))


def make_action_targets(valid, segments, intervals, reviewed, config, excluded=None, times=None):
    classes = config["labels"]["action_classes"]
    validate_action_intervals(intervals, len(valid), classes)
    targets = np.zeros((len(valid), len(classes)), np.float32)
    mask = np.zeros_like(targets, dtype=bool)
    required = int(config["data"]["action_background_visible_frames"])
    visible = 0
    previous_segment = None
    for frame, present in enumerate(valid):
        if segments[frame] != previous_segment:
            visible = 0
        previous_segment = segments[frame]
        visible = visible + 1 if present else 0
        if reviewed and visible >= required:
            mask[frame] = True
    if times is None:
        times = np.arange(len(valid)) / float(config["data"]["action_sample_rate_hz"])
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        channel = classes.index(interval["label"])
        pulse = action_pulse_frames(interval, times, config)
        targets[pulse, channel] = 1.0
        # A reset has erased the event history, so its delayed target is ignored.
        mask[pulse, channel] = valid[pulse] & (segments[pulse] == segments[start])
    if excluded is not None:
        mask[excluded] = False
    return targets, mask


def negative_input_regions(annotations, opportunities, frame_count, config, times):
    """Task boundaries and explicit exclusions define input history, not loss masks."""
    tasks = [item for item in opportunities if item.get("sample_type") == "negative"]
    # Pending tasks still need features for annotation; training requires full review.
    history_annotations = {**annotations, "action_background_prompts": [item["prompt_index"] for item in tasks]}
    excluded = action_excluded_frames(history_annotations, opportunities, frame_count, config, times)
    regions = np.full(frame_count, -1, np.int64)
    region = 0
    for task in tasks:
        frames = np.arange(task["start_frame"], task["end_frame"] + 1)
        frames = frames[~excluded[frames]]
        for block in np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1):
            if len(block):
                regions[block] = region
                region += 1
    return regions


def load_action_session(config, session_id, resample=False, isolate_negative=True):
    folder = project_path(config["data"]["sessions_dir"]) / session_id
    metadata = load_session(folder)
    if metadata.get("target_model") != "action":
        raise ValueError(f"{session_id}: requires target_model: action")
    sample_type = metadata.get("sample_type", "positive")
    if sample_type not in ("positive", "negative"):
        raise ValueError(f"{session_id}: unknown sample_type: {sample_type}")
    times, detected, points = load_landmarks_csv(folder / "landmarks.csv")
    frame_times = np.asarray(load_frame_times(folder / "frames.csv"))
    if len(times) == 0 or not np.array_equal(times, frame_times):
        raise ValueError(f"{session_id}: landmarks and frames.csv times must align")
    world_points = None
    if config["features"].get("action_include_finger_angles", False):
        world_path = folder / "world_landmarks.csv"
        if not world_path.is_file():
            raise FileNotFoundError(
                f"{session_id}: run python -m scripts.extract_action --session-id {session_id} "
                "to extract 3D landmarks before training or evaluating"
            )
        world_times, world_detected, world_points = load_world_landmarks_csv(world_path)
        if not np.array_equal(times, world_times) or not np.array_equal(detected, world_detected):
            raise ValueError(f"{session_id}: 2D and world landmark rows must align")
    scores = []
    with (folder / "state_scores.csv").open(newline="") as stream:
        for frame, row in enumerate(csv.DictReader(stream)):
            if int(row["frame_index"]) != frame:
                raise ValueError("State score frame indices must be contiguous")
            scores.append({name: float(row[name]) for name in config["features"]["action_state_classes"]} if detected[frame] else None)
    if len(scores) != len(times):
        raise ValueError(f"{session_id}: State scores must have one row per frame")
    annotations = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
    intervals = annotations.get("action_intervals", [])
    reviewed = bool(annotations.get("action_reviewed", False))
    image_size = (1, 1)
    if config["features"].get("action_align_palm_axis", False):
        capture = cv2.VideoCapture(str(folder / "video.mp4"))
        image_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        if min(image_size) <= 0:
            raise ValueError(f"{session_id}: video.mp4 is required for palm-axis aspect correction")
    opportunities = action_opportunities(metadata["prompts"], len(times))
    excluded = action_excluded_frames(annotations, opportunities, len(times), config, times)
    input_regions = (negative_input_regions(annotations, opportunities, len(times), config, times)
                     if sample_type == "negative" and isolate_negative else None)
    features, valid, segments = make_action_features(
        points, detected, scores, times, config, image_size, input_regions, world_points_by_frame=world_points)
    targets, mask = make_action_targets(valid, segments, intervals, reviewed, config, excluded, times)
    session = {
        "id": session_id, "sample_type": sample_type,
        "times": times, "points": points, "world_points": world_points, "detected": detected,
        "features": features, "valid": valid, "segments": segments,
        "intervals": intervals, "reviewed": reviewed, "targets": targets, "mask": mask,
        "opportunities": opportunities, "annotations": annotations, "ignored_frames": excluded,
        "state_scores": scores,
        "image_size": image_size,
        "input_regions": input_regions,
    }
    if resample:
        from gesture.action_resampling import resample_action_session
        session = resample_action_session(session, config)
    return session


def action_split_ids(config, split):
    path = project_path(config["data"]["splits_file"])
    splits = json.loads(path.read_text(encoding="utf-8"))
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Train and validation sessions overlap")
    for name in ("train", "val"):
        if len(splits[name]) != len(set(splits[name])):
            raise ValueError(f"Duplicate session IDs in {name}")
    root = project_path(config["data"]["sessions_dir"])
    return [session_id for session_id in splits[split]
            if load_session(root / session_id).get("target_model") == "action"]


def load_action_split(config, split, resample=False):
    sessions = [load_action_session(config, session_id, resample=resample) for session_id in action_split_ids(config, split)]
    if not sessions:
        raise ValueError(f"No Action sessions in {split}; collect, annotate, then add them to splits.json")
    for session in sessions:
        if not session["reviewed"]:
            raise ValueError(f"{session['id']}: review or cancel every Action/background prompt before training")
    return sessions


def mask_truncated_events(session, indices, config):
    mask = session["mask"][indices].copy()
    classes = config["labels"]["action_classes"]
    for interval in session["intervals"]:
        start_time = interval.get("start_time", session["times"][int(interval["start_frame"])])
        if start_time < session["times"][indices[0]]:
            pulse = np.isin(indices, action_pulse_frames(interval, session["times"], config))
            mask[pulse, classes.index(interval["label"])] = False
    return mask


def action_window_indices(session, config):
    seconds = float(config["data"]["action_window_seconds"])
    stride = int(config["data"]["action_stride_frames"])
    if seconds <= 0 or stride <= 0:
        raise ValueError("Action window duration and stride must be positive")
    times, valid, segments = session["times"], session["valid"], session["segments"]
    excluded = session.get("ignored_frames", np.zeros(len(times), bool))
    for segment in np.unique(segments[valid]):
        frames = np.flatnonzero(valid & (segments == segment))
        begins = range(int(frames[0]), int(frames[-1]) + 1, stride)
        segment_end = times[frames[-1]] + 1 / float(config["data"]["action_sample_rate_hz"])
        short_segment = segment_end - times[frames[0]] < seconds - 1e-9
        for start in begins:
            if not valid[start] or excluded[start]:
                continue
            if any(int(item["start_frame"]) <= start <= int(item["end_frame"]) for item in session["intervals"]):
                continue
            if not short_segment and times[start] + seconds > segment_end + 1e-9:
                break
            selected = frames[(frames >= start) & (times[frames] < times[start] + seconds - 1e-9)]
            if len(selected) and mask_truncated_events(session, selected, config).any():
                yield selected
                if short_segment:
                    break


class ActionWindows(Dataset):
    def __init__(self, sessions, mean, std, config, augment=False):
        self.sessions, self.mean, self.std, self.config = sessions, mean, std, config
        self.mirror_probability = float(config["training"].get("action_mirror_probability", 0.0)) if augment else 0.0
        self.windows = [(index, frames) for index, session in enumerate(sessions)
                        for frames in action_window_indices(session, config)]
        if not self.windows:
            raise ValueError("No valid Action windows; check intervals and recording duration")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        session_index, frames = self.windows[index]
        session = self.sessions[session_index]
        features = session["features"][frames].copy()
        # Reflect a whole window, keeping camera direction intact during inference.
        if self.mirror_probability and np.random.random() < self.mirror_probability:
            features[:, action_mirror_motion_indices(self.config)] *= -1
        features = (features - self.mean) / self.std
        return (torch.from_numpy(features), torch.from_numpy(session["targets"][frames]),
                torch.from_numpy(mask_truncated_events(session, frames, self.config)))


def collate_action_windows(batch):
    # Padding is at the end and excluded from every loss channel.
    return tuple(pad_sequence(items, batch_first=True) for items in zip(*batch))
