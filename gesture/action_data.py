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
    completed = set()
    for interval in sorted(intervals, key=lambda item: int(item["start_frame"])):
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        if interval["label"] not in classes or not 0 <= start <= end < frame_count:
            raise ValueError(f"Invalid Action interval: {interval}")
        key = (interval["label"], end)
        if key in completed:
            raise ValueError(f"Duplicate Action event completion: {interval}")
        completed.add(key)


def action_pulse_frames(interval, times, config):
    end = int(interval["end_frame"])
    labels = config["labels"]
    if "action_positive_frames" in labels:
        return np.arange(end, min(len(times), end + int(labels["action_positive_frames"])))
    if "action_positive_start_offset_frames" in labels:
        # Used only when inspecting a retained checkpoint from the earlier baseline.
        return np.arange(end + int(labels["action_positive_start_offset_frames"]),
                         min(len(times), end + int(labels["action_positive_end_offset_frames"]) + 1))
    completion = interval.get("end_time", times[end])
    direction = interval["label"] in config.get("inference", {}).get("action_event_groups", {}).get("direction", [])
    prefix = "direction_" if direction else "action_"
    delay = float(labels.get(prefix + "positive_delay_seconds", labels["action_positive_delay_seconds"]))
    duration = float(labels.get(prefix + "positive_duration_seconds", labels["action_positive_duration_seconds"]))
    relative = times - completion
    return np.flatnonzero((relative >= delay - 1e-9) & (relative < delay + duration - 1e-9))


def _reviewed_frames(annotations, field, frame_count):
    frames = np.zeros(frame_count, dtype=bool)
    for interval in annotations.get(field, []):
        start = max(0, int(interval["start_frame"]))
        end = min(frame_count - 1, int(interval["end_frame"]))
        if start <= end:
            frames[start:end + 1] = True
    return frames


def make_action_targets(valid, segments, intervals, reviewed, config, excluded=None, times=None,
                        annotations=None, opportunities=None, kind=None, sample_type=None):
    classes = config["labels"]["action_classes"]
    validate_action_intervals(intervals, len(valid), classes)
    targets = np.zeros((len(valid), len(classes)), np.float32)
    mask = np.zeros_like(targets, dtype=bool)
    required = int(config["data"]["action_background_visible_frames"])
    visible = 0
    previous_segment = None
    one_shot_reviewed = (_reviewed_frames(annotations, "one_shot_reviewed_intervals", len(valid))
                         if annotations is not None else np.zeros(len(valid), bool))
    direction_reviewed = (_reviewed_frames(annotations, "direction_reviewed_intervals", len(valid))
                          if annotations is not None else np.zeros(len(valid), bool))
    background_prompts = set(annotations.get("action_background_prompts", [])) if annotations is not None else set()
    groups = config.get("inference", {}).get("action_event_groups", {})
    direction_classes = set(groups.get("direction", ()))
    one_shot_classes = set(groups.get("one_shot", classes)) - direction_classes
    direction_background = (kind == "one_shot" and sample_type != "negative" and reviewed
                            and config["labels"].get("action_one_shot_direction_background", False))
    for frame, present in enumerate(valid):
        if segments[frame] != previous_segment:
            visible = 0
        previous_segment = segments[frame]
        visible = visible + 1 if present else 0
        if reviewed and visible >= required:
            if annotations is None or kind not in ("periodic", "continuous"):
                for channel, name in enumerate(classes):
                    if name in one_shot_classes or direction_background:
                        mask[frame, channel] = True
        if annotations is not None and visible >= required:
            if kind in ("periodic", "continuous"):
                for channel, name in enumerate(classes):
                    if name in one_shot_classes and one_shot_reviewed[frame]:
                        mask[frame, channel] = True
                    if name in direction_classes and direction_reviewed[frame]:
                        mask[frame, channel] = True
            elif reviewed:
                for channel, name in enumerate(classes):
                    if name in one_shot_classes or direction_background:
                        mask[frame, channel] = True
            for channel, name in enumerate(classes):
                if name in direction_classes and direction_reviewed[frame]:
                    mask[frame, channel] = True
            if opportunities:
                for opportunity in opportunities:
                    if opportunity["prompt_index"] not in background_prompts:
                        continue
                    if opportunity["start_frame"] <= frame <= opportunity["end_frame"]:
                        for channel, name in enumerate(classes):
                            if name in one_shot_classes:
                                mask[frame, channel] = True
    if times is None:
        times = np.arange(len(valid)) / float(config["data"]["action_sample_rate_hz"])
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        channel = classes.index(interval["label"])
        pulse = action_pulse_frames(interval, times, config)
        targets[pulse, channel] = 1.0
        # A reset has erased the event history, so its target is ignored.
        mask[pulse, channel] = valid[pulse] & (segments[pulse] == segments[start])
    if annotations is not None:
        direction_ignored = _reviewed_frames(annotations, "direction_ignored_intervals", len(valid))
        for channel, name in enumerate(classes):
            if name in direction_classes:
                mask[direction_ignored, channel] = False
    if excluded is not None:
        mask[excluded] = False
    return targets, mask


def mirror_action_targets(session, config):
    """Swap the camera-direction labels when a recording is horizontally reflected."""
    classes = config["labels"]["action_classes"]
    direction = config.get("inference", {}).get("action_event_groups", {}).get("direction", [])
    pairs = [(classes.index("left_to_right"), classes.index("right_to_left"))] if {
        "left_to_right", "right_to_left"}.issubset(classes) and set(direction) else []
    mirrored = {**session, "targets": session["targets"].copy(), "mask": session["mask"].copy(),
                "intervals": [dict(item) for item in session["intervals"]]}
    for left, right in pairs:
        mirrored["targets"][:, [left, right]] = session["targets"][:, [right, left]]
        mirrored["mask"][:, [left, right]] = session["mask"][:, [right, left]]
    for interval in mirrored["intervals"]:
        if interval["label"] == "left_to_right":
            interval["label"] = "right_to_left"
        elif interval["label"] == "right_to_left":
            interval["label"] = "left_to_right"
    return mirrored


def negative_input_regions(annotations, opportunities, frame_count, config, times, kind=None):
    """Task boundaries and explicit exclusions define input history, not loss masks."""
    tasks = [item for item in opportunities if item.get("sample_type") == "negative"]
    if kind == "periodic":
        regions = np.full(frame_count, -1, np.int64)
        ignored = np.zeros(frame_count, bool)
        for item in annotations.get("action_ignored_intervals", []):
            ignored[max(0, int(item["start_frame"])):min(frame_count, int(item["end_frame"]) + 1)] = True
        for region, task in enumerate(tasks):
            frames = np.arange(task["start_frame"], task["end_frame"] + 1)
            regions[frames[~ignored[frames]]] = region
        return regions
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


def periodic_input_regions(metadata, opportunities, frame_count, annotations=None, config=None):
    """Keep each task's history, extending it to cover manually confirmed events."""
    regions = np.full(frame_count, -1, np.int64)
    excluded = np.zeros(frame_count, bool)
    for item in (annotations or {}).get("action_ignored_intervals", []):
        excluded[max(0, int(item["start_frame"])):min(frame_count, int(item["end_frame"]) + 1)] = True
    region_id = 0
    for index, prompt in enumerate(metadata.get("prompts", [])):
        if prompt.get("sample_type") == "negative":
            continue
        start = prompt.get("start_frame")
        end = prompt.get("hold_end_frame", prompt.get("end_frame"))
        if start is None or end is None:
            continue
        for field in ("waving_intervals", "action_intervals"):
            for interval in (annotations or {}).get(field, []):
                if interval.get("prompt_index") != index:
                    continue
                start = min(int(start), int(interval["start_frame"]))
                finish = int(interval["end_frame"])
                if field == "action_intervals" and config is not None and "action_positive_frames" in config["labels"]:
                    finish += int(config["labels"]["action_positive_frames"]) - 1
                end = max(int(end), finish)
        start, end = max(0, int(start)), min(frame_count - 1, int(end))
        if start <= end:
            frames = np.arange(start, end + 1)
            frames = frames[~excluded[frames]]
            for block in np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1):
                if len(block):
                    regions[block] = region_id
                    region_id += 1
    return regions


def continuous_input_regions(annotations, frame_count):
    """Keep one continuous history, cutting only explicitly cancelled ranges."""
    regions = np.full(frame_count, -1, np.int64)
    excluded = np.zeros(frame_count, bool)
    for item in annotations.get("action_ignored_intervals", []):
        excluded[max(0, int(item["start_frame"])):min(frame_count, int(item["end_frame"]) + 1)] = True
    region, previous = 0, False
    for frame in range(frame_count):
        if excluded[frame]:
            previous = False
            continue
        if not previous:
            region += 1
        regions[frame] = region - 1
        previous = True
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
    classes = set(config["labels"]["action_classes"])
    intervals = [item for item in annotations.get("action_intervals", []) if item["label"] in classes]
    reviewed = bool(annotations.get("action_reviewed", False))
    image_size = (1, 1)
    if config["features"].get("action_align_palm_axis", False):
        capture = cv2.VideoCapture(str(folder / "video.mp4"))
        image_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        if min(image_size) <= 0:
            raise ValueError(f"{session_id}: video.mp4 is required for palm-axis aspect correction")
    opportunities = action_opportunities(metadata["prompts"], len(times), metadata)
    kind = metadata.get("kind", "one_shot")
    excluded = action_excluded_frames(annotations, opportunities, len(times), config, times, kind)
    if isolate_negative and sample_type == "negative":
        input_regions = negative_input_regions(annotations, opportunities, len(times), config, times, kind)
    elif isolate_negative and kind == "periodic":
        input_regions = periodic_input_regions(metadata, opportunities, len(times), annotations, config)
    elif isolate_negative and kind == "continuous":
        input_regions = continuous_input_regions(annotations, len(times))
    else:
        input_regions = None
    features, valid, segments = make_action_features(
        points, detected, scores, times, config, image_size, input_regions, world_points_by_frame=world_points)
    targets, mask = make_action_targets(valid, segments, intervals, reviewed, config, excluded, times,
                                        annotations, opportunities, kind, sample_type)
    session = {
        "id": session_id, "sample_type": sample_type, "kind": kind,
        "times": times, "points": points, "world_points": world_points, "detected": detected,
        "features": features, "valid": valid, "segments": segments,
        "intervals": intervals, "reviewed": reviewed, "targets": targets, "mask": mask,
        "opportunities": opportunities, "annotations": annotations, "ignored_frames": excluded,
        "state_scores": scores,
        "image_size": image_size,
        "input_regions": input_regions,
        "metadata": metadata,
    }
    if resample:
        from gesture.action_resampling import resample_action_session
        session = resample_action_session(session, config)
    return session


def action_split_ids(config, split):
    path = project_path(config["data"]["splits_file"])
    splits = json.loads(path.read_text(encoding="utf-8"))
    if split not in ("train", "val", "acceptance"):
        raise ValueError(f"Unknown Action split: {split}")
    names = [name for name in ("train", "val", "acceptance") if name in splits]
    if split not in splits:
        if split == "acceptance":
            return []
        raise ValueError(f"Missing {split} split in {path}")
    assigned = set()
    for name in names:
        if len(splits[name]) != len(set(splits[name])):
            raise ValueError(f"Duplicate session IDs in {name}")
        overlap = assigned.intersection(splits[name])
        if overlap:
            raise ValueError(f"Session IDs appear in multiple splits: {sorted(overlap)}")
        assigned.update(splits[name])
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
            if session.get("kind") != "periodic" and any(
                    int(item["start_frame"]) <= start <= int(item["end_frame"]) for item in session["intervals"]):
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
        targets, mask = session["targets"], session["mask"]
        # Reflect a whole window, keeping camera direction intact during inference.
        if self.mirror_probability and np.random.random() < self.mirror_probability:
            features[:, action_mirror_motion_indices(self.config)] *= -1
            mirrored = mirror_action_targets(session, self.config)
            targets, mask = mirrored["targets"], mirrored["mask"]
            session = mirrored
        features = (features - self.mean) / self.std
        return (torch.from_numpy(features), torch.from_numpy(targets[frames]),
                torch.from_numpy(mask_truncated_events(session, frames, self.config)))


def collate_action_windows(batch):
    # Padding is at the end and excluded from every loss channel.
    return tuple(pad_sequence(items, batch_first=True) for items in zip(*batch))
