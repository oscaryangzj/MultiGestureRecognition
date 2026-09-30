import csv
import json
from pathlib import Path

import numpy as np

from gesture.config import project_path
from gesture.features import FEATURE_SIZE, NUM_LANDMARKS, normalize_landmarks
from gesture.labels import IGNORE_TARGET, make_state_targets


def load_session(session_dir):
    with (session_dir / "session.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def load_frame_times(path):
    frames = []
    elapsed = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frames.append(int(row["frame_index"]))
            elapsed.append(float(row["elapsed_seconds"]))
    if frames != list(range(len(frames))):
        raise ValueError(f"Frame indices are not contiguous: {path}")
    return elapsed


def load_landmarks_csv(path):
    frames = []
    elapsed = []
    detected = []
    points_by_frame = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame_points = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
            present = row["hand_detected"] == "1"
            if present:
                for point_index in range(NUM_LANDMARKS):
                    frame_points[point_index] = (
                        float(row[f"x{point_index}"]),
                        float(row[f"y{point_index}"]),
                    )
            frames.append(int(row["frame_index"]))
            elapsed.append(float(row["elapsed_seconds"]))
            detected.append(present)
            points_by_frame.append(frame_points)
    if frames != list(range(len(frames))):
        raise ValueError(f"Landmark frame indices are not contiguous: {path}")
    return (
        np.asarray(elapsed, dtype=np.float64),
        np.asarray(detected, dtype=bool),
        points_by_frame,
    )


def state_examples_for_session(session_dir, classes, null_visible_frames, config):
    session = load_session(session_dir)
    if session.get("target_model") != "state":
        return np.empty((0, FEATURE_SIZE), np.float32), np.empty(0, np.int64)

    _, detected, points_by_frame = load_landmarks_csv(session_dir / "landmarks.csv")
    with (session_dir / "annotations.json").open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    targets = make_state_targets(
        detected,
        annotations.get("state_intervals", []),
        classes,
        null_visible_frames,
    )

    features = []
    labels = []
    for frame_index, target in enumerate(targets):
        if target == IGNORE_TARGET or not detected[frame_index]:
            continue
        feature = normalize_landmarks(
            points_by_frame[frame_index],
            config["min_hand_scale"],
            config["local_scale"],
        )
        if feature is not None:
            features.append(feature)
            labels.append(target)
    if not features:
        return np.empty((0, FEATURE_SIZE), np.float32), np.empty(0, np.int64)
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def load_state_split(config, split_name):
    sessions_dir = project_path(config["data"]["sessions_dir"])
    splits_path = project_path(config["data"]["splits_file"])
    with splits_path.open(encoding="utf-8") as stream:
        splits = json.load(stream)
    for name in ("train", "val"):
        if name not in splits:
            raise ValueError(f"Split {name!r} is absent from {splits_path}")
        if len(splits[name]) != len(set(splits[name])):
            raise ValueError(f"Split {name!r} contains duplicate session IDs")
    overlap = set(splits["train"]) & set(splits["val"])
    if overlap:
        raise ValueError(f"Train and val sessions overlap: {sorted(overlap)}")
    if split_name not in splits:
        raise ValueError(f"Split {split_name!r} is absent from {splits_path}")

    all_features = []
    all_labels = []
    used_sessions = []
    for session_id in splits[split_name]:
        session_dir = sessions_dir / session_id
        session = load_session(session_dir)
        if session.get("target_model") != "state":
            continue
        features, labels = state_examples_for_session(
            session_dir,
            config["labels"]["state_classes"],
            config["labels"]["state_null_visible_frames"],
            config["features"],
        )
        if not len(labels):
            raise ValueError(f"State session {session_id!r} has no valid labeled frames")
        all_features.append(features)
        all_labels.append(labels)
        used_sessions.append(session_id)
    if not all_features:
        raise ValueError(f"No labeled State frames found in split {split_name!r}")
    return np.concatenate(all_features), np.concatenate(all_labels), used_sessions
