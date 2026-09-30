import numpy as np


IGNORE_TARGET = -100


def make_state_targets(hand_detected, intervals, classes, null_visible_frames):
    """Create frame labels: manual positives first, then eligible NULL frames."""
    hand_detected = np.asarray(hand_detected, dtype=bool)
    targets = np.full(len(hand_detected), IGNORE_TARGET, dtype=np.int64)
    class_to_index = {name: index for index, name in enumerate(classes)}
    if "NULL" not in class_to_index:
        raise ValueError("labels.state_classes must include NULL")

    occupied = np.zeros(len(hand_detected), dtype=bool)
    for interval in intervals:
        label = interval["label"]
        if label not in class_to_index or label == "NULL":
            raise ValueError(f"State interval has invalid positive label: {label}")
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        if start < 0 or end < start or end >= len(hand_detected):
            raise ValueError(f"Invalid State interval: {interval}")
        if occupied[start : end + 1].any():
            raise ValueError(f"Overlapping State intervals: {interval}")
        occupied[start : end + 1] = True
        visible = hand_detected[start : end + 1]
        selected = np.arange(start, end + 1)[visible]
        targets[selected] = class_to_index[label]

    # An unlabeled frame is NULL only after K consecutive detected-hand frames.
    visible_run = 0
    null_index = class_to_index["NULL"]
    for frame_index, detected in enumerate(hand_detected):
        visible_run = visible_run + 1 if detected else 0
        if (
            visible_run >= int(null_visible_frames)
            and not occupied[frame_index]
            and targets[frame_index] == IGNORE_TARGET
        ):
            targets[frame_index] = null_index
    return targets
