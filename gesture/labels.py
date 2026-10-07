import numpy as np

from gesture.prompts import REST_LABEL


IGNORE_TARGET = -100


def make_state_targets(hand_detected, intervals, classes, prompt_labels):
    """Supervise manual hand poses; REST, missing hands and unlabeled frames are ignored."""
    hand_detected = np.asarray(hand_detected, dtype=bool)
    valid = hand_detected & (np.asarray(prompt_labels) != REST_LABEL)
    targets = np.full(len(hand_detected), IGNORE_TARGET, dtype=np.int64)
    class_to_index = {name: index for index, name in enumerate(classes)}

    occupied = np.zeros(len(hand_detected), dtype=bool)
    for interval in intervals:
        label = interval["label"]
        if label not in class_to_index:
            raise ValueError(f"State interval has invalid positive label: {label}")
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        if start < 0 or end < start or end >= len(hand_detected):
            raise ValueError(f"Invalid State interval: {interval}")
        if occupied[start : end + 1].any():
            raise ValueError(f"Overlapping State intervals: {interval}")
        occupied[start : end + 1] = True
        visible = valid[start : end + 1]
        selected = np.arange(start, end + 1)[visible]
        targets[selected] = class_to_index[label]

    return targets
