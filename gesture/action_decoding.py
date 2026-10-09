from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ActionEvent:
    label: str
    time_seconds: float
    score: float
    frame_index: int | None = None


class ActionOutput:
    """Expose one final action or NULL; keep one-shot outputs briefly visible."""

    def __init__(self, config):
        self.one_shot = set(config["inference"]["action_event_groups"]["one_shot"])
        self.hold_seconds = float(config["inference"]["action_output_hold_seconds"])
        self.label = "NULL"
        self.recent = None
        self.until = None

    def update(self, events, waving_active, elapsed):
        candidates = [event for event in events if event.label in self.one_shot]
        if candidates:
            event = max(candidates, key=lambda item: (item.time_seconds, item.score))
            self.recent = event.label
            self.until = event.time_seconds + self.hold_seconds
        if self.until is not None and elapsed >= self.until:
            self.recent = self.until = None
        self.label = self.recent or ("waving" if waving_active else "NULL")
        return self.label


class ActionThresholdDecoder:
    """Decode causal threshold crossings with channel rearming and group debounce."""

    def __init__(self, classes, config):
        self.classes = list(classes)
        settings = config["inference"]
        base_threshold = float(settings["action_threshold"])
        base_reset = float(settings.get("action_reset_threshold", base_threshold))
        thresholds = settings.get("action_thresholds", {})
        reset_thresholds = settings.get("action_reset_thresholds", {})
        self.thresholds = np.asarray([float(thresholds.get(name, base_threshold)) for name in self.classes])
        self.reset_thresholds = np.asarray([float(reset_thresholds.get(name, base_reset)) for name in self.classes])
        self.debounce = float(settings["action_debounce_seconds"])
        groups = settings.get("action_event_groups", {"one_shot": self.classes})
        self.groups = {
            group: [self.classes.index(name) for name in names if name in self.classes]
            for group, names in groups.items()
        }
        self.groups = {group: indices for group, indices in self.groups.items() if indices}
        configured_debounce = settings.get("action_debounce_by_group", {})
        self.group_debounce = {group: float(configured_debounce.get(group, self.debounce)) for group in self.groups}
        assigned = {index for indices in self.groups.values() for index in indices}
        for index, name in enumerate(self.classes):
            if index not in assigned:
                self.groups[name] = [index]
        self.reset()

    def reset(self):
        self.armed = np.ones(len(self.classes), dtype=bool)
        self.last_group_event = {group: None for group in self.groups}
        self.last_events = []
        self.conflicts = []

    def update(self, scores, elapsed):
        self.last_events, self.conflicts = [], []
        if scores is None:
            return []
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(self.classes),):
            raise ValueError(f"Expected {len(self.classes)} Action scores, received {scores.shape}")
        finite = np.isfinite(scores)
        self.armed |= finite & (scores < self.reset_thresholds)
        crossings = np.flatnonzero(finite & self.armed & (scores >= self.thresholds))
        if not len(crossings):
            return []
        self.armed[crossings] = False
        accepted = []
        for group, indices in self.groups.items():
            candidates = [int(index) for index in crossings if index in indices]
            if not candidates:
                continue
            if len(candidates) > 1:
                peak = max(scores[index] for index in candidates)
                winners = [index for index in candidates if scores[index] == peak]
                if len(winners) != 1:
                    self.conflicts.append({"time_seconds": float(elapsed), "group": group,
                                           "labels": [self.classes[index] for index in candidates]})
                    continue
                candidates = winners
            index = candidates[0]
            previous = self.last_group_event[group]
            if previous is not None and elapsed - previous < self.group_debounce[group]:
                continue
            self.last_group_event[group] = float(elapsed)
            self.last_events.append(ActionEvent(self.classes[index], float(elapsed), float(scores[index])))
            accepted.append(index)
        return accepted


def decode_action_session(session, predictions, classes, config, conflicts=None):
    decoder = ActionThresholdDecoder(classes, config)
    events, previous_segment = [], None
    segments = session.get("segments", np.zeros(len(session["times"]), dtype=int))
    prediction_times = session.get("available_times", session["times"])
    valid = session.get("valid", np.ones(len(session["times"]), dtype=bool))
    for frame, scores in enumerate(np.asarray(predictions)):
        now = float(prediction_times[frame])
        if not valid[frame]:
            decoder.update(None, now)
            continue
        segment = segments[frame]
        if previous_segment is not None and segment != previous_segment:
            decoder.reset()
        previous_segment = segment
        decoder.update(scores, now)
        events.extend(ActionEvent(event.label, event.time_seconds, event.score, frame)
                      for event in decoder.last_events)
        if conflicts is not None:
            conflicts.extend(decoder.conflicts)
    return events


def write_action_events(path, events, conflicts=()):
    import csv
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("kind", "frame_index", "time_seconds", "label", "score", "group"))
        writer.writeheader()
        for event in events:
            item = asdict(event) if isinstance(event, ActionEvent) else dict(event)
            writer.writerow({"kind": "atomic", "frame_index": item.get("frame_index", ""),
                             "time_seconds": item["time_seconds"], "label": item["label"],
                             "score": item.get("score", ""), "group": ""})
        for conflict in conflicts:
            writer.writerow({"kind": "conflict", "frame_index": "", "time_seconds": conflict["time_seconds"],
                             "label": json.dumps(conflict["labels"]), "score": "", "group": conflict["group"]})


def write_final_events(path, events):
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("frame_index", "time_seconds", "label", "score"))
        writer.writeheader()
        for event in events:
            item = asdict(event) if isinstance(event, ActionEvent) else dict(event)
            writer.writerow({"frame_index": item.get("frame_index", ""),
                             "time_seconds": item["time_seconds"], "label": item["label"],
                             "score": item.get("score", "")})


def events_on_source_frames(session, events):
    if "source_session" not in session:
        return events
    source_times = session["source_session"]["times"]
    mapped = []
    for event in events:
        frame = int(np.searchsorted(source_times, event.time_seconds, side="left"))
        frame = min(max(frame, 0), len(source_times) - 1)
        mapped.append(ActionEvent(event.label, event.time_seconds, event.score, frame))
    return mapped
