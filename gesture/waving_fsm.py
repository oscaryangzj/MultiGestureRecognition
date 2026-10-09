from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from gesture.action_decoding import ActionEvent, decode_action_session


@dataclass
class WavingSnapshot:
    state: str
    active: bool
    expected_direction: str | None
    deadline_seconds: float | None
    transitions: list


class WavingFSM:
    """Combine two opposite direction events and time out on the expected event."""

    OPPOSITE = {"left_to_right": "right_to_left", "right_to_left": "left_to_right"}

    def __init__(self, config):
        settings = config["inference"]
        self.pair_timeout = float(settings["waving_pair_timeout_seconds"])
        self.active_timeout = float(settings["waving_active_timeout_seconds"])
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.first_direction = None
        self.expected_direction = None
        self.deadline = None
        self.transitions = []

    def _expire(self, now):
        if self.state == "IDLE" or self.deadline is None or now < self.deadline:
            return
        if self.state == "ACTIVE":
            self.transitions.append(ActionEvent("waving_end", float(now), 0.0))
        self.state = "IDLE"
        self.first_direction = None
        self.expected_direction = None
        self.deadline = None

    def update(self, events, time_seconds, history_reset=False):
        now = float(time_seconds)
        self.transitions = []
        if history_reset and self.state == "WAIT_OPPOSITE":
            self.state = "IDLE"
            self.first_direction = None
            self.expected_direction = None
            self.deadline = None
        self._expire(now)
        for event in sorted(events, key=lambda item: item.time_seconds):
            direction = event.label
            if direction not in self.OPPOSITE:
                continue
            event_time = float(event.time_seconds)
            self._expire(event_time)
            if self.state == "IDLE":
                self.state = "WAIT_OPPOSITE"
                self.first_direction = direction
                self.deadline = event_time + self.pair_timeout
            elif self.state == "WAIT_OPPOSITE":
                if direction == self.OPPOSITE[self.first_direction]:
                    self.state = "ACTIVE"
                    self.expected_direction = self.first_direction
                    self.deadline = event_time + self.active_timeout
                    self.transitions.append(ActionEvent("waving_start", event_time, float(event.score),
                                                         event.frame_index))
            elif direction == self.expected_direction:
                self.expected_direction = self.OPPOSITE[direction]
                self.deadline = event_time + self.active_timeout
        self._expire(now)
        return WavingSnapshot(self.state, self.state == "ACTIVE", self.expected_direction,
                              self.deadline, list(self.transitions))


def trace_waving_session(session, predictions, classes, config):
    """Run the same causal atomic decoder and waving FSM used by live inference."""
    atomic = decode_action_session(session, predictions, classes, config)
    events_by_frame = defaultdict(list)
    for event in atomic:
        if event.label in WavingFSM.OPPOSITE and event.frame_index is not None:
            events_by_frame[int(event.frame_index)].append(event)
    times = np.asarray(session.get("available_times", session["times"]), dtype=float)
    valid = session.get("valid", np.ones(len(times), dtype=bool))
    segments = session.get("segments", np.zeros(len(times), dtype=int))
    fsm = WavingFSM(config)
    states, active, expected, transitions = [], [], [], []
    previous_segment = None
    for frame, now in enumerate(times):
        history_reset = bool(valid[frame] and previous_segment is not None and segments[frame] != previous_segment)
        if valid[frame]:
            previous_segment = segments[frame]
        snapshot = fsm.update(events_by_frame.get(frame, ()), now, history_reset)
        for event in snapshot.transitions:
            transitions.append(ActionEvent(event.label, event.time_seconds, event.score,
                                           frame if event.frame_index is None else event.frame_index))
        states.append(snapshot.state)
        active.append(snapshot.active)
        expected.append(snapshot.expected_direction or "")
    one_shot = set(config.get("inference", {}).get("action_event_groups", {}).get("one_shot", ()))
    final_events = [event for event in atomic if event.label in one_shot] + transitions
    final_events.sort(key=lambda event: (event.time_seconds, event.label))
    return {"atomic_events": atomic, "waving_events": transitions, "final_events": final_events,
            "state": np.asarray(states, dtype=object), "active": np.asarray(active, dtype=bool),
            "expected_direction": np.asarray(expected, dtype=object)}


def source_waving_trace(session, trace):
    """Map resampled states to each original video frame using only available outputs."""
    if "source_session" not in session:
        return trace
    source = session["source_session"]
    indices = np.searchsorted(session["available_times"], source["times"], side="right") - 1
    available = indices >= 0
    mapped = {key: np.full(len(source["times"]), "", dtype=object) for key in ("state", "expected_direction")}
    mapped["active"] = np.zeros(len(source["times"]), dtype=bool)
    mapped["state"][available] = trace["state"][indices[available]]
    mapped["expected_direction"][available] = trace["expected_direction"][indices[available]]
    mapped["active"][available] = trace["active"][indices[available]]
    return mapped


def waving_fsm_metrics(session, trace, config):
    """Score waving starts and active coverage only in direction-reviewed ranges."""
    times = np.asarray(session["times"], dtype=float)
    annotations = session.get("annotations", {})
    review_scope = np.zeros(len(times), dtype=bool)
    for interval in annotations.get("direction_reviewed_intervals", []):
        review_scope[max(0, int(interval["start_frame"])):min(len(times), int(interval["end_frame"]) + 1)] = True
    valid = session.get("valid", np.ones(len(times), dtype=bool))
    ignored = session.get("ignored_frames", np.zeros(len(times), dtype=bool))
    direction_classes = config["inference"]["action_event_groups"]["direction"]
    direction_channels = [config["labels"]["action_classes"].index(name) for name in direction_classes]
    if (session.get("kind") == "one_shot" and session.get("sample_type") != "negative"
            and session.get("reviewed") and "mask" in session
            and config["labels"].get("action_one_shot_direction_background", False)):
        review_scope |= session["mask"][:, direction_channels].all(axis=1)
    reviewed = review_scope & valid & ~ignored
    if "mask" in session:
        reviewed &= session["mask"][:, direction_channels].all(axis=1)
    excluded = np.asarray(ignored, dtype=bool).copy()
    for interval in annotations.get("direction_ignored_intervals", []):
        excluded[max(0, int(interval["start_frame"])):
                 min(len(times), int(interval["end_frame"]) + 1)] = True
    reviewed &= ~excluded
    duration = float(np.diff(times)[reviewed[:-1] & reviewed[1:]].sum()) if len(times) > 1 else 0.0
    tolerance = float(config["evaluation"].get("waving_start_match_tolerance_seconds", 1.5))
    truths = []
    for item in annotations.get("waving_intervals", []):
        start, end = int(item["start_frame"]), int(item["end_frame"])
        # Missing observations exclude frames, not the whole confirmed gesture.
        if (0 <= start <= end < len(times) and review_scope[start:end + 1].all()
                and not excluded[start:end + 1].any() and reviewed[start:end + 1].any()):
            truths.append((start, end, float(item.get("start_time", times[start])),
                           float(item.get("end_time", times[end]))))
    starts = [event for event in trace["waving_events"] if event.label == "waving_start"
              and event.frame_index is not None and reviewed[min(max(event.frame_index, 0), len(times) - 1)]]
    ends = [event for event in trace["waving_events"] if event.label == "waving_end"]
    unmatched = set(range(len(truths)))
    matches = []
    for event in starts:
        choices = [index for index in unmatched if abs(event.time_seconds - truths[index][2]) <= tolerance]
        if choices:
            index = min(choices, key=lambda value: abs(event.time_seconds - truths[value][2]))
            unmatched.remove(index)
            start, end, true_start, true_end = truths[index]
            second_event = None
            directions = sorted((item for item in session.get("intervals", [])
                                 if item["label"] in WavingFSM.OPPOSITE
                                 and start <= int(item["end_frame"]) <= end),
                                key=lambda item: item.get("end_time", times[int(item["end_frame"])]))
            for first, second in zip(directions, directions[1:]):
                if second["label"] == WavingFSM.OPPOSITE[first["label"]]:
                    second_event = float(second.get("end_time", times[int(second["end_frame"])]))
                    break
            active_start = second_event if second_event is not None else event.time_seconds
            edges = ((times[:-1] >= active_start) & (times[1:] <= true_end)
                     & reviewed[:-1] & reviewed[1:])
            active_seconds = float(np.diff(times)[edges & trace["active"][:-1]].sum())
            total_seconds = float(np.diff(times)[edges].sum())
            end_candidates = [item for item in ends if item.time_seconds >= event.time_seconds]
            ending = min(end_candidates, key=lambda item: abs(item.time_seconds - true_end)) if end_candidates else None
            matches.append({"truth_start_frame": start, "start_event_frame": event.frame_index,
                            "start_delay_seconds": event.time_seconds - true_start,
                            "active_coverage": active_seconds / total_seconds if total_seconds > 0 else None,
                            "true_active_seconds": total_seconds,
                            "end_event_time_seconds": ending.time_seconds if ending else None,
                            "stop_delay_seconds": ending.time_seconds - true_end if ending else None})
        else:
            matches.append({"truth_start_frame": None, "start_event_frame": event.frame_index,
                            "start_delay_seconds": None, "false_positive": True})
    tp = len(truths) - len(unmatched)
    fp = len(starts) - tp
    fn = len(unmatched)
    return {"reviewed_duration_seconds": duration, "start_match_tolerance_seconds": tolerance,
            "true_waving_intervals": len(truths), "starts": len(starts),
            "truth_start_frames": [item[0] for item in truths],
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "false_starts_per_minute": fp / (duration / 60) if duration > 0 else 0.0,
            "mean_active_coverage": float(np.mean([item["active_coverage"] for item in matches
                                                    if item.get("active_coverage") is not None]))
            if any(item.get("active_coverage") is not None for item in matches) else None,
            "matches": matches}
