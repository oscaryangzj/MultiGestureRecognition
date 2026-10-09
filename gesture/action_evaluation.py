import numpy as np

from gesture.action_data import action_pulse_frames
from gesture.action_decoding import decode_action_session
from gesture.waving_fsm import trace_waving_session, waving_fsm_metrics


def event_peaks(scores, threshold):
    """One peak per above-threshold run, retained as a model diagnostic."""
    above = np.isfinite(scores) & (scores >= threshold)
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    return [int(start + np.argmax(scores[start:end]))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _match_events(session, config, events):
    classes = config["labels"]["action_classes"]
    tolerance = float(config["evaluation"]["action_match_tolerance_seconds"])
    times = session["times"]
    excluded = session.get("ignored_frames", np.zeros(len(times), bool)).copy()
    excluded |= ~session.get("valid", np.ones(len(times), bool))
    observed = ~excluded[:-1] & ~excluded[1:]
    minutes = float(np.diff(times)[observed].sum()) / 60
    per_class = {}
    for channel, name in enumerate(classes):
        channel_valid = ~excluded
        if "mask" in session:
            channel_valid = channel_valid & session["mask"][:, channel]
        channel_observed = channel_valid[:-1] & channel_valid[1:]
        channel_minutes = float(np.diff(times)[channel_observed].sum()) / 60
        truth = []
        for interval in session["intervals"]:
            end = int(interval["end_frame"])
            if interval["label"] != name or excluded[end]:
                continue
            if "mask" in session:
                pulse = action_pulse_frames(interval, times, config)
                if not session["mask"][pulse, channel].any():
                    continue
            truth.append((end, interval.get("end_time", times[end])))
        detected = [event for event in events if event.label == name and
                    (event.frame_index is None or
                     (0 <= event.frame_index < len(channel_valid) and channel_valid[event.frame_index]))]
        unmatched = set(range(len(truth)))
        delays, matches = [], []
        for event in detected:
            choices = [index for index in unmatched
                       if abs(event.time_seconds - truth[index][1]) <= tolerance]
            if not choices:
                continue
            index = min(choices, key=lambda item: abs(event.time_seconds - truth[item][1]))
            unmatched.remove(index)
            delay = float(event.time_seconds - truth[index][1])
            delays.append(delay)
            matches.append({"truth_frame": truth[index][0], "event_frame": event.frame_index,
                            "event_time_seconds": event.time_seconds, "score": event.score,
                            "delay_seconds": delay})
        tp, fp, fn = len(delays), len(detected) - len(delays), len(unmatched)
        per_class[name] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "false_positives_per_minute": fp / channel_minutes if channel_minutes > 0 else 0.0,
            "duration_seconds": channel_minutes * 60,
            "mean_delay_seconds": float(np.mean(delays)) if delays else None,
            "mean_absolute_delay_seconds": float(np.mean(np.abs(delays))) if delays else None,
            "matches": matches,
            "events": [{"frame_index": event.frame_index, "time_seconds": event.time_seconds,
                        "score": event.score} for event in detected],
        }
    return {"match_tolerance_seconds": tolerance, "duration_seconds": minutes * 60,
            "per_class": per_class}


def action_decoder_metrics(session, predictions, config):
    events = decode_action_session(session, predictions, config["labels"]["action_classes"], config)
    result = _match_events(session, config, events)
    result["threshold"] = float(config["inference"]["action_threshold"])
    return result


def action_event_metrics(session, predictions, config):
    classes = config["labels"]["action_classes"]
    threshold = float(config["inference"]["action_threshold"])
    tolerance = float(config["evaluation"]["action_match_tolerance_seconds"])
    times = session["times"]
    prediction_times = session.get("available_times", times)
    excluded = session.get("ignored_frames", np.zeros(len(times), bool)).copy()
    excluded |= ~session.get("valid", np.ones(len(times), bool))
    observed = ~excluded[:-1] & ~excluded[1:]
    minutes = float(np.diff(times)[observed].sum()) / 60
    raw_predictions = np.asarray(predictions).copy()
    predictions = raw_predictions.copy()
    predictions[excluded] = np.nan
    result = {"threshold": threshold, "match_tolerance_seconds": tolerance,
              "duration_seconds": minutes * 60, "per_class": {}}
    for channel, name in enumerate(classes):
        channel_valid = ~excluded
        if "mask" in session:
            channel_valid = channel_valid & session["mask"][:, channel]
        channel_observed = channel_valid[:-1] & channel_valid[1:]
        channel_minutes = float(np.diff(times)[channel_observed].sum()) / 60
        truth = []
        for interval in session["intervals"]:
            end = int(interval["end_frame"])
            if interval["label"] != name or excluded[end]:
                continue
            if "mask" in session:
                pulse = action_pulse_frames(interval, times, config)
                if not session["mask"][pulse, channel].any():
                    continue
            truth.append((end, interval.get("end_time", times[end])))
        channel_predictions = predictions[:, channel].copy()
        channel_predictions[~channel_valid] = np.nan
        peaks = event_peaks(channel_predictions, threshold)
        unmatched = set(range(len(truth)))
        delays, matches = [], []
        for peak in peaks:
            choices = [index for index in unmatched
                       if abs(prediction_times[peak] - truth[index][1]) <= tolerance]
            if not choices:
                continue
            index = min(choices, key=lambda item: abs(prediction_times[peak] - truth[item][1]))
            unmatched.remove(index)
            delay = float(prediction_times[peak] - truth[index][1])
            delays.append(delay)
            matches.append({"truth_frame": truth[index][0], "peak_frame": peak, "delay_seconds": delay})
        tp, fp, fn = len(delays), len(peaks) - len(delays), len(unmatched)
        result["per_class"][name] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "false_positives_per_minute": fp / channel_minutes if channel_minutes > 0 else 0.0,
            "duration_seconds": channel_minutes * 60,
            "mean_delay_seconds": float(np.mean(delays)) if delays else None,
            "mean_absolute_delay_seconds": float(np.mean(np.abs(delays))) if delays else None,
            "matches": matches, "peak_frames": peaks,
        }
    result["decoded_events"] = action_decoder_metrics(session, raw_predictions, config)
    directions = set(config.get("inference", {}).get("action_event_groups", {}).get("direction", ()))
    if directions.issubset(classes):
        trace = trace_waving_session(session, raw_predictions, classes, config)
        result["waving_fsm"] = waving_fsm_metrics(session, trace, config)
    return result


def action_failure_cases(session, metrics, config):
    """List unmatched atomic and waving events with original video frame locations."""
    source = session.get("source_session", session)
    source_times = source["times"]

    def source_frame(time_seconds):
        return max(0, min(int(np.searchsorted(source_times, time_seconds, side="left")), len(source_times) - 1))

    failures = []
    classes = metrics["decoded_events"]["per_class"]
    for label, row in classes.items():
        matched_truth = {item["truth_frame"] for item in row["matches"]}
        matched_event_times = {item["event_time_seconds"] for item in row["matches"]}
        channel = list(metrics["per_class"]).index(label)
        for interval in session["intervals"]:
            end = int(interval["end_frame"])
            if interval["label"] != label:
                continue
            truth_time = float(interval.get("end_time", session["times"][end]))
            pulse_mask = session["mask"][action_pulse_frames(interval, session["times"], config), channel].any()
            if pulse_mask and end not in matched_truth:
                failures.append({"kind": "missed_atomic_event", "label": label,
                                 "frame_index": source_frame(truth_time), "time_seconds": truth_time})
        for event in row["events"]:
            if event["time_seconds"] not in matched_event_times:
                failures.append({"kind": "false_atomic_event", "label": label,
                                 "frame_index": source_frame(event["time_seconds"]),
                                 "time_seconds": event["time_seconds"]})
    wave = metrics.get("waving_fsm")
    if wave is None:
        return failures
    wave_matches = wave["matches"]
    matched_wave_starts = {item["truth_start_frame"] for item in wave_matches
                           if item.get("truth_start_frame") is not None}
    annotations = session.get("annotations", {})
    evaluated_wave_starts = set(wave["truth_start_frames"])
    for interval in annotations.get("waving_intervals", []):
        start = int(interval["start_frame"])
        if start not in evaluated_wave_starts:
            continue
        truth_time = float(interval.get("start_time", session["times"][start]))
        if start not in matched_wave_starts:
            failures.append({"kind": "missed_waving_start", "label": "waving_start",
                             "frame_index": source_frame(truth_time), "time_seconds": truth_time})
    for item in wave_matches:
        if item.get("false_positive"):
            frame = int(item["start_event_frame"])
            time_seconds = float(session.get("available_times", session["times"])[frame])
            failures.append({"kind": "false_waving_start", "label": "waving_start",
                             "frame_index": source_frame(time_seconds), "time_seconds": time_seconds})
    return failures


def aggregate_action_metrics(evaluations):
    """Pool complete sessions, including false alarms from background-only recordings."""
    if not evaluations:
        raise ValueError("No Action sessions to aggregate")
    duration = sum(item["duration_seconds"] for item in evaluations.values())
    minutes = duration / 60
    classes = next(iter(evaluations.values()))["per_class"]

    def summarize(rows):
        tp = sum(row["true_positive"] for row in rows)
        fp = sum(row["false_positive"] for row in rows)
        fn = sum(row["false_negative"] for row in rows)
        delays = [match["delay_seconds"] for row in rows for match in row["matches"]]
        exposure = sum(row.get("duration_seconds", 0.0) for row in rows)
        exposure_minutes = exposure / 60 if exposure else minutes
        return {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "false_positives_per_minute": fp / exposure_minutes if exposure_minutes > 0 else 0.0,
            "mean_delay_seconds": float(np.mean(delays)) if delays else None,
            "mean_absolute_delay_seconds": float(np.mean(np.abs(delays))) if delays else None,
        }

    def summarize_events(rows):
        event_duration = sum(row["duration_seconds"] for row in rows)
        names = next(iter(rows))["per_class"]

        def event_summary(items):
            return summarize(items)

        by_class = {name: event_summary([row["per_class"][name] for row in rows]) for name in names}
        overall = event_summary([item for row in rows for item in row["per_class"].values()])
        return {"duration_seconds": event_duration, "per_class": by_class, "overall": overall,
                "macro_f1": float(np.mean([row["f1"] for row in by_class.values()]))}

    per_class = {name: summarize([item["per_class"][name] for item in evaluations.values()])
                 for name in classes}
    overall = summarize([row for item in evaluations.values() for row in item["per_class"].values()])
    decoded_rows = [item["decoded_events"] for item in evaluations.values() if "decoded_events" in item]
    waving_rows = [item["waving_fsm"] for item in evaluations.values() if item.get("waving_fsm") is not None]
    waving = None
    if waving_rows:
        wave_duration = sum(item["reviewed_duration_seconds"] for item in waving_rows)
        wave_tp = sum(item["true_positive"] for item in waving_rows)
        wave_fp = sum(item["false_positive"] for item in waving_rows)
        wave_fn = sum(item["false_negative"] for item in waving_rows)
        coverages = [(item["mean_active_coverage"], item["true_waving_intervals"] - item["false_negative"])
                     for item in waving_rows if item["mean_active_coverage"] is not None]
        waving = {
            "reviewed_duration_seconds": wave_duration,
            "true_positive": wave_tp, "false_positive": wave_fp, "false_negative": wave_fn,
            "precision": wave_tp / (wave_tp + wave_fp) if wave_tp + wave_fp else 0.0,
            "recall": wave_tp / (wave_tp + wave_fn) if wave_tp + wave_fn else 0.0,
            "f1": 2 * wave_tp / (2 * wave_tp + wave_fp + wave_fn) if 2 * wave_tp + wave_fp + wave_fn else 0.0,
            "false_starts_per_minute": wave_fp / (wave_duration / 60) if wave_duration > 0 else 0.0,
            "mean_active_coverage": (sum(value * count for value, count in coverages) / sum(count for _, count in coverages)
                                     if coverages and sum(count for _, count in coverages) else None),
        }
    return {"duration_seconds": duration, "per_class": per_class, "overall": overall,
            "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
            "decoded_events": summarize_events(decoded_rows) if decoded_rows else None,
            "waving_fsm": waving}
