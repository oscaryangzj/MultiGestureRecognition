import numpy as np

from gesture.action_data import action_pulse_frames


def event_peaks(scores, threshold):
    """One peak per above-threshold run, used only for evaluation."""
    above = np.isfinite(scores) & (scores >= threshold)
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    return [int(start + np.argmax(scores[start:end]))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


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
    predictions = np.asarray(predictions).copy()
    predictions[excluded] = np.nan
    result = {"threshold": threshold, "match_tolerance_seconds": tolerance, "duration_seconds": minutes * 60, "per_class": {}}
    for channel, name in enumerate(classes):
        truth = []
        for item in session["intervals"]:
            end = int(item["end_frame"])
            if item["label"] != name or excluded[end]:
                continue
            if "mask" in session:
                pulse = action_pulse_frames(item, times, config)
                if not session["mask"][pulse, channel].any():
                    continue
            truth.append((end, item.get("end_time", times[end])))
        peaks = event_peaks(predictions[:, channel], threshold)
        unmatched = set(range(len(truth)))
        delays = []
        matches = []
        for peak in peaks:
            candidates = [index for index in unmatched if abs(prediction_times[peak] - truth[index][1]) <= tolerance]
            if not candidates:
                continue
            index = min(candidates, key=lambda index: abs(prediction_times[peak] - truth[index][1]))
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
            "false_positives_per_minute": fp / minutes if minutes > 0 else 0.0,
            "mean_delay_seconds": float(np.mean(delays)) if delays else None,
            "mean_absolute_delay_seconds": float(np.mean(np.abs(delays))) if delays else None,
            "matches": matches, "peak_frames": peaks,
        }
    return result


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
        return {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "false_positives_per_minute": fp / minutes if minutes > 0 else 0.0,
            "mean_delay_seconds": float(np.mean(delays)) if delays else None,
            "mean_absolute_delay_seconds": float(np.mean(np.abs(delays))) if delays else None,
        }

    per_class = {name: summarize([item["per_class"][name] for item in evaluations.values()])
                 for name in classes}
    overall = summarize([row for item in evaluations.values() for row in item["per_class"].values()])
    return {"duration_seconds": duration, "per_class": per_class, "overall": overall,
            "macro_f1": float(np.mean([row["f1"] for row in per_class.values()]))}
