import numpy as np


def action_opportunities(prompts, frame_count, session=None):
    recorded = [(index, prompt) for index, prompt in enumerate(prompts)
                if prompt.get("start_frame") is not None and prompt.get("end_frame") is not None]
    periodic = bool(session and session.get("kind") == "periodic")
    result = []
    for position, (index, prompt) in enumerate(recorded):
        negative = prompt.get("sample_type") == "negative"
        end = (int(prompt["end_frame"]) if negative else
               int(recorded[position + 1][1]["start_frame"]) - 1 if position + 1 < len(recorded) else frame_count - 1)
        if periodic and not negative:
            hold_end = prompt.get("hold_end_frame")
            end = min(frame_count - 1, int(prompt["end_frame"] if hold_end is None else hold_end))
        result.append({"prompt_index": index, "label": prompt["gesture"],
                       "sample_type": "negative" if negative else "positive",
                       "instruction": prompt.get("instruction", ""),
                       "start_frame": int(prompt["start_frame"]), "end_frame": end,
                       "prompt_end_frame": int(prompt["end_frame"]),
                       "activity_end_frame": int(prompt["end_frame"])})
    return result


def stable_starts(flags, start, end, required):
    return [frame for frame in range(start, end - required + 2)
            if flags[frame:frame + required].all()]


def propose_action_interval(session, opportunity, config):
    """Offline draft only. A human must confirm it before any positive supervision."""
    settings = config["annotation"]
    start, end = opportunity["start_frame"], opportunity["end_frame"]
    lookback = max(0, start - int(settings["action_lookback_frames"]))
    stable = int(settings["action_stable_frames"])
    source, target = ("opened", "closed") if opportunity["label"] == "close_hand" else ("closed", "opened")
    state_classes = config["features"]["action_state_classes"]
    scores = session["features"][:, -len(state_classes):]
    threshold = float(settings["action_state_threshold"])
    source_flags = session["valid"] & (scores[:, state_classes.index(source)] >= threshold)
    target_flags = session["valid"] & (scores[:, state_classes.index(target)] >= threshold)
    source_runs = stable_starts(source_flags, lookback, end, stable)
    draft = {"prompt_index": opportunity["prompt_index"], "label": opportunity["label"],
             "start_frame": None, "end_frame": None}
    before_prompt = [frame for frame in source_runs if frame + stable <= start]
    if not source_runs:
        return draft
    baseline_start = before_prompt[-1] if before_prompt else source_runs[0]
    baseline_end = baseline_start + stable - 1
    completed = stable_starts(target_flags, max(start, baseline_end + 1), end, stable)
    if not completed:
        return draft
    finish = completed[0]
    if session["segments"][baseline_end] != session["segments"][finish + stable - 1]:
        return draft
    local = session["features"][:, :42].reshape(-1, 21, 2)
    baseline = local[baseline_start:baseline_end + 1].mean(axis=0)
    movement = np.linalg.norm(local - baseline, axis=2).mean(axis=1)
    moving = session["valid"] & (movement >= float(settings["action_motion_threshold"]))
    candidates = stable_starts(moving, baseline_end + 1, finish + stable - 1,
                               int(settings["action_motion_confirm_frames"]))
    if candidates and candidates[0] <= finish:
        draft["start_frame"], draft["end_frame"] = candidates[0], finish
    return draft


def action_excluded_frames(annotations, opportunities, frame_count, config, times=None, kind=None):
    """Unreviewed/cancelled regions and negative preparation never supervise."""
    regions = list(annotations.get("action_ignored_intervals", []))
    excluded = np.zeros(frame_count, bool)
    negative = [item for item in opportunities if item.get("sample_type") == "negative"]
    background = set(annotations.get("action_background_prompts", []))
    if negative and kind != "periodic":
        excluded[:] = True
        for item in negative:
            if item["prompt_index"] in background:
                excluded[item["start_frame"]:item["end_frame"] + 1] = False
    if annotations.get("action_review_mode") == "prompts" and kind not in ("periodic", "continuous"):
        resolved = {item["prompt_index"] for item in annotations.get("action_intervals", []) if "prompt_index" in item}
        resolved |= {item["prompt_index"] for item in regions if "prompt_index" in item}
        lookback = int(config["annotation"]["action_lookback_frames"])
        regions += [{"start_frame": max(0, item["start_frame"] - lookback), "end_frame": item["end_frame"]}
                    for item in opportunities
                    if item.get("sample_type") != "negative"
                    and item.get("label") != "waving"
                    and item["prompt_index"] not in resolved]
    if times is None:
        times = np.arange(frame_count) / float(config["data"]["action_sample_rate_hz"])
    for region in regions:
        start = max(0, int(region["start_frame"]))
        finish = min(frame_count - 1, int(region["end_frame"]))
        if "action_positive_frames" in config["labels"]:
            end = min(frame_count, finish + int(config["labels"]["action_positive_frames"]))
        elif "action_positive_end_offset_frames" in config["labels"]:
            end = min(frame_count, finish + int(config["labels"]["action_positive_end_offset_frames"]) + 1)
        else:
            tail = (float(config["labels"]["action_positive_delay_seconds"]) +
                    float(config["labels"]["action_positive_duration_seconds"]))
            end = int(np.searchsorted(times, times[finish] + tail, side="left"))
        excluded[start:end] = True
    return excluded
