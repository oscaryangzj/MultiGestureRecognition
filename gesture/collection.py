import copy
import random


def alternating_action_prompts(pairs, initial_pose):
    if pairs <= 0 or initial_pose not in ("opened", "closed"):
        raise ValueError("Action collection needs positive pairs and an opened/closed initial pose")
    order = ("close_hand", "open_hand") if initial_pose == "opened" else ("open_hand", "close_hand")
    return [{"gesture": gesture} for _ in range(pairs) for gesture in order]


def prepare_action_session(template, sample_type, count=1, rng=None, negative_task=None):
    if sample_type not in ("positive", "negative") or count <= 0:
        raise ValueError("Choose positive/negative and a positive collection count")
    session = copy.deepcopy(template[sample_type])
    session["sample_type"] = sample_type
    if sample_type == "positive":
        session["pairs"] = count
        lower, upper = map(float, session["hold_seconds_range"])
        if not 0 < lower <= upper:
            raise ValueError("HOLD range must satisfy 0 < min <= max")
        rng = rng or random
        prompts = alternating_action_prompts(count, session["initial_pose"])
        for prompt in prompts:
            gesture = prompt["gesture"]
            prompt.update(sample_type=sample_type,
                          instruction=gesture.replace("_", " ").upper() + " ONCE, THEN HOLD",
                          instruction_zh="张开一次，然后保持" if gesture == "open_hand" else "握拳一次，然后保持",
                          duration_seconds=float(session["durations_seconds"][gesture]),
                          hold_seconds=rng.uniform(lower, upper))
    else:
        task = next((task for task in session["tasks"] if task["gesture"] == negative_task), None)
        if task is None:
            raise ValueError("Choose a negative task from the collection plan")
        session["selected_task"] = negative_task
        session["tasks"] = [task]
        prompts = [dict(task, sample_type=sample_type)]
    session["prompts"] = prompts
    return session


def prepare_periodic_session(template, sample_type, count=1, hand="right", motion="wrist",
                             rng=None, negative_task=None):
    if sample_type not in ("positive", "negative") or count <= 0:
        raise ValueError("Choose periodic positive/negative and a positive count")
    session = copy.deepcopy(template[sample_type])
    session.update(sample_type=sample_type, hand=hand, motion=motion if sample_type == "positive" else None,
                   initial_pose="opened" if sample_type == "positive" else None)
    rng = rng or random
    if sample_type == "positive":
        session["bouts"] = count
        source_cues = list(session["speed_cues"])
        cues = []
        prompts = []
        for index in range(count):
            if index % len(source_cues) == 0:
                cues = source_cues.copy()
                rng.shuffle(cues)
            cue = cues[index % len(source_cues)]
            duration = rng.uniform(*session["wave_seconds_range"])
            prompts.append({
                "gesture": "waving", "sample_type": "positive", "hand": hand, "motion": motion,
                "speed_cue": cue["id"], "instruction": cue["instruction"],
                "instruction_zh": cue["instruction_zh"], "duration_seconds": duration,
                "prepare_seconds": float(session["prepare_seconds"]),
                "tail_hold_seconds": float(session["tail_hold_seconds"]),
                "rest_seconds": float(session["rest_seconds"]),
            })
    else:
        task = next((item for item in session["tasks"] if item["gesture"] == negative_task), None)
        if task is None:
            raise ValueError("Choose a negative task from the periodic collection plan")
        session["selected_task"] = negative_task
        session["tasks"] = [task]
        prompts = [dict(task, sample_type="negative")]
    session["prompts"] = prompts
    return session


def collection_phases(session):
    """Capture schedule; the selected timings are persisted with each session."""
    action = session["target_model"] == "action"
    negative = session.get("sample_type") == "negative"
    phases, cursor = [], 0.0

    def add(kind, gesture, duration, prompt_index=None, instruction="", pose=None, instruction_zh=""):
        nonlocal cursor
        if duration <= 0:
            raise ValueError("Collection phase durations must be positive")
        phases.append({"phase": kind, "gesture": gesture, "begin": cursor,
                       "end": cursor + duration, "prompt_index": prompt_index,
                       "instruction": instruction, "instruction_zh": instruction_zh, "pose": pose})
        cursor += duration

    if session.get("kind") == "periodic":
        for index, prompt in enumerate(session["prompts"]):
            if prompt["sample_type"] == "negative":
                add("prepare", prompt["prepare_pose"], float(session["prepare_seconds"]), index)
                add("negative", prompt["gesture"], float(prompt["duration_seconds"]), index,
                    prompt["instruction"], prompt["prepare_pose"], prompt["instruction_zh"])
                continue
            add("prepare", "opened", float(prompt["prepare_seconds"]), index)
            add("wave", "waving", float(prompt["duration_seconds"]), index,
                prompt["instruction"], "opened", prompt["instruction_zh"])
            add("hold", "opened", float(prompt["tail_hold_seconds"]), index)
            add("rest", "REST", float(prompt["rest_seconds"]), index)
        return phases

    if action and not negative:
        add("hold", session["initial_pose"], float(session["prepare_seconds"]))
    for index, prompt in enumerate(session["prompts"]):
        gesture = prompt["gesture"]
        if negative:
            add("prepare", prompt["prepare_pose"], float(session["prepare_seconds"]), index)
        duration = float(prompt["duration_seconds"]) if action else float(session["durations_seconds"][gesture])
        add("negative" if negative else "action" if action else "state", gesture, duration,
            index, prompt.get("instruction", ""), prompt.get("prepare_pose"), prompt.get("instruction_zh", ""))
        if action and not negative:
            pose = "closed" if gesture == "close_hand" else "opened"
            add("hold", pose, float(prompt["hold_seconds"]), index)
        elif not action and index < len(session["prompts"]) - 1:
            add("rest", "REST", float(session["gap_seconds"]))
    return phases
