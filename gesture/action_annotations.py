from gesture.action_data import validate_action_intervals
from gesture.action_candidates import propose_action_interval


class ActionAnnotationEditor:
    def __init__(self, annotations, frame_count, detected, classes, session=None, config=None):
        self.annotations = annotations
        self.intervals = annotations.setdefault("action_intervals", [])
        self.frame_count, self.detected, self.classes = frame_count, detected, classes
        validate_action_intervals(self.intervals, frame_count, classes)
        self.label = classes[0]
        self.changed = False
        self.message = "Replay confirmed events and supervision."
        self.opportunities = session["opportunities"] if session is not None else []
        self.config = config
        self.prompt_position = 0
        if self.opportunities:
            self._initialize_drafts(session)

    def _initialize_drafts(self, session):
        self.annotations["action_review_mode"] = "prompts"
        candidates = self.annotations.setdefault("action_candidates", [])
        ignored = self.annotations.setdefault("action_ignored_intervals", [])
        self.annotations.setdefault("action_background_prompts", [])
        for opportunity in self.opportunities:
            if opportunity.get("sample_type") == "negative":
                continue
            index = opportunity["prompt_index"]
            for interval in self.intervals:
                if "prompt_index" not in interval and interval["label"] == opportunity["label"] and opportunity["start_frame"] <= interval["end_frame"] <= opportunity["end_frame"]:
                    interval["prompt_index"] = index
            if not any(item.get("prompt_index") == index for item in [*self.intervals, *ignored, *candidates]):
                candidates.append(propose_action_interval(session, opportunity, self.config))
        self.prompt_position = next((position for position in range(len(self.opportunities)) if self.status(position) == "PENDING"), 0)
        self.select_prompt(self.prompt_position)
        self._update_reviewed()
        self.changed = True
        self.message = "Review each prompt. Enter confirms; X cancels. Negative real events: 1/2, I/O, E ADD."

    @property
    def opportunity(self):
        return self.opportunities[self.prompt_position]

    @property
    def is_negative(self):
        return bool(self.opportunities) and self.opportunity.get("sample_type") == "negative"

    def select_prompt(self, position):
        self.prompt_position = max(0, min(len(self.opportunities) - 1, position))
        self.label = self.classes[0] if self.is_negative else self.opportunity["label"]

    def status(self, position=None):
        opportunity = self.opportunities[self.prompt_position if position is None else position]
        index = opportunity["prompt_index"]
        if any(item.get("prompt_index") == index for item in self.annotations["action_ignored_intervals"]):
            return "IGNORE"
        if opportunity.get("sample_type") == "negative":
            return "BACKGROUND" if index in self.annotations["action_background_prompts"] else "PENDING"
        return "CONFIRMED" if any(item.get("prompt_index") == index for item in self.intervals) else "PENDING"

    def current_interval(self):
        index = self.opportunity["prompt_index"]
        source = self.annotations["action_candidates"] if self.is_negative else [*self.intervals, *self.annotations["action_candidates"]]
        return next((item for item in source if item.get("prompt_index") == index), None)

    def context_start(self):
        return self.opportunity["start_frame"] if self.is_negative else max(0, self.opportunity["start_frame"] - int(self.config["annotation"]["action_lookback_frames"]))

    def focus_frame(self):
        interval = self.current_interval()
        if interval and interval["start_frame"] is not None:
            return max(self.context_start(), int(interval["start_frame"]) - int(self.config["annotation"]["action_lookback_frames"]))
        return self.context_start()

    def _update_reviewed(self):
        self.annotations["action_reviewed"] = all(self.status(position) != "PENDING" for position in range(len(self.opportunities)))

    def _unreview_negative(self, index):
        background = self.annotations["action_background_prompts"]
        background[:] = [item for item in background if item != index]

    def apply(self, action, frame):
        if not self.opportunities:
            return
        if action in ("previous_prompt", "next_prompt"):
            self.select_prompt(self.prompt_position + (-1 if action == "previous_prompt" else 1))
            return
        opportunity = self.opportunity
        index = opportunity["prompt_index"]
        candidates = self.annotations["action_candidates"]
        ignored = self.annotations["action_ignored_intervals"]
        interval = self.current_interval()
        if action.startswith("label_") and self.is_negative:
            self.label = self.classes[int(action.split("_")[1])]
            if interval:
                interval["label"] = self.label
                self._unreview_negative(index)
            else:
                self.message = f"Selected {self.label}. I START, O FINISH, E ADD EVENT."
                return
        elif action in ("start", "finish"):
            if not self.context_start() <= frame <= opportunity["end_frame"] or not self.detected[frame]:
                self.message = "Choose a visible frame within this prompt."
                return
            interval = dict(interval) if interval else {"prompt_index": index, "label": self.label, "start_frame": None, "end_frame": None}
            if self.is_negative:
                self._unreview_negative(index)
            else:
                self.intervals[:] = [item for item in self.intervals if item.get("prompt_index") != index]
            ignored[:] = [item for item in ignored if item.get("prompt_index") != index]
            candidates[:] = [item for item in candidates if item.get("prompt_index") != index]
            interval["start_frame" if action == "start" else "end_frame"] = frame
            candidates.append(interval)
            self.message = ("Draft adjusted. E ADD EVENT, then Enter ACCEPT BLOCK." if self.is_negative
                            else "Draft adjusted. Enter confirms this action.")
        elif action == "confirm" and self.is_negative:
            if interval:
                self.message = "Save this real event with E, or Backspace to discard the draft, before confirming the block."
                return
            self._unreview_negative(index)
            self.annotations["action_background_prompts"].append(index)
            ignored[:] = [item for item in ignored if item.get("prompt_index") != index]
            self.message = "Negative block reviewed: valid background plus any marked real events."
        elif action == "confirm" or (action == "add_event" and self.is_negative):
            if interval is None or interval["start_frame"] is None or interval["end_frame"] is None:
                self.message = "Mark I START and O FINISH first."
                return
            others = self.intervals if self.is_negative else [item for item in self.intervals if item.get("prompt_index") != index]
            try:
                validate_action_intervals([*others, interval], self.frame_count, self.classes)
            except ValueError:
                self.message = "Check start <= finish and overlapping events before saving."
                return
            self.intervals[:] = sorted([*others, dict(interval)], key=lambda item: item["start_frame"])
            candidates[:] = [item for item in candidates if item.get("prompt_index") != index]
            ignored[:] = [item for item in ignored if item.get("prompt_index") != index]
            if self.is_negative:
                self._unreview_negative(index)
            self.message = f"Saved {interval['label']}."
        elif action == "delete_event" and self.is_negative:
            if interval:
                candidates[:] = [item for item in candidates if item.get("prompt_index") != index]
            else:
                selected = next((item for item in self.intervals if item.get("prompt_index") == index and item["start_frame"] <= frame <= item["end_frame"]), None)
                if selected is None:
                    self.message = "Seek a saved event to delete it."
                    return
                self.intervals.remove(selected)
            self._unreview_negative(index)
            self.message = "Event/draft removed. Review this block again."
        elif action == "cancel":
            self.intervals[:] = [item for item in self.intervals if item.get("prompt_index") != index]
            candidates[:] = [item for item in candidates if item.get("prompt_index") != index]
            ignored[:] = [item for item in ignored if item.get("prompt_index") != index]
            self._unreview_negative(index)
            ignored.append({"prompt_index": index, "start_frame": self.context_start(), "end_frame": opportunity["end_frame"]})
            self.message = "Cancelled prompt: IGNORE."
        else:
            return
        self._update_reviewed()
        self.changed = True
        if action in ("confirm", "cancel"):
            pending = [position for position in range(len(self.opportunities)) if self.status(position) == "PENDING"]
            if pending:
                self.select_prompt(next((position for position in pending if position > self.prompt_position), pending[0]))
        if self.annotations["action_reviewed"]:
            self.message += " All prompts reviewed."
