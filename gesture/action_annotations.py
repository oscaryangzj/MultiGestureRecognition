from gesture.action_data import validate_action_intervals
from gesture.action_candidates import propose_action_interval
from gesture.periodic_candidates import propose_direction_events


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
        self.kind = session.get("kind", "one_shot") if session is not None else "one_shot"
        self.periodic = self.kind in ("periodic", "continuous")
        if self.kind == "continuous" and not self.opportunities:
            self.opportunities = [{"prompt_index": 0, "label": "continuous", "sample_type": "mixed",
                                  "start_frame": 0, "end_frame": frame_count - 1}]
        self.config = config
        self._session_data = session
        self.prompt_position = 0
        self.mode = annotations.get("periodic_edit_mode", "direction")
        self.range_start, self.range_end = None, None
        self.event_position = 0
        if self.opportunities:
            self._initialize_drafts(session)

    def _initialize_drafts(self, session):
        self.annotations["action_review_mode"] = "continuous" if self.kind == "continuous" else "prompts"
        candidates = self.annotations.setdefault("action_candidates", [])
        ignored = self.annotations.setdefault("action_ignored_intervals", [])
        self.annotations.setdefault("action_background_prompts", [])
        if self.periodic:
            self.annotations.setdefault("direction_candidates", [])
            self.annotations.setdefault("direction_reviewed_intervals", [])
            self.annotations.setdefault("direction_ignored_intervals", [])
            self.annotations.setdefault("one_shot_reviewed_intervals", [])
            self.annotations.setdefault("waving_intervals", [])
            self.annotations.setdefault("waving_candidates", [])
            self.annotations.setdefault("waving_reviewed_intervals", [])
            self.annotations.setdefault("periodic_reviewed_prompts", [])
        for opportunity in self.opportunities:
            if self.periodic or opportunity.get("sample_type") == "negative":
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
        self.message = "按段检查；Enter 确认，X 忽略本段。负样本中的真实开合用 1/2、I/O、E 补标。"

    @property
    def opportunity(self):
        return self.opportunities[self.prompt_position]

    @property
    def is_negative(self):
        return bool(self.opportunities) and self.opportunity.get("sample_type") == "negative"

    def select_prompt(self, position):
        position = max(0, min(len(self.opportunities) - 1, position))
        if position != self.prompt_position:
            self.range_start, self.range_end = None, None
            self.event_position = 0
            if self.periodic:
                self.message = (f"Task {position + 1}: set its WAVING start/end with I/O, then E and Enter."
                                if self.mode == "waving" else f"Task {position + 1}: review this task's events and background.")
        self.prompt_position = position
        self.label = (self.classes[2] if self.periodic and self.mode == "direction" and len(self.classes) > 2 else
                      self.classes[0] if self.is_negative or (self.periodic and self.mode == "one_shot") else
                      "waving" if self.periodic and self.mode == "waving" else self.opportunity["label"])

    def select_prompt_at(self, frame):
        for position, opportunity in enumerate(self.opportunities):
            if opportunity["start_frame"] <= frame <= opportunity.get("prompt_end_frame", opportunity["end_frame"]):
                self.select_prompt(position)
                break

    def status(self, position=None):
        opportunity = self.opportunities[self.prompt_position if position is None else position]
        index = opportunity["prompt_index"]
        if not self.periodic and any(item.get("prompt_index") == index for item in self.annotations["action_ignored_intervals"]):
            return "IGNORE"
        if opportunity.get("sample_type") == "negative" and not self.periodic:
            return "BACKGROUND" if index in self.annotations["action_background_prompts"] else "PENDING"
        if self.periodic:
            if self._range_covered("action_ignored_intervals", opportunity["start_frame"], opportunity["end_frame"], index):
                return "IGNORE"
            one_shot = self._range_covered("one_shot_reviewed_intervals", opportunity["start_frame"], opportunity["end_frame"], index)
            direction = self._range_covered("direction_reviewed_intervals", opportunity["start_frame"], opportunity["end_frame"], index)
            if one_shot and direction:
                return "REVIEWED"
            return "PENDING"
        return "CONFIRMED" if any(item.get("prompt_index") == index for item in self.intervals) else "PENDING"

    def current_interval(self):
        index = self.opportunity["prompt_index"]
        if self.periodic:
            sources = {
                "direction": (self.annotations.get("direction_candidates", []), self.intervals),
                "one_shot": (self.annotations["action_candidates"], self.intervals),
                "waving": (self.annotations.get("waving_candidates", []), self.annotations.get("waving_intervals", [])),
            }
            drafts, _confirmed = sources[self.mode]
            items = [item for item in drafts if item.get("prompt_index") == index]
            return items[self.event_position % len(items)] if items else None
        source = self.annotations["action_candidates"] if self.is_negative else [*self.intervals, *self.annotations["action_candidates"]]
        return next((item for item in source if item.get("prompt_index") == index), None)

    def context_bounds(self):
        if self.kind == "periodic":
            start = self.opportunity["start_frame"]
            metadata = self._session_data.get("metadata", {})
            for phase in metadata.get("phases", []):
                if (phase.get("prompt_index") == self.opportunity["prompt_index"]
                        and phase["phase"] == "rest" and phase.get("end_frame") is not None):
                    return start, min(self.frame_count - 1, int(phase["end_frame"]))
            if self.prompt_position + 1 < len(self.opportunities):
                end = self.opportunities[self.prompt_position + 1]["start_frame"] - 1
            else:
                end = self.frame_count - 1
            return start, end
        if self.kind == "continuous":
            return 0, self.frame_count - 1
        start = (self.opportunity["start_frame"] if self.is_negative else
                 max(0, self.opportunity["start_frame"] - int(self.config["annotation"]["action_lookback_frames"])))
        end = self.opportunity["end_frame"]
        return max(0, start), min(self.frame_count - 1, end)

    def context_start(self):
        return self.context_bounds()[0]

    def context_end(self):
        return self.context_bounds()[1]

    def selected_waving_interval(self):
        index = self.opportunity["prompt_index"]
        if self.mode == "waving":
            if self.range_start is not None or self.range_end is not None:
                if self.range_start is None or self.range_end is None or self.range_start > self.range_end:
                    return None
                return {"prompt_index": index, "start_frame": self.range_start, "end_frame": self.range_end}
            candidate = self.current_interval()
            if candidate is not None:
                return candidate
        intervals = [item for item in self.annotations.get("waving_intervals", [])
                     if item.get("prompt_index") == index]
        if not intervals:
            return None
        candidate = self.current_interval() if self.mode != "waving" else None
        anchor = self.range_start if candidate is None else candidate.get("extremum_frame", candidate.get("start_frame"))
        if anchor is not None:
            return next((item for item in intervals if item["start_frame"] <= anchor <= item["end_frame"]), None)
        return intervals[self.event_position % len(intervals)] if self.mode == "waving" else intervals[0]

    def focus_frame(self):
        interval = self.current_interval()
        if interval and interval["start_frame"] is not None:
            return max(self.context_start(), int(interval["start_frame"]) - int(self.config["annotation"]["action_lookback_frames"]))
        return self.opportunity["start_frame"] if self.periodic else self.context_start()

    def _update_reviewed(self):
        self.annotations["action_reviewed"] = all(
            self.status(position) not in ("PENDING",) for position in range(len(self.opportunities)))

    def _range_covered(self, field, start, end, prompt_index=None):
        covered = [False] * (end - start + 1)
        for interval in self.annotations.get(field, []):
            if prompt_index is not None and interval.get("prompt_index") not in (None, prompt_index):
                continue
            left = max(start, int(interval["start_frame"]))
            right = min(end, int(interval["end_frame"]))
            if left <= right:
                covered[left - start:right - start + 1] = [True] * (right - left + 1)
        return bool(covered) and all(covered)

    def _apply_periodic(self, action, frame):
        if action in ("previous_prompt", "next_prompt"):
            self.select_prompt(self.prompt_position + (-1 if action == "previous_prompt" else 1))
            self.event_position = 0
            return
        if action in ("mode_direction", "mode_one_shot", "mode_waving"):
            self.mode = action.removeprefix("mode_")
            self.range_start, self.range_end = None, None
            if self.mode == "one_shot":
                self.label = self.classes[0]
            elif self.mode == "direction" and len(self.classes) > 2:
                self.label = self.classes[2]
            elif self.mode == "waving":
                self.label = "waving"
            self.annotations["periodic_edit_mode"] = self.mode
            self.event_position = 0
            self.message = f"Edit mode: {self.mode}. Set I/O range, then E add or G generate directions."
            self.changed = True
            return
        if action in ("previous_event", "next_event"):
            self.event_position += -1 if action == "previous_event" else 1
            self.range_start, self.range_end = None, None
            return
        index = self.opportunity["prompt_index"]
        if action.startswith("label_"):
            self.label = self.classes[int(action.split("_")[1])]
            return
        if action in ("start", "finish"):
            if not 0 <= frame < self.frame_count or not self.detected[frame]:
                self.message = "Choose a visible frame in the video."
                return
            if action == "start":
                self.range_start, self.range_end = int(frame), None
            else:
                self.range_end = int(frame)
            interval = self.current_interval()
            if interval is not None and interval in (
                    self.annotations.get("direction_candidates", [])
                    + self.annotations.get("waving_candidates", [])
                    + self.annotations.get("action_candidates", [])):
                interval["start_frame" if action == "start" else "end_frame"] = int(frame)
            self.message = f"Range: [{self.range_start}, {self.range_end}]. E adds a draft; Enter confirms; G generates PCA drafts."
            self.changed = True
            return
        if action == "generate_directions":
            waving_range = self.selected_waving_interval()
            if waving_range is None or waving_range not in (
                    self.annotations["waving_intervals"] + self.annotations["waving_candidates"]):
                self.message = "Mark and save a WAVE interval first, then generate its direction drafts."
                return
            self.annotations["direction_candidates"] = [
                item for item in self.annotations["direction_candidates"] if item.get("prompt_index") != index]
            drafts = propose_direction_events(self._session_data, waving_range, self.config)
            self.annotations["direction_candidates"].extend(drafts)
            self.mode, self.event_position = "direction", 0
            self.annotations["periodic_edit_mode"] = self.mode
            self.range_start, self.range_end = None, None
            self.message = f"Generated {len(drafts)} direction drafts. Review each, edit I/O, then Enter to confirm."
            self.changed = True
            return
        if action in ("add_event", "add_wave"):
            if self.range_start is None or self.range_end is None or self.range_start > self.range_end:
                self.message = "Set a visible start with I and a finish with O first."
                return
            item = {"prompt_index": index, "start_frame": self.range_start,
                    "end_frame": self.range_end, "label": self.label}
            if action == "add_wave" or self.mode == "waving":
                self.annotations.setdefault("waving_candidates", []).append(
                    {"prompt_index": index, "start_frame": self.range_start, "end_frame": self.range_end})
                self.message = "Waving interval draft added; confirm with Enter."
            else:
                destination = self.annotations["direction_candidates"] if self.label in {
                    "left_to_right", "right_to_left"} else self.annotations["action_candidates"]
                destination.append(item)
                self.message = f"{self.label} event draft added; confirm with Enter."
            self.range_start, self.range_end = None, None
            self.changed = True
            return
        if action == "confirm":
            interval = self.current_interval()
            if interval is not None:
                if interval in self.annotations.get("direction_candidates", []):
                    try:
                        validate_action_intervals([*self.intervals, interval], self.frame_count, self.classes)
                    except (KeyError, TypeError, ValueError):
                        self.message = "Check the event label and start/end frames before confirming."
                        return
                    self.annotations["action_intervals"].append(dict(interval))
                    self.annotations["direction_candidates"].remove(interval)
                elif interval in self.annotations.get("waving_candidates", []):
                    self.annotations["waving_intervals"].append(dict(interval))
                    self.annotations["waving_candidates"].remove(interval)
                elif interval in self.annotations.get("action_candidates", []):
                    try:
                        validate_action_intervals([*self.intervals, interval], self.frame_count, self.classes)
                    except (KeyError, TypeError, ValueError):
                        self.message = "Check the event label and start/end frames before confirming."
                        return
                    self.annotations["action_intervals"].append(dict(interval))
                    self.annotations["action_candidates"].remove(interval)
                else:
                    self.event_position += 1
                    return
                self.annotations["action_intervals"].sort(key=lambda item: item["end_frame"])
                self.range_start, self.range_end = None, None
                self.message = f"Confirmed {interval.get('label', 'waving interval')}."
                self.changed = True
                return
            if self.mode == "waving":
                self.message = "Set this WAVING range with I/O, press E to add a draft, then Enter to confirm."
                return
            if any(item.get("prompt_index") == index for item in self.annotations["direction_candidates"]):
                self.message = "Review or delete every direction draft before confirming this block."
                return
            if any(item.get("prompt_index") == index for item in self.annotations["waving_candidates"]):
                self.message = "Review or delete every waving draft before confirming this block."
                return
            opportunity = self.opportunity
            start, end = opportunity["start_frame"], opportunity["end_frame"]
            if self.range_start is not None and self.range_end is not None:
                start, end = self.range_start, self.range_end
            elif self.kind == "continuous":
                self.message = "Set a review range with I and O first."
                return
            interval = {"prompt_index": index, "start_frame": start, "end_frame": end}
            if self.mode == "one_shot":
                self.annotations["one_shot_reviewed_intervals"].append(dict(interval))
            elif self.mode == "direction":
                self.annotations["direction_reviewed_intervals"].append(dict(interval))
            else:
                self.annotations.setdefault("waving_reviewed_intervals", []).append(dict(interval))
            self.range_start, self.range_end = None, None
            if (self._range_covered("one_shot_reviewed_intervals", opportunity["start_frame"], opportunity["end_frame"], index)
                    and self._range_covered("direction_reviewed_intervals", opportunity["start_frame"], opportunity["end_frame"], index)
                    and index not in self.annotations["periodic_reviewed_prompts"]):
                self.annotations["periodic_reviewed_prompts"].append(index)
            self._update_reviewed()
            self.changed = True
            self.message = f"{self.mode} range reviewed. Confirmed events remain supervised; this range's background is zero."
            return
        if action == "ignore_direction":
            if self.range_start is None or self.range_end is None or self.range_start > self.range_end:
                self.message = "Set a direction IGNORE range with I and O first."
                return
            interval = {"prompt_index": index, "start_frame": self.range_start, "end_frame": self.range_end}
            self.annotations["direction_reviewed_intervals"].append(dict(interval))
            self.annotations["direction_ignored_intervals"].append(dict(interval))
            self.range_start, self.range_end = None, None
            self._update_reviewed()
            self.message = "This range is reviewed for directions and IGNORE for both direction channels."
            self.changed = True
            return
        if action == "cancel":
            self.annotations["action_ignored_intervals"].append({
                "prompt_index": index, "start_frame": self.context_start(),
                "end_frame": self.opportunity["end_frame"]})
            self.annotations["direction_candidates"] = [
                item for item in self.annotations["direction_candidates"] if item.get("prompt_index") != index]
            self.annotations["waving_candidates"] = [
                item for item in self.annotations["waving_candidates"] if item.get("prompt_index") != index]
            self.annotations["waving_reviewed_intervals"] = [
                item for item in self.annotations.get("waving_reviewed_intervals", []) if item.get("prompt_index") != index]
            self.annotations["action_intervals"] = [
                item for item in self.annotations["action_intervals"] if item.get("prompt_index") != index]
            self.annotations["waving_intervals"] = [
                item for item in self.annotations["waving_intervals"] if item.get("prompt_index") != index]
            self._update_reviewed()
            self.message = "Recording range cancelled: every Action channel IGNORE."
            self.changed = True
            return
        if action == "delete_event":
            interval = self.current_interval()
            if interval is None:
                interval = next((item for item in self.annotations["action_intervals"]
                                 if item.get("prompt_index") == index
                                 and item["start_frame"] <= frame <= item["end_frame"]), None)
            if interval is None:
                interval = next((item for item in self.annotations["waving_intervals"]
                                 if item.get("prompt_index") == index
                                 and item["start_frame"] <= frame <= item["end_frame"]), None)
            if interval is None:
                self.message = "No selected event or draft."
                return
            for key in ("direction_candidates", "waving_candidates", "action_candidates",
                        "action_intervals", "waving_intervals"):
                if interval in self.annotations.get(key, []):
                    self.annotations[key].remove(interval)
                    break
            self.event_position = 0
            self.changed = True
            self.message = "Selected draft/event removed. Review the range again."
            return

    def attach_session(self, session):
        self._session_data = session

    def _unreview_negative(self, index):
        background = self.annotations["action_background_prompts"]
        background[:] = [item for item in background if item != index]

    def apply(self, action, frame):
        if not self.opportunities:
            return
        if self.periodic:
            return self._apply_periodic(action, frame)
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
                self.message = "已选择动作；I/O 设置起止，E 添加，之后 Enter 审核整段。"
                return
        elif action in ("start", "finish"):
            if not self.context_start() <= frame <= opportunity["end_frame"] or not self.detected[frame]:
                self.message = "请选择当前段内能检测到手的一帧。"
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
            self.message = ("边界已调整；E 添加真实动作，之后 Enter 审核整段。" if self.is_negative
                            else "边界已调整；检查完成后 Enter 确认动作。")
        elif action == "confirm" and self.is_negative:
            if interval:
                self.message = "这个动作还未添加：按 E 添加，或 Del 放弃，再审核整段。"
                return
            self._unreview_negative(index)
            self.annotations["action_background_prompts"].append(index)
            ignored[:] = [item for item in ignored if item.get("prompt_index") != index]
            self.message = "本段负样本已审核，包含可靠背景和已补标的真实动作。"
        elif action == "confirm" or (action == "add_event" and self.is_negative):
            if interval is None or interval["start_frame"] is None or interval["end_frame"] is None:
                self.message = "请先按 I 选择开始帧，按 O 选择完成帧。"
                return
            others = self.intervals if self.is_negative else [item for item in self.intervals if item.get("prompt_index") != index]
            try:
                validate_action_intervals([*others, interval], self.frame_count, self.classes)
            except ValueError:
                self.message = "请检查起点不晚于完成帧，且同类动作的完成帧不重复。"
                return
            self.intervals[:] = sorted([*others, dict(interval)], key=lambda item: item["start_frame"])
            candidates[:] = [item for item in candidates if item.get("prompt_index") != index]
            ignored[:] = [item for item in ignored if item.get("prompt_index") != index]
            if self.is_negative:
                self._unreview_negative(index)
            self.message = "动作已确认；继续检查下一项。"
        elif action == "delete_event" and self.is_negative:
            if interval:
                candidates[:] = [item for item in candidates if item.get("prompt_index") != index]
            else:
                selected = next((item for item in self.intervals if item.get("prompt_index") == index and item["start_frame"] <= frame <= item["end_frame"]), None)
                if selected is None:
                    self.message = "请先定位到要删除的动作区间。"
                    return
                self.intervals.remove(selected)
            self._unreview_negative(index)
            self.message = "动作已删除，请重新检查并审核本段。"
        elif action == "cancel":
            self.intervals[:] = [item for item in self.intervals if item.get("prompt_index") != index]
            candidates[:] = [item for item in candidates if item.get("prompt_index") != index]
            ignored[:] = [item for item in ignored if item.get("prompt_index") != index]
            self._unreview_negative(index)
            ignored.append({"prompt_index": index, "start_frame": self.context_start(), "end_frame": opportunity["end_frame"]})
            self.message = "本段已忽略，不参与训练。"
        else:
            return
        self._update_reviewed()
        self.changed = True
        if action in ("confirm", "cancel"):
            pending = [position for position in range(len(self.opportunities)) if self.status(position) == "PENDING"]
            if pending:
                self.select_prompt(next((position for position in pending if position > self.prompt_position), pending[0]))
        if self.annotations["action_reviewed"]:
            self.message += " 所有段均已处理。"
