from gesture.action_annotations import ActionAnnotationEditor
from gesture.action_data import validate_action_intervals
from gesture.periodic_candidates import propose_direction_events


def same_direction_event(left, right):
    if left.get("label") != right.get("label"):
        return False
    if left.get("extremum_frame") is not None and right.get("extremum_frame") is not None:
        return left["extremum_frame"] == right["extremum_frame"]
    return left["start_frame"] == right["start_frame"]


def repair_periodic_drafts(annotations, metadata, opportunities):
    """Recover draft ownership from recorded task phases, preserving confirmed truth."""
    bounds = {item["prompt_index"]: [item["start_frame"], item["end_frame"]] for item in opportunities}
    for phase in metadata.get("phases", []):
        index = phase.get("prompt_index")
        if index in bounds and phase.get("end_frame") is not None:
            bounds[index][0] = min(bounds[index][0], int(phase["start_frame"]))
            bounds[index][1] = max(bounds[index][1], int(phase["end_frame"]))
    moved, removed = 0, 0
    for field, confirmed_field in (("direction_candidates", "action_intervals"),
                                   ("waving_candidates", "waving_intervals")):
        kept = []
        for draft in annotations.get(field, []):
            anchor = draft.get("extremum_frame", draft.get("start_frame"))
            owners = [index for index, (start, end) in bounds.items()
                      if anchor is not None and start <= anchor <= end]
            reassigned = len(owners) == 1 and draft.get("prompt_index") != owners[0]
            if reassigned:
                draft["prompt_index"] = owners[0]
                moved += 1
            confirmed = [item for item in annotations.get(confirmed_field, [])
                         if item.get("prompt_index") == draft.get("prompt_index")]
            duplicate = (any(same_direction_event(draft, item) for item in confirmed)
                         if field == "direction_candidates" else
                         any(item["start_frame"] == draft["start_frame"] for item in confirmed))
            exact_duplicate = any(item["start_frame"] == draft["start_frame"]
                                  and item["end_frame"] == draft["end_frame"]
                                  and item.get("label") == draft.get("label") for item in confirmed)
            if duplicate and (reassigned or exact_duplicate):
                removed += 1
            else:
                kept.append(draft)
        annotations[field] = kept
    return moved, removed


class PeriodicAnnotationEditor(ActionAnnotationEditor):
    """One task: choose a waving range, edit events, then finish and save together."""

    task_workflow = True

    def __init__(self, annotations, frame_count, detected, classes, session, config):
        self.selected_event = None
        self.new_event_label = None
        super().__init__(annotations, frame_count, detected, classes, session, config)
        self.repaired_drafts = repair_periodic_drafts(
            self.annotations, self._session_data.get("metadata", {}), self.opportunities)
        self.mode = "waving"
        self.annotations["periodic_edit_mode"] = self.mode
        self.message = "① 标 waving 起止  ② G 生成候选  ③ 调整结束帧  ④ Enter 完成本段并保存"
        if any(self.repaired_drafts):
            self.message = f"已修复 {self.repaired_drafts[0]} 个候选归属，清除 {self.repaired_drafts[1]} 个重复草稿。请继续标注。"

    def select_prompt(self, position):
        changed = position != self.prompt_position
        super().select_prompt(position)
        if changed:
            self.selected_event, self.new_event_label = None, None
            self.mode, self.label = "waving", "waving"
            self.annotations["periodic_edit_mode"] = self.mode
            self.message = "当前段的 waving 起止由你选择；已有区间可直接检查候选，最后点完成本段。"

    def current_interval(self):
        return self.selected_event

    def task_events(self):
        index = self.opportunity["prompt_index"]
        events = [item for item in self.intervals if item.get("prompt_index") == index]
        for field in ("direction_candidates", "action_candidates"):
            for draft in self.annotations[field]:
                if draft.get("prompt_index") == index:
                    events = [item for item in events if not same_direction_event(item, draft)]
                    events.append(draft)
        return sorted(events, key=lambda item: (item["end_frame"], item["start_frame"]))

    def select_event(self, event):
        self.selected_event, self.new_event_label = event, None
        self.range_start, self.range_end = None, None
        self.mode = "direction" if event["label"] in ("left_to_right", "right_to_left") else "one_shot"
        self.label = event["label"]
        self.message = f"{self.label} [{event['start_frame']},{event['end_frame']}]：O 调整结束帧，逗号/句号微调，Del 删除。"

    def select_event_at(self, frame, label=None):
        events = [item for item in self.task_events() if item["start_frame"] <= frame <= item["end_frame"]
                  and (label is None or item["label"] == label)]
        if events:
            self.select_event(min(events, key=lambda item: abs(item["end_frame"] - frame)))
            return True
        return False

    def focus_frame(self):
        return int(self.selected_event["end_frame"]) if self.selected_event else self.opportunity["start_frame"]

    def editable_bounds(self):
        start, end = self.context_bounds()
        for phase in self._session_data.get("metadata", {}).get("phases", []):
            if phase.get("prompt_index") == self.opportunity["prompt_index"] and phase.get("end_frame") is not None:
                start = min(start, int(phase["start_frame"]))
                end = max(end, int(phase["end_frame"]))
        return start, end

    def _unreview_task(self):
        index = self.opportunity["prompt_index"]
        if self.status() == "IGNORE":
            self.annotations["action_ignored_intervals"] = [item for item in self.annotations["action_ignored_intervals"]
                                                           if item.get("prompt_index") != index]
        for field in ("one_shot_reviewed_intervals", "direction_reviewed_intervals"):
            self.annotations[field] = [item for item in self.annotations[field] if item.get("prompt_index") != index]
        self.annotations["periodic_reviewed_prompts"] = [item for item in self.annotations["periodic_reviewed_prompts"]
                                                        if item != index]
        self._update_reviewed()
        self.changed = True

    def _store_wave(self):
        if self.range_start is None or self.range_end is None or self.range_start > self.range_end:
            self.message = "先选 waving 第一帧按 I，再选最后一帧按 O，然后点生成候选。"
            return None
        index = self.opportunity["prompt_index"]
        wave = {"prompt_index": index, "start_frame": self.range_start, "end_frame": self.range_end}
        self.annotations["waving_intervals"] = [item for item in self.annotations["waving_intervals"]
                                                if item.get("prompt_index") != index]
        self.annotations["waving_intervals"].append(wave)
        self.annotations["waving_candidates"] = [item for item in self.annotations["waving_candidates"]
                                                 if item.get("prompt_index") != index]
        self.range_start, self.range_end = None, None
        self._unreview_task()
        return wave

    def _finish_task(self):
        if self.range_start is not None or self.range_end is not None:
            self.message = "当前选区尚未保存：waving 点 G 生成候选，补标事件点 E 添加。"
            return False
        index = self.opportunity["prompt_index"]
        waves = [item for field in ("waving_intervals", "waving_candidates")
                 for item in self.annotations[field] if item.get("prompt_index") == index]
        events = self.task_events()
        if not self.is_negative and not waves:
            self.message = "先手动标出本段 waving 起止，再生成并检查方向候选；没有 waving 可点取消本段。"
            return False
        if not self.is_negative and not any(item["label"] in ("left_to_right", "right_to_left") for item in events):
            self.message = "本段还没有方向事件：点 G 生成候选，或补标实际发生的方向事件。"
            return False
        others = [item for item in self.intervals if item.get("prompt_index") != index]
        try:
            validate_action_intervals([*others, *events], self.frame_count, self.classes)
        except (KeyError, TypeError, ValueError):
            self.message = "有事件的起止帧或标签无效，请检查当前候选后再完成。"
            return False
        self._unreview_task()
        self.intervals[:] = sorted([*others, *[dict(item) for item in events]], key=lambda item: item["end_frame"])
        for field in ("direction_candidates", "action_candidates", "waving_candidates"):
            self.annotations[field] = [item for item in self.annotations[field] if item.get("prompt_index") != index]
        self.annotations["waving_intervals"] = [item for item in self.annotations["waving_intervals"]
                                                if item.get("prompt_index") != index] + [dict(item) for item in waves]
        start = min([self.opportunity["start_frame"], *[item["start_frame"] for item in [*waves, *events]]])
        end = max([self.opportunity["end_frame"], *[item["end_frame"] for item in [*waves, *events]]])
        review = {"prompt_index": index, "start_frame": start, "end_frame": end}
        for field in ("one_shot_reviewed_intervals", "direction_reviewed_intervals"):
            self.annotations[field] = [item for item in self.annotations[field] if item.get("prompt_index") != index]
            self.annotations[field].append(dict(review))
        if index not in self.annotations["periodic_reviewed_prompts"]:
            self.annotations["periodic_reviewed_prompts"].append(index)
        self.selected_event = None
        self._update_reviewed()
        self.changed = True
        self.message = f"本段完成：{len(events)} 个事件，四个通道的背景一起审核。"
        return True

    def apply(self, action, frame):
        if action in ("previous_prompt", "next_prompt"):
            self.select_prompt(self.prompt_position + (-1 if action == "previous_prompt" else 1))
        elif action == "edit_wave":
            self.mode, self.label = "waving", "waving"
            self.selected_event, self.new_event_label = None, None
            self.range_start, self.range_end = None, None
            self.message = "选择 waving 第一帧按 I、最后一帧按 O；按 G 保存区间并生成方向候选。"
        elif action in ("previous_event", "next_event"):
            events = self.task_events()
            if not events:
                self.message = "尚无候选；先标 waving 区间并点 G 生成。"
                return
            position = next((i for i, item in enumerate(events) if item is self.selected_event),
                            0 if action == "previous_event" else -1)
            self.select_event(events[(position + (-1 if action == "previous_event" else 1)) % len(events)])
        elif action.startswith("new_"):
            self.new_event_label = action.removeprefix("new_")
            self.selected_event = None
            self.mode = "one_shot" if self.new_event_label in ("open_hand", "close_hand") else "direction"
            self.label = self.new_event_label
            self.range_start, self.range_end = None, None
            self.message = f"补标 {self.label}：I 起点、O 结束、E 添加；添加后仍在当前段检查。"
        elif action in ("start", "finish", "end_previous", "end_next"):
            if action in ("end_previous", "end_next"):
                if self.selected_event is None:
                    return
                frame = self.selected_event["end_frame"] + (-1 if action == "end_previous" else 1)
                action = "finish"
            if not 0 <= frame < self.frame_count or not self.detected[frame]:
                self.message = "请选择能检测到手的一帧。"
                return
            start, end = self.editable_bounds()
            if not start <= frame <= end:
                self.message = "这一帧属于其它段，请先切换到对应 task 再标注。"
                return
            if self.selected_event is not None:
                field = "start_frame" if action == "start" else "end_frame"
                adjusted = {**self.selected_event, field: int(frame)}
                others = [item for item in self.task_events() if item is not self.selected_event]
                try:
                    validate_action_intervals([*others, adjusted], self.frame_count, self.classes)
                except ValueError:
                    self.message = "请保持起点不晚于结束帧，并避免同方向事件结束帧重复。"
                    return
                self.selected_event[field] = int(frame)
                self._unreview_task()
                self.select_event(self.selected_event)
            else:
                if action == "start":
                    self.range_start, self.range_end = int(frame), None
                else:
                    self.range_end = int(frame)
                self.changed = True
                self.message = f"选区 [{self.range_start},{self.range_end}]；waving 按 G，补标事件按 E。"
        elif action == "generate_directions":
            if self.new_event_label is not None:
                self.message = "正在补标动作：先用 I/O 选择起止，再按 E 添加。"
                return
            if self.mode == "waving" and (self.range_start is not None or self.range_end is not None):
                wave = self._store_wave()
            else:
                wave = self.selected_waving_interval()
            if wave is None:
                self.message = "先标出当前段 waving 的第一帧和最后一帧。"
                return
            index = self.opportunity["prompt_index"]
            confirmed = [item for item in self.intervals if item.get("prompt_index") == index]
            drafts = [item for item in propose_direction_events(self._session_data, wave, self.config)
                      if not any(same_direction_event(item, saved) for saved in confirmed)]
            self.annotations["direction_candidates"] = [item for item in self.annotations["direction_candidates"]
                                                         if item.get("prompt_index") != index] + drafts
            self._unreview_task()
            events = self.task_events()
            if events:
                self.select_event(events[0])
                self.message = f"当前段有 {len(events)} 个事件；点候选或用上一/下一候选检查，最后 Enter 完成本段。"
            else:
                self.message = "未找到方向候选；检查 waving 区间，或补标实际发生的左右事件。"
        elif action == "add_event":
            if self.new_event_label is None or self.range_start is None or self.range_end is None:
                self.message = "先点补标类别，再用 I/O 选择起止，最后 E 添加。"
                return
            event = {"prompt_index": self.opportunity["prompt_index"], "label": self.new_event_label,
                     "start_frame": self.range_start, "end_frame": self.range_end}
            try:
                validate_action_intervals([*self.task_events(), event], self.frame_count, self.classes)
            except ValueError:
                self.message = "请检查事件起止，以及是否和已有同类事件重复。"
                return
            field = "direction_candidates" if self.mode == "direction" else "action_candidates"
            self.annotations[field].append(event)
            self._unreview_task()
            self.select_event(event)
        elif action == "delete_event":
            if self.selected_event is None:
                self.message = "先点击一个候选，再删除。"
                return
            index = self.opportunity["prompt_index"]
            for field in ("action_intervals", "direction_candidates", "action_candidates"):
                self.annotations[field][:] = [item for item in self.annotations[field]
                                             if item.get("prompt_index") != index
                                             or not same_direction_event(item, self.selected_event)]
            self.selected_event = None
            self._unreview_task()
            self.apply("next_event", frame)
            self.message = "当前事件已删除；检查剩余事件后点完成本段。"
        elif action in ("confirm", "finish_task"):
            return self._finish_task()
        elif action == "cancel":
            index = self.opportunity["prompt_index"]
            self.range_start, self.range_end = None, None
            self._unreview_task()
            self.intervals[:] = [item for item in self.intervals if item.get("prompt_index") != index]
            for field in ("direction_candidates", "action_candidates", "waving_candidates", "waving_intervals"):
                self.annotations[field] = [item for item in self.annotations[field] if item.get("prompt_index") != index]
            start, end = self.editable_bounds()
            self.annotations["action_ignored_intervals"].append(
                {"prompt_index": index, "start_frame": start, "end_frame": end})
            self.selected_event, self.new_event_label = None, None
            self.mode, self.label = "waving", "waving"
            self._update_reviewed()
            self.message = "本段已取消，所有 Action 通道忽略。"
            return True
        return False
