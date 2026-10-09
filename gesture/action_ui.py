import json
import time

import cv2
import numpy as np

from gesture.action_data import action_pulse_frames, make_action_targets, negative_input_regions
from gesture.action_features import make_action_features
from gesture.action_candidates import action_excluded_frames
from gesture.config import project_path
from gesture.hand_landmarker import draw_hand_landmarks
from gesture.prompts import label_segments, prompt_labels_for_session
from gesture.data import load_session
from gesture.periodic_candidates import direction_trace
from gesture.timeline import frame_at_x, frame_x
from gesture.display_text import draw_text


PANEL_HEIGHT = 570
MARGIN = 16
TIMELINE_BANDS = [(258, 275), (290, 307), (322, 339)]
ANNOTATION_HEADER_HEIGHT = 82
ACTION_NAMES = {"open_hand": "张开手掌", "close_hand": "握拳",
                "left_to_right": "左 → 右", "right_to_left": "右 → 左", "waving": "挥手"}


def action_buttons(width, playing, view_only=False, review_mode=True, negative=False, periodic=False,
                   task_workflow=False, extra_actions=True):
    rows = [[("previous", "A < PREV"), ("play", "P PAUSE" if playing else "P PLAY"),
             ("next", "D NEXT >"), ("save", "S SAVE"), ("quit", "Q EXIT")]]
    if task_workflow and not view_only:
        rows = [[("previous", "A 上一帧"), ("play", "P 暂停" if playing else "P 播放"),
                 ("next", "D 下一帧"), ("save", "S 保存进度"), ("quit", "Q 退出")],
                [("previous_prompt", "[ 上一段"), ("next_prompt", "] 下一段"),
                 ("finish_task", "Enter 完成并保存"), ("cancel", "X 忽略本段"),
                 ("more_actions", "B 收起补标" if extra_actions else "B 补标动作")],
                [("edit_wave", "W 重选挥手区间"), ("generate_directions", "G 自动生成候选"),
                 ("previous_event", "上一候选"), ("next_event", "下一候选")],
                [("goto_start", "J 跳到起点"), ("start", "I 设置起点"),
                  ("goto_finish", "K 跳到完成帧"), ("finish", "O 设置完成帧"), ("delete_event", "Del 删除事件")]]
        if extra_actions:
            rows += [[("new_open_hand", "1 补标张开"), ("new_close_hand", "2 补标握拳"),
                      ("new_left_to_right", "3 补标左→右"), ("new_right_to_left", "4 补标右→左"), ("add_event", "E 添加动作")]]
    elif not view_only:
        rows += [[("previous_prompt", "[ PREV TASK"), ("next_prompt", "] NEXT TASK"),
                  ("confirm", "ENTER REVIEW" if periodic else "ENTER ACCEPT BLOCK >" if negative else "ENTER CONFIRM >"),
                  ("cancel", "X CANCEL >")]]
        if periodic:
            rows += [[("mode_one_shot", "M OPEN/CLOSE"), ("mode_direction", "R DIRECTION"),
                      ("mode_waving", "W WAVING"), ("previous_event", "( PREV DRAFT"),
                      ("next_event", ") NEXT DRAFT")]]
        rows += [[("goto_start", "J GO START"), ("start", "I SET START"),
                  ("goto_finish", "K GO END"), ("finish", "O SET END")]]
        if periodic:
            rows += [[("label_0", "1 OPEN"), ("label_1", "2 CLOSE"),
                      ("label_2", "3 L->R"), ("label_3", "4 R->L"),
                      ("add_event", "E ADD DRAFT"), ("generate_directions", "G PCA"),
                      ("ignore_direction", "Z DIR IGNORE"), ("delete_event", "DEL")]]
        elif negative:
            rows += [[("label_0", "1 OPEN HAND"), ("label_1", "2 CLOSE HAND"),
                      ("add_event", "E ADD EVENT"), ("delete_event", "BACKSPACE DEL EVENT")]]
    if not view_only:
        rows[0].insert(3, ("details", "H 标签细节"))
        translations = {"previous": "A 上一帧", "play": "P 暂停" if playing else "P 播放",
                        "next": "D 下一帧", "save": "S 保存进度", "quit": "Q 退出",
                        "previous_prompt": "[ 上一段", "next_prompt": "] 下一段",
                        "confirm": "Enter 审核并保存" if negative or periodic else "Enter 确认并保存",
                        "cancel": "X 忽略本段", "goto_start": "J 跳到起点", "start": "I 设置起点",
                        "goto_finish": "K 跳到完成帧", "finish": "O 设置完成帧",
                        "label_0": "1 张开", "label_1": "2 握拳", "label_2": "3 左→右", "label_3": "4 右→左",
                        "add_event": "E 添加动作", "delete_event": "Del 删除动作"}
        rows = [[(action, translations.get(action, label)) for action, label in row] for row in rows]
    buttons = {}
    for row_index, row in enumerate(rows):
        slot = (width - 2 * MARGIN) / len(row)
        for index, (action, text) in enumerate(row):
            left, right = MARGIN + round(index * slot), MARGIN + round((index + 1) * slot) - 6
            top = 8 + 38 * row_index
            buttons[action] = ((left, top, right, top + 30), text)
    return buttons


def annotation_guide(editor):
    if editor.status() == "IGNORE":
        return "本段已忽略，不参与训练", "需要重标时重新设置起止，再确认本段。"
    if editor.status() != "PENDING":
        return "本段已审核完成", "可以查看其它段；修改边界后需要重新确认。"
    if editor.kind == "continuous":
        return "连续录像：手动标出实际发生的动作", "选择开合、方向或挥手模式；I/O 设置起止，E 添加，Enter 审核选区。"
    if getattr(editor, "task_workflow", False):
        if editor.new_event_label:
            return f"补标：{ACTION_NAMES[editor.new_event_label]}", "I 设置开始帧，O 设置完成帧，E 添加；补齐后再完成整段。"
        event = editor.current_interval()
        if event:
            events = editor.task_events()
            position = next(index for index, item in enumerate(events) if item is event) + 1
            direction = event["label"] in ("left_to_right", "right_to_left")
            return (f"第 2 步：检查候选 {position}/{len(events)} · {ACTION_NAMES[event['label']]}",
                    "方向自动分配；O 设置端点后开始回移的帧。逐个检查后 Enter 完成。" if direction else
                    "找到首次完整张开或握拳的帧，按 O 设置完成帧。")
        if editor.is_negative:
            return "检查负样本：先播放完整一段", "没有目标动作可直接完成；意外发生动作时点“补标动作”，如实记录后再完成。"
        if editor.range_start is None and editor.range_end is None:
            if editor.task_events():
                return "已有挥手区间和候选", "点击下一候选逐个检查；需要重选区间时点“重选挥手区间”。"
            return "第 1 步：选择本段挥手区间", "定位开始挥手的第一帧，按 I；定位停止挥手的最后一帧，按 O。"
        if editor.range_end is None:
            return f"第 1 步：已选开始帧 {editor.range_start}", "继续播放或逐帧移动，找到停止挥手的最后一帧，按 O。"
        return f"区间已选择 [{editor.range_start}, {editor.range_end}]", "下一步按 G 自动生成方向候选；左右方向会自动分配。"
    if editor.is_negative:
        if editor.current_interval():
            return f"补标真实动作：{ACTION_NAMES[editor.label]}", "I 设置开始帧，O 设置完成帧，E 添加动作；之后 Enter 审核整段。"
        return "检查负样本：先播放完整一段", "没有目标动作按 Enter；有开合时用 1/2 选动作、I/O 标起止、E 添加。"
    return (f"检查自动候选：{ACTION_NAMES.get(editor.label, editor.label)}",
            "I 设置开始离开原手型的帧；O 设置首次完整达到目标手型的帧；Enter 确认并进入下一段。")


def annotation_layout(width, editor, playing=False, extra_actions=False):
    buttons = action_buttons(width, playing, negative=editor.is_negative, periodic=editor.periodic,
                             task_workflow=getattr(editor, "task_workflow", False), extra_actions=extra_actions)
    guide_top = max(bounds[0][3] for bounds in buttons.values()) + 12
    top = guide_top + 86
    rows = [("提示 / 保持 / 休息", "prompt", None, (top, top + 18))]
    top += 38
    if editor.periodic:
        rows.append(("挥手区间", "wave", None, (top, top + 18)))
        top += 30
    classes = editor.classes if editor.periodic else [label for label in editor.classes
                                                      if label in ("open_hand", "close_hand")]
    for label in classes:
        rows.append((ACTION_NAMES[label], "event", label, (top, top + 18)))
        top += 26
    return buttons, guide_top, rows, top + 8


def draw_annotation_panel(width, session, prompts, config, frame, editor, playing, extra_actions):
    buttons, guide_top, rows, height = annotation_layout(width, editor, playing, extra_actions)
    panel = np.full((height, width, 3), (26, 22, 18), np.uint8)
    font = config["display"]["text_font_path"]
    colors = config["display"]["prompt_colors"]

    def text(value, position, size=13, color=(233, 230, 225), max_width=None):
        draw_text(panel, value, position, size, color, font, max_width=max_width)

    for action, ((left, top, right, bottom), label) in buttons.items():
        color = (53, 44, 36)
        if action in ("confirm", "finish_task"):
            color = (53, 116, 64)
        elif action == "cancel":
            color = (50, 47, 102)
        cv2.rectangle(panel, (left, top), (right, bottom), color, -1)
        text(label, (left + 7, top + 8), 13, max_width=right - left - 14)
    title, instruction = annotation_guide(editor)
    text(title, (MARGIN, guide_top), 16, (160, 218, 176), width - 2 * MARGIN)
    text(instruction, (MARGIN, guide_top + 25), 13, max_width=width - 2 * MARGIN)
    text("色块是动作区间，白框是当前选中；点击时间轴定位。检查完后确认整段，S 仅保存进度。",
         (MARGIN, guide_top + 49), 11, (169, 153, 136), width - 2 * MARGIN)
    text(editor.message, (MARGIN, guide_top + 68), 11, (130, 200, 233), width - 2 * MARGIN)
    begin, end = editor.context_bounds()
    timeline_left, timeline_right = 126, width - MARGIN

    def x(index):
        return frame_x(index, begin, end, timeline_left, timeline_right)

    selected = editor.current_interval()
    events = editor.task_events() if getattr(editor, "task_workflow", False) else [*editor.intervals,
                                                                               *editor.annotations.get("action_candidates", [])]
    wave = editor.selected_waving_interval() if editor.periodic else None
    for caption, kind, label, (top, bottom) in rows:
        text(caption, (MARGIN, top + 3), 12, max_width=timeline_left - MARGIN - 8)
        cv2.rectangle(panel, (timeline_left, top), (timeline_right, bottom), (53, 44, 36), -1)
        if kind == "prompt":
            intervals = [{"start_frame": first + begin, "end_frame": last + begin, "label": name}
                         for name, first, last in label_segments(prompts[begin:end + 1]) if name in colors]
        elif kind == "wave":
            intervals = [wave] if wave else []
        else:
            intervals = [item for item in events if item["label"] == label]
        for interval in intervals:
            if interval.get("start_frame") is None or interval.get("end_frame") is None:
                continue
            first, last = max(begin, interval["start_frame"]), min(end, interval["end_frame"])
            if first > last:
                continue
            key = interval.get("label", "waving")
            cv2.rectangle(panel, (x(first), top), (x(last), bottom), tuple(colors[key]), -1)
            if interval is selected or (kind == "wave" and editor.mode == "waving"):
                cv2.rectangle(panel, (x(first), top), (x(last), bottom), (245, 245, 245), 1)
        if begin <= frame <= end:
            cv2.line(panel, (x(frame), top), (x(frame), bottom), (245, 245, 245), 2)
    return panel


def draw_action_panel(width, session, prompts, config, frame, editor, playing, predictions=None,
                      view_only=False, event_marks=None, compact=False, extra_actions=True):
    if compact:
        return draw_annotation_panel(width, session, prompts, config, frame, editor, playing, extra_actions)
    panel = np.full((PANEL_HEIGHT, width, 3), 24, np.uint8)
    colors = config["display"]["prompt_colors"]
    classes = config["labels"]["action_classes"]
    review_mode = bool(editor.opportunities) and not view_only
    task_view = review_mode and editor.kind == "periodic"
    task_workflow = getattr(editor, "task_workflow", False) and not view_only
    candidate = editor.current_interval() if review_mode else None
    selected_label = candidate.get("label", "waving") if candidate else editor.label
    for action, ((left, top, right, bottom), text) in action_buttons(
            width, playing, view_only, review_mode, editor.is_negative, editor.periodic, task_workflow,
            extra_actions).items():
        color = (65, 65, 65)
        if action.startswith("label_") and classes[int(action[-1])] == selected_label:
            color = tuple(colors[selected_label])
        if action in ("confirm", "finish_task"):
            color = (45, 125, 45)
        elif action == "cancel":
            color = (50, 50, 150)
        elif action.startswith("mode_") and action.removeprefix("mode_") == editor.mode:
            color = (0, 125, 190)
        cv2.rectangle(panel, (left, top), (right, bottom), color, -1)
        if not view_only:
            draw_text(panel, text, (left + 6, top + 8), 14, (255, 255, 255),
                      config["display"]["text_font_path"], right - left - 12)
        else:
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            cv2.putText(panel, text, (left + max(3, (right - left - tw) // 2), top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    message = "White curves: predicted scores. Colored blocks: target pulses." if view_only else editor.message
    if task_workflow:
        draw_text(panel, message, (MARGIN, 205), 16, (80, 220, 255), config["display"]["text_font_path"], width - 2 * MARGIN)
        draw_text(panel, "点色块选事件；逗号/句号微调结束帧。检查整段后 Enter 完成。黄点：折返；青点：结束。",
                  (MARGIN, 228), 13, (210, 210, 210), config["display"]["text_font_path"], width - 2 * MARGIN)
    else:
        cv2.putText(panel, message, (MARGIN, 237), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 220, 255), 1)
    count = len(session["times"])
    local_start, local_end = 0, count - 1
    if review_mode:
        local_start, local_end = editor.context_bounds()

    def x(index, overview=False):
        begin, end = (0, count - 1) if overview and not task_view else (local_start, local_end)
        return frame_x(index, begin, end, MARGIN, width - MARGIN)

    def blocks(labels, band, overview=False):
        top, bottom = band
        begin, end = (0, count - 1) if overview and not task_view else (local_start, local_end)
        for label, start, finish in label_segments(labels[begin:end + 1]):
            if label not in colors:
                continue
            left, right = x(start + begin, overview), x(finish + begin, overview)
            cv2.rectangle(panel, (left, top), (right, bottom), tuple(colors[label]), -1)

    offset = 75 if editor.periodic else 0
    bands = [(top + offset, bottom + offset) for top, bottom in TIMELINE_BANDS[:3 if editor.periodic else 2]]
    captions = [f"TASK {editor.prompt_position + 1}/{len(editor.opportunities)} PROMPT + REST | frames {local_start}-{local_end}"
                if task_view else "SESSION PROMPTS (guide only; click to select task)"]
    if editor.periodic:
        captions += ["WAVING INTERVALS / yellow: pending draft", "CONFIRMED ATOMIC EVENTS / yellow: pending draft"]
    else:
        captions += ["CONFIRMED ACTIONS / yellow: pending draft"]
    for channel, name in enumerate(classes):
        captions.append(f"{name.upper()} TARGET / SCORE")
        top = 372 + offset + channel * 26
        bands.append((top, top + 11))
    captions.append("ANY CHANNEL SUPERVISED")
    top = 372 + offset + len(classes) * 26
    bands.append((top, top + 11))
    if editor.periodic:
        selected = editor.selected_waving_interval()
        trace = direction_trace(session, selected, config) if selected is not None else None
        left, right, top, bottom = MARGIN, width - MARGIN, 250, 322
        if not task_workflow:
            cv2.putText(panel, "PCA: fingertip path | dominant camera axis | projected position / frame", (left, 247),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (210, 210, 210), 1)
        cv2.rectangle(panel, (left, top), (right, bottom), (45, 45, 45), -1)
        if trace is not None:
            path_left, path_right = left + 4, left + round((right - left) * 0.31)
            tx, ty = trace["trajectory"][:, 0], trace["trajectory"][:, 1]
            sx = max(float(np.ptp(tx)), 1e-6)
            sy = max(float(np.ptp(ty)), 1e-6)
            path = np.asarray([(path_left + round((xv - tx.min()) / sx * (path_right - path_left - 8)),
                                bottom - 5 - round((yv - ty.min()) / sy * (bottom - top - 10)))
                               for xv, yv in zip(tx, ty)], np.int32)
            if len(path) > 1:
                cv2.polylines(panel, [path], False, (180, 190, 200), 1, cv2.LINE_AA)
            center = ((path_left + path_right) // 2, (top + bottom) // 2)
            vector = trace["axis"]
            end_axis = (center[0] + round(vector[0] * 28), center[1] + round(vector[1] * 28))
            cv2.arrowedLine(panel, center, end_axis, (70, 220, 240), 2, tipLength=0.2)
            chart_left, chart_right = path_right + 10, right - 4
            projection = trace["projection"]
            low, high = float(projection.min()), float(projection.max())
            scale = max(high - low, 1e-6)
            chart = np.asarray([(chart_left + round(index / max(1, len(projection) - 1) * (chart_right - chart_left)),
                                 bottom - 5 - round((value - low) / scale * (bottom - top - 10)))
                                for index, value in enumerate(projection)], np.int32)
            if len(chart) > 1:
                cv2.polylines(panel, [chart], False, (110, 220, 125), 1, cv2.LINE_AA)
            chart_events = editor.task_events() if task_workflow else editor.annotations.get("direction_candidates", [])
            for event in chart_events:
                if event.get("prompt_index") != editor.opportunity["prompt_index"]:
                    continue
                if trace["frames"][0] <= event.get("extremum_frame", -1) <= trace["frames"][-1]:
                    local = int(np.searchsorted(trace["frames"], event["extremum_frame"]))
                    cv2.circle(panel, tuple(chart[min(local, len(chart) - 1)]), 4, (0, 230, 255), -1)
                if trace["frames"][0] <= event.get("end_frame", -1) <= trace["frames"][-1]:
                    local = int(np.searchsorted(trace["frames"], event["end_frame"]))
                    cv2.circle(panel, tuple(chart[min(local, len(chart) - 1)]), 4, (255, 230, 50), -1)
            title = f"axis {trace['angle_degrees']:.1f} deg | {'usable' if trace['eligible'] else 'angle too steep'}"
            cv2.putText(panel, title, (chart_left, top + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (80, 235, 245) if trace["eligible"] else (40, 130, 255), 1)
        else:
            message = "Set this WAVING range with I/O first." if selected is None else "No continuous hand trajectory in this range"
            cv2.putText(panel, message, (left + 8, top + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    for caption, (top, bottom) in zip(captions, bands):
        if task_workflow:
            if caption.startswith("TASK"):
                caption = f"当前第 {editor.prompt_position + 1}/{len(editor.opportunities)} 段：提示 + 随后休息 | 帧 {local_start}–{local_end}"
            elif caption.startswith("WAVING"):
                caption = "你标记的 waving 区间"
            elif caption.startswith("CONFIRMED"):
                caption = "动作事件：点击色块选择，黄色为当前事件"
            elif caption.startswith("ANY"):
                caption = "参与训练监督的帧：正样本 + 已审核背景"
            elif caption.endswith("TARGET / SCORE"):
                caption = caption.replace("TARGET / SCORE", "目标预览：完成帧起连续 4 帧")
            draw_text(panel, caption, (MARGIN, top - 16), 12, (210, 210, 210),
                      config["display"]["text_font_path"], width - 2 * MARGIN)
        else:
            cv2.putText(panel, caption, (MARGIN, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (210, 210, 210), 1)
        cv2.rectangle(panel, (MARGIN, top), (width - MARGIN, bottom), (45, 45, 45), -1)
    blocks(prompts, bands[0], overview=True)
    if editor.periodic:
        waving = np.full(count, "", dtype=object)
        for interval in editor.annotations.get("waving_intervals", []):
            if task_view and interval.get("prompt_index") not in (None, editor.opportunity["prompt_index"]):
                continue
            waving[interval["start_frame"]:interval["end_frame"] + 1] = "waving"
        blocks(waving, bands[1])
        for interval in editor.annotations.get("waving_candidates", []):
            if task_view and interval.get("prompt_index") not in (None, editor.opportunity["prompt_index"]):
                continue
            start = max(local_start, int(interval["start_frame"]))
            finish = min(local_end, int(interval["end_frame"]))
            if start <= finish:
                cv2.rectangle(panel, (x(start), bands[1][0]), (x(finish), bands[1][1]), (30, 170, 220), -1)
    event_band = bands[2] if editor.periodic else bands[1]

    def event_lane(label):
        top, bottom = event_band
        channel = classes.index(label)
        height = bottom - top + 1
        return (top + channel * height // len(classes),
                top + (channel + 1) * height // len(classes) - 1)

    for interval in editor.intervals:
        if task_view and interval.get("prompt_index") not in (None, editor.opportunity["prompt_index"]):
            continue
        start = max(local_start, int(interval["start_frame"]))
        finish = min(local_end, int(interval["end_frame"]))
        if start <= finish:
            top, bottom = event_lane(interval["label"])
            cv2.rectangle(panel, (x(start), top), (x(finish), bottom), tuple(colors[interval["label"]]), -1)
    if task_workflow:
        for field in ("direction_candidates", "action_candidates"):
            for interval in editor.annotations[field]:
                if interval.get("prompt_index") != editor.opportunity["prompt_index"]:
                    continue
                start, finish = max(local_start, interval["start_frame"]), min(local_end, interval["end_frame"])
                if start <= finish:
                    top, bottom = event_lane(interval["label"])
                    cv2.rectangle(panel, (x(start), top), (x(finish), bottom), tuple(colors[interval["label"]]), -1)
    if review_mode:
        opportunity = editor.opportunity
        if not editor.periodic:
            cv2.rectangle(panel, (x(opportunity["start_frame"], True), bands[0][0]),
                          (x(opportunity["end_frame"], True), bands[0][1]), (0, 255, 255), 1)
        if editor.periodic and editor.mode == "waving" and editor.range_start is not None and editor.range_end is not None:
            start, finish = max(local_start, editor.range_start), min(local_end, editor.range_end)
            if start <= finish:
                cv2.rectangle(panel, (x(start), bands[1][0]),
                              (x(finish), bands[1][1]), (30, 170, 220), -1)
        if (task_workflow or editor.status() == "PENDING") and candidate and candidate.get("start_frame") is not None and candidate.get("end_frame") is not None:
            start = max(local_start, int(candidate["start_frame"]))
            finish = min(local_end, int(candidate["end_frame"]))
            if start <= finish:
                draft_band = bands[1] if editor.periodic and editor.mode == "waving" else event_lane(candidate["label"])
                cv2.rectangle(panel, (x(start), draft_band[0]),
                              (x(finish), draft_band[1]), (30, 170, 220), -1)
    displayed_targets = session["targets"].copy()
    if task_workflow:
        displayed_targets[:] = 0
        for interval in editor.task_events():
            displayed_targets[action_pulse_frames(interval, session["times"], config), classes.index(interval["label"])] = 1
    for channel, name in enumerate(classes):
        blocks(np.where(displayed_targets[:, channel] > 0, name, ""), bands[(3 if editor.periodic else 2) + channel])
        if predictions is not None:
            top, bottom = bands[(3 if editor.periodic else 2) + channel]
            finite = np.isfinite(predictions[local_start:local_end + 1, channel])
            for start, end in zip(np.flatnonzero(np.diff(np.r_[False, finite, False].astype(int)) == 1) + local_start,
                                  np.flatnonzero(np.diff(np.r_[False, finite, False].astype(int)) == -1) + local_start):
                curve = np.asarray([(x(index), bottom - round(float(predictions[index, channel]) * (bottom - top))) for index in range(start, end)], np.int32)
                cv2.polylines(panel, [curve], False, (255, 255, 255), 1, cv2.LINE_AA)
    supervised = np.any(session["mask"], axis=1)
    top, bottom = bands[-1]
    for present, start, end in label_segments(supervised[local_start:local_end + 1]):
        if present:
            cv2.rectangle(panel, (x(start + local_start), top), (x(end + local_start), bottom), (80, 130, 80), -1)
    for mark in event_marks or []:
        mark_frame = mark.get("frame_index")
        if mark_frame is None or not local_start <= int(mark_frame) <= local_end:
            continue
        label = mark.get("label", "")
        if mark.get("failure_kind"):
            band = bands[2] if editor.periodic else bands[1]
            color = (40, 40, 255)
        elif label in classes:
            band = bands[(3 if editor.periodic else 2) + classes.index(label)]
            color = tuple(colors[label])
        elif label in ("waving_start", "waving_end") and editor.periodic:
            band = bands[1]
            color = (70, 235, 70) if label == "waving_start" else (70, 70, 235)
        else:
            continue
        location = x(int(mark_frame))
        cv2.line(panel, (location, band[0]), (location, band[1]), color, 2)
    cv2.line(panel, (x(frame, True), bands[0][0]), (x(frame, True), bands[0][1]), (255, 255, 255), 2)
    if local_start <= frame <= local_end:
        cv2.line(panel, (x(frame), bands[1 if editor.periodic else 1][0]),
                 (x(frame), bands[-1][1]), (255, 255, 255), 2)
    return panel


def show_action_session(config, session, editor, predictions=None, view_only=False,
                        waving_trace=None, event_marks=None, start_frame=None):
    folder = project_path(config["data"]["sessions_dir"]) / session["id"]
    annotation_path = folder / "annotations.json"
    prompts = prompt_labels_for_session(load_session(folder), len(session["times"]))
    capture = cv2.VideoCapture(str(folder / "video.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {folder / 'video.mp4'}")
    window = "Action replay" if view_only else "Action annotation"
    review_mode = bool(editor.opportunities) and not view_only
    task_view = review_mode and editor.kind == "periodic"
    task_workflow = getattr(editor, "task_workflow", False) and not view_only
    count = len(session["times"])
    frame_index = editor.focus_frame() if review_mode else 0
    if start_frame is not None:
        frame_index = max(0, min(count - 1, int(start_frame)))
    displayed = None
    playing, running, redraw = False, True, True
    details, extra_actions = view_only or editor.kind == "continuous", False
    canvas_size = None
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    next_frame_at = 0.0

    def refresh_targets():
        session["ignored_frames"] = action_excluded_frames(
            editor.annotations, session["opportunities"], count, config, session["times"], session.get("kind"))
        session["reviewed"] = bool(editor.annotations.get("action_reviewed", False))
        if session.get("sample_type") == "negative":
            session["input_regions"] = negative_input_regions(
                editor.annotations, session["opportunities"], count, config, session["times"], session.get("kind"))
        elif session.get("kind") == "periodic":
            from gesture.action_data import periodic_input_regions
            session["input_regions"] = periodic_input_regions(
                session["metadata"], session["opportunities"], count, editor.annotations, config)
        elif session.get("kind") == "continuous":
            from gesture.action_data import continuous_input_regions
            session["input_regions"] = continuous_input_regions(editor.annotations, count)
        else:
            session["input_regions"] = None
        if session.get("kind") in ("periodic", "continuous") or session.get("sample_type") == "negative":
            session["features"], session["valid"], session["segments"] = make_action_features(
                session["points"], session["detected"], session["state_scores"], session["times"], config,
                session["image_size"], session["input_regions"],
                world_points_by_frame=session.get("world_points"),
            )
        session["targets"], session["mask"] = make_action_targets(
            session["valid"], session["segments"], editor.intervals,
            session["reviewed"], config, session["ignored_frames"], session["times"],
            editor.annotations, session["opportunities"], session.get("kind"),
        )

    if not view_only:
        refresh_targets()

    def seek(frame):
        nonlocal frame_index, playing, redraw
        frame_index = max(0, min(count - 1, int(frame)))
        playing, redraw = False, True

    def save(show_message=True):
        annotation_path.write_text(json.dumps(editor.annotations, indent=2), encoding="utf-8")
        editor.changed = False
        if show_message:
            editor.message = ("标注进度已保存；看完当前段后点完成本段，统一确认事件和背景。" if task_workflow else
                              "标注进度已保存；保存不会自动审核当前段。")

    def perform(action):
        nonlocal playing, running, next_frame_at, redraw, details, extra_actions
        if action == "quit":
            running = False
        elif action in ("previous", "next"):
            seek(frame_index + (-1 if action == "previous" else 1))
        elif action in ("details", "more_actions"):
            if action == "details":
                details = not details
            else:
                extra_actions = not extra_actions
        elif action == "play":
            stop = editor.context_end() if review_mode else count - 1
            if not playing and frame_index >= stop:
                seek(editor.focus_frame() if review_mode else 0)
            playing = not playing and count > 1
            if playing:
                next_frame_at = time.monotonic() + session["times"][frame_index + 1] - session["times"][frame_index]
        elif not view_only:
            seek(frame_index)
            if task_workflow and action.startswith("new_"):
                extra_actions = True
            if action == "save":
                save()
            elif action in ("goto_start", "goto_finish") and review_mode:
                interval = editor.current_interval()
                if interval is None and task_workflow and editor.mode == "waving":
                    interval = editor.selected_waving_interval()
                field = "start_frame" if action == "goto_start" else "end_frame"
                boundary = interval[field] if interval else editor.range_start if action == "goto_start" else editor.range_end
                if boundary is not None:
                    seek(boundary)
                else:
                    editor.message = "还没有设置这一边界，请先定位到对应帧，再按 I 或 O。"
            else:
                position = editor.prompt_position
                completed = editor.apply(action, frame_index)
                refresh_targets()
                if task_workflow and completed and action in ("confirm", "finish_task", "cancel"):
                    save(show_message=False)
                    message = f"第 {position + 1} 段已{'取消' if action == 'cancel' else '完成'}并保存。"
                    editor.select_prompt(position + 1)
                    editor.message = message + ("继续检查下一段。" if editor.prompt_position != position else "已到最后一段。")
                elif not task_workflow and action in ("confirm", "cancel") and (
                        editor.prompt_position != position or editor.status() != "PENDING"):
                    save(show_message=False)
                if review_mode and (editor.prompt_position != position or action in ("previous_prompt", "next_prompt")):
                    seek(editor.focus_frame())
                elif task_workflow and action in ("previous_event", "next_event", "generate_directions",
                                                   "end_previous", "end_next", "delete_event"):
                    seek(editor.focus_frame())
        redraw = True

    def mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        panel_y = y - video_height - (0 if view_only else ANNOTATION_HEADER_HEIGHT)
        compact = not view_only and not details
        buttons = (annotation_layout(video_width, editor, playing, extra_actions)[0] if compact else
                   action_buttons(video_width, playing, view_only, review_mode, editor.is_negative,
                                  editor.periodic, task_workflow, extra_actions))
        for action, ((left, top, right, bottom), _text) in buttons.items():
            if left <= x <= right and top <= panel_y <= bottom:
                perform(action)
                return
        if compact:
            if 126 <= x <= video_width - MARGIN:
                begin, end = editor.context_bounds()
                clicked_frame = frame_at_x(x, begin, end, 126, video_width - MARGIN)
                for _caption, kind, label, (top, bottom) in annotation_layout(
                        video_width, editor, playing, extra_actions)[2]:
                    if top <= panel_y <= bottom:
                        if task_workflow and kind == "event" and editor.select_event_at(clicked_frame, label):
                            seek(editor.focus_frame())
                        else:
                            if task_workflow and kind == "wave" and editor.mode != "waving":
                                editor.apply("edit_wave", clicked_frame)
                            seek(clicked_frame)
                        return
            return
        timeline_offset = 75 if editor.periodic else 0
        if MARGIN <= x <= video_width - MARGIN and TIMELINE_BANDS[0][0] + timeline_offset <= panel_y < PANEL_HEIGHT:
            if review_mode and (task_view or panel_y >= TIMELINE_BANDS[1][0] + timeline_offset):
                begin, end = editor.context_bounds()
                clicked_frame = frame_at_x(x, begin, end, MARGIN, video_width - MARGIN)
                if task_workflow and 397 <= panel_y <= 414:
                    channel = min(len(editor.classes) - 1, ((panel_y - 397 + 1) * len(editor.classes) - 1) // 18)
                    if editor.select_event_at(clicked_frame, editor.classes[channel]):
                        seek(editor.focus_frame())
                        return
                if task_workflow and 365 <= panel_y <= 382 and editor.mode != "waving":
                    editor.apply("edit_wave", clicked_frame)
                seek(clicked_frame)
            else:
                frame = frame_at_x(x, 0, count - 1, MARGIN, video_width - MARGIN)
                if review_mode:
                    editor.select_prompt_at(frame)
                    seek(frame)
                else:
                    seek(frame)

    keys = {ord(key): action for key, action in {
        "a": "previous", "d": "next", "p": "play", " ": "play", "q": "quit", "s": "save",
        "1": "label_0", "2": "label_1", "3": "label_2", "4": "label_3",
        "i": "start", "o": "finish", "e": "add_event", "x": "cancel", "j": "goto_start", "k": "goto_finish",
        "[": "previous_prompt", "]": "next_prompt", "g": "generate_directions", "r": "mode_direction", "z": "ignore_direction",
        "m": "mode_one_shot", "w": "mode_waving", "(": "previous_event", ")": "next_event",
        "h": "details",
    }.items()}
    keys.update({63234: "previous", 63235: "next", 13: "confirm", 10: "confirm", 8: "delete_event", 127: "delete_event"})
    if task_workflow:
        keys.update({13: "finish_task", 10: "finish_task", ord("w"): "edit_wave",
                     ord(","): "end_previous", ord("."): "end_next",
                     ord("1"): "new_open_hand", ord("2"): "new_close_hand",
                     ord("3"): "new_left_to_right", ord("4"): "new_right_to_left"})
        keys[ord("b")] = "more_actions"
        for key in (ord("m"), ord("r"), ord("z")):
            keys.pop(key)
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse)
        print("a/d frames, p play, [/] task, i/o bounds, g generate, ,/. adjust end, Enter finish and save, s save, q exit"
              if task_workflow else
              "a/d frames, p play, [/] task, i/o range, Enter confirm, x cancel, e add, 1-4 labels, g PCA, z dir IGNORE, m/w modes, s save, q exit")
        while running:
            if redraw or displayed != frame_index:
                if displayed is None or frame_index != displayed + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                max_height = config["display"]["annotation_max_height"] if view_only else config["display"]["annotation_preview_height"]
                scale = min(1.0, config["display"]["annotation_max_width"] / width, max_height / height)
                if scale < 1:
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                points = session["points"][frame_index] if session["detected"][frame_index] else None
                shown = draw_hand_landmarks(frame, points)
                video_height, video_width = shown.shape[:2]
                if view_only:
                    cv2.rectangle(shown, (0, 0), (video_width, 165), (0, 0, 0), -1)
                lines = [f"Frame {frame_index}/{count - 1} t={session['times'][frame_index]:.2f}s | {prompts[frame_index]}",
                         f"Selected: {editor.label} | {'SAVED' if not editor.changed else 'UNSAVED'} | valid={int(session['valid'][frame_index])}"]
                if review_mode:
                    interval = editor.current_interval()
                    bounds = f"[{interval['start_frame']}, {interval['end_frame']}]" if interval else "no candidate"
                    event_label = interval.get("label", "waving") if interval else editor.label
                    task = f"Task {editor.prompt_position + 1}/{len(editor.opportunities)} {editor.opportunity['label']} {editor.status()}"
                    if editor.periodic:
                        mode_name = {"one_shot": "OPEN/CLOSE", "direction": "DIRECTION", "waving": "WAVING INTERVAL"}[editor.mode]
                        lines[1] = f"MODE: {mode_name} | {task}"
                        if editor.mode == "waving":
                            selected = editor.selected_waving_interval()
                            bounds = f"[{selected['start_frame']}, {selected['end_frame']}]" if selected else f"[{editor.range_start}, {editor.range_end}]"
                            lines.append(f"WAVING range {bounds} | I START / O END | {'SAVED' if not editor.changed else 'UNSAVED'}")
                        else:
                            lines.append(f"Event {event_label} {bounds} | {'SAVED' if not editor.changed else 'UNSAVED'}")
                    else:
                        lines[1] = f"{task} | event {event_label} {bounds} | {'SAVED' if not editor.changed else 'UNSAVED'}"
                classes = config["labels"]["action_classes"]
                if view_only and waving_trace is not None:
                    lines.append(f"Waving FSM: {waving_trace['state'][frame_index]} active={int(waving_trace['active'][frame_index])} "
                                 f"expected={waving_trace['expected_direction'][frame_index] or '--'}")
                if view_only and event_marks:
                    current_events = [item["label"] for item in event_marks if item.get("frame_index") == frame_index]
                    if current_events:
                        lines.append("Events: " + ", ".join(current_events))
                    current_failures = [item for item in event_marks
                                        if item.get("frame_index") == frame_index and item.get("failure_kind")]
                    if current_failures:
                        lines.append("FAILURE: " + ", ".join(
                            f"{item['failure_kind']} {item['label']}" for item in current_failures))
                lines.append(" | ".join(f"{name}: target={session['targets'][frame_index, channel]:.0f} mask={int(session['mask'][frame_index, channel])}"
                                       + (f" score={predictions[frame_index, channel]:.2f}" if predictions is not None else "")
                                       for channel, name in enumerate(classes)))
                if not view_only:
                    header = np.full((ANNOTATION_HEADER_HEIGHT, video_width, 3), (26, 22, 18), np.uint8)
                    status = {"PENDING": "待审核", "REVIEWED": "已完成", "IGNORE": "已忽略",
                              "CONFIRMED": "已确认", "BACKGROUND": "背景已审核"}[editor.status()]
                    completed_count = sum(editor.status(index) != "PENDING" for index in range(len(editor.opportunities)))
                    category = "Waving" if task_workflow else "连续录像" if editor.kind == "continuous" else "开合动作"
                    lines = [f"{category} · {'负样本' if editor.is_negative else '正样本'}    已审核 {completed_count}/{len(editor.opportunities)} 段",
                             f"第 {editor.prompt_position + 1}/{len(editor.opportunities)} 段 · {status}    帧 {frame_index}/{count - 1} · {session['times'][frame_index]:.2f}s"]
                    interval = editor.current_interval()
                    if interval:
                        label = ACTION_NAMES.get(interval['label'], interval['label'])
                        lines.append(f"{label}  [{interval['start_frame']},{interval['end_frame']}]    {'未保存修改' if editor.changed else '进度已保存'}")
                    else:
                        selected = editor.selected_waving_interval() if editor.mode == "waving" else None
                        bounds = f"[{selected['start_frame']},{selected['end_frame']}]" if selected else f"起点 {editor.range_start if editor.range_start is not None else '未选'} / 完成 {editor.range_end if editor.range_end is not None else '未选'}"
                        lines.append(f"{bounds}    {'未保存修改' if editor.changed else '进度已保存'}")
                    for line, text in enumerate(lines):
                        draw_text(header, text, (16, 10 + 24 * line), 15 if line == 0 else 13, (245, 245, 245),
                                  config["display"]["text_font_path"], video_width - 32)
                else:
                    for line, text in enumerate(lines):
                        cv2.putText(shown, text, (16, 27 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1)
                panel = draw_action_panel(video_width, session, prompts, config, frame_index, editor,
                                          playing, predictions, view_only, event_marks,
                                          compact=not view_only and not details, extra_actions=extra_actions)
                canvas = np.vstack((shown, panel)) if view_only else np.vstack((header, shown, panel))
                size = (canvas.shape[1], canvas.shape[0])
                if canvas_size != size:
                    cv2.resizeWindow(window, *size)
                    canvas_size = size
                cv2.imshow(window, canvas)
                displayed, redraw = frame_index, False
            key = cv2.waitKeyEx(10 if playing else 30)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if ord("！") <= key <= ord("～"):
                key -= 0xFEE0
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if key in keys:
                perform(keys[key])
            if playing and time.monotonic() >= next_frame_at:
                frame_index += 1
                stop = editor.context_end() if review_mode else count - 1
                if frame_index >= stop:
                    playing = False
                else:
                    next_frame_at += session["times"][frame_index + 1] - session["times"][frame_index]
                redraw = True
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if not view_only and editor.changed:
            save()
