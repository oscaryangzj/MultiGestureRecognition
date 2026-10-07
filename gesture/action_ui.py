import json
import time

import cv2
import numpy as np

from gesture.action_data import make_action_targets, negative_input_regions
from gesture.action_features import make_action_features
from gesture.action_candidates import action_excluded_frames
from gesture.config import project_path
from gesture.hand_landmarker import draw_hand_landmarks
from gesture.prompts import label_segments, prompt_labels_for_session
from gesture.data import load_session


PANEL_HEIGHT = 388
MARGIN = 16
TIMELINE_BANDS = [(190, 209), (230, 249), (277, 296), (320, 339), (362, 374)]


def action_buttons(width, playing, view_only=False, review_mode=True, negative=False):
    rows = [[("previous", "A < PREV"), ("play", "P PAUSE" if playing else "P PLAY"),
             ("next", "D NEXT >"), ("quit", "Q EXIT")]]
    if not view_only:
        rows[0].insert(-1, ("save", "S SAVE"))
        rows += [[("previous_prompt", "[ PREV TASK"), ("next_prompt", "] NEXT TASK"),
                  ("confirm", "ENTER ACCEPT BLOCK >" if negative else "ENTER CONFIRM >"), ("cancel", "X CANCEL >")],
                 [("goto_start", "J GO TO START"), ("start", "I SET START"),
                  ("goto_finish", "K GO TO FINISH"), ("finish", "O SET FINISH")]]
        if negative:
            rows += [[("label_0", "1 OPEN HAND"), ("label_1", "2 CLOSE HAND"),
                      ("add_event", "E ADD EVENT"), ("delete_event", "BACKSPACE DEL EVENT")]]
    buttons = {}
    for row_index, row in enumerate(rows):
        slot = (width - 2 * MARGIN) / len(row)
        for index, (action, text) in enumerate(row):
            left, right = MARGIN + round(index * slot), MARGIN + round((index + 1) * slot) - 6
            top = 8 + 38 * row_index
            buttons[action] = ((left, top, right, top + 30), text)
    return buttons


def draw_action_panel(width, session, prompts, config, frame, editor, playing, predictions=None, view_only=False):
    panel = np.full((PANEL_HEIGHT, width, 3), 24, np.uint8)
    colors = config["display"]["prompt_colors"]
    classes = config["labels"]["action_classes"]
    review_mode = bool(editor.opportunities) and not view_only
    for action, ((left, top, right, bottom), text) in action_buttons(width, playing, view_only, review_mode, editor.is_negative).items():
        color = (65, 65, 65)
        if action.startswith("label_") and classes[int(action[-1])] == editor.label:
            color = tuple(colors[editor.label])
        if action == "confirm":
            color = (45, 125, 45)
        elif action == "cancel":
            color = (50, 50, 150)
        cv2.rectangle(panel, (left, top), (right, bottom), color, -1)
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(panel, text, (left + max(3, (right - left - tw) // 2), top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    message = "White curves: predicted scores. Colored blocks: target pulses." if view_only else editor.message
    cv2.putText(panel, message, (MARGIN, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 220, 255), 1)
    count = len(session["times"])
    span = width - 2 * MARGIN

    local_start, local_end = 0, count - 1
    if review_mode:
        local_start = editor.context_start()
        local_end = editor.opportunity["end_frame"]

    def x(index, overview=False):
        begin, length = (0, count) if overview else (local_start, local_end - local_start + 1)
        return MARGIN + round((index - begin) / length * span)

    def blocks(labels, band, overview=False):
        top, bottom = band
        begin, end = (0, count - 1) if overview else (local_start, local_end)
        for label, start, finish in label_segments(labels[begin:end + 1]):
            if label not in colors:
                continue
            left, right = x(start + begin, overview), x(finish + begin + 1, overview)
            cv2.rectangle(panel, (left, top), (max(left, right - 1), bottom), tuple(colors[label]), -1)

    captions = ["SESSION PROMPTS (click to select task)", "CONFIRMED ACTIONS / yellow: pending draft",
                classes[0] + " TARGET / SCORE", classes[1] + " TARGET / SCORE", "SUPERVISED FRAMES"]
    for caption, (top, bottom) in zip(captions, TIMELINE_BANDS):
        cv2.putText(panel, caption, (MARGIN, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (210, 210, 210), 1)
        cv2.rectangle(panel, (MARGIN, top), (width - MARGIN, bottom), (45, 45, 45), -1)
    blocks(prompts, TIMELINE_BANDS[0], overview=True)
    manual = np.full(count, "", dtype=object)
    for interval in editor.intervals:
        manual[interval["start_frame"]:interval["end_frame"] + 1] = interval["label"]
    blocks(manual, TIMELINE_BANDS[1])
    if review_mode:
        opportunity = editor.opportunity
        cv2.rectangle(panel, (x(opportunity["start_frame"], True), TIMELINE_BANDS[0][0]),
                      (x(opportunity["end_frame"] + 1, True) - 1, TIMELINE_BANDS[0][1]), (0, 255, 255), 1)
        candidate = editor.current_interval()
        if editor.status() == "PENDING" and candidate and candidate["start_frame"] is not None and candidate["end_frame"] is not None:
            start, finish = int(candidate["start_frame"]), int(candidate["end_frame"])
            if start <= finish:
                cv2.rectangle(panel, (x(max(local_start, start)), TIMELINE_BANDS[1][0]),
                              (x(min(local_end + 1, finish + 1)), TIMELINE_BANDS[1][1]), (30, 170, 220), -1)
    for channel, name in enumerate(classes):
        blocks(np.where(session["targets"][:, channel] > 0, name, ""), TIMELINE_BANDS[2 + channel])
        if predictions is not None:
            top, bottom = TIMELINE_BANDS[2 + channel]
            finite = np.isfinite(predictions[local_start:local_end + 1, channel])
            for start, end in zip(np.flatnonzero(np.diff(np.r_[False, finite, False].astype(int)) == 1) + local_start,
                                  np.flatnonzero(np.diff(np.r_[False, finite, False].astype(int)) == -1) + local_start):
                curve = np.asarray([(x(index), bottom - round(float(predictions[index, channel]) * (bottom - top))) for index in range(start, end)], np.int32)
                cv2.polylines(panel, [curve], False, (255, 255, 255), 1, cv2.LINE_AA)
    supervised = np.all(session["mask"], axis=1)
    top, bottom = TIMELINE_BANDS[-1]
    for present, start, end in label_segments(supervised[local_start:local_end + 1]):
        if present:
            cv2.rectangle(panel, (x(start + local_start), top), (max(x(start + local_start), x(end + local_start + 1) - 1), bottom), (80, 130, 80), -1)
    cv2.line(panel, (x(frame, True), TIMELINE_BANDS[0][0]), (x(frame, True), TIMELINE_BANDS[0][1]), (255, 255, 255), 2)
    if local_start <= frame <= local_end:
        cv2.line(panel, (x(frame), TIMELINE_BANDS[1][0]), (x(frame), TIMELINE_BANDS[-1][1]), (255, 255, 255), 2)
    return panel


def show_action_session(config, session, editor, predictions=None, view_only=False):
    folder = project_path(config["data"]["sessions_dir"]) / session["id"]
    annotation_path = folder / "annotations.json"
    prompts = prompt_labels_for_session(load_session(folder), len(session["times"]))
    capture = cv2.VideoCapture(str(folder / "video.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {folder / 'video.mp4'}")
    window = "Action replay" if view_only else "Action annotation"
    review_mode = bool(editor.opportunities) and not view_only
    frame_index, displayed = editor.focus_frame() if review_mode else 0, None
    playing, running, redraw = False, True, True
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    next_frame_at = 0.0
    count = len(session["times"])

    def refresh_targets():
        session["ignored_frames"] = action_excluded_frames(editor.annotations, session["opportunities"], count, config, session["times"])
        session["reviewed"] = bool(editor.annotations.get("action_reviewed", False))
        if session.get("sample_type") == "negative":
            session["input_regions"] = negative_input_regions(editor.annotations, session["opportunities"], count, config, session["times"])
            session["features"], session["valid"], session["segments"] = make_action_features(
                session["points"], session["detected"], session["state_scores"], session["times"], config,
                session["image_size"], session["input_regions"],
                world_points_by_frame=session.get("world_points"),
            )
        session["targets"], session["mask"] = make_action_targets(
            session["valid"], session["segments"], editor.intervals,
            session["reviewed"], config, session["ignored_frames"], session["times"],
        )

    if not view_only:
        refresh_targets()

    def seek(frame):
        nonlocal frame_index, playing, redraw
        frame_index = max(0, min(count - 1, int(frame)))
        playing, redraw = False, True

    def save():
        annotation_path.write_text(json.dumps(editor.annotations, indent=2), encoding="utf-8")
        editor.changed = False
        editor.message = f"Saved {len(editor.intervals)} actions; reviewed={editor.annotations.get('action_reviewed', False)}."

    def perform(action):
        nonlocal playing, running, next_frame_at, redraw
        if action == "quit":
            running = False
        elif action in ("previous", "next"):
            seek(frame_index + (-1 if action == "previous" else 1))
        elif action == "play":
            stop = editor.opportunity["end_frame"] if review_mode else count - 1
            if not playing and frame_index >= stop:
                seek(editor.focus_frame() if review_mode else 0)
            playing = not playing and count > 1
            if playing:
                next_frame_at = time.monotonic() + session["times"][frame_index + 1] - session["times"][frame_index]
        elif not view_only:
            seek(frame_index)
            if action == "save":
                save()
            elif action in ("goto_start", "goto_finish") and review_mode:
                interval = editor.current_interval()
                field = "start_frame" if action == "goto_start" else "end_frame"
                if interval and interval[field] is not None:
                    seek(interval[field])
                else:
                    editor.message = "No boundary yet: choose a frame and set it."
            else:
                position = editor.prompt_position
                editor.apply(action, frame_index)
                refresh_targets()
                if review_mode and (editor.prompt_position != position or action in ("previous_prompt", "next_prompt")):
                    seek(editor.focus_frame())
        redraw = True

    def mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        panel_y = y - video_height
        for action, ((left, top, right, bottom), _text) in action_buttons(video_width, playing, view_only, review_mode, editor.is_negative).items():
            if left <= x <= right and top <= panel_y <= bottom:
                perform(action)
                return
        if MARGIN <= x <= video_width - MARGIN and any(top <= panel_y <= bottom for top, bottom in TIMELINE_BANDS):
            fraction = (x - MARGIN) / (video_width - 2 * MARGIN)
            if review_mode and panel_y >= TIMELINE_BANDS[1][0]:
                begin = editor.context_start()
                seek(begin + fraction * (editor.opportunity["end_frame"] - begin + 1))
            else:
                frame = min(count - 1, int(fraction * count))
                if review_mode:
                    positions = [position for position, item in enumerate(editor.opportunities) if item["start_frame"] <= frame <= item["end_frame"]]
                    editor.select_prompt(positions[0] if positions else 0)
                    seek(editor.focus_frame())
                else:
                    seek(frame)

    keys = {ord(key): action for key, action in {
        "a": "previous", "d": "next", "p": "play", " ": "play", "q": "quit", "s": "save",
        "1": "label_0", "2": "label_1", "i": "start", "o": "finish", "e": "add_event",
        "x": "cancel", "j": "goto_start", "k": "goto_finish", "[": "previous_prompt", "]": "next_prompt",
    }.items()}
    keys.update({63234: "previous", 63235: "next", 13: "confirm", 10: "confirm", 8: "delete_event", 127: "delete_event"})
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, mouse)
        print("a/d frames, p/space play, [/] tasks, enter confirm, x cancel, j/k boundaries, i/o adjust, negative: 1/2 class, e add, backspace delete; s save, q exit")
        while running:
            if redraw or displayed != frame_index:
                if displayed is None or frame_index != displayed + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                scale = min(1.0, config["display"]["annotation_max_width"] / width, config["display"]["annotation_max_height"] / height)
                if scale < 1:
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                points = session["points"][frame_index] if session["detected"][frame_index] else None
                shown = draw_hand_landmarks(frame, points)
                video_height, video_width = shown.shape[:2]
                cv2.rectangle(shown, (0, 0), (video_width, 108), (0, 0, 0), -1)
                lines = [f"Frame {frame_index}/{count - 1} t={session['times'][frame_index]:.2f}s | {prompts[frame_index]}",
                         f"Selected: {editor.label} | {'SAVED' if not editor.changed else 'UNSAVED'} | valid={int(session['valid'][frame_index])}"]
                if review_mode:
                    interval = editor.current_interval()
                    bounds = f"[{interval['start_frame']}, {interval['end_frame']}]" if interval else "no candidate"
                    lines[1] = f"Task {editor.prompt_position + 1}/{len(editor.opportunities)} {editor.opportunity['label']} {editor.status()} | event {editor.label} {bounds} | {'SAVED' if not editor.changed else 'UNSAVED'}"
                classes = config["labels"]["action_classes"]
                lines.append(" | ".join(f"{name}: target={session['targets'][frame_index, channel]:.0f} mask={int(session['mask'][frame_index, channel])}"
                                       + (f" score={predictions[frame_index, channel]:.2f}" if predictions is not None else "")
                                       for channel, name in enumerate(classes)))
                for line, text in enumerate(lines):
                    cv2.putText(shown, text, (16, 27 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1)
                panel = draw_action_panel(video_width, session, prompts, config, frame_index, editor, playing, predictions, view_only)
                cv2.imshow(window, np.vstack((shown, panel)))
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
                stop = editor.opportunity["end_frame"] if review_mode else count - 1
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
