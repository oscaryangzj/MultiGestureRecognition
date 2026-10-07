import argparse
import json
import time

import cv2
import numpy as np

from gesture.config import load_config, project_path
from gesture.data import load_landmarks_csv
from gesture.hand_landmarker import draw_hand_landmarks
from gesture.labels import IGNORE_TARGET, make_state_targets
from gesture.prompts import REST_LABEL, label_segments, prompt_labels_for_session


PANEL_HEIGHT = 230
TIMELINE_MARGIN = 16
PROMPT_BAND = (116, 140)
LABEL_BAND = (184, 204)


def annotation_buttons(width, playing):
    rows = [
        [("previous", "< PREV A"), ("play", "PAUSE P" if playing else "PLAY P"),
         ("next", "NEXT D >"), ("save", "S SAVE"), ("quit", "Q EXIT")],
        [("start", "I START"), ("cancel", "X CANCEL PROMPT")],
    ]
    buttons = {}
    for row_index, row in enumerate(rows):
        slot_width = (width - 2 * TIMELINE_MARGIN) / len(row)
        top = 8 + row_index * 38
        for index, (name, text) in enumerate(row):
            left = TIMELINE_MARGIN + round(index * slot_width)
            right = TIMELINE_MARGIN + round((index + 1) * slot_width) - 6
            buttons[name] = ((left, top, right, top + 30), text)
    return buttons


def without_interval_range(intervals, start, end):
    """Remove labels in this prompt while preserving labels outside it."""
    remaining = []
    for interval in intervals:
        begin, finish = int(interval["start_frame"]), int(interval["end_frame"])
        if finish < start or begin > end:
            remaining.append(interval)
            continue
        if begin < start:
            remaining.append({**interval, "end_frame": start - 1})
        if finish > end:
            remaining.append({**interval, "start_frame": end + 1})
    return remaining


def draw_annotation_panel(width, elapsed, prompt_labels, targets, classes, colors, frame_index, playing, message=""):
    panel = np.full((PANEL_HEIGHT, width, 3), 24, dtype=np.uint8)
    for _name, ((left, top, right, bottom), text) in annotation_buttons(width, playing).items():
        cv2.rectangle(panel, (left, top), (right, bottom), (70, 70, 70), -1)
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0]
        cv2.putText(panel, text, (left + (right - left - text_width) // 2, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

    cv2.putText(panel, message, (TIMELINE_MARGIN, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 220, 255), 1)
    cv2.putText(panel, "PROMPTS (guide)", (TIMELINE_MARGIN, 109), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(panel, "LABELS (confirmed start; empty = IGNORE)", (TIMELINE_MARGIN, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    timeline_width = width - 2 * TIMELINE_MARGIN
    frame_count = len(elapsed)
    manual_labels = [classes[int(target)] if target != IGNORE_TARGET else "IGNORE" for target in targets]
    for labels, (top, bottom) in ((prompt_labels, PROMPT_BAND), (manual_labels, LABEL_BAND)):
        cv2.rectangle(panel, (TIMELINE_MARGIN, top), (width - TIMELINE_MARGIN, bottom), (45, 45, 45), -1)
        for label, start, end in label_segments(labels):
            if label == "IGNORE":
                continue
            left = TIMELINE_MARGIN + round(start / frame_count * timeline_width)
            right = TIMELINE_MARGIN + round((end + 1) / frame_count * timeline_width)
            cv2.rectangle(panel, (left, top), (right - 1, bottom), tuple(colors[label]), -1)
            text = label.replace("_", "").upper()
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            if right - left >= text_width + 8:
                cv2.putText(panel, text, (left + (right - left - text_width) // 2, bottom - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    for index in np.linspace(0, frame_count - 1, 5, dtype=int):
        x = TIMELINE_MARGIN + round((index + 0.5) / frame_count * timeline_width)
        text = f"{elapsed[index]:.1f}s"
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        text_x = max(TIMELINE_MARGIN, min(width - TIMELINE_MARGIN - text_width, x - text_width // 2))
        cv2.putText(panel, text, (text_x, 157), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cursor_x = TIMELINE_MARGIN + round((frame_index + 0.5) / frame_count * timeline_width)
    cv2.line(panel, (cursor_x, PROMPT_BAND[0]), (cursor_x, LABEL_BAND[1]), (255, 255, 255), 2)
    cv2.putText(panel, "START: current frame to prompt end | CANCEL: ignore whole prompt", (TIMELINE_MARGIN, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
    return panel


def main():
    parser = argparse.ArgumentParser(description="Confirm the stable start of each State prompt")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    config = load_config()
    session_dir = project_path(config["data"]["sessions_dir"]) / args.session_id
    with (session_dir / "session.json").open(encoding="utf-8") as stream:
        session = json.load(stream)
    if session.get("target_model") != "state":
        raise ValueError("State annotation requires target_model: state")

    annotation_path = session_dir / "annotations.json"
    annotations = (
        json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation_path.exists()
        else {"session_id": args.session_id, "state_intervals": []}
    )
    intervals = annotations.setdefault("state_intervals", [])
    elapsed, hand_detected, points_by_frame = load_landmarks_csv(session_dir / "landmarks.csv")
    cap = cv2.VideoCapture(str(session_dir / "video.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {session_dir / 'video.mp4'}")
    frame_count = len(hand_detected)
    if frame_count == 0:
        raise ValueError("Session has no landmark frames")

    classes = config["labels"]["state_classes"]
    key_actions = {ord(key): action for key, action in {
        "a": "previous", "d": "next", " ": "play", "p": "play",
        "[": "start", "i": "start", "【": "start",
        "x": "cancel", "s": "save", "q": "quit",
    }.items()}
    key_actions.update({63234: "previous", 63235: "next"})
    colors = config["display"]["prompt_colors"]
    prompt_labels = prompt_labels_for_session(session, frame_count)
    frame_index = 0
    changed = False
    targets = make_state_targets(hand_detected, intervals, classes, prompt_labels)
    playing = False
    next_frame_at = 0.0
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    displayed_frame = None
    redraw = True
    running = True
    message = "Choose the first stable frame, then click START."

    def current_prompt():
        return next((prompt for prompt in session["prompts"]
                     if prompt["gesture"] != REST_LABEL
                     and prompt.get("start_frame") is not None
                     and prompt.get("end_frame") is not None
                     and int(prompt["start_frame"]) <= frame_index <= int(prompt["end_frame"])), None)

    def seek(index):
        nonlocal frame_index, playing, redraw
        frame_index = max(0, min(frame_count - 1, index))
        playing = False
        redraw = True

    def toggle_playback():
        nonlocal frame_index, playing, next_frame_at, redraw
        if not playing and frame_index == frame_count - 1:
            frame_index = 0
        playing = not playing and frame_count > 1
        if playing:
            next_frame_at = time.monotonic() + float(elapsed[frame_index + 1] - elapsed[frame_index])
        redraw = True

    def save_annotations():
        nonlocal changed
        annotations["session_id"] = args.session_id
        annotation_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
        changed = False

    def perform_action(action):
        nonlocal targets, changed, running, message
        if action == "play":
            toggle_playback()
            return
        if action == "previous" or action == "next":
            seek(frame_index + (-1 if action == "previous" else 1))
            return
        if action == "quit":
            running = False
            return
        seek(frame_index)
        if action in ("start", "cancel"):
            prompt = current_prompt()
            if prompt is None:
                message = "REST: select a gesture prompt first."
                return
            start, end = int(prompt["start_frame"]), int(prompt["end_frame"])
            label = prompt["gesture"]
            intervals[:] = without_interval_range(intervals, start, end)
            if action == "start":
                intervals.append({"start_frame": frame_index, "end_frame": end, "label": label})
                message = f"Set {label.upper()} [{frame_index}, {end}]. Click SAVE."
            else:
                message = f"Cancelled {label.upper()} prompt [{start}, {end}]: IGNORE."
            intervals.sort(key=lambda item: int(item["start_frame"]))
            targets = make_state_targets(hand_detected, intervals, classes, prompt_labels)
            changed = True
        elif action == "save":
            save_annotations()
            message = f"Saved {len(intervals)} State interval(s)."
            print(message)

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        panel_y = y - video_height
        for name, ((left, top, right, bottom), _text) in annotation_buttons(video_width, playing).items():
            if left <= x <= right and top <= panel_y <= bottom:
                perform_action(name)
                return
        if TIMELINE_MARGIN <= x <= video_width - TIMELINE_MARGIN and any(
            top <= panel_y <= bottom for top, bottom in (PROMPT_BAND, LABEL_BAND)
        ):
            seek(int((x - TIMELINE_MARGIN) / (video_width - 2 * TIMELINE_MARGIN) * frame_count))

    cv2.namedWindow("State annotation", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("State annotation", on_mouse)

    print("Click the window buttons, or use: a/left previous | d/right next | space/p play/pause | i/[ start to prompt end | x cancel whole prompt | s save | q quit")
    try:
        while running:
            if redraw or frame_index != displayed_frame:
                if displayed_frame is None or frame_index != displayed_frame + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    break

                height, width = frame.shape[:2]
                scale = min(1.0, config["display"]["annotation_max_width"] / width, config["display"]["annotation_max_height"] / height)
                if scale < 1.0:
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                points = points_by_frame[frame_index] if hand_detected[frame_index] else None
                shown = draw_hand_landmarks(frame, points)
                video_height, video_width = shown.shape[:2]
                if prompt_labels[frame_index] == REST_LABEL:
                    status = "REST: IGNORE"
                elif not hand_detected[frame_index]:
                    status = "NO HAND: IGNORE"
                elif targets[frame_index] != IGNORE_TARGET:
                    status = f"MANUAL: {classes[int(targets[frame_index])]}"
                else:
                    status = "UNLABELED: IGNORE"

                cv2.rectangle(shown, (0, 0), (shown.shape[1], 104), (0, 0, 0), -1)
                cv2.putText(shown, f"Frame {frame_index}/{frame_count - 1}  t={elapsed[frame_index]:.2f}s  {'PLAYING' if playing else 'PAUSED'}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1)
                prompt_name = prompt_labels[frame_index].replace("_", "").upper()
                cv2.putText(shown, f"{prompt_name} | {status}", (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
                prompt = current_prompt()
                bounds = f"Prompt [{prompt['start_frame']}, {prompt['end_frame']}]" if prompt is not None else "REST"
                cv2.putText(shown, f"{bounds} | {'UNSAVED' if changed else 'SAVED'}", (16, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
                panel = draw_annotation_panel(video_width, elapsed, prompt_labels, targets, classes, colors, frame_index, playing, message)
                shown = np.vstack((shown, panel))
                cv2.imshow("State annotation", shown)
                displayed_frame = frame_index
                redraw = False

            key = cv2.waitKeyEx(10 if playing else 30)
            if cv2.getWindowProperty("State annotation", cv2.WND_PROP_VISIBLE) < 1:
                break
            if ord("！") <= key <= ord("～"):
                key -= 0xFEE0
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if key in key_actions:
                perform_action(key_actions[key])
            if playing and time.monotonic() >= next_frame_at:
                frame_index += 1
                if frame_index == frame_count - 1:
                    playing = False
                else:
                    next_frame_at += float(elapsed[frame_index + 1] - elapsed[frame_index])
                redraw = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if changed:
        save_annotations()
        print(f"Saved {len(intervals)} State intervals to {annotation_path}")


if __name__ == "__main__":
    main()
