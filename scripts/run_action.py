import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch

from gesture.action_decoding import (ActionEvent, ActionOutput, ActionThresholdDecoder, write_action_events,
                                     write_final_events)
from gesture.action_inference import (OnlineActionPredictor, action_output_directory,
                                      load_action_model, preprocessing_config)
from gesture.config import load_config, project_path
from gesture.data import load_frame_times
from gesture.display_text import draw_text
from gesture.hand_landmarker import draw_hand_landmarks
from gesture.state_recognizer import create_state_recognizer, recognize_video_frame
from gesture.waving_fsm import WavingFSM


def current_state(state_scores, config):
    if state_scores is None:
        return "no_hand", None
    label = max(state_scores, key=state_scores.get)
    confidence = float(state_scores[label])
    if confidence < float(config["inference"]["state_threshold"]):
        return "other", None
    return label, confidence


def rounded_rectangle(image, bounds, color, radius=12):
    left, top, right, bottom = bounds
    cv2.rectangle(image, (left + radius, top), (right - radius, bottom), color, -1)
    cv2.rectangle(image, (left, top + radius), (right, bottom - radius), color, -1)
    for x, y in ((left + radius, top + radius), (right - radius, top + radius),
                 (left + radius, bottom - radius), (right - radius, bottom - radius)):
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


def draw_action_preview(frame, points, scores, state_scores, history, counts, last_event, elapsed,
                        config, seconds, mirror, thresholds, waving_state, recording, output):
    height, width = frame.shape[:2]
    scale = min(1.0, config["display"]["annotation_max_width"] / width,
                config["display"]["annotation_max_height"] / height)
    shown = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame.copy()
    shown = draw_hand_landmarks(shown, points)
    if mirror:
        shown = cv2.flip(shown, 1)
    height, width = shown.shape[:2]
    margin, canvas_width = 16, width + 32
    background, card = (26, 22, 18), (42, 34, 28)
    text_color, muted = (244, 240, 233), (169, 153, 136)
    colors = config["display"]["prompt_colors"]
    font = config["display"]["action_font_path"]

    def text(image, value, position, size, color=text_color, max_width=None):
        draw_text(image, value, position, size, color, font, max_width=max_width)

    state, confidence = current_state(state_scores, config)
    state_names = {"opened": "OPEN PALM", "closed": "FIST", "thumb_up": "THUMB UP",
                   "other": "OTHER", "no_hand": "NO HAND"}
    state_hint = (f"Confidence {confidence:.0%}" if confidence is not None else
                  "No matching hand shape" if state == "other" else "Place your hand in view")
    action_name = "BYEBYE" if output == "waving" else output.replace("_", " ").upper()
    action_hint = ("Waiting for an action" if output == "NULL" else
                   "Continuous wave" if output == "waving" else "Gesture detected")
    header = np.full((160, canvas_width, 3), background, np.uint8)
    text(header, "Gesture recognition", (margin, 17), 18)
    if recording:
        rounded_rectangle(header, (canvas_width - 82, 12, canvas_width - margin, 36), (51, 44, 100), 8)
        cv2.circle(header, (canvas_width - 68, 24), 3, (111, 125, 255), -1, cv2.LINE_AA)
        text(header, "REC", (canvas_width - 58, 17), 12, (184, 187, 255))
    card_width = (canvas_width - margin * 3) // 2
    cards = [("STATE", state_names[state], state_hint, state),
             ("ACTION", action_name, action_hint, output)]
    for index, (caption, value, hint, key) in enumerate(cards):
        left = margin + index * (card_width + margin)
        accent = tuple(int(component * 0.5 + 122) for component in colors[key]) if key in colors else muted
        rounded_rectangle(header, (left, 50, left + card_width, 145), card)
        cv2.circle(header, (left + 18, 68), 3, accent, -1, cv2.LINE_AA)
        text(header, caption, (left + 29, 62), 11, muted)
        text(header, value, (left + 16, 84), min(36, round(width / 22)),
             accent if key in colors else muted, card_width - 32)
        text(header, hint, (left + 16, 125), 12, muted, card_width - 32)

    video = np.full((height, canvas_width, 3), background, np.uint8)
    mask = np.zeros((height, width), np.uint8)
    rounded_rectangle(mask, (0, 0, width - 1, height - 1), 255)
    cv2.copyTo(shown, mask, video[:, margin:margin + width])
    footer = np.full((56, canvas_width, 3), background, np.uint8)
    tracking_color = (156, 209, 144) if points is not None else muted
    cv2.circle(footer, (margin + 4, 27), 3, tracking_color, -1, cv2.LINE_AA)
    text(footer, "Hand tracked" if points is not None else "No hand detected", (margin + 14, 21), 12, muted)
    keys = f"R {'Stop' if recording else 'Record'}    D {'Hide' if config['display']['action_show_debug'] else 'Details'}    Q Quit"
    text(footer, keys, (canvas_width - margin - 240, 21), 12, muted, 240)
    if not config["display"]["action_show_debug"]:
        return np.vstack((header, video, footer))

    classes = config["labels"]["action_classes"]
    panel = np.full((88 + 72 * len(classes), canvas_width, 3), background, np.uint8)
    state_text = "--" if state_scores is None else "  ".join(f"{name} {value:.2f}" for name, value in state_scores.items())
    text(panel, f"State scores: {state_text}", (margin, 18), 12, muted)
    text(panel, f"FSM {waving_state.state}    Next: {waving_state.expected_direction or '--'}", (margin, 38), 12, muted)
    evidence = "--" if last_event is None else f"{last_event[0]} at {last_event[1]:.2f}s"
    text(panel, f"Last evidence: {evidence}    History: {seconds:g}s", (margin, 58), 12, muted)
    for channel, name in enumerate(classes):
        top, bottom = 107 + channel * 72, 145 + channel * 72
        color = tuple(colors[name])
        score = "--" if scores is None else f"{scores[channel]:.3f}"
        text(panel, f"{name}  {score} / {thresholds[channel]:.2f}    events {counts[channel]}",
             (margin, top - 19), 12, muted)
        cv2.rectangle(panel, (margin, top), (canvas_width - margin, bottom), card, -1)
        threshold_y = bottom - round(float(thresholds[channel]) * (bottom - top))
        cv2.line(panel, (margin, threshold_y), (canvas_width - margin, threshold_y), (88, 75, 62), 1)
        curve = []
        for observed, probabilities in history:
            if probabilities is None:
                if len(curve) > 1:
                    cv2.polylines(panel, [np.asarray(curve, np.int32)], False, color, 1, cv2.LINE_AA)
                curve = []
                continue
            x = margin + round((observed - elapsed + seconds) / seconds * (canvas_width - margin * 2))
            y = bottom - round(float(probabilities[channel]) * (bottom - top))
            curve.append((x, y))
        if len(curve) > 1:
            cv2.polylines(panel, [np.asarray(curve, np.int32)], False, color, 1, cv2.LINE_AA)
    return np.vstack((header, video, panel, footer))


def main():
    parser = argparse.ArgumentParser(description="Run Action inference on a camera or recorded session")
    parser.add_argument("--session-id", help="Run the same online pipeline on a recorded session")
    args = parser.parse_args()
    config = load_config()
    cv2.setNumThreads(int(config["inference"]["action_opencv_threads"]))
    device = config["inference"]["action_device"]
    if torch.device(device).type == "cpu":
        torch.set_num_threads(int(config["inference"]["action_cpu_threads"]))
    model, checkpoint = load_action_model(config, device=device)
    config = preprocessing_config(config, checkpoint)
    predictor = OnlineActionPredictor(model, checkpoint, config)
    decoder = ActionThresholdDecoder(checkpoint["classes"], config)
    is_video = args.session_id is not None
    if is_video:
        folder = project_path(config["data"]["sessions_dir"]) / args.session_id
        frame_times = load_frame_times(folder / "frames.csv")
        capture = cv2.VideoCapture(str(folder / "video.mp4"))
    else:
        capture = cv2.VideoCapture(int(config["data"]["camera_index"]))
        capture.set(cv2.CAP_PROP_FPS, float(config["inference"]["action_camera_fps"]))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open input video or camera")

    history = deque()
    counts = np.zeros(len(checkpoint["classes"]), int)
    last_event = None
    event_log, conflict_log = [], []
    final_event_log = []
    waving_fsm = WavingFSM(config)
    action_output = ActionOutput(config)
    thresholds = decoder.thresholds.copy()
    waving_state = waving_fsm.update([], 0.0)
    recording = None
    window = "Action recognition"
    started = time.monotonic()
    frame_index, previous_timestamp = 0, -1

    def advance_output(events, now, reset):
        nonlocal waving_state
        waving_state = waving_fsm.update(events, now, reset)
        for transition in waving_state.transitions:
            final_event_log.append(ActionEvent(transition.label, transition.time_seconds, transition.score,
                                               frame_index if is_video else None))
        previous = action_output.label
        label = action_output.update(events, waving_state.active, now)
        if label != previous:
            print(f"{now:.2f}s OUTPUT: {'BYEBYE' if label == 'waving' else label.upper()}")

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        print(f"Action on {next(model.parameters()).device}; {checkpoint['input_size']} features; press q to quit.")
        state_classes = config["labels"]["state_classes"]
        with create_state_recognizer(config, classes=state_classes) as recognizers:
            while True:
                frame_started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    break
                if is_video:
                    if frame_index >= len(frame_times):
                        raise ValueError("video.mp4 has more frames than frames.csv")
                    elapsed = frame_times[frame_index]
                else:
                    elapsed = time.monotonic() - started
                timestamp_ms = max(round(elapsed * 1000), previous_timestamp + 1)
                previous_timestamp = timestamp_ms
                include_world = predictor.include_world_landmarks
                max_width = int(config["inference"]["action_input_max_width"])
                inference_frame = (cv2.resize(frame, (max_width, round(frame.shape[0] * max_width / frame.shape[1])))
                                   if frame.shape[1] > max_width else frame)
                result = recognize_video_frame(recognizers, inference_frame, timestamp_ms,
                                               include_world_landmarks=include_world)
                if include_world:
                    points, probabilities, world_points = result
                else:
                    points, probabilities = result
                    world_points = None
                state_scores = None if probabilities is None else dict(zip(state_classes, probabilities))
                scores, reset = predictor.update(points, state_scores, elapsed,
                                                 image_size=(frame.shape[1], frame.shape[0]),
                                                 world_points=world_points)
                for sample_scores, sample_reset, _sample_time, available_time in predictor.timed_emissions:
                    if sample_reset:
                        decoder.reset()
                    atomic_events = []
                    for channel in decoder.update(sample_scores, available_time):
                        counts[channel] += 1
                        name = checkpoint["classes"][channel]
                        last_event = (name, available_time)
                        decoded = next(item for item in decoder.last_events if item.label == name)
                        logged = ActionEvent(decoded.label, decoded.time_seconds, decoded.score,
                                             frame_index if is_video else None)
                        event_log.append(logged)
                        if name in config["inference"]["action_event_groups"]["one_shot"]:
                            final_event_log.append(logged)
                        atomic_events.append(decoded)
                    advance_output(atomic_events, available_time, sample_reset)
                    conflict_log.extend(decoder.conflicts)
                if not predictor.timed_emissions:
                    if reset:
                        decoder.reset()
                    advance_output([], elapsed, reset)
                elif elapsed > predictor.timed_emissions[-1][3]:
                    advance_output([], elapsed, reset)
                if recording is not None:
                    record_time = elapsed - recording["started"]
                    recording["writer"].write(frame)
                    recording["rows"].append((len(recording["rows"]), record_time))
                history.append((elapsed, scores))
                while history and history[0][0] <= elapsed - predictor.seconds:
                    history.popleft()
                shown = draw_action_preview(frame, points, scores, state_scores, history, counts, last_event,
                                            elapsed, config, predictor.seconds,
                                            not is_video and bool(config["display"]["mirror_preview"]),
                                            thresholds, waving_state, recording is not None, action_output.label)
                cv2.imshow(window, shown)
                delay_ms = 1
                if is_video and frame_index + 1 < len(frame_times):
                    remaining = frame_times[frame_index + 1] - elapsed - (time.monotonic() - frame_started)
                    delay_ms = max(1, round(remaining * 1000))
                elif not is_video:
                    remaining = 1 / float(config["inference"]["action_camera_fps"]) - (time.monotonic() - frame_started)
                    delay_ms = max(1, round(remaining * 1000))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("d"), ord("D")):
                    config["display"]["action_show_debug"] = not config["display"]["action_show_debug"]
                if key in (ord("r"), ord("R")) and not is_video:
                    if recording is None:
                        collector = input("Collector name: ").strip()
                        slug = "".join(char if char.isalnum() or char in "-_" else "_" for char in collector).strip("-_")
                        if not slug:
                            print("Recording cancelled: enter a collector name.")
                        else:
                            session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"
                            folder = project_path(config["data"]["sessions_dir"]) / session_id
                            folder.mkdir(parents=True)
                            size = (frame.shape[1], frame.shape[0])
                            writer = cv2.VideoWriter(str(folder / "video.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), float(config["data"]["fps_target"]), size)
                            if not writer.isOpened():
                                writer.release()
                                (folder / "video.mp4").unlink(missing_ok=True)
                                folder.rmdir()
                                raise RuntimeError("Could not create recording video")
                            recording = {"id": session_id, "folder": folder, "collector": collector,
                                         "writer": writer, "rows": [], "started": elapsed}
                            print(f"Recording: {session_id}; press R to stop and save.")
                    else:
                        finish_online_recording(recording)
                        recording = None
                frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if recording is not None:
            finish_online_recording(recording)
        if is_video:
            output = action_output_directory(config) / "evaluation" / args.session_id
            write_action_events(output / "events.csv", event_log, conflict_log)
            write_final_events(output / "final_events.csv", final_event_log)


def finish_online_recording(recording):
    folder, writer, rows = recording["folder"], recording["writer"], recording["rows"]
    writer.release()
    with (folder / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        output = csv.writer(stream)
        output.writerow(["frame_index", "elapsed_seconds"])
        output.writerows(rows)
    session = {"session_id": recording["id"], "collector": recording["collector"],
               "target_model": "action", "kind": "continuous", "sample_type": "positive",
               "pattern": "online_continuous", "prompts": [], "phases": []}
    (folder / "session.json").write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    annotations = {"session_id": recording["id"], "action_review_mode": "continuous",
                   "action_reviewed": False, "action_intervals": [], "action_candidates": [],
                   "action_ignored_intervals": [], "action_background_prompts": [],
                   "direction_candidates": [], "direction_reviewed_intervals": [],
                   "direction_ignored_intervals": [], "one_shot_reviewed_intervals": [],
                   "waving_intervals": [], "waving_candidates": [], "waving_reviewed_intervals": []}
    (folder / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} frames: {folder}")
    print(f"Next: python -m scripts.extract_action --session-id {recording['id']}")


if __name__ == "__main__":
    main()
