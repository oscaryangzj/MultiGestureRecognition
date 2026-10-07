import argparse
import time
from collections import deque

import cv2
import numpy as np

from gesture.action_inference import ActionThresholdDecoder, OnlineActionPredictor, load_action_model, preprocessing_config
from gesture.config import load_config, project_path
from gesture.data import load_frame_times
from gesture.hand_landmarker import draw_hand_landmarks
from gesture.state_recognizer import create_state_recognizer, recognize_video_frame


def draw_action_preview(frame, points, scores, state_scores, history, counts, last_event, elapsed, config, seconds, mirror):
    height, width = frame.shape[:2]
    scale = min(1.0, config["display"]["annotation_max_width"] / width,
                config["display"]["annotation_max_height"] / height)
    shown = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame.copy()
    shown = draw_hand_landmarks(shown, points)
    if mirror:
        shown = cv2.flip(shown, 1)
    height, width = shown.shape[:2]
    classes = config["labels"]["action_classes"]
    threshold = float(config["inference"]["action_threshold"])
    header = np.zeros((165, width, 3), np.uint8)
    active = [] if scores is None else [name for name, score in zip(classes, scores) if score >= threshold]
    status = "NO HAND" if scores is None else " + ".join(active) if active else "NONE"
    lines = [f"Action: {status}  | threshold={threshold:.2f}"]
    for channel, name in enumerate(classes):
        score = "--" if scores is None else f"{scores[channel]:.3f}"
        lines.append(f"{name}: {score}  | events: {counts[channel]}")
    state_text = "--" if state_scores is None else "  ".join(
        f"{name}={state_scores[name]:.2f}" for name in config["features"]["action_state_classes"])
    lines.append(f"State: {state_text}")
    lines.append("Last event: --" if last_event is None else f"Last event: {last_event[0]} at {last_event[1]:.2f}s")
    for row, text in enumerate(lines):
        cv2.putText(header, text, (16, 28 + row * 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)

    panel = np.full((196, width, 3), 24, np.uint8)
    cv2.putText(panel, f"Last {seconds:g}s | colored curves: scores | gray line: threshold | Q quit",
                (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
    for channel, name in enumerate(classes):
        top, bottom = 49 + channel * 75, 99 + channel * 75
        color = tuple(config["display"]["prompt_colors"][name])
        cv2.putText(panel, name, (16, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.rectangle(panel, (16, top), (width - 16, bottom), (50, 50, 50), 1)
        threshold_y = bottom - round(threshold * (bottom - top))
        cv2.line(panel, (16, threshold_y), (width - 16, threshold_y), (110, 110, 110), 1)
        curve = []
        for observed, probabilities in history:
            if probabilities is None:
                if len(curve) > 1:
                    cv2.polylines(panel, [np.asarray(curve, np.int32)], False, color, 1, cv2.LINE_AA)
                curve = []
                continue
            x = 16 + round((observed - elapsed + seconds) / seconds * (width - 32))
            y = bottom - round(float(probabilities[channel]) * (bottom - top))
            curve.append((x, y))
        if len(curve) > 1:
            cv2.polylines(panel, [np.asarray(curve, np.int32)], False, color, 1, cv2.LINE_AA)
    return np.vstack((header, shown, panel))


def main():
    parser = argparse.ArgumentParser(description="Run Action inference on a camera or recorded session")
    parser.add_argument("--session-id", help="Run the same online pipeline on a recorded session")
    args = parser.parse_args()
    config = load_config()
    model, checkpoint = load_action_model(config)
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
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open input video or camera")

    history = deque()
    counts = np.zeros(len(checkpoint["classes"]), int)
    last_event = None
    window = "Action recognition"
    started = time.monotonic()
    frame_index, previous_timestamp = 0, -1
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        print(f"Action on {next(model.parameters()).device}; {checkpoint['input_size']} features; press q to quit.")
        with create_state_recognizer(config) as recognizers:
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
                result = recognize_video_frame(recognizers, frame, timestamp_ms, include_world_landmarks=include_world)
                if include_world:
                    points, probabilities, world_points = result
                else:
                    points, probabilities = result
                    world_points = None
                state_scores = None if probabilities is None else dict(zip(config["labels"]["state_classes"], probabilities))
                scores, reset = predictor.update(points, state_scores, elapsed,
                                                 image_size=(frame.shape[1], frame.shape[0]),
                                                 world_points=world_points)
                for sample_scores, sample_reset in predictor.emissions:
                    if sample_reset:
                        decoder.reset()
                    for channel in decoder.update(sample_scores, elapsed):
                        counts[channel] += 1
                        name = checkpoint["classes"][channel]
                        last_event = (name, elapsed)
                        print(f"{elapsed:.2f}s {name} score={sample_scores[channel]:.3f} count={counts[channel]}")
                history.append((elapsed, scores))
                while history and history[0][0] <= elapsed - predictor.seconds:
                    history.popleft()
                shown = draw_action_preview(frame, points, scores, state_scores, history, counts, last_event,
                                            elapsed, config, predictor.seconds,
                                            not is_video and bool(config["display"]["mirror_preview"]))
                cv2.imshow(window, shown)
                delay_ms = 1
                if is_video and frame_index + 1 < len(frame_times):
                    remaining = frame_times[frame_index + 1] - elapsed - (time.monotonic() - frame_started)
                    delay_ms = max(1, round(remaining * 1000))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
