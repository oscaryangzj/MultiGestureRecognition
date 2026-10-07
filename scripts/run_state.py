import argparse
import math
import time
from datetime import datetime

import cv2

from gesture.config import load_config, project_path
from gesture.data import load_frame_times
from gesture.hand_landmarker import draw_hand_landmarks
from gesture.state_recognizer import create_state_recognizer, recognize_video_frame


class Recording:
    def __init__(self, output_dir, fps):
        self.output_dir = output_dir
        self.fps = fps
        self.writer = None
        self.last_frame = None

    def start(self, frame, elapsed):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
        height, width = frame.shape[:2]
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(f"Could not create recording: {self.path}")
        self.started = elapsed
        self.frame_count = 0
        self.last_frame = None
        print(f"Recording: {self.path}")

    def _write_until(self, target_count, frame):
        while self.frame_count < target_count:
            self.writer.write(frame)
            self.frame_count += 1

    def write(self, frame, elapsed):
        # Hold the last displayed frame between observations to preserve playback speed.
        target_count = int((elapsed - self.started) * self.fps) + 1
        if self.last_frame is not None:
            self._write_until(target_count - 1, self.last_frame)
        self._write_until(target_count, frame)
        self.last_frame = frame.copy()

    def stop(self, elapsed):
        if self.writer is None:
            return
        if self.last_frame is not None:
            self._write_until(math.ceil((elapsed - self.started) * self.fps), self.last_frame)
        self.writer.release()
        self.writer = None
        print(f"Saved recording: {self.path}")


def main():
    parser = argparse.ArgumentParser(description="Run State classification on a camera or video")
    parser.add_argument("--session-id", help="Replay a recorded session video")
    args = parser.parse_args()
    config = load_config()
    classes = config["labels"]["state_classes"]
    threshold = float(config["inference"]["state_threshold"])

    is_video = args.session_id is not None
    if is_video:
        session_dir = project_path(config["data"]["sessions_dir"]) / args.session_id
        video_path = session_dir / "video.mp4"
        frame_times = load_frame_times(session_dir / "frames.csv")
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(config["data"]["fps_target"])
    else:
        capture = cv2.VideoCapture(int(config["data"]["camera_index"]))
        fps = float(config["data"]["fps_target"])
    if not capture.isOpened():
        raise RuntimeError("Could not open input video or camera")

    recording = Recording(project_path(config["inference"]["recordings_dir"]), fps)
    window_name = "Static State recognition"
    toggle_requested = False
    button_bounds = None

    def on_mouse(event, x, y, _flags, _param):
        nonlocal toggle_requested
        if event == cv2.EVENT_LBUTTONDOWN and button_bounds is not None:
            left, top, right, bottom = button_bounds
            if left <= x <= right and top <= y <= bottom:
                toggle_requested = True

    started = time.monotonic()
    elapsed = 0.0
    frame_index = 0
    previous_timestamp = -1
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)
        print("Click START REC / STOP REC, or press r to toggle recording; q to quit.")
        with create_state_recognizer(config) as recognizers:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if is_video:
                    if frame_index >= len(frame_times):
                        raise ValueError("video.mp4 has more frames than frames.csv")
                    elapsed = frame_times[frame_index]
                else:
                    elapsed = time.monotonic() - started
                timestamp_ms = max(int(round(elapsed * 1000)), previous_timestamp + 1)
                previous_timestamp = timestamp_ms

                points, probabilities = recognize_video_frame(recognizers, frame, timestamp_ms)
                views = [draw_hand_landmarks(frame.copy(), points)]
                if not is_video and config["display"]["mirror_preview"]:
                    views.append(cv2.flip(views[0], 1))
                for shown in views:
                    if points is None:
                        cv2.putText(shown, "NO HAND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 255), 2)
                    else:
                        active = [label for label, probability in zip(classes, probabilities) if probability >= threshold]
                        cv2.putText(
                            shown,
                            " + ".join(active) if active else "NONE",
                            (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (60, 240, 60) if active else (0, 120, 255),
                            2,
                        )
                        for index, (label, probability) in enumerate(zip(classes, probabilities)):
                            cv2.putText(
                                shown,
                                f"{label}: {probability:.2f}",
                                (20, 70 + index * 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (245, 245, 245),
                                1,
                            )
                if toggle_requested:
                    if recording.writer is None:
                        recording.start(views[0], elapsed)
                    else:
                        recording.stop(elapsed)
                    toggle_requested = False

                height, width = shown.shape[:2]
                button_bounds = (max(0, width - 240), height - 65, width - 10, height - 15)
                left, top, right, bottom = button_bounds
                is_recording = recording.writer is not None
                color = (40, 40, 200) if is_recording else (40, 140, 40)
                for shown in views:
                    cv2.rectangle(shown, (left, top), (right, bottom), color, -1)
                    cv2.putText(shown, "R STOP REC" if is_recording else "R START REC", (left + 12, top + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    if is_recording:
                        cv2.putText(shown, f"REC {elapsed - recording.started:.1f}s", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                if is_recording:
                    recording.write(views[0], elapsed)
                cv2.imshow(window_name, views[-1])
                delay_ms = max(1, int(1000 / fps)) if is_video else 1
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("r"), ord("R")):
                    toggle_requested = True
                frame_index += 1
    finally:
        recording.stop(elapsed + 1 / fps if is_video else time.monotonic() - started)
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
