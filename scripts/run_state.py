import argparse
import time

import cv2
import numpy as np
import mediapipe as mp
import torch

from gesture.config import load_config, project_path
from gesture.data import load_frame_times
from gesture.features import normalize_landmarks
from gesture.hand_landmarker import create_hand_landmarker, detect_video_frame, draw_hand_landmarks
from gesture.models import StateMLP


def load_model(config):
    checkpoint_path = project_path("runs/state/model.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Train the State model first with python -m scripts.train_state")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = StateMLP(
        input_size=checkpoint["input_size"],
        hidden_size=int(checkpoint["hidden_size"]),
        num_classes=len(checkpoint["classes"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["classes"]


def main():
    parser = argparse.ArgumentParser(description="Run State classification on a camera or video")
    parser.add_argument("--session-id", help="Replay a recorded session video")
    args = parser.parse_args()
    config = load_config()
    model, classes = load_model(config)

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

    started = time.monotonic()
    frame_index = 0
    previous_timestamp = -1
    try:
        with create_hand_landmarker(config, mp.tasks.vision.RunningMode.VIDEO) as landmarker:
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

                points = detect_video_frame(landmarker, frame, timestamp_ms)
                shown = draw_hand_landmarks(frame, points)
                if points is None:
                    cv2.putText(shown, "NO HAND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 255), 2)
                else:
                    features = normalize_landmarks(
                        points,
                        config["features"]["min_hand_scale"],
                        config["features"]["local_scale"],
                    )
                    if features is None:
                        cv2.putText(shown, "INVALID HAND SCALE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
                    else:
                        with torch.no_grad():
                            probabilities = torch.softmax(model(torch.from_numpy(features).unsqueeze(0)), dim=1)[0].numpy()
                        best_index = int(np.argmax(probabilities))
                        cv2.putText(
                            shown,
                            f"{classes[best_index]}  {probabilities[best_index]:.2f}",
                            (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (60, 240, 60),
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
                cv2.imshow("Static State recognition", shown)
                delay_ms = max(1, int(1000 / fps)) if is_video else 1
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break
                frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
