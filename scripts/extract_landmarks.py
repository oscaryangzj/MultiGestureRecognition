import argparse
import csv

import cv2
import mediapipe as mp

from gesture.config import load_config, project_path
from gesture.data import load_frame_times
from gesture.hand_landmarker import create_hand_landmarker, detect_video_frame


def main():
    parser = argparse.ArgumentParser(description="Extract one row of hand landmarks per video frame")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    config = load_config()
    session_dir = project_path(config["data"]["sessions_dir"]) / args.session_id
    video_path = session_dir / "video.mp4"
    frame_times_path = session_dir / "frames.csv"
    if not video_path.is_file() or not frame_times_path.is_file():
        raise FileNotFoundError("Session requires video.mp4 and frames.csv")

    frame_times = load_frame_times(frame_times_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    columns = ["frame_index", "elapsed_seconds", "hand_detected"]
    for index in range(21):
        columns.extend([f"x{index}", f"y{index}"])

    output_path = session_dir / "landmarks.csv"
    temp_path = session_dir / "landmarks.csv.tmp"
    rows_written = 0
    try:
        with create_hand_landmarker(config, mp.tasks.vision.RunningMode.VIDEO) as landmarker:
            with temp_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                frame_index = 0
                previous_timestamp = -1
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index >= len(frame_times):
                        raise ValueError("video.mp4 has more frames than frames.csv")
                    elapsed = frame_times[frame_index]
                    timestamp_ms = max(int(round(elapsed * 1000)), previous_timestamp + 1)
                    previous_timestamp = timestamp_ms
                    points = detect_video_frame(landmarker, frame, timestamp_ms)
                    row = {
                        "frame_index": frame_index,
                        "elapsed_seconds": f"{elapsed:.6f}",
                        "hand_detected": 1 if points is not None else 0,
                    }
                    for index in range(21):
                        row[f"x{index}"] = f"{points[index, 0]:.8f}" if points is not None else ""
                        row[f"y{index}"] = f"{points[index, 1]:.8f}" if points is not None else ""
                    writer.writerow(row)
                    frame_index += 1
                    rows_written += 1
        if rows_written != len(frame_times):
            raise ValueError(
                f"video.mp4 has {rows_written} frames but frames.csv has {len(frame_times)} rows"
            )
        temp_path.replace(output_path)
    finally:
        cap.release()
        temp_path.unlink(missing_ok=True)
    print(f"Saved {rows_written} landmark rows to {output_path}")


if __name__ == "__main__":
    main()
