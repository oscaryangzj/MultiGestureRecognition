import argparse
import csv

import cv2

from gesture.config import load_config, project_path
from gesture.data import load_frame_times, load_session
from gesture.state_recognizer import create_state_recognizer, recognize_video_frame


def extract_action_session(config, session_dir):
    if load_session(session_dir).get("target_model") != "action":
        raise ValueError("Action extraction requires target_model: action")
    times = load_frame_times(session_dir / "frames.csv")
    classes = config["labels"]["state_classes"]
    columns = ["frame_index", "elapsed_seconds", "hand_detected"]
    world_columns = ["frame_index", "elapsed_seconds", "hand_detected"]
    for index in range(21):
        columns.extend([f"x{index}", f"y{index}"])
        world_columns.extend([f"x{index}", f"y{index}", f"z{index}"])
    capture = cv2.VideoCapture(str(session_dir / "video.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {session_dir / 'video.mp4'}")
    landmark_path = session_dir / "landmarks.csv"
    world_path = session_dir / "world_landmarks.csv"
    scores_path = session_dir / "state_scores.csv"
    landmark_temp = landmark_path.with_suffix(".csv.tmp")
    world_temp = world_path.with_suffix(".csv.tmp")
    scores_temp = scores_path.with_suffix(".csv.tmp")
    count = 0
    try:
        with create_state_recognizer(config) as recognizers, landmark_temp.open("w", newline="") as landmarks, world_temp.open("w", newline="") as world_landmarks, scores_temp.open("w", newline="") as scores:
            landmark_writer = csv.DictWriter(landmarks, fieldnames=columns)
            world_writer = csv.DictWriter(world_landmarks, fieldnames=world_columns)
            score_writer = csv.DictWriter(scores, fieldnames=["frame_index", *classes])
            landmark_writer.writeheader()
            world_writer.writeheader()
            score_writer.writeheader()
            previous_timestamp = -1
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if count >= len(times):
                    raise ValueError("video.mp4 has more frames than frames.csv")
                timestamp = max(round(times[count] * 1000), previous_timestamp + 1)
                previous_timestamp = timestamp
                points, probabilities, world_points = recognize_video_frame(
                    recognizers, frame, timestamp, include_world_landmarks=True)
                row = {"frame_index": count, "elapsed_seconds": times[count], "hand_detected": int(points is not None)}
                world_row = {"frame_index": count, "elapsed_seconds": times[count],
                             "hand_detected": int(world_points is not None)}
                for index in range(21):
                    row[f"x{index}"] = float(points[index, 0]) if points is not None else ""
                    row[f"y{index}"] = float(points[index, 1]) if points is not None else ""
                    for axis, axis_name in enumerate(("x", "y", "z")):
                        world_row[f"{axis_name}{index}"] = float(world_points[index, axis]) if world_points is not None else ""
                landmark_writer.writerow(row)
                world_writer.writerow(world_row)
                score_writer.writerow({"frame_index": count, **{
                    label: float(probabilities[index]) if points is not None else ""
                    for index, label in enumerate(classes)
                }})
                count += 1
        if count != len(times):
            raise ValueError("video.mp4 and frames.csv must have the same frame count")
        landmark_temp.replace(landmark_path)
        world_temp.replace(world_path)
        scores_temp.replace(scores_path)
    finally:
        capture.release()
        landmark_temp.unlink(missing_ok=True)
        world_temp.unlink(missing_ok=True)
        scores_temp.unlink(missing_ok=True)
    print(f"Saved {count} rows: {landmark_path}, {world_path}, {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract Action landmarks and official State scores")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    config = load_config()
    extract_action_session(config, project_path(config["data"]["sessions_dir"]) / args.session_id)


if __name__ == "__main__":
    main()
