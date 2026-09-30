import argparse
import json

import cv2

from gesture.config import load_config, project_path
from gesture.data import load_landmarks_csv
from gesture.hand_landmarker import draw_hand_landmarks
from gesture.labels import IGNORE_TARGET, make_state_targets


def main():
    parser = argparse.ArgumentParser(description="Manually mark State positive intervals")
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
    key_labels = {ord(str(index)): label for index, label in enumerate(
        (name for name in classes if name != "NULL"), start=1
    )}
    null_index = classes.index("NULL")
    visible_frames = int(config["labels"]["state_null_visible_frames"])
    cv2.namedWindow("State annotation", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", "State annotation", 0, frame_count - 1, lambda _: None)
    frame_index = 0
    start_frame = None
    end_frame = None
    changed = False
    targets = make_state_targets(hand_detected, intervals, classes, visible_frames)

    label_keys = " | ".join(f"{chr(key)} {label}" for key, label in key_labels.items())
    print(f"Controls: [ start | ] end | {label_keys} | x remove current interval | s save | q quit")
    displayed_frame = None
    redraw = True
    try:
        while True:
            selected = cv2.getTrackbarPos("frame", "State annotation")
            frame_index = max(0, min(frame_count - 1, selected))
            if redraw or frame_index != displayed_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    break

                points = points_by_frame[frame_index] if hand_detected[frame_index] else None
                shown = draw_hand_landmarks(frame, points)
                label = next(
                    (
                        item["label"]
                        for item in intervals
                        if int(item["start_frame"]) <= frame_index <= int(item["end_frame"])
                    ),
                    None,
                )
                if not hand_detected[frame_index]:
                    status = "NO HAND: ignored"
                elif label:
                    status = f"MANUAL POSITIVE: {label}"
                elif targets[frame_index] == null_index:
                    status = "NULL (automatic)"
                else:
                    status = "UNLABELED: waiting for continuous hand frames"

                cv2.rectangle(shown, (0, 0), (shown.shape[1], 104), (0, 0, 0), -1)
                cv2.putText(shown, f"Frame {frame_index}/{frame_count - 1}  t={elapsed[frame_index]:.2f}s", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1)
                cv2.putText(shown, status, (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (70, 240, 70) if hand_detected[frame_index] else (70, 170, 255), 2)
                cv2.putText(shown, f"range start={start_frame} end={end_frame} | [ ] then labels | x delete | s save | q quit", (16, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
                cv2.imshow("State annotation", shown)
                displayed_frame = frame_index
                redraw = False

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                frame_index = max(0, frame_index - 1)
                cv2.setTrackbarPos("frame", "State annotation", frame_index)
            elif key == ord("d"):
                frame_index = min(frame_count - 1, frame_index + 1)
                cv2.setTrackbarPos("frame", "State annotation", frame_index)
            elif key == ord("["):
                start_frame = frame_index
                redraw = True
            elif key == ord("]"):
                end_frame = frame_index
                redraw = True
            elif key in key_labels:
                if start_frame is None or end_frame is None:
                    print("Set both interval endpoints first with [ and ]")
                    continue
                start, end = sorted((start_frame, end_frame))
                candidate = {"start_frame": start, "end_frame": end, "label": key_labels[key]}
                overlaps = any(
                    start <= int(item["end_frame"]) and end >= int(item["start_frame"])
                    for item in intervals
                )
                if overlaps:
                    print("This interval overlaps an existing positive interval")
                    continue
                intervals.append(candidate)
                intervals.sort(key=lambda item: int(item["start_frame"]))
                targets = make_state_targets(hand_detected, intervals, classes, visible_frames)
                start_frame = end_frame = None
                changed = True
                redraw = True
            elif key == ord("x"):
                old_count = len(intervals)
                intervals[:] = [
                    item
                    for item in intervals
                    if not int(item["start_frame"]) <= frame_index <= int(item["end_frame"])
                ]
                if len(intervals) != old_count:
                    targets = make_state_targets(hand_detected, intervals, classes, visible_frames)
                    changed = True
                    redraw = True
            elif key == ord("s"):
                annotations["session_id"] = args.session_id
                annotation_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
                changed = False
                print(f"Saved {len(intervals)} State intervals")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if changed:
        annotations["session_id"] = args.session_id
        annotation_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
        print(f"Saved {len(intervals)} State intervals to {annotation_path}")


if __name__ == "__main__":
    main()
