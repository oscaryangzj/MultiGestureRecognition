import argparse
import csv
import json
import time

import cv2

from gesture.config import load_config, project_path


def main():
    parser = argparse.ArgumentParser(description="Collect a prompted State session")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    config = load_config()
    session_dir = project_path(config["data"]["sessions_dir"]) / args.session_id
    session_path = session_dir / "session.json"
    if not session_path.is_file():
        raise FileNotFoundError(f"Create the session prompt plan first: {session_path}")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if session.get("target_model") != "state":
        raise ValueError("Stage 1 collection requires target_model: state")
    if session.get("kind") != "static":
        raise ValueError("Stage 1 collection requires kind: static")
    existing = [
        name
        for name in ("video.mp4", "frames.csv", "landmarks.csv", "annotations.json")
        if (session_dir / name).exists()
    ]
    if existing:
        raise FileExistsError(
            f"Session {args.session_id} already has data ({', '.join(existing)}); "
            "use a new session ID to record again"
        )

    prompts = session.get("prompts", [])
    durations = session.get("durations_seconds", {})
    classes = set(config["labels"]["state_classes"]) - {"NULL"}
    if not prompts:
        raise ValueError("session.json must have at least one prompt")
    for prompt in prompts:
        gesture = prompt["gesture"]
        if gesture not in classes:
            raise ValueError(f"Unknown State prompt: {gesture}")
        if float(durations[gesture]) <= 0:
            raise ValueError(f"Duration must be positive for {gesture}")

    gap_seconds = float(session["gap_seconds"])
    schedule = []
    cursor = 0.0
    for prompt in prompts:
        duration = float(durations[prompt["gesture"]])
        schedule.append((cursor, cursor + duration))
        cursor += duration + gap_seconds

    session_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(int(config["data"]["camera_index"]))
    if not cap.isOpened():
        raise RuntimeError("Could not open the configured camera")

    fps = float(config["data"]["fps_target"])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(session_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create video.mp4")

    rows = []
    prompt_ranges = [{"gesture": p["gesture"], "start_frame": None, "end_frame": None} for p in prompts]
    frame_index = 0
    start_time = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            elapsed = time.monotonic() - start_time
            if elapsed >= schedule[-1][1]:
                break

            active_index = next(
                (i for i, (begin, end) in enumerate(schedule) if begin <= elapsed < end),
                None,
            )
            shown = frame.copy()
            if active_index is None:
                next_item = next((i for i, (begin, _) in enumerate(schedule) if elapsed < begin), None)
                if next_item is None:
                    break
                label = "REST"
                remaining = schedule[next_item][0] - elapsed
            else:
                prompt = prompts[active_index]["gesture"]
                label = f"MAKE AND HOLD: {prompt}"
                remaining = schedule[active_index][1] - elapsed
                if prompt_ranges[active_index]["start_frame"] is None:
                    prompt_ranges[active_index]["start_frame"] = frame_index
                prompt_ranges[active_index]["end_frame"] = frame_index

            cv2.rectangle(shown, (0, 0), (width, 74), (0, 0, 0), -1)
            cv2.putText(shown, label, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(shown, f"{remaining:.1f}s   q: stop", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
            cv2.imshow("State data collection", shown)

            writer.write(frame)
            rows.append((frame_index, elapsed))
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    with (session_dir / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        output = csv.writer(stream)
        output.writerow(["frame_index", "elapsed_seconds"])
        output.writerows(rows)
    session["session_id"] = args.session_id
    session["prompts"] = prompt_ranges
    session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    annotation_path = session_dir / "annotations.json"
    if not annotation_path.exists():
        annotation_path.write_text(
            json.dumps({"session_id": args.session_id, "state_intervals": []}, indent=2),
            encoding="utf-8",
        )
    print(f"Saved {frame_index} frames to {session_dir}")


if __name__ == "__main__":
    main()
