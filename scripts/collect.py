import argparse
import csv
import json
import random
import time
from datetime import datetime

import cv2

from gesture.config import load_config, project_path
from gesture.collection import collection_phases, prepare_action_session, prepare_periodic_session
from gesture.display_text import draw_text


HAND_SHAPES = {"opened": ("张开手掌", "OPEN PALM"), "closed": ("握拳", "FIST"),
               "thumb_up": ("竖起拇指", "THUMB UP")}


def ask_count(message, default, minimum=1):
    while True:
        value = input(f"{message} (default {default}): ").strip()
        try:
            count = int(value) if value else default
            if count >= minimum:
                return count
        except ValueError:
            pass
        print(f"Enter an integer >= {minimum}.")


def ask_negative_task(tasks):
    for index, task in enumerate(tasks, 1):
        print(f"{index}. {task['gesture']}: {task['instruction']}")
    while True:
        value = input(f"Negative task [1-{len(tasks)}]: ").strip()
        if value.isdigit() and 1 <= int(value) <= len(tasks):
            return tasks[int(value) - 1]["gesture"]
        print(f"Choose 1-{len(tasks)}.")


def draw_pose_icon(frame, pose, config):
    if pose not in ("opened", "closed"):
        return
    height, width = frame.shape[:2]
    left, top = width - 170, height - 200
    cv2.rectangle(frame, (left, top), (width - 12, height - 55), (35, 35, 35), -1)
    color = (235, 235, 235)
    cx, cy = left + 88, top + 92
    cv2.rectangle(frame, (cx - 28, cy - 15), (cx + 28, cy + 30), color, 2)
    if pose == "opened":
        for dx, length in ((-24, 45), (-8, 58), (8, 52), (24, 35)):
            cv2.line(frame, (cx + dx, cy - 15), (cx + dx, cy - 15 - length), color, 7)
        cv2.line(frame, (cx - 28, cy + 8), (cx - 50, cy - 14), color, 7)
    else:
        cv2.rectangle(frame, (cx - 28, cy - 35), (cx + 28, cy - 15), color, 2)
        for dx in (-14, 0, 14):
            cv2.line(frame, (cx + dx, cy - 35), (cx + dx, cy - 15), color, 1)
        cv2.line(frame, (cx - 30, cy + 3), (cx + 5, cy - 17), color, 7)
    draw_text(frame, "张开 / OPEN" if pose == "opened" else "握拳 / FIST", (left + 12, top + 126),
              18, color, config["display"]["text_font_path"], max_width=134)


def show_phase(frame, phase, config, prompt_count, practice=False):
    shown = cv2.flip(frame, 1) if config["display"]["mirror_preview"] else frame.copy()
    height, width = shown.shape[:2]
    colors = config["display"]["prompt_colors"]
    font_path = config["display"]["text_font_path"]
    kind, gesture = phase["phase"], phase["gesture"]
    pose = gesture if kind in ("hold", "prepare") else phase.get("pose")
    if kind == "hold":
        name_zh, name_en = HAND_SHAPES[pose]
        label_zh, label = f"保持{name_zh}", f"HOLD {name_en}"
        color = colors["HOLD"]
    elif kind == "prepare":
        name_zh, name_en = HAND_SHAPES[pose]
        label_zh, label = f"准备：{name_zh}", f"PREPARE {name_en}"
        color = colors["PREPARE"]
    elif kind == "rest":
        label_zh, label, color = "休息", "REST", colors["REST"]
    elif kind == "negative":
        label_zh, label, color = phase["instruction_zh"], phase["instruction"], colors["NEGATIVE"]
    elif kind == "wave":
        label_zh, label = phase["instruction_zh"], phase["instruction"]
        color = colors.get("waving", colors["opened"])
    else:
        color = colors[gesture]
        if kind == "action":
            label_zh, label = phase["instruction_zh"], phase["instruction"]
            pose = "closed" if gesture == "close_hand" else "opened"
        else:
            name_zh, name_en = HAND_SHAPES[gesture]
            label_zh, label = f"{name_zh}并保持", f"MAKE AND HOLD: {name_en}"
    cv2.rectangle(shown, (0, 0), (width, 128), tuple(color), -1)
    draw_text(shown, label_zh, (20, 8), 28, (255, 255, 255), font_path, max_width=width - 40)
    draw_text(shown, label, (20, 44), 23, (255, 255, 255), font_path, max_width=width - 40)
    index = phase["prompt_index"]
    progress = f"{index + 1}/{prompt_count}" if index is not None else "准备 / ready"
    prefix = "练习 / PRACTICE" if practice else "录制 / REC"
    remaining = phase["end"] - phase["elapsed"]
    status = f"{prefix} | {remaining:.1f}s | {progress} | Q退出 / EXIT"
    draw_text(shown, status, (20, 80), 18, (255, 255, 255), font_path, max_width=width - 40)
    fraction = (phase["elapsed"] - phase["begin"]) / (phase["end"] - phase["begin"])
    cv2.rectangle(shown, (20, 108), (width - 20, 116), (55, 55, 55), -1)
    cv2.rectangle(shown, (20, 108), (20 + round(fraction * (width - 40)), 116), (235, 235, 235), -1)
    draw_pose_icon(shown, pose, config)
    return shown


def practice(cap, window, session, config):
    if session.get("kind") == "periodic":
        plan = dict(session, prompts=session["prompts"][:1])
    else:
        plan = prepare_action_session({"positive": session}, "positive", int(session["practice_pairs"]))
    phases = collection_phases(plan)
    started = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read camera preview")
        elapsed = time.monotonic() - started
        if elapsed >= phases[-1]["end"]:
            return True
        phase = next(item for item in phases if item["begin"] <= elapsed < item["end"])
        cv2.imshow(window, show_phase(frame, dict(phase, elapsed=elapsed), config, len(plan["prompts"]), True))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False


def wait_for_start(cap, window, session, config):
    selected, buttons = None, {}
    positive = session.get("sample_type") == "positive"
    negative = session.get("sample_type") == "negative"

    def mouse(event, x, y, _flags, _param):
        nonlocal selected
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, (left, top, right, bottom) in buttons.items():
                if left <= x <= right and top <= y <= bottom:
                    selected = action

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read camera preview")
        height, width = frame.shape[:2]
        shown = cv2.flip(frame, 1) if config["display"]["mirror_preview"] else frame.copy()
        cv2.rectangle(shown, (0, 0), (width, 140), (0, 0, 0), -1)
        periodic = session.get("kind") == "periodic"
        if periodic and positive:
            hand_zh = "左手" if session["hand"] == "left" else "右手"
            hand_en = "left" if session["hand"] == "left" else "right"
            motion_zh = "手腕" if session["motion"] == "wrist" else "肘部"
            motion_en = "wrist" if session["motion"] == "wrist" else "elbow"
            instruction = (f"掌心朝向摄像头，用{hand_zh}沿{motion_zh}左右挥动",
                           f"Face your palm to camera; wave with your {hand_en} {motion_en}.")
        elif periodic:
            instruction = ("张开手掌，按提示左右挥动", "Open your palm and wave as cued.")
        else:
            instruction = ("每次做一个动作，然后保持", "One motion per cue, then hold.")
        lines = [("整只手入镜，点击开始", "Keep the full hand in view; click START."),
                 instruction if positive else
                 (session["prompts"][0]["instruction_zh"], session["prompts"][0]["instruction"]) if negative else
                 ("做出手型并保持，休息时放松", "Hold the shape; relax during REST.")]
        font_path = config["display"]["text_font_path"]
        for row, (chinese, english) in enumerate(lines):
            draw_text(shown, chinese, (16, 8 + 54 * row), 22, (245, 245, 245), font_path, max_width=width - 32)
            draw_text(shown, english, (16, 36 + 54 * row), 18, (245, 245, 245), font_path, max_width=width - 32)
        hint = "练习不录制 / PRACTICE not recorded | Q退出 / EXIT" if positive else "Q退出 / EXIT"
        if negative:
            hint = (f"准备 / PREPARE {session['prepare_seconds']:g}s | "
                    f"采集 / RECORD {session['prompts'][0]['duration_seconds']:g}s | Q退出 / EXIT")
        elif periodic:
            hint = (f"{session['bouts']} 段 / bouts | WAVE {session['wave_seconds_range'][0]:g}–"
                    f"{session['wave_seconds_range'][1]:g}s | Q退出 / EXIT")
        draw_text(shown, hint, (16, 116), 16, (200, 200, 200), font_path, max_width=width - 32)
        buttons = {"start": (width // 2 + 10 if positive else width // 4, height - 90, width * 3 // 4, height - 30)}
        if positive:
            buttons["practice"] = (width // 4, height - 90, width // 2 - 10, height - 30)
            draw_pose_icon(shown, session.get("initial_pose", "opened"), config)
        for action, bounds in buttons.items():
            left, top, right, bottom = bounds
            cv2.rectangle(shown, (left, top), (right, bottom), (40, 140, 40) if action == "start" else (150, 100, 40), -1)
            label = "开始 / START" if action == "start" else "练习 / PRACTICE"
            draw_text(shown, label, (left + 10, top + 18), 24, (255, 255, 255), font_path, max_width=right - left - 20)
        cv2.imshow(window, shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        if selected == "start":
            return True
        if selected == "practice":
            selected = None
            if not practice(cap, window, session, config):
                return False


def main():
    parser = argparse.ArgumentParser(description="Collect gesture sessions")
    plans = tuple(name for name in ("state", "action", "periodic")
                  if project_path(f"session_plans/{name}.json").is_file())
    parser.add_argument("--plan", required=True, choices=plans, help="Collection plan")
    args = parser.parse_args()
    collector = input("Name: ").strip()
    collector_slug = "".join(char if char.isalnum() or char in "-_" else "_" for char in collector).strip("-_")
    if not collector_slug:
        raise ValueError("Name must contain a letter or digit")
    config = load_config()
    template = json.loads(project_path(f"session_plans/{args.plan}.json").read_text(encoding="utf-8"))
    if args.plan == "periodic" and template["negative"].get("reuse_action_tasks"):
        action_template = json.loads(project_path("session_plans/action.json").read_text(encoding="utf-8"))
        template["negative"]["tasks"] = action_template["negative"]["tasks"] + template["negative"]["tasks"]
        mode = input("Type [1 positive / 2 negative] (default 1): ").strip() or "1"
        if mode not in ("1", "2", "positive", "negative"):
            parser.error("Choose positive or negative")
        sample_type = "positive" if mode in ("1", "positive") else "negative"
        if sample_type == "positive":
            hands = ("left", "right")
            motions = ("wrist", "elbow")
            print("Hand: 1 left, 2 right; movement: 1 wrist, 2 elbow.")
            hand_choice = input("Hand [1/2] (default 2): ").strip() or "2"
            motion_choice = input("Movement [1/2] (default 1): ").strip() or "1"
            if hand_choice not in ("1", "2") or motion_choice not in ("1", "2"):
                parser.error("Choose hand and movement from 1 or 2")
            count = ask_count("Waving bouts", int(template["positive"]["bouts"]))
            session = prepare_periodic_session(
                template, sample_type, count, hands[int(hand_choice) - 1], motions[int(motion_choice) - 1])
            print("Keep an open palm facing the camera; wave continuously during each WAVE cue.")
        else:
            task = ask_negative_task(template["negative"]["tasks"])
            session = prepare_periodic_session(template, sample_type, negative_task=task)
            print("Perform only the selected background task.")
    elif args.plan == "action":
        while True:
            mode = input("Type [1 positive / 2 negative] (default 1): ").strip() or "1"
            if mode in ("1", "2", "positive", "negative"):
                break
        sample_type = "positive" if mode in ("1", "positive") else "negative"
        if sample_type == "positive":
            count = ask_count("Open-close pairs", int(template["positive"]["pairs"]))
            session = prepare_action_session(template, sample_type, count)
            print("One cue, one motion; hold until the next cue.")
        else:
            task = ask_negative_task(template["negative"]["tasks"])
            session = prepare_action_session(template, sample_type, negative_task=task)
            print("Prepare the pose, then repeat the selected task until recording ends.")
    else:
        session = template
        counts = session["prompt_counts"]
        for prompt in session["prompts"]:
            gesture = prompt["gesture"]
            counts[gesture] = ask_count(f"{gesture} repetitions (0 to skip)", int(counts[gesture]), 0)
        session["prompts"] = [dict(prompt) for prompt in session["prompts"] for _ in range(counts[prompt["gesture"]])]
        if session.get("shuffle"):
            random.shuffle(session["prompts"])
    if not session["prompts"]:
        parser.error("At least one prompt is required")
    phases = collection_phases(session)
    print(f"{len(session['prompts'])} prompts, {phases[-1]['end']:.1f}s. Click START.")
    cap = cv2.VideoCapture(int(config["data"]["camera_index"]))
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    window = f"{args.plan.title()} data collection"
    writer, rows = None, []
    prompts = [dict(prompt, start_frame=None, end_frame=None, hold_end_frame=None)
               for prompt in session["prompts"]]
    recorded_phases = []
    try:
        if not wait_for_start(cap, window, session, config):
            return
        session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{collector_slug}"
        folder = project_path(config["data"]["sessions_dir"]) / session_id
        folder.mkdir(parents=True)
        session.update(session_id=session_id, collector=collector)
        print(f"Recording: {session_id}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), float(config["data"]["fps_target"]), (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create video.mp4")
        started = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            elapsed = time.monotonic() - started
            if elapsed >= phases[-1]["end"]:
                break
            phase = next(item for item in phases if item["begin"] <= elapsed < item["end"])
            frame_index = len(rows)
            if phase["phase"] in ("state", "action", "negative", "wave"):
                prompt = prompts[phase["prompt_index"]]
                if prompt["start_frame"] is None:
                    prompt["start_frame"] = frame_index
                prompt["end_frame"] = frame_index
            if phase["phase"] == "hold" and phase["prompt_index"] is not None:
                prompts[phase["prompt_index"]]["hold_end_frame"] = frame_index
            if (not recorded_phases or recorded_phases[-1]["phase"] != phase["phase"]
                    or recorded_phases[-1]["prompt_index"] != phase["prompt_index"]):
                recorded_phases.append({
                    "phase": phase["phase"], "prompt_index": phase["prompt_index"],
                    "start_frame": frame_index, "end_frame": frame_index,
                    "start_seconds": elapsed, "end_seconds": elapsed,
                })
            else:
                recorded_phases[-1]["end_frame"] = frame_index
                recorded_phases[-1]["end_seconds"] = elapsed
            cv2.imshow(window, show_phase(frame, dict(phase, elapsed=elapsed), config, len(prompts)))
            writer.write(frame)
            rows.append((frame_index, elapsed))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    with (folder / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        output = csv.writer(stream)
        output.writerow(["frame_index", "elapsed_seconds"])
        output.writerows(rows)
    session["prompts"] = prompts
    session["phases"] = recorded_phases
    (folder / "session.json").write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    annotations = {"session_id": session_id, f"{args.plan}_intervals": []}
    if args.plan == "action":
        annotations.update(action_reviewed=False, action_review_mode="prompts", action_candidates=[],
                           action_ignored_intervals=[], action_background_prompts=[])
    elif args.plan == "periodic":
        annotations.update(
            action_review_mode="prompts", action_reviewed=False, action_candidates=[],
            action_ignored_intervals=[], action_background_prompts=[],
            direction_candidates=[], direction_reviewed_intervals=[],
            direction_ignored_intervals=[], one_shot_reviewed_intervals=[],
            waving_intervals=[],
        )
    (folder / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} frames: {folder}")
    extractor = "extract_landmarks" if args.plan == "state" else "extract_action"
    print(f"Next: python -m scripts.{extractor} --session-id {session_id}")
    annotator = "annotate_state" if args.plan == "state" else "annotate_action"
    print(f"Then: python -m scripts.{annotator} --session-id {session_id}")


if __name__ == "__main__":
    main()
