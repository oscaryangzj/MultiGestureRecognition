import json
from pathlib import Path
import random
import copy
import csv
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from gesture.collection import collection_phases, prepare_action_session
from gesture.config import load_config
from scripts import collect


class ActionCollectionPlanTest(unittest.TestCase):
    def setUp(self):
        self.template = json.loads((Path(__file__).resolve().parents[1] / "session_plans/action.json").read_text())

    def test_actual_random_hold_values_survive_json_and_define_capture_phases(self):
        session = prepare_action_session(self.template, "positive", 3, random.Random(123))
        saved = json.loads(json.dumps(session))
        self.assertEqual(saved["sample_type"], "positive")
        self.assertEqual([prompt["gesture"] for prompt in saved["prompts"]], ["close_hand", "open_hand"] * 3)
        holds = [prompt["hold_seconds"] for prompt in saved["prompts"]]
        low, high = self.template["positive"]["hold_seconds_range"]
        self.assertTrue(all(low <= hold <= high for hold in holds))
        self.assertGreater(len(set(holds)), 1)
        phases = collection_phases(saved)
        self.assertEqual(phases[0]["gesture"], saved["initial_pose"])
        for index, prompt in enumerate(saved["prompts"]):
            action, hold = phases[index * 2 + 1:index * 2 + 3]
            self.assertAlmostEqual(action["end"] - action["begin"], prompt["duration_seconds"])
            self.assertAlmostEqual(hold["end"] - hold["begin"], prompt["hold_seconds"])
            self.assertEqual(hold["gesture"], "closed" if prompt["gesture"] == "close_hand" else "opened")
        expected = saved["prepare_seconds"] + sum(prompt["duration_seconds"] + prompt["hold_seconds"] for prompt in saved["prompts"])
        self.assertAlmostEqual(phases[-1]["end"], expected)

    def test_negative_preparation_is_separate_from_each_recorded_background_task(self):
        for selected in self.template["negative"]["tasks"]:
            session = prepare_action_session(self.template, "negative", negative_task=selected["gesture"])
            self.assertEqual(session["sample_type"], "negative")
            self.assertEqual(session["selected_task"], selected["gesture"])
            self.assertEqual(len(session["prompts"]), 1)
            self.assertEqual(len(session["tasks"]), 1)
            prompt = session["prompts"][0]
            self.assertEqual(prompt["gesture"], selected["gesture"])
            self.assertEqual(prompt["duration_seconds"], 30)
            phases = collection_phases(session)
            self.assertEqual(len(phases), 2)
            prepare, task = phases
            self.assertEqual(prepare["phase"], "prepare")
            self.assertEqual(prepare["gesture"], prompt["prepare_pose"])
            self.assertAlmostEqual(prepare["end"] - prepare["begin"], session["prepare_seconds"])
            self.assertAlmostEqual(task["end"] - task["begin"], prompt["duration_seconds"])
            self.assertEqual(task["prompt_index"], 0)
            self.assertEqual(task["gesture"], prompt["gesture"])
            self.assertTrue(prompt["instruction"])


class FakeCamera:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def get(self, property_id):
        return {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480}[property_id]

    def release(self):
        self.released = True


class ActionCollectionFlowTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.template = json.loads((Path(__file__).resolve().parents[1] / "session_plans/action.json").read_text())
        self.camera = FakeCamera()

    def test_practice_is_two_pairs_even_when_recording_one_pair(self):
        session = prepare_action_session(self.template, "positive", 1, random.Random(123))
        seen = []
        tick = iter(np.arange(0, 100, .5))

        def render(frame, phase, _config, prompt_count, practice=False):
            seen.append((phase["phase"], phase["prompt_index"], prompt_count, practice))
            return frame

        with patch.object(collect, "show_phase", side_effect=render), patch.object(cv2, "imshow"), \
                patch.object(cv2, "waitKey", return_value=-1), patch.object(cv2, "VideoWriter") as writer, \
                patch.object(collect.time, "monotonic", side_effect=lambda: float(next(tick))):
            self.assertTrue(collect.practice(self.camera, "TEST", session, self.config))
        self.assertEqual({item[1] for item in seen if item[0] == "action"}, {0, 1, 2, 3})
        self.assertTrue(all(item[2:] == (4, True) for item in seen))
        self.assertEqual(len(session["prompts"]), 2)
        writer.assert_not_called()

    def test_preview_practice_returns_to_ready_and_start_button_is_clickable(self):
        session = prepare_action_session(self.template, "positive", 1, random.Random(123))
        callback, calls = None, 0

        def register(_window, mouse):
            nonlocal callback
            callback = mouse

        def key(_delay):
            nonlocal calls
            calls += 1
            callback(cv2.EVENT_LBUTTONDOWN, 240 if calls == 1 else 410, 420, 0, None)
            return -1

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "setMouseCallback", side_effect=register), \
                patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", side_effect=key), \
                patch.object(collect, "practice", return_value=True) as practice:
            self.assertTrue(collect.wait_for_start(self.camera, "TEST", session, self.config))
        self.assertEqual(calls, 2)
        practice.assert_called_once()

    def test_practice_then_cancel_preview_creates_no_session_or_writer(self):
        with tempfile.TemporaryDirectory(prefix="mgr-preview-test-") as root:
            self.config["data"]["sessions_dir"] = str(Path(root) / "sessions")
            callback = None
            keys = 0

            def register(_window, mouse):
                nonlocal callback
                callback = mouse

            def key(_delay):
                nonlocal keys
                keys += 1
                if keys == 1:
                    callback(cv2.EVENT_LBUTTONDOWN, 240, 420, 0, None)
                    return -1
                return ord("q")

            with patch("sys.argv", ["collect", "--plan", "action"]), patch("builtins.input", side_effect=["TEST", "1", "1"]), \
                    patch.object(collect, "load_config", return_value=self.config), \
                    patch.object(cv2, "VideoCapture", return_value=self.camera), \
                    patch.object(cv2, "VideoWriter") as writer, \
                    patch.object(cv2, "namedWindow"), patch.object(cv2, "setMouseCallback", side_effect=register), \
                    patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", side_effect=key), \
                    patch.object(cv2, "destroyAllWindows"), patch.object(collect, "practice", return_value=True) as practice:
                collect.main()
            practice.assert_called_once()
            writer.assert_not_called()
            self.assertFalse(Path(self.config["data"]["sessions_dir"]).exists())
            self.assertTrue(self.camera.released)

    def test_start_saves_raw_video_actual_timings_and_only_task_frame_ranges(self):
        for sample_type, mode in (("positive", "1"), ("negative", "2")):
            with self.subTest(sample_type=sample_type), tempfile.TemporaryDirectory(prefix="mgr-capture-test-") as root:
                config = copy.deepcopy(self.config)
                config["data"]["sessions_dir"] = str(Path(root) / "sessions")
                camera = FakeCamera()
                tick = iter(np.arange(0, 200, .5))

                def start(_cap, _window, _session, _config):
                    self.assertFalse(Path(config["data"]["sessions_dir"]).exists())
                    return True

                with patch("sys.argv", ["collect", "--plan", "action"]), patch("builtins.input", side_effect=["TEST", mode, "1" if sample_type == "positive" else "5"]), \
                        patch.object(collect, "load_config", return_value=config), \
                        patch.object(cv2, "VideoCapture", return_value=camera), \
                        patch.object(collect, "wait_for_start", side_effect=start), \
                        patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", return_value=-1), \
                        patch.object(cv2, "destroyAllWindows"), \
                        patch.object(collect.time, "monotonic", side_effect=lambda: float(next(tick))):
                    collect.main()
                folders = list(Path(config["data"]["sessions_dir"]).iterdir())
                self.assertEqual(len(folders), 1)
                folder = folders[0]
                session = json.loads((folder / "session.json").read_text())
                annotations = json.loads((folder / "annotations.json").read_text())
                self.assertEqual(session["sample_type"], sample_type)
                if sample_type == "negative":
                    self.assertEqual(session["selected_task"], "wrist_pitch_opened")
                    self.assertEqual(len(session["prompts"]), 1)
                    self.assertEqual(session["prompts"][0]["duration_seconds"], 30)
                self.assertEqual(annotations["action_background_prompts"], [])
                self.assertFalse(annotations["action_reviewed"])
                phases = collection_phases(session)
                with (folder / "frames.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                for index, prompt in enumerate(session["prompts"]):
                    matching = next(phase for phase in phases if phase["prompt_index"] == index and phase["phase"] in ("action", "negative"))
                    selected = [int(row["frame_index"]) for row in rows if matching["begin"] <= float(row["elapsed_seconds"]) < matching["end"]]
                    self.assertEqual((prompt["start_frame"], prompt["end_frame"]), (selected[0], selected[-1]))
                captured = cv2.VideoCapture(str(folder / "video.mp4"))
                try:
                    self.assertEqual(int(captured.get(cv2.CAP_PROP_FRAME_COUNT)), len(rows))
                finally:
                    captured.release()
                self.assertTrue(camera.released)


if __name__ == "__main__":
    unittest.main()
