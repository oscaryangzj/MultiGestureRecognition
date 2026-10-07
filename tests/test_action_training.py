import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from gesture.action_data import ActionWindows, load_action_session, load_action_split
from gesture.action_evaluation import action_event_metrics, aggregate_action_metrics
from gesture.action_features import ACTION_FEATURE_SIZE
from gesture.action_training import action_data_summary, action_normalization, checkpoint_rank
from gesture.config import load_config


class ActionTrainingTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())

    def write_session(self, root, session_id, sample_type="negative", intervals=None, reviewed=True):
        folder = root / session_id
        folder.mkdir()
        prompt = {"gesture": "hold_opened", "sample_type": "negative",
                  "instruction": "Keep the hand open", "start_frame": 10, "end_frame": 99}
        metadata = {"target_model": "action", "sample_type": sample_type,
                    "prompts": [prompt] if sample_type == "negative" else []}
        (folder / "session.json").write_text(json.dumps(metadata))
        annotations = {"action_intervals": intervals or [], "action_reviewed": reviewed,
                       "action_review_mode": "prompts" if sample_type == "negative" else "manual",
                       "action_candidates": [], "action_ignored_intervals": [],
                       "action_background_prompts": [0] if reviewed else []}
        (folder / "annotations.json").write_text(json.dumps(annotations))
        times = np.arange(120) / 10
        video = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 54))
        for _ in times:
            video.write(np.zeros((54, 96, 3), np.uint8))
        video.release()
        for filename, columns, rows in (
            ("frames.csv", ["frame_index", "elapsed_seconds"],
             [[frame, time] for frame, time in enumerate(times)]),
            ("state_scores.csv", ["frame_index", "opened", "closed"],
             [[frame, 0.9, 0.1] for frame in range(len(times))]),
            ("landmarks.csv", ["frame_index", "elapsed_seconds", "hand_detected"] +
             [axis + str(point) for point in range(21) for axis in ("x", "y")],
             [[frame, time, 1] + [value for point in range(21)
                                  for value in (0.3 + point * 0.01, 0.4 + point * 0.01)]
              for frame, time in enumerate(times)]),
        ):
            with (folder / filename).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(rows)

    def test_reviewed_negative_session_has_zero_targets_and_supervised_windows(self):
        with tempfile.TemporaryDirectory(prefix="mgr-negative-test-") as temporary:
            root = Path(temporary)
            self.write_session(root, "negative")
            self.config["data"]["sessions_dir"] = str(root)
            session = load_action_session(self.config, "negative")
            self.assertEqual(session["sample_type"], "negative")
            self.assertFalse(session["targets"].any())
            self.assertTrue(session["valid"][10:100].all())
            self.assertFalse(session["valid"][:10].any())
            self.assertFalse(session["valid"][100:].any())
            self.assertFalse(session["mask"][10:12].any())
            self.assertTrue(session["mask"][12:100].all())
            self.assertFalse(session["mask"][:10].any())
            self.assertFalse(session["mask"][100:].any())
            data = ActionWindows([session], np.zeros(ACTION_FEATURE_SIZE, np.float32), np.ones(ACTION_FEATURE_SIZE, np.float32), self.config)
            self.assertGreater(len(data), 0)
            for _features, target, mask in data:
                self.assertFalse(target.any())
                self.assertTrue(mask.any())

            continuous = load_action_session(self.config, "negative", isolate_negative=False)
            self.assertTrue(continuous["valid"].all())

    def test_pending_negative_task_remains_available_for_annotation(self):
        with tempfile.TemporaryDirectory(prefix="mgr-pending-negative-") as temporary:
            root = Path(temporary)
            self.write_session(root, "negative", reviewed=False)
            self.config["data"]["sessions_dir"] = str(root)
            session = load_action_session(self.config, "negative", resample=True)
            self.assertTrue(session["valid"].any())
            self.assertFalse(session["mask"].any())

    def test_negative_session_can_contain_manually_marked_real_event(self):
        with tempfile.TemporaryDirectory(prefix="mgr-negative-event-test-") as temporary:
            root = Path(temporary)
            self.write_session(root, "negative", intervals=[{"start_frame": 30, "end_frame": 40, "label": "open_hand"}])
            self.config["data"]["sessions_dir"] = str(root)
            session = load_action_session(self.config, "negative")
            self.assertTrue((session["targets"][:, 0] * session["mask"][:, 0]).any())
            self.assertFalse(session["targets"][:, 1].any())
            summary = action_data_summary([session], self.config)
            self.assertEqual(summary["sessions"], {"positive": 0, "negative": 1})
            self.assertEqual(summary["events"], {"open_hand": 1, "close_hand": 0})

    def test_unreviewed_negative_session_is_rejected_for_training(self):
        with tempfile.TemporaryDirectory(prefix="mgr-unreviewed-test-") as temporary:
            root = Path(temporary)
            self.write_session(root, "negative", reviewed=False)
            split_path = root / "splits.json"
            split_path.write_text(json.dumps({"train": ["negative"], "val": []}))
            self.config["data"].update(sessions_dir=str(root), splits_file=str(split_path))
            with self.assertRaisesRegex(ValueError, "review or cancel"):
                load_action_split(self.config, "train")

    def test_normalization_excludes_ignored_and_missing_frames(self):
        features = np.tile(np.asarray([1, 3, 1000, 2000], np.float32)[:, None], (1, ACTION_FEATURE_SIZE))
        session = {"features": features,
                   "valid": np.asarray([True, True, True, False]),
                   "mask": np.asarray([[True, True], [True, True], [False, False], [True, True]])}
        mean, std = action_normalization([session], self.config)
        np.testing.assert_allclose(mean[:42], 2)
        np.testing.assert_allclose(std[:42], 1)
        np.testing.assert_allclose(mean[[42, 44, 46]], 0)
        np.testing.assert_allclose(std[[42, 44, 46]], np.sqrt(5))
        self.config["training"]["action_mirror_probability"] = 0
        mean, std = action_normalization([session], self.config)
        np.testing.assert_allclose(mean, 2)
        np.testing.assert_allclose(std, 1)

    def test_mirror_augmentation_keeps_targets_and_validation_unchanged(self):
        features = np.tile(np.arange(ACTION_FEATURE_SIZE, dtype=np.float32), (30, 1))
        session = {"features": features.copy(), "valid": np.ones(30, bool),
                   "segments": np.zeros(30, int), "times": np.arange(30) / 10,
                   "mask": np.ones((30, 2), bool), "targets": np.ones((30, 2), np.float32),
                   "intervals": []}
        self.config["training"]["action_mirror_probability"] = 1.0
        mean, std = np.full(ACTION_FEATURE_SIZE, 2, np.float32), np.full(ACTION_FEATURE_SIZE, 3, np.float32)
        train = ActionWindows([session], mean, std, self.config, augment=True)
        val = ActionWindows([session], mean, std, self.config)
        expected = features.copy()
        expected[:, [42, 44, 46]] *= -1
        train_features, train_targets, train_mask = train[0]
        val_features, val_targets, val_mask = val[0]
        np.testing.assert_allclose(train_features, (expected - mean) / std)
        np.testing.assert_allclose(val_features, (features - mean) / std)
        np.testing.assert_array_equal(session["features"], features)
        np.testing.assert_array_equal(train_targets, val_targets)
        np.testing.assert_array_equal(train_mask, val_mask)

    def test_aggregate_metrics_include_negative_false_alarms_and_missing_time_is_ignored(self):
        positive = {"times": np.arange(61), "intervals": [{"start_frame": 5, "end_frame": 10, "label": "open_hand"}]}
        positive_scores = np.zeros((61, 2), np.float32)
        positive_scores[10, 0] = 0.9
        negative = {"times": np.arange(61), "intervals": [], "valid": np.ones(61, bool)}
        negative["valid"][30:] = False
        negative_scores = np.zeros((61, 2), np.float32)
        negative_scores[[15, 45], 0] = 0.9
        evaluations = {"positive": action_event_metrics(positive, positive_scores, self.config),
                       "negative": action_event_metrics(negative, negative_scores, self.config)}
        self.assertEqual(evaluations["negative"]["duration_seconds"], 29)
        summary = aggregate_action_metrics(evaluations)
        self.assertEqual(summary["overall"]["true_positive"], 1)
        self.assertEqual(summary["overall"]["false_positive"], 1)
        self.assertAlmostEqual(summary["per_class"]["open_hand"]["f1"], 2 / 3)
        self.assertAlmostEqual(summary["overall"]["false_positives_per_minute"], 60 / 89)

    def test_checkpoint_selection_prioritizes_events_over_lower_bce(self):
        baseline = {"macro_f1": 0.5, "overall": {"false_positives_per_minute": 1.0,
                                                "mean_absolute_delay_seconds": 0.2}}
        better_events = copy.deepcopy(baseline)
        better_events["macro_f1"] = 0.6
        self.assertGreater(checkpoint_rank(better_events, 2.0), checkpoint_rank(baseline, 0.1))
        fewer_false = copy.deepcopy(baseline)
        fewer_false["overall"]["false_positives_per_minute"] = 0.5
        self.assertGreater(checkpoint_rank(fewer_false, 2.0), checkpoint_rank(baseline, 0.1))
        lower_delay = copy.deepcopy(baseline)
        lower_delay["overall"]["mean_absolute_delay_seconds"] = 0.1
        self.assertGreater(checkpoint_rank(lower_delay, 2.0), checkpoint_rank(baseline, 0.1))

    def test_event_without_any_supervised_pulse_is_not_evaluated(self):
        session = {"times": np.arange(30) / 10,
                   "intervals": [{"start_frame": 25, "end_frame": 29, "label": "open_hand"}],
                   "mask": np.ones((30, 2), bool)}
        metrics = action_event_metrics(session, np.zeros((30, 2), np.float32), self.config)
        self.assertEqual(metrics["per_class"]["open_hand"]["false_negative"], 0)


if __name__ == "__main__":
    unittest.main()
