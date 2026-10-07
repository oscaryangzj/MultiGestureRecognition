import copy
import unittest

import numpy as np
import torch

from gesture.action_candidates import action_excluded_frames, action_opportunities
from gesture.action_data import action_window_indices, make_action_targets, negative_input_regions
from gesture.action_features import ACTION_FEATURE_SIZE, make_action_features
from gesture.action_inference import predict_action_session, rolling_contexts
from gesture.action_resampling import resample_action_session, scores_on_source_frames
from gesture.config import load_config
from gesture.models import ActionCNNLSTM


class NegativeHistoryTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.config["data"]["action_window_seconds"] = 0.8

    def session(self, ignored=None, reviewed=True, contiguous=False):
        times = np.arange(128) * 0.041
        prompts = [{"gesture": "three_fingers_to_fist", "sample_type": "negative", "start_frame": 10, "end_frame": 54},
                   {"gesture": "wrist_pitch_opened", "sample_type": "negative", "start_frame": 55 if contiguous else 64, "end_frame": 116}]
        opportunities = action_opportunities(prompts, len(times))
        annotations = {"action_background_prompts": [0, 1] if reviewed else [],
                       "action_ignored_intervals": ignored or [], "action_review_mode": "prompts"}
        points = np.tile(np.column_stack((np.linspace(0.2, 0.6, 21), np.linspace(0.3, 0.7, 21))), (len(times), 1, 1))
        points[:, :, 0] += (0.01 * times ** 2)[:, None]
        scores = [{"opened": 0.6 + time * 0.01, "closed": 0.2} for time in times]
        regions = negative_input_regions(annotations, opportunities, len(times), self.config, times)
        features, valid, segments = make_action_features(points, np.ones(len(times), bool), scores, times,
                                                        self.config, input_regions=regions)
        excluded = action_excluded_frames(annotations, opportunities, len(times), self.config, times)
        intervals = [{"start_frame": 75, "end_frame": 82, "label": "open_hand"}] if reviewed else []
        targets, mask = make_action_targets(valid, segments, intervals, reviewed, self.config, excluded, times)
        return {"times": times, "points": points, "state_scores": scores, "detected": np.ones(len(times), bool),
                "features": features, "valid": valid, "segments": segments, "intervals": intervals,
                "reviewed": reviewed, "targets": targets, "mask": mask, "ignored_frames": excluded,
                "input_regions": regions, "sample_type": "negative"}

    def test_preparation_changes_cannot_change_inputs_or_isolated_predictions(self):
        source = self.session()
        changed = copy.deepcopy(source)
        preparation = source["input_regions"] < 0
        changed["points"][preparation] += [0.4, -0.3]
        for index in np.flatnonzero(preparation):
            changed["state_scores"][index] = {"opened": 0.01, "closed": 0.99}
        original = resample_action_session(source, self.config)
        modified = resample_action_session(changed, self.config)
        np.testing.assert_array_equal(original["valid"], modified["valid"])
        np.testing.assert_array_equal(original["features"], modified["features"])
        np.testing.assert_array_equal(original["targets"], modified["targets"])
        np.testing.assert_array_equal(original["mask"], modified["mask"])

        model = ActionCNNLSTM(ACTION_FEATURE_SIZE, 4, 1, 2, 4, 3, 0).eval()
        checkpoint = {"feature_mean": torch.zeros(ACTION_FEATURE_SIZE), "feature_std": torch.ones(ACTION_FEATURE_SIZE),
                      "window_seconds": self.config["data"]["action_window_seconds"]}
        expected = predict_action_session(model, original, checkpoint, 16)
        actual = predict_action_session(model, modified, checkpoint, 16)
        np.testing.assert_allclose(expected, actual, equal_nan=True)
        self.assertTrue(np.isnan(expected[~original["valid"]]).all())
        self.assertTrue((original["targets"][:, 0] * original["mask"][:, 0]).any())
        _, mapped = scores_on_source_frames(original, expected)
        self.assertTrue(np.isnan(mapped[preparation]).all())

    def test_all_windows_and_rolling_contexts_stay_inside_one_task(self):
        session = resample_action_session(self.session(), self.config)
        windows = list(action_window_indices(session, self.config))
        self.assertTrue(windows)
        for indices in windows:
            self.assertTrue(session["valid"][indices].all())
            self.assertGreaterEqual(session["input_regions"][indices[0]], 0)
            self.assertEqual(len(np.unique(session["input_regions"][indices])), 1)
        for _frame, indices in rolling_contexts(session, 0.8):
            self.assertEqual(len(np.unique(session["input_regions"][indices])), 1)
        for region in np.unique(session["input_regions"][session["valid"]]):
            frames = np.flatnonzero(session["valid"] & (session["input_regions"] == region))
            np.testing.assert_array_equal(session["features"][frames[0], 42:47], 0)
            self.assertTrue(session["valid"][frames[:2]].all())
            self.assertFalse(session["mask"][frames[:2]].any())
            self.assertTrue(session["mask"][frames[2]].all())

    def test_adjacent_tasks_reset_interpolation_derivatives_and_visibility(self):
        source = self.session(contiguous=True)
        source["points"][55:] += [0.3, 0.1]
        session = resample_action_session(source, self.config)
        frames = np.flatnonzero(session["valid"] & (session["input_regions"] == 1))
        # The grid point straddling the boundary cannot borrow either task's history.
        boundary = np.searchsorted(session["times"], source["times"][55])
        self.assertFalse(session["valid"][boundary - 1])
        self.assertEqual(frames[0], boundary)
        np.testing.assert_array_equal(session["features"][frames[0], 42:47], 0)
        self.assertFalse(session["mask"][frames[:2]].any())
        previous = np.flatnonzero(session["valid"] & (session["input_regions"] == 0))[-1]
        self.assertNotEqual(session["segments"][previous], session["segments"][frames[0]])

    def test_explicit_ignored_interval_splits_history_and_pending_tasks_keep_inputs(self):
        source = self.session(ignored=[{"start_frame": 20, "end_frame": 30}])
        self.assertFalse(source["valid"][20:31].any())
        self.assertNotEqual(source["segments"][19], source["segments"][40])
        session = resample_action_session(source, self.config)
        for _frame, indices in rolling_contexts(session, 8):
            self.assertEqual(len(np.unique(session["input_regions"][indices])), 1)
        pending = resample_action_session(self.session(reviewed=False), self.config)
        self.assertTrue(pending["valid"].any())
        self.assertFalse(pending["mask"].any())

    def test_complete_fifteen_second_task_has_43_eight_second_windows(self):
        self.config["data"]["action_window_seconds"] = 8
        session = {"times": np.arange(450) / 30, "valid": np.ones(450, bool), "segments": np.zeros(450, int),
                   "intervals": [], "mask": np.ones((450, 2), bool)}
        windows = list(action_window_indices(session, self.config))
        self.assertEqual(len(windows), 43)
        self.assertTrue(all(len(indices) == 240 for indices in windows))
        self.assertEqual(windows[-1][-1], 449)

    def test_pulse_with_history_outside_the_task_is_ignored(self):
        session = self.session()
        intervals = [{"start_frame": 5, "end_frame": 15, "label": "open_hand"}]
        targets, mask = make_action_targets(session["valid"], session["segments"], intervals, True,
                                            self.config, session["ignored_frames"], session["times"])
        self.assertTrue(targets.any())
        self.assertFalse((targets * mask).any())


if __name__ == "__main__":
    unittest.main()
