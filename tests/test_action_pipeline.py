import copy
import unittest

import numpy as np
import torch

from gesture.action_data import action_window_indices, make_action_targets, mask_truncated_events
from gesture.action_evaluation import action_event_metrics
from gesture.action_features import ACTION_FEATURE_SIZE, ActionFeatureExtractor, make_action_features
from gesture.action_inference import rolling_contexts
from gesture.config import load_config
from gesture.models import ActionLSTM


def hand(angle=0.0, shift=(0.0, 0.0)):
    points = np.zeros((21, 2), np.float32)
    points[:, 0] = np.linspace(0.2, 0.6, 21)
    points[:, 1] = np.linspace(0.2, 0.7, 21)
    points[9] = points[0] + 0.1 * np.asarray([np.cos(angle), np.sin(angle)])
    return points + np.asarray(shift, np.float32)


class ActionPipelineTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.scores = {"opened": 0.7, "closed": 0.2}

    def test_geometry_actual_time_and_angle_wrap(self):
        extractor = ActionFeatureExtractor(self.config)
        first, _ = extractor.update(hand(np.deg2rad(179)), self.scores, 0.0)
        second, _ = extractor.update(hand(np.deg2rad(-179), (0.02, 0)), self.scores, 0.2)
        self.assertEqual(first.shape, (ACTION_FEATURE_SIZE,))
        np.testing.assert_allclose(first[42:46], 0)
        self.assertAlmostEqual(first[46], 0.0)
        self.assertAlmostEqual(second[46], np.deg2rad(2) / 0.2, places=5)
        np.testing.assert_allclose(second[44:46], 0)
        np.testing.assert_allclose(first[47:], [0.7, 0.2])
        extractor = ActionFeatureExtractor(self.config)
        for time in (0.0, 0.2, 0.7):
            features, _ = extractor.update(hand(shift=(time * 0.1, time * 0.2)), self.scores, time)
        np.testing.assert_allclose(features[42:44], [0.1, 0.2], atol=1e-6)
        np.testing.assert_allclose(features[44:46], 0, atol=1e-6)

    def test_mirrored_hands_share_local_shape_but_keep_camera_motion(self):
        original = ActionFeatureExtractor(self.config)
        mirrored = ActionFeatureExtractor(self.config)
        for time in (0.0, 0.2, 0.7):
            points = hand(angle=-np.pi / 2 + time * 0.1,
                          shift=(time * time * 0.1, time * 0.2))
            reflected = points.copy()
            reflected[:, 0] = 1 - reflected[:, 0]
            expected, _ = original.update(points, self.scores, time, (1920, 1080))
            actual, _ = mirrored.update(reflected, self.scores, time, (1920, 1080))
            np.testing.assert_allclose(actual[:42], expected[:42], atol=1e-5)
            np.testing.assert_allclose(actual[42:47], expected[42:47] * [-1, 1, -1, 1, -1], atol=1e-5)
            np.testing.assert_array_equal(actual[47:], expected[47:])
        self.assertGreater(abs(expected[42]), 0)
        self.assertGreater(abs(expected[44]), 0)
        self.assertGreater(abs(expected[46]), 0)

    def test_old_checkpoint_feature_config_keeps_mirrored_shape_distinct(self):
        self.config["features"].pop("action_canonicalize_hand")
        points = hand(angle=-np.pi / 2)
        reflected = points.copy()
        reflected[:, 0] = 1 - reflected[:, 0]
        original, _ = ActionFeatureExtractor(self.config).update(points, self.scores, 0)
        mirrored, _ = ActionFeatureExtractor(self.config).update(reflected, self.scores, 0)
        np.testing.assert_allclose(mirrored[:42:2], -original[:42:2], atol=2e-6)
        np.testing.assert_allclose(mirrored[1:42:2], original[1:42:2], atol=2e-6)

    def test_constant_translation_preserves_all_action_features(self):
        original = ActionFeatureExtractor(self.config)
        translated = ActionFeatureExtractor(self.config)
        for time in (0.0, 0.2, 0.7):
            points = hand(angle=time * 0.1, shift=(time * 0.1, time * 0.2))
            expected, _ = original.update(points, self.scores, time)
            actual, _ = translated.update(points + [0.12, -0.07], self.scores, time)
            np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_missing_frames_skip_and_reset_derivative_history(self):
        extractor = ActionFeatureExtractor(self.config)
        extractor.update(hand(), self.scores, 0)
        self.assertEqual(extractor.update(None, None, 0.1), (None, False))
        feature, reset = extractor.update(hand(shift=(0.02, 0)), self.scores, 0.2)
        self.assertFalse(reset)
        np.testing.assert_allclose(feature[42:44], [0.1, 0], atol=1e-6)
        for index in range(3):
            feature, reset = extractor.update(None, None, 0.3 + index * 0.1)
            self.assertIsNone(feature)
            self.assertEqual(reset, index == 2)
        feature, _ = extractor.update(hand(shift=(0.5, 0)), self.scores, 0.6)
        np.testing.assert_allclose(feature[42:46], 0)

    def test_targets_review_mask_and_reset_during_delayed_pulse(self):
        valid = np.ones(30, bool)
        valid[14:17] = False
        segments = np.zeros(30, int)
        segments[16:] = 1
        intervals = [{"start_frame": 5, "end_frame": 10, "label": "open_hand"}]
        target, mask = make_action_targets(valid, segments, intervals, False, self.config)
        np.testing.assert_array_equal(np.flatnonzero(target[:, 0]), np.arange(13, 18))
        self.assertTrue(mask[13, 0])
        self.assertFalse(mask[17, 0])
        self.assertFalse(mask[:, 1].any())
        _, mask = make_action_targets(valid, segments, intervals, True, self.config)
        self.assertFalse(mask[:2].any())
        self.assertTrue(mask[2:5].all())
        self.assertFalse(mask[17:19].any())
        self.assertTrue(mask[19].all())

    def test_stride_starts_and_cropped_event_mask(self):
        session = {"times": np.arange(60) / 10, "valid": np.ones(60, bool), "segments": np.zeros(60, int),
                   "intervals": [{"start_frame": 10, "end_frame": 12, "label": "open_hand"}]}
        session["targets"], session["mask"] = make_action_targets(session["valid"], session["segments"], session["intervals"], True, self.config, times=session["times"])
        self.config["data"]["action_window_seconds"] = 2.0
        windows = list(action_window_indices(session, self.config))
        self.assertTrue(windows)
        self.assertTrue(all(frames[0] % 5 == 0 for frames in windows))
        self.assertNotIn(10, [frames[0] for frames in windows])
        mask = mask_truncated_events(session, np.arange(13, 20), self.config)
        self.assertFalse(mask[:2, 0].any())
        self.assertTrue(mask[:, 1].all())

    def test_training_and_rolling_history_never_cross_reset(self):
        detected = np.ones(50, bool)
        detected[15:18] = False
        times = np.arange(50) / 10
        features, valid, segments = make_action_features([hand() for _ in times], detected, [self.scores for _ in times], times, self.config)
        self.assertFalse(valid[15:18].any())
        self.assertEqual(segments[18], 1)
        session = {"features": features, "valid": valid, "segments": segments, "times": times,
                   "mask": np.ones((50, 2), bool), "intervals": []}
        for frame, context in rolling_contexts(session, 8):
            self.assertTrue(valid[context].all())
            self.assertTrue((segments[context] == segments[frame]).all())
        for context in action_window_indices(session, self.config):
            self.assertTrue((segments[context] == segments[context[0]]).all())

    def test_short_segment_can_start_after_initial_incomplete_action(self):
        session = {"times": np.arange(30) / 10, "valid": np.ones(30, bool), "segments": np.zeros(30, int),
                   "intervals": [{"start_frame": 0, "end_frame": 7, "label": "close_hand"}],
                   "mask": np.ones((30, 2), bool)}
        windows = list(action_window_indices(session, self.config))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0], 10)
        self.assertTrue(mask_truncated_events(session, windows[0], self.config).all())

    def test_completion_event_supervision_is_exactly_e_plus_three_through_seven(self):
        valid = np.ones(40, bool)
        segments = np.zeros(40, int)
        intervals = [{"start_frame": 7, "end_frame": 12, "label": "close_hand"}]
        targets, mask = make_action_targets(valid, segments, intervals, True, self.config)
        np.testing.assert_array_equal(np.flatnonzero(targets[:, 1]), np.arange(15, 20))
        self.assertFalse(targets[:, 0].any())
        self.assertTrue(mask[15:20].all())
        np.testing.assert_array_equal(targets[20:], 0)
        self.assertTrue(mask[20:].all())  # Stable HOLD trains against repeated events.

    def test_evaluation_duplicate_and_late_predictions_count_as_false(self):
        session = {"times": np.arange(40) / 10, "intervals": [{"start_frame": 3, "end_frame": 10, "label": "open_hand"}]}
        predictions = np.zeros((40, 2), np.float32)
        predictions[11:14, 0] = [0.6, 0.9, 0.7]
        predictions[15, 0] = 0.8
        predictions[30, 0] = 0.9
        metrics = action_event_metrics(session, predictions, self.config)["per_class"]["open_hand"]
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 2)
        self.assertEqual(metrics["false_negative"], 0)
        self.assertAlmostEqual(metrics["mean_delay_seconds"], 0.2)

    def test_causal_model_padding_does_not_change_valid_prefix(self):
        model = ActionLSTM(ACTION_FEATURE_SIZE, 8, 1, 2).eval()
        features = torch.randn(1, 6, ACTION_FEATURE_SIZE)
        with torch.no_grad():
            short = model(features)
            padded = model(torch.cat((features, torch.zeros(1, 4, ACTION_FEATURE_SIZE)), dim=1))
        torch.testing.assert_close(short, padded[:, :6])


if __name__ == "__main__":
    unittest.main()
