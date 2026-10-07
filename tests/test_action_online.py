import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gesture.action_features import ACTION_FEATURE_SIZE, make_action_features
from gesture.action_inference import ActionThresholdDecoder, OnlineActionPredictor, predict_action_session, preprocessing_config
from gesture.action_resampling import resample_action_session, scores_on_source_frames
from gesture.config import load_config
from gesture.models import ActionCNNLSTM
from scripts import run_action


class OnlineActionTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        torch.manual_seed(42)
        self.model = ActionCNNLSTM(ACTION_FEATURE_SIZE, 8, 3, 2, 8, 3, 0.1).eval()
        self.checkpoint = {
            "classes": self.config["labels"]["action_classes"],
            "input_size": ACTION_FEATURE_SIZE,
            "feature_mean": torch.linspace(-0.2, 0.2, ACTION_FEATURE_SIZE),
            "feature_std": torch.linspace(0.8, 1.2, ACTION_FEATURE_SIZE),
            "feature_config": copy.deepcopy(self.config["features"]),
            "no_hand_reset_frames": self.config["data"]["no_hand_reset_frames"],
            "window_seconds": 0.5,
            "sample_rate_hz": self.config["data"]["action_sample_rate_hz"],
            "background_visible_frames": self.config["data"]["action_background_visible_frames"],
            "target_config": {key: self.config["labels"][key] for key in (
                "action_positive_delay_seconds", "action_positive_duration_seconds")},
        }
        self.points = np.column_stack((np.linspace(0.2, 0.6, 21), np.linspace(0.3, 0.7, 21))).astype(np.float32)
        self.state_scores = {"opened": 0.7, "closed": 0.2}

    def test_online_scores_match_offline_with_irregular_sampling_short_gaps_and_resets(self):
        times = np.r_[0.0, np.cumsum(np.tile([0.03, 0.07, 0.04], 20))]
        points = [self.points + [time * 0.01, np.sin(time) * 0.02] for time in times]
        detected = np.ones(len(times), bool)
        detected[12] = False
        detected[30:33] = False
        scores = [self.state_scores for _ in times]
        config = preprocessing_config(self.config, self.checkpoint)
        features, valid, segments = make_action_features(points, detected, scores, times, config)
        session = {"features": features, "valid": valid, "segments": segments, "times": times,
                   "targets": np.zeros((len(times), 2), np.float32), "points": points,
                   "detected": detected, "state_scores": scores, "intervals": [], "reviewed": True,
                   "ignored_frames": np.zeros(len(times), bool)}
        resampled = resample_action_session(session, config)
        _source, expected = scores_on_source_frames(resampled, predict_action_session(self.model, resampled, self.checkpoint, 8))
        predictor = OnlineActionPredictor(self.model, self.checkpoint, self.config)
        actual = np.full_like(expected, np.nan)
        resets = []
        for frame, elapsed in enumerate(times):
            prediction, reset = predictor.update(points[frame] if detected[frame] else None,
                                                 scores[frame] if detected[frame] else None, elapsed)
            if prediction is not None:
                actual[frame] = prediction
                newest = predictor.history[-1][0]
                self.assertTrue(all(observed > newest - predictor.seconds for observed, _ in predictor.history))
            if reset:
                resets.append(frame)
                self.assertEqual(len(predictor.history), 0)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5, equal_nan=True)
        self.assertEqual(resets, [32])

    def test_constant_pulse_is_counted_once_and_crossings_use_global_debounce(self):
        decoder = ActionThresholdDecoder(self.checkpoint["classes"], self.config)
        self.assertEqual(decoder.update([0.6, 0.1], 0.0), [0])
        self.assertEqual(decoder.update([0.9, 0.1], 0.01), [])
        self.assertEqual(decoder.update([0.1, 0.8], 0.02), [])
        self.assertEqual(decoder.update([0.1, 0.9], 0.2), [])
        self.assertEqual(decoder.update([0.1, 0.1], 0.3), [])
        self.assertEqual(decoder.update([0.1, 0.8], 0.4), [1])

    def test_short_missing_gap_keeps_crossing_state_and_reset_clears_it(self):
        decoder = ActionThresholdDecoder(self.checkpoint["classes"], self.config)
        self.assertEqual(decoder.update([0.6, 0.1], 0.0), [0])
        self.assertEqual(decoder.update(None, 0.1), [])
        self.assertEqual(decoder.update([0.7, 0.1], 0.2), [])
        decoder.reset()
        self.assertEqual(decoder.update([0.7, 0.1], 0.3), [0])

    def run_preview(self, session_id=None):
        frame = np.zeros((480, 640, 3), np.uint8)
        with patch("sys.argv", ["run_action"] + (["--session-id", session_id] if session_id else [])), \
                patch.object(run_action, "load_config", return_value=self.config), \
                patch.object(run_action, "load_action_model", return_value=(self.model, self.checkpoint)), \
                patch.object(run_action, "load_frame_times", return_value=[10.0, 10.04, 10.1]), \
                patch.object(run_action.cv2, "VideoCapture") as capture, \
                patch.object(run_action, "create_state_recognizer"), \
                patch.object(run_action, "recognize_video_frame", side_effect=[
                    (self.points, np.asarray([0.2, 0.7, 0.0])),
                    (None, None),
                    (self.points, np.asarray([0.2, 0.7, 0.0])),
                ]) as recognize, \
                patch.object(run_action.cv2, "namedWindow"), \
                patch.object(run_action.cv2, "imshow") as show, \
                patch.object(run_action.cv2, "waitKey", return_value=-1), \
                patch.object(run_action.cv2, "getWindowProperty", return_value=1), \
                patch.object(run_action.cv2, "destroyAllWindows") as close:
            capture.return_value.isOpened.return_value = True
            capture.return_value.read.side_effect = [(True, frame)] * 3 + [(False, None)]
            run_action.main()
            self.assertEqual(show.call_count, 3)
            self.assertEqual(show.call_args.args[1].shape, (841, 640, 3))
            capture.return_value.release.assert_called_once()
            close.assert_called_once()
            return capture.call_args.args[0], [call.args[2] for call in recognize.call_args_list]

    def test_camera_loop_processes_frames_without_zero_filling_missing_hand(self):
        source, timestamps = self.run_preview()
        self.assertEqual(source, self.config["data"]["camera_index"])
        self.assertTrue(all(second > first for first, second in zip(timestamps, timestamps[1:])))

    def test_video_uses_recorded_observation_times_for_online_pipeline(self):
        source, timestamps = self.run_preview("example")
        self.assertTrue(source.endswith("example/video.mp4"))
        self.assertEqual(timestamps, [10000, 10040, 10100])


if __name__ == "__main__":
    unittest.main()
