import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gesture.action_data import make_action_targets
from gesture.action_features import ACTION_FEATURE_SIZE, ActionFeatureExtractor
from gesture.action_resampling import ActionResampler
from gesture.config import load_config
from gesture.models import ActionCNNLSTM


class ActionResamplingTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.points = np.column_stack((np.linspace(0.2, 0.6, 21), np.linspace(0.3, 0.7, 21)))
        self.scores = {"opened": 0.8, "closed": 0.1}

    def test_uniform_grid_interpolates_before_derivatives_and_waits_for_observation(self):
        resampler = ActionResampler(30)
        extractor = ActionFeatureExtractor(self.config)
        emitted, features = [], []
        for elapsed in (0.0, 0.04, 0.11, 0.17):
            samples = resampler.update(self.points + [elapsed * 0.1, elapsed * 0.2], self.scores, elapsed)
            self.assertTrue(all(sample[0] <= elapsed for sample in samples))
            for time, points, scores in samples:
                emitted.append(time)
                feature, _reset = extractor.update(points, scores, time)
                features.append(feature)
        np.testing.assert_allclose(np.diff(emitted), 1 / 30)
        np.testing.assert_allclose(np.asarray(features)[1:, 42:44], np.tile([0.1, 0.2], (len(features) - 1, 1)), atol=1e-5)
        np.testing.assert_allclose(np.asarray(features)[2:, 44:46], 0, atol=1e-4)

    def test_missing_observations_are_never_interpolated_into_a_hand(self):
        resampler = ActionResampler(30)
        resampler.update(self.points, self.scores, 0.0)
        missing = resampler.update(None, None, 0.07)
        recovery = resampler.update(self.points, self.scores, 0.11)
        self.assertTrue(all(points is None and scores is None for _, points, scores in missing + recovery))
        returned = resampler.update(self.points, self.scores, 0.17)
        self.assertTrue(all(points is not None for _, points, _ in returned))

    def test_pulse_is_defined_by_elapsed_time_at_both_sampling_rates(self):
        for rate in (15, 30):
            times = np.arange(rate * 2) / rate
            intervals = [{"start_frame": 0, "end_frame": rate, "label": "close_hand"}]
            targets, _mask = make_action_targets(np.ones(len(times), bool), np.zeros(len(times), int),
                                                 intervals, True, self.config, times=times)
            expected = (times >= 1.1 - 1e-9) & (times < 1.25 - 1e-9)
            np.testing.assert_array_equal(targets[:, 1].astype(bool), expected)

    def test_meta_model_cannot_see_future_or_right_padding(self):
        torch.manual_seed(42)
        model = ActionCNNLSTM(ACTION_FEATURE_SIZE, 8, 3, 2, 8, 3, 0.1).eval()
        prefix = torch.randn(1, 9, ACTION_FEATURE_SIZE)
        with torch.no_grad():
            expected = model(prefix)
            with_future = model(torch.cat((prefix, torch.randn(1, 5, ACTION_FEATURE_SIZE)), dim=1))
            with_padding = model(torch.cat((prefix, torch.zeros(1, 5, ACTION_FEATURE_SIZE)), dim=1))
        self.assertEqual(expected.shape, (1, 9, 2))
        torch.testing.assert_close(expected, with_future[:, :9])
        torch.testing.assert_close(expected, with_padding[:, :9])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not accessible in this process")
    def test_mps_training_then_eval_matches_batch_sizes_and_reloaded_weights(self):
        torch.manual_seed(42)
        model = ActionCNNLSTM(ACTION_FEATURE_SIZE, 8, 3, 2, 8, 3, 0.1).to("mps")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model(torch.randn(3, 15, ACTION_FEATURE_SIZE, device="mps")).square().mean().backward()
        optimizer.step()
        model.eval()
        inputs = torch.randn(10, 15, ACTION_FEATURE_SIZE, device="mps")
        with torch.no_grad():
            expected = model(inputs)
            partial = model(inputs[:3])
        torch.testing.assert_close(expected[:3], partial, atol=1e-5, rtol=1e-5)
        with tempfile.TemporaryDirectory(prefix="mgr-meta-weights-") as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = ActionCNNLSTM(ACTION_FEATURE_SIZE, 8, 3, 2, 8, 3, 0.1)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            restored.to("mps").eval()
            with torch.no_grad():
                actual = restored(inputs[:3])
                cpu = restored.to("cpu")(inputs[:3].cpu())
            torch.testing.assert_close(partial.cpu(), actual.cpu(), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(partial.cpu(), cpu, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
