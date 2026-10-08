import unittest

import numpy as np
import torch

from deepforecast import nbeats
from deepforecast.data import Dataset, validation_split
from deepforecast.training import Sampler, TrainConfig, predict, train, windows


class WindowTests(unittest.TestCase):
    def test_short_history_is_left_padded_and_masked(self):
        x, mask = windows(np.array([1.0, 2.0, 3.0]), 3, 5)
        np.testing.assert_array_equal(x, [0, 0, 1, 2, 3])
        np.testing.assert_array_equal(mask, [0, 0, 1, 1, 1])

    def test_sampler_targets_come_after_the_window(self):
        series = [np.arange(1, 201, dtype=float)]
        sampler = Sampler(series, lookback=10, horizon=5, history=50, season=1, seed=0)
        x, mask, y, y_mask, _ = sampler.batch(64)
        for row in range(64):
            last_input = x[row][mask[row] > 0][-1]
            observed = y[row][y_mask[row] > 0]
            np.testing.assert_array_equal(observed, last_input + np.arange(1, len(observed) + 1))

    def test_validation_split_holds_out_the_last_horizon(self):
        dataset = Dataset("Weekly", ["a"], [np.arange(30.0)], np.zeros((1, 13)), 13, 1)
        histories, targets = validation_split(dataset)
        self.assertEqual(len(histories[0]), 17)
        np.testing.assert_array_equal(targets[0], np.arange(17.0, 30.0))


class ModelTests(unittest.TestCase):
    def test_shapes_for_both_architectures(self):
        x, mask = torch.ones(4, 30), torch.ones(4, 30)
        for model in (nbeats.generic(30, 10, blocks=2, width=16), nbeats.interpretable(30, 10, width=16)):
            self.assertEqual(model(x, mask).shape, (4, 10))

    def test_interpretable_parts_sum_to_the_forecast(self):
        model = nbeats.interpretable(30, 10, width=16)
        x, mask = torch.rand(3, 30) + 1, torch.ones(3, 30)
        parts = model.stacks_forecast(x, mask)
        torch.testing.assert_close(x[:, -1:] + sum(parts), model(x, mask))

    def test_learns_a_seasonal_pattern(self):
        t = np.arange(400)
        rng = np.random.default_rng(0)
        series = [10 + 3 * np.sin(2 * np.pi * (t + p) / 12) + rng.normal(0, 0.1, 400) for p in range(20)]
        dataset = Dataset("Weekly", [str(i) for i in range(20)], series, np.zeros((20, 12)), 12, 12)
        validation = validation_split(dataset)
        config = TrainConfig(architecture="generic", lookback_multiple=3, batch_size=128, blocks=2, width=64, eval_every=100)
        model, lookback, best_step, curve = train(config, validation[0], 12, 12, 400, validation, log=lambda _: None)
        self.assertLess(curve[-1]["validation_smape"], 5.0)  # a flat forecast scores about 19
        forecast = predict(model, validation[0], lookback)
        self.assertEqual(forecast.shape, (20, 12))


if __name__ == "__main__":
    unittest.main()
