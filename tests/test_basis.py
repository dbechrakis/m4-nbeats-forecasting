import unittest

import numpy as np
import torch

from deepforecast import nbeats
from deepforecast.data import Dataset, validation_split
from deepforecast.training import Sampler


class BasisTests(unittest.TestCase):
    def test_generic_basis_takes_the_first_lookback_and_last_horizon_coefficients(self):
        theta = torch.arange(15, dtype=torch.float32)[None, :]
        back, fore = nbeats.GenericBasis(10, 5)(theta)
        torch.testing.assert_close(back, theta[:, :10])
        torch.testing.assert_close(fore, theta[:, -5:])

    def test_trend_basis_reproduces_a_known_polynomial_on_both_sides(self):
        lookback, horizon = 12, 6
        basis = nbeats.TrendBasis(2, lookback, horizon)
        coefficients = np.array([1.0, 2.0, 3.0])
        theta = torch.tensor(np.concatenate([coefficients, coefficients * 2])[None, :], dtype=torch.float32)
        back, fore = basis(theta)
        t_back, t_fore = np.arange(lookback) / lookback, np.arange(horizon) / horizon
        expected_back = coefficients[0] + coefficients[1] * t_back + coefficients[2] * t_back**2
        expected_fore = 2 * (coefficients[0] + coefficients[1] * t_fore + coefficients[2] * t_fore**2)
        np.testing.assert_allclose(back[0].numpy(), expected_back, rtol=1e-5)
        np.testing.assert_allclose(fore[0].numpy(), expected_fore, rtol=1e-5)

    def test_seasonality_basis_reproduces_a_cosine_and_a_sine_with_a_period_of_one_horizon(self):
        lookback, horizon = 24, 12
        basis = nbeats.SeasonalityBasis(1, lookback, horizon)
        harmonics = (basis.size - 1) // 2
        theta = torch.zeros(1, 2 * basis.size)
        theta[0, 1] = 1.0
        theta[0, basis.size + 1 + harmonics] = 1.0
        back, fore = basis(theta)
        np.testing.assert_allclose(back[0].numpy(), np.cos(2 * np.pi * np.arange(lookback) / horizon), atol=1e-5)
        np.testing.assert_allclose(fore[0].numpy(), np.sin(2 * np.pi * np.arange(horizon) / horizon), atol=1e-5)

    def test_a_constant_coefficient_gives_a_flat_seasonality_term(self):
        basis = nbeats.SeasonalityBasis(1, 24, 12)
        theta = torch.zeros(1, 2 * basis.size)
        theta[0, 0] = 2.5
        theta[0, basis.size] = -1.5
        back, fore = basis(theta)
        np.testing.assert_allclose(back[0].numpy(), 2.5, atol=1e-6)
        np.testing.assert_allclose(fore[0].numpy(), -1.5, atol=1e-6)


class ValidationHorizonTests(unittest.TestCase):
    def test_training_windows_and_targets_never_reach_the_held_out_horizon(self):
        series = [np.arange(1, 121, dtype=float) + 1000 * i for i in range(3)]
        dataset = Dataset("Weekly", ["a", "b", "c"], series, np.zeros((3, 13)), 13, 1)
        histories, targets = validation_split(dataset)
        sampler = Sampler(histories, lookback=26, horizon=13, history=200, season=1, seed=0)
        x, mask, y, y_mask, _ = sampler.batch(500)
        held_out = {value for row in targets for value in row}
        seen = set(x[mask > 0].ravel().tolist()) | set(y[y_mask > 0].ravel().tolist())
        self.assertFalse(seen & held_out)
        self.assertTrue(all(len(h) == 107 for h in histories))


class EnsembleTests(unittest.TestCase):
    def test_the_median_ignores_one_member_that_is_far_off(self):
        members = np.stack([np.full(5, 10.0), np.full(5, 11.0), np.full(5, 1000.0)])
        np.testing.assert_array_equal(np.median(members, axis=0), np.full(5, 11.0))


if __name__ == "__main__":
    unittest.main()
