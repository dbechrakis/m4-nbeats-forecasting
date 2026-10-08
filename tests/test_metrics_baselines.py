import unittest

import numpy as np

from deepforecast import baselines
from deepforecast.metrics import mase, mase_scale, owa, smape


class MetricTests(unittest.TestCase):
    def test_smape_matches_definition(self):
        actual, forecast = np.array([[100.0, 200.0]]), np.array([[110.0, 180.0]])
        expected = 200 * np.mean([10 / 210, 20 / 380])
        self.assertAlmostEqual(smape(actual, forecast)[0], expected)

    def test_mase_scales_by_seasonal_naive_error(self):
        history = np.array([1.0, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(mase_scale(history, 2), 2.0)
        self.assertAlmostEqual(mase(np.array([[7.0, 8.0]]), np.array([[9.0, 8.0]]), [history], 2)[0], 0.5)

    def test_owa_of_naive2_is_one(self):
        self.assertEqual(owa(10, 2, 10, 2), 1.0)
        self.assertAlmostEqual(owa(9, 1.8, 10, 2), 0.9)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(24 * 30)
        self.seasonal = 100 * (1 + 0.3 * np.sin(2 * np.pi * t / 24)) + 0.05 * t

    def test_seasonality_test(self):
        self.assertTrue(baselines.is_seasonal(self.seasonal, 24))
        noise = np.random.default_rng(0).normal(100, 1, 24 * 30)
        self.assertFalse(baselines.is_seasonal(noise, 24))
        self.assertFalse(baselines.is_seasonal(self.seasonal[:50], 24))  # under three seasons

    def test_naive2_carries_seasonality_and_naive_does_not(self):
        naive2 = baselines.naive2(self.seasonal, 24, 24)
        self.assertGreater(naive2.std(), 10)
        self.assertEqual(baselines.naive(self.seasonal, 24).std(), 0)
        self.assertLess(smape(self.seasonal[-24:][None], naive2[None])[0], 5)  # recovers the shape

    def test_seasonal_indices_average_one(self):
        self.assertAlmostEqual(baselines.seasonal_indices(self.seasonal, 24).mean(), 1.0)

    def test_naive2_equals_naive_without_seasonality(self):
        series = np.linspace(10, 20, 100)
        np.testing.assert_allclose(baselines.naive2(series, 6, 1), baselines.naive(series, 6))

    def test_theta_follows_trend_and_stays_non_negative(self):
        trend = np.linspace(1, 100, 200)
        forecast = baselines.theta(trend, 10, 1)
        self.assertTrue(np.all(np.diff(forecast) > 0))
        self.assertTrue(np.all(baselines.theta(np.linspace(100, 1, 200), 500, 1) >= 0))


if __name__ == "__main__":
    unittest.main()
