"""Official M4 point-forecast metrics."""

import numpy as np


def smape(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Symmetric MAPE per series, in percent (M4 definition)."""
    return 200 * np.mean(np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast)), axis=-1)


def mase_scale(history: np.ndarray, season: int) -> float:
    """In-sample mean absolute seasonal-naive error (M4 uses the full training series)."""
    return float(np.mean(np.abs(history[season:] - history[:-season])))


def mase(actual: np.ndarray, forecast: np.ndarray, histories: list[np.ndarray], season: int) -> np.ndarray:
    scales = np.array([mase_scale(h, season) for h in histories])
    return np.mean(np.abs(actual - forecast), axis=-1) / scales


def owa(smape_model: float, mase_model: float, smape_naive2: float, mase_naive2: float) -> float:
    """Overall Weighted Average relative to Naive2; below 1 beats the official benchmark."""
    return 0.5 * (smape_model / smape_naive2 + mase_model / mase_naive2)
