"""M4 statistical benchmarks: Naive, Seasonal Naive, Naive2, SES, Holt, Damped, Comb and Theta.

Naive2 and Theta follow the competition's reference implementation: a 90% autocorrelation
test decides whether to apply classical multiplicative decomposition before forecasting.
"""

import numpy as np


def acf(series: np.ndarray, lag: int) -> float:
    centred = series - series.mean()
    return float(np.sum(centred[lag:] * centred[:-lag]) / np.sum(centred**2))


def is_seasonal(series: np.ndarray, season: int) -> bool:
    """M4 seasonality test: |ACF(season)| beyond the 90% limit using lags below it."""
    if season <= 1 or len(series) < 3 * season:
        return False
    coefficients = [acf(series, lag) for lag in range(1, season + 1)]
    limit = 1.645 * np.sqrt((1 + 2 * np.sum(np.square(coefficients[:-1]))) / len(series))
    return abs(coefficients[-1]) > limit


def seasonal_indices(series: np.ndarray, season: int) -> np.ndarray:
    """Classical multiplicative decomposition indices, one per position in the cycle."""
    if season % 2 == 0:
        kernel = np.r_[0.5, np.ones(season - 1), 0.5] / season
    else:
        kernel = np.ones(season) / season
    trend = np.convolve(series, kernel, mode="valid")
    offset = (len(series) - len(trend)) // 2
    ratios = series[offset: offset + len(trend)] / trend
    positions = (np.arange(len(trend)) + offset) % season
    indices = np.array([ratios[positions == p].mean() for p in range(season)])
    return indices * season / indices.sum()


def deseasonalise(series: np.ndarray, season: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (adjusted series, indices for the next len(series) + horizon positions)."""
    if not is_seasonal(series, season):
        return series, np.ones(len(series) * 2 + 1000)
    indices = seasonal_indices(series, season)
    cycle = np.resize(indices, len(series) + 1000)
    return series / cycle[: len(series)], cycle


def naive(series: np.ndarray, horizon: int, season: int = 1) -> np.ndarray:
    return np.repeat(series[-1], horizon)


def seasonal_naive(series: np.ndarray, horizon: int, season: int) -> np.ndarray:
    if season <= 1:
        return naive(series, horizon)
    return np.resize(series[-season:], horizon)


def naive2(series: np.ndarray, horizon: int, season: int) -> np.ndarray:
    adjusted, cycle = deseasonalise(series, season)
    return adjusted[-1] * cycle[len(series): len(series) + horizon]


def ses_fit(series: np.ndarray, alphas=np.linspace(0.05, 1.0, 20)) -> tuple[float, float]:
    """Simple exponential smoothing; alpha minimises one-step squared error."""
    best = (np.inf, 1.0, series[-1])
    for alpha in alphas:
        level, error = series[0], 0.0
        for value in series[1:]:
            error += (value - level) ** 2
            level = alpha * value + (1 - alpha) * level
        if error < best[0]:
            best = (error, alpha, level)
    return best[1], best[2]


def ses(series: np.ndarray, horizon: int, season: int) -> np.ndarray:
    adjusted, cycle = deseasonalise(series, season)
    _, level = ses_fit(adjusted)
    return level * cycle[len(series): len(series) + horizon]


def theta(series: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Classical Theta (theta = 2): SES on the series plus half the linear-trend slope."""
    adjusted, cycle = deseasonalise(series, season)
    n = len(adjusted)
    time = np.arange(n)
    slope, intercept = np.polyfit(time, adjusted, 1)
    alpha, level = ses_fit(adjusted)
    steps = np.arange(1, horizon + 1)
    drift = 0.5 * slope * (steps - 1 + 1 / alpha - ((1 - alpha) ** n) / alpha)
    return np.maximum((level + drift) * cycle[n: n + horizon], 0.0)  # the reference clips at zero


ALPHAS = np.linspace(0.05, 1.0, 20)
BETA_SHARES = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])   # beta = share x alpha, so beta never exceeds alpha
PHIS = np.array([0.80, 0.85, 0.90, 0.95, 0.98])


def holt_fit(series: np.ndarray, damped: bool) -> tuple[np.ndarray, float, float]:
    """Holt's linear trend, optionally damped, by grid search on one-step squared error.

    The M4 reference fits these with R's forecast::holt (likelihood optimisation). A grid, run for every parameter
    combination at once, follows this repository's SES and Theta and keeps the benchmark dependency-free.
    Returns (alpha, beta, phi) as an array, and the final level and trend.
    """
    a, b, f = np.meshgrid(ALPHAS, BETA_SHARES, PHIS if damped else np.array([1.0]), indexing="ij")
    alpha, beta, phi = a.ravel(), (a * b).ravel(), f.ravel()
    level = np.full(alpha.shape, series[0], dtype=float)
    trend = np.full(alpha.shape, series[1] - series[0] if len(series) > 1 else 0.0, dtype=float)
    error = np.zeros(alpha.shape)
    for value in series[1:]:
        forecast = level + phi * trend
        error += (value - forecast) ** 2
        new_level = alpha * value + (1 - alpha) * forecast
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    best = int(np.argmin(error))
    return np.array([alpha[best], beta[best], phi[best]]), float(level[best]), float(trend[best])


def _trend_forecast(series: np.ndarray, horizon: int, season: int, damped: bool) -> np.ndarray:
    adjusted, cycle = deseasonalise(series, season)
    (_, _, phi), level, trend = holt_fit(adjusted, damped)
    steps = np.arange(1, horizon + 1)
    cumulative = steps.astype(float) if phi == 1.0 else phi * (1 - phi**steps) / (1 - phi)
    return (level + trend * cumulative) * cycle[len(series): len(series) + horizon]


def holt(series: np.ndarray, horizon: int, season: int) -> np.ndarray:
    return _trend_forecast(series, horizon, season, damped=False)


def damped(series: np.ndarray, horizon: int, season: int) -> np.ndarray:
    return _trend_forecast(series, horizon, season, damped=True)


def comb(series: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """M4's Comb benchmark: the mean of SES, Holt and Damped."""
    return (ses(series, horizon, season) + holt(series, horizon, season) + damped(series, horizon, season)) / 3


BENCHMARKS = {
    "Naive": naive,
    "Seasonal naive": seasonal_naive,
    "Naive2": naive2,
    "SES": ses,
    "Holt": holt,
    "Damped": damped,
    "Comb": comb,
    "Theta": theta,
}
