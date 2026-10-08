"""Global training of N-BEATS on windows sampled across all series of one frequency.

Protocol (no test leakage):
1. Train on each series minus its last horizon; score that held-out horizon every
   ``eval_every`` steps and keep the weights with the best validation sMAPE.
2. Forecast the official test horizon once with those weights, feeding the full training
   series as input. The network never trained on targets from the last training horizon;
   it only reads those values as recent history, exactly as it would in production.
Windows are divided by the maximum of their lookback, so one network serves series whose
levels differ by orders of magnitude; the M4 metrics are scale-free.
"""

from dataclasses import dataclass, field

import copy

import numpy as np
import torch

from deepforecast.metrics import smape


@dataclass
class TrainConfig:
    lookback_multiple: int = 3  # lookback = multiple x horizon
    history_multiple: float = 10  # sample cut points from the last multiple x horizon points
    architecture: str = "generic"
    loss: str = "smape"
    steps: int = 3000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    eval_every: int = 250
    seed: int = 0
    width: int = 256
    blocks: int = 12
    tags: dict = field(default_factory=dict)


def windows(series: np.ndarray, cut: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Lookback window ending at ``cut`` (exclusive), left-padded with zeros, and its mask."""
    start = max(0, cut - lookback)
    values = series[start:cut]
    x = np.zeros(lookback, dtype=np.float32)
    mask = np.zeros(lookback, dtype=np.float32)
    x[lookback - len(values):] = values
    mask[lookback - len(values):] = 1.0
    return x, mask


def scale(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    peak = np.where(mask > 0, np.abs(x), 0).max(axis=-1, keepdims=True)
    return np.maximum(peak, 1e-8)


class Sampler:
    def __init__(self, series: list[np.ndarray], lookback: int, horizon: int, history: int, season: int, seed: int):
        self.series, self.lookback, self.horizon, self.history = series, lookback, horizon, history
        self.rng = np.random.default_rng(seed)
        self.mase_scales = np.array(
            [np.mean(np.abs(s[season:] - s[:-season])) if len(s) > season else 1.0 for s in series], dtype=np.float32
        )

    def batch(self, size: int):
        x = np.zeros((size, self.lookback), dtype=np.float32)
        mask = np.zeros_like(x)
        y = np.zeros((size, self.horizon), dtype=np.float32)
        y_mask = np.zeros_like(y)
        ids = self.rng.integers(len(self.series), size=size)
        for row, i in enumerate(ids):
            s = self.series[i]
            low = max(1, len(s) - self.history)
            cut = int(self.rng.integers(low, len(s)))  # at least one future point exists
            x[row], mask[row] = windows(s, cut, self.lookback)
            future = s[cut: cut + self.horizon]
            y[row, : len(future)] = future
            y_mask[row, : len(future)] = 1.0
        return x, mask, y, y_mask, self.mase_scales[ids]


def loss_fn(name: str, forecast, target, target_mask, mase_scale):
    if name == "smape":
        denominator = forecast.abs() + target.abs()
        ratio = torch.where(denominator > 0, (forecast - target).abs() / denominator.clamp_min(1e-8), torch.zeros_like(forecast))
        return 200 * (ratio * target_mask).sum() / target_mask.sum()
    if name == "mase":
        return (((forecast - target).abs() / mase_scale[:, None]) * target_mask).sum() / target_mask.sum()
    raise ValueError(name)


def build_model(config: TrainConfig, lookback: int, horizon: int):
    from deepforecast import nbeats

    if config.architecture == "generic":
        return nbeats.generic(lookback, horizon, blocks=config.blocks, width=config.width)
    if config.architecture == "interpretable":
        return nbeats.interpretable(lookback, horizon, width=config.width)
    raise ValueError(config.architecture)


def predict(model, histories: list[np.ndarray], lookback: int) -> np.ndarray:
    pairs = [windows(s, len(s), lookback) for s in histories]
    x = np.stack([p[0] for p in pairs])
    mask = np.stack([p[1] for p in pairs])
    peak = scale(x, mask)
    model.eval()
    with torch.no_grad():
        forecast = model(torch.from_numpy(x / peak), torch.from_numpy(mask)).numpy()
    model.train()
    return np.maximum(forecast * peak, 0.0)  # M4 series are positive


def train(
    config: TrainConfig,
    series: list[np.ndarray],
    horizon: int,
    season: int,
    steps: int,
    validation: tuple[list[np.ndarray], np.ndarray] | None = None,
    log=print,
):
    """Train for ``steps``; with ``validation`` return the best step by held-out sMAPE."""
    torch.manual_seed(config.seed)
    lookback = config.lookback_multiple * horizon
    sampler = Sampler(series, lookback, horizon, int(config.history_multiple * horizon), season, config.seed)
    model = build_model(config, lookback, horizon)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    curve = []
    best = (np.inf, steps)
    best_state = None
    for step in range(1, steps + 1):
        x, mask, y, y_mask, mase_scale = sampler.batch(config.batch_size)
        peak = scale(x, mask)
        forecast = model(torch.from_numpy(x / peak), torch.from_numpy(mask))
        loss = loss_fn(
            config.loss, forecast, torch.from_numpy(y / peak), torch.from_numpy(y_mask),
            torch.from_numpy(mase_scale / peak[:, 0]),
        )
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if validation is not None and (step % config.eval_every == 0 or step == steps):
            histories, targets = validation
            score = float(smape(targets, predict(model, histories, lookback)).mean())
            curve.append({"step": step, "train_loss": float(loss.detach()), "validation_smape": score})
            if score < best[0]:
                best = (score, step)
                best_state = copy.deepcopy(model.state_dict())
            log(f"  step {step:5d}  loss {loss.item():8.3f}  validation sMAPE {score:7.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, lookback, best[1], curve
