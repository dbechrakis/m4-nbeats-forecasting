"""N-BEATS (Oreshkin et al., 2020) in PyTorch.

Each block reads the residual of the lookback window, emits a backcast that is subtracted
from it and a forecast that is added to the output. Generic blocks learn their basis; the
interpretable variant constrains blocks to polynomial trend and Fourier seasonality bases.
"""

import numpy as np
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, lookback: int, horizon: int, width: int, layers: int, theta_size: int, basis: nn.Module):
        super().__init__()
        stack = [nn.Linear(lookback, width), nn.ReLU()]
        for _ in range(layers - 1):
            stack += [nn.Linear(width, width), nn.ReLU()]
        self.fc = nn.Sequential(*stack)
        self.theta = nn.Linear(width, theta_size)
        self.basis = basis

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.basis(self.theta(self.fc(x)))


class GenericBasis(nn.Module):
    def __init__(self, lookback: int, horizon: int):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return theta[:, : self.lookback], theta[:, -self.horizon:]


class TrendBasis(nn.Module):
    """Polynomial of degree ``degree`` over normalised time."""

    def __init__(self, degree: int, lookback: int, horizon: int):
        super().__init__()
        self.terms = degree + 1
        back = np.arange(lookback) / lookback
        fore = np.arange(horizon) / horizon
        self.register_buffer("back", torch.tensor(np.stack([back**i for i in range(self.terms)]), dtype=torch.float32))
        self.register_buffer("fore", torch.tensor(np.stack([fore**i for i in range(self.terms)]), dtype=torch.float32))

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return theta[:, : self.terms] @ self.back, theta[:, self.terms:] @ self.fore


class SeasonalityBasis(nn.Module):
    """Fourier terms up to ``harmonics`` cycles per horizon."""

    def __init__(self, harmonics: int, lookback: int, horizon: int):
        super().__init__()
        frequencies = np.arange(1, harmonics * horizon // 2 + 1) / harmonics

        def basis(length: int) -> np.ndarray:
            grid = 2 * np.pi * np.arange(length)[None, :] / horizon * frequencies[:, None]
            return np.concatenate([np.ones((1, length)), np.cos(grid), np.sin(grid)])

        self.register_buffer("back", torch.tensor(basis(lookback), dtype=torch.float32))
        self.register_buffer("fore", torch.tensor(basis(horizon), dtype=torch.float32))
        self.size = self.back.shape[0]

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return theta[:, : self.size] @ self.back, theta[:, self.size:] @ self.fore


class NBeats(nn.Module):
    def __init__(self, blocks: list[Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x.flip(dims=(1,))
        mask = mask.flip(dims=(1,))
        forecast = x[:, -1:]  # level of the last observation, as in the reference code
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = (residual - backcast) * mask
            forecast = forecast + block_forecast
        return forecast

    def stacks_forecast(self, x: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        """Per-block forecasts, used to plot the interpretable trend/seasonality split."""
        residual, mask = x.flip(dims=(1,)), mask.flip(dims=(1,))
        parts = []
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = (residual - backcast) * mask
            parts.append(block_forecast)
        return parts


def generic(lookback: int, horizon: int, blocks: int = 12, layers: int = 4, width: int = 256) -> NBeats:
    return NBeats(
        [Block(lookback, horizon, width, layers, lookback + horizon, GenericBasis(lookback, horizon)) for _ in range(blocks)]
    )


def interpretable(lookback: int, horizon: int, width: int = 256, trend_blocks: int = 3, season_blocks: int = 3) -> NBeats:
    trend = TrendBasis(2, lookback, horizon)
    season = SeasonalityBasis(1, lookback, horizon)
    return NBeats(
        [Block(lookback, horizon, width, 4, 2 * trend.terms, trend) for _ in range(trend_blocks)]
        + [Block(lookback, horizon, width * 2, 4, 2 * season.size, season) for _ in range(season_blocks)]
    )
