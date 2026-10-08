"""Fingerprinted M4 downloads and loading into per-series arrays."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


SOURCE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
SHA256 = {
    "Train/Hourly-train.csv": "ea59b7783573c49077a835ab6465c7d66f1474783360f310988a9a737fbca62f",
    "Test/Hourly-test.csv": "71a57fccb15e534d973626ada4fb87febf789e9317715b7cc6dd3b5f90db6f42",
    "Train/Weekly-train.csv": "d478d3f6ed673e6ed3c2cb0fbed5425f72eedd51318ebebeb0b5ecfcefc37714",
    "Test/Weekly-test.csv": "3b2bc4be8e636e802260dcd843ca3c5da40536a1fbe1ecb3229a97dae4f44688",
    "Train/Daily-train.csv": "78e94591c60c06309f1e544fd7b2ccba28f3616b01913dd336a1d1d98483a1ec",
    "Test/Daily-test.csv": "633b0815626ae266d36872c4d53e95eb93ee8d43e79587fceec519eba3e6b05e",
}
# Official M4 settings: forecast horizon and the seasonal period used by MASE and Naive2.
FREQUENCIES = {
    "Hourly": {"horizon": 48, "season": 24},
    "Daily": {"horizon": 14, "season": 1},
    "Weekly": {"horizon": 13, "season": 1},
}


@dataclass
class Dataset:
    frequency: str
    ids: list[str]
    train: list[np.ndarray]
    test: np.ndarray  # (series, horizon)
    horizon: int
    season: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(frequency: str, target: Path) -> None:
    """Download and verify the train/test files for one frequency."""
    for split in ("Train", "Test"):
        name = f"{split}/{frequency}-{split.lower()}.csv"
        path = target / name
        if path.exists() and sha256(path) == SHA256[name]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urlopen(f"{SOURCE}/{name}", timeout=300).read())
        if sha256(path) != SHA256[name]:
            raise ValueError(f"{name} does not match its pinned SHA-256")


def load(frequency: str, root: Path) -> Dataset:
    settings = FREQUENCIES[frequency]
    train_frame = pd.read_csv(root / f"Train/{frequency}-train.csv", index_col=0)
    test_frame = pd.read_csv(root / f"Test/{frequency}-test.csv", index_col=0)
    if not train_frame.index.equals(test_frame.index):
        raise ValueError("Train and test series ids differ")
    train = [row.dropna().to_numpy(dtype=float) for _, row in train_frame.iterrows()]
    test = test_frame.to_numpy(dtype=float)
    if test.shape[1] != settings["horizon"] or np.isnan(test).any():
        raise ValueError("Unexpected test horizon")
    if any((series <= 0).any() for series in train):
        raise ValueError("M4 series are expected to be positive")
    return Dataset(frequency, list(train_frame.index), train, test, settings["horizon"], settings["season"])


def validation_split(dataset: Dataset) -> tuple[list[np.ndarray], np.ndarray]:
    """Hold out the last horizon of every training series for model selection.

    The official test values are never seen while choosing epochs or ensembles.
    """
    h = dataset.horizon
    return [series[:-h] for series in dataset.train], np.stack([series[-h:] for series in dataset.train])
