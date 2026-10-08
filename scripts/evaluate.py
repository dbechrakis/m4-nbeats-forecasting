"""Score the N-BEATS ensemble against the M4 benchmarks on the official test horizon.

    python scripts/evaluate.py --frequency Hourly

Writes outputs/<frequency>/metrics.csv, summary.json and a forecast figure.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from deepforecast.baselines import BENCHMARKS  # noqa: E402
from deepforecast.data import load, validation_split  # noqa: E402
from deepforecast.metrics import mase, owa, smape  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def benchmark_forecasts(dataset, out: Path) -> dict[str, np.ndarray]:
    cache = out / "benchmarks.npz"
    if cache.exists():
        return dict(np.load(cache))
    forecasts = {
        name: np.stack([method(series, dataset.horizon, dataset.season) for series in dataset.train])
        for name, method in BENCHMARKS.items()
    }
    np.savez_compressed(cache, **forecasts)
    return forecasts


def bootstrap_owa_gain(per_series: dict, model: str, rival: str, naive2: tuple[float, float], draws: int = 2000) -> tuple[float, float]:
    """95% interval of OWA(rival) - OWA(model), resampling series."""
    rng = np.random.default_rng(0)
    n = len(per_series[model][0])
    gains = []
    for _ in range(draws):
        idx = rng.integers(n, size=n)
        model_owa = owa(per_series[model][0][idx].mean(), per_series[model][1][idx].mean(), *naive2)
        rival_owa = owa(per_series[rival][0][idx].mean(), per_series[rival][1][idx].mean(), *naive2)
        gains.append(rival_owa - model_owa)
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", required=True, choices=["Hourly", "Daily", "Weekly"])
    args = parser.parse_args()
    dataset = load(args.frequency, ROOT / "data" / "raw")
    out = ROOT / "outputs" / args.frequency.lower()
    _, validation_targets = validation_split(dataset)

    forecasts = benchmark_forecasts(dataset, out)
    members = sorted((out / "members").glob("*.npz"))
    if not members:
        raise FileNotFoundError("Train the ensemble first: scripts/train_nbeats.py")
    member_test = {m.stem: np.load(m)["test"] for m in members}
    member_validation = {m.stem: np.load(m)["validation"] for m in members}
    forecasts["N-BEATS ensemble (median)"] = np.median(np.stack(list(member_test.values())), axis=0)
    for architecture in ("generic", "interpretable"):
        chosen = [v for k, v in member_test.items() if k.startswith(architecture)]
        if chosen:
            forecasts[f"N-BEATS {architecture} only"] = np.median(np.stack(chosen), axis=0)

    per_series = {
        name: (smape(dataset.test, forecast), mase(dataset.test, forecast, dataset.train, dataset.season))
        for name, forecast in forecasts.items()
    }
    naive2 = (per_series["Naive2"][0].mean(), per_series["Naive2"][1].mean())
    rows = [
        {"method": name, "smape": s.mean(), "mase": m.mean(), "owa": owa(s.mean(), m.mean(), *naive2)}
        for name, (s, m) in per_series.items()
    ]
    table = pd.DataFrame(rows).sort_values("owa")
    table.to_csv(out / "metrics.csv", index=False, float_format="%.4f")

    ensemble = "N-BEATS ensemble (median)"
    best_benchmark = table[table["method"].isin(BENCHMARKS)].iloc[0]["method"]
    low, high = bootstrap_owa_gain(per_series, ensemble, best_benchmark, naive2)
    validation_ensemble = np.median(np.stack(list(member_validation.values())), axis=0)
    summary = {
        "frequency": args.frequency,
        "series": len(dataset.train),
        "horizon": dataset.horizon,
        "ensemble_members": len(members),
        "best_benchmark": best_benchmark,
        "owa_ensemble": float(table.set_index("method").loc[ensemble, "owa"]),
        "owa_best_benchmark": float(table.set_index("method").loc[best_benchmark, "owa"]),
        "owa_gain_vs_best_benchmark_95ci": [low, high],
        "share_of_series_ensemble_beats_best_benchmark_smape": float(
            (per_series[ensemble][0] < per_series[best_benchmark][0]).mean()
        ),
        "validation_smape_ensemble": float(smape(validation_targets, validation_ensemble).mean()),
        "member_validation_smape": {
            k: float(smape(validation_targets, v).mean()) for k, v in member_validation.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Forecasts for the series where the ensemble's sMAPE is at the 25th, 50th and 75th percentile.
    order = np.argsort(per_series[ensemble][0])
    picks = [order[int(q * (len(order) - 1))] for q in (0.25, 0.5, 0.75)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 8))
    for ax, i in zip(axes, picks):
        history = dataset.train[i][-4 * dataset.horizon:]
        t_hist = np.arange(len(history))
        t_test = np.arange(len(history), len(history) + dataset.horizon)
        ax.plot(t_hist, history, color="black", linewidth=1, label="History")
        ax.plot(t_test, dataset.test[i], color="black", linestyle="--", linewidth=1, label="Actual")
        ax.plot(t_test, forecasts[ensemble][i], color="#1f77b4", label=f"N-BEATS (sMAPE {per_series[ensemble][0][i]:.1f})")
        ax.plot(t_test, forecasts[best_benchmark][i], color="#ff7f0e", label=f"{best_benchmark} (sMAPE {per_series[best_benchmark][0][i]:.1f})")
        ax.set_title(f"{dataset.ids[i]}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
    fig.suptitle(f"M4 {args.frequency}: series at the 25th / 50th / 75th percentile of N-BEATS error")
    fig.tight_layout()
    fig.savefig(out / "example_forecasts.png", dpi=130)
    plt.close(fig)
    print(table.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "member_validation_smape"}, indent=2))


if __name__ == "__main__":
    main()
