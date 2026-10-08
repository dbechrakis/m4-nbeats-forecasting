"""Train the N-BEATS ensemble members for one M4 frequency and save their forecasts.

    python scripts/train_nbeats.py --frequency Hourly --steps 4000

Each member is saved as soon as it finishes, so an interrupted run resumes where it stopped.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from deepforecast.data import load, validation_split
from deepforecast.metrics import smape
from deepforecast.training import TrainConfig, predict, train


ROOT = Path(__file__).resolve().parents[1]
MEMBERS = [
    {"architecture": architecture, "lookback_multiple": lookback, "seed": seed}
    for seed, (architecture, lookback) in enumerate(
        [(a, m) for a in ("generic", "interpretable") for m in (2, 3, 5)]
    )
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", required=True, choices=["Hourly", "Daily", "Weekly"])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--members", type=int, default=len(MEMBERS))
    parser.add_argument("--subset", type=int, default=None, help="Use the first N series (smoke tests)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    dataset = load(args.frequency, ROOT / "data" / "raw")
    if args.subset:
        dataset.train, dataset.test, dataset.ids = dataset.train[: args.subset], dataset.test[: args.subset], dataset.ids[: args.subset]
    validation = validation_split(dataset)
    out = args.output or ROOT / "outputs" / args.frequency.lower() / "members"
    out.mkdir(parents=True, exist_ok=True)

    for spec in MEMBERS[: args.members]:
        name = f"{spec['architecture']}-L{spec['lookback_multiple']}-s{spec['seed']}"
        if (out / f"{name}.npz").exists():
            print(f"{name}: done already")
            continue
        config = TrainConfig(**spec, eval_every=args.eval_every)
        print(f"{name}: training {args.steps} steps", flush=True)
        started = time.perf_counter()
        model, lookback, best_step, curve = train(
            config, validation[0], dataset.horizon, dataset.season, args.steps, validation,
            log=lambda line: print(line, flush=True),
        )
        validation_forecast = predict(model, validation[0], lookback)
        test_forecast = predict(model, dataset.train, lookback)
        np.savez_compressed(out / f"{name}.npz", validation=validation_forecast, test=test_forecast)
        (out / f"{name}.json").write_text(json.dumps({
            "config": asdict(config), "best_step": best_step, "curve": curve,
            "validation_smape": float(smape(validation[1], validation_forecast).mean()),
            "train_seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
        }, indent=2) + "\n")
        print(f"{name}: best step {best_step}, {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
