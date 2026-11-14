# Validation record

## First full run — 2026-10-08

- **Data.** M4 Hourly, Daily and Weekly train/test files were downloaded from the M4-methods repository and verified against the SHA-256 values pinned in `src/deepforecast/data.py`.
- **Benchmarks.** They were computed on all 414 / 4,227 / 359 series. Naive2 sMAPE / MASE: Hourly 18.383 / 2.395, Daily 3.045 / 3.278, Weekly 9.161 / 2.777.
- **Training.**
  - 18 networks: 3 frequencies × (generic, interpretable) × lookbacks of 2, 3 and 5 horizons.
  - Run on a 4-core CPU with PyTorch 2.14.1; each took 221–789 s.
  - Each trained on the training series minus their last horizon and kept its best-validation weights.
  - The test horizon was scored once, after training.
- **Results.** Ensemble OWA: Hourly 0.442, Weekly 0.777, Daily 1.001.
  - The 95% bootstrap intervals for the gain over the best benchmark (2,000 resamples of series) exclude zero for Hourly and Weekly and include zero for Daily.
  - `ci/verify_evidence.py` recomputes every OWA from the committed sMAPE/MASE, and checks member counts and validation curves.
- **Tests.**
  - 14 unit tests cover the metric definitions, the seasonality test, Naive2/Theta behaviour, window padding and masking, and target placement (no look-ahead in sampled windows).
  - They also check that the interpretable parts sum to the forecast and that a small network learns a seasonal pattern.
  - CI also smoke-trains on real M4 Weekly data.
- **Not done.** No GPU-scale ensemble, no reproduction of the paper's exact numbers, and no comparison with the M4 competitors' submissions.

## The full M4 benchmark set, 2026-11-09

Holt, Damped and Comb were added to `deepforecast.baselines`, applied to the deseasonalised series as in the M4 reference code and fitted by grid search (alpha, beta as a share of alpha, and phi for Damped). Tests check that Holt continues an exact line exactly, that a simulated Holt series returns its smoothing parameters, that Damped flattens, and that Comb is the mean of its parts. `scripts/evaluate.py` now computes only benchmarks missing from each `benchmarks.npz`, and the five existing benchmark forecasts were confirmed byte-identical afterwards. The N-BEATS members were not retrained.

The best benchmark changed on two frequencies. Weekly: Damped (OWA 0.919) instead of Theta (0.969), so the N-BEATS gain falls from 20% to 15% of OWA, with a bootstrap interval of 0.08 to 0.21 that still excludes zero. Daily: Comb (0.985), which beats every N-BEATS variant. Hourly is unchanged, because Holt, Damped and Comb do poorly on hourly cycles (1.15 to 2.98).

## Tests for the network's building blocks — 2026-11-14

- Added `tests/test_basis.py`. The trend basis reproduces a known quadratic on the lookback and the forecast side. The seasonality basis reproduces a cosine and a sine with a period of one horizon, and a constant coefficient gives a flat term. The generic basis takes the first lookback and last horizon coefficients. After the validation split, no input or target value that the sampler draws is a value of the held-out horizon. The median of members ignores one member that is far off.
- 23 tests pass with the pinned requirements on Python 3.12, and the repository's lint passes. Nothing was retrained and no result changed.
