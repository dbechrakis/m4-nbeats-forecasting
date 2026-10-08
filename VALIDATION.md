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
