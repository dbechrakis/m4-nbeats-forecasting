# N-BEATS vs the M4 Benchmarks — Global Deep Learning Forecasting in PyTorch

A from-scratch PyTorch implementation of **N-BEATS** (Oreshkin et al., ICLR 2020), with generic and interpretable trend/seasonality architectures. It is trained as one global model per frequency on the **M4 competition** Hourly (414 series), Daily (4,227) and Weekly (359) data, and scored against the competition's statistical benchmarks with the official sMAPE, MASE and OWA.

> Results are being produced. This README will carry the measured tables once the ensemble runs finish.

**Stack:** Python · PyTorch · NumPy · pandas · Matplotlib · GitHub Actions

## Protocol

- **Data:** the official M4 train/test files, SHA-256 pinned ([`data.py`](src/deepforecast/data.py)).
- **Benchmarks:** Naive, Seasonal naive, Naive2, SES and Theta, following the M4 reference R code. That code applies a 90% autocorrelation seasonality test and, if it passes, classical multiplicative decomposition.
- **No test leakage:**
  - The last horizon of every training series is held out as validation.
  - Each network keeps the weights with the best validation sMAPE.
  - The official test horizon is forecast once.
- **Ensemble:** 6 networks (generic and interpretable × lookbacks of 2, 3 and 5 horizons), combined by the median.
- **Scale-free training:** each window is divided by the maximum of its lookback, so one network serves series whose levels differ by orders of magnitude.

## Run

```bash
python -m pip install -r requirements.txt
make data       # pinned M4 downloads
make train      # ensemble members per frequency (CPU: about 1-2 hours in total)
make evaluate   # metrics, OWA, bootstrap intervals, figures
make check      # lint + tests + evidence check
```
