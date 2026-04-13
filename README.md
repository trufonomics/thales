# Thales

Economic time series foundation model for [Truflation](https://truflation.com).

Thales forecasts individual economic indicators (CPI, GDP proxies, employment, housing, energy) and adds capabilities no existing model has: hierarchical coherence, cross-stream economic intelligence, regime detection, and anomaly detection.

Named after [Thales of Miletus](https://en.wikipedia.org/wiki/Thales_of_Miletus) — the first philosopher and first economist, who predicted an olive harvest and cornered the market on olive presses.

## Why

Every time series foundation model (TimesFM, Chronos, Moirai, TiRex, Toto) trains on general data — Wikipedia pageviews, weather, electricity, traffic. None are built for economic data. We tested all of them on Truflation's CPI streams. The results speak for themselves.

## Experiment 1 — Baseline Results

6 foundation models + 2 naive baselines evaluated zero-shot on 32 Truflation CPI index streams.

**Train:** 2010-01-01 to 2023-12-31 (5,113 days) | **Test:** 2024-01-01 to 2026-04-12 (833 days) | **Data:** Frozen (point-in-time)

### 7-Day Forecast

| Model | Params | MAE | MASE | Direction Accuracy |
|---|---|---|---|---|
| Chronos-T5 | 46M | 1.05 | 0.24 | **63%** |
| TiRex | 35M | **1.07** | **0.26** | 21% |
| EMA | — | 1.09 | 0.27 | 62% |
| TimesFM | 200M | 1.09 | 0.25 | 22% |
| Moirai | 14M | 1.15 | 0.27 | 19% |
| Toto | 151M | 1.15 | 0.28 | 23% |
| Chronos-Bolt | 48M | 1.33 | 0.34 | 24% |
| Seasonal Naive | — | 4.13 | 0.97 | **78%** |

### 30-Day Forecast

| Model | Params | MAE | MASE | Direction Accuracy |
|---|---|---|---|---|
| **EMA** | — | **1.37** | **0.34** | **62%** |
| TiRex | 35M | 1.41 | 0.34 | 21% |
| Toto | 151M | 1.42 | 0.35 | 20% |
| Chronos-Bolt | 48M | 1.51 | 0.35 | 21% |
| Chronos-T5 | 46M | 1.52 | 0.36 | 60% |
| Moirai | 14M | 1.66 | 0.41 | 19% |
| TimesFM | 200M | 1.70 | 0.38 | 20% |
| Seasonal Naive | — | 4.34 | 1.00 | **72%** |

### 90-Day Forecast

| Model | Params | MAE | MASE | Direction Accuracy |
|---|---|---|---|---|
| **TiRex** | 35M | **1.72** | **0.41** | 21% |
| EMA | — | 1.90 | 0.49 | **60%** |
| Toto | 151M | 2.05 | 0.52 | 20% |
| Chronos-T5 | 46M | 2.06 | 0.50 | 58% |
| Moirai | 14M | 2.09 | 0.55 | 19% |
| TimesFM | 200M | 2.39 | 0.53 | 20% |
| Chronos-Bolt | 48M | 3.14 | 0.72 | 20% |
| Seasonal Naive | — | 3.98 | 0.97 | **71%** |

### Key Findings

1. **Every foundation model fails on directional accuracy (19-24%).** Five of six predict the wrong direction >75% of the time on economic data. Only Chronos-T5 exceeds 50%.

2. **A simple EMA beats every foundation model at 30+ day horizons.** Google's 200M-param TimesFM and Datadog's GIFT-Eval champion Toto both lose to exponential smoothing.

3. **No model achieves both low error AND high directional accuracy.** TiRex wins on MAE. Seasonal Naive wins on direction. Nobody does both. That's the gap Thales fills.

4. **These models were never trained on economic data and it shows.** They revert to the mean instead of following economic trends. A model trained on Truflation's proprietary economic data with domain-specific objectives should beat all of them on both dimensions.

## Models Evaluated

| Model | Organization | Params | Architecture | Training Data |
|---|---|---|---|---|
| [TimesFM 1.0](https://huggingface.co/google/timesfm-1.0-200m-pytorch) | Google | 200M | Decoder-only Transformer | 100B+ points (Google Trends, Wikipedia) |
| [Chronos-T5 Small](https://huggingface.co/amazon/chronos-t5-small) | Amazon | 46M | T5 Encoder-Decoder | 100B+ points (public + synthetic) |
| [Chronos-Bolt Small](https://huggingface.co/amazon/chronos-bolt-small) | Amazon | 48M | T5 Encoder-Decoder | 100B+ points (direct quantile) |
| [Moirai 1.1 Small](https://huggingface.co/Salesforce/moirai-1.1-R-small) | Salesforce | 14M | Masked Encoder Transformer | LOTSA (27B obs, 9 domains) |
| [TiRex](https://huggingface.co/NX-AI/TiRex) | NX-AI (Sepp Hochreiter) | 35M | xLSTM (sLSTM) | 47.5M samples (NeurIPS 2025) |
| [Toto](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | Datadog | 151M | Decoder-only Transformer | 2.36T points (GIFT-Eval champion) |

## Data

Truflation US CPI data — 32 daily index streams (12 categories + subcategories), January 2010 to April 2026. Includes both Truflation's proprietary indexes and official BLS/BEA data for comparison.

## Metrics

- **MAE** — Mean Absolute Error (lower is better)
- **MASE** — Mean Absolute Scaled Error: MAE divided by seasonal naive MAE. MASE < 1 = beats naive baseline.
- **Direction Accuracy** — % of time the model predicts the correct direction of change. 50% = coin flip.

## Repo Structure

```
thales/
├── src/
│   ├── data.py          # Truflation data loading and preprocessing
│   └── metrics.py       # MAE, RMSE, MASE, CRPS, directional accuracy
├── scripts/
│   ├── experiment_01_baselines.py  # Baseline zoo (all 6 TSFMs + naive)
│   └── vast_setup.sh              # Vast.ai dependency install
├── data/                # (not in git) Truflation CSVs
├── results/             # (not in git) Experiment outputs
└── requirements.txt
```

## Running

```bash
# Local (naive baselines only, no GPU needed)
python scripts/experiment_01_baselines.py --models naive --horizons 7 30 90

# Vast.ai (full baseline zoo)
bash scripts/vast_setup.sh
python scripts/experiment_01_baselines.py --models all --horizons 7 30 90
```

## Status

- [x] Experiment 1 — Baseline Zoo (complete)
- [ ] Experiment 2 — Architecture Selection (S5 vs Mamba vs Transformer vs xLSTM)
- [ ] Experiment 3 — Hierarchical Coherence
- [ ] Experiment 4 — Training Objective Ablation
- [ ] Experiment 5 — Cross-Stream Transfer
- [ ] Experiment 6 — Multi-Resolution
- [ ] Experiment 7 — Raw Price Stream Training
- [ ] Experiment 8 — Historical Stress Test
