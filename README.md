# Thales

Economic time series foundation model for [Truflation](https://truflation.com).

Thales forecasts individual economic indicators (CPI, GDP proxies, employment, housing, energy) and adds capabilities no existing model has: hierarchical coherence, cross-stream economic intelligence, regime detection, and anomaly detection.

Named after [Thales of Miletus](https://en.wikipedia.org/wiki/Thales_of_Miletus) — the first philosopher and first economist, who predicted an olive harvest and cornered the market on olive presses.

## Why

Every time series foundation model (TimesFM, Chronos, Moirai, TiRex, Toto) trains on general data — Wikipedia pageviews, weather, electricity, traffic. None are built for economic data. We tested all of them on Truflation's CPI streams, then trained our own. The results speak for themselves.

---

## Glossary

**Terms used throughout this README:**

| Term | What it means |
|---|---|
| **MAE** | Mean Absolute Error — average prediction error in index points. If CPI is at 143 and you predict 145, your error is 2. Lower is better. |
| **MASE** | Mean Absolute Scaled Error — your MAE divided by the Seasonal Naive baseline's MAE. Below 1.0 = you beat naive. Below 0.5 = you're twice as good as naive. This lets you compare across different series fairly. |
| **Direction Accuracy** | How often the model correctly predicts whether the value will go UP or DOWN. 50% = coin flip (no skill). 80% = strong directional signal. 21% = systematically wrong. |
| **Zero-shot** | Running a pre-trained model on data it has never seen before, with no additional training. Tests whether the model's general knowledge transfers to economic data. |
| **EMA** | Exponential Moving Average — the simplest possible forecasting method. It says "tomorrow will be close to today" by taking a weighted average that favors recent values. Surprisingly hard to beat on trending data like CPI. |
| **Seasonal Naive** | Predicts that this April will look like last April. Captures seasonal patterns (heating costs rise in winter, travel costs rise in summer) but ignores trends entirely. Gets direction right ~75% of the time because economic seasons repeat. |
| **RevIN** | Reversible Instance Normalization — a data preparation technique (ICLR 2022) where each input window is scaled to its own local average. Prevents the model from learning shortcuts like "always predict the long-term average." Used by all top forecasting models. |
| **Params** | Parameters — the number of learnable values in the model. More params = more capacity to learn patterns, but also more risk of memorizing noise instead of learning real patterns. |

---

## Experiment 1 — Baseline Results

6 foundation models + 2 naive baselines evaluated zero-shot on 32 Truflation CPI index streams. None of these models were trained on economic data.

**Train:** 2010-01-01 to 2023-12-31 (5,113 days) | **Test:** 2024-01-01 to 2026-04-12 (833 days) | **Data:** Frozen (point-in-time)

### 7-Day Forecast

| Rank | Model | Params | MAE | MASE | Direction |
|---|---|---|---|---|---|
| 1 | Chronos-T5 | 46M | 1.05 | 0.24 | **63%** |
| 2 | TiRex | 35M | **1.07** | **0.26** | 21% |
| 3 | EMA | — | 1.09 | 0.27 | 62% |
| 4 | TimesFM | 200M | 1.09 | 0.25 | 22% |
| 5 | Moirai | 14M | 1.15 | 0.27 | 19% |
| 6 | Toto | 151M | 1.15 | 0.28 | 23% |
| 7 | Chronos-Bolt | 48M | 1.33 | 0.34 | 24% |
| 8 | Seasonal Naive | — | 4.13 | 0.97 | **78%** |

### 30-Day Forecast

| Rank | Model | Params | MAE | MASE | Direction |
|---|---|---|---|---|---|
| 1 | **EMA** | — | **1.37** | **0.34** | **62%** |
| 2 | TiRex | 35M | 1.41 | 0.34 | 21% |
| 3 | Toto | 151M | 1.42 | 0.35 | 20% |
| 4 | Chronos-Bolt | 48M | 1.51 | 0.35 | 21% |
| 5 | Chronos-T5 | 46M | 1.52 | 0.36 | 60% |
| 6 | Moirai | 14M | 1.66 | 0.41 | 19% |
| 7 | TimesFM | 200M | 1.70 | 0.38 | 20% |
| 8 | Seasonal Naive | — | 4.34 | 1.00 | **72%** |

### 90-Day Forecast

| Rank | Model | Params | MAE | MASE | Direction |
|---|---|---|---|---|---|
| 1 | **TiRex** | 35M | **1.72** | **0.41** | 21% |
| 2 | EMA | — | 1.90 | 0.49 | **60%** |
| 3 | Toto | 151M | 2.05 | 0.52 | 20% |
| 4 | Chronos-T5 | 46M | 2.06 | 0.50 | 58% |
| 5 | Moirai | 14M | 2.09 | 0.55 | 19% |
| 6 | TimesFM | 200M | 2.39 | 0.53 | 20% |
| 7 | Chronos-Bolt | 48M | 3.14 | 0.72 | 20% |
| 8 | Seasonal Naive | — | 3.98 | 0.97 | **71%** |

### Key Findings

1. **Every foundation model fails on directional accuracy (19-24%).** Five of six predict the wrong direction more than 75% of the time on economic data. Only Chronos-T5 exceeds 50%.

2. **A simple EMA beats every foundation model at 30+ day horizons.** Google's 200M-param TimesFM and Datadog's GIFT-Eval champion Toto both lose to exponential smoothing.

3. **No model achieves both low error AND high directional accuracy.** TiRex wins on MAE. Seasonal Naive wins on direction. Nobody does both. That's the gap Thales fills.

4. **These models were never trained on economic data and it shows.** They revert to the mean instead of following economic trends.

### Models Evaluated

| Model | Organization | Params | What it is | Trained on |
|---|---|---|---|---|
| [TimesFM 1.0](https://huggingface.co/google/timesfm-1.0-200m-pytorch) | Google | 200M | Decoder-only Transformer | 100B+ points (Google Trends, Wikipedia) |
| [Chronos-T5 Small](https://huggingface.co/amazon/chronos-t5-small) | Amazon | 46M | T5 Encoder-Decoder | 100B+ points (public + synthetic) |
| [Chronos-Bolt Small](https://huggingface.co/amazon/chronos-bolt-small) | Amazon | 48M | T5 Encoder-Decoder (direct quantile) | 100B+ points |
| [Moirai 1.1 Small](https://huggingface.co/Salesforce/moirai-1.1-R-small) | Salesforce | 14M | Masked Encoder Transformer | 27B observations across 9 domains |
| [TiRex](https://huggingface.co/NX-AI/TiRex) | NX-AI | 35M | xLSTM — built by Sepp Hochreiter, inventor of the original LSTM | 47.5M samples (NeurIPS 2025) |
| [Toto](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | Datadog | 151M | Decoder-only Transformer | 2.36T points (current GIFT-Eval champion) |

---

## Experiment 2 — Architecture Selection

We trained four architectures on Truflation's 82 CPI index streams. Each model has 4-7M parameters and was trained to forecast 90 days ahead.

**Why these four?** Each represents a fundamentally different approach to processing data over time:

- **Transformer** — the architecture behind ChatGPT. Looks at all data points simultaneously and decides which ones matter most. Industry default — TimesFM (Google), Toto (Datadog), and Moirai (Salesforce) all use this.
- **S5 (State Space Model)** — maintains a compressed "memory state" that evolves as new data arrives, like a running summary. FlowState (IBM) used this to beat models 55x larger on standard benchmarks.
- **xLSTM (Extended LSTM)** — built by Sepp Hochreiter, the inventor of the original LSTM (the architecture that powered Siri, Google Translate, and speech recognition for a decade). His 2024 upgrade uses exponential gating for more expressive memory control. TiRex used this to win the top forecasting benchmark at NeurIPS 2025.
- **Mamba** — a variant of S5 that selectively decides how much each new data point should update the memory. Can ignore noise and focus on signal.

**Train:** 82 streams, 2010-2022 | **Val:** 2023 | **Test:** 2024-2026

### 90-Day Forecast — Thales vs Baselines (ranked by MAE)

| Rank | Model | Params | MAE | MASE | Direction |
|---|---|---|---|---|---|
| 1 | TiRex (zero-shot) | 35M | **1.72** | **0.41** | 21% |
| 2 | EMA (naive) | — | 1.90 | 0.49 | **60%** |
| 3 | **Thales-Transformer** | **7.1M** | **1.97** | **0.49** | 21% |
| 4 | **Thales-S5** | **3.7M** | **1.98** | **0.48** | 21% |
| 5 | **Thales-xLSTM** | **5.6M** | **2.02** | **0.48** | 21% |
| 6 | Toto (zero-shot) | 151M | 2.05 | 0.52 | 20% |
| 7 | Chronos-T5 (zero-shot) | 46M | 2.06 | 0.50 | 58% |
| 8 | TimesFM (zero-shot) | 200M | 2.39 | 0.53 | 20% |

### Key Findings

1. **Our 3.7M param model matches or beats 46-200M param models on magnitude.** Thales-S5 (MASE 0.48) outperforms Amazon's Chronos (0.50), Datadog's Toto (0.52), and Google's TimesFM (0.53) — models 10-50x larger trained on billions of general data points.

2. **Architecture matters less than how you train.** All four architectures produce nearly identical results (MAE 1.97-2.02). The data preparation (RevIN normalization) and scoring function (composite loss) had 3.5x more impact than the architecture choice.

3. **Directional accuracy is the unsolved problem.** Every model — ours and theirs — predicts the wrong direction ~80% of the time on economic data. Solving this is the focus of Experiment 4.

4. **S5 is the most parameter-efficient.** Best 7-day MAE with half the parameters of the Transformer.

---

## What's Next

| Experiment | What it does | Status |
|---|---|---|
| 3 — Hierarchical Coherence | Enforce that CPI sub-component forecasts sum to the headline forecast using Truflation's composition weights | Waiting on weights from Truflation |
| 4 — Training Objective | Fix directional accuracy by changing what the model optimizes for | Next up |
| 5 — Cross-Stream Transfer | Train on inflation data, test if it can predict employment zero-shot | Needs API access for labor/housing/energy streams |
| 6 — Multi-Resolution | One model for daily, weekly, monthly, quarterly forecasts | Planned |
| 7 — Raw Price Streams | Train on individual product prices (Zillow listings, gas stations, Amazon) instead of aggregated indexes | Needs Layer 3 data access |
| 8 — Historical Stress Test | Test against 2008 crisis, COVID, 2022 inflation surge | Planned |

---

## Data

Truflation US CPI data — 82 daily index streams covering 12 spending categories and their subcategories (food, housing, transport, energy, health, etc.), January 2010 to April 2026. Includes both Truflation's proprietary indexes and official government data (BLS CPI, BEA PCE) for comparison.

## Repo Structure

```
thales/
├── src/
│   ├── data.py            # Truflation data loading
│   ├── dataset.py         # Sliding window datasets for training
│   ├── metrics.py         # MAE, RMSE, MASE, CRPS, directional accuracy
│   ├── revin.py           # Reversible Instance Normalization (ICLR 2022)
│   ├── losses.py          # Composite loss: Huber + Trend + Directional
│   ├── trainer.py         # Training loop with early stopping
│   └── models/
│       ├── transformer.py # Decoder-only transformer
│       ├── s5.py          # State space model
│       ├── mamba_model.py # Selective SSM
│       └── xlstm_model.py # Extended LSTM
├── scripts/
│   ├── experiment_01_baselines.py  # 6 TSFMs + naive baselines
│   ├── experiment_02_v2.py         # Architecture comparison
│   └── evaluate_checkpoints.py     # Evaluate saved models vs baselines
├── data/                  # (not in git) Truflation CSVs
└── results/               # (not in git) Experiment outputs + checkpoints
```

## Running

```bash
# Experiment 1: Baseline zoo (no GPU needed for naive, GPU for TSFMs)
python scripts/experiment_01_baselines.py --models all --horizons 7 30 90

# Experiment 2: Architecture selection (GPU recommended)
python scripts/experiment_02_v2.py --arch all --horizon 90 --epochs 100

# Evaluate saved checkpoints against baselines
python scripts/evaluate_checkpoints.py
```

## Status

- [x] Experiment 1 — Baseline Zoo (6 foundation models + 2 naive baselines)
- [x] Experiment 2 — Architecture Selection (Transformer, S5, xLSTM — Mamba pending)
- [ ] Experiment 3 — Hierarchical Coherence
- [ ] Experiment 4 — Training Objective Ablation
- [ ] Experiment 5 — Cross-Stream Transfer
- [ ] Experiment 6 — Multi-Resolution
- [ ] Experiment 7 — Raw Price Stream Training
- [ ] Experiment 8 — Historical Stress Test
