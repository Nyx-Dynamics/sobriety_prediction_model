# Sobriety Prediction Model

**Predicting relapse in stimulant and polysubstance use disorder using longitudinal clinical data and deep learning.**

Nyx Dynamics LLC | MIT MicroMasters in Statistics and Data Science — Capstone Project

## Overview

This project builds a survival-aware LSTM model to predict time-to-relapse among patients with stimulant use disorder (StUD) and polysubstance use disorder (PSUD). The pipeline generates a realistic synthetic longitudinal cohort informed by real EHR feature spaces, processes it into model-ready tensors, and provides a full suite of survival analysis diagnostics.

### Key Findings (Data Pipeline Phase)

| Metric | Value |
|---|---|
| Cohort size | 2,000 patients, 48,162 panel observations |
| Follow-up | Up to 34 months (30-day windows) |
| Event rate | 84.9% confirmed relapse |
| Median time to relapse | 276 days (stimulant), 354 days (polysubstance) |
| Log-rank test (StUD vs PSUD) | p = 0.222 (not significant) |
| Hartigan's dip test | D = 0.086, p < 0.0001 — **bimodal** (early vs. late relapsers) |

## Data Pipeline

The pipeline is fully reproducible from the `src/` scripts, run in order:

```
1. generate_synthetic_cohort.py   → Raw cohort (2,000 patients × 12 quarters)
2. build_static_matrix.py         → Patient-level features (30 cols)
3. build_panel.py                 → 30-day windowed longitudinal panel (48,162 rows × 19 cols)
4. build_outcomes.py              → Survival outcomes + KM analysis + dip test
5. build_lstm_tensors.py          → (N, T, F) = (2000, 34, 40) padded tensors
6. build_splits.py                → Stratified 70/15/15 splits with balance verification
```

### Feature Space (40 features)

**Time-varying (13):** active depression, medication adherence, employment, adverse events (none/mild/severe), positive events, PHQ-9, GAD-7, sobriety status, UDS result, meetings/month, social support, sparse window flag, cumulative stress

**Static (27):** SUD type, severity (0-10), prior treatment episodes, housing stability, social support baseline, age, sex, race/ethnicity, route of administration, treatment modality, insurance, age of first use, 8 MH diagnosis flags (MDD, PTSD, GAD, ADHD, bipolar I/II, psychosis, BPD), 7 medical comorbidities (HCV, HIV, chronic pain, TBI, liver disease, cardiovascular, MS/autoimmune)

### Cumulative Stress Model

A rolling weighted stress score captures temporal accumulation of adversity:

```
stress_t = 0.8 × stress_{t-1} + severity_weight(adverse_event_t) - 0.3 × positive_event_t
```

Severity weights: none = 0, mild = 0.5, severe = 1.5. Clipped to [0, 10].

## Project Structure

```
sobriety_prediction_model/
├── src/                          # Pipeline scripts (run in order)
│   ├── generate_synthetic_cohort.py
│   ├── build_static_matrix.py
│   ├── build_panel.py
│   ├── build_outcomes.py
│   ├── build_lstm_tensors.py
│   └── build_splits.py
├── data/
│   ├── raw/                      # Full merged cohort CSV
│   └── processed/
│       ├── static/               # sud_static.csv (patient-level)
│       ├── panel/                # sud_panel.csv (30-day windows)
│       ├── outcomes/             # sud_outcomes.csv (survival)
│       └── lstm_sequences/       # .npy tensors, scaler, split indices
├── models/                       # Trained model checkpoints
├── outputs/figures/              # Diagnostic plots
├── notebooks/                    # EDA and analysis notebooks
├── requirements.txt
└── README.md
```

## Splits

| Split | N | Event Rate | Polysubstance % | MH High % |
|---|---|---|---|---|
| Train | 1,400 (70%) | 0.849 | 52.7% | 67.8% |
| Validation | 300 (15%) | 0.850 | 52.7% | 67.7% |
| Test | 300 (15%) | 0.847 | 52.7% | 68.0% |

Stratified on event × SUD type × (n_mh_diagnoses ≥ 2). Maximum cross-split difference: 0.3%.

StandardScaler fitted exclusively on training data (1,400 patients).

## Diagnostic Figures

**MH Co-occurrence Matrix** — Highest overlap: MDD + PTSD (20.3% of cohort)

**Kaplan-Meier Curves** — Stratified by SUD type with log-rank test

**Event Time Distribution** — Bimodal: early relapsers (~90-180 days) and late relapsers (~500-600 days)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.12+.

## Reproducing the Data

```bash
cd src
python generate_synthetic_cohort.py
python build_static_matrix.py
python build_panel.py
python build_outcomes.py
python build_lstm_tensors.py
python build_splits.py
```

All scripts use `np.random.seed(42)` for reproducibility.

## Roadmap

- [ ] LSTM survival model (DeepSurv / DRSA architecture)
- [ ] Cox proportional hazards baseline
- [ ] Bayesian mixture model for early vs. late relapse subgroups
- [ ] SHAP feature importance analysis
- [ ] Sensitivity analysis on synthetic data assumptions

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```
Demidont, A.C. (2026). Sobriety Prediction Model for Stimulant and Polysubstance
Use Disorder. Nyx Dynamics LLC / MIT MicroMasters Capstone.
```
