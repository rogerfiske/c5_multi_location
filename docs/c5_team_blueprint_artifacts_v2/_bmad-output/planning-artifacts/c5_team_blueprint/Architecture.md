# Architecture — C5 Multi-Location Parts Forecasting (Research)

## 1. System Overview
A local research pipeline that ingests state matrices, constructs features (including cross-state pressure), trains forecasting models, backtests, and generates next-day ranked forecasts in two formats:
- **Pool**: ranked parts list
- **Sets**: ranked candidate 5-part combinations

## 2. Repository Structure (Target)
- `data/raw/` (source CSVs)
- `data/processed/` (cleaned, aligned, feature-ready tables)
- `src/`
  - `ingest/` (parsing, alignment, validation)
  - `features/` (pressure features, lags, seasonality, location transforms)
  - `models/` (baselines, ML models, calibration, ensembling)
  - `backtest/` (walk-forward evaluation)
  - `predict/` (next-day inference + outputs)
  - `sets/` (candidate set generation + scoring + optional assignment)
  - `utils/` (config, logging, seeds)
- `configs/` (YAML run configs)
- `reports/` (EDA + backtest summaries)
- `predictions/` (next-day outputs)
- `tests/` (data invariants + unit tests)

## 3. Data Flow
1. **Ingest**
   - Read CSVs with robust date parsing and column validation
2. **Validate**
   - Invariants:
     - state daily sum(P)=5 (tolerate exceptions flagged)
     - aggregated daily sum(P)=30 (tolerate known gaps)
     - duplicates resolved by rule (sum vs drop vs keep-last)
3. **Align**
   - Construct canonical date index; join CA target with aggregated features
4. **Feature Build**
   - Lags of CA P-vector
   - Lags of aggregated pressure counts
   - Day-of-week / holiday indicators (optional)
   - Location regime indicators (optional)
5. **Modeling**
   - Baselines → ML models → calibrated probabilities
6. **Pool Ranking**
   - Rank parts by calibrated score
7. **Set Generation**
   - Generate candidate sets from pool using:
     - stochastic sampling proportional to scores,
     - constrained search,
     - diversity constraints,
     - optional per-location priors
8. **Backtest**
   - Walk-forward evaluation for pool and sets
9. **Predict**
   - Produce next-day outputs with run manifest

## 4. Model Interfaces
- `fit(train_df, config) -> model_artifact`
- `predict_proba(context_df) -> np.ndarray[39]`
- `rank_parts(proba) -> ranked_table`
- `generate_sets(ranked_parts, proba, config) -> sets_table`

## 5. Reproducibility
- Deterministic seeds for numpy/torch/sklearn
- Persist:
  - configs, manifests, model versions, dataset hashes

## 6. Observability (Local)
- Structured logs to `reports/logs/`
- Run manifests with:
  - git hash (if applicable),
  - config snapshot,
  - start/end timestamps,
  - data coverage diagnostics

## 7. Security/Privacy
- Local-only research; no external services required.
