# PRD — C5 Multi-Location Parts Forecasting (Research)

## 1. Problem Statement
We need to forecast the **next-day parts** likely required across **five California manufacturing locations** using historical demand patterns and cross-state aggregated signals.

## 2. Goals
### 2.1 Primary Goal — Ranked Pool Forecast
Produce a ranked list of parts (P_1..P_39) for the next day in California.
- Output: `part_id`, `rank`, `score` (probability or calibrated score), optional `confidence_band`
- Pool size K is configurable and will be optimized via backtesting (e.g., K in {10, 15, 20, 25, 30})

### 2.2 Secondary Goal — Ranked Candidate Sets
Generate multiple candidate 5-part sets (default target **20 sets**).
- Each set contains 5 unique parts
- Each set has a set-level score (joint likelihood approximation)
- Optional resolver maps set parts to L1..L5 assignments when beneficial

## 3. Users
- Research operator (local PC): runs EDA, trains models, runs backtests, generates next-day forecasts.

## 4. Non-Functional Requirements
- Local execution only; no UI/web.
- Reproducibility: deterministic seeds, saved configs, versioned outputs.
- Robust ingestion: handle missing holidays, duplicates, mixed date formats.

## 5. Data Inputs
- `data/raw/CA5_matrix.csv` (target state)
- `data/raw/CA5_agregated_matrix.csv` (cross-state pressure features)
- Other states (NY5, OH5, MD5, MI5, MO5) may be used for:
  - feature construction,
  - auxiliary tasks,
  - transfer learning,
  - regime discovery.
- ME5 is excluded by default.

## 6. Forecast Formulation (to be selected during research)
- Multi-label next-step prediction in P-space (39 outputs)
- Potential model families:
  - probabilistic baselines,
  - tree ensembles (LightGBM/XGBoost),
  - temporal deep models,
  - hybrid ensemble + calibration,
  - per-location vs pooled strategies.

## 7. Baselines
- Frequency baseline (global and rolling window)
- Last-observed persistence
- Simple transition/Markov baseline (optional)

## 8. Evaluation
### 8.1 Pool Metrics
- Recall@K (primary)
- Precision@K
- PR-AUC (micro/macro)
- Calibration metrics (ECE / Brier)

### 8.2 Set Metrics
- Best-set hit rate: whether any predicted set matches the observed 5-part day (allowing ordering-insensitive comparison)
- Partial overlap distribution across sets
- Diversity/coverage of candidate sets

### 8.3 Backtest Method
- Rolling-origin evaluation (walk-forward)
- Train window options:
  - expanding window, or
  - fixed-length rolling window
- Horizon: 1-day ahead (default)

## 9. Outputs
- `predictions/ca_pool_next_day.csv`
- `predictions/ca_sets_next_day.csv`
- `reports/backtest_summary.md`
- `reports/eda/*`
- `models/*`
- `configs/run_*.yaml`

## 10. Risks & Mitigations
- Holiday gaps / duplicates → enforce invariants and cleaning rules
- Concept drift → rolling window evaluation and drift diagnostics
- Small label space (39) but structured combinations → use set generation strategies and calibrated ranking

## 11. Acceptance Criteria (Research)
- Demonstrated improvement over frequency baseline on the last 6–12 months holdout
- Documented methodology and reproducible runs producing:
  - ranked pool,
  - ranked sets,
  - backtest summary.
