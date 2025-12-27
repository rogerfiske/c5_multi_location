# TODO — C5 Multi-Location Parts Forecasting (Research)

## Phase 0 — Repository Hygiene
- [ ] Confirm canonical filenames (note: `CA5_agregated_matrix.csv` spelling)
- [ ] Add `configs/`, `src/`, `reports/`, `predictions/`, `tests/` scaffolding
- [ ] Add `docs/data_contract.md` summarizing invariants and cleaning rules

## Phase 1 — Ingestion + Validation
- [ ] Implement robust CSV loader with date parsing and schema checks
- [ ] Implement invariant checks:
  - [ ] state sum(P)=5 with exceptions report
  - [ ] aggregated sum(P)=30 with known holiday-gap exceptions report
  - [ ] duplicate date detection and resolution policy
- [ ] Produce `data/processed/ca_features.parquet` (or CSV)

## Phase 2 — EDA
- [ ] Global EDA: frequencies, seasonality, autocorrelation, transitions
- [ ] Per-location EDA: L1..L5 part distributions and regime differences
- [ ] Cross-state pressure EDA: correlation and lead/lag relationships
- [ ] Deliver: `reports/eda/` figures + `reports/eda_summary.md`

## Phase 3 — Baselines
- [ ] Frequency baseline (global + rolling)
- [ ] Persistence baseline
- [ ] Optional Markov/transition baseline
- [ ] Backtest + metrics for pool and sets

## Phase 4 — First ML Models
- [ ] Multi-label tree baseline (e.g., OneVsRest or native multi-output)
- [ ] Add lag features and pressure features
- [ ] Calibrate probabilities (Platt/isotonic/temperature scaling)
- [ ] Implement pool ranking

## Phase 5 — Candidate Set Generation
- [ ] Implement set generation methods:
  - [ ] weighted sampling
  - [ ] constrained search with diversity
- [ ] Score sets (joint likelihood approximation)
- [ ] Optional assignment resolver to map parts to L1..L5

## Phase 6 — Ensemble + Optimization
- [ ] Compare model families and ensembles
- [ ] Optimize K (pool size) and number of sets
- [ ] Robustness: drift tests and rolling window selection

## Phase 7 — Research Packaging
- [ ] One-command runs:
  - [ ] `python -m src.backtest.run --config configs/backtest.yaml`
  - [ ] `python -m src.predict.next_day --config configs/predict.yaml`
- [ ] Final report: `reports/backtest_summary.md`
