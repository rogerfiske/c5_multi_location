# C5 Multi-Location Parts Forecasting — Team Overview (BMAD Builder Input)

## Team Name
**C5 PartPulse Research Collective**

## Macro Objective
Use the historical multi-state, multi-location parts demand matrices to forecast **next-day California demand** using two complementary outputs:

1. **Pool Forecast (primary):** a ranked list of parts (P_1..P_39) with highest likelihood of being required the next day at any of California’s five locations. Pool size is **configurable** and will be **optimized** via backtesting.
2. **Set Forecast (secondary):** produce **multiple candidate 5-part sets** (e.g., 10–30 sets; default target 20) representing plausible next-day combinations. Each set is ranked by overall likelihood and can optionally be mapped to L1–L5.

This is a **research-only** project operated locally on a single PC. No UI/web app is required.

## Data Contract (Authoritative)
- State files: `data/raw/*5_matrix.csv`
  - Columns: `date`, `L_1..L_5`, `P_1..P_39`
  - `P_*` are binary; daily sum is typically **5**
- Aggregated master: `data/raw/CA5_agregated_matrix.csv`
  - Aggregation of **CA5, NY5, OH5, MD5, MI5, MO5** by date
  - `P_*` are counts **0..6**; daily sum typically **30**
  - ME5 excluded due to shorter history
- Known anomalies (must be handled):
  - Holiday gaps causing totals of 20/25
  - Duplicate date(s) can yield totals of 35
  - Date formatting inconsistencies (must parse dates robustly)

## Core Research Questions
1. How predictive is multi-state **pressure** (aggregated `P_*` counts) for CA next-day `P_*` outcomes?
2. Do locations L1–L5 exhibit distinct part regimes requiring per-location models?
3. What forecasting formulation is best:
   - multi-label classification (P-space),
   - sequence models,
   - count/intensity models,
   - hybrid + calibration,
   - or ensemble stacking?

## Target Outputs
All forecasts are **ranked lists**. Probability scores are computed internally for evaluation and ranking.
- `predictions/ca_pool_next_day.csv`
- `predictions/ca_sets_next_day.csv`
- `reports/backtest_summary.md`
- `reports/eda/` (figures + tables)
- `models/` (serialized artifacts)

## Success Criteria (Research)
- Demonstrate consistent lift over baselines:
  - frequency baseline, last-value baseline, simple Markov transitions
- Primary metric for pool:
  - Recall@K / HitRate@K on next-day CA `P_*`
- Secondary metrics:
  - precision@K, PR-AUC (macro/micro), calibration error, diversity of sets

## Non-Goals
- No UI, no web deployment, no multi-tenant serving.
- No hard real-time constraints; focus on correctness and reproducible science.

## Operating Constraints
- Single-machine execution. Optional RUNPOD GPU acceleration if used or required by chosen models, is available.
- Deterministic runs with seed control; reproducible environment capture.

## Working Agreement
- Every change that affects labels/features updates:
  - `docs/data_contract.md` and
  - `tests/data_invariants_test.py` (to be created).
