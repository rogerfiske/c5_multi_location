# TODO — C5 Multi-Location Parts Forecasting (Research)

**Version:** 2.0.0
**Last Updated:** 2025-12-29

> **Reference Documents:**
> - [`docs/data_contract.md`](../../../../data_contract.md) — Data specifications
> - [`PRD.md`](./PRD.md) — Requirements and acceptance criteria
> - [`Architecture.md`](./Architecture.md) — System design

---

## Phase 0 — Repository Hygiene

| Task | Status | Notes |
|------|--------|-------|
| Confirm canonical filenames | DONE | Renamed `CA5_agregated_matrix.csv` → `CA5_aggregated_matrix.csv` (2025-12-28) |
| Add `docs/data_contract.md` | DONE | Created 2025-12-28, updated to v1.1.0 on 2025-12-29 |
| Fix data quality issues | DONE | CA5_raw_data.txt (71), CA5_matrix.csv (2), CA5_aggregated_matrix.csv (2) |
| Add `configs/` scaffolding | TODO | |
| Add `src/` scaffolding | TODO | |
| Add `reports/` scaffolding | TODO | |
| Add `predictions/` scaffolding | TODO | |
| Add `tests/` scaffolding | TODO | |

---

## Phase 1 — Ingestion + Validation

| Task | Status | Notes |
|------|--------|-------|
| Implement robust CSV loader | TODO | Date parsing (M/D/YYYY, MM/DD/YYYY, YYYY-MM-DD) |
| Invariant: state sum(P)=5 | TODO | With exceptions report |
| Invariant: aggregated sum(P) in {20,25,30,35} | TODO | Holiday tolerance |
| Invariant: ascending order L_1 < L_2 < ... < L_5 | DONE | Fixed violations in all files |
| Duplicate date detection | TODO | Sum (aggregated) or flag (state) |
| Produce `data/processed/ca_features.parquet` | TODO | |

---

## Phase 2 — EDA

| Task | Status | Notes |
|------|--------|-------|
| Global EDA: frequencies, distributions | DONE | `scripts/comprehensive_eda.py` |
| Per-location EDA: L_1..L_5 distributions | DONE | Percentiles documented in PRD |
| Rolling window analysis | DONE | 30-day optimal (98.3% coverage) |
| Adjacent value tendency analysis | DONE | `scripts/recency_adjacency_analysis.py` - 5x random for L_1, L_5 |
| Set repetition analysis | DONE | 99.37% unique sets |
| Cross-state correlation analysis | DONE | `scripts/cross_state_analysis.py` - near zero correlation |
| Deliver `reports/eda/` | PARTIAL | Scripts exist, formal report TODO |
| Deliver `reports/eda_summary.md` | TODO | Findings documented in PRD Section 6 |

### Key EDA Findings (Summary)

| Finding | Value | Implication |
|---------|-------|-------------|
| Optimal rolling window | 30 days | 98.3% part coverage |
| L_1 adjacent tendency | 21.2% | 3x random, exploitable |
| L_5 adjacent tendency | 20.7% | 3x random, exploitable |
| L_4+L_5 both adjacent | 4.04% | 5x random, exploitable |
| Unique sets | 99.37% | No duplicate filtering needed |
| Cross-state correlation | ~0 | Use as consensus, not predictor |
| Part frequency spread | 12.1%-13.6% | Uniform, frequency alone insufficient |

---

## Phase 3 — Baselines

| Task | Status | Notes |
|------|--------|-------|
| Frequency baseline (global) | DONE | 88.77% Good+ (inverted signal!) |
| Frequency baseline (30-day rolling) | DONE | 89.32% Good+ (best baseline) |
| Persistence baseline | DONE | 87.12% Good+ |
| Adjacent tendency baseline | DONE | 86.85% Good+ |
| Backtest on 365-day holdout | DONE | `scripts/baseline_evaluation.py` |
| Calculate Good+ rate | DONE | See results below |
| Calculate Recall@K | DONE | See results below |

### Baseline Results (2025-12-29)

| Baseline | Good+ Rate | Recall@20 | Recall@30 |
|----------|-----------|-----------|-----------|
| Rolling Frequency (30d) | **89.32%** | **53.04%** | **79.01%** |
| Global Frequency | 88.77% | 49.92% | 74.74% |
| Persistence | 87.12% | 50.63% | 76.16% |
| Adjacent Tendency | 86.85% | 50.74% | 76.60% |

### CRITICAL FINDING: Inverted Signal

The high Good+ rate (89%) comes from **4-5 wrong predictions** (inverted signal), NOT 0-1 correct:
- 0-1 wrong: **0%** (baselines never get it right)
- 4-5 wrong: **89%** (baselines consistently wrong)

**Implication:** Frequency-based greedy set generation produces ANTI-predictions. The most frequent parts are NOT tomorrow's parts.

### Acceptance Criteria (from PRD)

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Good+ Rate | 89.32% (inverted) | >= 35% (correct) | >= 45% (correct) |
| Recall@20 | 53.04% | Beat baseline +10% | +20% |

**Note:** Need to distinguish between "Good+ via correct" vs "Good+ via inverted". Target should be 0-1 wrong, not 4-5 wrong.

---

## Phase 4 — First ML Models

| Task | Status | Notes |
|------|--------|-------|
| Multi-label tree baseline | TODO | LightGBM/XGBoost |
| Add lag features | TODO | L_1..L_5 lag-1, P_1..P_39 lag-1 |
| Add rolling features | TODO | 3, 7, 30 day windows |
| Add pressure features | TODO | Aggregated P lag-1 |
| Calibrate probabilities | TODO | Isotonic/Platt |
| Implement pool ranking | TODO | Top-K by score |

---

## Phase 5 — Candidate Set Generation

| Task | Status | Notes |
|------|--------|-------|
| Implement cascade model | TODO | Predict L_1 → filter → L_2 → ... |
| Implement weighted sampling | TODO | Score-proportional |
| Implement constrained search | TODO | Beam search with diversity |
| Score sets (joint likelihood) | TODO | |
| Optional L_1..L_5 assignment resolver | TODO | |

---

## Phase 6 — Ensemble + Optimization

| Task | Status | Notes |
|------|--------|-------|
| Compare model families | TODO | Frequency, Tree, Cascade |
| Ensemble models | TODO | |
| Optimize K (pool size) | TODO | Test K in {10, 15, 20, 25, 30} |
| Optimize number of sets | TODO | Default 20 |
| Drift tests | TODO | |
| Rolling window selection | TODO | |

---

## Phase 7 — Research Packaging

| Task | Status | Notes |
|------|--------|-------|
| One-command backtest | TODO | `python -m src.backtest.run --config configs/backtest.yaml` |
| One-command predict | TODO | `python -m src.predict.next_day --config configs/predict.yaml` |
| Final report | TODO | `reports/backtest_summary.md` |
| Reproducibility manifest | TODO | Git hash, config, seeds, data hash |

---

## Change Log

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-XX | 1.0.0 | Initial TODO from ChatGPT genesis | Original |
| 2025-12-29 | 2.0.0 | Updated with completed EDA tasks, added tables, linked to PRD/Architecture | Priya |
