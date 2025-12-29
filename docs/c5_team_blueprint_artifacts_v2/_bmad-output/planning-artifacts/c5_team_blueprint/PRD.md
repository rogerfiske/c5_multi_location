# PRD — C5 Multi-Location Parts Forecasting (Research)

**Version:** 2.0.0
**Last Updated:** 2025-12-29
**Owner:** Priya (Research Product Owner)

> **Data Contract Reference:** All data definitions, schemas, and invariants are specified in [`docs/data_contract.md`](../../../../data_contract.md). This PRD references that contract as the single source of truth for data specifications.

---

## 1. Problem Statement

We need to forecast the **next-day parts** likely required across **five California manufacturing locations** using historical demand patterns and cross-state aggregated signals.

---

## 2. Goals

### 2.1 Primary Goal — Ranked Pool Forecast
Produce a ranked list of parts (P_1..P_39) for the next day in California.
- Output: `part_id`, `rank`, `score` (probability or calibrated score), optional `confidence_band`
- Pool size K is configurable and will be optimized via backtesting (e.g., K in {10, 15, 20, 25, 30})

### 2.2 Secondary Goal — Ranked Candidate Sets
Generate multiple candidate 5-part sets (default target **20 sets**).
- Each set contains 5 unique parts satisfying the ascending constraint: L_1 < L_2 < L_3 < L_4 < L_5
- Each set has a set-level score (joint likelihood approximation)
- Optional resolver maps set parts to L1..L5 assignments when beneficial

---

## 3. Users

- Research operator (local PC): runs EDA, trains models, runs backtests, generates next-day forecasts.
- User is not a programmer — team executes all code and presents results.

---

## 4. Non-Functional Requirements

- Local execution only; no UI/web.
- Reproducibility: deterministic seeds, saved configs, versioned outputs.
- Robust ingestion: handle missing holidays, duplicates, mixed date formats (per data contract).
- Processing time: < 15 minutes for 500-event holdout backtest.

---

## 5. Data Inputs

> See [`docs/data_contract.md`](../../../../data_contract.md) for complete schema and invariant specifications.

### 5.1 Primary Files

| File | Description | Rows |
|------|-------------|------|
| `data/raw/CA5_matrix.csv` | Target state (CA) with L_1-L_5 and binary P_1-P_39 | 6,318 |
| `data/raw/CA5_aggregated_matrix.csv` | Cross-state aggregation (6 states: CA, MI, MO, NY, OH, MD) | 6,318 |

### 5.2 Supporting Files

| File | Description | Use Case |
|------|-------------|----------|
| `MI5_matrix.csv`, `MO5_matrix.csv`, `NY5_matrix.csv`, `OH5_matrix.csv`, `MD5_matrix.csv` | Other state matrices | Feature construction, cross-state analysis |
| `ME5_matrix.csv` | Maine (excluded) | Shorter history (starts 2013-05-15) |

---

## 6. EDA-Derived Data Characteristics

### 6.1 Location Distribution Profiles

| Position | Range | Mean | Mode | 92% Capture Threshold |
|----------|-------|------|------|----------------------|
| **L_1** | [1, 30] | 6.6 | 1 | **<= 15** |
| **L_2** | [2, 35] | 13.2 | 10 | **<= 24** |
| **L_3** | [3, 37] | 19.8 | 19 | **<= 30** |
| **L_4** | [5, 38] | 26.6 | 28 | **<= 35** |
| **L_5** | [9, 39] | 33.2 | 39 | **<= 39** |

**Key Insight:** L_1 is heavily skewed low (mode=1), L_5 is heavily skewed high (mode=39). These constrained ranges are exploitable for cascade filtering.

### 6.2 Part Frequency Distribution

| Metric | Value |
|--------|-------|
| Most frequent part | Part 11 (13.6%) |
| Least frequent part | Part 20 (12.1%) |
| Frequency range | 12.1% - 13.6% |
| Standard deviation | 26.7 occurrences (~1.5% spread) |

**Key Insight:** Part frequencies are remarkably uniform. Frequency-only baselines will struggle to differentiate.

### 6.3 Rolling Window Analysis (Recency Effect)

| Window Size | Part Coverage | L_1 Exact Match | L_5 Exact Match |
|-------------|---------------|-----------------|-----------------|
| 3-day | 34.2% | 21.1% | 20.3% |
| 7-day | 61.9% | 40.7% | 40.1% |
| 14-day | 85.3% | 62.2% | 61.5% |
| **30-day** | **98.3%** | **81.8%** | **80.9%** |
| 39-day | 99.5% | - | - |

**Recommendation:** 30-day rolling window is optimal — captures 98.3% of parts while maintaining ~70-80% exact match rates per location.

### 6.4 Adjacent Value Tendency (Exploitable Pattern)

| Position | Adjacent (+-1) Rate | Small Change (+-3) | vs. Random |
|----------|---------------------|-------------------|------------|
| **L_1** | **21.2%** | 43.1% | ~3x expected |
| L_2 | 12.6% | 29.7% | ~1.5x expected |
| L_3 | 11.1% | 26.1% | ~1.3x expected |
| L_4 | 12.9% | 29.3% | ~1.5x expected |
| **L_5** | **20.7%** | 41.6% | ~3x expected |

**Key Insight:** Both L_4 AND L_5 being adjacent (+-1) occurs 4.04% vs. ~0.8% expected — **5x higher than random chance**. This pattern is EXPLOITABLE for prediction.

### 6.5 Set Repetition Analysis

| Metric | Value | Implication |
|--------|-------|-------------|
| Unique sets | 6,278 / 6,318 | **99.37% unique** |
| Max repeat | 2 times | No set appears more than twice |
| Consecutive exact match | 0 (0.00%) | Never same set two days in a row |
| Consecutive 4+ match | 3 (0.05%) | Extremely rare |

**Implication:** No need to filter out "already seen" sets. Each day is effectively a new combination.

### 6.6 Cross-State Correlation

| Comparison | Avg Part Correlation | Lag-1 Autocorrelation |
|------------|---------------------|----------------------|
| CA vs MO (best) | 0.0045 | 0.0022 |
| CA vs OH | 0.0019 | -0.0008 |
| CA vs MD | -0.0010 | -0.0021 |
| CA vs MI | -0.0022 | -0.0003 |
| CA vs NY | -0.0031 | -0.0008 |

**Key Insight:** Cross-state same-day correlations are near zero. Aggregated P values are useful as **concurrent consensus signals**, not predictive lags.

---

## 7. Forecast Formulation (to be selected during research)

- Multi-label next-step prediction in P-space (39 outputs)
- Potential model families:
  - Probabilistic baselines (frequency, rolling window)
  - Tree ensembles (LightGBM/XGBoost)
  - Temporal deep models
  - Hybrid ensemble + calibration
  - Per-location vs pooled strategies
  - Cascade models (predict L_1, filter pool, predict L_2, etc.)

### 7.1 Feature Categories (from EDA)

| Category | Features | Priority |
|----------|----------|----------|
| **Lag Features** | Previous day L values, P values, rolling presence (3, 7, 30 day) | HIGH |
| **Temporal** | Day of week, month, year, day of year (cyclical) | MEDIUM |
| **Frequency** | Static frequency, recent frequency (30-day), momentum | HIGH |
| **Location-Specific** | Percentile boundaries, gap patterns, adjacent tendency weights | HIGH |
| **Cross-State** | Aggregated P values (consensus), lag-1 aggregated | MEDIUM |
| **Cascade** | Conditional distributions given L_1, pool filtering | HIGH |

---

## 8. Baselines

| Baseline | Description | Expected Performance |
|----------|-------------|---------------------|
| **Frequency baseline** | Rank parts by historical frequency | ~12.8% per part (uniform) |
| **Rolling window (30-day)** | Rank by recent 30-day frequency | 98.3% pool coverage |
| **Last-observed persistence** | Predict yesterday's parts | 12.9% hit rate |
| **Adjacent tendency** | Weight yesterday's L values +-1 | 20%+ for L_1, L_5 |

---

## 9. Evaluation

### 9.1 Pool Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Recall@K** | Fraction of actual parts in top-K pool | Primary metric |
| Precision@K | Fraction of pool parts that are correct | Secondary |
| PR-AUC | Area under precision-recall curve | Summary metric |
| Calibration (ECE/Brier) | Probability calibration quality | < 0.1 ECE |

### 9.2 Set Metrics (Good+ Rate)

| Outcome | Parts Correct | Classification |
|---------|---------------|----------------|
| 0 wrong | 5/5 | **GOOD** (Perfect) |
| 1 wrong | 4/5 | **GOOD** |
| 2 wrong | 3/5 | Ambiguous |
| 3 wrong | 2/5 | Ambiguous |
| 4 wrong | 1/5 | **GOOD** (Inverted signal) |
| 5 wrong | 0/5 | **GOOD** (Inverted signal) |

**Primary Set Metric:** `Good+ Rate = (0-1 wrong) + (4-5 wrong) / Total`

**Rationale:** Consistently wrong predictions (4-5 wrong) indicate inverted signal — equally valuable as correct predictions once identified.

### 9.3 Backtest Method

| Parameter | Value |
|-----------|-------|
| Method | Rolling-origin (walk-forward) |
| Holdout period | 6-12 months (~180-365 events) |
| Min holdout | 500 events for statistical significance |
| Train window | Expanding or fixed-length rolling |
| Horizon | 1-day ahead (default) |
| Processing time | < 15 minutes for full backtest |

---

## 10. Outputs

| File | Location | Description |
|------|----------|-------------|
| `ca_pool_next_day.csv` | `predictions/` | Ranked pool: part_id, rank, score |
| `ca_sets_next_day.csv` | `predictions/` | Ranked sets: set_id, parts, score |
| `backtest_summary.md` | `reports/` | Evaluation results and metrics |
| `eda/*` | `reports/` | EDA reports and visualizations |
| `models/*` | `models/` | Trained model artifacts |
| `run_*.yaml` | `configs/` | Reproducible run configurations |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Holiday gaps / duplicates | Enforce invariants per data contract |
| Concept drift | Rolling window evaluation, drift diagnostics |
| Uniform part frequencies | Exploit temporal patterns, not just frequency |
| Near-zero cross-state correlation | Use aggregated values as consensus, not predictive lag |
| Overfitting to historical patterns | Walk-forward validation, multiple holdout periods |

---

## 12. Acceptance Criteria (Research)

### 12.1 Quantified Targets

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| **Good+ Rate** | TBD (frequency baseline) | **>= 35%** | >= 45% |
| **Recall@20** | TBD | Beat baseline by 10%+ | Beat by 20%+ |
| **Holdout Events** | - | >= 500 | >= 1000 |
| **Processing Time** | - | < 15 min | < 5 min |

### 12.2 Deliverables

- [ ] Documented methodology and reproducible runs
- [ ] Frequency baseline performance established
- [ ] Improvement over baseline demonstrated on 6-12 month holdout
- [ ] Ranked pool output (ca_pool_next_day.csv)
- [ ] Ranked sets output (ca_sets_next_day.csv)
- [ ] Backtest summary with metrics (reports/backtest_summary.md)

### 12.3 Decision Criteria

| Outcome | Action |
|---------|--------|
| Good+ >= 35% | Success — proceed to production consideration |
| Good+ 25-35% | Partial success — investigate model improvements |
| Good+ < 25% | Insufficient — revisit problem formulation |

---

## 13. Change Log

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-XX | 1.0.0 | Initial PRD from ChatGPT genesis | Original |
| 2025-12-28 | 1.1.0 | Gap analysis identified issues | Priya |
| 2025-12-29 | 2.0.0 | Added EDA findings, quantified acceptance criteria, data contract reference | Priya |

---

## 14. References

- [`docs/data_contract.md`](../../../../data_contract.md) — Single source of truth for data definitions
- `Session_summary_2025-12-28.md` — Gap analysis session notes
- `scripts/comprehensive_eda.py` — EDA analysis code
- `scripts/recency_adjacency_analysis.py` — Recency and pattern analysis code
