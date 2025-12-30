# TODO — C5 Multi-Location Parts Forecasting (Research)

**Version:** 2.4.0
**Last Updated:** 2025-12-30

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
| Inverse frequency baseline | DONE | `scripts/inverse_frequency_baseline.py` - No improvement |
| Recency exclusion baseline | DONE | `scripts/recency_exclusion_baseline.py` - Pool ~0 (all parts in 30d) |
| Transition analysis | DONE | `scripts/transition_analysis.py` - Found adjacency signals |
| Adjacency-weighted baseline | DONE | `scripts/adjacency_weighted_baseline.py` - Recall improved |
| Stochastic sampling baseline | DONE | `scripts/stochastic_sampling_baseline.py` - Oracle 1.1% |
| Exclusion analysis | DONE | `scripts/exclusion_analysis.py` - Inversion not viable (1.37% exclusion@20) |
| Position-specific cascade model | DONE | `scripts/position_specific_baseline.py` - **TARGET MET: avg 2.69** |
| Cascade tuning sweep | DONE | `scripts/cascade_tuning.py` - Portfolio 200 best (avg 2.264) |
| Markov transition model | DONE | `scripts/markov_model.py` - Small additive lift (2%) |
| Multi-state consensus | DONE | `scripts/multistate_signal.py` - No improvement |
| Combined optimized model | DONE | `scripts/combined_model.py` - **ALL STRETCH GOALS MET: avg 2.217** |

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

### Transition Analysis (2025-12-29)

| Metric | Value | Implication |
|--------|-------|-------------|
| Next-day repeat rate | 12.77% | Some persistence exists |
| Mean gap between appearances | 7.79 days | Weekly-ish cycling |
| L_1 adjacency (+/-2) | **33.04%** | 3x random - edge positions predictable |
| L_5 adjacency (+/-2) | **31.45%** | 3x random - edge positions predictable |
| L_2/L_3/L_4 adjacency (+/-2) | ~20% | 2x random - harder to predict |
| +/-4 window hit rate | 71.9% | Adjacency is exploitable signal |

### Oracle Upper Bound (2025-12-29)

With adjacency-weighted scoring + 50 stochastic sets per day:

| Metric | Value |
|--------|-------|
| Days with CORRECT set (0-1 wrong) | **1.10%** |
| Days with 2-wrong set | 20.00% |
| Days with 3-wrong set | 50.41% |
| Avg best wrong | 3.09 |

**Paradigm Shift:** Perfect prediction (~1% ceiling) is nearly impossible. New target: **avg wrong < 3.0** via portfolio approach.

### Exclusion Analysis (2025-12-30)

Investigated whether top-K predictions consistently EXCLUDE correct parts (inversion hypothesis):

| K | Exclusion Rate | Anti-Recall | Conclusion |
|---|----------------|-------------|------------|
| 10 | 20.82% | 74.41% | Partial exclusion |
| 15 | 6.85% | 60.55% | Rare exclusion |
| **20** | **1.37%** | 46.52% | **Inversion NOT viable** |
| 25 | 0.00% | 32.88% | Never excluded |

**Finding:** The top-20 pool almost always contains SOME correct parts (Recall@20 ~51%). The problem is SET SELECTION from a decent pool, not pool exclusion. Inversion strategy is not viable.

### Position-Specific Cascade Model (2025-12-30) - TARGET MET

Exploits 33% adjacency signal in edge positions (L_1, L_5) using cascade prediction.

**Position Accuracy (Combined = Exact + Adjacent +/-2):**

| Position | Combined | Signal Strength |
|----------|----------|-----------------|
| L_1 * | **36.54%** | 3x random |
| L_2 | 22.53% | 2x random |
| L_3 | 23.08% | 2x random |
| L_4 | 20.05% | 2x random |
| L_5 * | **37.36%** | 3x random |

**Model Performance (Portfolio of 50 sets):**

| Model | Avg Wrong | Correct (0-1) | Good (0-2) |
|-------|-----------|---------------|------------|
| Rolling Frequency (Greedy) | 4.44 | 0.00% | ~11% |
| Stochastic Oracle (50 sets) | 3.09 | 1.10% | ~21% |
| **Cascade Portfolio** | **2.69** | **1.65%** | **32.69%** |

**Improvement:** 3.09 -> 2.69 (13% reduction, 0.4 fewer wrong per day)

**Distribution Shift:**
- Eliminated 5-wrong days (48% -> 0%)
- 95% of days now at 2-3 wrong

### Acceptance Criteria (UPDATED 2025-12-30)

| Metric | Previous | Current | Target | Stretch |
|--------|----------|---------|--------|---------|
| Avg best wrong (50 sets) | 3.09 | **2.69** | ~~< 3.0~~ MET | < 2.5 |
| Days with 0-2 wrong | 21% | **32.69%** | > 30% MET | > 40% |
| Correct rate (0-1 wrong) | 1.10% | **1.65%** | > 3% | > 5% |

**Note:** Primary target (avg < 3.0) achieved. New focus: push toward stretch goals.

### Combined Optimized Model (2025-12-30) - ALL STRETCH GOALS MET

After systematic tuning and signal combination:

**Configuration:**
- Portfolio size: 200 sets (biggest lever - 15.8% improvement alone)
- Adjacency window: +/-3 (slight improvement over +/-2)
- Markov signal: weight 0.3 (additional 2% improvement)
- Multi-state: No improvement (cross-state correlation ~0)

**Tuning Results:**

| Parameter | Tested Values | Best |
|-----------|---------------|------|
| Portfolio size | 25, 50, 100, 200 | **200** |
| Adjacency window | +/-2, +/-3, +/-4, +/-5 | **+/-3** |
| Edge boost | 2.0, 3.0, 4.0, 5.0 | **3.0** |
| Markov weight | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 | **0.3** |

**Final Model Performance:**

| Model | Avg Wrong | Good Rate | Correct Rate |
|-------|-----------|-----------|--------------|
| Original baseline | 2.69 | 32.69% | 1.65% |
| Cascade (50 sets) | 2.69 | 32.69% | 1.65% |
| Tuned (200 sets, adj=3) | 2.264 | 68.96% | 4.67% |
| **Combined (+Markov)** | **2.217** | **72.25%** | **6.04%** |

**Distribution Shift:**
```
Original: 0% 0-wrong, 2% 1-wrong, 31% 2-wrong, 64% 3-wrong, 3% 4-wrong, 0% 5-wrong
Combined: 0% 0-wrong, 6% 1-wrong, 66% 2-wrong, 28% 3-wrong, 0% 4-wrong, 0% 5-wrong
```

**Improvement:** 17.6% reduction in avg wrong (2.69 -> 2.217)

### Acceptance Criteria (FINAL 2025-12-30)

| Metric | Original | Current | Target | Stretch | Status |
|--------|----------|---------|--------|---------|--------|
| Avg best wrong | 2.69 | **2.217** | < 3.0 | < 2.5 | **ALL MET** |
| Days 0-2 wrong | 32.69% | **72.25%** | > 30% | > 60% | **ALL MET** |
| Correct rate | 1.65% | **6.04%** | > 3% | > 5% | **ALL MET** |
| 4-5 wrong days | 3.3% | **0%** | - | - | **Eliminated** |

### CRITICAL INVESTIGATION: Prediction Viability (2025-12-30)

Following the "stretch goals met" milestone, deeper investigation revealed fundamental issues:

#### User Insight: 2-3 Wrong is the Unusable Zone

The 72% "good rate" (0-2 wrong) is misleading. User clarified:
- **0-1 wrong** = Actionable (can use the prediction directly)
- **4-5 wrong** = Actionable (can use the INVERSE prediction)
- **2-3 wrong** = UNUSABLE (neither direct nor inverse works)

Current model: 6% at 0-1 wrong, 66% at 2 wrong, 28% at 3 wrong = **94% in unusable zone**

#### Aggregated Matrix Analysis (No Signal)

Tested if multi-state consensus (P_1 to P_39) predicts CA5:

| Test | Result | Script |
|------|--------|--------|
| Lag-0 to Lag-5 correlation | All < 0.01, lift ~1.0x | `aggregated_signal_analysis.py` |
| Velocity (P_today - P_yesterday) | All lifts ~1.0x | `aggregated_velocity_analysis.py` |
| Concentration as regime indicator | No correlation (max +0.035) | `aggregated_concentration_analysis.py` |

**Conclusion:** Multi-state consensus provides NO predictive signal for CA5.

#### Dual Portfolio / Confidence Classifier (Cannot Predict Regime)

Attempted to predict which days would be 0-1 wrong vs 4-5 wrong:

| Approach | Result | Script |
|----------|--------|--------|
| Regime classifier | Cannot predict, always abstains | `dual_model_classifier.py` |
| Adjacency vs Anti-adjacency | Portfolios converge (not opposites) | `hybrid_portfolio_model.py` |
| Pool overlap as confidence | 98.4% days HIGH confidence (not discriminating) | `confidence_prediction_system.py` |

**Conclusion:** Cannot reliably predict which strategy will work on a given day.

#### Unordered Set Analysis (Doesn't Help)

Removed ascending constraint (L_1 < L_2 < ... < L_5), just require 5 unique parts:

| Portfolio Size | Avg Wrong | Good Rate | Script |
|----------------|-----------|-----------|--------|
| 200 (unordered) | 2.370 | 58.4% | `unordered_set_model.py` |
| 400 (combined) | 1.995 | 91.2% | `unordered_set_model.py` |
| 1000 (combined) | 1.764 | 98.9% | `unordered_scaling_test.py` |

User correctly rejected 1000-set approach: "like saying if i pick a pool of 39 parts i will have 100% accuracy" - brute force, not prediction.

#### DEVASTATING FINDING: Random Beats the Model

Fair comparison at SAME portfolio size (`fair_comparison.py`):

| Strategy | Portfolio Size | Avg Wrong | Good Rate |
|----------|----------------|-----------|-----------|
| **Random (no model)** | 200 | **2.055** | **87.1%** |
| Unordered Model | 200 | 2.370 | 58.4% |
| Ordered Model | 200 | 2.438 | 53.7% |

**The prediction model is WORSE than random sampling!**

Why? Biased sampling reduces portfolio diversity:
- Random 200 sets covers ~39 parts uniformly
- Model-biased 200 sets concentrates on ~15-20 high-scoring parts
- Uniform coverage wins because there is NO exploitable signal

### Decision: Prediction Approach ABANDONED

**Date:** 2025-12-30

**Rationale:**
1. Multi-state consensus provides no signal
2. Cannot predict which days will be correct vs inverted
3. Random sampling outperforms the model at equal portfolio size
4. The "model" actually reduces performance by biasing toward fewer parts

**What the data tells us:**
- Parts cycle uniformly through the 39-part space every ~30 days
- Day-over-day adjacency (~33%) is the ONLY weak signal, but exploiting it hurts diversity more than it helps
- At equal portfolio sizes, uniform random sampling is optimal

**Recommendation:** This research demonstrates that deterministic prediction of lottery-style CA5 outcomes is not viable. The data exhibits near-uniform randomness with no exploitable patterns beyond trivial adjacency. Future work should either:
1. Accept random baseline performance (~2.05 avg wrong, ~87% good rate)
2. Focus on portfolio diversity optimization rather than prediction
3. Investigate entirely different data sources or problem formulations

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
| Implement cascade model | **DONE** | `position_specific_baseline.py` - L_1 -> filter -> L_2 -> ... |
| Implement weighted sampling | **DONE** | Score-proportional stochastic sampling in cascade |
| Implement constrained search | PARTIAL | Cascade filtering respects L_1 < L_2 < ... < L_5 |
| Score sets (joint likelihood) | TODO | Currently using portfolio best-of selection |
| Optional L_1..L_5 assignment resolver | N/A | Cascade inherently assigns positions |
| Portfolio generation | **DONE** | 50 diverse sets per day, best-of selection |

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
| 2025-12-29 | 2.1.0 | Added transition analysis, Oracle upper bound, revised acceptance criteria | Priya |
| 2025-12-30 | 2.2.0 | **TARGET MET** - Added exclusion analysis (inversion not viable), position-specific cascade model (avg 2.69), updated acceptance criteria | Priya + C5 Team |
| 2025-12-30 | 2.3.0 | **ALL STRETCH GOALS MET** - Tuning (portfolio 200, adj=3) + Markov signal achieves avg 2.217, 72.25% good rate, 6.04% correct rate | Priya + C5 Team |
| 2025-12-30 | 2.4.0 | **PREDICTION ABANDONED** - Aggregated matrix no signal, dual portfolio cannot predict regime, random beats model (2.05 vs 2.44 avg wrong). Research concludes prediction is not viable. | Priya + C5 Team |
