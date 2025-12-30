# Session Summary - 2025-12-30

**Project:** C5 Multi-Location Parts Forecasting (Research)
**Duration:** ~4 hours
**Focus:** Exclusion analysis, Position-specific cascade model, Tuning, Markov, ALL STRETCH GOALS MET

---

## Executive Summary

Today's session achieved the **primary target** (avg wrong < 3.0) through the position-specific cascade model. Key activities:

1. **Exclusion Analysis** - Investigated inversion hypothesis (not viable)
2. **Position-Specific Cascade Model** - Exploited 33% adjacency signal in L_1/L_5
3. **Target Met** - Reduced avg wrong from 3.09 to 2.69 (13% improvement)

---

## Key Findings

### Finding 1: Inversion Strategy Not Viable

Investigated whether top-K predictions consistently EXCLUDE correct parts:

| K | Exclusion Rate | Conclusion |
|---|----------------|------------|
| 10 | 20.82% | Partial |
| 15 | 6.85% | Rare |
| **20** | **1.37%** | Not viable |
| 25 | 0.00% | Never |

**Conclusion:** The top-20 pool almost always contains SOME correct parts (Recall@20 ~51%). The problem is SET SELECTION from a decent pool, not pool exclusion. The 89% "inverted" signal is at the SET level (greedy picks wrong combination), not POOL level (pool doesn't exclude correct parts).

### Finding 2: Position-Specific Adjacency Confirmed

| Position | Combined (Exact + Adjacent +/-2) | Signal |
|----------|----------------------------------|--------|
| L_1 * | **36.54%** | 3x random |
| L_2 | 22.53% | 2x random |
| L_3 | 23.08% | 2x random |
| L_4 | 20.05% | 2x random |
| L_5 * | **37.36%** | 3x random |

Edge positions (L_1, L_5) are significantly more predictable than interior positions.

### Finding 3: TARGET MET - Cascade Model Performance

| Model | Avg Wrong | Correct (0-1) | Good (0-2) |
|-------|-----------|---------------|------------|
| Rolling Frequency (Greedy) | 4.44 | 0.00% | ~11% |
| Stochastic Oracle (50 sets) | 3.09 | 1.10% | ~21% |
| **Cascade Portfolio (50 sets)** | **2.69** | **1.65%** | **32.69%** |

**Improvement:** 13% reduction in avg wrong (0.4 fewer wrong per day)

**Distribution Shift:**
```
Before: 0% 0-wrong, 0% 1-wrong, 1% 2-wrong, 9% 3-wrong, 42% 4-wrong, 48% 5-wrong
After:  0% 0-wrong, 2% 1-wrong, 31% 2-wrong, 64% 3-wrong, 3% 4-wrong, 0% 5-wrong
```

Key win: Completely eliminated 5-wrong days. 95% of days now at 2-3 wrong.

---

## Scripts Created

| Script | Purpose | Key Output |
|--------|---------|------------|
| `scripts/exclusion_analysis.py` | Test inversion hypothesis | 1.37% exclusion@20 - not viable |
| `scripts/position_specific_baseline.py` | Cascade model with portfolio | **avg 2.69** - TARGET MET |

---

## Party Mode Session

**Participants:** Theo (Forecasting), Nova (Set Generation), Winston (Architect), Dev (ML Engineer), Priya (PO)

**Key Discussion Points:**
1. User asked about inversion strategy - if top-20 excludes all correct parts, use remaining 19
2. Team clarified: Recall@20 = 51% means partial overlap, not exclusion
3. Ran exclusion analysis to quantify - only 1.37% of days have complete exclusion
4. Proceeded with position-specific cascade as Priority 1
5. Target achieved on first implementation

**Decision Logged:** Position-specific cascade model is new baseline. Tradeoff: slower than frequency-only, but 13% improvement justifies compute for daily prediction.

---

## Acceptance Criteria Status

| Metric | Previous | Current | Target | Status |
|--------|----------|---------|--------|--------|
| Avg best wrong | 3.09 | **2.69** | < 3.0 | **MET** |
| Days 0-2 wrong | 21% | **32.69%** | > 30% | **MET** |
| Correct rate | 1.10% | 1.65% | > 3% | In progress |

---

## Files Modified/Created

### New Scripts:
- `scripts/exclusion_analysis.py`
- `scripts/position_specific_baseline.py`

### Updated Documents:
- `docs/.../TODO.md` - v2.2.0 with new results and updated acceptance criteria

---

## Next Steps (Stretch Goals)

1. **Tune cascade model** - Try adjacency window +/-3, different boost factors
2. **Sequence prediction** - Markov model on part transitions
3. **Push toward stretch goals:**
   - Avg wrong < 2.5 (currently 2.69)
   - Correct rate > 3% (currently 1.65%)
   - Good rate > 40% (currently 32.69%)

---

## Git Status

Ready to commit:
- 2 new scripts
- Updated TODO.md
- Session summary
- Start_Here_Tomorrow guide

---

## Key Takeaway (Morning Session)

The position-specific cascade model works because:
1. Edge positions (L_1, L_5) have 3x random adjacency signal
2. Cascade respects ascending constraint (L_1 < L_2 < ... < L_5)
3. Portfolio diversity captures good sets that greedy misses
4. Combined effect: 13% improvement, target met on first try

---

## Afternoon Session: Tuning + Signal Combination (Hours 1-3)

### Hour 1: Cascade Tuning - MAJOR WIN

Systematic parameter sweep revealed **portfolio size is the biggest lever**:

| Portfolio Size | Avg Wrong | Good Rate | Correct Rate |
|----------------|-----------|-----------|--------------|
| 25 | 2.909 | 19.78% | 0.82% |
| 50 (baseline) | 2.690 | 32.69% | 1.65% |
| 100 | 2.492 | 49.73% | 1.65% |
| **200** | **2.264** | **68.96%** | **4.67%** |

Other parameters:
- Adjacency window +/-3 slightly better than +/-2
- Edge boost 3.0 remains optimal

### Hour 2: Markov Transition Model - Small Additive

Built P(part_tomorrow | part_today) transition matrix:
- Average repeat probability: 12.68% (near random baseline 12.82%)
- Best Markov weight: 0.3
- Additional improvement: 2% on top of tuning

### Hour 3: Multi-State Consensus - No Improvement

- Cross-state correlation is near zero (as EDA found)
- Multi-state signal not useful as standalone
- Confirms states are independent

### Combined Model: ALL STRETCH GOALS MET

Final configuration:
- Portfolio size: 200
- Adjacency window: +/-3
- Markov weight: 0.3

| Metric | Original | Final | Improvement |
|--------|----------|-------|-------------|
| Avg wrong | 2.69 | **2.217** | **17.6%** |
| Good rate | 32.69% | **72.25%** | **2.2x** |
| Correct rate | 1.65% | **6.04%** | **3.7x** |
| 4-5 wrong days | 3.3% | **0%** | Eliminated |

---

## Scripts Created (Full Session)

| Script | Purpose | Key Output |
|--------|---------|------------|
| `exclusion_analysis.py` | Test inversion hypothesis | 1.37% exclusion - not viable |
| `position_specific_baseline.py` | Cascade model | avg 2.69 - TARGET MET |
| `cascade_tuning.py` | Parameter sweep | Portfolio 200 best |
| `markov_model.py` | Transition probabilities | Small additive lift |
| `multistate_signal.py` | Cross-state consensus | No improvement |
| `combined_model.py` | Optimized model | **avg 2.217 - ALL STRETCH MET** |

---

## Final Key Takeaway

**Portfolio size is the biggest lever.** More candidate sets = better chance of finding a good one.

The optimal configuration (200 sets, adj=3, Markov=0.3) achieves:
- 72% of days with 0-2 wrong parts
- 6% of days with perfect/near-perfect predictions
- Zero days with worst-case (4-5 wrong) outcomes

This is a research success: from 0% correct baseline to 6% correct with systematic optimization.

---

## Evening Session: Prediction Viability Investigation (Hours 3-4)

### User Insight: 2-3 Wrong is the Unusable Zone

User corrected our understanding of the "72% good rate":

| Wrong Count | Category | Actionability |
|-------------|----------|---------------|
| 0-1 wrong | Excellent | Use prediction directly |
| 4-5 wrong | Poor | Use INVERSE prediction |
| **2-3 wrong** | **Unusable** | Neither works |

Our "optimized" model: 6% at 0-1, 66% at 2, 28% at 3 = **94% in unusable zone**

### Investigation 1: Aggregated Matrix Analysis

Tested if CA5_aggregated_matrix.csv (multi-state consensus P_1 to P_39) provides signal:

| Analysis | Result | Script |
|----------|--------|--------|
| Lag correlation (0-5 days) | All < 0.01, lift ~1.0x | `aggregated_signal_analysis.py` |
| Velocity (P_t - P_t-1) | All lifts ~1.0x | `aggregated_velocity_analysis.py` |
| Concentration as regime indicator | No correlation | `aggregated_concentration_analysis.py` |

**Conclusion:** Multi-state consensus provides NO predictive signal for CA5. States are independent.

### Investigation 2: Dual Portfolio / Confidence Classifier

Attempted to predict which days would be 0-1 wrong vs 4-5 wrong:

| Approach | Result | Script |
|----------|--------|--------|
| Regime classifier (adj vs anti) | Cannot predict, always abstains | `dual_model_classifier.py` |
| Pool overlap as confidence | 98.4% days HIGH (not discriminating) | `confidence_prediction_system.py` |
| Adj vs Anti-adj portfolios | Both converge to similar results | `hybrid_portfolio_model.py` |

**Conclusion:** Cannot reliably predict which strategy will work on a given day.

### Investigation 3: Unordered Sets

Removed L_1 < L_2 < ... < L_5 constraint:

| Portfolio Size | Approach | Avg Wrong | Good Rate |
|----------------|----------|-----------|-----------|
| 200 | Ordered model | 2.438 | 53.7% |
| 200 | Unordered model | 2.370 | 58.4% |
| 1000 | Combined | 1.764 | 98.9% |

User correctly rejected 1000-set approach: "like saying if i pick a pool of 39 parts i will have 100% accuracy" - brute force, not prediction.

### DEVASTATING FINDING: Random Beats the Model

Fair comparison at SAME portfolio size (`fair_comparison.py`):

| Strategy | Portfolio | Avg Wrong | Good Rate |
|----------|-----------|-----------|-----------|
| **Random (no model)** | 200 sets | **2.055** | **87.1%** |
| Unordered Model | 200 sets | 2.370 | 58.4% |
| Ordered Model | 200 sets | 2.438 | 53.7% |

**The prediction model is WORSE than random sampling!**

**Why?** Biased sampling reduces portfolio diversity:
- Random 200 sets covers ~39 parts uniformly
- Model-biased 200 sets concentrates on ~15-20 high-scoring parts
- Uniform coverage wins because there is NO exploitable signal

---

## Decision: Prediction Approach ABANDONED

**Date:** 2025-12-30

**Rationale:**
1. Multi-state consensus provides no signal
2. Cannot predict which days will be correct vs inverted
3. Random sampling outperforms the model at equal portfolio size
4. The "model" reduces performance by biasing toward fewer parts

**What the data tells us:**
- Parts cycle uniformly through the 39-part space every ~30 days
- Day-over-day adjacency (~33%) is the ONLY weak signal
- Exploiting adjacency hurts diversity more than it helps
- At equal portfolio sizes, uniform random sampling is optimal

---

## Research Methodology

The methodology was primarily **statistical and heuristic-based**, not machine learning in the modern sense.

### Approaches Used

#### 1. Exploratory Data Analysis (EDA)
- Frequency distributions across 39 parts
- Rolling window coverage analysis
- Gap/recency distributions
- Cross-state correlation analysis

#### 2. Statistical Signal Detection
- Lag correlation (0-5 days) with aggregated matrix
- Lift calculations (observed vs expected frequency)
- Transition probability matrices (Markov)
- Adjacency tendency quantification (+/-2, +/-3, +/-4 windows)

#### 3. Heuristic Rule-Based Models
- **Frequency scoring**: P(part) based on rolling 30-day window
- **Adjacency boosting**: 3x weight for parts near yesterday's values
- **Position-specific cascade**: L_1 → filter → L_2 → ... → L_5 with ascending constraint
- **Edge position exploitation**: Higher boost for L_1/L_5 (37% signal vs 22% for middle)

#### 4. Probabilistic Sampling
- Score-proportional stochastic sampling (not greedy)
- Portfolio generation (50-200 diverse candidate sets)
- Best-of-N selection strategy

#### 5. Comparative Benchmarking
- Multiple baselines (global freq, rolling freq, persistence, random)
- Fair comparison at equal portfolio sizes
- Holdout validation (365-day backtest)

### What Was NOT Used

| Technique | Status | Reason |
|-----------|--------|--------|
| Neural networks (LSTM, Transformer) | Not implemented | Abandoned before reaching this phase |
| Gradient boosting (XGBoost, LightGBM) | Not implemented | Research concluded at baseline phase |
| Deep learning | Not implemented | No signal to learn from |
| Supervised classification | Attempted (regime classifier) | Failed - could not predict regime |

### The One "ML" Attempt

A simple **regime classifier** was attempted (`dual_model_classifier.py`) to predict which days would favor adjacency vs anti-adjacency strategies. It used basic features (pool overlap, score distributions) but failed - it could not reliably predict regime and defaulted to always abstaining.

### Methodology Summary

The research was fundamentally **statistical exploration + heuristic optimization**, not ML. The key finding - that random sampling beats the model - was discovered through systematic benchmarking, not through training and evaluating ML models. The project never progressed to Phase 4 (ML models) because Phase 3 (baselines) revealed there was no signal to model.

---

## Scripts Created (Evening Session)

| Script | Purpose | Key Output |
|--------|---------|------------|
| `holdout_report.py` | Generate holdout test reports | Formatted results |
| `aggregated_signal_analysis.py` | Test P_values as predictors | No signal (lift ~1.0x) |
| `aggregated_velocity_analysis.py` | Test P velocity | No signal |
| `aggregated_concentration_analysis.py` | Test concentration | No correlation |
| `dual_model_classifier.py` | Predict optimal strategy | Cannot classify |
| `hybrid_portfolio_model.py` | Compare adj vs anti | Both converge |
| `portfolio_agreement_analysis.py` | Pool overlap metric | High overlap always |
| `confidence_prediction_system.py` | Confidence-based routing | Not discriminating |
| `unordered_set_model.py` | Remove ascending constraint | Doesn't help fairly |
| `unordered_scaling_test.py` | Scale portfolio sizes | Brute force only |
| `fair_comparison.py` | Same-size comparison | **Random wins** |
| `check_overlap_distribution.py` | Overlap distribution | Min 19, Max 34 |

---

## Final Key Takeaways (Full Day)

### Morning: Success (Apparent)
- Position-specific cascade: 2.69 avg wrong
- Tuning + Markov: 2.217 avg wrong, 72% good rate
- "All stretch goals met"

### Evening: Reality Check
- 72% good rate is misleading (94% in unusable 2-3 zone)
- Aggregated matrix provides no signal
- Cannot predict regime (correct vs inverted days)
- **Random sampling BEATS the model** at equal portfolio size

### Conclusion
The prediction approach has been abandoned. The data exhibits near-uniform randomness with no exploitable patterns beyond trivial adjacency, and exploiting adjacency hurts diversity more than it helps.

---

## Tomorrow's Starting Point

The prediction research has concluded. Possible directions:

1. **Close the project** - document findings, archive
2. **Pivot to different problem** - not set prediction
3. **Accept random baseline** - use uniform sampling (2.05 avg wrong)
4. **Investigate external data** - not historical CA5 patterns
