# Session Summary — 2025-12-29

**Project:** C5 Multi-Location Parts Forecasting (Research)
**Duration:** ~3 hours
**Focus:** Baseline evaluation, transition analysis, scoring strategy pivot

---

## Executive Summary

Today's session produced a **critical paradigm shift**: frequency-based prediction is fundamentally broken for this problem. Through systematic baseline evaluation and transition analysis, we discovered that:

1. **All frequency baselines produce 0% correct predictions** (but 89% inverted)
2. **Parts cycle uniformly** through the 39-part space every ~30 days
3. **Day-over-day adjacency** is the strongest exploitable signal (33% for L_1/L_5)
4. **Oracle upper bound is ~1% correct** even with optimal scoring + 50 diverse sets
5. **Portfolio approach needed**: Optimize for "avg wrong < 3.0" not "correct prediction"

---

## Key Findings

### Finding 1: Inverted Signal Discovery

| Baseline | Correct (0-1 wrong) | Inverted (4-5 wrong) |
|----------|---------------------|----------------------|
| Global Frequency | 0.00% | 88.77% |
| Rolling Frequency (30d) | 0.00% | 89.32% |
| Persistence | 0.00% | 87.12% |
| Adjacent Tendency | 0.00% | 86.85% |

**Implication:** Greedy selection from frequency scores consistently picks the WRONG parts. The most frequent parts are NOT tomorrow's parts.

### Finding 2: Part Frequency is Uniform

```
Most frequent:  P_15 = 13.69%
Least frequent: P_20 = 11.96%
Spread: only 1.73%
```

All 39 parts appear within ~30 days. There are no "hot" or "cold" parts to exploit. The 30-39 day "sweet spot" from previous studies was about **coverage**, not exclusion.

### Finding 3: Day-Over-Day Transition Structure

| Metric | Value | Implication |
|--------|-------|-------------|
| Next-day repeat rate | 12.77% | Some persistence exists |
| Mean gap between appearances | 7.79 days | Weekly-ish cycling |
| L_1 adjacency (+/-2) | 33.04% | 3x random - exploitable |
| L_5 adjacency (+/-2) | 31.45% | 3x random - exploitable |
| L_2/L_3/L_4 adjacency (+/-2) | ~20% | 2x random - weaker |

**Key insight:** Edge positions (L_1 lowest, L_5 highest) are more predictable than interior positions.

### Finding 4: Adjacency Window Analysis

| Window | Pool Size | Hit Rate | 5/5 Coverage |
|--------|-----------|----------|--------------|
| +/-2 | ~25 | 50.1% | 2.9% |
| +/-3 | ~35 | 62.4% | 10.1% |
| +/-4 | ~39 | 71.9% | 21.7% |
| +/-5 | ~39 | 79.1% | 35.7% |

**Sweet spot:** +/-4 to +/-5 captures 70-80% of tomorrow's parts from today's adjacency.

### Finding 5: Oracle Upper Bound

With adjacency-weighted scoring + 50 stochastic sets per day:

| Metric | Value |
|--------|-------|
| Days with CORRECT set (0-1 wrong) | **1.10%** (4 days/year) |
| Days with 2-wrong set | 20.00% |
| Days with 3-wrong set | 50.41% |
| Avg best wrong | 3.09 |

**Critical insight:** Even with optimal scoring, CORRECT prediction is nearly impossible (~1%). The problem is fundamentally combinatorial - predicting 1 of 575,757 possible sets.

---

## Scripts Created

| Script | Purpose | Key Output |
|--------|---------|------------|
| `baseline_evaluation.py` | Evaluate 4 frequency baselines | 0% correct, 89% inverted |
| `inverse_frequency_baseline.py` | Test if rare parts predict better | No improvement |
| `recency_exclusion_baseline.py` | Test recency window exclusion | Pool size ~0 (all parts appear in 30d) |
| `transition_analysis.py` | Day-over-day pattern analysis | Adjacency signals discovered |
| `adjacency_weighted_baseline.py` | Combine temporal signals | Recall@20 improved, still 0% correct |
| `stochastic_sampling_baseline.py` | Random sampling vs greedy | Oracle 1.1% upper bound |

---

## Party Mode Insights (Team Discussion)

**Theo (Forecasting):** "Start with the stupidest baseline that could work" - we did, and it revealed the inverted signal. The ~8-day mean gap suggests weekly cycling.

**Quinn (EDA):** The gap distribution is exponential - 50% of parts reappear within 5 days. Short-term memory (1-5 days) matters most.

**Nova (Set Generation):** "Joint probability is not the product of marginals." Greedy selection is systematically biased. Portfolio of diverse sets needed.

**Winston (Architecture):** The system should generate N diverse candidate sets and track "avg wrong" across the portfolio. Target: push avg wrong below 3.0.

---

## Paradigm Shift

### OLD Thinking:
- Predict the correct 5-part set
- Measure: % days with 0-1 wrong
- Binary success/failure

### NEW Thinking:
- Generate diverse portfolio of candidate sets
- Measure: avg wrong across best set in portfolio
- Continuous optimization toward partial accuracy

### Target Metrics (Revised):
| Metric | Current | Target |
|--------|---------|--------|
| Avg best wrong (50 sets) | 3.09 | < 3.0 |
| Days with 2-wrong set | 20% | > 30% |
| Recall@20 (pool quality) | 51% | > 60% |

---

## Files Modified/Created

### New Scripts (6):
- `scripts/baseline_evaluation.py`
- `scripts/inverse_frequency_baseline.py`
- `scripts/recency_exclusion_baseline.py`
- `scripts/transition_analysis.py`
- `scripts/adjacency_weighted_baseline.py`
- `scripts/stochastic_sampling_baseline.py`

### Pending Updates:
- `TODO.md` - needs baseline results added
- `README.md` - needs status update

---

## Tomorrow's Starting Point

**Option D (All of the above in parallel):**

1. **Portfolio Approach**
   - Generate 50+ diverse stochastic sets per day
   - Optimize scoring to minimize avg wrong
   - Track partial accuracy (3/5, 4/5 matches)

2. **Position-Specific Models**
   - L_1 and L_5 show 33% adjacency (vs 20% for middle)
   - Build separate predictors for edge vs interior positions
   - Cascade: predict L_1 first, then L_2|L_1, etc.

3. **Sequence Prediction**
   - Markov model on part transitions
   - LSTM/Transformer on sequence history
   - Treat as next-day prediction from sliding window

---

## Git Status (Pre-Commit)

```
Modified:
- README.md (to be updated)
- docs/.../TODO.md (to be updated)

New (untracked):
- scripts/baseline_evaluation.py (already committed earlier)
- scripts/inverse_frequency_baseline.py
- scripts/recency_exclusion_baseline.py
- scripts/transition_analysis.py
- scripts/adjacency_weighted_baseline.py
- scripts/stochastic_sampling_baseline.py
- Session_summary_2025-12-29.md
- Start_Here_Tomorrow_2025-12-30.md
```

---

## Key Takeaways

1. **Frequency doesn't work** - uniform distribution, no hot/cold parts
2. **Adjacency is the signal** - 33% for L_1/L_5, use +/-4 window
3. **Greedy is broken** - systematically anti-predicts
4. **Portfolio is the path** - diverse sets, optimize avg wrong
5. **1% Oracle ceiling** - correct prediction is nearly impossible; aim for partial accuracy
