# Session Summary - 2025-12-30

**Project:** C5 Multi-Location Parts Forecasting (Research)
**Duration:** ~1.5 hours
**Focus:** Exclusion analysis, Position-specific cascade model, TARGET MET

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

## Key Takeaway

The position-specific cascade model works because:
1. Edge positions (L_1, L_5) have 3x random adjacency signal
2. Cascade respects ascending constraint (L_1 < L_2 < ... < L_5)
3. Portfolio diversity captures good sets that greedy misses
4. Combined effect: 13% improvement, target met on first try
