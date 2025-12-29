# Start Here Tomorrow — 2025-12-30

**Project:** C5 Multi-Location Parts Forecasting (Research)
**Last Session:** 2025-12-29
**Context:** Completed baseline evaluation, discovered paradigm shift needed

---

## Quick Context (30-second recap)

- **Problem:** Predict tomorrow's 5-part set from 39 possible parts
- **Discovery:** Frequency-based prediction produces 0% correct (89% inverted)
- **Root cause:** Parts cycle uniformly (~30 days), greedy selection anti-predicts
- **Key signal:** Day-over-day adjacency (33% for L_1/L_5 positions)
- **Oracle ceiling:** Even with optimal scoring + 50 sets, only 1.1% days correct
- **New approach:** Portfolio of diverse sets, optimize avg wrong (target < 3.0)

---

## Immediate Next Steps (Option D: All in Parallel)

### Step 1: Portfolio Approach Enhancement

**Goal:** Improve stochastic set generation to push avg wrong below 3.0

**Actions:**
1. Review `scripts/stochastic_sampling_baseline.py` for optimization opportunities
2. Test different sampling strategies:
   - Temperature-scaled sampling (more/less random)
   - Diversity constraints (force sets to be different)
   - Ensemble scoring (combine multiple signals)
3. Target metrics:
   - Current avg best wrong: 3.09
   - Target: < 3.0 (would be first improvement)

**Run command:**
```bash
python scripts/stochastic_sampling_baseline.py
```

### Step 2: Position-Specific Models

**Goal:** Exploit the 33% adjacency signal in L_1 and L_5

**Actions:**
1. Create `scripts/position_specific_baseline.py`
2. Build separate predictors:
   - L_1 predictor (lowest part) - 33% adjacency
   - L_5 predictor (highest part) - 31% adjacency
   - L_2/L_3/L_4 predictors - 20% adjacency (harder)
3. Cascade approach: Predict L_1 → filter pool → predict L_2 → ...

**Key insight:** Edge positions are more predictable. Model them separately.

### Step 3: Sequence Prediction Exploration

**Goal:** Test if temporal sequence models can capture patterns

**Actions:**
1. Build simple Markov transition model from `transition_analysis.py` output
2. Consider LSTM/Transformer if Markov shows promise
3. Feature: sliding window of last N days' parts

---

## Key Data Files

| File | Description | Rows |
|------|-------------|------|
| `data/raw/CA5_matrix.csv` | California state matrix (primary) | 6,318 |
| `data/raw/CA5_aggregated_matrix.csv` | 6-state aggregate | 6,318 |

**Schema:** `date, L_1, L_2, L_3, L_4, L_5, P_1..P_39`

**Constraint:** L_1 < L_2 < L_3 < L_4 < L_5 (always ascending)

---

## Key Scripts Reference

| Script | Purpose | Last Result |
|--------|---------|-------------|
| `baseline_evaluation.py` | 4 frequency baselines | 0% correct, 89% inverted |
| `transition_analysis.py` | Day-over-day patterns | 33% L_1/L_5 adjacency |
| `stochastic_sampling_baseline.py` | Random vs greedy | Oracle 1.1% ceiling |
| `adjacency_weighted_baseline.py` | Temporal signal scoring | Recall@20 = 51% |

---

## Critical Numbers to Remember

| Metric | Value | Meaning |
|--------|-------|---------|
| 39 | Number of unique parts | Search space |
| 575,757 | Possible 5-part sets (39C5) | Combinatorial challenge |
| 12.77% | Next-day repeat rate | Some persistence |
| 7.79 days | Mean gap between appearances | Weekly-ish cycling |
| 33% | L_1/L_5 adjacency (+/-2) | Best exploitable signal |
| 1.1% | Oracle correct rate (50 sets) | Upper bound for current approach |
| 3.09 | Oracle avg best wrong | Current performance |

---

## Team Agents Available

Invoke party mode for brainstorming: `/bmad:core:workflows:party-mode`

**C5 Team:**
- 🔬 **Theo** - Forecasting Scientist ("Start with stupidest baseline")
- 📊 **Quinn** - EDA Analyst (pattern discovery)
- 🎯 **Nova** - Set Generation ("Joint prob ≠ product of marginals")
- 🏗️ **Winston** - System Architect (pipeline design)
- ⚙️ **Dev** - ML Engineer (implementation)
- 📋 **Priya** - Product Owner (you are here)

---

## Documents to Reference

| Document | Location | Purpose |
|----------|----------|---------|
| Data Contract | `docs/data_contract.md` | Schema, invariants, quality rules |
| PRD | `docs/c5_team_blueprint_artifacts_v2/_bmad-output/.../PRD.md` | Requirements |
| Architecture | `docs/c5_team_blueprint_artifacts_v2/_bmad-output/.../Architecture.md` | System design |
| TODO | `docs/c5_team_blueprint_artifacts_v2/_bmad-output/.../TODO.md` | Task tracking |
| Yesterday's Summary | `Session_summary_2025-12-29.md` | Full details |

---

## Suggested Opening Command

```
Good morning! Continue from where we left off - implementing Option D
(portfolio approach + position-specific models + sequence prediction).

Start with position-specific models since L_1/L_5 showed 33% adjacency
vs 20% for middle positions. This seems like the most promising signal.
```

---

## Success Criteria for Tomorrow

| Metric | Current | Tomorrow's Target |
|--------|---------|-------------------|
| Avg best wrong (50 sets) | 3.09 | < 3.0 |
| Days with 2-wrong set | 20% | > 25% |
| Position-specific model | N/A | L_1/L_5 beating random |

---

## Git Status

All work from 2025-12-29 committed and pushed. Clean slate for tomorrow.

Repository: `https://github.com/rogerfiske/c5_multi_location`
Branch: `main`
