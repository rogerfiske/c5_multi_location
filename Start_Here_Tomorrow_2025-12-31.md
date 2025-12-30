# Start Here Tomorrow - 2025-12-31

**Project:** C5 Multi-Location Parts Forecasting (Research)
**Last Session:** 2025-12-30
**Context:** PRIMARY TARGET MET - Cascade model achieved avg wrong 2.69

---

## Quick Context (30-second recap)

- **Problem:** Predict tomorrow's 5-part set from 39 possible parts
- **Previous baseline:** Stochastic Oracle avg wrong = 3.09
- **New baseline:** Cascade Portfolio avg wrong = **2.69** (TARGET MET)
- **Key signal:** Position-specific adjacency (37% for L_1/L_5 vs 22% for middle)
- **Distribution:** Eliminated 5-wrong days, 95% now at 2-3 wrong

---

## Current Status: Stretch Goals

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Avg best wrong | **2.69** | ~~< 3.0~~ MET | < 2.5 |
| Days 0-2 wrong | **32.69%** | ~~> 30%~~ MET | > 40% |
| Correct rate | 1.65% | > 3% | > 5% |

---

## Immediate Next Steps

### Option A: Tune Cascade Model (Low Risk)

**Goal:** Push avg wrong toward 2.5

**Actions:**
1. Try adjacency window +/-3 instead of +/-2
2. Experiment with boost factors (currently 3.0 for edges, 2.0 for middle)
3. Add recency weighting (recent days weighted higher)
4. Test different portfolio sizes (25, 100 sets)

**Run current model:**
```bash
python scripts/position_specific_baseline.py
```

### Option B: Sequence Prediction (Medium Risk)

**Goal:** Add temporal signal via Markov/LSTM

**Actions:**
1. Build Markov transition matrix from `transition_analysis.py` output
2. Predict P(part_tomorrow | part_today) for each part
3. Combine Markov scores with cascade model
4. If Markov shows promise, consider LSTM

### Option C: Conditional Position Models (Medium Risk)

**Goal:** Model P(L_2 | L_1), P(L_3 | L_1, L_2), etc.

**Actions:**
1. Build transition probabilities between positions
2. Use cascade but with conditional probabilities
3. May capture patterns like "if L_1 is low, L_5 tends to be high"

---

## Key Scripts Reference

| Script | Purpose | Last Result |
|--------|---------|-------------|
| `position_specific_baseline.py` | Cascade model | **avg 2.69** - TARGET MET |
| `exclusion_analysis.py` | Inversion hypothesis | 1.37% exclusion - not viable |
| `baseline_evaluation.py` | Frequency baselines | 0% correct, 89% inverted |
| `transition_analysis.py` | Day-over-day patterns | 33% L_1/L_5 adjacency |

---

## Critical Numbers to Remember

| Metric | Value | Meaning |
|--------|-------|---------|
| 2.69 | Current avg wrong | **New baseline** |
| 36-37% | L_1/L_5 adjacency | Best signal (3x random) |
| 22-23% | L_2/L_3/L_4 adjacency | Weaker signal (2x random) |
| 32.69% | Good rate (0-2 wrong) | Almost 1/3 of days |
| 1.65% | Correct rate (0-1 wrong) | ~6 days per year |
| 0% | 5-wrong days | Eliminated worst case |

---

## Team Agents Available

Invoke party mode for brainstorming: `/bmad:core:workflows:party-mode`

**C5 Team:**
- Priya (PO) - `/bmad:c5:agents:c5-product-owner`
- Theo (Forecasting) - `/bmad:c5:agents:c5-forecasting-scientist`
- Nova (Set Generation) - `/bmad:c5:agents:c5-set-generation-specialist`
- Winston (Architect) - `/bmad:c5:agents:c5-architect`
- Dev (ML Engineer) - `/bmad:c5:agents:c5-ml-engineer`

---

## Documents to Reference

| Document | Location | Purpose |
|----------|----------|---------|
| TODO | `docs/.../TODO.md` | Task tracking (v2.2.0) |
| Data Contract | `docs/data_contract.md` | Schema, invariants |
| Yesterday's Summary | `Session_summary_2025-12-30.md` | Full details |

---

## Suggested Opening Command

```
Good morning! Continue pushing toward stretch goals.

The cascade model hit avg 2.69 (target was < 3.0).
Let's try Option A first - tune the adjacency window and boost factors
to see if we can get closer to 2.5.
```

---

## Success Criteria for Tomorrow

| Metric | Current | Tomorrow's Target |
|--------|---------|-------------------|
| Avg best wrong | 2.69 | < 2.6 |
| Good rate (0-2) | 32.69% | > 35% |
| Correct rate | 1.65% | > 2% |

---

## Git Status

All work from 2025-12-30 committed and pushed. Clean slate for tomorrow.

Repository: `https://github.com/rogerfiske/c5_multi_location`
Branch: `main`
