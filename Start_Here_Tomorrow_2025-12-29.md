# Start Here Tomorrow — December 29, 2025

## Quick Context

Yesterday (Dec 28) you completed a **full gap analysis** of the ChatGPT genesis documents with the C5 team in Party Mode. Key housekeeping was done:
- Renamed `CA5_agregated_matrix.csv` → `CA5_aggregated_matrix.csv`
- Created `docs/data_contract.md` as single source of truth
- Documented critical data insights (ascending constraint, cascade filtering)

**The PRD and Architecture still need updating based on the gap analysis.**

---

## What's Ready

### C5 Agent Team (Invoke with slash commands)
```
/bmad:c5:agents:c5-product-owner      # Priya 📋 - Research PO
/bmad:c5:agents:c5-data-steward       # Iris 🔍 - Data Quality
/bmad:c5:agents:c5-eda-analyst        # Quinn 📊 - EDA
/bmad:c5:agents:c5-forecasting-scientist  # Theo 🔬 - ML/Forecasting
/bmad:c5:agents:c5-set-generation-specialist  # Nova 🎯 - Combinatorics
/bmad:c5:agents:c5-architect          # Winston 🏗️ - System Architect
/bmad:c5:agents:c5-ml-engineer        # Dev ⚙️ - Implementation
```

### Key Files
| File | Purpose | Status |
|------|---------|--------|
| `docs/data_contract.md` | Single source of truth for data | **NEW** - Created yesterday |
| `docs/c5_team_blueprint_artifacts_v2/.../PRD.md` | Product requirements | Needs updating |
| `docs/c5_team_blueprint_artifacts_v2/.../Architecture.md` | System design | Needs updating |
| `docs/c5_team_blueprint_artifacts_v2/.../TODO.md` | Phased plan | Needs updating |
| `Session_summary_2025-12-28.md` | Yesterday's full session notes | Reference |

---

## Critical Decisions Already Made

| Decision | Rationale |
|----------|-----------|
| Start date: 2008-09-08 | Cross-state consistency |
| Don't extend backward | Maintain 6-state shape parity |
| CA5_raw_data.txt excluded | Known quality issues |
| ME5 excluded | Shorter history |
| L values always ascending | Constraint documented in data contract |

---

## Recommended Next Steps

### Option A: Update Documents First (Recommended)

**Step 1:** Start Party Mode with C5 team
```
/bmad:core:workflows:party-mode
```

**Step 2:** Have Priya update PRD.md with:
- Reference to `docs/data_contract.md`
- Quantified acceptance criteria:
  - Good+ rate >= 35% (0-1 or 4-5 wrong)
  - Holdout: 500 events, <15 min processing
- Clarified prediction target (pool ranking vs constrained sets)
- Fixed file references

**Step 3:** Have Winston update Architecture.md with:
- Reference to `docs/data_contract.md`
- Corrected file names
- Model interface expansions

**Step 4:** Update TODO.md to reflect completed Phase 0 tasks

**Step 5:** Parse PRD into Epics and begin Story creation

---

### Option B: EDA Spike First

If you want data-driven acceptance criteria:

**Step 1:** Invoke Quinn
```
/bmad:c5:agents:c5-eda-analyst
```

**Step 2:** Select `GE` (Global EDA) to analyze:
- Part frequency distribution
- Location ranges (L_1 typical 1-17? L_2 typical 5-25?)
- Cross-state correlation strength
- Baseline difficulty estimate

**Step 3:** Use EDA findings to set quantified acceptance criteria

**Step 4:** Then proceed with document updates

---

## Gap Analysis Summary (From Yesterday)

### Critical Gaps (Must Fix)
1. Acceptance criteria vague - needs quantified thresholds
2. ~~Data contract missing~~ **DONE**
3. ~~File typo~~ **DONE**
4. Duplicate/holiday handling - documented in data contract
5. Location assignment scope - decision needed (defer to future phase?)

### Significant Gaps (Should Fix)
6. Multi-label vs constrained-5 framing unclear
7. Baseline specifications incomplete (window sizes)
8. Set diversity metric undefined
9. EDA success criteria missing
10. Environment/tooling not specified (Python 3.10+, pytest)

---

## Data Insights to Remember

### 1. Ascending Constraint
```
L_1 < L_2 < L_3 < L_4 < L_5 (always)
```
No duplicates. This enables cascade filtering for set generation.

### 2. Cascade Filtering Approach
If L_1 = 6, then L_2 ∈ {7..39}, L_3 ∈ {L_2+1..39}, etc.
Predict locations sequentially, each filters the pool for the next.

### 3. Holdout Test Format
```
0 wrong: Perfect
1 wrong: 4/5 correct
2-3 wrong: Ambiguous (poor)
4-5 wrong: Inverted signal (also useful!)

Good+ = (0-1 wrong) + (4-5 wrong)
Target: Good+ rate >= 35%
```

### 4. CA Has No Holiday Gaps
California operates 365/year. Holiday gaps only affect supporting states in aggregated data.

---

## Project Phase Status

| Phase | Status | Next Action |
|-------|--------|-------------|
| Team Setup | ✅ Complete | - |
| Gap Analysis | ✅ Complete | - |
| Data Contract | ✅ Complete | `docs/data_contract.md` |
| File Housekeeping | ✅ Complete | Renamed aggregated file |
| PRD Update | ⏳ Not Started | **Priority 1** |
| Architecture Update | ⏳ Not Started | **Priority 2** |
| Phase 0: Repo Hygiene | ⏳ Not Started | After doc updates |
| Phase 1: Ingestion | ⏳ Not Started | After Phase 0 |
| Phase 2: EDA | ⏳ Not Started | Can start in parallel |

---

## Quick Commands Reference

| Action | Command |
|--------|---------|
| Start Party Mode | `/bmad:core:workflows:party-mode` |
| Invoke Priya (PO) | `/bmad:c5:agents:c5-product-owner` |
| Invoke Quinn (EDA) | `/bmad:c5:agents:c5-eda-analyst` |
| Invoke Winston (Architect) | `/bmad:c5:agents:c5-architect` |

---

## Suggested First Message Tomorrow

> "Start Party Mode. Let's update PRD.md and Architecture.md based on yesterday's gap analysis, then parse into Epics."

or

> "Invoke Quinn to run a quick EDA spike - I want data-driven acceptance criteria before we finalize the PRD."

---

## Files Changed Yesterday (2025-12-28)
- Created: `docs/data_contract.md`
- Renamed: `data/raw/CA5_aggregated_matrix.csv` (fixed typo)
- Created: `Session_summary_2025-12-28.md`
- Created: `Start_Here_Tomorrow_2025-12-29.md`
- Updated: `README.md` (pending)
