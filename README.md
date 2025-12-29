# C5 Multi-Location Parts Demand Forecasting

## Project Overview

This research project builds ML pipelines to **predict next-day California manufacturing parts demand** using 16+ years of multi-state historical data. The project uses the **BMAD Method** with a specialized C5 agent team.

### Forecasting Objectives
1. **Ranked Pool**: Probability-ranked list of parts (P_1..P_39) likely needed tomorrow
2. **Ranked Sets**: Multiple candidate 5-part combinations for location assignments

### Project Status
- **Phase**: Baseline Evaluation Complete, Paradigm Shift Identified
- **Last Session**: 2025-12-29 (Baseline eval, Transition analysis, Oracle bound)
- **Key Finding**: Frequency-based prediction yields 0% correct (89% inverted). Oracle upper bound ~1%.
- **Next Steps**: Option D - Portfolio approach + Position-specific models + Sequence prediction

---

## Dataset Overview

This project contains manufacturing parts demand data from 6 US state locations, each with 5 manufacturing facilities. The data spans from September 2008 to 2025.

## Dataset Structure

### Source Files (data/raw/)

| File | State | Rows | Date Range |
|------|-------|------|------------|
| CA5_matrix.csv | California | 6,318 | 9/8/2008 - 2025 |
| MI5_matrix.csv | Michigan | 6,313 | 9/8/2008 - 2025 |
| MO5_matrix.csv | Missouri | 6,318 | 9/8/2008 - 2025 |
| NY5_matrix.csv | New York | 6,316 | 9/8/2008 - 2025 |
| OH5_matrix.csv | Ohio | 6,318 | 9/8/2008 - 2025 |
| MD5_matrix.csv | Maryland | 6,317 | 9/8/2008 - 2025 |
| ME5_matrix.csv | Maine | 2,542 | 5/15/2013 - 2025 |

### Aggregated File
| File | Description | Rows |
|------|-------------|------|
| CA5_aggregated_matrix.csv | Combined demand from CA, MI, MO, NY, OH, MD | 6,318 |

**Note:** File renamed from `CA5_agregated_matrix.csv` on 2025-12-28 to fix typo.

## Column Definitions

### Location Columns (L_1 to L_5)
- Each column represents one of 5 manufacturing locations within a state
- Values: Part numbers (1-39) required at that location for the given date

### Part Matrix Columns (P_1 to P_39)
- Binary indicator (0/1) in source files showing if part was required that day
- In aggregated file: Count (0-6) showing how many states required that part

## Data Characteristics

- **Parts Range**: 39 unique part numbers (1-39)
- **Locations per State**: 5
- **Total Locations (excluding ME)**: 30 (6 states x 5 locations)
- **Daily Parts per State**: 5 (one per location)
- **Daily Parts in Aggregated**: 30 (sum across 6 states)

## Notes
- ME5_matrix.csv excluded from aggregation due to shorter date range (2,542 vs ~6,318 records)
- Minor date gaps exist on holidays (Christmas, etc.) in some supporting state files
- California (CA) operates 365 days/year with **no holiday gaps**
- All files aligned by date for accurate aggregation

## Critical Data Constraints

### Location Ordering (Discovered 2025-12-28)
In all matrix files, location values are **always strictly ascending**:
```
L_1 < L_2 < L_3 < L_4 < L_5
```
- No duplicate parts on same day for a given state
- This constraint enables cascade filtering for set generation
- See `docs/data_contract.md` for full invariant specifications

---

## BMAD Agent Team: C5 PartPulse Research Collective

This project uses the BMAD Method with a specialized team of 7 agents for forecasting research:

| Agent | Name | Role | Invoke Command |
|-------|------|------|----------------|
| 📋 | Priya | Research Product Owner | `/bmad:c5:agents:c5-product-owner` |
| 🔍 | Iris | Data Quality Guardian | `/bmad:c5:agents:c5-data-steward` |
| 📊 | Quinn | Exploratory Data Analyst | `/bmad:c5:agents:c5-eda-analyst` |
| 🔬 | Theo | Forecasting Scientist | `/bmad:c5:agents:c5-forecasting-scientist` |
| 🎯 | Nova | Set Generation Specialist | `/bmad:c5:agents:c5-set-generation-specialist` |
| 🏗️ | Winston | System Architect | `/bmad:c5:agents:c5-architect` |
| ⚙️ | Dev | ML Implementation Engineer | `/bmad:c5:agents:c5-ml-engineer` |

### Key Documentation
- **Data Contract**: `docs/data_contract.md` - **Single source of truth** for data definitions, schemas, invariants
- **PRD (Draft)**: `docs/c5_team_blueprint_artifacts_v2/_bmad-output/planning-artifacts/c5_team_blueprint/PRD.md`
- **Architecture (Draft)**: `docs/c5_team_blueprint_artifacts_v2/_bmad-output/planning-artifacts/c5_team_blueprint/Architecture.md`
- **TODO**: `docs/c5_team_blueprint_artifacts_v2/_bmad-output/planning-artifacts/c5_team_blueprint/TODO.md`
- **C5 Module**: `_bmad/c5/`
- **Session Notes**: `Session_summary_2025-12-29.md`, `Start_Here_Tomorrow_2025-12-30.md`

---

## Project Structure

```
c5_multi_location/
├── data/
│   └── raw/                    # Source CSV files (7 state matrices + aggregated)
├── docs/
│   ├── data_contract.md        # Single source of truth for data specs
│   ├── pc_specs.md             # Hardware specifications
│   └── c5_team_blueprint_artifacts_v2/  # ChatGPT genesis documents
├── scripts/                    # Analysis and baseline scripts
│   ├── baseline_evaluation.py      # 4 frequency baselines
│   ├── transition_analysis.py      # Day-over-day pattern analysis
│   ├── stochastic_sampling_baseline.py  # Oracle upper bound analysis
│   └── ...
├── _bmad/
│   ├── c5/                     # C5 Agent Team Module
│   │   ├── agents/             # 7 specialized agents
│   │   ├── teams/              # Party mode configuration
│   │   └── config.yaml
│   ├── bmm/                    # BMAD Method Module
│   ├── bmb/                    # BMAD Builder Module
│   └── core/                   # BMAD Core
├── Session_summary_2025-12-29.md   # Session notes
├── Start_Here_Tomorrow_2025-12-30.md  # Next session guide
└── README.md
```
