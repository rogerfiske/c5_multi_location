# Data Contract — C5 Multi-Location Parts Forecasting

**Version:** 1.1.0
**Last Updated:** 2025-12-29
**Owner:** Iris (Data Steward)

This document is the **single source of truth** for all data definitions, schemas, invariants, and quality rules. Both PRD.md and Architecture.md reference this contract.

---

## 1. Canonical Date Range

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Start Date** | 2008-09-08 | Earliest date with consistent data across all 6 states |
| **End Date** | Present (ongoing) | Data collection continues |
| **Do NOT extend backward** | Decision logged | Pre-2008 data exists for some states but lacks cross-state consistency |

---

## 2. Source Files

### 2.1 State Matrix Files

| File | State | Location | Rows (approx) | Status |
|------|-------|----------|---------------|--------|
| `CA5_matrix.csv` | California | `data/raw/` | 6,318 | **Target state** |
| `MI5_matrix.csv` | Michigan | `data/raw/` | 6,313 | Supporting |
| `MO5_matrix.csv` | Missouri | `data/raw/` | 6,318 | Supporting |
| `NY5_matrix.csv` | New York | `data/raw/` | 6,316 | Supporting |
| `OH5_matrix.csv` | Ohio | `data/raw/` | 6,318 | Supporting |
| `MD5_matrix.csv` | Maryland | `data/raw/` | 6,317 | Supporting |
| `ME5_matrix.csv` | Maine | `data/raw/` | 2,542 | **Excluded** (shorter history, starts 2013-05-15) |

### 2.2 Aggregated File

| File | Description | Location |
|------|-------------|----------|
| `CA5_aggregated_matrix.csv` | 6-state aggregation (CA, MI, MO, NY, OH, MD) | `data/raw/` |

**Note:** Filename corrected from `CA5_agregated_matrix.csv` on 2025-12-28.

### 2.3 Raw Data Files (Reference Only)

| File | Description | Status |
|------|-------------|--------|
| `CA5_raw_data.txt` | Original CA data from 1992 (11,658 rows) | **FIXED 2025-12-29** - 71 ascending violations corrected. Available for matrix expansion if needed. |
| `*_raw_data.Txt` | Other state raw files | Reference only |

---

## 3. Column Schema

### 3.1 State Matrix Files

| Column | Type | Description | Valid Range |
|--------|------|-------------|-------------|
| `date` | Date | Daily timestamp | 2008-09-08 to present |
| `L_1` | Integer | Part number at Location 1 | 1-39 |
| `L_2` | Integer | Part number at Location 2 | 1-39 |
| `L_3` | Integer | Part number at Location 3 | 1-39 |
| `L_4` | Integer | Part number at Location 4 | 1-39 |
| `L_5` | Integer | Part number at Location 5 | 1-39 |
| `P_1` to `P_39` | Binary (0/1) | Part indicator: 1 if part was required that day | 0 or 1 |

### 3.2 Aggregated Matrix File

| Column | Type | Description | Valid Range |
|--------|------|-------------|-------------|
| `date` | Date | Daily timestamp | 2008-09-08 to present |
| `L_1` to `L_5` | Integer | **CA only** location assignments | 1-39 |
| `P_1` to `P_39` | Integer (count) | Count of states requiring that part | 0-6 |

---

## 4. Data Invariants

### 4.1 Location Ordering Constraint (CRITICAL)

In all **matrix files**, location values are **always in strictly ascending order**:

```
L_1 < L_2 < L_3 < L_4 < L_5
```

- No duplicate parts on the same day for a given state
- This constraint reduces the search space for set generation
- Raw data files may NOT have this ordering (sorting was applied during matrix creation)

### 4.2 Daily Sum Invariants

| File Type | Expected Daily Sum | Column | Tolerance |
|-----------|-------------------|--------|-----------|
| State matrix | `sum(P_1..P_39) = 5` | P_* columns | Exact (5 parts per day) |
| Aggregated matrix | `sum(P_1..P_39) = 30` | P_* columns | Exceptions documented below |

### 4.3 Known Exceptions

| Exception Type | Cause | Expected Values | Affected Files |
|----------------|-------|-----------------|----------------|
| Holiday gaps | Some states closed on holidays | Sum = 20, 25 (missing 1-2 states) | Aggregated matrix |
| Duplicate dates | Data entry errors | Sum = 35 (extra state day) | Aggregated matrix |

**Note:** California (CA) operates 365 days/year with **no holiday gaps**.

---

## 5. Data Quality Rules

### 5.1 Date Parsing

- Accept multiple formats: `M/D/YYYY`, `MM/DD/YYYY`, `YYYY-MM-DD`
- Normalize to ISO 8601 (`YYYY-MM-DD`) in processed outputs
- Flag and log any unparseable dates

### 5.2 Duplicate Date Handling

| Policy | Action |
|--------|--------|
| Detection | Log all duplicate dates with affected file |
| Resolution | **Sum** the part counts (for aggregated) or **Flag for review** (for state) |
| Documentation | Record in validation report |

### 5.3 Holiday Gap Handling

| Policy | Action |
|--------|--------|
| Detection | Identify days where aggregated sum < 30 |
| Resolution | **Do NOT skip** - include in training data |
| Documentation | Flag as regime indicator if beneficial for modeling |

### 5.4 Missing Data Handling

| Scenario | Action |
|----------|--------|
| Missing row (date gap) | Log the gap; do not impute |
| Missing column value | Fail validation; require manual review |

---

## 6. Derived Outputs

### 6.1 Processed Files

| File | Location | Description |
|------|----------|-------------|
| `ca_features.parquet` | `data/processed/` | Feature-engineered CA data (TBD) |

### 6.2 Prediction Outputs

| File | Location | Description |
|------|----------|-------------|
| `ca_pool_next_day.csv` | `predictions/` | Ranked pool of likely parts |
| `ca_sets_next_day.csv` | `predictions/` | Ranked 5-part candidate sets |

---

## 7. Validation Test Requirements

All data transformations must maintain invariants. The following tests are required:

```python
# tests/data_invariants_test.py (to be implemented)

def test_state_daily_sum_equals_five():
    """Each state matrix row should have exactly 5 parts (sum of P_* = 5)"""
    pass

def test_location_values_ascending():
    """L_1 < L_2 < L_3 < L_4 < L_5 for all rows"""
    pass

def test_no_duplicate_parts_per_day():
    """All L_* values unique within each row"""
    pass

def test_part_values_in_range():
    """All L_* and non-zero P_* indices in range 1-39"""
    pass

def test_aggregated_sum_tolerances():
    """Aggregated sum in {20, 25, 30, 35} with exceptions logged"""
    pass
```

---

## 8. Change Log

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-28 | 1.0.0 | Initial data contract created | Iris |
| 2025-12-28 | 1.0.0 | Renamed `CA5_agregated_matrix.csv` -> `CA5_aggregated_matrix.csv` | Dev |
| 2025-12-29 | 1.1.0 | Fixed 71 ascending violations in `CA5_raw_data.txt` | Dev |
| 2025-12-29 | 1.1.0 | Fixed 2 ascending violations in `CA5_matrix.csv` (rows 1021, 1974) | Dev |
| 2025-12-29 | 1.1.0 | Fixed 2 ascending violations in `CA5_aggregated_matrix.csv` (rows 1021, 1974) | Dev |

---

## 9. References

- PRD.md - Product Requirements Document
- Architecture.md - System Architecture
- team_overview.md - Original team blueprint with data contract draft
