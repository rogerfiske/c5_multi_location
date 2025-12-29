# Architecture — C5 Multi-Location Parts Forecasting (Research)

**Version:** 2.0.0
**Last Updated:** 2025-12-29
**Owner:** Winston (System Architect)

> **Data Contract Reference:** All data definitions, schemas, and invariants are specified in [`docs/data_contract.md`](../../../../data_contract.md). This Architecture document references that contract as the single source of truth for data specifications.

> **PRD Reference:** Requirements and acceptance criteria are specified in [`PRD.md`](./PRD.md).

---

## 1. System Overview

A local research pipeline that ingests state matrices, constructs features (including cross-state pressure), trains forecasting models, backtests, and generates next-day ranked forecasts in two formats:
- **Pool**: ranked parts list (P_1..P_39 with scores)
- **Sets**: ranked candidate 5-part combinations (satisfying L_1 < L_2 < L_3 < L_4 < L_5)

### 1.1 Key Architectural Decisions (from EDA)

| Decision | Rationale | Impact |
|----------|-----------|--------|
| 30-day rolling window as default | 98.3% part coverage, 70-80% exact match per location | Feature engineering, baseline models |
| Cascade model support | Adjacent value tendency (5x random) is exploitable | Model interface, set generation |
| No duplicate set filtering | 99.37% unique sets historically | Simplifies set generation |
| Consensus as concurrent signal | Cross-state correlation ~0, not predictive | Feature priority |

---

## 2. Repository Structure (Target)

```
c5_multi_location/
├── data/
│   ├── raw/                    # Source CSVs (per data contract)
│   │   ├── CA5_matrix.csv      # Target state
│   │   ├── CA5_aggregated_matrix.csv  # Cross-state aggregation
│   │   ├── {MI,MO,NY,OH,MD}5_matrix.csv  # Supporting states
│   │   └── ME5_matrix.csv      # Excluded (shorter history)
│   └── processed/              # Cleaned, aligned, feature-ready tables
├── src/
│   ├── ingest/                 # Parsing, alignment, validation
│   ├── features/               # Feature engineering (see Section 5)
│   ├── models/                 # Baselines, ML models, calibration
│   │   ├── baselines/          # Frequency, rolling window, persistence
│   │   ├── cascade/            # Cascade models (L_1 -> L_2 -> ...)
│   │   └── ensemble/           # Model combination, calibration
│   ├── backtest/               # Walk-forward evaluation
│   ├── predict/                # Next-day inference + outputs
│   ├── sets/                   # Candidate set generation + scoring
│   └── utils/                  # Config, logging, seeds
├── configs/                    # YAML run configs
├── reports/                    # EDA + backtest summaries
│   ├── eda/                    # EDA outputs
│   └── logs/                   # Structured logs
├── predictions/                # Next-day outputs
├── models/                     # Trained model artifacts
├── scripts/                    # Utility scripts (EDA, fixes, etc.)
├── tests/                      # Data invariants + unit tests
└── docs/                       # Documentation
    ├── data_contract.md        # Single source of truth for data
    └── c5_team_blueprint_artifacts_v2/  # Planning artifacts
```

---

## 3. Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Ingest    │───>│   Validate   │───>│    Align    │───>│ Feature Build│
│  (CSV read) │    │ (invariants) │    │ (date join) │    │  (lags, etc) │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  │
                                                                  v
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Predict   │<───│   Backtest   │<───│ Set Generate│<───│   Modeling   │
│ (next-day)  │    │(walk-forward)│    │  (ranking)  │    │ (train/calib)│
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### 3.1 Stage Details

| Stage | Input | Output | Key Logic |
|-------|-------|--------|-----------|
| **Ingest** | Raw CSVs | Parsed DataFrames | Robust date parsing (M/D/YYYY, MM/DD/YYYY, YYYY-MM-DD) |
| **Validate** | Parsed DataFrames | Validated DataFrames | Invariants per data contract (sum=5, ascending order) |
| **Align** | Validated DataFrames | Canonical DataFrame | Date index alignment, CA + aggregated join |
| **Feature Build** | Canonical DataFrame | Feature matrix | Lags, rolling windows, temporal, cascade features |
| **Modeling** | Feature matrix | Trained models | Baselines -> ML -> calibration |
| **Set Generate** | Probabilities | Ranked sets | Pool ranking -> constrained set generation |
| **Backtest** | Models + data | Metrics | Walk-forward evaluation (Good+ rate, Recall@K) |
| **Predict** | Today's features | Next-day forecast | Pool CSV + Sets CSV + manifest |

---

## 4. Data Validation Rules

> See [`docs/data_contract.md`](../../../../data_contract.md) for complete specifications.

### 4.1 Invariants Enforced

| Rule | Check | Action on Failure |
|------|-------|-------------------|
| State daily sum(P) = 5 | `sum(P_1..P_39) == 5` | Flag and log |
| Aggregated daily sum(P) in {20,25,30,35} | Holiday tolerance | Flag regime indicator |
| Ascending order | `L_1 < L_2 < L_3 < L_4 < L_5` | Fail validation |
| Part range | `1 <= L_i <= 39` | Fail validation |
| No duplicate dates | Unique date index | Sum (aggregated) or flag (state) |

---

## 5. Feature Engineering Specification

### 5.1 Feature Categories (from PRD Section 7.1)

| Category | Features | Window/Params | Priority |
|----------|----------|---------------|----------|
| **Lag Features** | L_1..L_5 lag-1, P_1..P_39 lag-1 | 1 day | HIGH |
| **Rolling Presence** | P_i rolling sum | 3, 7, 30 days | HIGH |
| **Recent Frequency** | P_i count in window / window size | 30 days | HIGH |
| **Frequency Momentum** | Recent (7d) - Recent (30d) | 7d, 30d | MEDIUM |
| **Temporal** | Day of week (one-hot), month, year | - | MEDIUM |
| **Cyclical Temporal** | sin/cos day-of-year | 365-day cycle | LOW |
| **Adjacent Weights** | Boost for yesterday's L_i +/- 1 | - | HIGH |
| **Gap Features** | L_2-L_1, L_3-L_2, L_4-L_3, L_5-L_4 lag-1 | 1 day | MEDIUM |
| **Aggregated Consensus** | P_i aggregated value (0-5) lag-1 | 1 day | MEDIUM |
| **Percentile Position** | Where L_i falls in historical distribution | Per position | HIGH |

### 5.2 EDA-Derived Feature Recommendations

| Feature | Rationale | Expected Impact |
|---------|-----------|-----------------|
| **30-day rolling presence** | 98.3% coverage, optimal balance | Pool recall |
| **Adjacent tendency boost** | 21% rate vs 8% random for L_1, L_5 | Set accuracy |
| **Percentile boundaries** | 92% capture thresholds per position | Pool pruning |
| **Lag-1 L values** | Direct input for cascade models | Cascade accuracy |

### 5.3 Feature Matrix Schema

```python
# Feature matrix columns (example)
feature_columns = [
    # Lag features (45 columns)
    'L_1_lag1', 'L_2_lag1', 'L_3_lag1', 'L_4_lag1', 'L_5_lag1',
    'P_1_lag1', ..., 'P_39_lag1',

    # Rolling features (39 * 3 = 117 columns)
    'P_1_roll3', ..., 'P_39_roll3',
    'P_1_roll7', ..., 'P_39_roll7',
    'P_1_roll30', ..., 'P_39_roll30',

    # Temporal features (7 + 12 + 1 = 20 columns)
    'dow_0', ..., 'dow_6',  # Day of week one-hot
    'month_1', ..., 'month_12',  # Month one-hot
    'year',

    # Gap features (4 columns)
    'gap_L1_L2_lag1', 'gap_L2_L3_lag1', 'gap_L3_L4_lag1', 'gap_L4_L5_lag1',

    # Aggregated consensus (39 columns)
    'P_1_agg_lag1', ..., 'P_39_agg_lag1',
]
# Total: ~260 features (before selection)
```

---

## 6. Model Architecture

### 6.1 Model Interfaces

```python
class BaseModel(Protocol):
    """Base interface for all models"""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, config: dict) -> 'BaseModel':
        """Train model on feature matrix X and target y"""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability scores for each part (shape: [n_samples, 39])"""
        ...

class PoolRanker(Protocol):
    """Interface for pool ranking"""

    def rank_parts(self, proba: np.ndarray, k: int = 20) -> pd.DataFrame:
        """Return top-k parts ranked by score

        Returns DataFrame with columns: part_id, rank, score
        """
        ...

class SetGenerator(Protocol):
    """Interface for set generation"""

    def generate_sets(
        self,
        ranked_parts: pd.DataFrame,
        proba: np.ndarray,
        n_sets: int = 20,
        config: dict = None
    ) -> pd.DataFrame:
        """Generate candidate 5-part sets

        Returns DataFrame with columns: set_id, L_1, L_2, L_3, L_4, L_5, score
        """
        ...
```

### 6.2 Model Types

| Type | Description | Use Case |
|------|-------------|----------|
| **FrequencyBaseline** | Rank by historical/rolling frequency | Baseline comparison |
| **AdjacentBoostModel** | Weight yesterday's L +/- 1 higher | Exploit adjacent tendency |
| **CascadeModel** | Predict L_1, filter, predict L_2, ... | Sequential prediction |
| **TreeEnsemble** | LightGBM/XGBoost on feature matrix | ML baseline |
| **CalibratedEnsemble** | Isotonic/Platt calibration on ensemble | Production model |

### 6.3 Cascade Model Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Predict L_1 │───>│ Filter Pool  │───>│  Predict L_2 │───> ...
│  (1-30 range)│    │ (L_2 > L_1)  │    │  (2-35 range)│
└──────────────┘    └──────────────┘    └──────────────┘
       │                                        │
       v                                        v
  92% threshold: <= 15                    92% threshold: <= 24
```

**Cascade Filtering Logic:**
- If L_1 = 10, then L_2 candidates = {11, 12, ..., 39}
- If L_2 = 15, then L_3 candidates = {16, 17, ..., 39}
- Dramatically reduces search space at each step

---

## 7. Set Generation Strategy

### 7.1 Constraints (from Data Contract)

- L_1 < L_2 < L_3 < L_4 < L_5 (ascending order, mandatory)
- All L_i in range [1, 39]
- 5 unique parts per set

### 7.2 Generation Methods

| Method | Description | Trade-off |
|--------|-------------|-----------|
| **Top-K Greedy** | Take top parts respecting constraints | Fast, may miss diversity |
| **Stochastic Sampling** | Sample proportional to scores | More diverse, slower |
| **Beam Search** | Expand top-k candidates at each position | Balanced |
| **Cascade Sequential** | Use cascade model predictions directly | Exploits structure |

### 7.3 Diversity Considerations

- EDA showed 99.37% unique historical sets
- No need to filter duplicates
- Focus on score maximization, not novelty

---

## 8. Evaluation Architecture

### 8.1 Backtest Configuration

```yaml
# configs/backtest_config.yaml
backtest:
  method: rolling_origin  # walk-forward
  holdout_events: 500     # minimum for significance
  holdout_months: 12      # ~365 events
  train_window: expanding # or fixed_length
  horizon: 1              # 1-day ahead

metrics:
  pool:
    - recall_at_k: [10, 15, 20, 25, 30]
    - precision_at_k: [10, 15, 20, 25, 30]
    - pr_auc: true
    - calibration_ece: true

  sets:
    - good_plus_rate: true  # (0-1 wrong) + (4-5 wrong) / total
    - exact_match_rate: true
    - partial_overlap_distribution: true
```

### 8.2 Good+ Rate Calculation

```python
def calculate_good_plus_rate(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """
    Calculate Good+ rate per PRD Section 9.2

    Good+ = events with 0-1 wrong OR 4-5 wrong (inverted signal)
    """
    results = []
    for pred_set, actual_set in zip(predictions, actuals):
        wrong_count = len(set(pred_set) - set(actual_set))
        is_good_plus = wrong_count <= 1 or wrong_count >= 4
        results.append(is_good_plus)

    return sum(results) / len(results)
```

---

## 9. Output Specifications

### 9.1 Pool Output Schema

**File:** `predictions/ca_pool_next_day.csv`

| Column | Type | Description |
|--------|------|-------------|
| `part_id` | int | Part number (1-39) |
| `rank` | int | Rank (1 = highest score) |
| `score` | float | Calibrated probability or score |
| `confidence_lower` | float | Optional lower bound |
| `confidence_upper` | float | Optional upper bound |

### 9.2 Sets Output Schema

**File:** `predictions/ca_sets_next_day.csv`

| Column | Type | Description |
|--------|------|-------------|
| `set_id` | int | Set identifier (1-20) |
| `L_1` | int | Part at position 1 (lowest) |
| `L_2` | int | Part at position 2 |
| `L_3` | int | Part at position 3 |
| `L_4` | int | Part at position 4 |
| `L_5` | int | Part at position 5 (highest) |
| `score` | float | Set-level score |

### 9.3 Run Manifest

**File:** `predictions/manifest_YYYYMMDD_HHMMSS.yaml`

```yaml
run:
  timestamp: "2025-12-29T10:00:00"
  git_hash: "abc123"
  config_file: "configs/production.yaml"

data:
  train_start: "2008-09-08"
  train_end: "2025-12-28"
  predict_date: "2025-12-29"
  n_train_events: 6317

model:
  type: "CalibratedEnsemble"
  version: "1.0.0"

outputs:
  pool_file: "ca_pool_next_day.csv"
  sets_file: "ca_sets_next_day.csv"
```

---

## 10. Reproducibility

### 10.1 Seed Management

```python
# src/utils/seeds.py
import random
import numpy as np

def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed) if using PyTorch
    # sklearn uses numpy random state
```

### 10.2 Artifacts to Persist

| Artifact | Location | Purpose |
|----------|----------|---------|
| Run configs | `configs/` | Reproduce settings |
| Model artifacts | `models/` | Reproduce predictions |
| Data hashes | Manifest | Verify data integrity |
| Git hash | Manifest | Code version |
| Random seeds | Config | Reproduce randomness |

---

## 11. Observability (Local)

### 11.1 Logging

- Structured JSON logs to `reports/logs/`
- Log levels: DEBUG, INFO, WARNING, ERROR
- Key events: data load, validation, train start/end, predict

### 11.2 Monitoring Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `data_rows_loaded` | Rows ingested | < expected |
| `validation_failures` | Invariant violations | > 0 |
| `train_time_seconds` | Model training time | > 900 (15 min) |
| `prediction_confidence_mean` | Average score | < 0.1 or > 0.9 |

---

## 12. Security/Privacy

- Local-only research; no external services required
- No PII in data (part numbers only)
- No network access needed for core pipeline

---

## 13. Change Log

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-12-XX | 1.0.0 | Initial architecture from ChatGPT genesis | Original |
| 2025-12-29 | 2.0.0 | Added EDA-derived decisions, feature specs, cascade model, data contract reference | Winston |

---

## 14. References

- [`docs/data_contract.md`](../../../../data_contract.md) — Single source of truth for data definitions
- [`PRD.md`](./PRD.md) — Product Requirements Document
- `scripts/comprehensive_eda.py` — EDA analysis code
- `scripts/recency_adjacency_analysis.py` — Recency and pattern analysis code
