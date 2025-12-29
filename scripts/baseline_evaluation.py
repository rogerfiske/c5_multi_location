"""
Baseline Evaluation for C5 Multi-Location Parts Forecasting
Establishes performance floor with multiple baseline strategies.

Baselines:
1. Global Frequency - Rank parts by historical occurrence rate
2. Rolling Frequency (30-day) - Rank by recent occurrence rate
3. Persistence - Predict yesterday's parts
4. Adjacent Tendency - Weight yesterday's L values +/- 1

Metrics:
- Good+ Rate: (0-1 wrong) + (4-5 wrong) / total
- Recall@K: Fraction of actual parts captured in top-K pool
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Configuration
HOLDOUT_DAYS = 365  # Last year as holdout
POOL_SIZES = [10, 15, 20, 25, 30]
NUM_SETS = 20  # Number of candidate sets to generate


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def split_data(df, holdout_days=365):
    """Split into train and holdout"""
    cutoff_idx = len(df) - holdout_days
    train = df.iloc[:cutoff_idx].copy()
    holdout = df.iloc[cutoff_idx:].copy()
    return train, holdout


def get_actual_parts(row):
    """Extract actual parts from a row"""
    return set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def get_actual_set(row):
    """Extract actual set as sorted tuple"""
    return tuple(sorted([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']]))


# ============================================================
# BASELINE 1: Global Frequency
# ============================================================

def global_frequency_baseline(train_df):
    """
    Rank parts by their global frequency in training data.
    Returns: dict mapping part_id to frequency score
    """
    p_cols = [f'P_{i}' for i in range(1, 40)]
    freq = train_df[p_cols].sum()
    total = len(train_df)

    # Convert to dict: part_id -> frequency
    scores = {}
    for col in p_cols:
        part_id = int(col.replace('P_', ''))
        scores[part_id] = freq[col] / total

    return scores


# ============================================================
# BASELINE 2: Rolling Frequency (30-day)
# ============================================================

def rolling_frequency_baseline(df, current_idx, window=30):
    """
    Rank parts by their frequency in the last `window` days.
    Returns: dict mapping part_id to frequency score
    """
    p_cols = [f'P_{i}' for i in range(1, 40)]

    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    if len(window_df) == 0:
        # Fallback to uniform
        return {i: 1/39 for i in range(1, 40)}

    freq = window_df[p_cols].sum()
    total = len(window_df)

    scores = {}
    for col in p_cols:
        part_id = int(col.replace('P_', ''))
        scores[part_id] = freq[col] / total

    return scores


# ============================================================
# BASELINE 3: Persistence (Yesterday's Parts)
# ============================================================

def persistence_baseline(df, current_idx):
    """
    Predict yesterday's parts will repeat.
    Returns: dict mapping part_id to score (1.0 for yesterday's parts, 0 otherwise)
    """
    if current_idx == 0:
        return {i: 1/39 for i in range(1, 40)}

    prev_row = df.iloc[current_idx - 1]
    prev_parts = get_actual_parts(prev_row)

    scores = {}
    for i in range(1, 40):
        scores[i] = 1.0 if i in prev_parts else 0.0

    return scores


# ============================================================
# BASELINE 4: Adjacent Tendency
# ============================================================

def adjacent_tendency_baseline(df, current_idx, train_freq):
    """
    Boost parts that are adjacent (+/- 1) to yesterday's L values.
    Combines with global frequency as base.
    """
    if current_idx == 0:
        return train_freq.copy()

    prev_row = df.iloc[current_idx - 1]
    prev_l_values = [prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                     prev_row['L_4'], prev_row['L_5']]

    # Adjacent parts (within +/- 1 of any previous L value)
    adjacent_parts = set()
    for l_val in prev_l_values:
        for adj in [l_val - 1, l_val, l_val + 1]:
            if 1 <= adj <= 39:
                adjacent_parts.add(adj)

    # Boost adjacent parts
    scores = train_freq.copy()
    boost_factor = 2.0  # Double the score for adjacent parts

    for part in adjacent_parts:
        scores[part] = scores.get(part, 0) * boost_factor

    # Normalize
    total = sum(scores.values())
    if total > 0:
        scores = {k: v/total for k, v in scores.items()}

    return scores


# ============================================================
# SET GENERATION
# ============================================================

def generate_set_from_scores(scores, method='greedy'):
    """
    Generate a valid 5-part set from scores.
    Must satisfy L_1 < L_2 < L_3 < L_4 < L_5
    """
    # Sort parts by score descending
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    if method == 'greedy':
        # Greedy: take highest scoring parts that form valid ascending set
        selected = []
        for part_id, score in ranked:
            if len(selected) == 5:
                break
            # Check if this part can be added (must be > all selected)
            if not selected or part_id > max(selected):
                selected.append(part_id)
            elif part_id < min(selected):
                # Can insert at beginning
                selected.insert(0, part_id)
            else:
                # Try to insert in correct position
                for i in range(len(selected)):
                    if i == 0 and part_id < selected[i]:
                        selected.insert(0, part_id)
                        break
                    elif i > 0 and selected[i-1] < part_id < selected[i]:
                        selected.insert(i, part_id)
                        break

            # Trim to 5 if needed
            if len(selected) > 5:
                selected = sorted(selected)[:5]

        # If we don't have 5, fill with remaining parts
        if len(selected) < 5:
            selected = sorted(selected)
            remaining = [p for p in range(1, 40) if p not in selected]
            while len(selected) < 5:
                for p in remaining:
                    if not selected or p > selected[-1]:
                        selected.append(p)
                        break
                    elif p < selected[0]:
                        selected.insert(0, p)
                        selected = selected[:5]
                        break
                else:
                    # Fallback: just take any valid set
                    break

        return tuple(sorted(selected)[:5])

    return tuple(sorted(list(scores.keys())[:5]))


def generate_multiple_sets(scores, n_sets=20):
    """
    Generate multiple candidate sets using stochastic sampling.
    """
    sets = []

    # Always include greedy best
    greedy_set = generate_set_from_scores(scores, method='greedy')
    sets.append(greedy_set)

    # Generate additional sets with randomization
    np.random.seed(42)
    parts = list(range(1, 40))
    probs = np.array([scores.get(p, 0) for p in parts])
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(39) / 39

    attempts = 0
    max_attempts = n_sets * 10

    while len(sets) < n_sets and attempts < max_attempts:
        attempts += 1

        # Sample 5 parts weighted by score
        sampled = np.random.choice(parts, size=8, replace=False, p=probs)
        sampled = sorted(sampled)

        # Take first 5 that form valid ascending set
        valid_set = []
        for p in sampled:
            if not valid_set or p > valid_set[-1]:
                valid_set.append(p)
            if len(valid_set) == 5:
                break

        if len(valid_set) == 5:
            candidate = tuple(valid_set)
            if candidate not in sets:
                sets.append(candidate)

    # Fill remaining with variations if needed
    while len(sets) < n_sets:
        # Random valid set
        sampled = sorted(np.random.choice(parts, size=5, replace=False))
        candidate = tuple(sampled)
        if candidate not in sets:
            sets.append(candidate)

    return sets[:n_sets]


# ============================================================
# METRICS
# ============================================================

def calculate_recall_at_k(scores, actual_parts, k):
    """Calculate Recall@K: fraction of actual parts in top-K ranked parts"""
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_k = set([p[0] for p in ranked[:k]])
    hits = len(actual_parts.intersection(top_k))
    return hits / len(actual_parts)


def calculate_set_accuracy(pred_set, actual_set):
    """
    Calculate how many parts are wrong in predicted set.
    Returns: (wrong_count, is_good_plus)
    """
    pred = set(pred_set)
    actual = set(actual_set)
    wrong = len(pred - actual)
    is_good_plus = wrong <= 1 or wrong >= 4
    return wrong, is_good_plus


def evaluate_baseline(df, train_end_idx, baseline_fn, baseline_name, train_freq=None):
    """
    Evaluate a baseline on holdout data.

    Returns dict with metrics.
    """
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    print(f"\n--- Evaluating: {baseline_name} ---")
    print(f"Holdout period: {holdout_df['date'].iloc[0].strftime('%Y-%m-%d')} to {holdout_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Holdout events: {n_holdout}")

    # Metrics accumulators
    recall_at_k = {k: [] for k in POOL_SIZES}
    wrong_counts = []
    good_plus_count = 0

    for i in range(n_holdout):
        global_idx = train_end_idx + i

        # Get predictions
        if baseline_name == 'Adjacent Tendency':
            scores = baseline_fn(df, global_idx, train_freq)
        elif baseline_name in ['Rolling Frequency (30d)']:
            scores = baseline_fn(df, global_idx, window=30)
        elif baseline_name == 'Persistence':
            scores = baseline_fn(df, global_idx)
        else:
            # Global frequency - use pre-computed
            scores = train_freq

        # Get actual parts
        actual_row = df.iloc[global_idx]
        actual_parts = get_actual_parts(actual_row)
        actual_set = get_actual_set(actual_row)

        # Pool metrics: Recall@K
        for k in POOL_SIZES:
            recall = calculate_recall_at_k(scores, actual_parts, k)
            recall_at_k[k].append(recall)

        # Set metrics: Generate best set and compare
        pred_set = generate_set_from_scores(scores, method='greedy')
        wrong, is_good_plus = calculate_set_accuracy(pred_set, actual_set)
        wrong_counts.append(wrong)
        if is_good_plus:
            good_plus_count += 1

    # Aggregate metrics
    results = {
        'baseline': baseline_name,
        'holdout_events': n_holdout,
        'good_plus_rate': good_plus_count / n_holdout * 100,
        'wrong_distribution': Counter(wrong_counts),
    }

    for k in POOL_SIZES:
        results[f'recall_at_{k}'] = np.mean(recall_at_k[k]) * 100

    # Print results
    print(f"\nPool Metrics (Recall@K):")
    for k in POOL_SIZES:
        print(f"  Recall@{k}: {results[f'recall_at_{k}']:.2f}%")

    print(f"\nSet Metrics:")
    print(f"  Good+ Rate: {results['good_plus_rate']:.2f}%")
    print(f"  Wrong distribution:")
    for wrong in range(6):
        count = results['wrong_distribution'].get(wrong, 0)
        pct = count / n_holdout * 100
        label = "GOOD" if wrong <= 1 or wrong >= 4 else ""
        print(f"    {wrong} wrong: {count:4d} ({pct:5.2f}%) {label}")

    return results


def main():
    print("="*70)
    print("BASELINE EVALUATION - C5 Multi-Location Parts Forecasting")
    print("="*70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Total events: {len(df)}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Split data
    train_end_idx = len(df) - HOLDOUT_DAYS
    train_df = df.iloc[:train_end_idx]
    print(f"\nTrain events: {len(train_df)}")
    print(f"Holdout events: {HOLDOUT_DAYS}")

    # Pre-compute global frequency
    global_freq = global_frequency_baseline(train_df)

    # Evaluate baselines
    results = []

    # 1. Global Frequency
    r1 = evaluate_baseline(
        df, train_end_idx,
        lambda d, i: global_freq,
        "Global Frequency",
        train_freq=global_freq
    )
    results.append(r1)

    # 2. Rolling Frequency (30-day)
    r2 = evaluate_baseline(
        df, train_end_idx,
        rolling_frequency_baseline,
        "Rolling Frequency (30d)",
        train_freq=global_freq
    )
    results.append(r2)

    # 3. Persistence
    r3 = evaluate_baseline(
        df, train_end_idx,
        persistence_baseline,
        "Persistence",
        train_freq=global_freq
    )
    results.append(r3)

    # 4. Adjacent Tendency
    r4 = evaluate_baseline(
        df, train_end_idx,
        adjacent_tendency_baseline,
        "Adjacent Tendency",
        train_freq=global_freq
    )
    results.append(r4)

    # Summary table
    print("\n" + "="*70)
    print("BASELINE COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Baseline':<25} {'Good+ Rate':>12} {'Recall@20':>12} {'Recall@30':>12}")
    print("-" * 65)
    for r in results:
        print(f"{r['baseline']:<25} {r['good_plus_rate']:>11.2f}% {r['recall_at_20']:>11.2f}% {r['recall_at_30']:>11.2f}%")

    # Find best baseline
    best_good_plus = max(results, key=lambda x: x['good_plus_rate'])
    best_recall = max(results, key=lambda x: x['recall_at_20'])

    print(f"\nBest Good+ Rate: {best_good_plus['baseline']} ({best_good_plus['good_plus_rate']:.2f}%)")
    print(f"Best Recall@20: {best_recall['baseline']} ({best_recall['recall_at_20']:.2f}%)")

    # Acceptance criteria check
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*70)
    print(f"\nTarget Good+ Rate: >= 35%")
    print(f"Best Baseline Good+ Rate: {best_good_plus['good_plus_rate']:.2f}%")

    if best_good_plus['good_plus_rate'] >= 35:
        print("STATUS: Baseline MEETS target (unexpected - implies easy problem)")
    elif best_good_plus['good_plus_rate'] >= 25:
        print("STATUS: Baseline provides reasonable floor, improvement needed")
    else:
        print("STATUS: Baseline is low, significant improvement potential")

    return results


if __name__ == "__main__":
    results = main()
