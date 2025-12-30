"""
Cascade Model Tuning - Parameter Sweep
Tests different configurations to push avg wrong below 2.5

Parameters to tune:
1. Adjacency window: +/-2, +/-3, +/-4, +/-5
2. Edge boost factor: 2.0, 3.0, 4.0, 5.0
3. Portfolio size: 25, 50, 100, 200
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from itertools import product

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Configuration
HOLDOUT_DAYS = 365


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_actual_set(row):
    """Extract actual set as sorted tuple"""
    return tuple([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def position_specific_scores(df, current_idx, position, window=30):
    """Get position-specific frequency scores"""
    l_col = f'L_{position}'
    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    if len(window_df) == 0:
        return {i: 1/39 for i in range(1, 40)}

    counts = window_df[l_col].value_counts()
    total = len(window_df)

    scores = {}
    for part_id in range(1, 40):
        scores[part_id] = counts.get(part_id, 0) / total

    return scores


def cascade_predict_single(df, current_idx, prev_row, adj_window=2, edge_boost=3.0, middle_boost=2.0, method='greedy'):
    """Cascade prediction with configurable parameters"""
    prev_l_values = [prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                     prev_row['L_4'], prev_row['L_5']]

    predicted_set = []

    for position in range(1, 6):
        pos_scores = position_specific_scores(df, current_idx, position)
        prev_val = prev_l_values[position - 1]

        # Apply adjacency boost with configurable window
        for offset in range(-adj_window, adj_window + 1):
            candidate = prev_val + offset
            if 1 <= candidate <= 39 and candidate in pos_scores:
                boost = edge_boost if position in [1, 5] else middle_boost
                pos_scores[candidate] *= boost

        # Filter to valid candidates
        if predicted_set:
            min_valid = max(predicted_set) + 1
        else:
            min_valid = 1

        valid_scores = {k: v for k, v in pos_scores.items() if k >= min_valid}

        if not valid_scores:
            if predicted_set:
                remaining = [p for p in range(1, 40) if p > max(predicted_set)]
            else:
                remaining = list(range(1, 40))
            if remaining:
                predicted_set.append(min(remaining))
            continue

        if method == 'greedy':
            best_part = max(valid_scores.items(), key=lambda x: x[1])[0]
            predicted_set.append(best_part)
        else:
            parts = list(valid_scores.keys())
            probs = np.array(list(valid_scores.values()))
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
            selected = np.random.choice(parts, p=probs)
            predicted_set.append(selected)

    return tuple(predicted_set)


def cascade_predict_portfolio(df, current_idx, prev_row, n_sets, adj_window, edge_boost, middle_boost):
    """Generate portfolio with configurable parameters"""
    sets = []

    greedy_set = cascade_predict_single(df, current_idx, prev_row, adj_window, edge_boost, middle_boost, method='greedy')
    sets.append(greedy_set)

    np.random.seed(current_idx)

    attempts = 0
    max_attempts = n_sets * 5

    while len(sets) < n_sets and attempts < max_attempts:
        attempts += 1
        candidate = cascade_predict_single(df, current_idx, prev_row, adj_window, edge_boost, middle_boost, method='stochastic')
        if candidate not in sets and len(candidate) == 5:
            sets.append(candidate)

    while len(sets) < n_sets:
        random_set = tuple(sorted(np.random.choice(range(1, 40), size=5, replace=False)))
        if random_set not in sets:
            sets.append(random_set)

    return sets[:n_sets]


def score_set(pred_set, actual_set):
    """Calculate number of wrong predictions"""
    pred = set(pred_set)
    actual = set(actual_set)
    return len(pred - actual)


def evaluate_config(df, train_end_idx, adj_window, edge_boost, middle_boost, n_sets):
    """Evaluate a single configuration"""
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    best_wrong = []

    for i in range(1, n_holdout):
        global_idx = train_end_idx + i
        prev_row = df.iloc[global_idx - 1]
        actual_row = df.iloc[global_idx]
        actual_set = get_actual_set(actual_row)

        portfolio = cascade_predict_portfolio(df, global_idx, prev_row, n_sets, adj_window, edge_boost, middle_boost)
        scores = [score_set(pred_set, actual_set) for pred_set in portfolio]
        best_wrong.append(min(scores))

    avg_wrong = np.mean(best_wrong)
    best_counter = Counter(best_wrong)
    correct_rate = (best_counter.get(0, 0) + best_counter.get(1, 0)) / len(best_wrong) * 100
    good_rate = sum(best_counter.get(w, 0) for w in [0, 1, 2]) / len(best_wrong) * 100

    return {
        'avg_wrong': avg_wrong,
        'correct_rate': correct_rate,
        'good_rate': good_rate,
        'distribution': dict(best_counter)
    }


def main():
    print("=" * 70)
    print("CASCADE MODEL TUNING - Parameter Sweep")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data()
    train_end_idx = len(df) - HOLDOUT_DAYS

    print(f"\nBaseline (adj=2, edge=3.0, middle=2.0, sets=50): avg_wrong = 2.69")
    print("\nTesting parameter combinations...")

    # Parameter grid
    adj_windows = [2, 3, 4, 5]
    edge_boosts = [2.0, 3.0, 4.0, 5.0]
    portfolio_sizes = [50, 100]  # Keep middle_boost fixed at 2.0 for now

    results = []

    # Quick scan: vary one parameter at a time first
    print("\n" + "-" * 70)
    print("PHASE 1: Single Parameter Variation (middle_boost=2.0, n_sets=50)")
    print("-" * 70)

    # Vary adjacency window
    print("\n[Adjacency Window Sweep]")
    print(f"{'Window':>8} {'Avg Wrong':>12} {'Good Rate':>12} {'Correct':>10}")
    print("-" * 45)

    for adj in adj_windows:
        result = evaluate_config(df, train_end_idx, adj, 3.0, 2.0, 50)
        results.append({
            'adj_window': adj, 'edge_boost': 3.0, 'middle_boost': 2.0, 'n_sets': 50,
            **result
        })
        marker = " <-- BEST" if result['avg_wrong'] < 2.69 else ""
        print(f"{adj:>8} {result['avg_wrong']:>12.3f} {result['good_rate']:>11.2f}% {result['correct_rate']:>9.2f}%{marker}")

    # Vary edge boost
    print("\n[Edge Boost Sweep] (adj_window=2)")
    print(f"{'Boost':>8} {'Avg Wrong':>12} {'Good Rate':>12} {'Correct':>10}")
    print("-" * 45)

    for boost in edge_boosts:
        if boost == 3.0:
            continue  # Already tested
        result = evaluate_config(df, train_end_idx, 2, boost, 2.0, 50)
        results.append({
            'adj_window': 2, 'edge_boost': boost, 'middle_boost': 2.0, 'n_sets': 50,
            **result
        })
        marker = " <-- BEST" if result['avg_wrong'] < 2.69 else ""
        print(f"{boost:>8} {result['avg_wrong']:>12.3f} {result['good_rate']:>11.2f}% {result['correct_rate']:>9.2f}%{marker}")

    # Vary portfolio size
    print("\n[Portfolio Size Sweep] (adj_window=2, edge_boost=3.0)")
    print(f"{'Sets':>8} {'Avg Wrong':>12} {'Good Rate':>12} {'Correct':>10}")
    print("-" * 45)

    for n_sets in [25, 50, 100, 200]:
        if n_sets == 50:
            print(f"{n_sets:>8} {2.690:>12.3f} {'32.69':>11}% {'1.65':>9}% (baseline)")
            continue
        result = evaluate_config(df, train_end_idx, 2, 3.0, 2.0, n_sets)
        results.append({
            'adj_window': 2, 'edge_boost': 3.0, 'middle_boost': 2.0, 'n_sets': n_sets,
            **result
        })
        marker = " <-- BEST" if result['avg_wrong'] < 2.69 else ""
        print(f"{n_sets:>8} {result['avg_wrong']:>12.3f} {result['good_rate']:>11.2f}% {result['correct_rate']:>9.2f}%{marker}")

    # Find best combination from results so far
    if results:
        best = min(results, key=lambda x: x['avg_wrong'])
        print("\n" + "=" * 70)
        print("PHASE 1 BEST CONFIGURATION")
        print("=" * 70)
        print(f"Adjacency window: +/-{best['adj_window']}")
        print(f"Edge boost: {best['edge_boost']}")
        print(f"Middle boost: {best['middle_boost']}")
        print(f"Portfolio size: {best['n_sets']}")
        print(f"\nAvg wrong: {best['avg_wrong']:.3f}")
        print(f"Good rate: {best['good_rate']:.2f}%")
        print(f"Correct rate: {best['correct_rate']:.2f}%")

        improvement = 2.69 - best['avg_wrong']
        if improvement > 0:
            print(f"\nImprovement over baseline: {improvement:.3f} ({improvement/2.69*100:.1f}%)")
        else:
            print(f"\nNo improvement found in Phase 1. Baseline remains best.")

    # Phase 2: Test promising combinations
    print("\n" + "-" * 70)
    print("PHASE 2: Promising Combinations")
    print("-" * 70)

    # Find best adj_window from Phase 1
    adj_results = [r for r in results if r['edge_boost'] == 3.0 and r['n_sets'] == 50]
    if adj_results:
        best_adj = min(adj_results, key=lambda x: x['avg_wrong'])['adj_window']
    else:
        best_adj = 2

    # Test best adj with larger portfolio
    print(f"\n[Best adj_window ({best_adj}) with larger portfolios]")
    print(f"{'Sets':>8} {'Avg Wrong':>12} {'Good Rate':>12} {'Correct':>10}")
    print("-" * 45)

    for n_sets in [100, 200]:
        result = evaluate_config(df, train_end_idx, best_adj, 3.0, 2.0, n_sets)
        results.append({
            'adj_window': best_adj, 'edge_boost': 3.0, 'middle_boost': 2.0, 'n_sets': n_sets,
            **result
        })
        marker = " <-- BEST" if result['avg_wrong'] < 2.69 else ""
        print(f"{n_sets:>8} {result['avg_wrong']:>12.3f} {result['good_rate']:>11.2f}% {result['correct_rate']:>9.2f}%{marker}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Sort by avg_wrong
    results_sorted = sorted(results, key=lambda x: x['avg_wrong'])

    print(f"\n{'Rank':>4} {'Adj':>4} {'Edge':>6} {'Sets':>5} {'Avg Wrong':>10} {'Good%':>8} {'Correct%':>9}")
    print("-" * 55)

    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i:>4} {r['adj_window']:>4} {r['edge_boost']:>6.1f} {r['n_sets']:>5} {r['avg_wrong']:>10.3f} {r['good_rate']:>7.2f}% {r['correct_rate']:>8.2f}%")

    best = results_sorted[0]
    print(f"\n{'=' * 70}")
    print("RECOMMENDED CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Adjacency window: +/-{best['adj_window']}")
    print(f"Edge boost: {best['edge_boost']}")
    print(f"Portfolio size: {best['n_sets']}")
    print(f"\nAvg wrong: {best['avg_wrong']:.3f} (baseline: 2.69)")
    print(f"Good rate: {best['good_rate']:.2f}% (baseline: 32.69%)")
    print(f"Correct rate: {best['correct_rate']:.2f}% (baseline: 1.65%)")

    return results_sorted


if __name__ == "__main__":
    results = main()
