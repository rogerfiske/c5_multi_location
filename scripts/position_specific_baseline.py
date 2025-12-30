"""
Position-Specific Cascade Model for C5 Multi-Location Parts Forecasting

Exploits the 33% adjacency signal discovered in L_1/L_5 positions.
Uses cascade prediction: L_1 → filter → L_2 → filter → ... → L_5

Key insight: Edge positions (L_1, L_5) are more predictable than interior.
- L_1 adjacency (+/-2): 33.04% (3x random)
- L_5 adjacency (+/-2): 31.45% (3x random)
- L_2/L_3/L_4 adjacency: ~20% (2x random)

Cascade approach respects the ascending constraint: L_1 < L_2 < L_3 < L_4 < L_5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Configuration
HOLDOUT_DAYS = 365
ADJACENCY_WINDOW = 2  # +/- 2 from previous day's value
NUM_CANDIDATE_SETS = 50  # Generate multiple sets for portfolio


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_actual_parts(row):
    """Extract actual parts from a row as set"""
    return set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def get_actual_set(row):
    """Extract actual set as sorted tuple"""
    return tuple([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def get_adjacency_candidates(prev_value, window=2, min_val=1, max_val=39):
    """Get candidate parts within +/- window of previous value"""
    candidates = set()
    for offset in range(-window, window + 1):
        val = prev_value + offset
        if min_val <= val <= max_val:
            candidates.add(val)
    return candidates


def rolling_frequency_scores(df, current_idx, window=30):
    """Get frequency scores for all parts in rolling window"""
    p_cols = [f'P_{i}' for i in range(1, 40)]
    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    if len(window_df) == 0:
        return {i: 1/39 for i in range(1, 40)}

    freq = window_df[p_cols].sum()
    total = len(window_df)

    scores = {}
    for col in p_cols:
        part_id = int(col.replace('P_', ''))
        scores[part_id] = freq[col] / total

    return scores


def position_specific_scores(df, current_idx, position, window=30):
    """
    Get position-specific frequency scores.
    position: 1-5 for L_1 through L_5
    """
    l_col = f'L_{position}'
    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    if len(window_df) == 0:
        return {i: 1/39 for i in range(1, 40)}

    # Count occurrences at this specific position
    counts = window_df[l_col].value_counts()
    total = len(window_df)

    scores = {}
    for part_id in range(1, 40):
        scores[part_id] = counts.get(part_id, 0) / total

    return scores


def adjacency_weighted_scores(base_scores, prev_l_values, adjacency_window=2, boost=3.0):
    """
    Boost scores for parts adjacent to previous day's L values.
    Different boost per position based on adjacency strength.
    """
    # Position-specific boost factors (based on analysis)
    # L_1 and L_5 have stronger adjacency signal
    position_boost = {1: 3.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 3.0}

    boosted = base_scores.copy()

    for pos_idx, prev_val in enumerate(prev_l_values):
        position = pos_idx + 1
        boost_factor = position_boost.get(position, 2.0)

        for offset in range(-adjacency_window, adjacency_window + 1):
            candidate = prev_val + offset
            if 1 <= candidate <= 39 and candidate in boosted:
                boosted[candidate] = boosted[candidate] * boost_factor

    # Normalize
    total = sum(boosted.values())
    if total > 0:
        boosted = {k: v/total for k, v in boosted.items()}

    return boosted


def cascade_predict_single(df, current_idx, prev_row, method='greedy'):
    """
    Cascade prediction for a single set.
    Predict L_1 → constrain → L_2 → constrain → ... → L_5

    method: 'greedy' or 'stochastic'
    """
    prev_l_values = [prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                     prev_row['L_4'], prev_row['L_5']]

    predicted_set = []

    for position in range(1, 6):
        # Get position-specific base scores
        pos_scores = position_specific_scores(df, current_idx, position)

        # Boost for adjacency to previous day's value at this position
        prev_val = prev_l_values[position - 1]
        adjacent_candidates = get_adjacency_candidates(prev_val, window=ADJACENCY_WINDOW)

        # Apply adjacency boost
        for part_id in pos_scores:
            if part_id in adjacent_candidates:
                # Edge positions get stronger boost
                boost = 3.0 if position in [1, 5] else 2.0
                pos_scores[part_id] *= boost

        # Filter to valid candidates (must be > all previously selected)
        if predicted_set:
            min_valid = max(predicted_set) + 1
        else:
            min_valid = 1

        valid_scores = {k: v for k, v in pos_scores.items() if k >= min_valid}

        if not valid_scores:
            # Fallback: take smallest available part greater than current max
            if predicted_set:
                remaining = [p for p in range(1, 40) if p > max(predicted_set)]
            else:
                remaining = list(range(1, 40))
            if remaining:
                predicted_set.append(min(remaining))
            continue

        if method == 'greedy':
            # Take highest scoring valid part
            best_part = max(valid_scores.items(), key=lambda x: x[1])[0]
            predicted_set.append(best_part)
        else:
            # Stochastic: sample proportional to score
            parts = list(valid_scores.keys())
            probs = np.array(list(valid_scores.values()))
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
            selected = np.random.choice(parts, p=probs)
            predicted_set.append(selected)

    return tuple(predicted_set)


def cascade_predict_portfolio(df, current_idx, prev_row, n_sets=50):
    """
    Generate portfolio of diverse candidate sets using cascade approach.
    """
    sets = []

    # Always include greedy prediction
    greedy_set = cascade_predict_single(df, current_idx, prev_row, method='greedy')
    sets.append(greedy_set)

    # Generate stochastic variations
    np.random.seed(current_idx)  # Reproducible per day

    attempts = 0
    max_attempts = n_sets * 5

    while len(sets) < n_sets and attempts < max_attempts:
        attempts += 1
        candidate = cascade_predict_single(df, current_idx, prev_row, method='stochastic')
        if candidate not in sets and len(candidate) == 5:
            sets.append(candidate)

    # Fill remaining with random valid sets if needed
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


def evaluate_position_accuracy(df, train_end_idx):
    """
    Evaluate per-position prediction accuracy.
    """
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    print("\n" + "=" * 70)
    print("POSITION-SPECIFIC ACCURACY ANALYSIS")
    print("=" * 70)

    position_results = {pos: {
        'exact_hits': 0,
        'adjacent_hits': 0,  # within +/- 2
        'predictions': []
    } for pos in range(1, 6)}

    for i in range(1, n_holdout):  # Start at 1 to have previous day
        global_idx = train_end_idx + i
        prev_row = df.iloc[global_idx - 1]
        actual_row = df.iloc[global_idx]

        # Predict each position
        predicted_set = cascade_predict_single(df, global_idx, prev_row, method='greedy')

        for pos in range(1, 6):
            pred_val = predicted_set[pos - 1]
            actual_val = actual_row[f'L_{pos}']

            position_results[pos]['predictions'].append((pred_val, actual_val))

            if pred_val == actual_val:
                position_results[pos]['exact_hits'] += 1
            elif abs(pred_val - actual_val) <= 2:
                position_results[pos]['adjacent_hits'] += 1

    # Print results
    n_eval = n_holdout - 1

    print(f"\nEvaluated {n_eval} days")
    print(f"\n{'Position':>10} {'Exact Match':>14} {'Adjacent (+/-2)':>16} {'Combined':>12}")
    print("-" * 55)

    for pos in range(1, 6):
        r = position_results[pos]
        exact_pct = r['exact_hits'] / n_eval * 100
        adj_pct = r['adjacent_hits'] / n_eval * 100
        combined_pct = (r['exact_hits'] + r['adjacent_hits']) / n_eval * 100

        # Mark edge positions
        marker = " *" if pos in [1, 5] else ""
        print(f"L_{pos}{marker:>8} {exact_pct:>13.2f}% {adj_pct:>15.2f}% {combined_pct:>11.2f}%")

    print("\n* Edge positions (L_1, L_5) expected to have stronger signal")

    return position_results


def evaluate_cascade_model(df, train_end_idx):
    """
    Full evaluation of cascade model with portfolio.
    """
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    print("\n" + "=" * 70)
    print("CASCADE MODEL EVALUATION (Portfolio of {} sets)".format(NUM_CANDIDATE_SETS))
    print("=" * 70)

    # Track metrics
    greedy_wrong = []
    best_wrong = []
    wrong_distribution = []

    for i in range(1, n_holdout):  # Start at 1 to have previous day
        global_idx = train_end_idx + i
        prev_row = df.iloc[global_idx - 1]
        actual_row = df.iloc[global_idx]
        actual_set = get_actual_set(actual_row)

        # Generate portfolio
        portfolio = cascade_predict_portfolio(df, global_idx, prev_row, n_sets=NUM_CANDIDATE_SETS)

        # Score each set
        scores = [score_set(pred_set, actual_set) for pred_set in portfolio]

        greedy_wrong.append(scores[0])  # First set is greedy
        best_wrong.append(min(scores))
        wrong_distribution.append(Counter(scores))

    n_eval = n_holdout - 1

    # Summary statistics
    print(f"\nEvaluated {n_eval} days")

    # Greedy performance
    greedy_counter = Counter(greedy_wrong)
    print("\n--- GREEDY CASCADE (single best prediction) ---")
    print(f"{'Wrong':>6} {'Count':>8} {'Percent':>10}")
    print("-" * 28)
    for wrong in range(6):
        count = greedy_counter.get(wrong, 0)
        pct = count / n_eval * 100
        print(f"{wrong:>6} {count:>8} {pct:>9.2f}%")

    avg_greedy = np.mean(greedy_wrong)
    print(f"\nAvg wrong (greedy): {avg_greedy:.3f}")

    # Portfolio performance
    best_counter = Counter(best_wrong)
    print("\n--- PORTFOLIO CASCADE (best of {} sets) ---".format(NUM_CANDIDATE_SETS))
    print(f"{'Wrong':>6} {'Count':>8} {'Percent':>10}")
    print("-" * 28)
    for wrong in range(6):
        count = best_counter.get(wrong, 0)
        pct = count / n_eval * 100
        marker = " <-- CORRECT" if wrong <= 1 else ""
        print(f"{wrong:>6} {count:>8} {pct:>9.2f}%{marker}")

    avg_best = np.mean(best_wrong)
    correct_rate = (best_counter.get(0, 0) + best_counter.get(1, 0)) / n_eval * 100
    good_rate = sum(best_counter.get(w, 0) for w in [0, 1, 2]) / n_eval * 100

    print(f"\nAvg wrong (best of portfolio): {avg_best:.3f}")
    print(f"Correct rate (0-1 wrong): {correct_rate:.2f}%")
    print(f"Good rate (0-2 wrong): {good_rate:.2f}%")

    return {
        'greedy_wrong': greedy_wrong,
        'best_wrong': best_wrong,
        'avg_greedy': avg_greedy,
        'avg_best': avg_best,
        'correct_rate': correct_rate,
        'good_rate': good_rate
    }


def compare_to_baseline(cascade_results):
    """
    Compare cascade model to previous baselines.
    """
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINES")
    print("=" * 70)

    # Previous baseline results (from baseline_evaluation.py)
    baselines = {
        'Rolling Frequency (30d) - Greedy': {'avg_wrong': 4.44, 'correct_rate': 0.0},
        'Stochastic Sampling Oracle (50 sets)': {'avg_wrong': 3.09, 'correct_rate': 1.10},
    }

    print(f"\n{'Model':<40} {'Avg Wrong':>12} {'Correct %':>12}")
    print("-" * 66)

    for name, metrics in baselines.items():
        print(f"{name:<40} {metrics['avg_wrong']:>12.3f} {metrics['correct_rate']:>11.2f}%")

    print(f"{'Cascade Greedy':<40} {cascade_results['avg_greedy']:>12.3f} {'N/A':>12}")
    print(f"{'Cascade Portfolio (50 sets)':<40} {cascade_results['avg_best']:>12.3f} {cascade_results['correct_rate']:>11.2f}%")

    # Improvement calculation
    baseline_avg = 3.09  # Oracle baseline
    improvement = baseline_avg - cascade_results['avg_best']
    improvement_pct = improvement / baseline_avg * 100

    print(f"\nImprovement over Oracle baseline:")
    print(f"  Avg wrong: {baseline_avg:.3f} -> {cascade_results['avg_best']:.3f} ({improvement:+.3f}, {improvement_pct:+.1f}%)")

    if cascade_results['avg_best'] < 3.0:
        print("\n[SUCCESS] TARGET MET: Avg wrong < 3.0")
    else:
        gap = cascade_results['avg_best'] - 3.0
        print(f"\n[MISS] TARGET NOT MET: Need to reduce avg wrong by {gap:.3f} more")


def main():
    print("=" * 70)
    print("POSITION-SPECIFIC CASCADE MODEL")
    print("C5 Multi-Location Parts Forecasting")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nStrategy: Predict L_1 -> filter -> L_2 -> ... -> L_5")
    print(f"Key signal: 33% adjacency for edge positions (L_1, L_5)")

    # Load data
    df = load_data()
    train_end_idx = len(df) - HOLDOUT_DAYS

    print(f"\nData: {len(df)} total days")
    print(f"Training: {train_end_idx} days")
    print(f"Holdout: {HOLDOUT_DAYS} days")

    # Evaluate position-specific accuracy
    position_results = evaluate_position_accuracy(df, train_end_idx)

    # Evaluate full cascade model
    cascade_results = evaluate_cascade_model(df, train_end_idx)

    # Compare to baselines
    compare_to_baseline(cascade_results)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. If avg_best < 3.0: Target met, document and move to sequence prediction
2. If not: Try tuning adjacency window, boost factors, or add more signals
3. Consider adding recency weighting (recent patterns weighted higher)
4. Explore conditional position models (L_2 | L_1, etc.)
""")

    return cascade_results


if __name__ == "__main__":
    results = main()
