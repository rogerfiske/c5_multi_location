"""
Multi-State Consensus Signal for C5 Parts Forecasting

Uses the aggregated 6-state data as an additional signal.
Key idea: If multiple states had part X yesterday, it might be a stronger signal.

Cross-state consensus as tie-breaker or boost for cascade model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Configuration
HOLDOUT_DAYS = 365
NUM_CANDIDATE_SETS = 50


def load_data():
    """Load both CA5 and aggregated matrix data"""
    ca_df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    ca_df['date'] = pd.to_datetime(ca_df['date'], format='%m/%d/%Y')
    ca_df = ca_df.sort_values('date').reset_index(drop=True)

    agg_df = pd.read_csv(DATA_DIR / "CA5_aggregated_matrix.csv")
    agg_df['date'] = pd.to_datetime(agg_df['date'], format='%m/%d/%Y')
    agg_df = agg_df.sort_values('date').reset_index(drop=True)

    return ca_df, agg_df


def get_actual_set(row):
    """Extract actual set as tuple"""
    return tuple([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def get_multistate_signal(agg_row, threshold=3):
    """
    Get parts that appeared in multiple states yesterday.

    Returns dict: part_id -> state_count (0-6)
    """
    signal = {}
    for part in range(1, 40):
        col = f'P_{part}'
        if col in agg_row.index:
            signal[part] = agg_row[col]
        else:
            signal[part] = 0
    return signal


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


def cascade_with_multistate(ca_df, agg_df, current_idx, prev_ca_row, prev_agg_row,
                             adj_window=2, edge_boost=3.0, multistate_boost=1.5,
                             multistate_threshold=3, method='greedy'):
    """
    Cascade prediction enhanced with multi-state consensus signal.

    If a part appeared in >= threshold states yesterday, boost its score.
    """
    prev_l_values = [prev_ca_row['L_1'], prev_ca_row['L_2'], prev_ca_row['L_3'],
                     prev_ca_row['L_4'], prev_ca_row['L_5']]

    # Get multi-state signal
    multistate = get_multistate_signal(prev_agg_row)

    predicted_set = []

    for position in range(1, 6):
        # Position-specific scores
        pos_scores = position_specific_scores(ca_df, current_idx, position)
        prev_val = prev_l_values[position - 1]

        # Apply adjacency boost
        for offset in range(-adj_window, adj_window + 1):
            candidate = prev_val + offset
            if 1 <= candidate <= 39 and candidate in pos_scores:
                boost = edge_boost if position in [1, 5] else 2.0
                pos_scores[candidate] *= boost

        # Apply multi-state boost
        for part, state_count in multistate.items():
            if state_count >= multistate_threshold and part in pos_scores:
                # Scale boost by number of states (3->1.5x, 4->2x, 5->2.5x, 6->3x)
                scale = 1 + (state_count - multistate_threshold + 1) * 0.5
                pos_scores[part] *= scale * multistate_boost

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


def cascade_multistate_portfolio(ca_df, agg_df, current_idx, prev_ca_row, prev_agg_row,
                                  n_sets, adj_window, edge_boost, multistate_boost, multistate_threshold):
    """Generate portfolio with multi-state enhanced cascade"""
    sets = []

    greedy_set = cascade_with_multistate(
        ca_df, agg_df, current_idx, prev_ca_row, prev_agg_row,
        adj_window, edge_boost, multistate_boost, multistate_threshold, method='greedy'
    )
    sets.append(greedy_set)

    np.random.seed(current_idx)

    attempts = 0
    max_attempts = n_sets * 5

    while len(sets) < n_sets and attempts < max_attempts:
        attempts += 1
        candidate = cascade_with_multistate(
            ca_df, agg_df, current_idx, prev_ca_row, prev_agg_row,
            adj_window, edge_boost, multistate_boost, multistate_threshold, method='stochastic'
        )
        if candidate not in sets and len(candidate) == 5:
            sets.append(candidate)

    while len(sets) < n_sets:
        random_set = tuple(sorted(np.random.choice(range(1, 40), size=5, replace=False)))
        if random_set not in sets:
            sets.append(random_set)

    return sets[:n_sets]


def score_set(pred_set, actual_set):
    """Calculate number of wrong predictions"""
    return len(set(pred_set) - set(actual_set))


def analyze_multistate_correlation(ca_df, agg_df, train_end_idx):
    """
    Analyze correlation between multi-state consensus and CA outcomes.
    """
    print("\n" + "-" * 70)
    print("MULTI-STATE CORRELATION ANALYSIS")
    print("-" * 70)

    # For each day, check if high-consensus parts appear in CA
    correlations = []

    for i in range(1, train_end_idx):
        prev_agg = agg_df.iloc[i-1]
        today_ca = ca_df.iloc[i]
        today_parts = set([today_ca['L_1'], today_ca['L_2'], today_ca['L_3'],
                          today_ca['L_4'], today_ca['L_5']])

        multistate = get_multistate_signal(prev_agg)

        # Check different thresholds
        for threshold in [3, 4, 5, 6]:
            high_consensus = set([p for p, count in multistate.items() if count >= threshold])
            if high_consensus:
                overlap = len(today_parts & high_consensus)
                correlations.append({
                    'threshold': threshold,
                    'consensus_size': len(high_consensus),
                    'overlap': overlap
                })

    # Aggregate by threshold
    print(f"\n{'Threshold':>10} {'Avg Pool Size':>15} {'Avg Overlap':>12} {'Hit Rate':>10}")
    print("-" * 50)

    for threshold in [3, 4, 5, 6]:
        subset = [c for c in correlations if c['threshold'] == threshold]
        if subset:
            avg_pool = np.mean([c['consensus_size'] for c in subset])
            avg_overlap = np.mean([c['overlap'] for c in subset])
            hit_rate = np.mean([c['overlap'] > 0 for c in subset]) * 100
            print(f"{threshold:>10} {avg_pool:>15.1f} {avg_overlap:>12.2f} {hit_rate:>9.1f}%")

    return correlations


def evaluate_multistate_model(ca_df, agg_df, train_end_idx, multistate_boost, multistate_threshold):
    """Evaluate multi-state enhanced cascade model"""
    n_holdout = len(ca_df) - train_end_idx

    best_wrong = []

    for i in range(1, n_holdout):
        global_idx = train_end_idx + i
        prev_ca_row = ca_df.iloc[global_idx - 1]
        prev_agg_row = agg_df.iloc[global_idx - 1]
        actual_row = ca_df.iloc[global_idx]
        actual_set = get_actual_set(actual_row)

        portfolio = cascade_multistate_portfolio(
            ca_df, agg_df, global_idx, prev_ca_row, prev_agg_row,
            n_sets=50, adj_window=2, edge_boost=3.0,
            multistate_boost=multistate_boost, multistate_threshold=multistate_threshold
        )

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
    print("MULTI-STATE CONSENSUS SIGNAL")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    ca_df, agg_df = load_data()
    train_end_idx = len(ca_df) - HOLDOUT_DAYS

    print(f"\nCA data: {len(ca_df)} days")
    print(f"Aggregated data: {len(agg_df)} days")
    print(f"Training: {train_end_idx} days")
    print(f"Holdout: {HOLDOUT_DAYS} days")

    # Analyze multi-state correlation
    analyze_multistate_correlation(ca_df, agg_df, train_end_idx)

    # Test different configurations
    print("\n" + "-" * 70)
    print("MULTI-STATE BOOST SWEEP")
    print("-" * 70)
    print(f"\nBaseline (cascade only): avg_wrong = 2.69")

    results = []

    # Test different thresholds and boosts
    print(f"\n{'Threshold':>10} {'Boost':>8} {'Avg Wrong':>12} {'Good Rate':>12} {'Correct':>10}")
    print("-" * 55)

    for threshold in [3, 4, 5]:
        for boost in [1.0, 1.5, 2.0]:
            if threshold == 3 and boost == 1.0:
                # This is effectively no multi-state signal
                continue

            result = evaluate_multistate_model(ca_df, agg_df, train_end_idx, boost, threshold)
            results.append({
                'threshold': threshold, 'boost': boost, **result
            })
            marker = " <-- BEST" if result['avg_wrong'] < 2.69 else ""
            print(f"{threshold:>10} {boost:>8.1f} {result['avg_wrong']:>12.3f} {result['good_rate']:>11.2f}% {result['correct_rate']:>9.2f}%{marker}")

    # Find best
    if results:
        best = min(results, key=lambda x: x['avg_wrong'])

        print("\n" + "=" * 70)
        print("MULTI-STATE MODEL RESULTS")
        print("=" * 70)
        print(f"\nBest configuration:")
        print(f"  Threshold: {best['threshold']} states")
        print(f"  Boost: {best['boost']}")
        print(f"\nAvg wrong: {best['avg_wrong']:.3f} (baseline: 2.69)")
        print(f"Good rate: {best['good_rate']:.2f}% (baseline: 32.69%)")
        print(f"Correct rate: {best['correct_rate']:.2f}% (baseline: 1.65%)")

        improvement = 2.69 - best['avg_wrong']
        if improvement > 0:
            print(f"\nImprovement: {improvement:.3f} ({improvement/2.69*100:.1f}%)")
        else:
            print(f"\nNo improvement from multi-state signal alone.")
            print("Cross-state correlation is near zero (as found in EDA).")
            print("Multi-state may work better as tie-breaker rather than primary signal.")

    return results


if __name__ == "__main__":
    results = main()
