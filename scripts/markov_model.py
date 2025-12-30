"""
Markov Transition Model for C5 Parts Forecasting

Models P(part_tomorrow | part_today) to capture temporal dependencies.
Can be combined with cascade model for additional signal.

Key idea: If part X appeared today, what's the probability it appears tomorrow?
Also: If part X appeared today, what other parts tend to appear tomorrow?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Configuration
HOLDOUT_DAYS = 365
NUM_CANDIDATE_SETS = 50


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_parts_set(row):
    """Extract parts as set"""
    return set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def get_actual_set(row):
    """Extract actual set as tuple"""
    return tuple([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def build_transition_matrix(df, end_idx):
    """
    Build Markov transition matrix from training data.

    Returns:
    - repeat_prob[part]: P(part appears tomorrow | part appeared today)
    - co_occurrence[part]: Counter of parts that appear tomorrow when part appeared today
    - overall_prob[part]: Baseline P(part appears) across all days
    """
    train_df = df.iloc[:end_idx]

    # Track transitions
    repeat_counts = defaultdict(int)  # How often part repeats next day
    appearance_counts = defaultdict(int)  # How often part appears
    co_occurrence = defaultdict(Counter)  # What appears tomorrow given today

    for i in range(1, len(train_df)):
        today_parts = get_parts_set(train_df.iloc[i-1])
        tomorrow_parts = get_parts_set(train_df.iloc[i])

        for part in today_parts:
            appearance_counts[part] += 1
            if part in tomorrow_parts:
                repeat_counts[part] += 1
            # Track co-occurrence
            for tomorrow_part in tomorrow_parts:
                co_occurrence[part][tomorrow_part] += 1

    # Calculate probabilities
    repeat_prob = {}
    for part in range(1, 40):
        if appearance_counts[part] > 0:
            repeat_prob[part] = repeat_counts[part] / appearance_counts[part]
        else:
            repeat_prob[part] = 0.0

    # Overall probability (baseline)
    p_cols = [f'P_{i}' for i in range(1, 40)]
    overall_prob = {}
    for col in p_cols:
        part_id = int(col.replace('P_', ''))
        overall_prob[part_id] = train_df[col].sum() / len(train_df)

    return repeat_prob, co_occurrence, overall_prob


def markov_scores(today_parts, repeat_prob, co_occurrence, overall_prob, alpha=0.5):
    """
    Generate scores for tomorrow's parts using Markov model.

    Score = alpha * markov_signal + (1-alpha) * baseline

    markov_signal = weighted combination of:
    - repeat probability (if part was in today)
    - co-occurrence probability (given today's parts)
    """
    scores = {}

    for part in range(1, 40):
        # Baseline
        base_score = overall_prob.get(part, 1/39)

        # Markov signal
        markov_signal = 0.0

        # Repeat component: was this part in today?
        if part in today_parts:
            markov_signal += repeat_prob.get(part, 0) * 2  # Boost repeat

        # Co-occurrence component: how often does this part follow today's parts?
        for today_part in today_parts:
            co_count = co_occurrence[today_part].get(part, 0)
            total_transitions = sum(co_occurrence[today_part].values())
            if total_transitions > 0:
                markov_signal += co_count / total_transitions

        # Normalize markov signal
        markov_signal = markov_signal / (len(today_parts) + 1)  # +1 for repeat component

        # Combine
        scores[part] = alpha * markov_signal + (1 - alpha) * base_score

    # Normalize
    total = sum(scores.values())
    if total > 0:
        scores = {k: v/total for k, v in scores.items()}

    return scores


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


def cascade_with_markov(df, current_idx, prev_row, repeat_prob, co_occurrence, overall_prob,
                        adj_window=2, edge_boost=3.0, markov_weight=0.3, method='greedy'):
    """
    Cascade prediction enhanced with Markov signal.
    """
    prev_l_values = [prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                     prev_row['L_4'], prev_row['L_5']]
    today_parts = set(prev_l_values)

    # Get Markov scores for today's context
    markov = markov_scores(today_parts, repeat_prob, co_occurrence, overall_prob)

    predicted_set = []

    for position in range(1, 6):
        # Position-specific scores
        pos_scores = position_specific_scores(df, current_idx, position)
        prev_val = prev_l_values[position - 1]

        # Apply adjacency boost
        for offset in range(-adj_window, adj_window + 1):
            candidate = prev_val + offset
            if 1 <= candidate <= 39 and candidate in pos_scores:
                boost = edge_boost if position in [1, 5] else 2.0
                pos_scores[candidate] *= boost

        # Combine with Markov signal
        combined_scores = {}
        for part in range(1, 40):
            pos_score = pos_scores.get(part, 0)
            markov_score = markov.get(part, 0)
            combined_scores[part] = (1 - markov_weight) * pos_score + markov_weight * markov_score

        # Filter to valid candidates
        if predicted_set:
            min_valid = max(predicted_set) + 1
        else:
            min_valid = 1

        valid_scores = {k: v for k, v in combined_scores.items() if k >= min_valid}

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


def cascade_markov_portfolio(df, current_idx, prev_row, repeat_prob, co_occurrence, overall_prob,
                              n_sets, adj_window, edge_boost, markov_weight):
    """Generate portfolio with Markov-enhanced cascade"""
    sets = []

    greedy_set = cascade_with_markov(df, current_idx, prev_row, repeat_prob, co_occurrence, overall_prob,
                                      adj_window, edge_boost, markov_weight, method='greedy')
    sets.append(greedy_set)

    np.random.seed(current_idx)

    attempts = 0
    max_attempts = n_sets * 5

    while len(sets) < n_sets and attempts < max_attempts:
        attempts += 1
        candidate = cascade_with_markov(df, current_idx, prev_row, repeat_prob, co_occurrence, overall_prob,
                                         adj_window, edge_boost, markov_weight, method='stochastic')
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


def evaluate_markov_model(df, train_end_idx, markov_weight):
    """Evaluate Markov-enhanced cascade model"""
    # Build transition matrix from training data
    repeat_prob, co_occurrence, overall_prob = build_transition_matrix(df, train_end_idx)

    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    best_wrong = []

    for i in range(1, n_holdout):
        global_idx = train_end_idx + i
        prev_row = df.iloc[global_idx - 1]
        actual_row = df.iloc[global_idx]
        actual_set = get_actual_set(actual_row)

        portfolio = cascade_markov_portfolio(
            df, global_idx, prev_row, repeat_prob, co_occurrence, overall_prob,
            n_sets=50, adj_window=2, edge_boost=3.0, markov_weight=markov_weight
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
    print("MARKOV TRANSITION MODEL")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data()
    train_end_idx = len(df) - HOLDOUT_DAYS

    # Build and analyze transition matrix
    print("\nBuilding Markov transition matrix...")
    repeat_prob, co_occurrence, overall_prob = build_transition_matrix(df, train_end_idx)

    # Analyze repeat probabilities
    print("\n" + "-" * 70)
    print("REPEAT PROBABILITY ANALYSIS")
    print("-" * 70)

    repeat_sorted = sorted(repeat_prob.items(), key=lambda x: -x[1])

    print("\nTop 10 parts most likely to repeat:")
    print(f"{'Part':>6} {'Repeat Prob':>12} {'Overall Prob':>12} {'Lift':>8}")
    print("-" * 42)
    for part, prob in repeat_sorted[:10]:
        overall = overall_prob[part]
        lift = prob / overall if overall > 0 else 0
        print(f"P_{part:>3} {prob:>11.2%} {overall:>11.2%} {lift:>7.2f}x")

    avg_repeat = np.mean(list(repeat_prob.values()))
    print(f"\nAverage repeat probability: {avg_repeat:.2%}")
    print(f"(Random baseline: {5/39:.2%})")

    # Test different Markov weights
    print("\n" + "-" * 70)
    print("MARKOV WEIGHT SWEEP")
    print("-" * 70)
    print(f"\nBaseline (cascade only, markov_weight=0): avg_wrong = 2.69")
    print(f"\n{'Weight':>8} {'Avg Wrong':>12} {'Good Rate':>12} {'Correct':>10}")
    print("-" * 45)

    results = []
    for weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        result = evaluate_markov_model(df, train_end_idx, weight)
        results.append({'markov_weight': weight, **result})
        marker = " <-- BEST" if result['avg_wrong'] < 2.69 else ""
        print(f"{weight:>8.1f} {result['avg_wrong']:>12.3f} {result['good_rate']:>11.2f}% {result['correct_rate']:>9.2f}%{marker}")

    # Find best
    best = min(results, key=lambda x: x['avg_wrong'])

    print("\n" + "=" * 70)
    print("MARKOV MODEL RESULTS")
    print("=" * 70)
    print(f"\nBest Markov weight: {best['markov_weight']}")
    print(f"Avg wrong: {best['avg_wrong']:.3f} (baseline: 2.69)")
    print(f"Good rate: {best['good_rate']:.2f}% (baseline: 32.69%)")
    print(f"Correct rate: {best['correct_rate']:.2f}% (baseline: 1.65%)")

    improvement = 2.69 - best['avg_wrong']
    if improvement > 0:
        print(f"\nImprovement: {improvement:.3f} ({improvement/2.69*100:.1f}%)")
    else:
        print(f"\nNo improvement from Markov signal.")

    return results


if __name__ == "__main__":
    results = main()
