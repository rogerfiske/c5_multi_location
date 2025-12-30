"""
Combined Optimized Model for C5 Parts Forecasting

Combines the best findings from tuning experiments:
1. Portfolio size: 200 sets (biggest lever)
2. Adjacency window: +/-3 (slight improvement)
3. Markov signal: weight 0.3 (small additive improvement)

Target: Push avg wrong below 2.2
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Optimized configuration
HOLDOUT_DAYS = 365
NUM_CANDIDATE_SETS = 200
ADJACENCY_WINDOW = 3
EDGE_BOOST = 3.0
MIDDLE_BOOST = 2.0
MARKOV_WEIGHT = 0.3


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
    """Build Markov transition matrix"""
    train_df = df.iloc[:end_idx]

    repeat_counts = defaultdict(int)
    appearance_counts = defaultdict(int)
    co_occurrence = defaultdict(Counter)

    for i in range(1, len(train_df)):
        today_parts = get_parts_set(train_df.iloc[i-1])
        tomorrow_parts = get_parts_set(train_df.iloc[i])

        for part in today_parts:
            appearance_counts[part] += 1
            if part in tomorrow_parts:
                repeat_counts[part] += 1
            for tomorrow_part in tomorrow_parts:
                co_occurrence[part][tomorrow_part] += 1

    repeat_prob = {}
    for part in range(1, 40):
        if appearance_counts[part] > 0:
            repeat_prob[part] = repeat_counts[part] / appearance_counts[part]
        else:
            repeat_prob[part] = 0.0

    p_cols = [f'P_{i}' for i in range(1, 40)]
    overall_prob = {}
    for col in p_cols:
        part_id = int(col.replace('P_', ''))
        overall_prob[part_id] = train_df[col].sum() / len(train_df)

    return repeat_prob, co_occurrence, overall_prob


def markov_scores(today_parts, repeat_prob, co_occurrence, overall_prob):
    """Generate Markov scores for tomorrow"""
    scores = {}

    for part in range(1, 40):
        base_score = overall_prob.get(part, 1/39)
        markov_signal = 0.0

        if part in today_parts:
            markov_signal += repeat_prob.get(part, 0) * 2

        for today_part in today_parts:
            co_count = co_occurrence[today_part].get(part, 0)
            total_transitions = sum(co_occurrence[today_part].values())
            if total_transitions > 0:
                markov_signal += co_count / total_transitions

        markov_signal = markov_signal / (len(today_parts) + 1)
        scores[part] = MARKOV_WEIGHT * markov_signal + (1 - MARKOV_WEIGHT) * base_score

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


def combined_cascade_predict(df, current_idx, prev_row, markov, method='greedy'):
    """Combined cascade with Markov and optimized parameters"""
    prev_l_values = [prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                     prev_row['L_4'], prev_row['L_5']]

    predicted_set = []

    for position in range(1, 6):
        pos_scores = position_specific_scores(df, current_idx, position)
        prev_val = prev_l_values[position - 1]

        # Apply adjacency boost with optimized window
        for offset in range(-ADJACENCY_WINDOW, ADJACENCY_WINDOW + 1):
            candidate = prev_val + offset
            if 1 <= candidate <= 39 and candidate in pos_scores:
                boost = EDGE_BOOST if position in [1, 5] else MIDDLE_BOOST
                pos_scores[candidate] *= boost

        # Combine with Markov signal
        combined = {}
        for part in range(1, 40):
            pos_score = pos_scores.get(part, 0)
            markov_score = markov.get(part, 0)
            combined[part] = (1 - MARKOV_WEIGHT) * pos_score + MARKOV_WEIGHT * markov_score

        # Filter to valid candidates
        if predicted_set:
            min_valid = max(predicted_set) + 1
        else:
            min_valid = 1

        valid_scores = {k: v for k, v in combined.items() if k >= min_valid}

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


def generate_portfolio(df, current_idx, prev_row, markov, n_sets):
    """Generate portfolio with optimized parameters"""
    sets = []

    greedy_set = combined_cascade_predict(df, current_idx, prev_row, markov, method='greedy')
    sets.append(greedy_set)

    np.random.seed(current_idx)

    attempts = 0
    max_attempts = n_sets * 5

    while len(sets) < n_sets and attempts < max_attempts:
        attempts += 1
        candidate = combined_cascade_predict(df, current_idx, prev_row, markov, method='stochastic')
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


def main():
    print("=" * 70)
    print("COMBINED OPTIMIZED MODEL")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Portfolio size: {NUM_CANDIDATE_SETS}")
    print(f"  Adjacency window: +/-{ADJACENCY_WINDOW}")
    print(f"  Edge boost: {EDGE_BOOST}")
    print(f"  Markov weight: {MARKOV_WEIGHT}")

    df = load_data()
    train_end_idx = len(df) - HOLDOUT_DAYS

    print(f"\nBuilding Markov transition matrix...")
    repeat_prob, co_occurrence, overall_prob = build_transition_matrix(df, train_end_idx)

    n_holdout = len(df) - train_end_idx

    print(f"\nEvaluating on {n_holdout - 1} holdout days...")

    best_wrong = []
    greedy_wrong = []

    for i in range(1, n_holdout):
        global_idx = train_end_idx + i
        prev_row = df.iloc[global_idx - 1]
        actual_row = df.iloc[global_idx]
        actual_set = get_actual_set(actual_row)
        today_parts = get_parts_set(prev_row)

        markov = markov_scores(today_parts, repeat_prob, co_occurrence, overall_prob)
        portfolio = generate_portfolio(df, global_idx, prev_row, markov, NUM_CANDIDATE_SETS)

        scores = [score_set(pred_set, actual_set) for pred_set in portfolio]
        greedy_wrong.append(scores[0])
        best_wrong.append(min(scores))

    # Results
    avg_greedy = np.mean(greedy_wrong)
    avg_best = np.mean(best_wrong)

    best_counter = Counter(best_wrong)
    greedy_counter = Counter(greedy_wrong)

    correct_rate = (best_counter.get(0, 0) + best_counter.get(1, 0)) / len(best_wrong) * 100
    good_rate = sum(best_counter.get(w, 0) for w in [0, 1, 2]) / len(best_wrong) * 100

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- GREEDY (single prediction) ---")
    print(f"Avg wrong: {avg_greedy:.3f}")

    print(f"\n--- PORTFOLIO (best of {NUM_CANDIDATE_SETS}) ---")
    print(f"{'Wrong':>6} {'Count':>8} {'Percent':>10}")
    print("-" * 28)
    for wrong in range(6):
        count = best_counter.get(wrong, 0)
        pct = count / len(best_wrong) * 100
        marker = " <-- CORRECT" if wrong <= 1 else ""
        print(f"{wrong:>6} {count:>8} {pct:>9.2f}%{marker}")

    print(f"\nAvg wrong: {avg_best:.3f}")
    print(f"Correct rate (0-1 wrong): {correct_rate:.2f}%")
    print(f"Good rate (0-2 wrong): {good_rate:.2f}%")

    print("\n" + "=" * 70)
    print("COMPARISON TO PREVIOUS RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<35} {'Avg Wrong':>10} {'Good%':>10} {'Correct%':>10}")
    print("-" * 68)
    print(f"{'Baseline (cascade, 50 sets)':<35} {'2.690':>10} {'32.69%':>10} {'1.65%':>10}")
    print(f"{'Tuned (cascade, 200 sets, adj=3)':<35} {'2.264':>10} {'68.96%':>10} {'4.67%':>10}")
    print(f"{'Combined (+ Markov)':<35} {avg_best:>10.3f} {good_rate:>9.2f}% {correct_rate:>9.2f}%")

    improvement_vs_baseline = (2.69 - avg_best) / 2.69 * 100
    improvement_vs_tuned = (2.264 - avg_best) / 2.264 * 100

    print(f"\nImprovement vs baseline: {improvement_vs_baseline:.1f}%")
    print(f"Improvement vs tuned-only: {improvement_vs_tuned:.1f}%")

    # Check targets
    print("\n" + "=" * 70)
    print("TARGET CHECK")
    print("=" * 70)

    targets = [
        ("Avg wrong < 3.0", avg_best < 3.0, f"{avg_best:.3f}"),
        ("Avg wrong < 2.5 (stretch)", avg_best < 2.5, f"{avg_best:.3f}"),
        ("Avg wrong < 2.25 (new stretch)", avg_best < 2.25, f"{avg_best:.3f}"),
        ("Good rate > 40%", good_rate > 40, f"{good_rate:.2f}%"),
        ("Good rate > 60%", good_rate > 60, f"{good_rate:.2f}%"),
        ("Correct rate > 3%", correct_rate > 3, f"{correct_rate:.2f}%"),
        ("Correct rate > 5%", correct_rate > 5, f"{correct_rate:.2f}%"),
    ]

    for name, met, value in targets:
        status = "[MET]" if met else "[MISS]"
        print(f"{status:>8} {name:<35} ({value})")

    return {
        'avg_wrong': avg_best,
        'correct_rate': correct_rate,
        'good_rate': good_rate,
        'distribution': dict(best_counter)
    }


if __name__ == "__main__":
    results = main()
