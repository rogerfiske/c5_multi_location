"""
Recency Exclusion Baseline
Tests the hypothesis that parts appearing in recent window are LESS likely tomorrow.

Based on party mode insight: 30-39 days may be the sweet spot for exclusion window.
Tests windows: 30, 33, 35, 37, 39 days

Hypothesis: Parts NOT seen in last N days are more likely to appear tomorrow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
HOLDOUT_DAYS = 365
WINDOWS = [30, 33, 35, 37, 39]  # Sweet spot range from previous studies


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_actual_parts(row):
    """Extract actual parts from a row"""
    return set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def get_recent_parts(df, current_idx, window):
    """Get all parts that appeared in the last N days"""
    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    recent = set()
    for _, row in window_df.iterrows():
        recent.update(get_actual_parts(row))
    return recent


def get_excluded_parts(df, current_idx, window):
    """Get parts that DID NOT appear in last N days (eligible pool)"""
    recent = get_recent_parts(df, current_idx, window)
    all_parts = set(range(1, 40))
    return all_parts - recent


def recency_exclusion_scores(df, current_idx, window):
    """
    Score parts by recency exclusion:
    - Parts NOT in recent window get high scores
    - Parts IN recent window get low scores
    """
    recent = get_recent_parts(df, current_idx, window)

    scores = {}
    for part in range(1, 40):
        if part in recent:
            scores[part] = 0.01  # Very low score for recent parts
        else:
            scores[part] = 1.0  # High score for excluded parts

    # Normalize
    total = sum(scores.values())
    return {k: v/total for k, v in scores.items()}


def generate_set_from_scores(scores):
    """Generate valid ascending 5-part set from scores"""
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    selected = []
    for part_id, score in ranked:
        if len(selected) == 5:
            break
        if not selected or part_id > max(selected):
            selected.append(part_id)
        elif part_id < min(selected):
            selected.insert(0, part_id)
        else:
            for i in range(len(selected)):
                if i == 0 and part_id < selected[i]:
                    selected.insert(0, part_id)
                    break
                elif i > 0 and selected[i-1] < part_id < selected[i]:
                    selected.insert(i, part_id)
                    break
        if len(selected) > 5:
            selected = sorted(selected)[:5]

    # Fill if needed
    if len(selected) < 5:
        remaining = [p for p in range(1, 40) if p not in selected]
        for p in remaining:
            if not selected or p > selected[-1]:
                selected.append(p)
            if len(selected) == 5:
                break

    return tuple(sorted(selected)[:5])


def calculate_set_accuracy(pred_set, actual_set):
    """Calculate wrong count"""
    pred = set(pred_set)
    actual = set(actual_set)
    wrong = len(pred - actual)
    correct = 5 - wrong
    return wrong, correct


def analyze_exclusion_coverage(df, train_end_idx, window):
    """
    Analyze how well recency exclusion predicts actual parts.
    Key metric: What % of actual parts were NOT in the recent window?
    """
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    # Track coverage stats
    actual_in_excluded = []  # How many actual parts were in excluded pool
    excluded_pool_sizes = []

    for i in range(n_holdout):
        global_idx = train_end_idx + i

        excluded_pool = get_excluded_parts(df, global_idx, window)
        actual_parts = get_actual_parts(df.iloc[global_idx])

        # How many of the 5 actual parts were in the excluded pool?
        hits = len(actual_parts.intersection(excluded_pool))
        actual_in_excluded.append(hits)
        excluded_pool_sizes.append(len(excluded_pool))

    return {
        'window': window,
        'avg_hits': np.mean(actual_in_excluded),
        'avg_pool_size': np.mean(excluded_pool_sizes),
        'hit_rate': np.mean(actual_in_excluded) / 5 * 100,
        'perfect_coverage': sum(1 for h in actual_in_excluded if h == 5) / n_holdout * 100,
        'hit_distribution': Counter(actual_in_excluded)
    }


def evaluate_recency_baseline(df, train_end_idx, window):
    """Full evaluation of recency exclusion baseline"""
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    wrong_counts = []

    for i in range(n_holdout):
        global_idx = train_end_idx + i

        scores = recency_exclusion_scores(df, global_idx, window)
        actual_parts = get_actual_parts(df.iloc[global_idx])
        actual_set = tuple(sorted(actual_parts))

        pred_set = generate_set_from_scores(scores)
        wrong, correct = calculate_set_accuracy(pred_set, actual_set)
        wrong_counts.append(wrong)

    wrong_dist = Counter(wrong_counts)
    correct_rate = sum(1 for w in wrong_counts if w <= 1) / n_holdout * 100
    inverted_rate = sum(1 for w in wrong_counts if w >= 4) / n_holdout * 100

    return {
        'window': window,
        'correct_rate': correct_rate,
        'inverted_rate': inverted_rate,
        'total_good_plus': correct_rate + inverted_rate,
        'wrong_dist': wrong_dist,
        'avg_wrong': np.mean(wrong_counts)
    }


def main():
    print("=" * 70)
    print("RECENCY EXCLUSION BASELINE ANALYSIS")
    print("Hypothesis: Parts NOT seen in last N days are more likely tomorrow")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing windows: {WINDOWS}")

    # Load data
    df = load_data()
    train_end_idx = len(df) - HOLDOUT_DAYS

    print(f"\nTotal events: {len(df)}")
    print(f"Train: {train_end_idx}, Holdout: {HOLDOUT_DAYS}")

    # Phase 1: Analyze exclusion coverage
    print("\n" + "=" * 70)
    print("PHASE 1: EXCLUSION COVERAGE ANALYSIS")
    print("How many actual parts were NOT in the recent window?")
    print("=" * 70)

    coverage_results = []
    for window in WINDOWS:
        result = analyze_exclusion_coverage(df, train_end_idx, window)
        coverage_results.append(result)

        print(f"\n--- Window: {window} days ---")
        print(f"  Avg excluded pool size: {result['avg_pool_size']:.1f} parts")
        print(f"  Avg actual parts in excluded pool: {result['avg_hits']:.2f} / 5")
        print(f"  Hit rate: {result['hit_rate']:.1f}%")
        print(f"  Perfect coverage (5/5): {result['perfect_coverage']:.1f}%")
        print(f"  Distribution: {dict(sorted(result['hit_distribution'].items()))}")

    # Phase 2: Set generation evaluation
    print("\n" + "=" * 70)
    print("PHASE 2: SET GENERATION EVALUATION")
    print("Generate sets from excluded pool, measure accuracy")
    print("=" * 70)

    eval_results = []
    for window in WINDOWS:
        result = evaluate_recency_baseline(df, train_end_idx, window)
        eval_results.append(result)

        print(f"\n--- Window: {window} days ---")
        print(f"  Correct (0-1 wrong): {result['correct_rate']:.2f}%")
        print(f"  Inverted (4-5 wrong): {result['inverted_rate']:.2f}%")
        print(f"  Total Good+: {result['total_good_plus']:.2f}%")
        print(f"  Avg wrong: {result['avg_wrong']:.2f}")
        print(f"  Distribution: {dict(sorted(result['wrong_dist'].items()))}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n{'Window':>8} | {'Pool Size':>10} | {'Hit Rate':>10} | {'Correct':>10} | {'Inverted':>10}")
    print("-" * 60)
    for cov, evl in zip(coverage_results, eval_results):
        print(f"{cov['window']:>8}d | {cov['avg_pool_size']:>10.1f} | {cov['hit_rate']:>9.1f}% | {evl['correct_rate']:>9.2f}% | {evl['inverted_rate']:>9.2f}%")

    # Best window
    best_coverage = max(coverage_results, key=lambda x: x['hit_rate'])
    best_correct = max(eval_results, key=lambda x: x['correct_rate'])

    print(f"\nBest hit rate: {best_coverage['window']} days ({best_coverage['hit_rate']:.1f}%)")
    print(f"Best correct rate: {best_correct['window']} days ({best_correct['correct_rate']:.2f}%)")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)

    if best_correct['correct_rate'] > 5:
        print(f"\nRECENCY EXCLUSION SHOWS PROMISE!")
        print(f"  {best_correct['window']}-day window achieves {best_correct['correct_rate']:.2f}% correct")
        print(f"  This is a significant improvement over 0% from frequency baselines")
    else:
        print(f"\nRecency exclusion alone doesn't flip the signal.")
        print(f"  The anti-correlation may require more sophisticated modeling.")
        print(f"  Consider: weighted recency decay, part-specific windows, or")
        print(f"  combining exclusion with other features.")

    return coverage_results, eval_results


if __name__ == "__main__":
    coverage_results, eval_results = main()
