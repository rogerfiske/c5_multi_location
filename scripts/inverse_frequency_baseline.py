"""
Inverse Frequency Baseline Investigation
Tests if the inverted signal can be exploited.

Hypothesis: If most frequent parts are consistently WRONG,
then LEAST frequent parts might be more likely to appear.

Baselines tested:
1. Inverse Frequency - Rank by (1 - frequency), preferring rare parts
2. Exclusion - Exclude top-K most frequent, pick from remainder
3. Anti-Greedy - Take the LEAST likely valid set
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
HOLDOUT_DAYS = 365
POOL_SIZES = [10, 15, 20, 25, 30]


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_actual_parts(row):
    """Extract actual parts from a row"""
    return set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def get_actual_set(row):
    """Extract actual set as sorted tuple"""
    return tuple(sorted([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']]))


def global_frequency(train_df):
    """Calculate global frequency scores"""
    p_cols = [f'P_{i}' for i in range(1, 40)]
    freq = train_df[p_cols].sum()
    total = len(train_df)
    scores = {}
    for col in p_cols:
        part_id = int(col.replace('P_', ''))
        scores[part_id] = freq[col] / total
    return scores


def inverse_frequency_scores(freq_scores):
    """Invert frequency scores: rare parts get high scores"""
    max_freq = max(freq_scores.values())
    inverse = {}
    for part, freq in freq_scores.items():
        # Higher score for less frequent parts
        inverse[part] = max_freq - freq + 0.01  # Small offset to avoid zero
    # Normalize
    total = sum(inverse.values())
    return {k: v/total for k, v in inverse.items()}


def exclusion_scores(freq_scores, exclude_top_k=10):
    """Exclude top-K most frequent parts, uniform over remainder"""
    ranked = sorted(freq_scores.items(), key=lambda x: -x[1])
    excluded = set([p[0] for p in ranked[:exclude_top_k]])

    scores = {}
    remaining = [p for p in range(1, 40) if p not in excluded]
    uniform_score = 1.0 / len(remaining) if remaining else 0

    for part in range(1, 40):
        scores[part] = uniform_score if part not in excluded else 0

    return scores


def generate_set_from_scores(scores, method='greedy'):
    """
    Generate a valid 5-part set from scores.
    Must satisfy L_1 < L_2 < L_3 < L_4 < L_5
    """
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    if method == 'greedy':
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
                    break

        return tuple(sorted(selected)[:5])

    return tuple(sorted(list(scores.keys())[:5]))


def calculate_set_accuracy(pred_set, actual_set):
    """Calculate how many parts are wrong"""
    pred = set(pred_set)
    actual = set(actual_set)
    wrong = len(pred - actual)
    correct = 5 - wrong
    is_good_plus = wrong <= 1 or wrong >= 4
    return wrong, correct, is_good_plus


def calculate_recall_at_k(scores, actual_parts, k):
    """Recall@K: fraction of actual parts in top-K"""
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_k = set([p[0] for p in ranked[:k]])
    hits = len(actual_parts.intersection(top_k))
    return hits / len(actual_parts)


def evaluate_baseline(df, train_end_idx, score_fn, name):
    """Evaluate a scoring function on holdout"""
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    print(f"\n--- Evaluating: {name} ---")

    wrong_counts = []
    correct_counts = []
    recall_at_k = {k: [] for k in POOL_SIZES}

    for i in range(n_holdout):
        global_idx = train_end_idx + i
        scores = score_fn(df, global_idx)

        actual_row = df.iloc[global_idx]
        actual_parts = get_actual_parts(actual_row)
        actual_set = get_actual_set(actual_row)

        for k in POOL_SIZES:
            recall = calculate_recall_at_k(scores, actual_parts, k)
            recall_at_k[k].append(recall)

        pred_set = generate_set_from_scores(scores, method='greedy')
        wrong, correct, is_good_plus = calculate_set_accuracy(pred_set, actual_set)
        wrong_counts.append(wrong)
        correct_counts.append(correct)

    # Aggregate
    wrong_dist = Counter(wrong_counts)
    good_plus_correct = sum(1 for w in wrong_counts if w <= 1)  # 0-1 wrong = correct
    good_plus_inverted = sum(1 for w in wrong_counts if w >= 4)  # 4-5 wrong = inverted

    print(f"  Correct (0-1 wrong): {good_plus_correct/n_holdout*100:.2f}%")
    print(f"  Inverted (4-5 wrong): {good_plus_inverted/n_holdout*100:.2f}%")
    print(f"  Total Good+: {(good_plus_correct + good_plus_inverted)/n_holdout*100:.2f}%")
    print(f"  Wrong distribution:")
    for w in range(6):
        count = wrong_dist.get(w, 0)
        pct = count / n_holdout * 100
        label = "CORRECT" if w <= 1 else ("INVERTED" if w >= 4 else "")
        print(f"    {w} wrong: {count:4d} ({pct:5.2f}%) {label}")

    print(f"  Recall@20: {np.mean(recall_at_k[20])*100:.2f}%")
    print(f"  Recall@30: {np.mean(recall_at_k[30])*100:.2f}%")

    return {
        'name': name,
        'correct_rate': good_plus_correct / n_holdout * 100,
        'inverted_rate': good_plus_inverted / n_holdout * 100,
        'total_good_plus': (good_plus_correct + good_plus_inverted) / n_holdout * 100,
        'wrong_dist': wrong_dist,
        'recall_20': np.mean(recall_at_k[20]) * 100,
        'recall_30': np.mean(recall_at_k[30]) * 100,
    }


def main():
    print("=" * 70)
    print("INVERSE FREQUENCY BASELINE INVESTIGATION")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Total events: {len(df)}")

    train_end_idx = len(df) - HOLDOUT_DAYS
    train_df = df.iloc[:train_end_idx]

    # Pre-compute frequencies
    freq_scores = global_frequency(train_df)
    inv_scores = inverse_frequency_scores(freq_scores)

    # Show frequency distribution
    print("\n--- Part Frequency Analysis ---")
    ranked_freq = sorted(freq_scores.items(), key=lambda x: -x[1])
    print("Most frequent parts:")
    for part, freq in ranked_freq[:5]:
        print(f"  P_{part}: {freq*100:.2f}%")
    print("Least frequent parts:")
    for part, freq in ranked_freq[-5:]:
        print(f"  P_{part}: {freq*100:.2f}%")

    results = []

    # 1. Standard frequency (for comparison)
    def freq_fn(df, idx):
        return freq_scores
    r1 = evaluate_baseline(df, train_end_idx, freq_fn, "Global Frequency (Standard)")
    results.append(r1)

    # 2. Inverse frequency
    def inv_fn(df, idx):
        return inv_scores
    r2 = evaluate_baseline(df, train_end_idx, inv_fn, "Inverse Frequency (Rare First)")
    results.append(r2)

    # 3. Exclusion baselines
    for exclude_k in [5, 10, 15, 20]:
        exc_scores = exclusion_scores(freq_scores, exclude_top_k=exclude_k)
        def exc_fn(df, idx, scores=exc_scores):
            return scores
        r = evaluate_baseline(df, train_end_idx, exc_fn, f"Exclude Top-{exclude_k}")
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: CORRECT vs INVERTED SIGNAL")
    print("=" * 70)
    print(f"\n{'Baseline':<30} {'Correct':>10} {'Inverted':>10} {'Recall@20':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<30} {r['correct_rate']:>9.2f}% {r['inverted_rate']:>9.2f}% {r['recall_20']:>9.2f}%")

    # Analysis
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    std_inverted = results[0]['inverted_rate']
    inv_inverted = results[1]['inverted_rate']

    if results[1]['correct_rate'] > results[0]['correct_rate']:
        print("\nINVERSE FREQUENCY IMPROVED CORRECT PREDICTIONS!")
        print(f"  Standard correct: {results[0]['correct_rate']:.2f}%")
        print(f"  Inverse correct: {results[1]['correct_rate']:.2f}%")
        print("  -> Rare parts ARE more predictive")
    else:
        print("\nINVERSE FREQUENCY DID NOT IMPROVE correct predictions.")
        print("  The inverted signal may not be directly exploitable.")
        print("  Possible explanations:")
        print("    1. All parts have similar baseline frequency (~12-13%)")
        print("    2. The signal is more complex than simple frequency inversion")
        print("    3. Temporal patterns matter more than global frequency")

    return results


if __name__ == "__main__":
    results = main()
