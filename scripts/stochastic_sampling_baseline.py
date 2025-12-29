"""
Stochastic Sampling Baseline
Instead of greedy selection, use weighted random sampling from adjacency scores.

Key insight from Nova: Greedy selection is systematically biased toward anti-predictions.
Stochastic sampling should produce diverse sets that sometimes hit CORRECT.

Strategy:
1. Generate multiple candidate sets via weighted sampling
2. Evaluate each set and pick the best (or track individual performance)
3. Compare greedy vs stochastic approaches
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
HOLDOUT_DAYS = 365
NUM_CANDIDATE_SETS = 50  # Generate this many candidate sets per day


def load_data():
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_parts_set(row):
    return set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def adjacency_weighted_scores(df, current_idx, adjacency_window=4):
    """Adjacency-weighted scoring from previous analysis"""
    scores = {p: 1.0 for p in range(1, 40)}

    if current_idx == 0:
        return scores

    yesterday_parts = get_parts_set(df.iloc[current_idx - 1])

    # Adjacency boost with distance decay
    for yp in yesterday_parts:
        for delta in range(-adjacency_window, adjacency_window + 1):
            candidate = yp + delta
            if 1 <= candidate <= 39:
                distance = abs(delta)
                boost = 3.0 * (1 - distance / (adjacency_window + 1))
                scores[candidate] *= (1 + boost)

    # Normalize to probabilities
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}


def generate_greedy_set(scores):
    """Greedy ascending set generation (original approach)"""
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

    if len(selected) < 5:
        remaining = sorted([p for p in range(1, 40) if p not in selected])
        for p in remaining:
            if not selected or p > selected[-1]:
                selected.append(p)
            if len(selected) == 5:
                break

    return tuple(sorted(selected)[:5])


def generate_stochastic_set(scores, rng):
    """
    Generate a set via weighted random sampling.
    Sample 8-10 parts, then take 5 that form valid ascending set.
    """
    parts = list(range(1, 40))
    probs = np.array([scores[p] for p in parts])
    probs = probs / probs.sum()

    # Sample more than 5 to ensure we can form a valid ascending set
    sampled = rng.choice(parts, size=min(12, 39), replace=False, p=probs)
    sampled = sorted(sampled)

    # Take first 5 that form valid ascending sequence
    selected = []
    for p in sampled:
        if not selected or p > selected[-1]:
            selected.append(p)
        if len(selected) == 5:
            break

    # If we don't have 5, fill from remaining
    if len(selected) < 5:
        remaining = sorted([p for p in range(1, 40) if p not in selected])
        for p in remaining:
            if not selected or p > selected[-1]:
                selected.append(p)
            if len(selected) == 5:
                break

    return tuple(sorted(selected)[:5])


def generate_inverse_greedy_set(scores):
    """
    Generate set from LOWEST scored parts.
    If greedy picks the wrong parts, inverse-greedy might pick the right ones.
    """
    # Invert scores
    max_score = max(scores.values())
    inverse_scores = {k: max_score - v + 0.01 for k, v in scores.items()}

    return generate_greedy_set(inverse_scores)


def calculate_set_accuracy(pred_set, actual_set):
    pred = set(pred_set)
    actual = set(actual_set)
    wrong = len(pred - actual)
    return wrong


def evaluate_method(df, train_end_idx, method_name, set_generator_fn, n_seeds=10):
    """Evaluate a set generation method on holdout"""
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*60}")

    all_wrong_counts = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        wrong_counts = []

        for i in range(n_holdout):
            global_idx = train_end_idx + i
            scores = adjacency_weighted_scores(df, global_idx)
            actual_parts = get_parts_set(df.iloc[global_idx])

            pred_set = set_generator_fn(scores, rng)
            wrong = calculate_set_accuracy(pred_set, actual_parts)
            wrong_counts.append(wrong)

        all_wrong_counts.append(wrong_counts)

    # Aggregate across seeds
    avg_by_day = np.mean(all_wrong_counts, axis=0)
    overall_wrong = np.mean(avg_by_day)

    # Calculate rates
    correct_rates = []
    inverted_rates = []
    for seed_wrong in all_wrong_counts:
        correct = sum(1 for w in seed_wrong if w <= 1) / n_holdout * 100
        inverted = sum(1 for w in seed_wrong if w >= 4) / n_holdout * 100
        correct_rates.append(correct)
        inverted_rates.append(inverted)

    avg_correct = np.mean(correct_rates)
    avg_inverted = np.mean(inverted_rates)
    std_correct = np.std(correct_rates)

    print(f"\nResults (averaged over {n_seeds} seeds):")
    print(f"  Correct (0-1 wrong): {avg_correct:.2f}% (+/- {std_correct:.2f}%)")
    print(f"  Inverted (4-5 wrong): {avg_inverted:.2f}%")
    print(f"  Total Good+: {avg_correct + avg_inverted:.2f}%")
    print(f"  Avg wrong: {overall_wrong:.2f}")

    # Best seed performance
    best_seed = np.argmax(correct_rates)
    print(f"\nBest seed ({best_seed}): {correct_rates[best_seed]:.2f}% correct")

    return {
        'name': method_name,
        'avg_correct': avg_correct,
        'std_correct': std_correct,
        'avg_inverted': avg_inverted,
        'avg_wrong': overall_wrong,
        'best_correct': max(correct_rates)
    }


def oracle_analysis(df, train_end_idx, n_sets=50):
    """
    Oracle analysis: Generate N stochastic sets per day,
    see how often at least one set is CORRECT.
    """
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    print(f"\n{'='*60}")
    print(f"ORACLE ANALYSIS: Best of {n_sets} stochastic sets per day")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    best_wrong_per_day = []
    any_correct_count = 0
    any_good_plus_count = 0

    for i in range(n_holdout):
        global_idx = train_end_idx + i
        scores = adjacency_weighted_scores(df, global_idx)
        actual_parts = get_parts_set(df.iloc[global_idx])

        # Generate N sets
        day_wrong_counts = []
        for _ in range(n_sets):
            pred_set = generate_stochastic_set(scores, rng)
            wrong = calculate_set_accuracy(pred_set, actual_parts)
            day_wrong_counts.append(wrong)

        best_wrong = min(day_wrong_counts)
        best_wrong_per_day.append(best_wrong)

        if best_wrong <= 1:
            any_correct_count += 1
        if best_wrong <= 1 or best_wrong >= 4:
            any_good_plus_count += 1

    print(f"\nWith {n_sets} candidate sets per day:")
    print(f"  Days where at least 1 set is CORRECT (0-1 wrong): {any_correct_count/n_holdout*100:.2f}%")
    print(f"  Days where at least 1 set is Good+: {any_good_plus_count/n_holdout*100:.2f}%")
    print(f"  Avg best wrong per day: {np.mean(best_wrong_per_day):.2f}")

    print(f"\nBest-wrong distribution:")
    dist = Counter(best_wrong_per_day)
    for w in range(6):
        count = dist.get(w, 0)
        pct = count / n_holdout * 100
        label = "CORRECT" if w <= 1 else ""
        print(f"    Best = {w} wrong: {count:4d} ({pct:5.2f}%) {label}")

    return best_wrong_per_day


def main():
    print("=" * 70)
    print("STOCHASTIC SAMPLING BASELINE")
    print("Testing whether random sampling beats greedy selection")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data()
    train_end_idx = len(df) - HOLDOUT_DAYS

    print(f"\nTotal events: {len(df)}")
    print(f"Train: {train_end_idx}, Holdout: {HOLDOUT_DAYS}")

    results = []

    # 1. Greedy baseline
    def greedy_fn(scores, rng):
        return generate_greedy_set(scores)
    r1 = evaluate_method(df, train_end_idx, "Greedy (original)", greedy_fn, n_seeds=1)
    results.append(r1)

    # 2. Inverse greedy
    def inverse_fn(scores, rng):
        return generate_inverse_greedy_set(scores)
    r2 = evaluate_method(df, train_end_idx, "Inverse Greedy", inverse_fn, n_seeds=1)
    results.append(r2)

    # 3. Stochastic sampling
    def stochastic_fn(scores, rng):
        return generate_stochastic_set(scores, rng)
    r3 = evaluate_method(df, train_end_idx, "Stochastic Sampling", stochastic_fn, n_seeds=20)
    results.append(r3)

    # 4. Oracle analysis
    oracle_results = oracle_analysis(df, train_end_idx, n_sets=50)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n{'Method':<30} | {'Correct':>10} | {'Inverted':>10} | {'Avg Wrong':>10}")
    print("-" * 70)
    for r in results:
        correct_str = f"{r['avg_correct']:.2f}%" if 'std_correct' not in r or r['std_correct'] == 0 else f"{r['avg_correct']:.2f}% +/-{r['std_correct']:.1f}"
        print(f"{r['name']:<30} | {correct_str:>10} | {r['avg_inverted']:>9.2f}% | {r['avg_wrong']:>10.2f}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    if results[2]['avg_correct'] > results[0]['avg_correct']:
        print(f"\n*** Stochastic sampling BEATS greedy! ***")
        print(f"    Greedy correct: {results[0]['avg_correct']:.2f}%")
        print(f"    Stochastic correct: {results[2]['avg_correct']:.2f}%")
    else:
        print(f"\n*** Both methods show similar (low) correct rates ***")
        print(f"    The problem is fundamental to the scoring, not just greedy vs stochastic")

    oracle_correct = sum(1 for w in oracle_results if w <= 1) / len(oracle_results) * 100
    print(f"\n*** Oracle (best of 50 sets): {oracle_correct:.2f}% days have a CORRECT set ***")
    print(f"    This is the UPPER BOUND for this scoring function")

    return results, oracle_results


if __name__ == "__main__":
    results, oracle_results = main()
