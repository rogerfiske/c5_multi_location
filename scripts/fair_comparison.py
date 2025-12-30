"""
Fair Comparison: Ordered vs Unordered at SAME portfolio size

The question: Does removing the ascending constraint improve predictions
when we use the SAME number of sets?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_data():
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_part_scores(df, current_idx, prev_parts, rolling_window=30):
    """Score parts with adjacency boost"""
    start_idx = max(0, current_idx - rolling_window)
    window_df = df.iloc[start_idx:current_idx]

    scores = {}
    for part_id in range(1, 40):
        freq = 0
        for pos in range(1, 6):
            col = f'L_{pos}'
            freq += (window_df[col] == part_id).sum()
        scores[part_id] = freq / (len(window_df) * 5) + 0.01

    for prev_part in prev_parts:
        for offset in range(-3, 4):
            candidate = prev_part + offset
            if 1 <= candidate <= 39:
                distance_factor = 1.0 - (abs(offset) / 4) * 0.5
                scores[candidate] *= (1 + 3.0 * distance_factor)

    return scores


def generate_ordered_set(df, current_idx, scores, method='stochastic'):
    """Generate set with ascending constraint L_1 < L_2 < L_3 < L_4 < L_5"""
    predicted = []

    for position in range(5):
        if predicted:
            min_valid = max(predicted) + 1
        else:
            min_valid = 1

        valid_scores = {k: v for k, v in scores.items() if k >= min_valid}

        if not valid_scores:
            remaining = [p for p in range(1, 40) if p > (max(predicted) if predicted else 0)]
            if remaining:
                predicted.append(min(remaining))
            continue

        if method == 'greedy':
            best = max(valid_scores.items(), key=lambda x: x[1])[0]
        else:
            parts = list(valid_scores.keys())
            probs = np.array(list(valid_scores.values()))
            probs = probs / probs.sum()
            best = np.random.choice(parts, p=probs)

        predicted.append(best)

    return tuple(sorted(predicted)) if len(predicted) == 5 else None


def generate_unordered_set(scores, method='stochastic'):
    """Generate set WITHOUT ascending constraint - just 5 unique parts"""
    selected = []
    available = dict(scores)

    for _ in range(5):
        if not available:
            break

        if method == 'greedy':
            best = max(available.items(), key=lambda x: x[1])[0]
        else:
            parts = list(available.keys())
            probs = np.array(list(available.values()))
            probs = probs / probs.sum()
            best = np.random.choice(parts, p=probs)

        selected.append(best)
        del available[best]

    return frozenset(selected) if len(selected) == 5 else None


def generate_random_set():
    """Generate completely random set of 5 parts"""
    return frozenset(np.random.choice(range(1, 40), size=5, replace=False))


def run_comparison(df, portfolio_size, n_days=365):
    """Compare ordered, unordered, and random at same portfolio size"""

    ordered_results = []
    unordered_results = []
    random_results = []

    for i in range(n_days, 0, -1):
        idx = len(df) - i
        if idx < 31:
            continue

        target = df.iloc[idx]
        actual = frozenset([target['L_1'], target['L_2'], target['L_3'],
                           target['L_4'], target['L_5']])

        prev = df.iloc[idx - 1]
        prev_parts = set([prev['L_1'], prev['L_2'], prev['L_3'],
                         prev['L_4'], prev['L_5']])

        scores = get_part_scores(df, idx, prev_parts)

        np.random.seed(idx)

        # Generate ORDERED portfolio
        ordered_portfolio = set()
        greedy = generate_ordered_set(df, idx, scores, method='greedy')
        if greedy:
            ordered_portfolio.add(greedy)

        attempts = 0
        while len(ordered_portfolio) < portfolio_size and attempts < portfolio_size * 10:
            attempts += 1
            candidate = generate_ordered_set(df, idx, scores, method='stochastic')
            if candidate:
                ordered_portfolio.add(candidate)

        # Generate UNORDERED portfolio
        np.random.seed(idx)
        unordered_portfolio = set()
        greedy = generate_unordered_set(scores, method='greedy')
        if greedy:
            unordered_portfolio.add(greedy)

        attempts = 0
        while len(unordered_portfolio) < portfolio_size and attempts < portfolio_size * 10:
            attempts += 1
            candidate = generate_unordered_set(scores, method='stochastic')
            if candidate:
                unordered_portfolio.add(candidate)

        # Generate RANDOM portfolio (baseline)
        np.random.seed(idx)
        random_portfolio = set()
        while len(random_portfolio) < portfolio_size:
            random_portfolio.add(generate_random_set())

        # Evaluate best wrong for each
        best_ordered = min(len(set(s) - actual) for s in ordered_portfolio) if ordered_portfolio else 5
        best_unordered = min(len(s - actual) for s in unordered_portfolio) if unordered_portfolio else 5
        best_random = min(len(s - actual) for s in random_portfolio) if random_portfolio else 5

        ordered_results.append(best_ordered)
        unordered_results.append(best_unordered)
        random_results.append(best_random)

    return ordered_results, unordered_results, random_results


def print_results(name, results):
    """Print summary statistics"""
    n = len(results)
    avg = np.mean(results)
    excellent = sum(1 for r in results if r <= 1) / n * 100
    good = sum(1 for r in results if r <= 2) / n * 100

    counter = Counter(results)

    print(f"\n{name}:")
    print(f"  Avg Wrong: {avg:.3f}")
    print(f"  Excellent (0-1): {excellent:.1f}%")
    print(f"  Good (0-2): {good:.1f}%")
    print(f"  Distribution: ", end="")
    for w in range(6):
        pct = counter.get(w, 0) / n * 100
        print(f"{w}:{pct:.1f}% ", end="")
    print()

    return avg, excellent, good


def main():
    print("=" * 70)
    print("FAIR COMPARISON: ORDERED vs UNORDERED vs RANDOM")
    print("=" * 70)
    print("Same portfolio size, different generation strategies")

    df = load_data()
    print(f"Data loaded: {len(df)} days")

    for portfolio_size in [50, 100, 200]:
        print(f"\n{'='*70}")
        print(f"PORTFOLIO SIZE: {portfolio_size} sets")
        print("=" * 70)

        ordered, unordered, random_res = run_comparison(df, portfolio_size, n_days=365)

        ord_avg, ord_exc, ord_good = print_results("ORDERED (with L_1 < L_2 < ... constraint)", ordered)
        unord_avg, unord_exc, unord_good = print_results("UNORDERED (just 5 unique parts)", unordered)
        rand_avg, rand_exc, rand_good = print_results("RANDOM (baseline - no model)", random_res)

        print(f"\n  Improvement vs Random:")
        print(f"    Ordered:   {(rand_avg - ord_avg)/rand_avg*100:+.1f}% avg wrong reduction")
        print(f"    Unordered: {(rand_avg - unord_avg)/rand_avg*100:+.1f}% avg wrong reduction")

        print(f"\n  Ordered vs Unordered:")
        if unord_avg < ord_avg:
            print(f"    Unordered is {(ord_avg - unord_avg)/ord_avg*100:.1f}% better")
        else:
            print(f"    Ordered is {(unord_avg - ord_avg)/unord_avg*100:.1f}% better")


if __name__ == "__main__":
    main()
