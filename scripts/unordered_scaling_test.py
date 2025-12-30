"""
Test scaling of unordered set model with different portfolio sizes
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
    """Score all 39 parts"""
    start_idx = max(0, current_idx - rolling_window)
    window_df = df.iloc[start_idx:current_idx]

    scores = {}
    for part_id in range(1, 40):
        freq = 0
        for pos in range(1, 6):
            col = f'L_{pos}'
            freq += (window_df[col] == part_id).sum()
        scores[part_id] = freq / (len(window_df) * 5) + 0.01

    # Adjacency boost
    for prev_part in prev_parts:
        for offset in range(-3, 4):
            candidate = prev_part + offset
            if 1 <= candidate <= 39:
                distance_factor = 1.0 - (abs(offset) / 4) * 0.5
                scores[candidate] *= (1 + 3.0 * distance_factor)

    return scores


def generate_unordered_set(scores, method='stochastic'):
    """Generate a set of 5 unique parts"""
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


def generate_portfolio(scores, size, seed):
    """Generate portfolio of unordered sets"""
    np.random.seed(seed)
    portfolio = set()

    # Greedy first
    greedy = generate_unordered_set(scores, method='greedy')
    if greedy:
        portfolio.add(greedy)

    # Stochastic
    attempts = 0
    while len(portfolio) < size and attempts < size * 20:
        attempts += 1
        candidate = generate_unordered_set(scores, method='stochastic')
        if candidate:
            portfolio.add(candidate)

    return list(portfolio)


def evaluate_portfolio(portfolio, actual_parts):
    """Find best wrong count"""
    best = 5
    for s in portfolio:
        wrong = len(s - actual_parts)
        if wrong < best:
            best = wrong
    return best


def test_portfolio_size(df, portfolio_size, n_days=365):
    """Test a specific portfolio size"""
    results = []

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
        portfolio = generate_portfolio(scores, portfolio_size, seed=idx)

        best_wrong = evaluate_portfolio(portfolio, actual)
        results.append(best_wrong)

    return results


def main():
    print("=" * 70)
    print("UNORDERED SET MODEL - SCALING TEST")
    print("=" * 70)

    df = load_data()
    print(f"Data loaded: {len(df)} days")
    print("\nTesting different portfolio sizes (365-day backtest)...\n")

    sizes = [50, 100, 200, 400, 600, 800, 1000]

    print(f"{'Size':>8} {'Avg Wrong':>12} {'Excellent':>12} {'Good':>10} {'Poor+':>10}")
    print("-" * 56)

    for size in sizes:
        results = test_portfolio_size(df, size, n_days=365)
        n = len(results)

        avg = np.mean(results)
        excellent = sum(1 for r in results if r <= 1) / n * 100
        good = sum(1 for r in results if r <= 2) / n * 100
        poor_plus = sum(1 for r in results if r >= 3) / n * 100

        print(f"{size:>8} {avg:>12.3f} {excellent:>11.1f}% {good:>9.1f}% {poor_plus:>9.1f}%")

    # Also test combined adj + anti at best size
    print("\n" + "=" * 70)
    print("COMBINED (ADJ + ANTI) AT OPTIMAL SIZE")
    print("=" * 70)

    best_size = 500  # Half for each

    combined_results = []

    for i in range(365, 0, -1):
        idx = len(df) - i
        if idx < 31:
            continue

        target = df.iloc[idx]
        actual = frozenset([target['L_1'], target['L_2'], target['L_3'],
                           target['L_4'], target['L_5']])

        prev = df.iloc[idx - 1]
        prev_parts = set([prev['L_1'], prev['L_2'], prev['L_3'],
                         prev['L_4'], prev['L_5']])

        # Adjacency portfolio
        scores_adj = get_part_scores(df, idx, prev_parts)
        portfolio_adj = generate_portfolio(scores_adj, best_size, seed=idx)

        # Anti-adjacency portfolio (suppress adjacent)
        scores_anti = dict(scores_adj)
        for prev_part in prev_parts:
            for offset in range(-3, 4):
                candidate = prev_part + offset
                if 1 <= candidate <= 39:
                    scores_anti[candidate] *= 0.3
        portfolio_anti = generate_portfolio(scores_anti, best_size, seed=idx + 10000)

        combined = list(set(portfolio_adj) | set(portfolio_anti))
        best_wrong = evaluate_portfolio(combined, actual)
        combined_results.append(best_wrong)

    n = len(combined_results)
    avg = np.mean(combined_results)
    excellent = sum(1 for r in combined_results if r <= 1) / n * 100
    good = sum(1 for r in combined_results if r <= 2) / n * 100

    print(f"\nCombined {best_size}+{best_size} = {best_size*2} sets:")
    print(f"  Avg Wrong: {avg:.3f}")
    print(f"  Excellent (0-1): {excellent:.1f}%")
    print(f"  Good (0-2): {good:.1f}%")

    counter = Counter(combined_results)
    print(f"\nDistribution:")
    for w in range(6):
        count = counter.get(w, 0)
        pct = count / n * 100
        print(f"  {w} wrong: {count:>3} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
