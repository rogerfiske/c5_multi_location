"""
Unordered Set Model

Removes the ascending constraint (L_1 < L_2 < L_3 < L_4 < L_5).
Just predicts 5 unique values - order doesn't matter.

This fundamentally changes the prediction space:
- Old: Must pick parts in ascending order per position
- New: Pick any 5 unique parts, compare as sets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

CONFIG = {
    'portfolio_size': 200,
    'adjacency_window': 3,
    'adjacency_boost': 3.0,
    'rolling_window': 30,
}


def load_data():
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_part_scores(df, current_idx, prev_parts, bias='adjacency'):
    """
    Score all 39 parts based on:
    1. Overall frequency in rolling window
    2. Adjacency to previous day's parts (boosted)
    """
    window = CONFIG['rolling_window']
    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    scores = {}

    # Base: overall frequency across all positions
    for part_id in range(1, 40):
        freq = 0
        for pos in range(1, 6):
            col = f'L_{pos}'
            freq += (window_df[col] == part_id).sum()
        scores[part_id] = freq / (len(window_df) * 5) + 0.01  # Smoothing

    # Apply adjacency boost
    if bias == 'adjacency':
        adj_window = CONFIG['adjacency_window']
        boost = CONFIG['adjacency_boost']

        for prev_part in prev_parts:
            for offset in range(-adj_window, adj_window + 1):
                candidate = prev_part + offset
                if 1 <= candidate <= 39:
                    distance_factor = 1.0 - (abs(offset) / (adj_window + 1)) * 0.5
                    scores[candidate] *= (1 + boost * distance_factor)

    elif bias == 'anti':
        adj_window = CONFIG['adjacency_window']
        for prev_part in prev_parts:
            for offset in range(-adj_window, adj_window + 1):
                candidate = prev_part + offset
                if 1 <= candidate <= 39:
                    scores[candidate] *= 0.3  # Suppress adjacent

    return scores


def generate_unordered_set(scores, method='greedy', existing=None):
    """
    Generate a set of 5 unique parts (unordered).
    """
    if existing is None:
        existing = set()

    selected = []
    available_scores = {k: v for k, v in scores.items() if k not in existing}

    for _ in range(5):
        if not available_scores:
            break

        if method == 'greedy':
            best_part = max(available_scores.items(), key=lambda x: x[1])[0]
            selected.append(best_part)
        else:  # stochastic
            parts = list(available_scores.keys())
            probs = np.array(list(available_scores.values()))
            probs = probs / probs.sum()
            best_part = np.random.choice(parts, p=probs)
            selected.append(best_part)

        del available_scores[best_part]

    return frozenset(selected) if len(selected) == 5 else None


def generate_portfolio(df, current_idx, bias='adjacency'):
    """Generate portfolio of unordered sets"""
    np.random.seed(current_idx)

    prev_row = df.iloc[current_idx - 1]
    prev_parts = set([prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                     prev_row['L_4'], prev_row['L_5']])

    scores = get_part_scores(df, current_idx, prev_parts, bias)

    portfolio = set()

    # Greedy set first
    greedy = generate_unordered_set(scores, method='greedy')
    if greedy:
        portfolio.add(greedy)

    # Stochastic sets
    attempts = 0
    while len(portfolio) < CONFIG['portfolio_size'] and attempts < CONFIG['portfolio_size'] * 10:
        attempts += 1
        candidate = generate_unordered_set(scores, method='stochastic')
        if candidate and candidate not in portfolio:
            portfolio.add(candidate)

    return list(portfolio)


def evaluate_set(pred_set, actual_parts):
    """
    Evaluate prediction - just compare as sets (unordered).
    Returns: wrong count (0-5)
    """
    return len(pred_set - actual_parts)


def evaluate_portfolio(portfolio, actual_parts):
    """Find best set in portfolio"""
    best_wrong = 5
    for pred_set in portfolio:
        wrong = evaluate_set(pred_set, actual_parts)
        if wrong < best_wrong:
            best_wrong = wrong
    return best_wrong


def run_backtest(df, n_days=365):
    """Run backtest comparing ordered vs unordered approaches"""

    results = []

    for i in range(n_days, 0, -1):
        target_idx = len(df) - i
        if target_idx < CONFIG['rolling_window'] + 1:
            continue

        target_row = df.iloc[target_idx]
        actual_parts = frozenset([target_row['L_1'], target_row['L_2'],
                                  target_row['L_3'], target_row['L_4'],
                                  target_row['L_5']])

        # Generate unordered portfolio (adjacency bias)
        portfolio_adj = generate_portfolio(df, target_idx, bias='adjacency')
        best_adj = evaluate_portfolio(portfolio_adj, actual_parts)

        # Generate unordered portfolio (anti-adjacency bias)
        portfolio_anti = generate_portfolio(df, target_idx, bias='anti')
        best_anti = evaluate_portfolio(portfolio_anti, actual_parts)

        # Combined
        combined = list(set(portfolio_adj) | set(portfolio_anti))
        best_combined = evaluate_portfolio(combined, actual_parts)

        # Check diversity
        unique_parts_adj = set()
        for s in portfolio_adj:
            unique_parts_adj.update(s)

        unique_parts_anti = set()
        for s in portfolio_anti:
            unique_parts_anti.update(s)

        results.append({
            'date': target_row['date'],
            'best_adj': best_adj,
            'best_anti': best_anti,
            'best_combined': best_combined,
            'portfolio_adj_size': len(portfolio_adj),
            'portfolio_anti_size': len(portfolio_anti),
            'unique_parts_adj': len(unique_parts_adj),
            'unique_parts_anti': len(unique_parts_anti),
            'combined_size': len(combined),
        })

    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze backtest results"""

    print("\n" + "=" * 70)
    print("UNORDERED SET MODEL - ADJACENCY PORTFOLIO")
    print("=" * 70)

    n = len(results_df)
    counter = Counter(results_df['best_adj'])

    for wrong in range(6):
        count = counter.get(wrong, 0)
        pct = count / n * 100
        print(f"  {wrong} wrong: {count:>4} ({pct:>5.1f}%)")

    avg = results_df['best_adj'].mean()
    excellent = (results_df['best_adj'] <= 1).sum() / n * 100
    good = (results_df['best_adj'] <= 2).sum() / n * 100
    print(f"\n  Avg Wrong: {avg:.3f}")
    print(f"  Excellent (0-1): {excellent:.1f}%")
    print(f"  Good (0-2): {good:.1f}%")
    print(f"  Avg portfolio size: {results_df['portfolio_adj_size'].mean():.0f}")
    print(f"  Avg unique parts covered: {results_df['unique_parts_adj'].mean():.1f}")

    print("\n" + "=" * 70)
    print("UNORDERED SET MODEL - ANTI-ADJACENCY PORTFOLIO")
    print("=" * 70)

    counter = Counter(results_df['best_anti'])

    for wrong in range(6):
        count = counter.get(wrong, 0)
        pct = count / n * 100
        print(f"  {wrong} wrong: {count:>4} ({pct:>5.1f}%)")

    avg = results_df['best_anti'].mean()
    excellent = (results_df['best_anti'] <= 1).sum() / n * 100
    good = (results_df['best_anti'] <= 2).sum() / n * 100
    print(f"\n  Avg Wrong: {avg:.3f}")
    print(f"  Excellent (0-1): {excellent:.1f}%")
    print(f"  Good (0-2): {good:.1f}%")
    print(f"  Avg portfolio size: {results_df['portfolio_anti_size'].mean():.0f}")
    print(f"  Avg unique parts covered: {results_df['unique_parts_anti'].mean():.1f}")

    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO (ADJ + ANTI)")
    print("=" * 70)

    counter = Counter(results_df['best_combined'])

    for wrong in range(6):
        count = counter.get(wrong, 0)
        pct = count / n * 100
        print(f"  {wrong} wrong: {count:>4} ({pct:>5.1f}%)")

    avg = results_df['best_combined'].mean()
    excellent = (results_df['best_combined'] <= 1).sum() / n * 100
    good = (results_df['best_combined'] <= 2).sum() / n * 100
    print(f"\n  Avg Wrong: {avg:.3f}")
    print(f"  Excellent (0-1): {excellent:.1f}%")
    print(f"  Good (0-2): {good:.1f}%")
    print(f"  Avg combined size: {results_df['combined_size'].mean():.0f}")

    # Compare to original ordered model baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO ORDERED MODEL BASELINE")
    print("=" * 70)
    print(f"{'Model':<30} {'Avg Wrong':>12} {'Excellent':>12} {'Good':>10}")
    print("-" * 66)
    print(f"{'Original Ordered (200 sets)':<30} {'2.205':>12} {'6.3%':>12} {'73.2%':>10}")

    adj_avg = results_df['best_adj'].mean()
    adj_exc = (results_df['best_adj'] <= 1).sum() / n * 100
    adj_good = (results_df['best_adj'] <= 2).sum() / n * 100
    print(f"{'Unordered Adjacency (200)':<30} {adj_avg:>12.3f} {adj_exc:>11.1f}% {adj_good:>9.1f}%")

    comb_avg = results_df['best_combined'].mean()
    comb_exc = (results_df['best_combined'] <= 1).sum() / n * 100
    comb_good = (results_df['best_combined'] <= 2).sum() / n * 100
    print(f"{'Unordered Combined (400)':<30} {comb_avg:>12.3f} {comb_exc:>11.1f}% {comb_good:>9.1f}%")


def main():
    print("=" * 70)
    print("UNORDERED SET MODEL")
    print("=" * 70)
    print("Removing ascending constraint - just predict 5 unique parts")
    print(f"Portfolio size: {CONFIG['portfolio_size']}")

    df = load_data()
    print(f"Data loaded: {len(df)} days")

    print("\nRunning backtest (365 days)...")
    results_df = run_backtest(df, n_days=365)
    print(f"Backtest complete: {len(results_df)} days")

    analyze_results(results_df)


if __name__ == "__main__":
    main()
