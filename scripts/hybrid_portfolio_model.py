"""
Hybrid Portfolio Model

Strategy: Generate sets from BOTH approaches:
1. Set A: Adjacency-biased (follow recent pattern)
2. Set B: Anti-adjacency biased (fade recent pattern)

This ensures we always have coverage in both directions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

CONFIG = {
    'portfolio_a_size': 100,  # Adjacency-biased sets
    'portfolio_b_size': 100,  # Anti-adjacency biased sets
    'adjacency_window': 3,
    'rolling_window': 30,
}


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_parts_set(row):
    """Extract parts as set"""
    return set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def position_scores(df, current_idx, position, bias='adjacency'):
    """
    Get position-specific scores with adjacency or anti-adjacency bias

    bias='adjacency': Boost parts near previous day's value
    bias='anti': Boost parts FAR from previous day's value
    """
    window = CONFIG['rolling_window']
    l_col = f'L_{position}'
    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    if len(window_df) == 0:
        return {i: 1/39 for i in range(1, 40)}

    # Base: position frequency
    counts = window_df[l_col].value_counts()
    total = len(window_df)

    scores = {}
    for part_id in range(1, 40):
        scores[part_id] = counts.get(part_id, 0) / total + 0.01  # Small smoothing

    # Apply adjacency or anti-adjacency bias
    if current_idx > 0:
        prev_row = df.iloc[current_idx - 1]
        prev_val = prev_row[l_col]
        adj_window = CONFIG['adjacency_window']

        for part_id in range(1, 40):
            distance = abs(part_id - prev_val)

            if bias == 'adjacency':
                # Boost nearby parts
                if distance <= adj_window:
                    boost = 3.0 - (distance * 0.5)  # 3.0 for exact, 2.5, 2.0, 1.5
                    scores[part_id] *= max(boost, 1.0)
            else:  # anti-adjacency
                # Boost distant parts
                if distance > adj_window:
                    boost = 1.0 + (distance / 39)  # More boost for further parts
                    scores[part_id] *= boost
                else:
                    # Suppress nearby parts
                    scores[part_id] *= 0.3

    return scores


def generate_set(df, current_idx, bias='adjacency', method='greedy'):
    """Generate a single candidate set with specified bias"""
    predicted_set = []

    for position in range(1, 6):
        pos_scores = position_scores(df, current_idx, position, bias)

        # Filter to valid candidates (must be > all previously selected)
        if predicted_set:
            min_valid = max(predicted_set) + 1
        else:
            min_valid = 1

        valid_scores = {k: v for k, v in pos_scores.items() if k >= min_valid}

        if not valid_scores:
            # Fallback: pick smallest available
            remaining = [p for p in range(1, 40) if p not in predicted_set and p >= min_valid]
            if remaining:
                predicted_set.append(min(remaining))
            continue

        if method == 'greedy':
            best_part = max(valid_scores.items(), key=lambda x: x[1])[0]
            predicted_set.append(best_part)
        else:  # stochastic
            parts = list(valid_scores.keys())
            probs = np.array(list(valid_scores.values()))
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
            selected = np.random.choice(parts, p=probs)
            predicted_set.append(selected)

    return tuple(sorted(predicted_set)) if len(predicted_set) == 5 else None


def generate_portfolio(df, current_idx):
    """Generate hybrid portfolio with both adjacency and anti-adjacency sets"""
    np.random.seed(current_idx)  # Reproducibility

    portfolio_a = []  # Adjacency-biased
    portfolio_b = []  # Anti-adjacency biased

    # Generate Set A portfolio (adjacency)
    greedy_a = generate_set(df, current_idx, bias='adjacency', method='greedy')
    if greedy_a:
        portfolio_a.append(greedy_a)

    attempts = 0
    while len(portfolio_a) < CONFIG['portfolio_a_size'] and attempts < CONFIG['portfolio_a_size'] * 5:
        attempts += 1
        candidate = generate_set(df, current_idx, bias='adjacency', method='stochastic')
        if candidate and candidate not in portfolio_a:
            portfolio_a.append(candidate)

    # Generate Set B portfolio (anti-adjacency)
    greedy_b = generate_set(df, current_idx, bias='anti', method='greedy')
    if greedy_b:
        portfolio_b.append(greedy_b)

    attempts = 0
    while len(portfolio_b) < CONFIG['portfolio_b_size'] and attempts < CONFIG['portfolio_b_size'] * 5:
        attempts += 1
        candidate = generate_set(df, current_idx, bias='anti', method='stochastic')
        if candidate and candidate not in portfolio_b:
            portfolio_b.append(candidate)

    return portfolio_a, portfolio_b


def evaluate_portfolio(portfolio, actual_set):
    """Evaluate a portfolio against actual outcome"""
    actual_parts = set(actual_set)

    wrong_counts = []
    for pred_set in portfolio:
        if pred_set is None:
            continue
        pred_parts = set(pred_set)
        wrong = len(pred_parts - actual_parts)
        wrong_counts.append(wrong)

    if not wrong_counts:
        return 5, 5, []

    best_wrong = min(wrong_counts)
    greedy_wrong = wrong_counts[0] if wrong_counts else 5

    return best_wrong, greedy_wrong, wrong_counts


def run_backtest(df, n_days=365):
    """Run backtest on last n_days"""
    results = []

    for i in range(n_days, 0, -1):
        target_idx = len(df) - i
        if target_idx < CONFIG['rolling_window'] + 1:
            continue

        target_row = df.iloc[target_idx]
        actual_set = (target_row['L_1'], target_row['L_2'], target_row['L_3'],
                     target_row['L_4'], target_row['L_5'])

        # Generate portfolios
        portfolio_a, portfolio_b = generate_portfolio(df, target_idx)

        # Evaluate each portfolio
        best_a, greedy_a, wrongs_a = evaluate_portfolio(portfolio_a, actual_set)
        best_b, greedy_b, wrongs_b = evaluate_portfolio(portfolio_b, actual_set)

        # Combined portfolio
        combined = portfolio_a + portfolio_b
        best_combined, _, wrongs_combined = evaluate_portfolio(combined, actual_set)

        results.append({
            'date': target_row['date'],
            'best_a': best_a,
            'best_b': best_b,
            'best_combined': best_combined,
            'greedy_a': greedy_a,
            'greedy_b': greedy_b,
            'portfolio_a_size': len(portfolio_a),
            'portfolio_b_size': len(portfolio_b),
        })

    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze backtest results"""
    print("\n" + "=" * 70)
    print("PORTFOLIO A (Adjacency-Biased) RESULTS")
    print("=" * 70)

    counter_a = Counter(results_df['best_a'])
    for wrong in range(6):
        count = counter_a.get(wrong, 0)
        pct = count / len(results_df) * 100
        print(f"  {wrong} wrong: {count:>4} ({pct:>5.1f}%)")

    avg_a = results_df['best_a'].mean()
    print(f"\n  Average wrong: {avg_a:.3f}")

    print("\n" + "=" * 70)
    print("PORTFOLIO B (Anti-Adjacency Biased) RESULTS")
    print("=" * 70)

    counter_b = Counter(results_df['best_b'])
    for wrong in range(6):
        count = counter_b.get(wrong, 0)
        pct = count / len(results_df) * 100
        print(f"  {wrong} wrong: {count:>4} ({pct:>5.1f}%)")

    avg_b = results_df['best_b'].mean()
    print(f"\n  Average wrong: {avg_b:.3f}")

    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO (A + B) RESULTS")
    print("=" * 70)

    counter_combined = Counter(results_df['best_combined'])
    for wrong in range(6):
        count = counter_combined.get(wrong, 0)
        pct = count / len(results_df) * 100
        print(f"  {wrong} wrong: {count:>4} ({pct:>5.1f}%)")

    avg_combined = results_df['best_combined'].mean()
    print(f"\n  Average wrong: {avg_combined:.3f}")

    # Comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    # Best of A alone
    excellent_a = sum(1 for w in results_df['best_a'] if w <= 1)
    good_a = sum(1 for w in results_df['best_a'] if w <= 2)

    # Best of B alone
    excellent_b = sum(1 for w in results_df['best_b'] if w <= 1)
    good_b = sum(1 for w in results_df['best_b'] if w <= 2)

    # Best of combined
    excellent_combined = sum(1 for w in results_df['best_combined'] if w <= 1)
    good_combined = sum(1 for w in results_df['best_combined'] if w <= 2)

    n = len(results_df)
    print(f"{'Strategy':<25} {'Avg Wrong':>12} {'Excellent (0-1)':>16} {'Good (0-2)':>12}")
    print("-" * 70)
    print(f"{'Portfolio A (100 sets)':<25} {avg_a:>12.3f} {excellent_a/n*100:>15.1f}% {good_a/n*100:>11.1f}%")
    print(f"{'Portfolio B (100 sets)':<25} {avg_b:>12.3f} {excellent_b/n*100:>15.1f}% {good_b/n*100:>11.1f}%")
    print(f"{'Combined (200 sets)':<25} {avg_combined:>12.3f} {excellent_combined/n*100:>15.1f}% {good_combined/n*100:>11.1f}%")

    # Oracle: if we knew which to use
    oracle_best = []
    for _, row in results_df.iterrows():
        oracle_best.append(min(row['best_a'], row['best_b']))

    oracle_avg = np.mean(oracle_best)
    oracle_excellent = sum(1 for w in oracle_best if w <= 1)
    oracle_good = sum(1 for w in oracle_best if w <= 2)

    print(f"{'Oracle (best of A or B)':<25} {oracle_avg:>12.3f} {oracle_excellent/n*100:>15.1f}% {oracle_good/n*100:>11.1f}%")

    # Analyze when each portfolio wins
    print("\n" + "=" * 70)
    print("WHEN DOES EACH PORTFOLIO WIN?")
    print("=" * 70)

    a_wins = sum(1 for _, r in results_df.iterrows() if r['best_a'] < r['best_b'])
    b_wins = sum(1 for _, r in results_df.iterrows() if r['best_b'] < r['best_a'])
    ties = sum(1 for _, r in results_df.iterrows() if r['best_a'] == r['best_b'])

    print(f"Portfolio A wins: {a_wins} ({a_wins/n*100:.1f}%)")
    print(f"Portfolio B wins: {b_wins} ({b_wins/n*100:.1f}%)")
    print(f"Ties:             {ties} ({ties/n*100:.1f}%)")


def main():
    print("=" * 70)
    print("HYBRID PORTFOLIO MODEL")
    print("=" * 70)
    print(f"Portfolio A size: {CONFIG['portfolio_a_size']} (adjacency-biased)")
    print(f"Portfolio B size: {CONFIG['portfolio_b_size']} (anti-adjacency biased)")

    df = load_data()
    print(f"Data loaded: {len(df)} days")

    # Run backtest
    print("\nRunning backtest (365 days)...")
    results_df = run_backtest(df, n_days=365)
    print(f"Backtest complete: {len(results_df)} days evaluated")

    # Analyze results
    analyze_results(results_df)


if __name__ == "__main__":
    main()
