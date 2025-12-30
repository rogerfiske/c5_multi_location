"""
Portfolio Agreement Analysis

Tests whether agreement between Portfolio A (adjacency) and Portfolio B (anti-adjacency)
predicts better outcomes.

Hypothesis: When both strategies agree, the prediction is more confident.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

CONFIG = {
    'portfolio_a_size': 100,
    'portfolio_b_size': 100,
    'adjacency_window': 3,
    'rolling_window': 30,
}


def load_data():
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def position_scores(df, current_idx, position, bias='adjacency'):
    window = CONFIG['rolling_window']
    l_col = f'L_{position}'
    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    if len(window_df) == 0:
        return {i: 1/39 for i in range(1, 40)}

    counts = window_df[l_col].value_counts()
    total = len(window_df)

    scores = {}
    for part_id in range(1, 40):
        scores[part_id] = counts.get(part_id, 0) / total + 0.01

    if current_idx > 0:
        prev_row = df.iloc[current_idx - 1]
        prev_val = prev_row[l_col]
        adj_window = CONFIG['adjacency_window']

        for part_id in range(1, 40):
            distance = abs(part_id - prev_val)

            if bias == 'adjacency':
                if distance <= adj_window:
                    boost = 3.0 - (distance * 0.5)
                    scores[part_id] *= max(boost, 1.0)
            else:
                if distance > adj_window:
                    boost = 1.0 + (distance / 39)
                    scores[part_id] *= boost
                else:
                    scores[part_id] *= 0.3

    return scores


def generate_set(df, current_idx, bias='adjacency', method='greedy'):
    predicted_set = []

    for position in range(1, 6):
        pos_scores = position_scores(df, current_idx, position, bias)

        if predicted_set:
            min_valid = max(predicted_set) + 1
        else:
            min_valid = 1

        valid_scores = {k: v for k, v in pos_scores.items() if k >= min_valid}

        if not valid_scores:
            remaining = [p for p in range(1, 40) if p not in predicted_set and p >= min_valid]
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

    return tuple(sorted(predicted_set)) if len(predicted_set) == 5 else None


def generate_portfolio(df, current_idx, bias='adjacency'):
    np.random.seed(current_idx + (0 if bias == 'adjacency' else 1000))

    portfolio = []
    greedy = generate_set(df, current_idx, bias=bias, method='greedy')
    if greedy:
        portfolio.append(greedy)

    attempts = 0
    size = CONFIG['portfolio_a_size'] if bias == 'adjacency' else CONFIG['portfolio_b_size']
    while len(portfolio) < size and attempts < size * 5:
        attempts += 1
        candidate = generate_set(df, current_idx, bias=bias, method='stochastic')
        if candidate and candidate not in portfolio:
            portfolio.append(candidate)

    return portfolio


def calculate_overlap(set1, set2):
    """Calculate overlap between two sets (0-5)"""
    return len(set(set1) & set(set2))


def get_best_set(portfolio, actual_set):
    """Get the best set and its wrong count"""
    actual_parts = set(actual_set)
    best_wrong = 5
    best_set = None

    for pred_set in portfolio:
        if pred_set is None:
            continue
        pred_parts = set(pred_set)
        wrong = len(pred_parts - actual_parts)
        if wrong < best_wrong:
            best_wrong = wrong
            best_set = pred_set

    return best_set, best_wrong


def run_analysis(df, n_days=365):
    """Run agreement analysis"""
    results = []

    for i in range(n_days, 0, -1):
        target_idx = len(df) - i
        if target_idx < CONFIG['rolling_window'] + 1:
            continue

        target_row = df.iloc[target_idx]
        actual_set = (target_row['L_1'], target_row['L_2'], target_row['L_3'],
                     target_row['L_4'], target_row['L_5'])

        # Generate portfolios
        portfolio_a = generate_portfolio(df, target_idx, bias='adjacency')
        portfolio_b = generate_portfolio(df, target_idx, bias='anti')

        # Get best sets from each
        best_a, wrong_a = get_best_set(portfolio_a, actual_set)
        best_b, wrong_b = get_best_set(portfolio_b, actual_set)

        if best_a is None or best_b is None:
            continue

        # Calculate agreement metrics
        greedy_a = portfolio_a[0] if portfolio_a else None
        greedy_b = portfolio_b[0] if portfolio_b else None

        # Overlap between greedy predictions
        greedy_overlap = calculate_overlap(greedy_a, greedy_b) if greedy_a and greedy_b else 0

        # Overlap between best predictions
        best_overlap = calculate_overlap(best_a, best_b)

        # Pool overlap (top 10 parts from each portfolio)
        pool_a = set()
        pool_b = set()
        for s in portfolio_a[:20]:
            if s:
                pool_a.update(s)
        for s in portfolio_b[:20]:
            if s:
                pool_b.update(s)
        pool_overlap = len(pool_a & pool_b)

        results.append({
            'date': target_row['date'],
            'wrong_a': wrong_a,
            'wrong_b': wrong_b,
            'best_wrong': min(wrong_a, wrong_b),
            'greedy_overlap': greedy_overlap,
            'best_overlap': best_overlap,
            'pool_overlap': pool_overlap,
        })

    return pd.DataFrame(results)


def analyze_agreement(results_df):
    """Analyze whether agreement predicts better outcomes"""

    print("\n" + "=" * 70)
    print("GREEDY OVERLAP ANALYSIS")
    print("=" * 70)
    print("Does agreement between greedy predictions predict better outcomes?")
    print()

    print(f"{'Overlap':>10} {'Count':>8} {'Avg Wrong':>12} {'Excellent%':>12} {'Good%':>10}")
    print("-" * 56)

    for overlap in range(6):
        subset = results_df[results_df['greedy_overlap'] == overlap]
        if len(subset) > 10:
            avg_wrong = subset['best_wrong'].mean()
            excellent = (subset['best_wrong'] <= 1).sum() / len(subset) * 100
            good = (subset['best_wrong'] <= 2).sum() / len(subset) * 100
            print(f"{overlap:>10} {len(subset):>8} {avg_wrong:>12.3f} {excellent:>11.1f}% {good:>9.1f}%")

    print("\n" + "=" * 70)
    print("POOL OVERLAP ANALYSIS")
    print("=" * 70)
    print("Does larger pool overlap predict better outcomes?")
    print()

    # Group by pool overlap quartiles
    quartiles = pd.qcut(results_df['pool_overlap'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'], duplicates='drop')

    print(f"{'Quartile':>15} {'Count':>8} {'Avg Wrong':>12} {'Excellent%':>12} {'Good%':>10}")
    print("-" * 62)

    for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        mask = quartiles == q
        subset = results_df[mask]
        if len(subset) > 0:
            avg_wrong = subset['best_wrong'].mean()
            excellent = (subset['best_wrong'] <= 1).sum() / len(subset) * 100
            good = (subset['best_wrong'] <= 2).sum() / len(subset) * 100
            print(f"{q:>15} {len(subset):>8} {avg_wrong:>12.3f} {excellent:>11.1f}% {good:>9.1f}%")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    print(f"Correlation between overlap metrics and best_wrong:")
    print(f"  greedy_overlap vs best_wrong: {results_df['greedy_overlap'].corr(results_df['best_wrong']):+.4f}")
    print(f"  best_overlap vs best_wrong:   {results_df['best_overlap'].corr(results_df['best_wrong']):+.4f}")
    print(f"  pool_overlap vs best_wrong:   {results_df['pool_overlap'].corr(results_df['best_wrong']):+.4f}")

    # Strategy: Use high-agreement days only
    print("\n" + "=" * 70)
    print("SELECTIVE PREDICTION STRATEGY")
    print("=" * 70)

    for threshold in [3, 4, 5]:
        high_agreement = results_df[results_df['greedy_overlap'] >= threshold]
        if len(high_agreement) > 10:
            avg_wrong = high_agreement['best_wrong'].mean()
            excellent = (high_agreement['best_wrong'] <= 1).sum() / len(high_agreement) * 100
            good = (high_agreement['best_wrong'] <= 2).sum() / len(high_agreement) * 100
            coverage = len(high_agreement) / len(results_df) * 100

            print(f"Predict only when greedy_overlap >= {threshold}:")
            print(f"  Coverage: {coverage:.1f}% of days ({len(high_agreement)} days)")
            print(f"  Avg Wrong: {avg_wrong:.3f}")
            print(f"  Excellent (0-1): {excellent:.1f}%")
            print(f"  Good (0-2): {good:.1f}%")
            print()


def main():
    print("=" * 70)
    print("PORTFOLIO AGREEMENT ANALYSIS")
    print("=" * 70)

    df = load_data()
    print(f"Data loaded: {len(df)} days")

    print("\nRunning analysis (365 days)...")
    results_df = run_analysis(df, n_days=365)
    print(f"Analysis complete: {len(results_df)} days")

    analyze_agreement(results_df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Key question: Does agreement between adjacency and anti-adjacency")
    print("portfolios indicate more confident/accurate predictions?")


if __name__ == "__main__":
    main()
