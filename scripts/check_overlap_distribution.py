"""
Check the actual pool overlap distribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

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

    size = CONFIG['portfolio_a_size'] if bias == 'adjacency' else CONFIG['portfolio_b_size']
    attempts = 0
    while len(portfolio) < size and attempts < size * 5:
        attempts += 1
        candidate = generate_set(df, current_idx, bias=bias, method='stochastic')
        if candidate and candidate not in portfolio:
            portfolio.append(candidate)

    return portfolio


def calculate_pool_overlap(portfolio_a, portfolio_b, top_n=20):
    pool_a = set()
    pool_b = set()

    for s in portfolio_a[:top_n]:
        if s:
            pool_a.update(s)
    for s in portfolio_b[:top_n]:
        if s:
            pool_b.update(s)

    return len(pool_a & pool_b), len(pool_a), len(pool_b)


def main():
    df = load_data()
    print(f"Data loaded: {len(df)} days")

    overlaps = []

    for i in range(365, 0, -1):
        idx = len(df) - i
        if idx < CONFIG['rolling_window']:
            continue

        portfolio_a = generate_portfolio(df, idx, bias='adjacency')
        portfolio_b = generate_portfolio(df, idx, bias='anti')

        overlap, size_a, size_b = calculate_pool_overlap(portfolio_a, portfolio_b)
        overlaps.append(overlap)

    print(f"\nPool Overlap Distribution (365 days):")
    print(f"  Min: {min(overlaps)}")
    print(f"  Max: {max(overlaps)}")
    print(f"  Mean: {np.mean(overlaps):.1f}")
    print(f"  Median: {np.median(overlaps):.0f}")

    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th: {np.percentile(overlaps, p):.0f}")

    print(f"\nDistribution:")
    counter = Counter(overlaps)
    for overlap in sorted(counter.keys()):
        count = counter[overlap]
        pct = count / len(overlaps) * 100
        bar = '#' * int(pct / 2)
        print(f"  {overlap:>2}: {count:>3} ({pct:>5.1f}%) {bar}")


if __name__ == "__main__":
    main()
