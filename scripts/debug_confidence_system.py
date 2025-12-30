"""
Debug the confidence prediction system
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

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


def main():
    df = load_data()

    # Test on a single day
    target_idx = len(df) - 10  # 10 days ago
    target_row = df.iloc[target_idx]
    actual_set = (target_row['L_1'], target_row['L_2'], target_row['L_3'],
                 target_row['L_4'], target_row['L_5'])
    actual_parts = set(actual_set)

    print(f"Target date: {target_row['date']}")
    print(f"Actual set: {actual_set}")
    print(f"Actual parts: {actual_parts}")

    # Previous day
    prev_row = df.iloc[target_idx - 1]
    prev_set = (prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                prev_row['L_4'], prev_row['L_5'])
    print(f"\nPrevious day set: {prev_set}")

    # Generate adjacency set
    print("\n--- ADJACENCY BIAS ---")
    adj_set = generate_set(df, target_idx, bias='adjacency', method='greedy')
    print(f"Predicted set: {adj_set}")
    adj_wrong = len(set(adj_set) - actual_parts)
    print(f"Wrong: {adj_wrong}")

    # Generate anti-adjacency set
    print("\n--- ANTI-ADJACENCY BIAS ---")
    anti_set = generate_set(df, target_idx, bias='anti', method='greedy')
    print(f"Predicted set: {anti_set}")
    anti_wrong = len(set(anti_set) - actual_parts)
    print(f"Wrong: {anti_wrong}")

    # Check scores for position 1
    print("\n--- POSITION 1 SCORES ---")
    adj_scores = position_scores(df, target_idx, 1, bias='adjacency')
    anti_scores = position_scores(df, target_idx, 1, bias='anti')

    print(f"Previous L_1: {prev_row['L_1']}")
    print(f"\nTop 5 adjacency scores:")
    for p, s in sorted(adj_scores.items(), key=lambda x: -x[1])[:5]:
        print(f"  Part {p}: {s:.4f}")

    print(f"\nTop 5 anti-adjacency scores:")
    for p, s in sorted(anti_scores.items(), key=lambda x: -x[1])[:5]:
        print(f"  Part {p}: {s:.4f}")

    # Run small backtest
    print("\n" + "=" * 60)
    print("SMALL BACKTEST (30 days)")
    print("=" * 60)

    adj_wrongs = []
    anti_wrongs = []

    for i in range(30, 0, -1):
        idx = len(df) - i
        actual_row = df.iloc[idx]
        actual = set([actual_row['L_1'], actual_row['L_2'], actual_row['L_3'],
                     actual_row['L_4'], actual_row['L_5']])

        adj = generate_set(df, idx, bias='adjacency', method='greedy')
        anti = generate_set(df, idx, bias='anti', method='greedy')

        if adj:
            adj_wrongs.append(len(set(adj) - actual))
        if anti:
            anti_wrongs.append(len(set(anti) - actual))

    print(f"Adjacency greedy - avg wrong: {np.mean(adj_wrongs):.3f}")
    print(f"Anti-adjacency greedy - avg wrong: {np.mean(anti_wrongs):.3f}")


if __name__ == "__main__":
    main()
