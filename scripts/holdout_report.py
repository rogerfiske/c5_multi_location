"""
Holdout Test Reports for C5 Parts Forecasting

Generates reports in user-specified format:
1. Pooled 20 Most Likely Parts report
2. N-sets per day report

Usage:
    python scripts/holdout_report.py
    python scripts/holdout_report.py --sets 50    # Use 50 sets instead of 200
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import Counter

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_actual_parts(df, target_date):
    """Get actual parts for a specific date"""
    row = df[df['date'] == target_date]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    return set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def get_actual_set(df, target_date):
    """Get actual set for a specific date"""
    row = df[df['date'] == target_date]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    return tuple([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])


def load_pool_prediction(date_str):
    """Load pool prediction CSV"""
    path = PREDICTIONS_DIR / f"{date_str}_pool.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_sets_prediction(date_str):
    """Load sets prediction CSV"""
    path = PREDICTIONS_DIR / f"{date_str}_sets.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def evaluate_pool_wrong(pool_df, actual_parts, k=20):
    """
    Calculate how many actual parts are NOT in top-K pool.
    Returns: number of "wrong" (0-5)
    """
    top_k = set(pool_df.head(k)['part_id'].tolist())
    hits = len(actual_parts & top_k)
    wrong = 5 - hits  # How many actual parts are missing from top-K
    return wrong


def evaluate_set_wrong(pred_set, actual_set):
    """
    Calculate how many parts in predicted set are wrong.
    Returns: number of "wrong" (0-5)
    """
    pred = set(pred_set)
    actual = set(actual_set)
    wrong = len(pred - actual)
    return wrong


def generate_pool_report(df, n_sets_limit=None):
    """Generate Pool-based holdout report"""
    # Find all prediction files
    pool_files = list(PREDICTIONS_DIR.glob("*_pool.csv"))
    dates = sorted([f.stem.replace('_pool', '') for f in pool_files])

    if not dates:
        print("No prediction files found")
        return

    # Evaluate each date
    wrong_counts = []

    for date_str in dates:
        target_date = pd.to_datetime(date_str)
        pool_df = load_pool_prediction(date_str)
        actual_parts = get_actual_parts(df, target_date)

        if pool_df is None or actual_parts is None:
            continue

        wrong = evaluate_pool_wrong(pool_df, actual_parts, k=20)
        wrong_counts.append(wrong)

    n_events = len(wrong_counts)
    counter = Counter(wrong_counts)

    # Print report
    print("=" * 60)
    print("Pooled 20 Most Likely parts next prediction Day")
    print("=" * 60)
    print(f"  HOLDOUT TEST SUMMARY - {n_events} Events")
    print("  " + "-" * 56)

    for wrong in range(6):
        count = counter.get(wrong, 0)
        pct = count / n_events * 100
        print(f"  {wrong} wrong: {count:>4} events ({pct:>5.2f}%)")

    print()
    print("  ACCURACY DISTRIBUTION")
    print("  " + "-" * 56)

    # Excellent: 0 wrong OR 5 wrong (all in or all out - predictable)
    excellent = counter.get(0, 0) + counter.get(5, 0)
    excellent_pct = excellent / n_events * 100

    # Good: 1 wrong OR 4 wrong
    good = counter.get(1, 0) + counter.get(4, 0)
    good_pct = good / n_events * 100

    # Poor: 2-3 wrong
    poor = counter.get(2, 0) + counter.get(3, 0)
    poor_pct = poor / n_events * 100

    print(f"  Excellent : {excellent:>4} events ({excellent_pct:>5.2f}%)  [0 wrong OR 5 wrong]")
    print(f"  Good      : {good:>4} events ({good_pct:>5.2f}%)  [1 wrong OR 4 wrong]")
    print(f"  Poor      : {poor:>4} events ({poor_pct:>5.2f}%)  [2-3 wrong]")

    print()
    good_plus = excellent + good
    good_plus_pct = good_plus / n_events * 100
    print(f"  Good+ (0-1 or 4-5 wrong): {good_plus} events ({good_plus_pct:.1f}%)")
    print("-" * 60)

    return counter


def generate_sets_report(df, n_sets_limit=50):
    """Generate Sets-based holdout report"""
    # Find all prediction files
    sets_files = list(PREDICTIONS_DIR.glob("*_sets.csv"))
    dates = sorted([f.stem.replace('_sets', '') for f in sets_files])

    if not dates:
        print("No prediction files found")
        return

    # Evaluate each date
    best_wrong_counts = []

    for date_str in dates:
        target_date = pd.to_datetime(date_str)
        sets_df = load_sets_prediction(date_str)
        actual_set = get_actual_set(df, target_date)

        if sets_df is None or actual_set is None:
            continue

        # Limit to first N sets if specified
        if n_sets_limit:
            sets_df = sets_df.head(n_sets_limit)

        # Find best set (minimum wrong)
        best_wrong = 5
        for _, row in sets_df.iterrows():
            pred_set = (row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5'])
            wrong = evaluate_set_wrong(pred_set, actual_set)
            if wrong < best_wrong:
                best_wrong = wrong

        best_wrong_counts.append(best_wrong)

    n_events = len(best_wrong_counts)
    counter = Counter(best_wrong_counts)

    # Print report
    print()
    print("=" * 60)
    print(f"Holdout Test of {n_sets_limit}-sets / prediction day")
    print("=" * 60)
    print(f"  HOLDOUT TEST - {n_events} days")
    print(f"  {n_sets_limit}-sets (L_1 to L_5 values)/day")
    print("  " + "-" * 56)

    for wrong in range(6):
        count = counter.get(wrong, 0)
        pct = count / n_events * 100
        print(f"  {wrong} wrong: {count:>4} events ({pct:>5.2f}%)")

    print()
    print("  ACCURACY DISTRIBUTION")
    print("  " + "-" * 56)

    # Correct: 0-1 wrong
    correct = counter.get(0, 0) + counter.get(1, 0)
    correct_pct = correct / n_events * 100

    # Good: 0-2 wrong
    good = correct + counter.get(2, 0)
    good_pct = good / n_events * 100

    # Acceptable: 0-3 wrong
    acceptable = good + counter.get(3, 0)
    acceptable_pct = acceptable / n_events * 100

    print(f"  Correct (0-1 wrong): {correct:>4} events ({correct_pct:>5.2f}%)")
    print(f"  Good    (0-2 wrong): {good:>4} events ({good_pct:>5.2f}%)")
    print(f"  Acceptable (0-3):    {acceptable:>4} events ({acceptable_pct:>5.2f}%)")

    print()
    avg_wrong = np.mean(best_wrong_counts)
    print(f"  Average wrong: {avg_wrong:.3f}")
    print("-" * 60)

    return counter


def main():
    parser = argparse.ArgumentParser(description='Generate holdout test reports')
    parser.add_argument('--sets', type=int, default=50, help='Number of sets to evaluate (default: 50)')
    args = parser.parse_args()

    # Load actual data
    df = load_data()

    # Generate reports
    print()
    generate_pool_report(df)
    print()
    generate_sets_report(df, n_sets_limit=args.sets)
    print()


if __name__ == "__main__":
    main()
