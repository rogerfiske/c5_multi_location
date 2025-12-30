"""
Holdout Evaluation Script for C5 Parts Forecasting

Reads predictions from CSV files and compares to actual outcomes.
Provides strict holdout evaluation metrics.

Usage:
    python scripts/evaluate_predictions.py                     # Evaluate all predictions
    python scripts/evaluate_predictions.py --date 2025-01-15   # Evaluate specific date
    python scripts/evaluate_predictions.py --summary           # Summary statistics only

Input Files (from predict_next_day.py):
    predictions/{date}_pool.csv
    predictions/{date}_sets.csv

Output:
    Metrics comparing predictions to actual outcomes
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
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


def evaluate_pool(pool_df, actual_parts, k_values=[10, 15, 20, 25, 30]):
    """Evaluate pool ranking against actual parts"""
    results = {}

    for k in k_values:
        top_k = set(pool_df.head(k)['part_id'].tolist())
        hits = len(actual_parts & top_k)
        recall = hits / len(actual_parts)
        results[f'recall@{k}'] = recall
        results[f'hits@{k}'] = hits

    # Find ranks of actual parts
    ranks = []
    for part in actual_parts:
        rank_row = pool_df[pool_df['part_id'] == part]
        if len(rank_row) > 0:
            ranks.append(rank_row.iloc[0]['rank'])
        else:
            ranks.append(40)  # Not in pool

    results['actual_part_ranks'] = sorted(ranks)
    results['mean_rank'] = np.mean(ranks)

    return results


def evaluate_sets(sets_df, actual_set):
    """Evaluate candidate sets against actual outcome"""
    actual_set_parts = set(actual_set)
    results = {}

    # Score each set
    wrong_counts = []
    for _, row in sets_df.iterrows():
        pred_set = set([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])
        wrong = len(pred_set - actual_set_parts)
        wrong_counts.append(wrong)

    results['all_wrong_counts'] = wrong_counts
    results['best_wrong'] = min(wrong_counts)
    results['greedy_wrong'] = wrong_counts[0]  # First set is greedy
    results['best_set_rank'] = wrong_counts.index(min(wrong_counts)) + 1

    # Distribution
    results['wrong_distribution'] = dict(Counter(wrong_counts))

    # Check if any set matches exactly
    for i, row in sets_df.iterrows():
        pred_set = tuple([row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']])
        if pred_set == actual_set:
            results['exact_match_rank'] = i + 1
            break
    else:
        results['exact_match_rank'] = None

    return results


def evaluate_single_date(df, date_str, verbose=True):
    """Evaluate predictions for a single date"""
    target_date = pd.to_datetime(date_str)

    # Load predictions
    pool_df = load_pool_prediction(date_str)
    sets_df = load_sets_prediction(date_str)

    if pool_df is None or sets_df is None:
        if verbose:
            print(f"No predictions found for {date_str}")
        return None

    # Get actuals
    actual_parts = get_actual_parts(df, target_date)
    actual_set = get_actual_set(df, target_date)

    if actual_parts is None:
        if verbose:
            print(f"No actual data found for {date_str}")
        return None

    # Evaluate
    pool_results = evaluate_pool(pool_df, actual_parts)
    sets_results = evaluate_sets(sets_df, actual_set)

    results = {
        'date': date_str,
        'actual_parts': list(actual_parts),
        'actual_set': list(actual_set),
        **pool_results,
        **sets_results
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EVALUATION: {date_str}")
        print(f"{'=' * 60}")

        print(f"\nActual parts: {sorted(actual_parts)}")
        print(f"Actual set:   [{actual_set[0]}, {actual_set[1]}, {actual_set[2]}, {actual_set[3]}, {actual_set[4]}]")

        print(f"\n--- POOL METRICS ---")
        print(f"Recall@10: {pool_results['recall@10']:.1%} ({pool_results['hits@10']}/5 parts)")
        print(f"Recall@15: {pool_results['recall@15']:.1%} ({pool_results['hits@15']}/5 parts)")
        print(f"Recall@20: {pool_results['recall@20']:.1%} ({pool_results['hits@20']}/5 parts)")
        print(f"Mean rank of actual parts: {pool_results['mean_rank']:.1f}")
        print(f"Actual part ranks: {pool_results['actual_part_ranks']}")

        print(f"\n--- SET METRICS ---")
        print(f"Greedy set wrong: {sets_results['greedy_wrong']}")
        print(f"Best set wrong:   {sets_results['best_wrong']} (rank {sets_results['best_set_rank']})")
        if sets_results['exact_match_rank']:
            print(f"EXACT MATCH at rank {sets_results['exact_match_rank']}!")

    return results


def evaluate_all_predictions(df, verbose=True):
    """Evaluate all prediction files in predictions/ directory"""
    if not PREDICTIONS_DIR.exists():
        print("No predictions directory found")
        return []

    # Find all prediction files
    pool_files = list(PREDICTIONS_DIR.glob("*_pool.csv"))
    dates = [f.stem.replace('_pool', '') for f in pool_files]
    dates.sort()

    if not dates:
        print("No prediction files found")
        return []

    print(f"Found {len(dates)} prediction files")

    results = []
    for date_str in dates:
        result = evaluate_single_date(df, date_str, verbose=False)
        if result:
            results.append(result)

    return results


def print_summary(results):
    """Print summary statistics across all evaluated predictions"""
    if not results:
        print("No results to summarize")
        return

    print(f"\n{'=' * 70}")
    print("HOLDOUT EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Evaluated: {len(results)} days")
    print(f"Date range: {results[0]['date']} to {results[-1]['date']}")

    # Pool metrics
    recall_20 = np.mean([r['recall@20'] for r in results])
    recall_10 = np.mean([r['recall@10'] for r in results])
    mean_rank = np.mean([r['mean_rank'] for r in results])

    print(f"\n--- POOL PERFORMANCE ---")
    print(f"Avg Recall@10: {recall_10:.1%}")
    print(f"Avg Recall@20: {recall_20:.1%}")
    print(f"Avg Mean Rank: {mean_rank:.1f}")

    # Set metrics
    best_wrong = [r['best_wrong'] for r in results]
    greedy_wrong = [r['greedy_wrong'] for r in results]

    avg_best_wrong = np.mean(best_wrong)
    avg_greedy_wrong = np.mean(greedy_wrong)

    best_counter = Counter(best_wrong)
    correct_rate = (best_counter.get(0, 0) + best_counter.get(1, 0)) / len(results) * 100
    good_rate = sum(best_counter.get(w, 0) for w in [0, 1, 2]) / len(results) * 100

    print(f"\n--- SET PERFORMANCE ---")
    print(f"Avg Greedy Wrong: {avg_greedy_wrong:.3f}")
    print(f"Avg Best Wrong:   {avg_best_wrong:.3f}")
    print(f"Correct Rate (0-1 wrong): {correct_rate:.2f}%")
    print(f"Good Rate (0-2 wrong):    {good_rate:.2f}%")

    print(f"\n--- BEST WRONG DISTRIBUTION ---")
    print(f"{'Wrong':>6} {'Count':>8} {'Percent':>10}")
    print("-" * 28)
    for wrong in range(6):
        count = best_counter.get(wrong, 0)
        pct = count / len(results) * 100
        print(f"{wrong:>6} {count:>8} {pct:>9.2f}%")

    # Exact matches
    exact_matches = sum(1 for r in results if r['exact_match_rank'] is not None)
    if exact_matches > 0:
        print(f"\nExact set matches: {exact_matches} ({exact_matches/len(results)*100:.2f}%)")

    # Compare to expected
    print(f"\n--- COMPARISON TO BACKTEST ---")
    print(f"{'Metric':<25} {'Backtest':>12} {'Holdout':>12} {'Delta':>10}")
    print("-" * 62)
    print(f"{'Avg Best Wrong':<25} {'2.217':>12} {avg_best_wrong:>12.3f} {avg_best_wrong - 2.217:>+10.3f}")
    print(f"{'Good Rate (0-2)':<25} {'72.25%':>12} {good_rate:>11.2f}% {good_rate - 72.25:>+9.2f}%")
    print(f"{'Correct Rate (0-1)':<25} {'6.04%':>12} {correct_rate:>11.2f}% {correct_rate - 6.04:>+9.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions against actual outcomes')
    parser.add_argument('--date', type=str, help='Evaluate specific date (YYYY-MM-DD)')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    args = parser.parse_args()

    print("=" * 70)
    print("C5 PARTS FORECASTING - HOLDOUT EVALUATION")
    print("=" * 70)

    # Load actual data
    df = load_data()
    print(f"Data loaded: {len(df)} days")

    if args.date:
        # Single date evaluation
        evaluate_single_date(df, args.date, verbose=True)
    else:
        # Evaluate all
        results = evaluate_all_predictions(df, verbose=not args.summary)
        if results:
            print_summary(results)

    return 0


if __name__ == "__main__":
    exit(main())
