"""
Forward Prediction Script for C5 Parts Forecasting

Generates predictions for a target date using only data available BEFORE that date.
Outputs CSV files for strict holdout evaluation.

Usage:
    python scripts/predict_next_day.py                    # Predict for tomorrow
    python scripts/predict_next_day.py --date 2025-01-15  # Predict for specific date
    python scripts/predict_next_day.py --backtest 30      # Generate predictions for last 30 days

Output Files:
    predictions/{date}_pool.csv      - Ranked pool of parts with scores
    predictions/{date}_sets.csv      - Ranked candidate sets with scores
    predictions/{date}_metadata.json - Model configuration and generation info
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"

# Optimal configuration (from tuning)
CONFIG = {
    'portfolio_size': 200,
    'adjacency_window': 3,
    'edge_boost': 3.0,
    'middle_boost': 2.0,
    'markov_weight': 0.3,
    'rolling_window': 30,
    'pool_size': 20,
    'model_version': '1.0.0',
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


def build_markov_matrix(df, end_idx):
    """Build Markov transition matrix from training data"""
    train_df = df.iloc[:end_idx]

    repeat_counts = defaultdict(int)
    appearance_counts = defaultdict(int)
    co_occurrence = defaultdict(Counter)

    for i in range(1, len(train_df)):
        today_parts = get_parts_set(train_df.iloc[i-1])
        tomorrow_parts = get_parts_set(train_df.iloc[i])

        for part in today_parts:
            appearance_counts[part] += 1
            if part in tomorrow_parts:
                repeat_counts[part] += 1
            for tomorrow_part in tomorrow_parts:
                co_occurrence[part][tomorrow_part] += 1

    repeat_prob = {}
    for part in range(1, 40):
        if appearance_counts[part] > 0:
            repeat_prob[part] = repeat_counts[part] / appearance_counts[part]
        else:
            repeat_prob[part] = 0.0

    p_cols = [f'P_{i}' for i in range(1, 40)]
    overall_prob = {}
    for col in p_cols:
        part_id = int(col.replace('P_', ''))
        overall_prob[part_id] = train_df[col].sum() / len(train_df)

    return repeat_prob, co_occurrence, overall_prob


def markov_scores(today_parts, repeat_prob, co_occurrence, overall_prob):
    """Generate Markov scores for tomorrow"""
    scores = {}
    markov_weight = CONFIG['markov_weight']

    for part in range(1, 40):
        base_score = overall_prob.get(part, 1/39)
        markov_signal = 0.0

        if part in today_parts:
            markov_signal += repeat_prob.get(part, 0) * 2

        for today_part in today_parts:
            co_count = co_occurrence[today_part].get(part, 0)
            total_transitions = sum(co_occurrence[today_part].values())
            if total_transitions > 0:
                markov_signal += co_count / total_transitions

        markov_signal = markov_signal / (len(today_parts) + 1)
        scores[part] = markov_weight * markov_signal + (1 - markov_weight) * base_score

    total = sum(scores.values())
    if total > 0:
        scores = {k: v/total for k, v in scores.items()}

    return scores


def position_specific_scores(df, current_idx, position):
    """Get position-specific frequency scores"""
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
        scores[part_id] = counts.get(part_id, 0) / total

    return scores


def generate_pool_ranking(df, current_idx, markov):
    """Generate ranked pool of parts for tomorrow"""
    # Combine position-specific scores with Markov
    combined = defaultdict(float)

    for position in range(1, 6):
        pos_scores = position_specific_scores(df, current_idx, position)
        for part, score in pos_scores.items():
            combined[part] += score

    # Add Markov signal
    for part, score in markov.items():
        combined[part] += score * CONFIG['markov_weight']

    # Normalize
    total = sum(combined.values())
    if total > 0:
        combined = {k: v/total for k, v in combined.items()}

    # Rank by score
    ranked = sorted(combined.items(), key=lambda x: -x[1])

    return ranked


def cascade_predict_single(df, current_idx, prev_row, markov, method='greedy'):
    """Generate a single candidate set using cascade approach"""
    prev_l_values = [prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                     prev_row['L_4'], prev_row['L_5']]

    adj_window = CONFIG['adjacency_window']
    edge_boost = CONFIG['edge_boost']
    middle_boost = CONFIG['middle_boost']
    markov_weight = CONFIG['markov_weight']

    predicted_set = []

    for position in range(1, 6):
        pos_scores = position_specific_scores(df, current_idx, position)
        prev_val = prev_l_values[position - 1]

        # Apply adjacency boost
        for offset in range(-adj_window, adj_window + 1):
            candidate = prev_val + offset
            if 1 <= candidate <= 39 and candidate in pos_scores:
                boost = edge_boost if position in [1, 5] else middle_boost
                pos_scores[candidate] *= boost

        # Combine with Markov
        combined = {}
        for part in range(1, 40):
            pos_score = pos_scores.get(part, 0)
            markov_score = markov.get(part, 0)
            combined[part] = (1 - markov_weight) * pos_score + markov_weight * markov_score

        # Filter to valid candidates (must be > all previously selected)
        if predicted_set:
            min_valid = max(predicted_set) + 1
        else:
            min_valid = 1

        valid_scores = {k: v for k, v in combined.items() if k >= min_valid}

        if not valid_scores:
            if predicted_set:
                remaining = [p for p in range(1, 40) if p > max(predicted_set)]
            else:
                remaining = list(range(1, 40))
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

    return tuple(predicted_set)


def generate_candidate_sets(df, current_idx, prev_row, markov):
    """Generate portfolio of candidate sets"""
    n_sets = CONFIG['portfolio_size']
    sets = []
    scores = []

    # Greedy set first
    greedy_set = cascade_predict_single(df, current_idx, prev_row, markov, method='greedy')
    sets.append(greedy_set)

    # Use current_idx as seed for reproducibility
    np.random.seed(current_idx)

    attempts = 0
    max_attempts = n_sets * 5

    while len(sets) < n_sets and attempts < max_attempts:
        attempts += 1
        candidate = cascade_predict_single(df, current_idx, prev_row, markov, method='stochastic')
        if candidate not in sets and len(candidate) == 5:
            sets.append(candidate)

    # Fill with random if needed
    while len(sets) < n_sets:
        random_set = tuple(sorted(np.random.choice(range(1, 40), size=5, replace=False)))
        if random_set not in sets:
            sets.append(random_set)

    # Score each set (sum of part scores from pool)
    pool_scores = dict(generate_pool_ranking(df, current_idx, markov))
    for s in sets:
        score = sum(pool_scores.get(p, 0) for p in s)
        scores.append(score)

    # Rank by score
    ranked = sorted(zip(sets, scores), key=lambda x: -x[1])

    return ranked


def predict_for_date(df, target_date, verbose=True):
    """
    Generate predictions for a specific date using only prior data.

    Returns:
        pool_ranking: List of (part_id, score) tuples
        candidate_sets: List of (set_tuple, score) tuples
        metadata: Dict with model info
    """
    # Find index for target date
    target_idx = df[df['date'] == target_date].index
    if len(target_idx) == 0:
        raise ValueError(f"Date {target_date} not found in data")

    target_idx = target_idx[0]

    if target_idx == 0:
        raise ValueError("Cannot predict for first date in dataset (no prior data)")

    # Previous day's data (what we can see)
    prev_row = df.iloc[target_idx - 1]
    prev_date = prev_row['date']

    if verbose:
        print(f"Predicting for: {target_date.strftime('%Y-%m-%d')}")
        print(f"Using data up to: {prev_date.strftime('%Y-%m-%d')}")
        print(f"Training samples: {target_idx}")

    # Build Markov matrix from all data before target
    repeat_prob, co_occurrence, overall_prob = build_markov_matrix(df, target_idx)

    # Get today's parts for Markov context
    today_parts = get_parts_set(prev_row)
    markov = markov_scores(today_parts, repeat_prob, co_occurrence, overall_prob)

    # Generate pool ranking
    pool_ranking = generate_pool_ranking(df, target_idx, markov)

    # Generate candidate sets
    candidate_sets = generate_candidate_sets(df, target_idx, prev_row, markov)

    # Metadata (convert numpy types to native Python)
    metadata = {
        'target_date': target_date.strftime('%Y-%m-%d'),
        'data_through': prev_date.strftime('%Y-%m-%d'),
        'training_samples': int(target_idx),
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': CONFIG,
        'previous_day_parts': [int(p) for p in today_parts],
    }

    return pool_ranking, candidate_sets, metadata


def save_predictions(target_date, pool_ranking, candidate_sets, metadata):
    """Save predictions to CSV files"""
    PREDICTIONS_DIR.mkdir(exist_ok=True)

    date_str = target_date.strftime('%Y-%m-%d')

    # Save pool ranking
    pool_df = pd.DataFrame([
        {'rank': i+1, 'part_id': part, 'score': score}
        for i, (part, score) in enumerate(pool_ranking)
    ])
    pool_path = PREDICTIONS_DIR / f"{date_str}_pool.csv"
    pool_df.to_csv(pool_path, index=False)

    # Save candidate sets
    sets_data = []
    for i, (s, score) in enumerate(candidate_sets):
        sets_data.append({
            'rank': i+1,
            'L_1': s[0],
            'L_2': s[1],
            'L_3': s[2],
            'L_4': s[3],
            'L_5': s[4],
            'score': score
        })
    sets_df = pd.DataFrame(sets_data)
    sets_path = PREDICTIONS_DIR / f"{date_str}_sets.csv"
    sets_df.to_csv(sets_path, index=False)

    # Save metadata
    meta_path = PREDICTIONS_DIR / f"{date_str}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return pool_path, sets_path, meta_path


def run_backtest(df, n_days):
    """Generate predictions for last n_days for holdout evaluation"""
    results = []

    for i in range(n_days, 0, -1):
        target_idx = len(df) - i
        target_date = df.iloc[target_idx]['date']

        try:
            pool_ranking, candidate_sets, metadata = predict_for_date(df, target_date, verbose=False)
            pool_path, sets_path, meta_path = save_predictions(target_date, pool_ranking, candidate_sets, metadata)
            results.append({
                'date': target_date,
                'pool_path': pool_path,
                'sets_path': sets_path,
                'status': 'success'
            })
            print(f"Generated: {target_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            results.append({
                'date': target_date,
                'status': 'error',
                'error': str(e)
            })
            print(f"Error for {target_date.strftime('%Y-%m-%d')}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for C5 Parts Forecasting')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD). Default: tomorrow')
    parser.add_argument('--backtest', type=int, help='Generate predictions for last N days')
    args = parser.parse_args()

    print("=" * 70)
    print("C5 PARTS FORECASTING - PREDICTION GENERATOR")
    print("=" * 70)
    print(f"Model version: {CONFIG['model_version']}")
    print(f"Portfolio size: {CONFIG['portfolio_size']}")
    print(f"Adjacency window: +/-{CONFIG['adjacency_window']}")

    # Load data
    df = load_data()
    print(f"\nData loaded: {len(df)} days")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    if args.backtest:
        # Backtest mode
        print(f"\n--- BACKTEST MODE: Last {args.backtest} days ---")
        results = run_backtest(df, args.backtest)
        success = sum(1 for r in results if r['status'] == 'success')
        print(f"\nGenerated {success}/{len(results)} predictions")
        print(f"Output directory: {PREDICTIONS_DIR}")

    else:
        # Single date mode
        if args.date:
            target_date = pd.to_datetime(args.date)
        else:
            # Default to last date in data (for testing)
            target_date = df['date'].max()

        print(f"\n--- SINGLE DATE MODE ---")

        pool_ranking, candidate_sets, metadata = predict_for_date(df, target_date)
        pool_path, sets_path, meta_path = save_predictions(target_date, pool_ranking, candidate_sets, metadata)

        # Display results
        print(f"\n--- POOL RANKING (Top {CONFIG['pool_size']}) ---")
        print(f"{'Rank':>4} {'Part':>6} {'Score':>10}")
        print("-" * 24)
        for i, (part, score) in enumerate(pool_ranking[:CONFIG['pool_size']]):
            print(f"{i+1:>4} P_{part:>3} {score:>10.4f}")

        print(f"\n--- TOP 10 CANDIDATE SETS ---")
        print(f"{'Rank':>4} {'Set':>25} {'Score':>10}")
        print("-" * 42)
        for i, (s, score) in enumerate(candidate_sets[:10]):
            set_str = f"[{s[0]:>2}, {s[1]:>2}, {s[2]:>2}, {s[3]:>2}, {s[4]:>2}]"
            print(f"{i+1:>4} {set_str:>25} {score:>10.4f}")

        print(f"\n--- OUTPUT FILES ---")
        print(f"Pool:     {pool_path}")
        print(f"Sets:     {sets_path}")
        print(f"Metadata: {meta_path}")

    return 0


if __name__ == "__main__":
    exit(main())
