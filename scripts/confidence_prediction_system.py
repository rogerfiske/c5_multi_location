"""
Confidence-Aware Prediction System

Uses dual portfolios (adjacency + anti-adjacency) to:
1. Generate predictions
2. Calculate confidence via portfolio agreement (pool overlap)
3. Flag high/low confidence predictions
4. Provide actionable output with confidence indicators

Output Format:
- predictions/{date}_confidence.json
- predictions/{date}_pool.csv (updated with confidence)
- predictions/{date}_sets.csv (updated with confidence)
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"

CONFIG = {
    'portfolio_a_size': 100,
    'portfolio_b_size': 100,
    'adjacency_window': 3,
    'rolling_window': 30,
    'high_confidence_threshold': 22,  # Pool overlap >= this = high confidence (top quartile)
    'medium_confidence_threshold': 18,  # Pool overlap >= this = medium confidence
    'model_version': '2.0.0-confidence',
}


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def position_scores(df, current_idx, position, bias='adjacency'):
    """Get position-specific scores with adjacency or anti-adjacency bias"""
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
    """Generate a single candidate set"""
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
    """Generate portfolio of sets with specified bias"""
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
    """Calculate overlap between top parts in each portfolio"""
    pool_a = set()
    pool_b = set()

    for s in portfolio_a[:top_n]:
        if s:
            pool_a.update(s)
    for s in portfolio_b[:top_n]:
        if s:
            pool_b.update(s)

    return len(pool_a & pool_b)


def generate_pool_ranking(portfolio_a, portfolio_b):
    """Generate combined pool ranking from both portfolios"""
    part_scores = defaultdict(float)

    # Score from Portfolio A
    for rank, s in enumerate(portfolio_a):
        if s:
            for part in s:
                part_scores[part] += 1.0 / (rank + 1)

    # Score from Portfolio B
    for rank, s in enumerate(portfolio_b):
        if s:
            for part in s:
                part_scores[part] += 1.0 / (rank + 1)

    # Normalize
    total = sum(part_scores.values())
    if total > 0:
        part_scores = {k: v/total for k, v in part_scores.items()}

    # Rank
    ranked = sorted(part_scores.items(), key=lambda x: -x[1])
    return ranked


def score_sets(combined_portfolio, pool_ranking):
    """Score each set based on pool ranking"""
    pool_scores = dict(pool_ranking)
    scored_sets = []

    seen = set()
    for s in combined_portfolio:
        if s and s not in seen:
            seen.add(s)
            score = sum(pool_scores.get(p, 0) for p in s)
            scored_sets.append((s, score))

    # Sort by score descending
    scored_sets.sort(key=lambda x: -x[1])
    return scored_sets


def evaluate_portfolio_best(portfolio, actual_parts):
    """Find best wrong count in portfolio"""
    best_wrong = 5
    for pred_set in portfolio:
        if pred_set:
            wrong = len(set(pred_set) - actual_parts)
            if wrong < best_wrong:
                best_wrong = wrong
    return best_wrong


def predict_for_date(df, target_date, verbose=True):
    """
    Generate confidence-aware predictions for a specific date.
    """
    target_idx = df[df['date'] == target_date].index
    if len(target_idx) == 0:
        raise ValueError(f"Date {target_date} not found in data")

    target_idx = target_idx[0]

    if target_idx < CONFIG['rolling_window']:
        raise ValueError("Not enough historical data for this date")

    prev_row = df.iloc[target_idx - 1]
    prev_date = prev_row['date']

    if verbose:
        print(f"Predicting for: {target_date.strftime('%Y-%m-%d')}")
        print(f"Using data through: {prev_date.strftime('%Y-%m-%d')}")

    # Generate both portfolios
    portfolio_a = generate_portfolio(df, target_idx, bias='adjacency')
    portfolio_b = generate_portfolio(df, target_idx, bias='anti')

    # Calculate confidence metrics
    pool_overlap = calculate_pool_overlap(portfolio_a, portfolio_b)

    # Determine confidence level
    if pool_overlap >= CONFIG['high_confidence_threshold']:
        confidence_level = 'HIGH'
        confidence_note = 'Both strategies strongly agree - higher confidence signal'
    elif pool_overlap >= CONFIG['medium_confidence_threshold']:
        confidence_level = 'MEDIUM'
        confidence_note = 'Moderate agreement - typical prediction'
    else:
        confidence_level = 'LOW'
        confidence_note = 'Strategies diverge - weaker signal, consider caution'

    # Generate combined pool ranking
    pool_ranking = generate_pool_ranking(portfolio_a, portfolio_b)

    # Combine and score sets
    combined_portfolio = list(set(portfolio_a + portfolio_b))
    scored_sets = score_sets(combined_portfolio, pool_ranking)

    # Previous day parts
    prev_parts = [int(prev_row['L_1']), int(prev_row['L_2']), int(prev_row['L_3']),
                  int(prev_row['L_4']), int(prev_row['L_5'])]

    # Metadata
    metadata = {
        'target_date': target_date.strftime('%Y-%m-%d'),
        'data_through': prev_date.strftime('%Y-%m-%d'),
        'training_samples': int(target_idx),
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': CONFIG['model_version'],
        'config': {k: v for k, v in CONFIG.items()},
        'previous_day_parts': prev_parts,
        'confidence': {
            'level': confidence_level,
            'pool_overlap': int(pool_overlap),
            'threshold': int(CONFIG['high_confidence_threshold']),
            'note': confidence_note,
        },
        'portfolio_sizes': {
            'adjacency': len(portfolio_a),
            'anti_adjacency': len(portfolio_b),
            'combined_unique': len(combined_portfolio),
        }
    }

    return pool_ranking, scored_sets, metadata


def save_predictions(target_date, pool_ranking, scored_sets, metadata):
    """Save predictions with confidence indicators"""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    date_str = target_date.strftime('%Y-%m-%d')

    # Save pool ranking
    pool_df = pd.DataFrame([
        {'rank': i+1, 'part_id': int(part), 'score': float(score)}
        for i, (part, score) in enumerate(pool_ranking)
    ])
    pool_path = PREDICTIONS_DIR / f"{date_str}_pool.csv"
    pool_df.to_csv(pool_path, index=False)

    # Save sets
    sets_data = []
    for i, (s, score) in enumerate(scored_sets[:200]):  # Top 200
        sets_data.append({
            'rank': i+1,
            'L_1': int(s[0]),
            'L_2': int(s[1]),
            'L_3': int(s[2]),
            'L_4': int(s[3]),
            'L_5': int(s[4]),
            'score': float(score)
        })
    sets_df = pd.DataFrame(sets_data)
    sets_path = PREDICTIONS_DIR / f"{date_str}_sets.csv"
    sets_df.to_csv(sets_path, index=False)

    # Save confidence metadata
    meta_path = PREDICTIONS_DIR / f"{date_str}_confidence.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return pool_path, sets_path, meta_path


def run_backtest(df, n_days=365):
    """Run backtest and evaluate confidence-based strategy"""
    results = []

    for i in range(n_days, 0, -1):
        target_idx = len(df) - i
        if target_idx < CONFIG['rolling_window']:
            continue

        target_row = df.iloc[target_idx]
        target_date = target_row['date']
        actual_set = (target_row['L_1'], target_row['L_2'], target_row['L_3'],
                     target_row['L_4'], target_row['L_5'])
        actual_parts = set(actual_set)

        try:
            pool_ranking, scored_sets, metadata = predict_for_date(df, target_date, verbose=False)

            # Evaluate BEST set in entire portfolio (not just top-scored)
            all_sets = [s for s, score in scored_sets]
            best_wrong = evaluate_portfolio_best(all_sets, actual_parts)

            # Also check top-scored set
            if scored_sets:
                top_scored_wrong = len(set(scored_sets[0][0]) - actual_parts)
            else:
                top_scored_wrong = 5

            # Evaluate pool (top 20)
            top_20_parts = set([p for p, s in pool_ranking[:20]])
            pool_wrong = 5 - len(actual_parts & top_20_parts)

            results.append({
                'date': target_date,
                'confidence_level': metadata['confidence']['level'],
                'pool_overlap': metadata['confidence']['pool_overlap'],
                'best_wrong': best_wrong,
                'top_scored_wrong': top_scored_wrong,
                'pool_wrong': pool_wrong,
            })

        except Exception as e:
            print(f"Error for {target_date}: {e}")

    return pd.DataFrame(results)


def analyze_backtest(results_df):
    """Analyze backtest results by confidence level"""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS BY CONFIDENCE LEVEL")
    print("=" * 70)

    for level in ['HIGH', 'MEDIUM', 'LOW']:
        subset = results_df[results_df['confidence_level'] == level]
        if len(subset) == 0:
            continue

        n = len(subset)
        pct = n / len(results_df) * 100
        avg_wrong = subset['best_wrong'].mean()
        excellent = (subset['best_wrong'] <= 1).sum() / n * 100
        good = (subset['best_wrong'] <= 2).sum() / n * 100

        print(f"\n{level} CONFIDENCE: {n} days ({pct:.1f}%)")
        print(f"  Avg Wrong:        {avg_wrong:.3f}")
        print(f"  Excellent (0-1):  {excellent:.1f}%")
        print(f"  Good (0-2):       {good:.1f}%")

        # Distribution
        counter = Counter(subset['best_wrong'])
        print(f"  Distribution:")
        for w in range(6):
            count = counter.get(w, 0)
            wpct = count / n * 100
            print(f"    {w} wrong: {count:>3} ({wpct:>5.1f}%)")

    # Overall
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    n = len(results_df)
    avg_wrong = results_df['best_wrong'].mean()
    excellent = (results_df['best_wrong'] <= 1).sum() / n * 100
    good = (results_df['best_wrong'] <= 2).sum() / n * 100

    print(f"Total days: {n}")
    print(f"Avg Wrong: {avg_wrong:.3f}")
    print(f"Excellent (0-1): {excellent:.1f}%")
    print(f"Good (0-2): {good:.1f}%")

    # Strategy comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON: HIGH-CONFIDENCE ONLY vs ALL")
    print("=" * 70)

    high_only = results_df[results_df['confidence_level'] == 'HIGH']
    if len(high_only) > 0:
        h_n = len(high_only)
        h_avg = high_only['best_wrong'].mean()
        h_excellent = (high_only['best_wrong'] <= 1).sum() / h_n * 100
        h_good = (high_only['best_wrong'] <= 2).sum() / h_n * 100

        print(f"{'Strategy':<25} {'Coverage':>10} {'Avg Wrong':>12} {'Excellent':>10} {'Good':>10}")
        print("-" * 70)
        print(f"{'All predictions':<25} {'100%':>10} {avg_wrong:>12.3f} {excellent:>9.1f}% {good:>9.1f}%")
        print(f"{'HIGH confidence only':<25} {h_n/n*100:>9.1f}% {h_avg:>12.3f} {h_excellent:>9.1f}% {h_good:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Confidence-Aware Prediction System')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--backtest', type=int, help='Run backtest for N days')
    args = parser.parse_args()

    print("=" * 70)
    print("CONFIDENCE-AWARE PREDICTION SYSTEM v2.0")
    print("=" * 70)
    print(f"Model version: {CONFIG['model_version']}")
    print(f"High confidence threshold: pool_overlap >= {CONFIG['high_confidence_threshold']}")

    df = load_data()
    print(f"Data loaded: {len(df)} days")

    if args.backtest:
        print(f"\n--- BACKTEST MODE: {args.backtest} days ---")
        results_df = run_backtest(df, args.backtest)
        analyze_backtest(results_df)

    else:
        # Single date prediction
        if args.date:
            target_date = pd.to_datetime(args.date)
        else:
            target_date = df['date'].max()

        print(f"\n--- SINGLE DATE PREDICTION ---")
        pool_ranking, scored_sets, metadata = predict_for_date(df, target_date)
        pool_path, sets_path, meta_path = save_predictions(target_date, pool_ranking, scored_sets, metadata)

        # Display results
        conf = metadata['confidence']
        print(f"\n{'='*60}")
        print(f"CONFIDENCE: {conf['level']}")
        print(f"Pool Overlap: {conf['pool_overlap']} (threshold: {conf['threshold']})")
        print(f"Note: {conf['note']}")
        print(f"{'='*60}")

        print(f"\n--- TOP 20 POOL ---")
        print(f"{'Rank':>4} {'Part':>6} {'Score':>10}")
        print("-" * 24)
        for i, (part, score) in enumerate(pool_ranking[:20]):
            print(f"{i+1:>4} P_{part:>3} {score:>10.4f}")

        print(f"\n--- TOP 10 SETS ---")
        print(f"{'Rank':>4} {'Set':>25} {'Score':>10}")
        print("-" * 42)
        for i, (s, score) in enumerate(scored_sets[:10]):
            set_str = f"[{s[0]:>2}, {s[1]:>2}, {s[2]:>2}, {s[3]:>2}, {s[4]:>2}]"
            print(f"{i+1:>4} {set_str:>25} {score:>10.4f}")

        print(f"\n--- OUTPUT FILES ---")
        print(f"Pool:       {pool_path}")
        print(f"Sets:       {sets_path}")
        print(f"Confidence: {meta_path}")


if __name__ == "__main__":
    main()
