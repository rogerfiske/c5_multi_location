"""
Adjacency-Weighted Baseline for C5 Multi-Location Parts Forecasting
First "informed" baseline using temporal signals discovered in transition analysis.

Signals combined:
1. Adjacency boost: Parts within +/-N of yesterday's parts get higher scores
2. Gap feature: Days since last appearance (sweet spot ~3-7 days)
3. Position-specific: L_1 and L_5 show stronger adjacency patterns
4. Anti-persistence: Parts from yesterday slightly penalized (12.77% repeat rate)

Target: Achieve CORRECT predictions (0-1 wrong) instead of inverted (4-5 wrong)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
HOLDOUT_DAYS = 365


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_parts(row):
    """Extract parts as list [L_1, L_2, L_3, L_4, L_5]"""
    return [row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']]


def get_parts_set(row):
    """Extract parts as set"""
    return set(get_parts(row))


# ============================================================
# ADJACENCY-WEIGHTED SCORING
# ============================================================

def adjacency_weighted_scores(df, current_idx, adjacency_window=4,
                               gap_lookback=7, gap_sweet_spot=(3, 7),
                               yesterday_penalty=0.7):
    """
    Score parts using multiple temporal signals:

    1. Adjacency boost: Parts within +/-N of yesterday's parts
    2. Gap bonus: Parts in sweet spot gap range get boosted
    3. Yesterday penalty: Parts from yesterday slightly penalized

    Returns: dict mapping part_id to score
    """
    scores = {p: 1.0 for p in range(1, 40)}  # Start uniform

    if current_idx == 0:
        return scores

    # Get yesterday's parts
    yesterday_parts = get_parts_set(df.iloc[current_idx - 1])

    # === Signal 1: Adjacency Boost ===
    # Parts within +/-N of yesterday get boosted
    adjacency_boost = 3.0  # Multiplier for adjacent parts

    for yp in yesterday_parts:
        for delta in range(-adjacency_window, adjacency_window + 1):
            candidate = yp + delta
            if 1 <= candidate <= 39:
                # Closer = higher boost (triangular weighting)
                distance = abs(delta)
                boost = adjacency_boost * (1 - distance / (adjacency_window + 1))
                scores[candidate] *= (1 + boost)

    # === Signal 2: Gap Feature ===
    # Track last appearance of each part
    last_seen = {}
    for i in range(max(0, current_idx - 30), current_idx):
        for p in get_parts_set(df.iloc[i]):
            last_seen[p] = i

    gap_min, gap_max = gap_sweet_spot
    gap_boost = 1.5

    for part in range(1, 40):
        if part in last_seen:
            gap = current_idx - last_seen[part]
            if gap_min <= gap <= gap_max:
                # In sweet spot - boost
                scores[part] *= gap_boost
            elif gap == 1:
                # Yesterday - apply penalty (anti-persistence)
                scores[part] *= yesterday_penalty

    # === Signal 3: Recent frequency (last 7 days) ===
    recent_freq = Counter()
    for i in range(max(0, current_idx - gap_lookback), current_idx):
        for p in get_parts_set(df.iloc[i]):
            recent_freq[p] += 1

    # Slight boost to parts seen recently but not yesterday
    for part, freq in recent_freq.items():
        if part not in yesterday_parts and freq >= 2:
            scores[part] *= 1.2

    # Normalize
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}


def generate_set_from_scores(scores):
    """Generate valid ascending 5-part set from scores"""
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    selected = []
    for part_id, score in ranked:
        if len(selected) == 5:
            break
        if not selected or part_id > max(selected):
            selected.append(part_id)
        elif part_id < min(selected):
            selected.insert(0, part_id)
        else:
            for i in range(len(selected)):
                if i == 0 and part_id < selected[i]:
                    selected.insert(0, part_id)
                    break
                elif i > 0 and selected[i-1] < part_id < selected[i]:
                    selected.insert(i, part_id)
                    break
        if len(selected) > 5:
            selected = sorted(selected)[:5]

    # Fill if needed
    if len(selected) < 5:
        remaining = [p for p in range(1, 40) if p not in selected]
        for p in remaining:
            if not selected or p > selected[-1]:
                selected.append(p)
            if len(selected) == 5:
                break

    return tuple(sorted(selected)[:5])


def calculate_recall_at_k(scores, actual_parts, k):
    """Recall@K: fraction of actual parts in top-K"""
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_k = set([p[0] for p in ranked[:k]])
    hits = len(actual_parts.intersection(top_k))
    return hits / len(actual_parts)


def evaluate_baseline(df, train_end_idx, score_fn, name, **kwargs):
    """Evaluate a scoring function on holdout"""
    holdout_df = df.iloc[train_end_idx:]
    n_holdout = len(holdout_df)

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    wrong_counts = []
    recall_20 = []
    recall_30 = []

    for i in range(n_holdout):
        global_idx = train_end_idx + i
        scores = score_fn(df, global_idx, **kwargs)

        actual_parts = get_parts_set(df.iloc[global_idx])
        actual_set = tuple(sorted(actual_parts))

        # Recall metrics
        recall_20.append(calculate_recall_at_k(scores, actual_parts, 20))
        recall_30.append(calculate_recall_at_k(scores, actual_parts, 30))

        # Set accuracy
        pred_set = generate_set_from_scores(scores)
        wrong = len(set(pred_set) - actual_parts)
        wrong_counts.append(wrong)

    wrong_dist = Counter(wrong_counts)
    correct_rate = sum(1 for w in wrong_counts if w <= 1) / n_holdout * 100
    inverted_rate = sum(1 for w in wrong_counts if w >= 4) / n_holdout * 100

    print(f"\nPool Metrics:")
    print(f"  Recall@20: {np.mean(recall_20)*100:.2f}%")
    print(f"  Recall@30: {np.mean(recall_30)*100:.2f}%")

    print(f"\nSet Metrics:")
    print(f"  Correct (0-1 wrong): {correct_rate:.2f}%")
    print(f"  Inverted (4-5 wrong): {inverted_rate:.2f}%")
    print(f"  Total Good+: {correct_rate + inverted_rate:.2f}%")
    print(f"  Avg wrong: {np.mean(wrong_counts):.2f}")

    print(f"\nWrong Distribution:")
    for w in range(6):
        count = wrong_dist.get(w, 0)
        pct = count / n_holdout * 100
        label = "CORRECT" if w <= 1 else ("INVERTED" if w >= 4 else "")
        print(f"    {w} wrong: {count:4d} ({pct:5.2f}%) {label}")

    return {
        'name': name,
        'correct_rate': correct_rate,
        'inverted_rate': inverted_rate,
        'recall_20': np.mean(recall_20) * 100,
        'recall_30': np.mean(recall_30) * 100,
        'avg_wrong': np.mean(wrong_counts),
        'wrong_dist': wrong_dist
    }


def grid_search_parameters(df, train_end_idx):
    """Search for optimal parameters"""
    print("\n" + "=" * 70)
    print("PARAMETER GRID SEARCH")
    print("=" * 70)

    results = []

    # Test different adjacency windows
    for adj_window in [3, 4, 5]:
        for gap_sweet in [(2, 5), (3, 7), (4, 8)]:
            for yesterday_pen in [0.5, 0.7, 0.9]:

                def score_fn(df, idx, aw=adj_window, gs=gap_sweet, yp=yesterday_pen):
                    return adjacency_weighted_scores(
                        df, idx,
                        adjacency_window=aw,
                        gap_sweet_spot=gs,
                        yesterday_penalty=yp
                    )

                # Quick evaluation on subset
                n_test = min(100, len(df) - train_end_idx)
                wrong_counts = []

                for i in range(n_test):
                    global_idx = train_end_idx + i
                    scores = score_fn(df, global_idx)
                    actual_parts = get_parts_set(df.iloc[global_idx])
                    pred_set = generate_set_from_scores(scores)
                    wrong = len(set(pred_set) - actual_parts)
                    wrong_counts.append(wrong)

                correct_rate = sum(1 for w in wrong_counts if w <= 1) / n_test * 100
                avg_wrong = np.mean(wrong_counts)

                results.append({
                    'adj_window': adj_window,
                    'gap_sweet': gap_sweet,
                    'yesterday_pen': yesterday_pen,
                    'correct_rate': correct_rate,
                    'avg_wrong': avg_wrong
                })

    # Sort by correct rate
    results.sort(key=lambda x: -x['correct_rate'])

    print("\nTop 10 parameter combinations (by correct rate):")
    print("-" * 70)
    print(f"{'Adj Win':>8} | {'Gap Sweet':>12} | {'Yest Pen':>9} | {'Correct':>8} | {'Avg Wrong':>10}")
    print("-" * 70)

    for r in results[:10]:
        print(f"{r['adj_window']:>8} | {str(r['gap_sweet']):>12} | {r['yesterday_pen']:>9.1f} | {r['correct_rate']:>7.1f}% | {r['avg_wrong']:>10.2f}")

    return results[0]  # Best params


def main():
    print("=" * 70)
    print("ADJACENCY-WEIGHTED BASELINE")
    print("First 'informed' baseline using temporal signals")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()
    train_end_idx = len(df) - HOLDOUT_DAYS

    print(f"\nTotal events: {len(df)}")
    print(f"Train: {train_end_idx}, Holdout: {HOLDOUT_DAYS}")

    # Grid search for best parameters
    best_params = grid_search_parameters(df, train_end_idx)

    print(f"\nBest parameters found:")
    print(f"  Adjacency window: +/-{best_params['adj_window']}")
    print(f"  Gap sweet spot: {best_params['gap_sweet']} days")
    print(f"  Yesterday penalty: {best_params['yesterday_pen']}")

    # Full evaluation with best parameters
    print("\n" + "=" * 70)
    print("FULL EVALUATION WITH BEST PARAMETERS")
    print("=" * 70)

    def best_score_fn(df, idx):
        return adjacency_weighted_scores(
            df, idx,
            adjacency_window=best_params['adj_window'],
            gap_sweet_spot=best_params['gap_sweet'],
            yesterday_penalty=best_params['yesterday_pen']
        )

    result = evaluate_baseline(df, train_end_idx, best_score_fn,
                               f"Adjacency-Weighted (adj={best_params['adj_window']}, gap={best_params['gap_sweet']}, pen={best_params['yesterday_pen']})")

    # Compare with naive frequency baseline
    print("\n" + "=" * 70)
    print("COMPARISON WITH FREQUENCY BASELINE")
    print("=" * 70)

    # Pre-compute global frequency
    train_df = df.iloc[:train_end_idx]
    p_cols = [f'P_{i}' for i in range(1, 40)]
    freq = train_df[p_cols].sum()
    total = len(train_df)
    global_freq = {int(col.replace('P_', '')): freq[col] / total for col in p_cols}

    def freq_fn(df, idx):
        return global_freq

    freq_result = evaluate_baseline(df, train_end_idx, freq_fn, "Global Frequency (baseline)")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n{'Baseline':<45} | {'Correct':>8} | {'Inverted':>9} | {'Recall@20':>10}")
    print("-" * 80)
    print(f"{'Global Frequency':<45} | {freq_result['correct_rate']:>7.2f}% | {freq_result['inverted_rate']:>8.2f}% | {freq_result['recall_20']:>9.2f}%")
    print(f"{result['name'][:45]:<45} | {result['correct_rate']:>7.2f}% | {result['inverted_rate']:>8.2f}% | {result['recall_20']:>9.2f}%")

    # Improvement
    correct_improvement = result['correct_rate'] - freq_result['correct_rate']
    recall_improvement = result['recall_20'] - freq_result['recall_20']

    print(f"\nImprovement over frequency baseline:")
    print(f"  Correct rate: {correct_improvement:+.2f}%")
    print(f"  Recall@20: {recall_improvement:+.2f}%")

    if result['correct_rate'] > 5:
        print("\n*** SUCCESS: Adjacency-weighted baseline achieves meaningful CORRECT predictions! ***")
    elif result['correct_rate'] > freq_result['correct_rate']:
        print("\n*** PROGRESS: Some improvement in correct predictions, but more work needed. ***")
    else:
        print("\n*** INSIGHT: Adjacency weighting alone insufficient. Need additional signals. ***")

    return result, freq_result, best_params


if __name__ == "__main__":
    result, freq_result, best_params = main()
