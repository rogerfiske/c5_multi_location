"""
Aggregated Matrix Concentration & Polarization Analysis

Tests whether the DISTRIBUTION of P_values (concentrated vs dispersed)
predicts CA5's predictability or behavior.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_data():
    """Load both datasets and verify alignment"""
    agg_df = pd.read_csv(DATA_DIR / "CA5_aggregated_matrix.csv")
    agg_df['date'] = pd.to_datetime(agg_df['date'], format='%m/%d/%Y')
    agg_df = agg_df.sort_values('date').reset_index(drop=True)

    ca5_df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    ca5_df['date'] = pd.to_datetime(ca5_df['date'], format='%m/%d/%Y')
    ca5_df = ca5_df.sort_values('date').reset_index(drop=True)

    return agg_df, ca5_df


def verify_alignment(agg_df, ca5_df):
    """Verify L_1 to L_5 match between files"""
    print("=" * 70)
    print("DATA ALIGNMENT VERIFICATION")
    print("=" * 70)

    # Find common dates
    common_dates = set(agg_df['date']) & set(ca5_df['date'])
    print(f"Aggregated dates: {len(agg_df)}")
    print(f"CA5 matrix dates: {len(ca5_df)}")
    print(f"Common dates: {len(common_dates)}")

    # Check if L values match
    mismatches = 0
    for date in list(common_dates)[:100]:  # Check first 100
        agg_row = agg_df[agg_df['date'] == date].iloc[0]
        ca5_row = ca5_df[ca5_df['date'] == date].iloc[0]

        for col in ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']:
            if agg_row[col] != ca5_row[col]:
                mismatches += 1

    print(f"L-value mismatches in first 100 common dates: {mismatches}")

    if mismatches == 0:
        print("[OK] L_1 to L_5 values are identical between files")
    else:
        print("[WARNING] Data mismatch detected!")

    return common_dates


def calculate_concentration_metrics(row):
    """Calculate various concentration metrics for a day's P_values"""
    p_values = [row[f'P_{i}'] for i in range(1, 40)]

    total = sum(p_values)
    max_p = max(p_values)
    hot_count = sum(1 for p in p_values if p >= 2)
    cold_count = sum(1 for p in p_values if p == 0)

    # Entropy (higher = more dispersed)
    if total > 0:
        probs = [p / total for p in p_values if p > 0]
        entropy = -sum(p * np.log(p) for p in probs)
    else:
        entropy = 0

    # Gini coefficient (higher = more concentrated)
    sorted_p = sorted(p_values)
    n = len(sorted_p)
    cumsum = np.cumsum(sorted_p)
    gini = (2 * sum((i + 1) * sorted_p[i] for i in range(n)) - (n + 1) * total) / (n * total) if total > 0 else 0

    return {
        'total_p': total,
        'max_p': max_p,
        'hot_count': hot_count,
        'cold_count': cold_count,
        'entropy': entropy,
        'gini': gini
    }


def analyze_concentration_vs_predictability(agg_df):
    """
    Does concentration predict CA5's predictability?
    """
    print("\n" + "=" * 70)
    print("CONCENTRATION VS PREDICTABILITY")
    print("=" * 70)

    results = []

    for i in range(len(agg_df) - 1):
        today = agg_df.iloc[i]
        tomorrow = agg_df.iloc[i + 1]

        # Today's concentration
        metrics = calculate_concentration_metrics(today)

        # Tomorrow's predictability (how many parts repeat from today)
        today_parts = set([today['L_1'], today['L_2'], today['L_3'], today['L_4'], today['L_5']])
        tomorrow_parts = set([tomorrow['L_1'], tomorrow['L_2'], tomorrow['L_3'], tomorrow['L_4'], tomorrow['L_5']])

        repeats = len(today_parts & tomorrow_parts)

        # Adjacent hits
        adjacent_pool = set()
        for p in today_parts:
            for offset in range(-3, 4):
                candidate = p + offset
                if 1 <= candidate <= 39:
                    adjacent_pool.add(candidate)
        adjacent_hits = len(tomorrow_parts & adjacent_pool)

        # Did tomorrow include any hot parts?
        hot_parts_today = set([j for j in range(1, 40) if today[f'P_{j}'] >= 2])
        tomorrow_hot = len(tomorrow_parts & hot_parts_today)

        results.append({
            **metrics,
            'repeats': repeats,
            'adjacent_hits': adjacent_hits,
            'tomorrow_hot': tomorrow_hot
        })

    df_results = pd.DataFrame(results)

    # Analyze by concentration quartiles
    print("\nCA5 predictability by today's consensus concentration:")
    print(f"{'Metric':>20} {'Q1 (low)':>12} {'Q2':>12} {'Q3':>12} {'Q4 (high)':>12}")
    print("-" * 60)

    for metric in ['total_p', 'hot_count', 'entropy']:
        try:
            quartiles = pd.qcut(df_results[metric], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        except ValueError:
            # If quartiles can't be computed, use manual binning
            quartiles = pd.cut(df_results[metric], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        grouped = df_results.groupby(quartiles)['adjacent_hits'].mean()
        print(f"{metric:>20} {grouped['Q1']:>12.2f} {grouped['Q2']:>12.2f} "
              f"{grouped['Q3']:>12.2f} {grouped['Q4']:>12.2f}")

    # Check correlation
    print("\n\nCorrelations with tomorrow's adjacent_hits:")
    for metric in ['total_p', 'hot_count', 'entropy', 'gini', 'cold_count']:
        corr = df_results[metric].corr(df_results['adjacent_hits'])
        print(f"  {metric:>15}: {corr:+.4f}")

    return df_results


def analyze_hot_following(agg_df):
    """
    Detailed analysis: Does CA5 follow or fade hot parts?
    """
    print("\n" + "=" * 70)
    print("HOT PART FOLLOWING ANALYSIS")
    print("=" * 70)

    # Track daily results
    days_follow = 0
    days_fade = 0
    days_neutral = 0

    for i in range(len(agg_df) - 1):
        today = agg_df.iloc[i]
        tomorrow = agg_df.iloc[i + 1]

        tomorrow_parts = set([tomorrow['L_1'], tomorrow['L_2'], tomorrow['L_3'],
                             tomorrow['L_4'], tomorrow['L_5']])

        # Hot parts today (P >= 2)
        hot_parts = set([j for j in range(1, 40) if today[f'P_{j}'] >= 2])

        if len(hot_parts) == 0:
            continue

        # How many of tomorrow's parts are hot?
        hits = len(tomorrow_parts & hot_parts)
        expected = 5 * len(hot_parts) / 39

        if hits > expected + 0.5:
            days_follow += 1
        elif hits < expected - 0.5:
            days_fade += 1
        else:
            days_neutral += 1

    total = days_follow + days_fade + days_neutral
    print(f"Days CA5 FOLLOWS hot parts: {days_follow} ({days_follow/total*100:.1f}%)")
    print(f"Days CA5 FADES hot parts:   {days_fade} ({days_fade/total*100:.1f}%)")
    print(f"Days CA5 is NEUTRAL:        {days_neutral} ({days_neutral/total*100:.1f}%)")


def analyze_extreme_days(agg_df):
    """
    Focus on extreme consensus days - are they more predictable?
    """
    print("\n" + "=" * 70)
    print("EXTREME CONCENTRATION DAY ANALYSIS")
    print("=" * 70)

    results = []

    for i in range(len(agg_df) - 1):
        today = agg_df.iloc[i]
        tomorrow = agg_df.iloc[i + 1]

        metrics = calculate_concentration_metrics(today)
        today_parts = set([today['L_1'], today['L_2'], today['L_3'], today['L_4'], today['L_5']])
        tomorrow_parts = set([tomorrow['L_1'], tomorrow['L_2'], tomorrow['L_3'], tomorrow['L_4'], tomorrow['L_5']])

        # Adjacent hits as predictability measure
        adjacent_pool = set()
        for p in today_parts:
            for offset in range(-3, 4):
                candidate = p + offset
                if 1 <= candidate <= 39:
                    adjacent_pool.add(candidate)
        adjacent_hits = len(tomorrow_parts & adjacent_pool)

        results.append({
            'hot_count': metrics['hot_count'],
            'total_p': metrics['total_p'],
            'adjacent_hits': adjacent_hits
        })

    df = pd.DataFrame(results)

    # Extreme days (top/bottom 10% by hot_count)
    low_threshold = df['hot_count'].quantile(0.1)
    high_threshold = df['hot_count'].quantile(0.9)

    low_days = df[df['hot_count'] <= low_threshold]
    high_days = df[df['hot_count'] >= high_threshold]
    middle_days = df[(df['hot_count'] > low_threshold) & (df['hot_count'] < high_threshold)]

    print(f"\nPredictability by consensus extremity:")
    print(f"{'Day Type':>25} {'Count':>8} {'Avg Adjacent Hits':>18}")
    print("-" * 55)
    print(f"{'Low consensus (bottom 10%)':>25} {len(low_days):>8} {low_days['adjacent_hits'].mean():>18.2f}")
    print(f"{'Middle (10-90%)':>25} {len(middle_days):>8} {middle_days['adjacent_hits'].mean():>18.2f}")
    print(f"{'High consensus (top 10%)':>25} {len(high_days):>8} {high_days['adjacent_hits'].mean():>18.2f}")


def main():
    print("=" * 70)
    print("AGGREGATED MATRIX - CONCENTRATION ANALYSIS")
    print("=" * 70)

    agg_df, ca5_df = load_data()

    # Verify data alignment
    verify_alignment(agg_df, ca5_df)

    # Run analyses
    analyze_concentration_vs_predictability(agg_df)
    analyze_hot_following(agg_df)
    analyze_extreme_days(agg_df)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
