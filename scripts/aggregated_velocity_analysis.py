"""
Aggregated Matrix Velocity & Interaction Analysis

Tests whether CHANGES in P_values (velocity) predict CA5's future parts,
and whether CA5's behavior depends on consensus patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_aggregated_data():
    """Load aggregated matrix"""
    df = pd.read_csv(DATA_DIR / "CA5_aggregated_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def analyze_velocity_signal(df):
    """
    Analyze if CHANGES in P_values predict CA5's future parts.

    Velocity = P_today - P_yesterday
    - Rising parts (velocity > 0): Getting more popular
    - Falling parts (velocity < 0): Getting less popular
    """
    print("\n" + "=" * 70)
    print("VELOCITY SIGNAL ANALYSIS")
    print("=" * 70)

    results_by_lag = []

    for lag in range(1, 6):
        rising_hits = 0
        rising_total = 0
        falling_hits = 0
        falling_total = 0
        stable_hits = 0
        stable_total = 0

        for i in range(1, len(df) - lag):
            yesterday = df.iloc[i - 1]
            today = df.iloc[i]
            future = df.iloc[i + lag]

            future_parts = set([future['L_1'], future['L_2'], future['L_3'],
                              future['L_4'], future['L_5']])

            for part in range(1, 40):
                p_yesterday = yesterday[f'P_{part}']
                p_today = today[f'P_{part}']
                velocity = p_today - p_yesterday

                is_winner = part in future_parts

                if velocity > 0:  # Rising
                    rising_total += 1
                    if is_winner:
                        rising_hits += 1
                elif velocity < 0:  # Falling
                    falling_total += 1
                    if is_winner:
                        falling_hits += 1
                else:  # Stable
                    stable_total += 1
                    if is_winner:
                        stable_hits += 1

        rising_precision = rising_hits / rising_total if rising_total > 0 else 0
        falling_precision = falling_hits / falling_total if falling_total > 0 else 0
        stable_precision = stable_hits / stable_total if stable_total > 0 else 0

        results_by_lag.append({
            'lag': lag,
            'rising_precision': rising_precision,
            'falling_precision': falling_precision,
            'stable_precision': stable_precision,
            'rising_lift': rising_precision / (5/39),
            'falling_lift': falling_precision / (5/39),
            'stable_lift': stable_precision / (5/39),
        })

    print(f"\n{'Lag':>4} {'Rising':>12} {'Falling':>12} {'Stable':>12} {'Best Signal':>14}")
    print("-" * 58)

    for r in results_by_lag:
        lifts = [('Rising', r['rising_lift']), ('Falling', r['falling_lift']), ('Stable', r['stable_lift'])]
        best = max(lifts, key=lambda x: abs(x[1] - 1))
        print(f"{r['lag']:>4} {r['rising_lift']:>11.3f}x {r['falling_lift']:>11.3f}x "
              f"{r['stable_lift']:>11.3f}x {best[0]:>10} {best[1]:.3f}x")

    return results_by_lag


def analyze_ca5_consensus_interaction(df):
    """
    Analyze if CA5's adjacency behavior depends on consensus.

    Question: When CA5 picks a hot part today, is tomorrow's part more/less
    likely to be adjacent?
    """
    print("\n" + "=" * 70)
    print("CA5 + CONSENSUS INTERACTION ANALYSIS")
    print("=" * 70)

    # Track adjacency rates by today's consensus level
    adjacency_by_consensus = defaultdict(lambda: {'adjacent': 0, 'total': 0})

    for i in range(1, len(df) - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        today_parts = [today['L_1'], today['L_2'], today['L_3'], today['L_4'], today['L_5']]
        tomorrow_parts = [tomorrow['L_1'], tomorrow['L_2'], tomorrow['L_3'], tomorrow['L_4'], tomorrow['L_5']]

        # For each position
        for pos in range(5):
            today_val = today_parts[pos]
            tomorrow_val = tomorrow_parts[pos]
            today_p = today[f'P_{today_val}']

            # Categorize consensus
            if today_p >= 2:
                consensus = 'hot'
            elif today_p == 1:
                consensus = 'warm'
            else:
                consensus = 'cold'

            # Check if adjacent (within +/- 3)
            is_adjacent = abs(tomorrow_val - today_val) <= 3

            adjacency_by_consensus[consensus]['total'] += 1
            if is_adjacent:
                adjacency_by_consensus[consensus]['adjacent'] += 1

    print("\nAdjacency rate by CA5's part consensus level:")
    print(f"{'Consensus':>10} {'Adjacency Rate':>15} {'Sample Size':>12}")
    print("-" * 40)

    for consensus in ['hot', 'warm', 'cold']:
        data = adjacency_by_consensus[consensus]
        rate = data['adjacent'] / data['total'] if data['total'] > 0 else 0
        print(f"{consensus:>10} {rate:>14.2%} {data['total']:>12,}")

    return adjacency_by_consensus


def analyze_regime_detection(df):
    """
    Can we use P_value distribution to predict CA5's predictability?

    Hypothesis: On days with high consensus (many parts with P >= 2),
    CA5 might be more/less predictable.
    """
    print("\n" + "=" * 70)
    print("REGIME DETECTION ANALYSIS")
    print("=" * 70)

    # Calculate daily consensus spread
    regime_results = []

    for i in range(len(df) - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        # Count how many parts have P >= 2 today
        hot_count = sum(1 for j in range(1, 40) if today[f'P_{j}'] >= 2)

        # Count how many of tomorrow's parts we could predict from adjacency
        today_parts = [today['L_1'], today['L_2'], today['L_3'], today['L_4'], today['L_5']]
        tomorrow_parts = set([tomorrow['L_1'], tomorrow['L_2'], tomorrow['L_3'],
                             tomorrow['L_4'], tomorrow['L_5']])

        # Adjacent pool from today
        adjacent_pool = set()
        for p in today_parts:
            for offset in range(-3, 4):
                candidate = p + offset
                if 1 <= candidate <= 39:
                    adjacent_pool.add(candidate)

        adjacent_hits = len(tomorrow_parts & adjacent_pool)

        regime_results.append({
            'hot_count': hot_count,
            'adjacent_hits': adjacent_hits,
            'date': today['date']
        })

    # Group by hot_count ranges
    print("\nCA5 predictability by consensus concentration:")
    print(f"{'Hot Parts Today':>16} {'Avg Adjacent Hits':>18} {'Sample Size':>12}")
    print("-" * 50)

    df_regime = pd.DataFrame(regime_results)

    for low, high, label in [(0, 3, '0-3 (dispersed)'), (4, 7, '4-7 (moderate)'),
                              (8, 12, '8-12 (concentrated)'), (13, 39, '13+ (very hot)')]:
        subset = df_regime[(df_regime['hot_count'] >= low) & (df_regime['hot_count'] <= high)]
        if len(subset) > 0:
            avg_hits = subset['adjacent_hits'].mean()
            print(f"{label:>16} {avg_hits:>17.2f} {len(subset):>12,}")

    return regime_results


def analyze_exclusion_power(df):
    """
    Can we use high-P parts to EXCLUDE from consideration?

    If hot parts are less likely to appear in CA5, excluding them shrinks the pool.
    """
    print("\n" + "=" * 70)
    print("EXCLUSION POWER ANALYSIS")
    print("=" * 70)

    for lag in range(1, 4):
        for threshold in [2, 3]:
            excluded_wrong = 0
            excluded_right = 0

            for i in range(len(df) - lag):
                today = df.iloc[i]
                future = df.iloc[i + lag]

                future_parts = set([future['L_1'], future['L_2'], future['L_3'],
                                  future['L_4'], future['L_5']])

                # Parts to exclude (high consensus)
                exclude_set = set([j for j in range(1, 40) if today[f'P_{j}'] >= threshold])

                # How many excluded parts actually appeared?
                wrongly_excluded = len(exclude_set & future_parts)
                correctly_excluded = len(exclude_set - future_parts)

                excluded_wrong += wrongly_excluded
                excluded_right += correctly_excluded

            total_excluded = excluded_wrong + excluded_right
            accuracy = excluded_right / total_excluded if total_excluded > 0 else 0
            avg_excluded = total_excluded / (len(df) - lag)

            print(f"Lag={lag}, P>={threshold}: Exclude avg {avg_excluded:.1f} parts, "
                  f"accuracy {accuracy:.2%} (wrongly excluded {excluded_wrong/(len(df)-lag):.2f}/day)")


def main():
    print("=" * 70)
    print("AGGREGATED MATRIX - ADVANCED SIGNAL ANALYSIS")
    print("=" * 70)

    df = load_aggregated_data()
    print(f"Data loaded: {len(df)} days")

    # Run all analyses
    analyze_velocity_signal(df)
    analyze_ca5_consensus_interaction(df)
    analyze_regime_detection(df)
    analyze_exclusion_power(df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
