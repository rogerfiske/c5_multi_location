"""
Aggregated Matrix Signal Analysis

Analyzes whether multi-state consensus (P_values) predicts CA5's future parts.
Tests multiple lag values to find optimal signal delay.

Key Question: Do high P_values predict INCLUSION or EXCLUSION of parts?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_aggregated_data():
    """Load aggregated matrix with multi-state consensus"""
    df = pd.read_csv(DATA_DIR / "CA5_aggregated_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def analyze_lag(df, lag=1):
    """
    Analyze correlation between P_values at day T and CA5 parts at day T+lag

    Returns:
        avg_p_of_winners: Average P_value of parts that appear in CA5
        avg_p_of_losers: Average P_value of parts that DON'T appear in CA5
        signal_strength: Difference (positive = follow crowd, negative = fade crowd)
    """
    winner_p_values = []
    loser_p_values = []

    p_cols = [f'P_{i}' for i in range(1, 40)]

    for i in range(len(df) - lag):
        # Today's P_values (consensus signal)
        today = df.iloc[i]
        today_p = {j: today[f'P_{j}'] for j in range(1, 40)}

        # Future CA5 parts (what we're trying to predict)
        future = df.iloc[i + lag]
        future_parts = set([future['L_1'], future['L_2'], future['L_3'],
                          future['L_4'], future['L_5']])

        # Collect P_values for winners vs losers
        for part in range(1, 40):
            if part in future_parts:
                winner_p_values.append(today_p[part])
            else:
                loser_p_values.append(today_p[part])

    avg_winner = np.mean(winner_p_values)
    avg_loser = np.mean(loser_p_values)
    signal = avg_winner - avg_loser

    return {
        'lag': lag,
        'avg_p_winners': avg_winner,
        'avg_p_losers': avg_loser,
        'signal_strength': signal,
        'direction': 'FOLLOW' if signal > 0 else 'FADE',
        'n_samples': len(df) - lag
    }


def analyze_threshold_performance(df, lag=1):
    """
    Analyze how well different P_value thresholds predict inclusion/exclusion
    """
    results = []
    p_cols = [f'P_{i}' for i in range(1, 40)]

    for threshold in range(5):  # P >= 0, 1, 2, 3, 4
        hits = 0
        misses = 0
        total_predictions = 0

        for i in range(len(df) - lag):
            today = df.iloc[i]
            future = df.iloc[i + lag]
            future_parts = set([future['L_1'], future['L_2'], future['L_3'],
                              future['L_4'], future['L_5']])

            # Parts with P >= threshold today
            hot_parts = set([j for j in range(1, 40) if today[f'P_{j}'] >= threshold])

            if len(hot_parts) > 0:
                hits += len(hot_parts & future_parts)
                misses += len(hot_parts - future_parts)
                total_predictions += len(hot_parts)

        precision = hits / total_predictions if total_predictions > 0 else 0
        avg_hot_count = total_predictions / (len(df) - lag)

        results.append({
            'threshold': f'P >= {threshold}',
            'avg_hot_parts': avg_hot_count,
            'precision': precision,
            'expected_random': 5/39,  # ~12.8%
            'lift': precision / (5/39) if precision > 0 else 0
        })

    return results


def analyze_bimodality(df, lag=1):
    """
    Check if there are distinct regimes (days where follow works vs fade works)
    """
    daily_signals = []

    for i in range(len(df) - lag):
        today = df.iloc[i]
        future = df.iloc[i + lag]
        future_parts = set([future['L_1'], future['L_2'], future['L_3'],
                          future['L_4'], future['L_5']])

        # Calculate day's signal
        winner_p = [today[f'P_{p}'] for p in future_parts]
        all_p = [today[f'P_{j}'] for j in range(1, 40)]

        avg_winner = np.mean(winner_p)
        avg_all = np.mean(all_p)

        daily_signals.append({
            'date': future['date'],
            'avg_winner_p': avg_winner,
            'avg_all_p': avg_all,
            'signal': avg_winner - avg_all
        })

    signals = [d['signal'] for d in daily_signals]

    return {
        'mean_signal': np.mean(signals),
        'std_signal': np.std(signals),
        'pct_positive': sum(1 for s in signals if s > 0) / len(signals) * 100,
        'pct_negative': sum(1 for s in signals if s < 0) / len(signals) * 100,
        'min_signal': np.min(signals),
        'max_signal': np.max(signals),
        'quartiles': np.percentile(signals, [25, 50, 75])
    }


def main():
    print("=" * 70)
    print("AGGREGATED MATRIX SIGNAL ANALYSIS")
    print("=" * 70)

    df = load_aggregated_data()
    print(f"Data loaded: {len(df)} days")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Check P_value distribution
    p_cols = [f'P_{i}' for i in range(1, 40)]
    all_p = df[p_cols].values.flatten()
    print(f"\nP_value distribution:")
    for v in range(6):
        pct = (all_p == v).sum() / len(all_p) * 100
        print(f"  P={v}: {pct:.1f}%")

    # Analyze multiple lags
    print("\n" + "=" * 70)
    print("LAG ANALYSIS: Does consensus predict future CA5 parts?")
    print("=" * 70)
    print(f"{'Lag':>4} {'Avg P Winners':>14} {'Avg P Losers':>14} {'Signal':>10} {'Direction':>10}")
    print("-" * 56)

    best_lag = None
    best_signal = 0

    for lag in range(1, 8):
        result = analyze_lag(df, lag)
        print(f"{result['lag']:>4} {result['avg_p_winners']:>14.4f} {result['avg_p_losers']:>14.4f} "
              f"{result['signal_strength']:>+10.4f} {result['direction']:>10}")

        if abs(result['signal_strength']) > abs(best_signal):
            best_signal = result['signal_strength']
            best_lag = lag

    print(f"\nBest lag: {best_lag} (signal: {best_signal:+.4f})")

    # Threshold analysis for best lag
    print("\n" + "=" * 70)
    print(f"THRESHOLD ANALYSIS (Lag={best_lag})")
    print("=" * 70)
    print(f"{'Threshold':>10} {'Avg Hot Parts':>14} {'Precision':>12} {'Lift vs Random':>16}")
    print("-" * 56)

    threshold_results = analyze_threshold_performance(df, best_lag)
    for r in threshold_results:
        print(f"{r['threshold']:>10} {r['avg_hot_parts']:>14.1f} {r['precision']:>11.2%} {r['lift']:>15.2f}x")

    # Bimodality analysis
    print("\n" + "=" * 70)
    print(f"BIMODALITY ANALYSIS (Lag={best_lag})")
    print("=" * 70)

    bimodal = analyze_bimodality(df, best_lag)
    print(f"Mean daily signal: {bimodal['mean_signal']:+.4f}")
    print(f"Std daily signal:  {bimodal['std_signal']:.4f}")
    print(f"Days with FOLLOW signal (positive): {bimodal['pct_positive']:.1f}%")
    print(f"Days with FADE signal (negative):   {bimodal['pct_negative']:.1f}%")
    print(f"Signal range: [{bimodal['min_signal']:.3f}, {bimodal['max_signal']:.3f}]")
    print(f"Quartiles (25/50/75): {bimodal['quartiles']}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if abs(best_signal) < 0.05:
        print("WEAK SIGNAL: Multi-state consensus has minimal predictive power.")
        print("The aggregated P_values don't strongly predict CA5's future parts.")
    elif best_signal > 0:
        print(f"FOLLOW SIGNAL: Parts popular across states (high P) tend to appear in CA5.")
        print(f"Strategy: Include parts with high P_values in predictions.")
    else:
        print(f"FADE SIGNAL: Parts popular across states (high P) tend to NOT appear in CA5.")
        print(f"Strategy: EXCLUDE parts with high P_values from predictions.")

    print(f"\nSignal strength: {abs(best_signal):.4f}")
    if abs(best_signal) > 0.1:
        print("This is a STRONG signal - worth building a model around.")
    elif abs(best_signal) > 0.05:
        print("This is a MODERATE signal - may provide incremental improvement.")
    else:
        print("This is a WEAK signal - unlikely to help significantly.")


if __name__ == "__main__":
    main()
