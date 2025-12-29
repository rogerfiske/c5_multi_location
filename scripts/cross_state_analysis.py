"""
Cross-State Correlation Analysis for C5 Forecasting
Analyze correlations between CA and other states to identify predictive relationships
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

def load_state_data(state_code):
    """Load matrix data for a state"""
    filepath = DATA_DIR / f"{state_code}5_matrix.csv"
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    return df

def main():
    print("="*70)
    print("CROSS-STATE CORRELATION ANALYSIS")
    print("="*70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load CA (target state)
    ca = load_state_data('CA')
    print(f"\nTarget: CA5_matrix.csv")
    print(f"  Date range: {ca['date'].min().strftime('%Y-%m-%d')} to {ca['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Records: {len(ca)}")

    # Load other states
    state_codes = ['MD', 'ME', 'MI', 'MO', 'NY', 'OH']
    states = {}

    print(f"\n--- State Files Overview ---")
    for code in state_codes:
        df = load_state_data(code)
        if df is not None:
            states[code] = df
            print(f"{code}5: {len(df)} records, {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        else:
            print(f"{code}5: NOT FOUND")

    # Align all states to CA date range
    print(f"\n--- Aligning States to CA Date Range ---")
    ca_dates = set(ca['date'])

    aligned = {'CA': ca.set_index('date')}
    for code, df in states.items():
        common_dates = ca_dates.intersection(set(df['date']))
        print(f"{code}: {len(common_dates)} common dates with CA ({len(common_dates)/len(ca)*100:.1f}%)")
        aligned[code] = df.set_index('date')

    # P-value correlation analysis (binary part presence)
    print(f"\n{'='*70}")
    print("PART-LEVEL CORRELATION ANALYSIS (CA vs Other States)")
    print("="*70)

    p_cols = [f'P_{i}' for i in range(1, 40)]

    # For each state, compute correlation of each P column with CA
    state_correlations = {}
    for code in states.keys():
        if code == 'ME':
            continue  # Skip ME (shorter history)

        # Get common dates
        common_dates = aligned['CA'].index.intersection(aligned[code].index)

        if len(common_dates) < 100:
            print(f"\n{code}: Insufficient common dates ({len(common_dates)}), skipping")
            continue

        ca_p = aligned['CA'].loc[common_dates, p_cols]
        state_p = aligned[code].loc[common_dates, p_cols]

        # Compute correlation for each part
        correlations = []
        for p_col in p_cols:
            corr = ca_p[p_col].corr(state_p[p_col])
            correlations.append(corr)

        avg_corr = np.nanmean(correlations)
        max_corr = np.nanmax(correlations)
        min_corr = np.nanmin(correlations)

        state_correlations[code] = {
            'avg': avg_corr,
            'max': max_corr,
            'min': min_corr,
            'correlations': correlations,
            'common_dates': len(common_dates)
        }

        print(f"\n--- CA vs {code} ({len(common_dates)} common dates) ---")
        print(f"  Average correlation: {avg_corr:.4f}")
        print(f"  Max correlation: {max_corr:.4f}")
        print(f"  Min correlation: {min_corr:.4f}")

        # Top correlated parts
        sorted_parts = sorted(zip(range(1, 40), correlations), key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
        print(f"  Top 5 correlated parts:")
        for part, corr in sorted_parts[:5]:
            print(f"    Part {part:2d}: {corr:.4f}")

    # Rank states by predictive potential
    print(f"\n{'='*70}")
    print("STATE RANKING BY PREDICTIVE POTENTIAL")
    print("="*70)

    ranked = sorted(state_correlations.items(), key=lambda x: x[1]['avg'], reverse=True)
    print("\nRanked by average part correlation with CA:")
    for rank, (code, stats) in enumerate(ranked, 1):
        print(f"  {rank}. {code}: avg={stats['avg']:.4f} (range: {stats['min']:.4f} to {stats['max']:.4f})")

    # L-value correlation analysis
    print(f"\n{'='*70}")
    print("LOCATION-LEVEL CORRELATION ANALYSIS (L_1 to L_5)")
    print("="*70)

    l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

    for code in states.keys():
        if code == 'ME':
            continue

        common_dates = aligned['CA'].index.intersection(aligned[code].index)
        if len(common_dates) < 100:
            continue

        ca_l = aligned['CA'].loc[common_dates, l_cols]
        state_l = aligned[code].loc[common_dates, l_cols]

        print(f"\n--- CA vs {code} L-value correlations ---")
        for l_col in l_cols:
            corr = ca_l[l_col].corr(state_l[l_col])
            print(f"  {l_col}: {corr:.4f}")

    # Same-day vs lag correlation
    print(f"\n{'='*70}")
    print("LAG CORRELATION ANALYSIS (Does yesterday's state X predict today's CA?)")
    print("="*70)

    for code in ['MD', 'MI', 'MO', 'NY', 'OH']:
        if code not in states:
            continue

        common_dates = aligned['CA'].index.intersection(aligned[code].index)
        if len(common_dates) < 100:
            continue

        # Create lagged version
        state_df = aligned[code].loc[common_dates, p_cols].copy()
        state_df_lagged = state_df.shift(1)  # Yesterday's state values

        ca_df = aligned['CA'].loc[common_dates, p_cols]

        # Drop first row (NaN from shift)
        valid_dates = state_df_lagged.dropna().index
        ca_valid = ca_df.loc[valid_dates]
        state_lagged_valid = state_df_lagged.loc[valid_dates]

        # Compute lag-1 correlations
        lag_correlations = []
        for p_col in p_cols:
            corr = ca_valid[p_col].corr(state_lagged_valid[p_col])
            lag_correlations.append(corr)

        avg_lag_corr = np.nanmean(lag_correlations)

        print(f"\n{code} (t-1) -> CA (t):")
        print(f"  Average lag-1 correlation: {avg_lag_corr:.4f}")

        # Compare to same-day correlation
        same_day_corr = state_correlations.get(code, {}).get('avg', np.nan)
        print(f"  Same-day correlation: {same_day_corr:.4f}")
        print(f"  Lag correlation as % of same-day: {avg_lag_corr/same_day_corr*100:.1f}%" if same_day_corr else "  N/A")

    # Cross-state consensus analysis
    print(f"\n{'='*70}")
    print("CROSS-STATE CONSENSUS ANALYSIS")
    print("="*70)
    print("How often do multiple states agree on a part being present?")

    # Load aggregated data
    ca_agg = pd.read_csv(DATA_DIR / "CA5_aggregated_matrix.csv")
    ca_agg['date'] = pd.to_datetime(ca_agg['date'], format='%m/%d/%Y')

    print(f"\nAggregated P-value distribution (0 = no states, 5 = all 5 states):")
    all_p = ca_agg[p_cols].values.flatten()
    for val in range(6):
        count = (all_p == val).sum()
        pct = count / len(all_p) * 100
        print(f"  {val} states agree: {count:6d} ({pct:.2f}%)")

    # Predictive value of consensus
    print(f"\nPredictive value of cross-state consensus:")
    print("When aggregated P value is X, what % of time is CA's P value = 1?")

    for val in range(6):
        mask = (ca_agg[p_cols] == val).values
        ca_binary = (ca_agg[p_cols] > 0).values  # CA's actual binary presence
        if mask.sum() > 0:
            ca_present_rate = ca_binary[mask].mean() * 100
            print(f"  Agg={val}: CA present {ca_present_rate:.1f}% of time")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: KEY CROSS-STATE INSIGHTS")
    print("="*70)

    print("""
1. CORRELATION RANKING
   - States ranked by average part-level correlation with CA
   - Higher correlation = better same-day predictive alignment

2. LAG CORRELATIONS
   - Tests if yesterday's state data predicts today's CA
   - Lag correlations typically lower than same-day
   - Even small positive lag correlation is useful for forecasting

3. CONSENSUS AS SIGNAL
   - Aggregated P values (0-5) indicate cross-state agreement
   - Higher consensus = higher confidence in part prediction
   - Can be used as feature weight or threshold

4. RECOMMENDED FEATURES
   - Use aggregated P values as continuous predictors
   - Include lag-1 aggregated values
   - Weight predictions by cross-state consensus
   - Consider state-specific lags for high-correlation states
""")

if __name__ == "__main__":
    main()
