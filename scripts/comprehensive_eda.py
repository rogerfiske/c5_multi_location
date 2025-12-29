"""
Comprehensive EDA for C5 Multi-Location Parts Forecasting
Primary focus: CA5_matrix.csv and CA5_aggregated_matrix.csv

Outputs:
- Data profiling summary
- Part frequency analysis
- Location (L_1 to L_5) distribution profiles
- Percentile boundaries for each location
- Co-occurrence patterns
- Temporal patterns
- Feature candidates for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load primary data files"""
    ca_matrix = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    ca_agg = pd.read_csv(DATA_DIR / "CA5_aggregated_matrix.csv")

    # Parse dates
    ca_matrix['date'] = pd.to_datetime(ca_matrix['date'], format='%m/%d/%Y')
    ca_agg['date'] = pd.to_datetime(ca_agg['date'], format='%m/%d/%Y')

    return ca_matrix, ca_agg

def data_profiling(df, name):
    """Generate data profiling summary"""
    print(f"\n{'='*60}")
    print(f"DATA PROFILE: {name}")
    print(f"{'='*60}")

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Date span: {(df['date'].max() - df['date'].min()).days} days")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Check for duplicate dates
    dupes = df['date'].duplicated().sum()
    print(f"Duplicate dates: {dupes}")

    # Check ascending constraint
    l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']
    violations = 0
    for idx, row in df[l_cols].iterrows():
        if list(row) != sorted(row):
            violations += 1
    print(f"Ascending constraint violations: {violations}")

    return {
        'rows': df.shape[0],
        'cols': df.shape[1],
        'date_min': df['date'].min(),
        'date_max': df['date'].max(),
        'missing': df.isnull().sum().sum(),
        'ascending_violations': violations
    }

def part_frequency_analysis(df, name):
    """Analyze part frequency distribution"""
    print(f"\n{'='*60}")
    print(f"PART FREQUENCY ANALYSIS: {name}")
    print(f"{'='*60}")

    p_cols = [f'P_{i}' for i in range(1, 40)]

    # Sum occurrences of each part
    part_sums = df[p_cols].sum()
    total_events = len(df)

    print(f"\nTotal events (days): {total_events}")
    print(f"Parts per day: 5 (fixed)")
    print(f"Total part occurrences: {part_sums.sum()}")

    # Frequency table
    freq_df = pd.DataFrame({
        'Part': range(1, 40),
        'Count': part_sums.values,
        'Frequency': (part_sums.values / total_events * 100).round(2)
    }).sort_values('Count', ascending=False)

    print(f"\n--- Top 10 Most Frequent Parts ---")
    for _, row in freq_df.head(10).iterrows():
        print(f"  Part {int(row['Part']):2d}: {int(row['Count']):4d} occurrences ({row['Frequency']:.1f}%)")

    print(f"\n--- Bottom 10 Least Frequent Parts ---")
    for _, row in freq_df.tail(10).iterrows():
        print(f"  Part {int(row['Part']):2d}: {int(row['Count']):4d} occurrences ({row['Frequency']:.1f}%)")

    # Frequency distribution stats
    print(f"\n--- Frequency Statistics ---")
    print(f"  Mean frequency: {part_sums.mean():.1f} ({part_sums.mean()/total_events*100:.2f}%)")
    print(f"  Std deviation: {part_sums.std():.1f}")
    print(f"  Min: {part_sums.min()} (Part {part_sums.idxmin().replace('P_', '')})")
    print(f"  Max: {part_sums.max()} (Part {part_sums.idxmax().replace('P_', '')})")

    return freq_df

def location_distribution_analysis(df, name):
    """Analyze L_1 to L_5 distributions individually"""
    print(f"\n{'='*60}")
    print(f"LOCATION DISTRIBUTION ANALYSIS: {name}")
    print(f"{'='*60}")

    l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

    results = {}
    for col in l_cols:
        data = df[col]

        print(f"\n--- {col} Distribution ---")
        print(f"  Range: [{data.min()}, {data.max()}]")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Median: {data.median():.1f}")
        print(f"  Std: {data.std():.2f}")
        print(f"  Mode: {data.mode().values[0]}")

        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        pct_values = np.percentile(data, percentiles)
        print(f"  Percentiles:")
        for p, v in zip(percentiles, pct_values):
            print(f"    {p:3d}th: {v:.0f}")

        # Value counts for top values
        vc = data.value_counts().sort_index()
        print(f"  Top 5 values: {dict(data.value_counts().head(5))}")

        results[col] = {
            'min': data.min(),
            'max': data.max(),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'mode': data.mode().values[0],
            'percentiles': dict(zip(percentiles, pct_values))
        }

    return results

def percentile_boundary_analysis(df, name):
    """Optimize percentile boundaries for minimizing false negatives"""
    print(f"\n{'='*60}")
    print(f"PERCENTILE BOUNDARY OPTIMIZATION: {name}")
    print(f"{'='*60}")
    print("\nGoal: Find percentile thresholds that capture most values (minimize false negatives)")

    l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

    results = {}
    for col in l_cols:
        data = df[col]

        print(f"\n--- {col} ---")

        # Test different percentile thresholds
        for pct in [85, 90, 92, 95, 98]:
            threshold = np.percentile(data, pct)
            captured = (data <= threshold).sum()
            capture_rate = captured / len(data) * 100
            print(f"  {pct}th percentile ({threshold:.0f}): captures {capture_rate:.1f}% of values")

        # Find optimal threshold for 90%, 92%, 95% capture
        for target_capture in [90, 92, 95]:
            optimal_threshold = np.percentile(data, target_capture)
            print(f"  -> To capture {target_capture}% of {col}, use threshold <= {optimal_threshold:.0f}")

        results[col] = {
            f'p{pct}': np.percentile(data, pct) for pct in [85, 90, 92, 95, 98]
        }

    return results

def location_gap_analysis(df, name):
    """Analyze gaps between consecutive L values"""
    print(f"\n{'='*60}")
    print(f"LOCATION GAP ANALYSIS: {name}")
    print(f"{'='*60}")
    print("\nAnalyzing gaps between consecutive locations (useful for cascade filtering)")

    l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

    gaps = {}
    for i in range(len(l_cols) - 1):
        gap_name = f"Gap_{l_cols[i]}_{l_cols[i+1]}"
        gap_values = df[l_cols[i+1]] - df[l_cols[i]]

        print(f"\n--- {gap_name} (L_{i+2} - L_{i+1}) ---")
        print(f"  Range: [{gap_values.min()}, {gap_values.max()}]")
        print(f"  Mean: {gap_values.mean():.2f}")
        print(f"  Median: {gap_values.median():.1f}")
        print(f"  Mode: {gap_values.mode().values[0]}")

        # Minimum gap is always 1 (ascending constraint)
        pct_gap_1 = (gap_values == 1).sum() / len(gap_values) * 100
        print(f"  Gap = 1 (consecutive): {pct_gap_1:.1f}%")

        gaps[gap_name] = {
            'min': gap_values.min(),
            'max': gap_values.max(),
            'mean': gap_values.mean(),
            'median': gap_values.median(),
            'pct_gap_1': pct_gap_1
        }

    return gaps

def aggregated_value_analysis(df, name):
    """Analyze aggregated P values (counts across states)"""
    print(f"\n{'='*60}")
    print(f"AGGREGATED VALUE ANALYSIS: {name}")
    print(f"{'='*60}")

    p_cols = [f'P_{i}' for i in range(1, 40)]

    # Distribution of aggregated values
    all_p_values = df[p_cols].values.flatten()
    value_counts = pd.Series(all_p_values).value_counts().sort_index()

    print(f"\n--- Distribution of Aggregated P Values ---")
    total = len(all_p_values)
    for val, count in value_counts.items():
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {val}: {count:6d} ({pct:5.2f}%) {bar}")

    print(f"\n--- Per-Part Aggregation Statistics ---")
    part_stats = []
    for col in p_cols:
        part_num = int(col.replace('P_', ''))
        mean_agg = df[col].mean()
        max_agg = df[col].max()
        pct_zero = (df[col] == 0).sum() / len(df) * 100
        part_stats.append({
            'Part': part_num,
            'Mean_Agg': mean_agg,
            'Max_Agg': max_agg,
            'Pct_Zero': pct_zero
        })

    stats_df = pd.DataFrame(part_stats).sort_values('Mean_Agg', ascending=False)

    print(f"\nTop 10 by Mean Aggregation:")
    for _, row in stats_df.head(10).iterrows():
        print(f"  Part {int(row['Part']):2d}: mean={row['Mean_Agg']:.3f}, max={int(row['Max_Agg'])}, zero%={row['Pct_Zero']:.1f}%")

    return stats_df

def cooccurrence_analysis(df, name, top_n=20):
    """Analyze part co-occurrence patterns"""
    print(f"\n{'='*60}")
    print(f"CO-OCCURRENCE ANALYSIS: {name}")
    print(f"{'='*60}")

    l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

    # Count pair co-occurrences
    pair_counts = Counter()
    for idx, row in df[l_cols].iterrows():
        parts = sorted(row.values)
        for i in range(len(parts)):
            for j in range(i+1, len(parts)):
                pair_counts[(parts[i], parts[j])] += 1

    print(f"\n--- Top {top_n} Most Common Part Pairs ---")
    for (p1, p2), count in pair_counts.most_common(top_n):
        freq = count / len(df) * 100
        print(f"  ({p1:2d}, {p2:2d}): {count:4d} ({freq:.2f}%)")

    # Analyze position-specific co-occurrence
    print(f"\n--- Position Correlation (L_1 vs L_2, etc.) ---")
    for i in range(len(l_cols) - 1):
        col1, col2 = l_cols[i], l_cols[i+1]
        corr = df[col1].corr(df[col2])
        print(f"  Corr({col1}, {col2}): {corr:.4f}")

    return pair_counts

def temporal_analysis(df, name):
    """Analyze temporal patterns"""
    print(f"\n{'='*60}")
    print(f"TEMPORAL ANALYSIS: {name}")
    print(f"{'='*60}")

    df = df.copy()
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear

    l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

    # Day of week analysis
    print(f"\n--- Day of Week Distribution ---")
    dow_counts = df['dayofweek'].value_counts().sort_index()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for dow, count in dow_counts.items():
        print(f"  {dow_names[dow]}: {count:4d} events ({count/len(df)*100:.1f}%)")

    # Monthly seasonality
    print(f"\n--- Monthly Distribution ---")
    monthly = df.groupby('month').size()
    for month, count in monthly.items():
        print(f"  Month {month:2d}: {count:4d} events")

    # Year-over-year trend
    print(f"\n--- Yearly Event Counts ---")
    yearly = df.groupby('year').size()
    for year, count in yearly.items():
        print(f"  {year}: {count:4d} events")

    # L value trends by year
    print(f"\n--- L_1 Mean by Year (first 5, last 5) ---")
    yearly_l1 = df.groupby('year')['L_1'].mean()
    years = sorted(yearly_l1.index)
    print("  First 5 years:")
    for y in years[:5]:
        print(f"    {y}: {yearly_l1[y]:.2f}")
    print("  Last 5 years:")
    for y in years[-5:]:
        print(f"    {y}: {yearly_l1[y]:.2f}")

    return {
        'dow_counts': dow_counts,
        'monthly': monthly,
        'yearly': yearly
    }

def feature_candidates(df, name):
    """Generate feature candidate recommendations"""
    print(f"\n{'='*60}")
    print(f"FEATURE CANDIDATES FOR MODELING: {name}")
    print(f"{'='*60}")

    recommendations = []

    print("\n--- Lag Features ---")
    print("  - Previous day's L values (L_1_lag1, ..., L_5_lag1)")
    print("  - Previous day's P binary vector (P_1_lag1, ..., P_39_lag1)")
    print("  - Previous 3-day rolling presence (P_i_rolling3)")
    print("  - Previous 7-day rolling presence (P_i_rolling7)")
    recommendations.extend(['lag1', 'rolling3', 'rolling7'])

    print("\n--- Temporal Features ---")
    print("  - Day of week (one-hot encoded)")
    print("  - Month (one-hot or cyclical encoding)")
    print("  - Day of year (cyclical encoding)")
    print("  - Year (for trend capture)")
    recommendations.extend(['dow', 'month', 'doy', 'year'])

    print("\n--- Part Frequency Features ---")
    print("  - Historical part frequency (static)")
    print("  - Recent part frequency (last 30 days)")
    print("  - Part frequency momentum (last 7 vs last 30)")
    recommendations.extend(['freq_static', 'freq_recent', 'freq_momentum'])

    print("\n--- Location-Specific Features ---")
    print("  - L_1 historical range (min, max, percentiles)")
    print("  - L_1 recent trend (increasing/decreasing)")
    print("  - Gap patterns (L_2-L_1, L_3-L_2, etc.)")
    recommendations.extend(['l_range', 'l_trend', 'gaps'])

    print("\n--- Cross-State Features (for aggregated data) ---")
    print("  - Aggregated P value from previous day")
    print("  - Cross-state consensus (# states with part)")
    print("  - Cross-state trend (increasing/decreasing consensus)")
    recommendations.extend(['agg_lag', 'consensus', 'consensus_trend'])

    print("\n--- Cascade Features ---")
    print("  - Given L_1, conditional distribution of L_2")
    print("  - Given L_1, L_2, conditional distribution of L_3")
    print("  - Pool filtering based on ascending constraint")
    recommendations.extend(['cascade_conditional', 'pool_filter'])

    print("\n--- Imputation Features (for quantum approach) ---")
    print("  - Aggregated P values as continuous (0-5 scale)")
    print("  - Normalized aggregation (0-1 probability proxy)")
    print("  - Weighted aggregation by state correlation")
    recommendations.extend(['agg_continuous', 'agg_normalized', 'agg_weighted'])

    return recommendations

def main():
    """Run comprehensive EDA"""
    print("=" * 70)
    print("C5 MULTI-LOCATION PARTS FORECASTING - COMPREHENSIVE EDA")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    ca_matrix, ca_agg = load_data()

    # Data profiling
    profile_matrix = data_profiling(ca_matrix, "CA5_matrix.csv")
    profile_agg = data_profiling(ca_agg, "CA5_aggregated_matrix.csv")

    # Part frequency (binary for matrix, aggregated for agg)
    freq_matrix = part_frequency_analysis(ca_matrix, "CA5_matrix.csv (Binary)")

    # Aggregated value analysis (only for aggregated file)
    agg_stats = aggregated_value_analysis(ca_agg, "CA5_aggregated_matrix.csv")

    # Location distribution (same L values in both files)
    loc_dist = location_distribution_analysis(ca_matrix, "CA5_matrix.csv")

    # Percentile boundary optimization
    pct_bounds = percentile_boundary_analysis(ca_matrix, "CA5_matrix.csv")

    # Gap analysis
    gaps = location_gap_analysis(ca_matrix, "CA5_matrix.csv")

    # Co-occurrence analysis
    cooccur = cooccurrence_analysis(ca_matrix, "CA5_matrix.csv")

    # Temporal analysis
    temporal = temporal_analysis(ca_matrix, "CA5_matrix.csv")

    # Feature candidates
    features = feature_candidates(ca_matrix, "CA5_matrix.csv")

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    # Summary
    print("\n--- KEY FINDINGS SUMMARY ---")
    print(f"\n1. Data spans {profile_matrix['date_min'].strftime('%Y-%m-%d')} to {profile_matrix['date_max'].strftime('%Y-%m-%d')}")
    print(f"   Total events: {profile_matrix['rows']}")

    print(f"\n2. Location Ranges (CA5_matrix):")
    for col in ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']:
        print(f"   {col}: [{loc_dist[col]['min']}, {loc_dist[col]['max']}], mean={loc_dist[col]['mean']:.1f}")

    print(f"\n3. Recommended Percentile Thresholds (92% capture):")
    for col in ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']:
        print(f"   {col}: <= {pct_bounds[col]['p92']:.0f}")

    print(f"\n4. Aggregation Value Distribution (CA5_aggregated_matrix):")
    print(f"   Most P values are 0-2 (cross-state consensus indicator)")

    print(f"\n5. Feature Categories Recommended:")
    print(f"   - Lag features (previous day, rolling windows)")
    print(f"   - Temporal features (DOW, month, year)")
    print(f"   - Frequency features (static, recent, momentum)")
    print(f"   - Location-specific (ranges, trends, gaps)")
    print(f"   - Cross-state (aggregation, consensus)")
    print(f"   - Cascade (conditional distributions)")

if __name__ == "__main__":
    main()
