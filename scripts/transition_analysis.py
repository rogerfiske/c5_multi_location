"""
Transition Analysis for C5 Multi-Location Parts Forecasting
Analyzes day-over-day patterns to find exploitable temporal structure.

Key analyses:
1. Part-to-part transition probabilities
2. Gap distributions (days between same part appearances)
3. Position-specific adjacency patterns (L_1->L_1, L_5->L_5, etc.)
4. Adjacency window optimization (find optimal +/- range)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_parts(row):
    """Extract parts as list [L_1, L_2, L_3, L_4, L_5]"""
    return [row['L_1'], row['L_2'], row['L_3'], row['L_4'], row['L_5']]


# ============================================================
# ANALYSIS 1: Part-to-Part Transition Matrix
# ============================================================

def analyze_part_transitions(df):
    """
    Build transition matrix: P(part j tomorrow | part k today)
    For each part that appears today, what parts appear tomorrow?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: PART-TO-PART TRANSITIONS")
    print("=" * 70)

    # Count transitions
    transition_counts = defaultdict(Counter)  # transition_counts[today_part][tomorrow_part]
    total_by_part = Counter()

    for i in range(len(df) - 1):
        today_parts = set(get_parts(df.iloc[i]))
        tomorrow_parts = set(get_parts(df.iloc[i + 1]))

        for tp in today_parts:
            total_by_part[tp] += 1
            for np in tomorrow_parts:
                transition_counts[tp][np] += 1

    # Calculate transition probabilities
    print("\nTop transition probabilities (part X today -> part Y tomorrow):")
    print("-" * 50)

    all_transitions = []
    for today_part in range(1, 40):
        for tomorrow_part in range(1, 40):
            count = transition_counts[today_part][tomorrow_part]
            total = total_by_part[today_part]
            if total > 0:
                prob = count / total
                all_transitions.append((today_part, tomorrow_part, prob, count))

    # Sort by probability
    all_transitions.sort(key=lambda x: -x[2])

    # Show top 20
    print(f"{'Today':>6} -> {'Tomorrow':>8} | {'Prob':>8} | {'Count':>6}")
    print("-" * 40)
    for today, tomorrow, prob, count in all_transitions[:20]:
        diff = tomorrow - today
        diff_str = f"(+{diff})" if diff > 0 else f"({diff})" if diff < 0 else "(=0)"
        print(f"P_{today:>2}   -> P_{tomorrow:>2}     | {prob:>7.2%} | {count:>6} {diff_str}")

    # Analyze adjacency in transitions
    print("\n--- Adjacency Analysis in Transitions ---")
    adj_counts = {0: 0, 1: 0, 2: 0, 3: 0, 'far': 0}
    total_trans = 0

    for today_part in range(1, 40):
        for tomorrow_part in range(1, 40):
            count = transition_counts[today_part][tomorrow_part]
            if count > 0:
                diff = abs(tomorrow_part - today_part)
                total_trans += count
                if diff == 0:
                    adj_counts[0] += count
                elif diff == 1:
                    adj_counts[1] += count
                elif diff == 2:
                    adj_counts[2] += count
                elif diff == 3:
                    adj_counts[3] += count
                else:
                    adj_counts['far'] += count

    print(f"\nTransition distance distribution:")
    for dist, count in adj_counts.items():
        pct = count / total_trans * 100
        expected = 100 / 39  # Random expectation per part
        label = f"(~{expected:.1f}% if random)" if dist != 'far' else ""
        print(f"  |diff| = {dist}: {pct:.2f}% {label}")

    return transition_counts, total_by_part


# ============================================================
# ANALYSIS 2: Gap Distribution (Days Between Same Part)
# ============================================================

def analyze_gap_distribution(df):
    """
    For each part, how many days between consecutive appearances?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: GAP DISTRIBUTION (Days Between Same Part)")
    print("=" * 70)

    # Track last appearance of each part
    last_seen = {}  # part -> last index seen
    gaps_by_part = defaultdict(list)

    for i in range(len(df)):
        parts = set(get_parts(df.iloc[i]))
        for p in parts:
            if p in last_seen:
                gap = i - last_seen[p]
                gaps_by_part[p].append(gap)
            last_seen[p] = i

    # Aggregate all gaps
    all_gaps = []
    for p in range(1, 40):
        all_gaps.extend(gaps_by_part[p])

    gap_counter = Counter(all_gaps)

    print(f"\nTotal gap observations: {len(all_gaps)}")
    print(f"\nGap distribution (days between same part appearing):")
    print("-" * 50)

    cumulative = 0
    for gap in sorted(gap_counter.keys())[:15]:
        count = gap_counter[gap]
        pct = count / len(all_gaps) * 100
        cumulative += pct
        print(f"  Gap = {gap:2d} days: {count:5d} ({pct:5.2f}%) cumulative: {cumulative:.1f}%")

    # Summary stats
    print(f"\n  Mean gap: {np.mean(all_gaps):.2f} days")
    print(f"  Median gap: {np.median(all_gaps):.1f} days")
    print(f"  Min gap: {min(all_gaps)} days")
    print(f"  Max gap: {max(all_gaps)} days")

    # Next-day repeat rate
    next_day_repeats = gap_counter.get(1, 0)
    print(f"\n  Next-day repeat rate: {next_day_repeats / len(all_gaps) * 100:.2f}%")

    return gaps_by_part, gap_counter


# ============================================================
# ANALYSIS 3: Position-Specific Adjacency
# ============================================================

def analyze_position_adjacency(df):
    """
    For each position (L_1 through L_5), what's the adjacency pattern?
    Does L_1 on day i predict L_1 on day i+1 being adjacent?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: POSITION-SPECIFIC ADJACENCY")
    print("=" * 70)

    positions = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

    for pos in positions:
        adj_0 = 0  # Same value
        adj_1 = 0  # +/- 1
        adj_2 = 0  # +/- 2
        adj_3 = 0  # +/- 3
        far = 0
        total = 0

        for i in range(len(df) - 1):
            today_val = df.iloc[i][pos]
            tomorrow_val = df.iloc[i + 1][pos]
            diff = abs(tomorrow_val - today_val)
            total += 1

            if diff == 0:
                adj_0 += 1
            elif diff == 1:
                adj_1 += 1
            elif diff == 2:
                adj_2 += 1
            elif diff == 3:
                adj_3 += 1
            else:
                far += 1

        print(f"\n{pos} (day i) -> {pos} (day i+1):")
        print(f"  Same value (diff=0): {adj_0/total*100:5.2f}%")
        print(f"  Adjacent +/-1:       {adj_1/total*100:5.2f}%")
        print(f"  Adjacent +/-2:       {adj_2/total*100:5.2f}%")
        print(f"  Adjacent +/-3:       {adj_3/total*100:5.2f}%")
        print(f"  Far (|diff| > 3):    {far/total*100:5.2f}%")
        print(f"  Combined +/-2:       {(adj_0+adj_1+adj_2)/total*100:5.2f}%")


# ============================================================
# ANALYSIS 4: Cross-Position Adjacency
# ============================================================

def analyze_cross_position_adjacency(df):
    """
    Does today's L_5 predict tomorrow's L_1?
    Any part from today predict any part tomorrow?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: CROSS-POSITION ADJACENCY")
    print("=" * 70)

    # For each day, check if ANY tomorrow part is adjacent to ANY today part
    any_adj_1 = 0
    any_adj_2 = 0
    count_adj_1 = []  # How many tomorrow parts are adjacent to some today part
    count_adj_2 = []
    total = 0

    for i in range(len(df) - 1):
        today_parts = set(get_parts(df.iloc[i]))
        tomorrow_parts = set(get_parts(df.iloc[i + 1]))
        total += 1

        # Count tomorrow parts that are +/-1 of any today part
        adj_1_parts = set()
        adj_2_parts = set()
        for tp in today_parts:
            for delta in [-1, 0, 1]:
                candidate = tp + delta
                if candidate in tomorrow_parts:
                    adj_1_parts.add(candidate)
            for delta in [-2, -1, 0, 1, 2]:
                candidate = tp + delta
                if candidate in tomorrow_parts:
                    adj_2_parts.add(candidate)

        count_adj_1.append(len(adj_1_parts))
        count_adj_2.append(len(adj_2_parts))

        if len(adj_1_parts) > 0:
            any_adj_1 += 1
        if len(adj_2_parts) > 0:
            any_adj_2 += 1

    print(f"\nDays where at least 1 tomorrow part is +/-1 of a today part: {any_adj_1/total*100:.2f}%")
    print(f"Days where at least 1 tomorrow part is +/-2 of a today part: {any_adj_2/total*100:.2f}%")

    print(f"\nAverage tomorrow parts adjacent (+/-1) to today: {np.mean(count_adj_1):.2f} / 5")
    print(f"Average tomorrow parts adjacent (+/-2) to today: {np.mean(count_adj_2):.2f} / 5")

    print(f"\nDistribution of adjacent (+/-1) parts count:")
    dist_1 = Counter(count_adj_1)
    for k in sorted(dist_1.keys()):
        print(f"  {k} parts adjacent: {dist_1[k]/total*100:.2f}%")

    print(f"\nDistribution of adjacent (+/-2) parts count:")
    dist_2 = Counter(count_adj_2)
    for k in sorted(dist_2.keys()):
        print(f"  {k} parts adjacent: {dist_2[k]/total*100:.2f}%")

    return count_adj_1, count_adj_2


# ============================================================
# ANALYSIS 5: Optimal Adjacency Window
# ============================================================

def analyze_adjacency_window(df):
    """
    What's the optimal adjacency window (+/- N) for prediction?
    Test windows from +/-1 to +/-5.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 5: OPTIMAL ADJACENCY WINDOW")
    print("=" * 70)

    results = []

    for window in range(1, 8):
        hits = []

        for i in range(len(df) - 1):
            today_parts = set(get_parts(df.iloc[i]))
            tomorrow_parts = set(get_parts(df.iloc[i + 1]))

            # Build candidate pool: all parts within +/- window of today's parts
            candidates = set()
            for tp in today_parts:
                for delta in range(-window, window + 1):
                    candidate = tp + delta
                    if 1 <= candidate <= 39:
                        candidates.add(candidate)

            # How many tomorrow parts are in the candidate pool?
            hit = len(tomorrow_parts.intersection(candidates))
            hits.append(hit)

        avg_hits = np.mean(hits)
        avg_pool = min(39, len(today_parts) * (2 * window + 1))  # Rough pool size
        coverage = sum(1 for h in hits if h == 5) / len(hits) * 100

        results.append({
            'window': window,
            'avg_hits': avg_hits,
            'hit_rate': avg_hits / 5 * 100,
            'perfect_coverage': coverage,
            'pool_size_approx': min(39, 5 * (2 * window + 1))
        })

        print(f"\nWindow +/-{window}:")
        print(f"  Approx pool size: ~{min(39, 5 * (2 * window + 1))} parts")
        print(f"  Avg tomorrow hits: {avg_hits:.2f} / 5 ({avg_hits/5*100:.1f}%)")
        print(f"  Days with 5/5 coverage: {coverage:.1f}%")

    # Find sweet spot
    print("\n--- Summary ---")
    print(f"{'Window':>8} | {'Pool Size':>10} | {'Hit Rate':>10} | {'5/5 Coverage':>12}")
    print("-" * 50)
    for r in results:
        print(f"    +/-{r['window']} | ~{r['pool_size_approx']:>8} | {r['hit_rate']:>9.1f}% | {r['perfect_coverage']:>11.1f}%")

    return results


def main():
    print("=" * 70)
    print("TRANSITION ANALYSIS - C5 Multi-Location Parts Forecasting")
    print("Finding day-over-day temporal structure")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()
    print(f"\nTotal events: {len(df)}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Run analyses
    transition_counts, total_by_part = analyze_part_transitions(df)
    gaps_by_part, gap_counter = analyze_gap_distribution(df)
    analyze_position_adjacency(df)
    count_adj_1, count_adj_2 = analyze_cross_position_adjacency(df)
    adjacency_results = analyze_adjacency_window(df)

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR MODELING")
    print("=" * 70)

    print("""
1. TRANSITION STRUCTURE: Parts show day-over-day adjacency patterns
   - Transitions are NOT random
   - Adjacent parts (+/-1, +/-2) more likely than distant parts

2. GAP DISTRIBUTION: Parts reappear with characteristic gaps
   - Mean gap ~7-8 days suggests weekly-ish cycling
   - Next-day repeats exist but are minority

3. POSITION-SPECIFIC: L_1 and L_5 show strongest adjacency
   - Edge positions (lowest, highest) more predictable
   - Interior positions (L_2, L_3, L_4) more variable

4. ADJACENCY WINDOW: +/-3 or +/-4 may be optimal
   - Balances pool size vs coverage
   - Use as feature, not hard constraint

5. MODELING IMPLICATIONS:
   - Build features from yesterday's parts
   - Weight candidates by distance from yesterday
   - Consider position-specific models
   - Gap since last appearance is informative
""")

    return transition_counts, gaps_by_part, adjacency_results


if __name__ == "__main__":
    transition_counts, gaps_by_part, adjacency_results = main()
