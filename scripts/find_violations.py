"""Find and report ascending constraint violations in CA5_matrix.csv"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

print("Searching for ascending constraint violations...")
print("="*70)

violations = []
for idx, row in df.iterrows():
    l_values = [row[col] for col in l_cols]
    if l_values != sorted(l_values):
        violations.append({
            'row': idx + 2,  # +2 for header and 0-index
            'date': row['date'].strftime('%Y-%m-%d'),
            'L_values': l_values,
            'sorted': sorted(l_values)
        })
        print(f"Row {idx+2}: {row['date'].strftime('%Y-%m-%d')}")
        print(f"  Current: {l_values}")
        print(f"  Should be: {sorted(l_values)}")
        print()

print(f"Total violations found: {len(violations)}")

# Also check aggregated file
print("\n" + "="*70)
print("Checking CA5_aggregated_matrix.csv...")
print("="*70)

df_agg = pd.read_csv(DATA_DIR / "CA5_aggregated_matrix.csv")
df_agg['date'] = pd.to_datetime(df_agg['date'], format='%m/%d/%Y')

violations_agg = []
for idx, row in df_agg.iterrows():
    l_values = [row[col] for col in l_cols]
    if l_values != sorted(l_values):
        violations_agg.append({
            'row': idx + 2,
            'date': row['date'].strftime('%Y-%m-%d'),
            'L_values': l_values
        })
        print(f"Row {idx+2}: {row['date'].strftime('%Y-%m-%d')}")
        print(f"  Current: {l_values}")
        print(f"  Should be: {sorted(l_values)}")
        print()

print(f"Total violations in aggregated file: {len(violations_agg)}")
