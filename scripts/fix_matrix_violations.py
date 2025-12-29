"""Fix ascending constraint violations in CA5_matrix.csv and CA5_aggregated_matrix.csv"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

l_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']

def fix_file(filename):
    filepath = DATA_DIR / filename
    print(f"\nFixing {filename}...")

    df = pd.read_csv(filepath)

    fixed_count = 0
    for idx, row in df.iterrows():
        l_values = [row[col] for col in l_cols]
        if l_values != sorted(l_values):
            sorted_values = sorted(l_values)
            print(f"  Row {idx+2}: {l_values} -> {sorted_values}")
            for i, col in enumerate(l_cols):
                df.at[idx, col] = sorted_values[i]
            fixed_count += 1

    # Save back
    df.to_csv(filepath, index=False)
    print(f"  Fixed {fixed_count} violations")
    print(f"  Saved to {filepath}")

# Fix both files
fix_file("CA5_matrix.csv")
fix_file("CA5_aggregated_matrix.csv")

print("\n" + "="*50)
print("VERIFICATION")
print("="*50)

# Verify fixes
for filename in ["CA5_matrix.csv", "CA5_aggregated_matrix.csv"]:
    df = pd.read_csv(DATA_DIR / filename)
    violations = 0
    for idx, row in df.iterrows():
        l_values = [row[col] for col in l_cols]
        if l_values != sorted(l_values):
            violations += 1
    print(f"{filename}: {violations} violations remaining")
