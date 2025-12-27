import pandas as pd

# Base file (CA5) - will be the master
base_file = r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\CA5_matrix.csv'

# Files to aggregate onto the master
files_to_add = [
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\NY5_matrix.csv',
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\OH5_matrix.csv',
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\MD5_matrix.csv',
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\MI5_matrix.csv',
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\MO5_matrix.csv',
]

output_file = r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\CA5_agregated_matrix.csv'

part_cols = [f'P_{i}' for i in range(1, 40)]

# Read base file
print(f"Reading base file: CA5_matrix.csv")
master = pd.read_csv(base_file)
master['date'] = pd.to_datetime(master['date'])
print(f"  Base has {len(master)} rows")

# Create a dictionary to store aggregated P values by date
# First, build from master
date_data = {}
for idx, row in master.iterrows():
    d = row['date']
    if d not in date_data:
        date_data[d] = {p: 0 for p in part_cols}
        date_data[d]['L_1'] = row['L_1']
        date_data[d]['L_2'] = row['L_2']
        date_data[d]['L_3'] = row['L_3']
        date_data[d]['L_4'] = row['L_4']
        date_data[d]['L_5'] = row['L_5']
    for p_col in part_cols:
        date_data[d][p_col] += int(row[p_col])

# Aggregate each file
for filepath in files_to_add:
    filename = filepath.split('\\')[-1]
    print(f"\nAdding: {filename}")

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  File has {len(df)} rows")

    added_count = 0
    for idx, row in df.iterrows():
        d = row['date']
        if d in date_data:
            for p_col in part_cols:
                date_data[d][p_col] += int(row[p_col])
            added_count += 1

    print(f"  Added values for {added_count} matching dates")

# Convert back to DataFrame
result_rows = []
for d in sorted(date_data.keys()):
    row_data = {'date': d}
    row_data['L_1'] = date_data[d]['L_1']
    row_data['L_2'] = date_data[d]['L_2']
    row_data['L_3'] = date_data[d]['L_3']
    row_data['L_4'] = date_data[d]['L_4']
    row_data['L_5'] = date_data[d]['L_5']
    for p_col in part_cols:
        row_data[p_col] = date_data[d][p_col]
    result_rows.append(row_data)

result = pd.DataFrame(result_rows)

# Format date as M/D/YYYY (Windows compatible)
result['date'] = result['date'].dt.strftime('%#m/%#d/%Y')

# Save aggregated file
result.to_csv(output_file, index=False)

print(f"\n=== Aggregation Complete ===")
print(f"Output saved to: CA5_agregated_matrix.csv")
print(f"Total rows: {len(result)}")

# Show sample with max values
print("\nSample (first 5 rows, selected P columns):")
print(result[['date', 'P_1', 'P_10', 'P_20', 'P_30', 'P_39']].head())

# Show max value across all P columns
max_val = result[part_cols].max().max()
print(f"\nMax aggregated value in any P column: {max_val}")
