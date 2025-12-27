import pandas as pd

# List of CSV files to process
files = [
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\MO5_matrix.csv',
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\NY5_matrix.csv',
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\OH5_matrix.csv',
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\MD5_matrix.csv',
    r'C:\Users\Minis\CascadeProjects\c5_multi_location\data\raw\ME5_matrix.csv',
]

location_cols = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5']
part_cols = [f'P_{i}' for i in range(1, 40)]
all_cols = ['date'] + location_cols + part_cols

for filepath in files:
    filename = filepath.split('\\')[-1]
    print(f"\nProcessing: {filename}")

    # Read first row to check if headers exist
    first_row = pd.read_csv(filepath, nrows=1, header=None)
    first_val = str(first_row.iloc[0, 0])

    # Check if first row is a header (contains 'date' or 'L_')
    has_header = 'date' in first_val.lower() or 'L_' in first_val

    if has_header:
        df = pd.read_csv(filepath)
    else:
        # No header - read with custom column names
        df = pd.read_csv(filepath, header=None, names=['date'] + location_cols)

    # Add P columns if they don't exist
    for p_col in part_cols:
        if p_col not in df.columns:
            df[p_col] = 0

    # Ensure correct column order
    df = df[all_cols]

    # For each row, set P columns based on L values
    for idx, row in df.iterrows():
        # Initialize all P columns to 0
        for p_col in part_cols:
            df.at[idx, p_col] = 0

        # Set P columns to 1 based on L values
        for l_col in location_cols:
            part_num = row[l_col]
            if pd.notna(part_num):
                part_num = int(part_num)
                if 1 <= part_num <= 39:
                    df.at[idx, f'P_{part_num}'] = 1

    # Convert P columns to integers
    for p_col in part_cols:
        df[p_col] = df[p_col].astype(int)

    # Save the updated CSV
    df.to_csv(filepath, index=False)

    print(f"  Processed {len(df)} rows")

print("\nAll files completed!")
