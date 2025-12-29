"""
Fix CA5_raw_data.txt data quality issues:
1. Sort L values within each row to be ascending (L_1 < L_2 < L_3 < L_4 < L_5)
2. Sort all rows by date chronologically
"""
import csv
from datetime import datetime
from pathlib import Path

def parse_date(date_str):
    """Parse date in M/D/YYYY format"""
    return datetime.strptime(date_str, "%m/%d/%Y")

def fix_raw_data(input_path, output_path=None):
    """Fix ordering issues in raw data file"""
    if output_path is None:
        output_path = input_path

    rows = []
    violations_found = 0

    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for line_num, row in enumerate(reader, start=2):
            date_str = row[0]
            l_values = [int(x) for x in row[1:6]]

            # Check if already sorted
            if l_values != sorted(l_values):
                violations_found += 1
                print(f"Line {line_num}: {date_str} - {l_values} -> {sorted(l_values)}")

            # Sort L values
            sorted_l = sorted(l_values)

            # Parse date for sorting
            try:
                parsed_date = parse_date(date_str)
            except ValueError:
                # Try alternate format
                parsed_date = datetime.strptime(date_str, "%Y-%m-%d")

            rows.append((parsed_date, date_str, sorted_l))

    # Sort by date
    rows.sort(key=lambda x: x[0])

    # Check if dates were out of order
    date_order_issues = 0
    prev_date = None
    for parsed_date, date_str, _ in rows:
        if prev_date and parsed_date < prev_date:
            date_order_issues += 1
        prev_date = parsed_date

    # Write corrected file
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for parsed_date, date_str, l_values in rows:
            writer.writerow([date_str] + l_values)

    print(f"\n=== Summary ===")
    print(f"Total rows processed: {len(rows)}")
    print(f"Ascending order violations fixed: {violations_found}")
    print(f"Date order issues: {date_order_issues}")
    print(f"Output written to: {output_path}")

    return violations_found, date_order_issues

if __name__ == "__main__":
    input_file = Path(__file__).parent.parent / "data" / "raw" / "CA5_raw_data.txt"
    fix_raw_data(input_file)
