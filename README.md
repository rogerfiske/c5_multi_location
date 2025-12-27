# C5 Multi-Location Parts Demand Dataset

## Overview
This project contains manufacturing parts demand data from 6 US state locations, each with 5 manufacturing facilities. The data spans from September 2008 to 2025.

## Dataset Structure

### Source Files (data/raw/)

| File | State | Rows | Date Range |
|------|-------|------|------------|
| CA5_matrix.csv | California | 6,318 | 9/8/2008 - 2025 |
| MI5_matrix.csv | Michigan | 6,313 | 9/8/2008 - 2025 |
| MO5_matrix.csv | Missouri | 6,318 | 9/8/2008 - 2025 |
| NY5_matrix.csv | New York | 6,316 | 9/8/2008 - 2025 |
| OH5_matrix.csv | Ohio | 6,318 | 9/8/2008 - 2025 |
| MD5_matrix.csv | Maryland | 6,317 | 9/8/2008 - 2025 |
| ME5_matrix.csv | Maine | 2,542 | 5/15/2013 - 2025 |

### Aggregated File
| File | Description | Rows |
|------|-------------|------|
| CA5_agregated_matrix.csv | Combined demand from CA, MI, MO, NY, OH, MD | 6,318 |

## Column Definitions

### Location Columns (L_1 to L_5)
- Each column represents one of 5 manufacturing locations within a state
- Values: Part numbers (1-39) required at that location for the given date

### Part Matrix Columns (P_1 to P_39)
- Binary indicator (0/1) in source files showing if part was required that day
- In aggregated file: Count (0-6) showing how many states required that part

## Data Characteristics

- **Parts Range**: 39 unique part numbers (1-39)
- **Locations per State**: 5
- **Total Locations (excluding ME)**: 30 (6 states x 5 locations)
- **Daily Parts per State**: 5 (one per location)
- **Daily Parts in Aggregated**: 30 (sum across 6 states)

## Notes
- ME5_matrix.csv excluded from aggregation due to shorter date range (2,542 vs ~6,318 records)
- Minor date gaps exist on holidays (Christmas, etc.) in some files
- All files aligned by date for accurate aggregation
