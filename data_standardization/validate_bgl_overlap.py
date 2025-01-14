import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare_data(file_path):
    """
    Load a CSV file and prepare it for comparison by standardizing timestamps
    and selecting relevant columns.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert timestamp to datetime with proper timezone handling
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], format='mixed')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    else:
        raise ValueError(f"No timestamp column found in {file_path}")
    
    # First localize naive timestamps to UTC, then convert any timezone-aware ones
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    # Round timestamps to the nearest minute to ensure matching
    df['timestamp'] = df['timestamp'].dt.round('min')
    
    # Select only timestamp and blood glucose columns
    if 'bgl' in df.columns:
        return df[['timestamp', 'bgl']].dropna(subset=['bgl'])
    else:
        return df[['timestamp', 'value']].dropna(subset=['value'])

def compare_dataframes(df1, df2, name1, name2):
    """
    Compare two dataframes and return statistics about their overlap
    """
    # Merge the dataframes on timestamp
    merged = pd.merge(df1, df2, on='timestamp', how='outer', suffixes=(f'_{name1}', f'_{name2}'))
    
    # Calculate statistics
    total_readings_1 = len(df1)
    total_readings_2 = len(df2)
    common_timestamps = merged.dropna().shape[0]
    only_in_1 = merged[merged[f'bgl_{name2}'].isna()].shape[0]
    only_in_2 = merged[merged[f'bgl_{name1}'].isna()].shape[0]
    
    # Check for value mismatches where timestamps overlap
    matching_times = merged.dropna()
    value_diffs = matching_times[f'bgl_{name1}'] - matching_times[f'bgl_{name2}']
    mismatches = (value_diffs.abs() > 0.01).sum()  # Using small threshold for float comparison
    
    if mismatches > 0:
        print("\nValue mismatches found:")
        print(f"Total mismatches: {mismatches}")
        
        # Analyze the distribution of differences
        diff_abs = value_diffs.abs()
        print("\nDifference distribution:")
        print(f"Mean difference: {diff_abs.mean():.2f}")
        print(f"Median difference: {diff_abs.median():.2f}")
        print(f"Max difference: {diff_abs.max():.2f}")
        print("\nDifference ranges:")
        ranges = [(0, 1), (1, 5), (5, 10), (10, 20), (20, float('inf'))]
        for start, end in ranges:
            count = ((diff_abs > start) & (diff_abs <= end)).sum()
            print(f"{start}-{end if end != float('inf') else '+'} mg/dL: {count} mismatches")
        
        # Show examples of largest mismatches
        print("\nLargest mismatches:")
        largest_mismatches = matching_times.loc[diff_abs.nlargest(5).index]
        for _, row in largest_mismatches.iterrows():
            print(f"\nTimestamp: {row['timestamp']}")
            print(f"{name1}: {row[f'bgl_{name1}']:.1f}")
            print(f"{name2}: {row[f'bgl_{name2}']:.1f}")
            print(f"Difference: {abs(row[f'bgl_{name1}'] - row[f'bgl_{name2}']):.1f}")
        
        # Check for patterns in mismatches
        print("\nTemporal pattern of mismatches:")
        matching_times['hour'] = matching_times['timestamp'].dt.hour
        mismatches_by_hour = (diff_abs > 0.01).groupby(matching_times['hour']).sum()
        print("Mismatches by hour:")
        for hour, count in mismatches_by_hour.items():
            if count > 0:
                print(f"{hour:02d}:00 - {count} mismatches")
    
    return {
        f'total_{name1}': total_readings_1,
        f'total_{name2}': total_readings_2,
        'common_timestamps': common_timestamps,
        f'only_in_{name1}': only_in_1,
        f'only_in_{name2}': only_in_2,
        'value_mismatches': mismatches
    }

def validate_overlap():
    """
    Main function to validate overlap between original and merged files
    """
    # File paths
    file1 = 'Data/679372_5th-7th.csv'
    file2 = 'Data/679372_7th-9th.csv'
    merged_file = 'merged_health_data.csv'
    
    print("Loading data files...")
    df1 = load_and_prepare_data(file1)
    df2 = load_and_prepare_data(file2)
    merged_df = load_and_prepare_data(merged_file)
    
    print("\nComparing first file with merged data...")
    stats1 = compare_dataframes(df1, merged_df, 'original1', 'merged')
    print("\nStatistics for first file vs merged:")
    for key, value in stats1.items():
        print(f"{key}: {value}")
    
    print("\nComparing second file with merged data...")
    stats2 = compare_dataframes(df2, merged_df, 'original2', 'merged')
    print("\nStatistics for second file vs merged:")
    for key, value in stats2.items():
        print(f"{key}: {value}")
    
    print("\nChecking for overlap between original files...")
    stats3 = compare_dataframes(df1, df2, 'file1', 'file2')
    print("\nStatistics for overlap between original files:")
    for key, value in stats3.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    validate_overlap() 