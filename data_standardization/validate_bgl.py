import pandas as pd
import numpy as np
from datetime import datetime
import glob

def standardize_timestamp(ts):
    """Convert timestamp to UTC and remove timezone info."""
    if pd.isna(ts):
        return pd.NaT
    
    ts = pd.to_datetime(ts, format='mixed')
    if ts.tzinfo is not None:
        # Convert timezone-aware timestamps to UTC
        ts = ts.tz_convert('UTC').tz_localize(None)
    return ts

def load_and_prepare_bgl(file_path):
    """Load BGL data and prepare it for comparison."""
    df = pd.read_csv(file_path)
    df['date'] = df['date'].apply(standardize_timestamp)
    return df[['date', 'bgl']].sort_values('date')

def compare_bgl_values(original_df, merged_df, source_name):
    """Compare BGL values between original and merged datasets."""
    print(f"\nValidating against {source_name}:")
    
    # Merge on exact timestamps
    comparison = pd.merge(
        original_df,
        merged_df[['timestamp', 'bgl']],
        left_on='date',
        right_on='timestamp',
        how='outer',
        suffixes=('_orig', '_merged')
    )
    
    # Find mismatches
    mismatches = comparison[
        (comparison['bgl_orig'].notna() & comparison['bgl_merged'].notna()) &
        (comparison['bgl_orig'] != comparison['bgl_merged'])
    ]
    
    # Find missing values
    missing_in_merged = comparison[
        comparison['bgl_orig'].notna() & comparison['bgl_merged'].isna()
    ]
    extra_in_merged = comparison[
        comparison['bgl_orig'].isna() & comparison['bgl_merged'].notna()
    ]
    
    print(f"Total readings in original: {len(original_df):,}")
    print(f"Matching timestamps: {len(comparison[comparison['bgl_orig'].notna() & comparison['bgl_merged'].notna()]):,}")
    print(f"Missing in merged: {len(missing_in_merged):,}")
    print(f"Extra in merged: {len(extra_in_merged):,}")
    print(f"Value mismatches: {len(mismatches):,}")
    
    if len(mismatches) > 0:
        print("\nSample of mismatches:")
        print(mismatches[['date', 'bgl_orig', 'bgl_merged']].head())
        print(f"\nLargest discrepancy: {abs(mismatches['bgl_orig'] - mismatches['bgl_merged']).max():.1f} mg/dL")
        
        # Print timestamp analysis for mismatches
        print("\nTimestamp analysis of mismatches:")
        time_diff = pd.to_datetime(mismatches['timestamp']) - pd.to_datetime(mismatches['date'])
        print(f"Average time difference: {time_diff.mean()}")
        print(f"Max time difference: {time_diff.max()}")

def main():
    # Load original files
    original_files = sorted(glob.glob("Data/679372_*.csv"))
    print("\nFound BGL files:")
    for i, file in enumerate(original_files):
        print(f"{i+1}. {file}")
    
    # Load all files
    original_dfs = []
    for file in original_files:
        print(f"\nLoading {file}")
        df = load_and_prepare_bgl(file)
        original_dfs.append((df, file))
    
    # Load merged file
    print("\nLoading merged_health_data.csv")
    merged_df = pd.read_csv("merged_health_data.csv")
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    
    # Print basic stats
    print("\n=== Basic Statistics ===")
    print(f"Records in merged file: {len(merged_df):,}")
    print(f"BGL readings in merged file: {merged_df['bgl'].notna().sum():,}")
    
    # Compare with each original file
    for df, file in original_dfs:
        compare_bgl_values(df, merged_df, file)
    
    # Analyze timestamp distribution
    print("\n=== Timestamp Analysis ===")
    bgl_times = merged_df[merged_df['bgl'].notna()]['timestamp']
    if len(bgl_times) > 0:
        print(f"Date range: {bgl_times.min()} to {bgl_times.max()}")
        days = (bgl_times.max() - bgl_times.min()).days
        if days > 0:
            print(f"Average readings per day: {len(bgl_times) / days:.1f}")
        else:
            print("Data spans less than one day")
    else:
        print("No BGL readings found in merged file")

if __name__ == "__main__":
    main() 