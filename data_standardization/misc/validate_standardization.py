import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import warnings

def validate_timestamps(df):
    """Validate timestamp standardization and handling."""
    print("\n=== Timestamp Validation ===")
    
    # Check for timezone info (should be None after standardization)
    if hasattr(df['timestamp'].dt, 'tz'):
        print("Warning: Found timezone info in timestamps")
    
    # Check for minute-level rounding
    time_components = pd.DataFrame({
        'second': df['timestamp'].dt.second,
        'microsecond': df['timestamp'].dt.microsecond
    })
    if (time_components != 0).any().any():
        print("Warning: Found timestamps not rounded to minute level")
        print("Non-zero components:")
        print(time_components[(time_components != 0).any(axis=1)].head())
    
    # Check for gaps in timeline
    time_diff = df['timestamp'].diff()
    gaps = time_diff != timedelta(minutes=1)
    if gaps.any():
        print(f"\nFound {gaps.sum():,} gaps in timeline")
        print("Largest gaps:")
        largest_gaps = time_diff[gaps].sort_values(ascending=False).head()
        for i, gap in enumerate(largest_gaps):
            print(f"{gap} at {df['timestamp'].iloc[largest_gaps.index[i]]}")
    
    print("\nTimestamp range:", df['timestamp'].min(), "to", df['timestamp'].max())
    print("Total minutes:", len(df))

def validate_insulin_doses(df):
    """Validate insulin dose handling."""
    print("\n=== Insulin Dose Validation ===")
    
    # Get rows with insulin doses
    insulin_rows = df[df['msg_type'].str.contains('INSULIN', na=False)]
    print(f"Total rows with insulin doses: {len(insulin_rows)}")
    
    # Find timestamps with multiple doses
    dose_groups = insulin_rows.groupby('timestamp')
    multiple_doses = dose_groups.filter(lambda x: len(x) > 1)
    print(f"\nFound {len(multiple_doses)//2} rows with multiple insulin doses at same timestamp")
    
    if len(multiple_doses) > 0:
        print("\nSample of multiple doses:\n")
        for timestamp, group in multiple_doses.groupby('timestamp'):
            print(f"\nTimestamp: {timestamp}")
            print(group[['dose_units', 'msg_type', 'text']].to_string())
    
    # Calculate statistics by type
    print("\nDose statistics by type:\n")
    for msg_type in insulin_rows['msg_type'].unique():
        type_doses = insulin_rows[insulin_rows['msg_type'] == msg_type]['dose_units']
        # Convert to numeric, dropping any non-numeric values
        type_doses = pd.to_numeric(type_doses, errors='coerce')
        print(f"{msg_type}:")
        print(f"Count: {len(type_doses)}")
        print(f"Mean: {type_doses.mean():.1f}u")
        print(f"Min: {type_doses.min():.1f}u")
        print(f"Max: {type_doses.max():.1f}u\n")

def validate_food_amounts(df):
    """Validate food amount handling."""
    print("\n=== Food Amount Validation ===")
    
    # Find rows with food amounts
    food_rows = df[df['food_g'].notna()]
    print(f"Total rows with food amounts: {len(food_rows):,}")
    
    # Group by timestamp and analyze
    food_groups = food_rows.groupby('timestamp')
    multiple_foods = food_groups.filter(lambda x: len(x) > 1)
    
    if len(multiple_foods) > 0:
        print(f"\nFound {len(multiple_foods):,} rows with multiple food entries at same timestamp")
        print("\nSample of multiple entries:")
        for ts, group in multiple_foods.groupby('timestamp'):
            print(f"\nTimestamp: {ts}")
            print(group[['food_g', 'msg_type', 'text']].to_string())
            if len(group) >= 5:  # Only show first few groups
                break
    
    # Analyze food amount distributions
    print("\nFood amount statistics:")
    food_amounts = food_rows['food_g']
    print(f"Mean: {food_amounts.mean():.1f}g")
    print(f"Max: {food_amounts.max():.1f}g")
    print(f"Common amounts:")
    print(food_amounts.value_counts().head())

def validate_bgl_conflicts(df):
    """Validate BGL conflict resolution."""
    print("\n=== BGL Conflict Validation ===")
    
    # Find rows with BGL values
    bgl_rows = df[df['bgl'].notna()]
    print(f"Total rows with BGL values: {len(bgl_rows):,}")
    
    # Group by timestamp and analyze
    bgl_groups = bgl_rows.groupby('timestamp')
    multiple_bgls = bgl_groups.filter(lambda x: len(x) > 1)
    
    if len(multiple_bgls) > 0:
        print(f"\nFound {len(multiple_bgls):,} rows with multiple BGL values at same timestamp")
        print("\nSample of multiple values:")
        for ts, group in multiple_bgls.groupby('timestamp'):
            print(f"\nTimestamp: {ts}")
            print(group[['bgl', 'msg_type', 'trend', 'text']].to_string())
            if len(group) >= 5:  # Only show first few groups
                break
    
    # Analyze BGL distributions by source
    print("\nBGL statistics by source:")
    for msg_type in bgl_rows['msg_type'].unique():
        type_label = 'Sensor readings' if pd.isna(msg_type) else msg_type
        type_bgls = bgl_rows[bgl_rows['msg_type'].isna() if pd.isna(msg_type) else bgl_rows['msg_type'] == msg_type]['bgl']
        if len(type_bgls) > 0:
            print(f"\n{type_label}:")
            print(f"Count: {len(type_bgls):,}")
            print(f"Mean: {type_bgls.mean():.1f}")
            print(f"Std Dev: {type_bgls.std():.1f}")

def validate_data_merging(df):
    """Validate data merging from multiple sources."""
    print("\n=== Data Merging Validation ===")
    
    # Check column presence
    expected_cols = {
        'timestamp', 'bgl', 'trend', 'msg_type', 'text', 'dose_units', 'food_g',
        'value', 'value_minute', 'value_device', 'STRESS_SCORE', 'overall_score'
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        print("Warning: Missing expected columns:", missing_cols)
    
    # Check data presence by source
    print("\nData presence by source:")
    sources = {
        'Blood Glucose': 'bgl',
        'SpO2': 'value_minute',
        'Temperature': 'value_device',
        'Stress Score': 'STRESS_SCORE',
        'Sleep Score': 'overall_score'
    }
    
    for source, col in sources.items():
        if col in df.columns:
            present = df[col].notna().sum()
            print(f"{source}: {present:,} records ({present/len(df)*100:.1f}% coverage)")

def analyze_missing_patterns(df):
    """Analyze patterns in missing data."""
    print("\n=== Missing Data Pattern Analysis ===")
    
    # Calculate missing percentages
    missing_pct = df.isnull().mean() * 100
    print("\nMissing percentages by column:")
    for col, pct in missing_pct.sort_values(ascending=False).items():
        if pct > 0:
            print(f"{col}: {pct:.1f}%")
    
    # Analyze missing data patterns by time of day
    df['hour'] = df['timestamp'].dt.hour
    print("\nMissing data patterns by hour:")
    
    key_cols = ['bgl', 'value_minute', 'value_device', 'STRESS_SCORE']
    for col in key_cols:
        if col in df.columns:
            missing_by_hour = df.groupby('hour')[col].apply(lambda x: x.isnull().mean() * 100)
            print(f"\n{col}:")
            for hour, pct in missing_by_hour.items():
                print(f"Hour {hour:02d}: {pct:.1f}%")

def main():
    print("Loading merged dataset...")
    df = pd.read_csv('output/merged_health_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Run all validations
    validate_timestamps(df)
    validate_insulin_doses(df)
    validate_food_amounts(df)
    validate_bgl_conflicts(df)
    validate_data_merging(df)
    analyze_missing_patterns(df)
    
    print("\nValidation complete.")

if __name__ == "__main__":
    main() 