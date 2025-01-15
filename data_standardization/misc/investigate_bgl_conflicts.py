import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta

def standardize_timestamp(df, date_col='date'):
    """Standardize timestamps to UTC."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], format='mixed')
    if df[date_col].dt.tz is None:
        df[date_col] = df[date_col].dt.tz_localize('UTC')
    df[date_col] = df[date_col].dt.tz_convert('UTC').dt.tz_localize(None)
    return df

def analyze_bgl_conflicts():
    print("Analyzing BGL conflicts...")
    
    # Read all blood glucose data
    all_data = []
    for file in glob.glob('data/679372_*.csv'):
        print(f"\nReading {file}")
        df = pd.read_csv(file, low_memory=False)
        df = standardize_timestamp(df)
        all_data.append(df)
    
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df = merged_df.sort_values('date')
    
    # Find timestamps with multiple rows
    duplicates = merged_df[merged_df.duplicated('date', keep=False)].copy()
    print(f"\nFound {len(duplicates)} rows with duplicate timestamps")
    print(f"This corresponds to {len(duplicates['date'].unique())} unique timestamps")
    
    # Analyze BGL differences for each timestamp
    conflicts = []
    for timestamp, group in duplicates.groupby('date'):
        # Get all BGL values
        bgls = group['bgl'].values
        # Remove NaN values
        valid_bgls = bgls[~np.isnan(bgls)]
        
        if len(valid_bgls) > 1:  # If we have multiple BGL values
            # Check if they're actually different
            unique_bgls = np.unique(valid_bgls)
            if len(unique_bgls) > 1:  # If we have different values
                max_diff = np.max(valid_bgls) - np.min(valid_bgls)
                mean_val = np.mean(valid_bgls)
                max_diff_pct = (max_diff / mean_val) * 100 if mean_val != 0 else 0
                
                # Count occurrences of each BGL value (including NaN)
                bgl_counts = group['bgl'].value_counts(dropna=False).to_dict()
                
                conflicts.append({
                    'timestamp': timestamp,
                    'bgl_values': unique_bgls,
                    'bgl_value_counts': bgl_counts,
                    'max_difference': max_diff,
                    'max_difference_pct': max_diff_pct,
                    'n_readings': len(unique_bgls),
                    'msg_types': group['msg_type'].unique(),
                    'texts': group['text'].unique(),
                    'rows': group[['date', 'bgl', 'msg_type', 'text']].to_dict('records')
                })
    
    if not conflicts:
        print("\nNo BGL conflicts found!")
        return
    
    conflicts_df = pd.DataFrame(conflicts)
    
    # Basic statistics
    print("\nBGL Conflict Statistics:")
    print(f"Total timestamps with different BGL values: {len(conflicts_df)}")
    print(f"Average number of different BGL values per conflict: {conflicts_df['n_readings'].mean():.1f}")
    print("\nDistribution of BGL differences:")
    print(conflicts_df['max_difference'].describe())
    print("\nDistribution of BGL differences (%):")
    print(conflicts_df['max_difference_pct'].describe())
    
    # Analyze all conflicts
    print("\nAnalyzing all BGL conflicts (sorted by difference %):")
    for _, conflict in conflicts_df.sort_values('max_difference_pct', ascending=False).iterrows():
        print(f"\nTimestamp: {conflict['timestamp']}")
        print(f"BGL values and counts: {conflict['bgl_value_counts']}")
        print(f"Max difference: {conflict['max_difference']:.1f} ({conflict['max_difference_pct']:.1f}%)")
        print("Rows (sorted by BGL value):")
        for row in sorted(conflict['rows'], key=lambda x: (pd.isna(x['bgl']), x['bgl'] if pd.notna(x['bgl']) else 0)):
            bgl_str = f"{row['bgl']:.1f}" if pd.notna(row['bgl']) else "NaN"
            msg_type_str = row['msg_type'] if pd.notna(row['msg_type']) else "NaN"
            text_str = row['text'] if pd.notna(row['text']) else "NaN"
            print(f"- BGL: {bgl_str}, Type: {msg_type_str}, Text: {text_str}")
    
    # Save conflicts to CSV
    conflicts_expanded = []
    for _, conflict in conflicts_df.iterrows():
        for row in conflict['rows']:
            conflicts_expanded.append({
                'timestamp': conflict['timestamp'],
                'max_difference': conflict['max_difference'],
                'max_difference_pct': conflict['max_difference_pct'],
                'n_different_bgls': conflict['n_readings'],
                'bgl': row['bgl'],
                'msg_type': row['msg_type'],
                'text': row['text'],
                'bgl_value_counts': str(conflict['bgl_value_counts'])
            })
    
    output_df = pd.DataFrame(conflicts_expanded)
    output_df = output_df.sort_values(['max_difference_pct', 'timestamp'], ascending=[False, True])
    output_df.to_csv('bgl_conflicts.csv', index=False)
    print("\nSaved all conflicts to 'bgl_conflicts.csv' (sorted by difference %)")
    
    # Analyze patterns
    print("\nAnalyzing patterns in conflicts:")
    print("\nMessage types involved in conflicts:")
    msg_types = [mt for conflict in conflicts_df['msg_types'] for mt in conflict if pd.notna(mt)]
    msg_type_counts = pd.Series(msg_types).value_counts()
    print(msg_type_counts)
    
    # Time analysis
    conflicts_df['hour'] = pd.to_datetime(conflicts_df['timestamp']).dt.hour
    print("\nDistribution of conflicts by hour of day:")
    print(conflicts_df['hour'].value_counts().sort_index())

if __name__ == "__main__":
    analyze_bgl_conflicts() 