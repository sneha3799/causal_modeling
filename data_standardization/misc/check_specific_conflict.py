"""
Investigation of apparent BGL conflicts in the data.

Initial investigation revealed what looked like conflicting BGL values at the same timestamp
(e.g., 138 vs 126 at '2024-05-21 18:16:00'). However, detailed analysis showed these are
actually separate readings taken seconds apart:

1. 18:15:42 - BGL: 138.0 (regular reading)
2. 18:16:14 - BGL: 126.0 (TEXT message about Shawarma)
3. 18:16:15 - BGL: 126.0 (ANNOUNCE_MEAL with carbs)

This suggests these aren't conflicts but rather rapid BGL changes around meal times,
which is expected. The apparent conflict was due to timestamp rounding in the standardization process.

Proposed solution:
1. Round timestamps to the minute
2. For multiple readings within the same minute:
   - Use BGL from the reading without a msg_type (actual sensor reading)
   - Keep all events (meals, insulin, etc.) but use the BGL from the closest sensor reading
This preserves event information while maintaining consistent BGL values.
"""

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

def round_and_standardize_bgls(group):
    # If there's only one reading, return it as is
    if len(group) == 1:
        return group
    
    # Try to find a reading without a message type (sensor reading)
    sensor_readings = group[group['msg_type'].isna()]
    if not sensor_readings.empty:
        correct_bgl = sensor_readings.iloc[0]['bgl']
    else:
        # If no sensor reading, use the most common BGL value
        mode_values = group['bgl'].mode()
        if len(mode_values) > 0:
            correct_bgl = mode_values.iloc[0]
        else:
            # If no mode exists, use the first BGL value
            correct_bgl = group['bgl'].iloc[0]
    
    # Keep all rows but standardize the BGL values
    group['bgl'] = correct_bgl
    return group

# Read all blood glucose data
all_data = []
for file in glob.glob('data/679372_*.csv'):
    print(f"\nReading {file}")
    df = pd.read_csv(file, low_memory=False)
    df = standardize_timestamp(df)
    all_data.append(df)

merged_df = pd.concat(all_data)
merged_df = merged_df.sort_values('date')

# Round timestamps to the minute
merged_df['date_minute'] = merged_df['date'].dt.floor('min')

# Check specific timestamp where we know there's a conflict
target_time = pd.to_datetime('2024-05-21 18:16:00')

# Look at original data
print("\nOriginal data around target time:")
time_window = merged_df[
    (merged_df['date'] >= target_time - timedelta(minutes=1)) &
    (merged_df['date'] <= target_time + timedelta(minutes=1))
]
print(time_window[['date', 'bgl', 'msg_type', 'text']].sort_values('date'))

# Apply the standardization
print("\nAfter standardization:")
standardized_df = merged_df.copy()
standardized_df = standardized_df.groupby('date_minute', group_keys=False).apply(round_and_standardize_bgls)

# Look at standardized data
time_window = standardized_df[
    (standardized_df['date_minute'] >= target_time - timedelta(minutes=1)) &
    (standardized_df['date_minute'] <= target_time + timedelta(minutes=1))
]
print(time_window[['date', 'date_minute', 'bgl', 'msg_type', 'text']].sort_values('date'))

# Verify no more conflicts
conflicts = standardized_df.groupby('date_minute').agg({
    'bgl': lambda x: len(x.unique()) if not x.isna().all() else 0
}).query('bgl > 1')

if len(conflicts) > 0:
    print("\nWarning: Still found timestamps with multiple BGL values:")
    for time, row in conflicts.iterrows():
        print(f"\nAt {time}:")
        print(standardized_df[standardized_df['date_minute'] == time][['date', 'bgl', 'msg_type', 'text']]) 