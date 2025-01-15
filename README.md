# Health Data Processing and Causal Modeling

This repository consists of two main components:

1. **Health Data Processing**: Tools for cleaning, merging, validating, and visualizing health data from Gluroo (blood glucose, insulin, meals) and FitBit (sleep, stress, HRV).
2. **Causal Modeling**: Counterfactual modeling regarding insulin timing and dosage on blood glucose evolution using causal inference and causal ML methods.

## Part 1: Health Data Processing

### Overview
The data processing pipeline standardizes and combines data from multiple sources, ensuring consistent timestamps, handling duplicates, and merging related data points. The process is implemented in `data_standardization/standardize_health_data.py`.

### Data Sources
```
data/
├── *_5th-7th.csv                 # Blood glucose data files
├── *_7th-9th.csv                 # Blood glucose data files
└── fitness/
    ├── DailyReadiness/
    │   └── Daily Readiness Score - *.csv
    ├── HeartRateVariability/
    │   ├── HRVDetails/
    │   │   └── Heart Rate Variability Details - *.csv
    │   ├── HRVSummary/
    │   │   └── Daily Heart Rate Variability Summary - *.csv
    │   └── RespiratoryRateSummary/
    │       └── Daily Respiratory Rate Summary - *.csv
    ├── SPO2/
    │   └── Minute SpO2 - *.csv
    ├── SleepScore/
    │   ├── Sleep Profile.csv
    │   └── sleep_score.csv
    ├── StressScore/
    │   └── Stress Score.csv
    └── Temperature/
        ├── Computed Temperature - *.csv
        └── Device Temperature - *.csv
```

Each data source provides specific health metrics:
- **Blood Glucose**: Continuous monitoring and finger prick readings (mg/dL)
- **Daily Readiness**: Overall readiness score and subcomponents
- **Sleep**: Sleep stages, duration, and quality metrics
- **Temperature**: Body temperature from wearable device
- **SpO2**: Blood oxygen saturation measurements
- **Stress**: Continuous stress level monitoring
- **HRV**: Heart rate variability metrics and respiratory rate

### Standardization Steps

1. **Timestamp Standardization**
   - Converts all timestamps to UTC
   - Rounds to nearest minute
   - Handles both naive and timezone-aware timestamps
   - Logs any timestamp adjustments for verification

2. **Duplicate Handling**
   - First removes exact duplicates (all columns identical)
   - Special handling for rows with same timestamp but different information:
     
     a. **Blood Glucose**:
     - Prioritizes sensor readings with valid trend values (not 'NONE')
     - If multiple readings with valid trends exist:
       - Uses average if values are within 5% of each other
       - Raises error if difference exceeds 5%
     - Preserves associated metadata (trend direction, device info)
     
     b. **Insulin Doses**:
     - Sums doses only for same insulin type at same timestamp
     - Example: At 2024-05-21 18:16:00
       - DOSE_INSULIN: 4u + 1u = 5u total regular insulin
       - DOSE_BASAL_INSULIN: 4u Toujeo (kept separate)
     - Maintains separation between regular and basal insulin
     - Preserves information about automatic vs manual doses
     
     c. **Food Amounts**:
     - Sums carbohydrates only for same food type at same timestamp
     - Example: At 2024-05-21 18:16:00
       - "Skittles": 5g + 10g = 15g total
       - "Apple": 25g (kept separate)
     - Maintains distinct glycemic indices
     - Preserves meal vs snack categorization
     
     d. **Boolean Columns**:
     - Uses OR operation for flags and indicators
     - Examples:
       - affects_fob (food on board): TRUE + FALSE = TRUE
       - affects_iob (insulin on board): TRUE + FALSE = TRUE
       - dose_automatic: FALSE + FALSE = FALSE
     - Ensures no information loss for tracking active insulin/food

3. **Data Quality Checks**
   - Verifies no BGL values were modified during processing
   - Checks for NaT (Not a Time) timestamps
   - Validates data consistency across files
   - Reports statistics on duplicates and adjustments

4. **Merging Process**
   - Concatenates data from multiple files
   - Sorts by timestamp
   - Handles overlapping time periods
   - Preserves all relevant metadata

## Interactive Visualization

The visualization tool (`data_standardization/visualize_health_data.py`) creates an interactive HTML plot combining all health metrics.

### Features

1. **Main Plot**
   - Blood Glucose line (primary y-axis, 70-180 mg/dL target range)
   - Normalized health metrics (secondary y-axis, 0-100 scale)
   - Event markers (meals, insulin doses, etc.)
   - Interactive legend for toggling metrics

2. **Time Controls**
   - Range slider for time window selection
   - Quick selection buttons (6h, 12h, 1d, 3d, 1w, All)
   - Pan and zoom capabilities

3. **Hover Information**
   - Original values for all metrics
   - Normalized values where applicable
   - Timestamps and event details
   - Trend information for CGM readings

4. **Metric Details Panel**
   - Dynamic updates based on selected metrics
   - Shows scale, description, and data processing info
   - Measurement frequency for each metric
   - Organized in a grid layout

### How to Use

1. **Running the Visualization**
   ```bash
   python data_standardization/visualize_health_data.py
   ```
   This generates `visualizations/health_metrics.html`

2. **Interacting with the Plot**
   - Click legend items to show/hide metrics
   - Use time controls to focus on specific periods
   - Hover over points for detailed information
   - Click and drag to zoom
   - Double-click to reset view

3. **Understanding the Metrics**
   - Blood Glucose: Original scale (mg/dL)
   - Sleep Metrics: Normalized from minutes to 0-100 scale
   - Stress Score: Original 0-100 scale
   - HRV: Normalized from milliseconds to 0-100 scale

4. **Event Types**
   - DOSE_INSULIN: Regular insulin doses
   - DOSE_BASAL_INSULIN: Long-acting insulin
   - ANNOUNCE_MEAL: Meal announcements
   - INTERVENTION_SNACK: Fast-acting carbs
   - BGL_FP_READING: Finger prick readings

## Data Processing Notes

1. **Insulin Doses**
   - Multiple doses at same timestamp are summed
   - Preserves distinction between regular and basal insulin
   - Frequency: Event-based (when administered)

2. **Meal Announcements**
   - Multiple food amounts at same timestamp are summed
   - Includes glycemic index information
   - Frequency: Event-based (when consumed)

3. **Blood Glucose Readings**
   - Prioritizes sensor readings with valid trends
   - Handles conflicts within 5% difference
   - Frequency: Every 5 minutes from CGM

4. **Sleep Metrics**
   - Normalized to 0-100 scale for comparison
   - Original values preserved in hover data
   - Frequency: Daily totals

5. **Stress and HRV**
   - Stress: Original 0-100 scale
   - HRV: Normalized from typical 0-150ms range
   - Frequency: Throughout day/night

## Part 2: Causal Modeling

### Overview
This section focuses on building causal models to model how different insulin timing and dosage decisions affect blood glucose levels as counterfcatuals to the observed blood glucose evolution over time. The goal is to understand counterfactuals in order to build intuition with an interactive method - "What if insulin was taken at a different time?" or "What if the dose was different?"

### Methodology
TBC

### Usage
TBC
