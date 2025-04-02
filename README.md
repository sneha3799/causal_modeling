# Health Data Processing and Causal Modeling

This repository consists of two main components:

1. **Health Data Processing**: Tools for cleaning, merging, validating, and visualizing health data from Gluroo (blood glucose, insulin, meals) and FitBit (sleep, stress, HRV).
2. **Causal Modeling**: Counterfactual modeling regarding insulin timing and dosage on blood glucose evolution using causal inference and causal ML methods.

## Part 1: Health Data Processing

### Overview
The data processing pipeline standardizes and combines data from Gluroo's app and FitBit exported data files, ensuring consistent timestamps, handling duplicates, and merging related data points. The process is implemented in `data_standardization/standardize_health_data.py`.

### Data Sources
```
data/
├── {user_id}_5th-7th.csv                 # Gluroo data months 5-7
├── {user_id}_7th-9th.csv                 # Gluroo data months 7-9
└── fitness/ # FitBit data
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

Visualization of the standardized data is available too.

## Part 2: Causal Modeling

### Overview
This section focuses on building causal models to model how different insulin timing and dosage decisions affect blood glucose levels as counterfcatuals to the observed blood glucose evolution over time. The goal is to understand counterfactuals in order to build intuition with an interactive method - "What if insulin was taken at a different time?" or "What if the dose was different?"

### Methodology
For our project we needed researched methods that handle:
* Time deconfounding
* Autocorrelation
* CATEs/ITEs rather than ATEs, i.e. causal effect on one person with any particular characteristics rather than a group
* Non-binary treatments and outcomes - since treatments are not binary, and neither is our outcome of intrest (note that we can discretize continuous values but not reduce to binary).

Our project builds on the following methods to find best suited methods that satisfy the above:
* T4 - [Estimating treatment effects for time-to-treatment antibiotic stewardship in sepsis
](https://pmc.ncbi.nlm.nih.gov/articles/PMC10135432/)
* the Causal Transformer - [Causal Transformer for Estimating Counterfactual Outcomes](https://arxiv.org/abs/2204.07258)
* G-Net - [A Recurrent Network Approach to G-Computation for Counterfactual Prediction Under a Dynamic Treatment Regime](https://proceedings.mlr.press/v158/li21a.html)
* Interupted Time Series with [causalimpact](https://pypi.org/project/causalimpact/)
