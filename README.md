# Diabetes Causal Modeling

## Project Overview
This repository implements a comprehensive pipeline for processing multi-source health data and building causal models to understand the impact of insulin timing and dosage on blood glucose levels.

## Repository Components

### 1. Health Data Processing
Tools for cleaning, merging, validating, and visualizing health data from multiple sources:
- **Gluroo**: Blood glucose monitoring, insulin administration, and meal tracking
- **FitBit**: Sleep metrics, stress levels, heart rate variability (HRV), and other biometrics

### 2. Causal Modeling
Advanced counterfactual modeling to understand how different insulin timing and dosage decisions affect blood glucose evolution, using causal inference and machine learning methods.

---

## Health Data Processing

### Pipeline Overview
The data processing pipeline standardizes and combines data from Gluroo and FitBit sources, ensuring:
- Consistent timestamp formats
- Duplicate handling
- Proper merging of related data points
- Data validation and quality checks

**Implementation**: `data_standardization/standardize_health_data.py`

### Data Sources Structure
```
data/
├── {user_id}_5th-7th.csv                 # Gluroo data months 5-7
├── {user_id}_7th-9th.csv                 # Gluroo data months 7-9
└── fitness/                              # FitBit data
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

### Data Metrics

| Source | Metrics | Description |
|--------|---------|-------------|
| **Gluroo** | Blood Glucose | Continuous monitoring (mg/dL) |
| **Gluroo** | Insulin | Timing, dosage, and type of insulin administration |
| **Gluroo** | Meals | Carbohydrate intake, meal timing, and food categories |
| **FitBit** | Daily Readiness | Overall readiness score and subcomponents |
| **FitBit** | Sleep | Sleep stages, duration, and quality metrics |
| **FitBit** | Temperature | Body temperature from wearable device |
| **FitBit** | SpO2 | Blood oxygen saturation measurements |
| **FitBit** | Stress | Continuous stress level monitoring |
| **FitBit** | HRV | Heart rate variability metrics and respiratory rate |

### Visualization
The repository includes tools for visualizing the standardized data, allowing for intuitive exploration of relationships between different health metrics.

---

## Causal Modeling

### Objective
Build causal models to understand how different insulin timing and dosage decisions affect blood glucose levels as counterfactuals to the observed blood glucose evolution over time.

### Key Questions Addressed
- "What if insulin was taken at a different time?"
- "What if the insulin dose was different?"
- "What if I had more carbs?"
- "What if I had the meal at a different time?"

### Methodological Challenges

Our project addresses several key challenges in health-related causal inference:

1. **Time Deconfounding**: Capturing exact causal effects in time series data
2. **Autocorrelation**: Managing recursive effects where outcome at $t=t_i$ is affected by $t_{i-1}$
3. **Individual Treatment Effects / Conditional Average Treatment Effects**: Identifying causal effects for specific individuals rather than population averages
4. **Continuous Treatments and Outcomes**: Handling non-binary treatments and continuous outcome variables

### Applied Methods

Our causal modeling approach builds on several advanced techniques:

1. **T4 Framework** - [Estimating treatment effects for time-to-treatment antibiotic stewardship in sepsis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10135432/)
   - Addresses time-varying treatments in clinical settings

2. **Causal Transformer** - [Estimating Counterfactual Outcomes](https://arxiv.org/abs/2204.07258)
   - Leverages transformer architecture for temporal dependencies in causal inference

3. **G-Net** - [A Recurrent Network Approach to G-Computation for Counterfactual Prediction Under a Dynamic Treatment Regime](https://proceedings.mlr.press/v158/li21a.html)
   - Implements G-computation using recurrent neural networks for dynamic treatments

4. **Interrupted Time Series Analysis** - Using [causalimpact](https://pypi.org/project/causalimpact/)
   - Evaluates the impact of interventions on time series data

### Implementation and Results

## Acknowledgments
- This work is still ongoing, but we'd like to acknowledge the Wat.ai design team at the University of Waterloo, which we're apart of, the authors of the above papers, Gluroo Inc., and all our colleagues at the Blood Glucose Control group at Wat.ai
