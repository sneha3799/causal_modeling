#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python run_its_models.py --analysis counterfactual --data-path synthetic_data/data/dose_counterfactuals --output-dir output/counterfactual_analysis --max-events 5
# python run_its_models.py --analysis single --data-path synthetic_data/data/ml_dataset.csv --output-dir output/counterfactual_analysis --max-events 5

"""
ITS Model Runner
---------------
This script runs Interrupted Time Series (ITS) models using both CausalImpact and 
statsmodels on time series data. It can handle both single dataset analysis and 
counterfactual analysis with multiple datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os
import glob
import argparse
import warnings
import json
from datetime import datetime, timedelta
from causalimpact import CausalImpact

# Suppress common warnings
warnings.filterwarnings("ignore", message="DataFrame.fillna with 'method' is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame.applymap has been deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="No frequency information was provided", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", message="No frequency information was provided", category=UserWarning)


def to_timestamp(dt):
    """Convert datetime to timestamp integer for indexing"""
    if isinstance(dt, pd.Timestamp):
        return dt.value // 10**9
    return pd.Timestamp(dt).value // 10**9


def load_dataset(file_path):
    """
    Load a single dataset from CSV file
    """
    data = pd.read_csv(file_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    print(f"Loaded dataset from {file_path}")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data


def load_counterfactual_datasets(data_path, file_pattern="insulin_factor_*.csv", use_timestamps=False):
    """
    Load multiple counterfactual datasets
    """
    dataset_files = glob.glob(os.path.join(data_path, file_pattern))
    
    datasets = {}
    for file in dataset_files:
        factor = os.path.basename(file).replace("insulin_factor_", "").replace("_", ".").replace(".csv", "")
        
        # Load data
        df = pd.read_csv(file, index_col=0)
        df.index = pd.to_datetime(df.index)
        
        if use_timestamps:
            # Store original datetime as a column
            df['datetime'] = df.index.copy()
            
            # Convert datetime index to numerical timestamp format (seconds since epoch)
            df.index = df.index.astype(np.int64) // 10**9
        
        datasets[factor] = df
    
    print(f"Loaded {len(datasets)} counterfactual datasets with insulin factors:")
    for factor in sorted(datasets.keys()):
        print(f"  - {factor}: {len(datasets[factor])} data points")
    
    return datasets


def detect_events(data, event_column='insulin', threshold=0, min_gap_hours=1, max_events=None):
    """
    Detect events in the data with minimum gap
    """
    all_events = data.index[data[event_column] > threshold]
    
    filtered_events = []
    last_event = None
    
    for event in all_events:
        if last_event is None or (event - last_event).total_seconds() >= min_gap_hours * 3600:
            filtered_events.append(event)
            last_event = event
    
    print(f"Found {len(filtered_events)} filtered events")
    
    # Limit number of events if specified
    if max_events:
        filtered_events = filtered_events[:max_events]
        print(f"Using first {len(filtered_events)} events")
    
    return filtered_events


def prepare_window_data(data, event, pre_window, post_window):
    """
    Extract window data around an event
    """
    # Define the window
    window_start = event - pd.Timedelta(pre_window)
    window_end = event + pd.Timedelta(post_window)
    
    # Extract the data
    window_data = data.loc[window_start:window_end].copy()
    
    # Forward fill missing values
    window_data = window_data.fillna(method='ffill')
    
    return window_data


def clean_data_for_causalimpact(window_data):
    """
    Clean data for CausalImpact analysis
    """
    ci_data = window_data.copy()
    ci_data = ci_data.replace([np.inf, -np.inf], np.nan)
    ci_data = ci_data.dropna()
    
    # Only keep columns with more than one unique value
    ci_data = ci_data.loc[:, ci_data.nunique() > 1]
    
    return ci_data


def run_causalimpact(ci_data, pre_period, post_period, event_time=None, debug=False):
    """
    Run CausalImpact analysis
    """
    # Adjust pre-data if needed to avoid constant values
    pre_data = ci_data.loc[pre_period[0]:pre_period[1]].copy()
    for col in pre_data.columns:
        if pre_data[col].nunique() == 1:
            constant_val = pre_data[col].iloc[0]
            ci_data.loc[pre_period[1], col] = constant_val + 0.001
            if debug:
                print(f"Adjusted pre-data for column {col} to avoid constant value")
    
    if debug:
        print(f"Running CausalImpact with data shape: {ci_data.shape}")
        print(f"Columns: {ci_data.columns.tolist()}")
        print(f"Pre-period: {pre_period}")
        print(f"Post-period: {post_period}")
    
    # Run the model
    impact = CausalImpact(ci_data, pre_period, post_period)
    
    # Extract results
    post_inferences = impact.inferences.loc[impact.inferences.index >= post_period[0]]
    avg_effect = post_inferences['point_effects'].mean()
    cum_effect = post_inferences['post_cum_effects'].iloc[-1]
    
    results = {
        'event_time': event_time,
        'avg_effect': avg_effect,
        'cum_effect': cum_effect,
        'impact': impact
    }
    
    return results


def run_its_statsmodel(window_data, event, time_unit='minutes'):
    """
    Run Interrupted Time Series analysis using statsmodels
    """
    # Reset index to prepare for statsmodels
    data_reset = window_data.reset_index()
    
    # Create time variables
    divisor = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}.get(time_unit, 60)
    
    if 'index' in data_reset.columns:
        data_reset['time'] = (data_reset['index'] - data_reset['index'].min()).dt.total_seconds() / divisor
    else:
        data_reset['time'] = np.arange(len(data_reset))
    
    # Create intervention indicator and interaction term
    data_reset['post'] = (data_reset['index'] > event).astype(int)
    data_reset['time_post'] = data_reset['time'] * data_reset['post']
    
    # Fit the model
    model = smf.ols("glucose ~ time + post + time_post", data=data_reset).fit()
    
    # Extract results
    level_change = model.params['post']
    slope_change = model.params['time_post']
    
    results = {
        'model': model,
        'level_change': level_change,
        'slope_change': slope_change,
        'pvalue_level': model.pvalues['post'],
        'pvalue_slope': model.pvalues['time_post'],
        'fitted_values': model.fittedvalues,
        'data': data_reset
    }
    
    return results


def plot_causalimpact_result(impact, save_path=None):
    """
    Plot CausalImpact results
    """
    impact.plot()
    fig = plt.gcf()
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    
    return fig


def plot_its_statsmodel_result(its_results, event, save_path=None):
    """
    Plot statsmodels ITS results
    """
    fig = plt.figure(figsize=(10, 5))
    
    data = its_results['data']
    
    # Plot observed values
    plt.plot(data['index'], data['glucose'], 'o-', alpha=0.7, label='Observed Glucose')
    
    # Plot fitted values
    plt.plot(data['index'], its_results['fitted_values'], 'r--', linewidth=2, label='Fitted')
    
    # Mark intervention
    plt.axvline(x=event, color='k', linestyle='--', label='Intervention')
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Glucose (mg/dL)')
    plt.title('Interrupted Time Series Analysis')
    plt.legend(loc='best')
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    
    return fig


def plot_counterfactual_comparison(datasets, save_path=None):
    """
    Plot comparison of counterfactual datasets
    """
    fig = plt.figure(figsize=(12, 6))
    colors = {"0.8": "red", "0.9": "orange", "1.0": "green", "1.1": "blue", "1.2": "purple"}
    
    for factor, df in datasets.items():
        color = colors.get(factor, "gray")
        label = f"Insulin × {factor}"
        plt.plot(df.index, df['glucose'], label=label, color=color, linewidth=2 if factor == "1.0" else 1.5)
    
    # Add guidelines for glucose ranges
    plt.axhline(y=180, color='red', linestyle='--', alpha=0.7, label='High threshold')
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Low threshold')
    plt.axhline(y=100, color='green', linestyle=':', alpha=0.7, label='Target')
    
    plt.title('Insulin Dose Counterfactual Comparison')
    plt.xlabel('Time')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.ylim(40, 250)
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    
    return fig


def run_single_dataset_analysis(data_path, output_dir, max_events=10, pre_window='45min', post_window='30min'):
    """
    Run analysis on a single dataset
    """
    print("\n===== Single Dataset Analysis =====\n")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(os.path.join(output_dir, "causalimpact"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "statsmodels"), exist_ok=True)
    
    # Load the dataset
    data = load_dataset(data_path)
    
    # Detect events
    events = detect_events(data, max_events=max_events)
    
    if not events:
        print("No events detected. Analysis cannot proceed.")
        return None
    
    # Store results
    causalimpact_results = []
    statsmodels_results = []
    
    # Process each event
    for event_idx, event in enumerate(events):
        print(f"\n{'='*50}")
        print(f"Processing event {event_idx+1}/{len(events)} at {event}")
        
        try:
            # Extract window data
            window_data = prepare_window_data(data, event, pre_window, post_window)
            
            if window_data.empty:
                print(f"Skipping event - no data in window")
                continue
            
            # Define pre/post periods for CausalImpact
            closest_pre = window_data.index[window_data.index <= event].max()
            closest_post = window_data.index[window_data.index > event].min()
            
            if pd.isna(closest_pre) or pd.isna(closest_post):
                print(f"Skipping event - cannot establish pre/post boundaries")
                continue
            
            pre_period = [window_data.index.min(), closest_pre]
            post_period = [closest_post, window_data.index.max()]
            
            # Run CausalImpact analysis
            try:
                # Clean data
                ci_data = clean_data_for_causalimpact(window_data)
                
                # Run the model
                ci_results = run_causalimpact(ci_data, pre_period, post_period, event)
                
                # Save plot
                if output_dir:
                    plot_path = os.path.join(output_dir, "causalimpact", f"event_{event_idx}_impact.png")
                    plot_causalimpact_result(ci_results['impact'], save_path=plot_path)
                
                # Store results
                result_item = {
                    'event_time': event,
                    'event_index': event_idx,
                    'insulin_dose': window_data.loc[event, 'insulin'] if event in window_data.index else None,
                    'avg_effect': ci_results['avg_effect'],
                    'cum_effect': ci_results['cum_effect']
                }
                causalimpact_results.append(result_item)
                
                print(f"CausalImpact analysis successful:")
                print(f"  Average effect: {ci_results['avg_effect']:.2f}")
                print(f"  Cumulative effect: {ci_results['cum_effect']:.2f}")
            
            except Exception as e:
                print(f"Error in CausalImpact analysis: {type(e).__name__}: {str(e)}")
            
            # Run statsmodels ITS analysis
            try:
                its_results = run_its_statsmodel(window_data, event)
                
                # Save plot
                if output_dir:
                    plot_path = os.path.join(output_dir, "statsmodels", f"event_{event_idx}_its.png")
                    plot_its_statsmodel_result(its_results, event, save_path=plot_path)
                
                # Store results
                result_item = {
                    'event_time': event,
                    'event_index': event_idx,
                    'insulin_dose': window_data.loc[event, 'insulin'] if event in window_data.index else None,
                    'level_change': its_results['level_change'],
                    'slope_change': its_results['slope_change'],
                    'pvalue_level': its_results['pvalue_level'],
                    'pvalue_slope': its_results['pvalue_slope']
                }
                statsmodels_results.append(result_item)
                
                print(f"Statsmodels ITS analysis successful:")
                print(f"  Level change: {its_results['level_change']:.2f}")
                print(f"  Slope change: {its_results['slope_change']:.4f}")
            
            except Exception as e:
                print(f"Error in statsmodels ITS analysis: {type(e).__name__}: {str(e)}")
        
        except Exception as e:
            print(f"Error processing event: {type(e).__name__}: {str(e)}")
    
    # Create summary dataframes
    ci_df = pd.DataFrame(causalimpact_results) if causalimpact_results else pd.DataFrame()
    its_df = pd.DataFrame(statsmodels_results) if statsmodels_results else pd.DataFrame()
    
    # Print summary
    if not ci_df.empty:
        print("\n===== CausalImpact Results Summary =====")
        print(ci_df)
    
    if not its_df.empty:
        print("\n===== Statsmodels ITS Results Summary =====")
        print(its_df)
    
    # Save results to CSV if output directory provided
    if output_dir and not ci_df.empty:
        ci_df.to_csv(os.path.join(output_dir, "causalimpact_results.csv"))
    
    if output_dir and not its_df.empty:
        its_df.to_csv(os.path.join(output_dir, "statsmodels_results.csv"))
    
    return {
        'causalimpact': ci_df,
        'statsmodels': its_df
    }


def run_counterfactual_analysis(data_path, output_dir, max_events=5, pre_window='45min', post_window='30min'):
    """
    Run analysis on counterfactual datasets
    """
    print("\n===== Counterfactual Analysis =====\n")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(os.path.join(output_dir, "causalimpact"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "statsmodels"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "counterfactual"), exist_ok=True)
    
    # Load counterfactual datasets
    datasets = load_counterfactual_datasets(data_path)
    
    if not datasets:
        print("No datasets found. Analysis cannot proceed.")
        return None
    
    # Use baseline dataset to identify events
    baseline_key = "1.0"
    if baseline_key not in datasets:
        print(f"Warning: Baseline dataset (factor {baseline_key}) not found")
        baseline_key = list(datasets.keys())[0]
        print(f"Using {baseline_key} as baseline")
    
    baseline_df = datasets[baseline_key]
    
    # Plot counterfactual comparison
    if output_dir:
        plot_path = os.path.join(output_dir, "counterfactual", "counterfactual_comparison.png")
        plot_counterfactual_comparison(datasets, save_path=plot_path)
    
    # Detect events
    events = detect_events(baseline_df, max_events=max_events)
    
    if not events:
        print("No events detected. Analysis cannot proceed.")
        return None
    
    # Store results for each factor
    causalimpact_results = {}
    statsmodels_results = {}
    
    # Process each factor
    for factor, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Analyzing insulin factor: {factor}")
        
        ci_factor_results = []
        sm_factor_results = []
        
        # Process each event
        for event_idx, event in enumerate(events):
            print(f"\nProcessing event {event_idx+1}/{len(events)} at {event}")
            
            try:
                # Extract window data
                window_data = prepare_window_data(data, event, pre_window, post_window)
                
                if window_data.empty:
                    print(f"Skipping event - no data in window")
                    continue
                
                # Define pre/post periods for CausalImpact
                closest_pre = window_data.index[window_data.index <= event].max()
                closest_post = window_data.index[window_data.index > event].min()
                
                if pd.isna(closest_pre) or pd.isna(closest_post):
                    print(f"Skipping event - cannot establish pre/post boundaries")
                    continue
                
                pre_period = [window_data.index.min(), closest_pre]
                post_period = [closest_post, window_data.index.max()]
                
                # Run CausalImpact analysis
                try:
                    # Clean data
                    ci_data = clean_data_for_causalimpact(window_data)
                    
                    # Run the model
                    ci_results = run_causalimpact(ci_data, pre_period, post_period, event)
                    
                    # Save plot
                    if output_dir:
                        plot_path = os.path.join(output_dir, "causalimpact", f"factor_{factor}_event_{event_idx}_impact.png")
                        plot_causalimpact_result(ci_results['impact'], save_path=plot_path)
                    
                    # Store results
                    result_item = {
                        'event_time': event,
                        'event_index': event_idx,
                        'insulin_factor': factor,
                        'insulin_dose': window_data.loc[event, 'insulin'] if event in window_data.index else None,
                        'avg_effect': ci_results['avg_effect'],
                        'cum_effect': ci_results['cum_effect']
                    }
                    ci_factor_results.append(result_item)
                    
                    print(f"CausalImpact analysis successful:")
                    print(f"  Average effect: {ci_results['avg_effect']:.2f}")
                    print(f"  Cumulative effect: {ci_results['cum_effect']:.2f}")
                
                except Exception as e:
                    print(f"Error in CausalImpact analysis: {type(e).__name__}: {str(e)}")
                
                # Run statsmodels ITS analysis
                try:
                    its_results = run_its_statsmodel(window_data, event)
                    
                    # Save plot
                    if output_dir:
                        plot_path = os.path.join(output_dir, "statsmodels", f"factor_{factor}_event_{event_idx}_its.png")
                        plot_its_statsmodel_result(its_results, event, save_path=plot_path)
                    
                    # Store results
                    result_item = {
                        'event_time': event,
                        'event_index': event_idx,
                        'insulin_factor': factor,
                        'insulin_dose': window_data.loc[event, 'insulin'] if event in window_data.index else None,
                        'level_change': its_results['level_change'],
                        'slope_change': its_results['slope_change'],
                        'pvalue_level': its_results['pvalue_level'],
                        'pvalue_slope': its_results['pvalue_slope']
                    }
                    sm_factor_results.append(result_item)
                    
                    print(f"Statsmodels ITS analysis successful:")
                    print(f"  Level change: {its_results['level_change']:.2f}")
                    print(f"  Slope change: {its_results['slope_change']:.4f}")
                
                except Exception as e:
                    print(f"Error in statsmodels ITS analysis: {type(e).__name__}: {str(e)}")
            
            except Exception as e:
                print(f"Error processing event: {type(e).__name__}: {str(e)}")
        
        # Store results for this factor
        if ci_factor_results:
            causalimpact_results[factor] = pd.DataFrame(ci_factor_results)
        
        if sm_factor_results:
            statsmodels_results[factor] = pd.DataFrame(sm_factor_results)
    
    # Create summary dataframes
    ci_summary = []
    sm_summary = []
    
    for factor in sorted(datasets.keys()):
        if factor in causalimpact_results:
            df = causalimpact_results[factor]
            ci_summary.append({
                'insulin_factor': factor,
                'avg_effect_mean': df['avg_effect'].mean(),
                'avg_effect_std': df['avg_effect'].std(),
                'cum_effect_mean': df['cum_effect'].mean(),
                'cum_effect_std': df['cum_effect'].std(),
                'num_events': len(df)
            })
        
        if factor in statsmodels_results:
            df = statsmodels_results[factor]
            sm_summary.append({
                'insulin_factor': factor,
                'level_change_mean': df['level_change'].mean(),
                'level_change_std': df['level_change'].std(),
                'slope_change_mean': df['slope_change'].mean(),
                'slope_change_std': df['slope_change'].std(),
                'num_events': len(df)
            })
    
    ci_summary_df = pd.DataFrame(ci_summary) if ci_summary else pd.DataFrame()
    sm_summary_df = pd.DataFrame(sm_summary) if sm_summary else pd.DataFrame()
    
    # Print summary
    if not ci_summary_df.empty:
        print("\n===== CausalImpact Summary Across Factors =====")
        print(ci_summary_df)
    
    if not sm_summary_df.empty:
        print("\n===== Statsmodels ITS Summary Across Factors =====")
        print(sm_summary_df)
    
    # Plot comparison across factors
    if not ci_summary_df.empty and output_dir:
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            ci_summary_df['insulin_factor'].astype(float),
            ci_summary_df['avg_effect_mean'],
            yerr=ci_summary_df['avg_effect_std'],
            fmt='o-',
            capsize=5
        )
        plt.title('CausalImpact: Average Effect by Insulin Factor')
        plt.xlabel('Insulin Factor')
        plt.ylabel('Average Glucose Effect (mg/dL)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "counterfactual", "ci_factor_effect_comparison.png"))
        plt.close()
    
    if not sm_summary_df.empty and output_dir:
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            sm_summary_df['insulin_factor'].astype(float),
            sm_summary_df['level_change_mean'],
            yerr=sm_summary_df['level_change_std'],
            fmt='o-',
            capsize=5
        )
        plt.title('ITS: Level Change by Insulin Factor')
        plt.xlabel('Insulin Factor')
        plt.ylabel('Glucose Level Change (mg/dL)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "counterfactual", "its_factor_effect_comparison.png"))
        plt.close()
    
    # Save results to CSV if output directory provided
    if output_dir:
        # Save factor-specific results
        for factor, df in causalimpact_results.items():
            df.to_csv(os.path.join(output_dir, f"causalimpact_factor_{factor}_results.csv"))
        
        for factor, df in statsmodels_results.items():
            df.to_csv(os.path.join(output_dir, f"statsmodels_factor_{factor}_results.csv"))
        
        # Save summary results
        if not ci_summary_df.empty:
            ci_summary_df.to_csv(os.path.join(output_dir, "causalimpact_summary.csv"))
        
        if not sm_summary_df.empty:
            sm_summary_df.to_csv(os.path.join(output_dir, "statsmodels_summary.csv"))
    
    return {
        'causalimpact_results': causalimpact_results,
        'statsmodels_results': statsmodels_results,
        'causalimpact_summary': ci_summary_df,
        'statsmodels_summary': sm_summary_df
    }


def main():
    """
    Main function to parse args and run analysis
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run ITS models on time series data')
    
    # Analysis type
    parser.add_argument('--analysis', type=str, choices=['single', 'counterfactual'], required=True,
                      help='Type of analysis to run')
    
    # Data inputs
    parser.add_argument('--data-path', type=str, required=True,
                      help='Path to data file or directory with counterfactual datasets')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save outputs')
    
    # Analysis parameters
    parser.add_argument('--max-events', type=int, default=10,
                      help='Maximum number of events to analyze')
    parser.add_argument('--pre-window', type=str, default='45min',
                      help='Time window before event (e.g., 45min, 1h)')
    parser.add_argument('--post-window', type=str, default='30min',
                      help='Time window after event (e.g., 30min, 2h)')
    
    # Advanced options
    parser.add_argument('--debug', action='store_true',
                      help='Print debug information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected analysis
    if args.analysis == 'single':
        results = run_single_dataset_analysis(
            args.data_path, 
            args.output_dir,
            max_events=args.max_events,
            pre_window=args.pre_window,
            post_window=args.post_window
        )
    elif args.analysis == 'counterfactual':
        results = run_counterfactual_analysis(
            args.data_path,
            args.output_dir,
            max_events=args.max_events,
            pre_window=args.pre_window,
            post_window=args.post_window
        )
    
    # Save run information
    if args.output_dir:
        run_info = {
            'analysis_type': args.analysis,
            'data_path': args.data_path,
            'max_events': args.max_events,
            'pre_window': args.pre_window,
            'post_window': args.post_window,
            'run_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'success': results is not None
        }
        
        with open(os.path.join(args.output_dir, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2)


if __name__ == "__main__":
    main()
