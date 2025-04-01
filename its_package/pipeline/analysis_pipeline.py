import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

from ..its_models.causalimpact_model import CausalImpactModel
from ..its_models.statsmodels_its import StatsmodelsITSModel
from ..data_handling.data_loader import load_csv_data, load_counterfactual_datasets, prepare_window_data, clean_data_for_causalimpact
from ..data_handling.event_detection import detect_insulin_events, detect_meal_events
from ..visualization.plot_results import plot_event_data, plot_counterfactual_comparison, plot_comparison_across_factors
from ..utils.time_utils import to_timestamp

class CausalAnalysisPipeline:
    """Pipeline for causal analysis of time series data."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save outputs
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Initialize results storage
        self.events = []
        self.causal_impact_results = {}
        self.statsmodels_results = {}
        self.counterfactual_results = {}
        
    def run_single_dataset_analysis(self, data_path, max_events=5, 
                                  pre_window='45min', post_window='30min'):
        """
        Run analysis on a single dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to dataset file
        max_events : int
            Maximum number of events to analyze
        pre_window : str
            Time window before event
        post_window : str
            Time window after event
            
        Returns:
        --------
        dict
            Results of the analysis
        """
        print(f"Loading data from {data_path}")
        data = load_csv_data(data_path)
        
        print("Detecting insulin events")
        events = detect_insulin_events(data)
        print(f"Found {len(events)} insulin events")
        
        # Limit number of events
        events = events[:max_events]
        self.events = events
        
        results = []
        
        for i, event in enumerate(events):
            print(f"\nProcessing event {i+1}/{len(events)} at {event}")
            
            # Extract window data
            window_data = prepare_window_data(
                data, event, pre_window, post_window, ffill=True
            )
            
            if window_data.empty:
                print(f"Skipping event - no data in window")
                continue
                
            # Define pre/post periods
            closest_pre = window_data.index[window_data.index <= event].max()
            closest_post = window_data.index[window_data.index > event].min()
            
            if pd.isna(closest_pre) or pd.isna(closest_post):
                print(f"Skipping event - cannot establish pre/post boundaries")
                continue
                
            pre_period = [window_data.index.min(), closest_pre]
            post_period = [closest_post, window_data.index.max()]
            
            # Clean data for CausalImpact
            ci_data = clean_data_for_causalimpact(window_data)
            
            # Plot event data
            plot_name = f"event_{i}_data.png"
            plot_event_data(
                window_data, event, 'glucose', 
                output_dir=self.output_dir, 
                filename=plot_name,
                title=f"Event {i+1} at {event}"
            )
            
            # Run CausalImpact analysis
            try:
                ci_model = CausalImpactModel(output_dir=self.output_dir)
                ci_model.fit(ci_data, pre_period, post_period)
                ci_results = ci_model.get_results()
                
                # Save plot
                ci_model.save_plot(f"event_{i}_causalimpact.png")
                
                print(f"CausalImpact results:")
                print(f"  Average effect: {ci_results['avg_effect']:.2f}")
                print(f"  Cumulative effect: {ci_results['cum_effect']:.2f}")
                
                # Store results
                self.causal_impact_results[event] = ci_results
            except Exception as e:
                print(f"Error in CausalImpact analysis: {str(e)}")
                ci_results = None
            
            # Run Statsmodels ITS analysis
            try:
                sm_model = StatsmodelsITSModel(output_dir=self.output_dir)
                sm_model.fit(window_data, pre_period, post_period)
                sm_results = sm_model.get_results()
                
                # Save plot
                sm_model.save_plot(f"event_{i}_statsmodels.png")
                
                print(f"Statsmodels ITS results:")
                print(f"  Level change: {sm_results['level_change']:.2f}")
                print(f"  Slope change: {sm_results['slope_change']:.4f}")
                
                # Store results
                self.statsmodels_results[event] = sm_results
            except Exception as e:
                print(f"Error in Statsmodels ITS analysis: {str(e)}")
                sm_results = None
            
            # Combine results
            event_result = {
                'event_time': event,
                'insulin_dose': window_data.loc[event, 'insulin'] if event in window_data.index else None,
                'ci_results': ci_results,
                'sm_results': sm_results
            }
            
            results.append(event_result)
        
        return results
        
    def run_counterfactual_analysis(self, data_path, file_pattern="insulin_factor_*.csv",
                                  max_events=5, pre_window='45min', post_window='30min'):
        """
        Run analysis on counterfactual datasets.
        
        Parameters:
        -----------
        data_path : str
            Path to directory with counterfactual datasets
        file_pattern : str
            Pattern to match counterfactual files
        max_events : int
            Maximum number of events to analyze
        pre_window : str
            Time window before event
        post_window : str
            Time window after event
            
        Returns:
        --------
        dict
            Results of the analysis
        """
        print(f"Loading counterfactual datasets from {data_path}")
        datasets = load_counterfactual_datasets(data_path, file_pattern)
        print(f"Loaded {len(datasets)} counterfactual datasets")
        
        # Use baseline dataset to identify events
        baseline_key = "1.0"
        if baseline_key not in datasets:
            print(f"Warning: Baseline dataset (factor {baseline_key}) not found")
            baseline_key = list(datasets.keys())[0]
            print(f"Using {baseline_key} as baseline")
        
        baseline_df = datasets[baseline_key]
        
        print("Detecting insulin events in baseline dataset")
        events = detect_insulin_events(baseline_df)
        print(f"Found {len(events)} insulin events")
        
        # Limit number of events
        events = events[:max_events]
        self.events = events
        
        # Plot overall comparison
        plot_counterfactual_comparison(
            datasets, 
            output_dir=self.output_dir,
            filename="counterfactual_comparison.png",
            title="Comparison of Counterfactual Datasets"
        )
        
        factor_results = {}
        
        # Analyze each factor
        for factor, data in datasets.items():
            print(f"\n{'='*50}")
            print(f"Analyzing insulin factor: {factor}")
            
            event_results = []
            
            for i, event in enumerate(events):
                print(f"\nProcessing event {i+1}/{len(events)} at {event}")
                
                # Extract window data
                window_data = prepare_window_data(
                    data, event, pre_window, post_window, ffill=True
                )
                
                if window_data.empty:
                    print(f"Skipping event - no data in window")
                    continue
                    
                # Define pre/post periods
                closest_pre = window_data.index[window_data.index <= event].max()
                closest_post = window_data.index[window_data.index > event].min()
                
                if pd.isna(closest_pre) or pd.isna(closest_post):
                    print(f"Skipping event - cannot establish pre/post boundaries")
                    continue
                    
                pre_period = [window_data.index.min(), closest_pre]
                post_period = [closest_post, window_data.index.max()]
                
                # Clean data for CausalImpact
                ci_data = clean_data_for_causalimpact(window_data)
                
                # Plot event data
                plot_name = f"factor_{factor}_event_{i}_data.png"
                plot_event_data(
                    window_data, event, 'glucose', 
                    output_dir=self.output_dir, 
                    filename=plot_name,
                    title=f"Factor {factor}, Event {i+1} at {event}"
                )
                
                # Run CausalImpact analysis
                try:
                    ci_model = CausalImpactModel(output_dir=self.output_dir)
                    ci_model.fit(ci_data, pre_period, post_period)
                    ci_results = ci_model.get_results()
                    
                    # Save plot
                    ci_model.save_plot(f"factor_{factor}_event_{i}_causalimpact.png")
                    
                    # Store results
                    event_results.append({
                        'event_time': event,
                        'insulin_dose': window_data.loc[event, 'insulin'] if event in window_data.index else None,
                        'avg_effect': ci_results['avg_effect'],
                        'cum_effect': ci_results['cum_effect']
                    })
                    
                    print(f"Success: Average effect: {ci_results['avg_effect']:.2f}, Cumulative effect: {ci_results['cum_effect']:.2f}")
                except Exception as e:
                    print(f"Error in CausalImpact analysis: {str(e)}")
            
            # Store results for this factor
            if event_results:
                factor_results[factor] = pd.DataFrame(event_results)
                print(f"\nSummary for insulin factor {factor}:")
                print(factor_results[factor])
        
        # Store the results
        self.counterfactual_results = factor_results
        
        # Create summary dataframe
        summary = []
        for factor, df in factor_results.items():
            if not df.empty:
                summary.append({
                    'insulin_factor': factor,
                    'avg_effect_mean': df['avg_effect'].mean(),
                    'avg_effect_std': df['avg_effect'].std(),
                    'cum_effect_mean': df['cum_effect'].mean(),
                    'cum_effect_std': df['cum_effect'].std(),
                    'num_events': len(df)
                })
        
        summary_df = pd.DataFrame(summary)
        
        # Plot summary
        if not summary_df.empty:
            plot_comparison_across_factors(
                summary_df,
                title='Average Effect by Insulin Factor',
                output_dir=self.output_dir,
                filename="factor_effect_comparison.png"
            )
        
        return summary_df
        
    def save_results(self, filename=None):
        """
        Save results to file.
        
        Parameters:
        -----------
        filename : str
            Filename to save results to
        """
        if not self.output_dir:
            raise ValueError("Output directory not set")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"causal_analysis_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert results to serializable format
        results = {
            'events': [str(e) for e in self.events],
            'causal_impact_results': {str(k): v for k, v in self.causal_impact_results.items()},
            'statsmodels_results': {str(k): v for k, v in self.statsmodels_results.items()},
        }
        
        # Save counterfactual results if available
        if self.counterfactual_results:
            cf_results = {}
            for factor, df in self.counterfactual_results.items():
                cf_results[factor] = df.to_dict(orient='records')
            results['counterfactual_results'] = cf_results
        
        # Save to JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"Results saved to {filepath}")
