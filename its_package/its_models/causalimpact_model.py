import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalimpact import CausalImpact
from .base import BaseITSModel
import os

class CausalImpactModel(BaseITSModel):
    """ITS model using CausalImpact."""
    
    def __init__(self, output_dir=None):
        """Initialize the CausalImpact model."""
        super().__init__(name="CausalImpactModel", output_dir=output_dir)
        self.impact = None
        self.data = None
        self.pre_period = None
        self.post_period = None
        
    def fit(self, data, pre_period, post_period):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to fit the model to
        pre_period : list
            [start, end] of pre-intervention period
        post_period : list
            [start, end] of post-intervention period
        prior_level_sd : float
            Standard deviation of the prior level
            
        Returns:
        --------
        self
        """
        # Handle edge cases first
        if data.empty:
            raise ValueError("Empty dataset provided")
        
        # Ensure we have at least one predictor variable
        if data.shape[1] <= 1:
            raise ValueError("CausalImpact requires at least one covariate/predictor variable")
        
        # Clean data - replace infinities, drop NAs, and ensure column variability
        clean_data = data.replace([np.inf, -np.inf], np.nan)
        clean_data = clean_data.dropna()
        
        # Keep only columns with variation (more than one unique value)
        clean_data = clean_data.loc[:, clean_data.nunique() > 1]
        
        # Check if we still have enough data
        if clean_data.empty or clean_data.shape[1] <= 1:
            raise ValueError("After cleaning, not enough data or variables remain")
        
        # Handle columns with constant values in pre-period
        pre_data = clean_data.loc[pre_period[0]:pre_period[1]].copy()
        for col in pre_data.columns:
            if pre_data[col].nunique() == 1:
                constant_val = pre_data[col].iloc[0]
                # Add a small variation to avoid constant values
                clean_data.loc[pre_period[1], col] = constant_val + 1
        
        # Store processed data
        self.data = clean_data
        self.pre_period = pre_period
        self.post_period = post_period
        
        # Fit the model
        self.impact = CausalImpact(clean_data, pre_period, post_period)
        self.impact.run()
        
        return self
    
    def get_results(self):
        """
        Get the results of the model.
        
        Returns:
        --------
        dict
            Dictionary of results
        """
        if self.impact is None:
            raise ValueError("Model not fitted yet")
        
        # Extract key results
        post_inferences = self.impact.inferences.loc[self.impact.inferences.index >= self.post_period[0]]
        
        results = {
            'avg_effect': post_inferences['point_effects'].mean(),
            'cum_effect': post_inferences['post_cum_effects'].iloc[-1],
            'post_pred_mean': post_inferences['point_pred'].mean(),
            'post_pred_lower': post_inferences['point_pred_lower'].mean(),
            'post_pred_upper': post_inferences['point_pred_upper'].mean(),
            'significance': self.impact.summary_data.get('p', None),
            'model_summary': self.impact.summary()
        }
        
        return results
    
    def plot(self, fig=None, figsize=(12, 8)):
        """
        Plot the results of the model.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to plot on
        figsize : tuple
            Size of the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with plot
        """
        if self.impact is None:
            raise ValueError("Model not fitted yet")
        
        # Generate the plot
        if fig is None:
            fig = plt.figure(figsize=figsize)
        
        self.impact.plot(fig=fig)
        
        return fig
    
    def plot_original_data(self, target_col='glucose', figsize=(12, 6)):
        """
        Plot the original data with intervention time.
        
        Parameters:
        -----------
        target_col : str
            Name of the target column to plot
        figsize : tuple
            Size of the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with plot
        """
        if self.data is None:
            raise ValueError("Model not fitted yet")
        
        fig = plt.figure(figsize=figsize)
        
        # Plot target variable
        plt.plot(self.data.index, self.data[target_col], 'b-', label=target_col)
        
        # Mark intervention time
        intervention_time = self.pre_period[1]
        plt.axvline(x=intervention_time, color='r', linestyle='--', label='Intervention')
        
        plt.title(f'Original Data with Intervention at {intervention_time}')
        plt.ylabel(target_col)
        plt.legend()
        
        return fig
