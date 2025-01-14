import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os

class HealthDataVisualizer:
    def __init__(self, data_path="merged_health_data.csv"):
        print("Loading data...")
        self.df = pd.read_csv(data_path, low_memory=False)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Debug information
        print("\nAvailable columns:")
        print(self.df.columns.tolist())
        print("\nFirst few rows:")
        print(self.df.head())
        
    def plot_bgl_time_series(self, days=7):
        """Plot blood glucose time series for the specified number of days."""
        print(f"\nPlotting {days}-day blood glucose time series...")
        
        # Get the last N days of data
        end_date = self.df['timestamp'].max()
        start_date = end_date - timedelta(days=days)
        mask = (self.df['timestamp'] >= start_date) & (self.df['timestamp'] <= end_date)
        plot_df = self.df[mask].copy()
        
        fig = go.Figure()
        
        # Plot CGM readings
        readings = plot_df[plot_df['__typename'] == 'Reading']
        if not readings.empty:
            fig.add_trace(go.Scatter(
                x=readings['timestamp'],
                y=readings['bgl'],
                mode='lines+markers',
                name='CGM Readings',
                line=dict(color='blue', width=1),
                marker=dict(size=4),
                hovertemplate="<br>".join([
                    "Time: %{x}",
                    "BGL: %{y} mg/dL",
                    "Trend: %{customdata}"
                ]),
                customdata=readings['trend']
            ))
        
        # Plot events (meals, insulin doses, etc.)
        events = plot_df[plot_df['__typename'] == 'Message']
        if not events.empty:
            fig.add_trace(go.Scatter(
                x=events['timestamp'],
                y=events['bgl'],
                mode='markers',
                name='Events',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='star'
                ),
                hovertemplate="<br>".join([
                    "Time: %{x}",
                    "BGL: %{y} mg/dL",
                    "Event: %{customdata}"
                ]),
                customdata=events['text']
            ))
        
        # Add target range
        fig.add_hrect(
            y0=70, y1=180,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            name="Target Range"
        )
        
        # Add threshold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, name="Low Threshold")
        fig.add_hline(y=180, line_dash="dash", line_color="red", opacity=0.5, name="High Threshold")
        
        fig.update_layout(
            title=f'Blood Glucose Readings and Events Over {days} Days',
            xaxis_title='Time',
            yaxis_title='Blood Glucose (mg/dL)',
            showlegend=True,
            height=600,
            hovermode='closest'
        )
        
        os.makedirs('visualizations', exist_ok=True)
        fig.write_html(f'visualizations/bgl_time_series_{days}days.html')

    def plot_value_distribution(self):
        """Plot distribution of blood glucose values."""
        print("\nPlotting blood glucose value distribution...")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.df['bgl'].dropna(),
            nbinsx=50,
            name='Blood Glucose Values'
        ))
        
        # Add threshold lines
        fig.add_vline(x=70, line_dash="dash", line_color="red", opacity=0.5, name="Low Threshold")
        fig.add_vline(x=180, line_dash="dash", line_color="red", opacity=0.5, name="High Threshold")
        
        fig.update_layout(
            title='Distribution of Blood Glucose Values',
            xaxis_title='Blood Glucose (mg/dL)',
            yaxis_title='Frequency',
            showlegend=True,
            height=600
        )
        
        os.makedirs('visualizations', exist_ok=True)
        fig.write_html('visualizations/bgl_distribution.html')
        
    def generate_all_plots(self):
        """Generate all visualization plots."""
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Generate all plots
        self.plot_bgl_time_series(days=7)  # Week view
        self.plot_bgl_time_series(days=1)  # Day view
        self.plot_value_distribution()
        
        print("\nAll visualizations have been generated in the 'visualizations' directory.")

def main():
    visualizer = HealthDataVisualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main() 