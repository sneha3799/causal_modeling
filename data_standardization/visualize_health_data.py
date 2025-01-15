import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os
import json

class HealthDataVisualizer:
    def __init__(self, data_path="output/merged_health_data.csv"):
        print("Loading data...")
        self.df = pd.read_csv(data_path, low_memory=False)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Debug information
        print("\nAvailable columns:")
        print(self.df.columns.tolist())
        print("\nFirst few rows:")
        print(self.df.head())
        
    def plot_bgl_time_series(self):
        """Plot interactive blood glucose time series with hover data."""
        print("Plotting interactive blood glucose time series...")
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add main BGL trace (keep original scale)
        readings = self.df[self.df['bgl'].notna()].copy()
        fig.add_trace(go.Scatter(
            x=readings['timestamp'],
            y=readings['bgl'],
        mode='lines',
        name='Blood Glucose',
        line=dict(color='blue', width=2),
        hovertemplate='<b>BGL:</b> %{y:.1f} mg/dL<br><b>Time:</b> %{x}',
        legendgroup='glucose',
        legendgrouptitle_text="Glucose"
        ))
        
        # Add BGL events (keep original scale)
        events = readings[readings['msg_type'].notna()]
        event_colors = {
            'DOSE_INSULIN': 'red',
            'DOSE_BASAL_INSULIN': 'darkred',
            'ANNOUNCE_MEAL': 'green',
            'INTERVENTION_SNACK': 'orange',
            'BGL_FP_READING': 'purple',
            'NEW_PEN': 'brown',
            'NEW_SENSOR': 'gray',
            'ANNOUNCE_EXERCISE': 'blue',
            'MEDICAL_TEST_RESULT': 'black'
        }
        
        # Create descriptions dictionary for each metric and event type
        descriptions = {
            'Blood Glucose': {
                'original_scale': 'mg/dL (70-180 target range)',
                'description': 'Continuous glucose monitor readings showing blood glucose levels.',
                'modifications': 'No scaling applied. Shown in original units.',
                'frequency': 'Every 5 minutes from CGM, or on-demand for finger prick readings.'
            },
            'DOSE_INSULIN': {
                'description': 'Regular insulin doses administered.',
                'modifications': 'Multiple doses at same timestamp are summed.',
                'frequency': 'Event-based (when insulin is administered).'
            },
            'DOSE_BASAL_INSULIN': {
                'description': 'Long-acting basal insulin doses.',
                'modifications': 'Multiple doses at same timestamp are summed.',
                'frequency': 'Typically once or twice daily.'
            },
            'ANNOUNCE_MEAL': {
                'description': 'Meal announcements with carbohydrate amounts.',
                'modifications': 'Multiple food amounts at same timestamp are summed.',
                'frequency': 'Event-based (when meals are consumed).'
            },
            'INTERVENTION_SNACK': {
                'description': 'Fast-acting carbohydrates for low blood sugar treatment.',
                'modifications': 'Multiple food amounts at same timestamp are summed.',
                'frequency': 'Event-based (when treating low blood sugar).'
            },
            'Sleep Score': {
                'original_scale': '0-100',
                'description': 'Overall sleep quality score.',
                'modifications': 'No scaling applied. Original 0-100 scale.',
                'frequency': 'Once daily (calculated after each sleep session).'
            },
            'Deep Sleep': {
                'original_scale': 'minutes (typically 0-500)',
                'description': 'Time spent in deep sleep phase.',
                'modifications': 'Scaled to 0-100 range (divided by 5) for visualization.',
                'frequency': 'Once daily (total minutes from sleep session).'
            },
            'REM Sleep': {
                'original_scale': 'minutes (typically 0-500)',
                'description': 'Time spent in REM sleep phase.',
                'modifications': 'Scaled to 0-100 range (divided by 5) for visualization.',
                'frequency': 'Once daily (total minutes from sleep session).'
            },
            'Stress Score': {
                'original_scale': '0-100',
                'description': 'Overall stress level based on various metrics.',
                'modifications': 'No scaling applied. Original 0-100 scale.',
                'frequency': 'Updated throughout the day based on HRV and activity.'
            },
            'HRV (RMSSD)': {
                'original_scale': 'milliseconds (typically 0-150)',
                'description': 'Heart Rate Variability - Root Mean Square of Successive Differences.',
                'modifications': 'Scaled to 0-100 range (multiplied by 100/150) for visualization.',
                'frequency': 'Measured during sleep, updated several times per night.'
            }
        }
        
        for msg_type in events['msg_type'].unique():
            if pd.isna(msg_type):
                continue
            
            type_events = events[events['msg_type'] == msg_type]
            color = event_colors.get(msg_type, 'gray')
            
            fig.add_trace(go.Scatter(
                x=type_events['timestamp'],
                y=type_events['bgl'],
                mode='markers',
                name=msg_type,
                marker=dict(size=8, color=color),
                customdata=type_events[['trend', 'text']].fillna('').values,
                hovertemplate=(
                    '<b>BGL:</b> %{y:.1f} mg/dL<br>' +
                    '<b>Time:</b> %{x}<br>' +
                    '<b>Trend:</b> %{customdata[0]}<br>' +
                    '<b>Note:</b> %{customdata[1]}'
                ),
                visible='legendonly',
                legendgroup='glucose',
                legendgrouptitle_text="Glucose"
            ))
        
        # Add Sleep Metrics (normalized to 0-100 scale)
        sleep_metrics = {
            'overall_score': {'color': 'purple', 'name': 'Sleep Score', 'scale': 1},  # Already 0-100
            'deep_sleep_in_minutes': {'color': 'darkblue', 'name': 'Deep Sleep', 'scale': 1/5},  # Typically 0-500 minutes
            'rem_sleep': {'color': 'lightblue', 'name': 'REM Sleep', 'scale': 1/5}  # Typically 0-500 minutes
        }
        
        for metric, props in sleep_metrics.items():
            metric_data = self.df[self.df[metric].notna()]
            if not metric_data.empty:
                original_values = metric_data[metric].values
                scaled_values = original_values * props['scale']
                
                fig.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=scaled_values,
                    mode='lines+markers',
                    name=props['name'],
                    line=dict(color=props['color']),
                    customdata=np.column_stack((original_values, scaled_values)),
                    hovertemplate=(
                        f'<b>{props["name"]}:</b><br>' +
                        'Original: %{customdata[0]:.1f}' + (' minutes' if 'sleep' in metric else '') + '<br>' +
                        'Normalized: %{customdata[1]:.1f}/100<br>' +
                        '<b>Time:</b> %{x}'
                    ),
                    visible='legendonly',
                    legendgroup='sleep',
                    legendgrouptitle_text="Sleep",
                    yaxis='y2'
                ))
        
        # Add Stress Score (already 0-100)
        stress_data = self.df[self.df['STRESS_SCORE'].notna()]
        if not stress_data.empty:
            fig.add_trace(go.Scatter(
                x=stress_data['timestamp'],
                y=stress_data['STRESS_SCORE'],
                mode='lines+markers',
                name='Stress Score',
                line=dict(color='red'),
                customdata=stress_data['STRESS_SCORE'],
                hovertemplate='<b>Stress Score:</b> %{customdata:.1f}/100<br><b>Time:</b> %{x}',
                visible='legendonly',
                legendgroup='stress',
                legendgrouptitle_text="Stress",
                yaxis='y2'
            ))
        
        # Add Heart Rate Variability (normalized to 0-100 scale)
        hrv_data = self.df[self.df['rmssd'].notna()]
        if not hrv_data.empty:
            original_hrv = hrv_data['rmssd'].values
            scaled_hrv = original_hrv * (100/150)  # Assuming typical range 0-150ms
            
            fig.add_trace(go.Scatter(
                x=hrv_data['timestamp'],
                y=scaled_hrv,
                mode='lines+markers',
                name='HRV (RMSSD)',
                line=dict(color='pink'),
                customdata=np.column_stack((original_hrv, scaled_hrv)),
                hovertemplate=(
                    '<b>HRV:</b><br>' +
                    'Original: %{customdata[0]:.1f} ms<br>' +
                    'Normalized: %{customdata[1]:.1f}/100<br>' +
                    '<b>Time:</b> %{x}'
                ),
                visible='legendonly',
                legendgroup='hrv',
                legendgrouptitle_text="Heart Rate",
                yaxis='y2'
            ))
        
        # Add target range for BGL
        fig.add_hrect(
            y0=70, y1=180,
            fillcolor="lightgreen", opacity=0.1,
            layer="below", line_width=0,
            name="Target Range (70-180)",
            legendgroup='glucose'
        )
        
        # Create description text with grid styling
        description_html = """
        <div style='margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;'>
            <h2 style='margin: 0 0 15px 0; color: #2c3e50; font-size: 1.2em;'>📊 Active Metrics</h2>
            <div id='metric-descriptions' style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px;'>
                <p><i>Select metrics in the legend above to view their details here.</i></p>
            </div>
        </div>
        """
        
        # Add JavaScript to update descriptions based on selected traces
        description_script = """
        <script>
            const descriptions = %s;
            const gd = document.getElementById('health-metrics');
            let lastUpdate = 0;
            
            function formatDescription(traceName, desc) {
                return `
                    <div class="metric-card" style="background: white; padding: 12px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="color: #2c3e50; font-weight: 600; font-size: 1.1em; margin-bottom: 8px;">
                            ${traceName}
                        </div>
                        ${desc.original_scale ? 
                            `<div style="font-size: 0.9em; color: #666; margin-bottom: 6px;">
                                <span style="color: #2c3e50; font-weight: 500;">Scale:</span> ${desc.original_scale}
                            </div>` : ''
                        }
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 6px;">
                            <span style="color: #2c3e50; font-weight: 500;">Description:</span> ${desc.description}
                        </div>
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 6px;">
                            <span style="color: #2c3e50; font-weight: 500;">Frequency:</span> ${desc.frequency}
                        </div>
                        <div style="font-size: 0.9em; color: #666;">
                            <span style="color: #2c3e50; font-weight: 500;">Processing:</span> ${desc.modifications}
                        </div>
                    </div>
                `.trim();
            }
            
            function updateDescriptions() {
                const now = Date.now();
                if (now - lastUpdate < 100) return; // Debounce updates
                lastUpdate = now;
                
                const activeTraces = gd.data
                    .filter(trace => trace.visible === true)
                    .map(trace => trace.name)
                    .filter(name => descriptions[name]);
                
                const descriptionDiv = document.getElementById('metric-descriptions');
                if (activeTraces.length === 0) {
                    descriptionDiv.innerHTML = '<p><i>Select metrics in the legend above to view their details here.</i></p>';
                    return;
                }
                
                const cards = activeTraces
                    .map(name => formatDescription(name, descriptions[name]))
                    .join('');
                
                descriptionDiv.innerHTML = cards;
            }
            
            // Initial update
            setTimeout(updateDescriptions, 1000);
            
            // Update on legend clicks
            gd.on('plotly_legendclick', updateDescriptions);
            gd.on('plotly_legenddoubleclick', updateDescriptions);
        </script>
        """ % json.dumps(descriptions)
        
        # Update layout with normalized secondary y-axis
        fig.update_layout(
            title='Health Metrics Over Time',
            xaxis_title='Time',
            yaxis_title='Blood Glucose (mg/dL)',
            yaxis2=dict(
                title='Normalized Scale (0-100)',
                overlaying='y',
                side='right',
                range=[0, 100],  # Fix range for normalized metrics
                showgrid=False
            ),
            hovermode='closest',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.15,
                bgcolor="rgba(255, 255, 255, 0.8)",
                groupclick="toggleitem"
            ),
            margin=dict(r=200),
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=12, label="12h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=3, label="3d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            )
        )
        
        # Save plot with descriptions
        os.makedirs('visualizations', exist_ok=True)
        html_content = fig.to_html(
            include_plotlyjs=True,
            full_html=True,
            div_id='health-metrics'
        )
        # Insert description section and script before </body>
        html_content = html_content.replace('</body>', description_html + description_script + '</body>')
        
        with open('visualizations/health_metrics.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("Interactive plot saved to visualizations/health_metrics.html")
        
    def generate_all_plots(self):
        """Generate all visualization plots."""
        os.makedirs('visualizations', exist_ok=True)
        
        # Generate interactive time series plot
        self.plot_bgl_time_series()
        
        print("\nAll visualizations have been generated in the 'visualizations' directory.")

def main():
    visualizer = HealthDataVisualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main() 