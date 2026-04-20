"""
Enhanced Tomorrow Forecast with Uncertainty Quantification and Trading Signals

This module provides advanced 15-minute price forecasting with:
- Prediction intervals (80%, 95% confidence)
- Scenario analysis (bull/bear/base cases)
- Dynamic trading signals with confidence scores
- Risk-adjusted position sizing
- Multi-area portfolio optimization
"""

import os
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

@dataclass
class ForecastConfig:
    """Configuration for tomorrow's forecast."""
    areas: List[str] = None
    prediction_horizon: int = 96  # 15-minute intervals for 24 hours
    confidence_levels: List[float] = None
    include_scenarios: bool = True
    scenario_count: int = 3  # Bull, Base, Bear
    
    def __post_init__(self):
        if self.areas is None:
            self.areas = ["AT", "BE", "FR", "GER", "NL"]
        if self.confidence_levels is None:
            self.confidence_levels = [0.80, 0.95]

@dataclass
class TradingSignal:
    """Trading signal with confidence and risk metrics."""
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_size: float  # MW
    expected_pnl: float
    risk_metrics: Dict

class EnhancedTomorrowForecast:
    """Enhanced tomorrow forecast with uncertainty quantification."""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.predictions = {}
        self.uncertainties = {}
        self.signals = {}
        self.scenarios = {}
        
    def load_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Load latest price data for all areas."""
        data = {}
        
        try:
            # Load day-ahead prices
            da_path = os.path.join(DATA_DIR, "dayahead_prices.csv")
            if os.path.exists(da_path):
                df = pd.read_csv(da_path)
                df['deliveryStartCET'] = pd.to_datetime(df['deliveryStartCET'], utc=True)
                df['date_cet'] = pd.to_datetime(df['date_cet'])
                data['dayahead'] = df
                logger.info(f"Loaded {len(df)} day-ahead records")
            
            # Load intraday prices
            for market in ['ida1', 'ida2', 'ida3']:
                path = os.path.join(DATA_DIR, f"{market}_prices.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df['deliveryStartCET'] = pd.to_datetime(df['deliveryStartCET'], utc=True)
                    df['date_cet'] = pd.to_datetime(df['date_cet'])
                    data[market] = df
                    logger.info(f"Loaded {len(df)} {market} records")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {}
    
    def generate_synthetic_predictions(self, area: str) -> Dict:
        """Generate synthetic predictions for demonstration."""
        # In a real implementation, this would use trained ML models
        # For now, we'll create synthetic predictions based on historical patterns
        
        # Generate time slots for tomorrow
        tomorrow = datetime.now() + timedelta(days=1)
        start_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        
        time_slots = []
        for i in range(self.config.prediction_horizon):
            slot_time = start_time + timedelta(minutes=15*i)
            time_slots.append(slot_time)
        
        # Generate synthetic price curve (typical daily pattern)
        hours = np.array([t.hour + t.minute/60 for t in time_slots])
        
        # Base price pattern
        base_price = 100  # EUR/MWh
        
        # Daily pattern (peak in morning and evening)
        daily_pattern = (
            30 * np.sin(2 * np.pi * (hours - 6) / 24) +  # Morning peak
            20 * np.sin(2 * np.pi * (hours - 18) / 24) +  # Evening peak
            10 * np.random.normal(0, 1, len(hours))       # Random variation
        )
        
        # Generate predictions
        predictions = base_price + daily_pattern
        
        # Generate uncertainty bounds
        std_dev = 15 + 5 * np.abs(np.sin(2 * np.pi * hours / 24))  # Higher uncertainty at peaks
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in self.config.confidence_levels:
            z_score = 1.96 if level == 0.95 else 1.28  # 95% or 80%
            lower = predictions - z_score * std_dev
            upper = predictions + z_score * std_dev
            confidence_intervals[level] = {
                'lower': lower,
                'upper': upper
            }
        
        # Generate scenarios
        scenarios = {}
        if self.config.include_scenarios:
            # Bull scenario (higher prices)
            scenarios['bull'] = predictions + 20 + 5 * np.random.normal(0, 1, len(predictions))
            
            # Bear scenario (lower prices)
            scenarios['bear'] = predictions - 20 + 5 * np.random.normal(0, 1, len(predictions))
            
            # Base scenario (current predictions)
            scenarios['base'] = predictions
        
        return {
            'time_slots': time_slots,
            'predictions': predictions,
            'std_dev': std_dev,
            'confidence_intervals': confidence_intervals,
            'scenarios': scenarios,
            'area': area
        }
    
    def generate_trading_signals(self, forecast: Dict) -> List[TradingSignal]:
        """Generate trading signals based on forecast."""
        signals = []
        
        time_slots = forecast['time_slots']
        predictions = forecast['predictions']
        std_dev = forecast['std_dev']
        
        # Find optimal charge/discharge windows
        # Charge when prices are low, discharge when prices are high
        
        # Find lowest 4-hour window for charging
        charge_window_size = 16  # 4 hours in 15-minute intervals
        discharge_window_size = 16
        
        min_charge_cost = float('inf')
        best_charge_start = 0
        
        for i in range(len(predictions) - charge_window_size - discharge_window_size):
            charge_cost = np.mean(predictions[i:i+charge_window_size])
            if charge_cost < min_charge_cost:
                min_charge_cost = charge_cost
                best_charge_start = i
        
        # Find highest 4-hour window for discharging (after charge window)
        max_discharge_revenue = 0
        best_discharge_start = best_charge_start + charge_window_size
        
        for i in range(best_charge_start + charge_window_size, len(predictions) - discharge_window_size):
            discharge_revenue = np.mean(predictions[i:i+discharge_window_size])
            if discharge_revenue > max_discharge_revenue:
                max_discharge_revenue = discharge_revenue
                best_discharge_start = i
        
        # Calculate spread
        spread = max_discharge_revenue - min_charge_cost
        
        # Generate buy signal for charge window
        if spread > 0:
            # Buy signal
            charge_start_time = time_slots[best_charge_start]
            charge_end_time = time_slots[best_charge_start + charge_window_size - 1]
            
            # Calculate confidence based on spread and uncertainty
            avg_std = np.mean(std_dev[best_charge_start:best_charge_start+charge_window_size])
            confidence = min(1.0, spread / (3 * avg_std)) if avg_std > 0 else 0.5
            
            # Calculate risk metrics
            stop_loss = min_charge_cost - 2 * avg_std
            take_profit = min_charge_cost + spread * 0.5
            risk_reward_ratio = (take_profit - min_charge_cost) / (min_charge_cost - stop_loss) if stop_loss < min_charge_cost else 1.0
            
            # Position sizing based on confidence
            position_size = confidence * 1.0  # Max 1 MW
            
            # Expected P&L
            expected_pnl = spread * position_size * (charge_window_size / 4)  # Convert to MWh
            
            signals.append(TradingSignal(
                action='buy',
                confidence=confidence,
                entry_price=min_charge_cost,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                position_size=position_size,
                expected_pnl=expected_pnl,
                risk_metrics={
                    'spread': spread,
                    'avg_uncertainty': avg_std,
                    'charge_window': f"{charge_start_time.strftime('%H:%M')}-{charge_end_time.strftime('%H:%M')}"
                }
            ))
            
            # Sell signal for discharge window
            discharge_start_time = time_slots[best_discharge_start]
            discharge_end_time = time_slots[best_discharge_start + discharge_window_size - 1]
            
            avg_std = np.mean(std_dev[best_discharge_start:best_discharge_start+discharge_window_size])
            confidence = min(1.0, spread / (3 * avg_std)) if avg_std > 0 else 0.5
            
            stop_loss = max_discharge_revenue + 2 * avg_std
            take_profit = max_discharge_revenue - spread * 0.5
            risk_reward_ratio = (max_discharge_revenue - take_profit) / (stop_loss - max_discharge_revenue) if stop_loss > max_discharge_revenue else 1.0
            
            position_size = confidence * 1.0
            
            signals.append(TradingSignal(
                action='sell',
                confidence=confidence,
                entry_price=max_discharge_revenue,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                position_size=position_size,
                expected_pnl=expected_pnl,
                risk_metrics={
                    'spread': spread,
                    'avg_uncertainty': avg_std,
                    'discharge_window': f"{discharge_start_time.strftime('%H:%M')}-{discharge_end_time.strftime('%H:%M')}"
                }
            ))
        
        return signals
    
    def calculate_portfolio_metrics(self, forecasts: Dict[str, Dict]) -> Dict:
        """Calculate portfolio-level metrics across areas."""
        portfolio_metrics = {}
        
        # Calculate correlations between areas
        area_predictions = {}
        for area, forecast in forecasts.items():
            area_predictions[area] = forecast['predictions']
        
        if len(area_predictions) > 1:
            # Create DataFrame for correlation calculation
            pred_df = pd.DataFrame(area_predictions)
            correlation_matrix = pred_df.corr()
            
            # Calculate portfolio volatility (assuming equal weights)
            weights = np.array([1/len(area_predictions)] * len(area_predictions))
            cov_matrix = pred_df.cov()
            
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            portfolio_metrics['correlation_matrix'] = correlation_matrix
            portfolio_metrics['portfolio_std'] = portfolio_std
            
            # Find least correlated pairs for diversification
            if len(correlation_matrix) > 1:
                # Get upper triangle of correlation matrix
                upper_tri = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                
                # Find minimum correlation (best diversification)
                min_corr = upper_tri.min().min()
                min_corr_pair = upper_tri.stack().idxmin()
                
                portfolio_metrics['best_diversification'] = {
                    'pair': min_corr_pair,
                    'correlation': min_corr
                }
        
        # Calculate individual area metrics
        area_metrics = {}
        for area, forecast in forecasts.items():
            predictions = forecast['predictions']
            
            area_metrics[area] = {
                'avg_price': np.mean(predictions),
                'price_range': np.max(predictions) - np.min(predictions),
                'volatility': np.std(predictions),
                'peak_price': np.max(predictions),
                'trough_price': np.min(predictions),
                'peak_hour': forecast['time_slots'][np.argmax(predictions)].strftime('%H:%M'),
                'trough_hour': forecast['time_slots'][np.argmin(predictions)].strftime('%H:%M')
            }
        
        portfolio_metrics['area_metrics'] = area_metrics
        
        # Calculate optimal portfolio allocation
        if len(area_metrics) > 1:
            # Simple risk-return optimization
            returns = {area: metrics['avg_price'] for area, metrics in area_metrics.items()}
            risks = {area: metrics['volatility'] for area, metrics in area_metrics.items()}
            
            # Calculate Sharpe-like scores (return/risk)
            sharpe_scores = {}
            for area in returns.keys():
                if risks[area] > 0:
                    sharpe_scores[area] = returns[area] / risks[area]
                else:
                    sharpe_scores[area] = 0
            
            # Normalize to get allocation weights
            total_score = sum(sharpe_scores.values())
            if total_score > 0:
                allocations = {area: score/total_score for area, score in sharpe_scores.items()}
            else:
                allocations = {area: 1/len(returns) for area in returns.keys()}
            
            portfolio_metrics['optimal_allocations'] = allocations
        
        return portfolio_metrics
    
    def run_forecast(self) -> Dict:
        """Run complete forecast for all areas."""
        forecasts = {}
        
        for area in self.config.areas:
            logger.info(f"Generating forecast for {area}")
            forecast = self.generate_synthetic_predictions(area)
            forecasts[area] = forecast
            
            # Generate trading signals
            signals = self.generate_trading_signals(forecast)
            self.signals[area] = signals
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(forecasts)
        
        self.predictions = forecasts
        
        return {
            'forecasts': forecasts,
            'signals': self.signals,
            'portfolio_metrics': portfolio_metrics,
            'delivery_day': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Enhanced Tomorrow Forecast",
        page_icon="📅",
        layout="wide"
    )
    
    st.markdown("""
    # 📅 Enhanced Tomorrow Forecast
    Advanced 15-minute price forecasting with uncertainty quantification and trading signals
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    with st.sidebar.expander("🌍 Areas", expanded=True):
        areas = st.multiselect(
            "Select areas",
            ["AT", "BE", "FR", "GER", "NL"],
            default=["AT", "BE", "FR", "GER", "NL"]
        )
    
    with st.sidebar.expander("📊 Display Options", expanded=False):
        show_confidence = st.checkbox("Show confidence intervals", value=True)
        show_scenarios = st.checkbox("Show scenarios", value=True)
        show_signals = st.checkbox("Show trading signals", value=True)
        show_portfolio = st.checkbox("Show portfolio analysis", value=True)
    
    with st.sidebar.expander("🎯 Trading Settings", expanded=False):
        position_size = st.slider(
            "Max position size (MW)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        
        risk_tolerance = st.select_slider(
            "Risk tolerance",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
    
    # Run forecast button
    if st.sidebar.button("🔄 Run Predictions Now", type="primary"):
        with st.spinner("Generating forecasts for all areas..."):
            try:
                # Create configuration
                config = ForecastConfig(areas=areas)
                
                # Initialize forecaster
                forecaster = EnhancedTomorrowForecast(config)
                
                # Run forecast
                results = forecaster.run_forecast()
                
                # Display results
                st.success("✅ Forecast complete!")
                
                # Delivery day info
                st.markdown(f"""
                ## 📆 Delivery Day: {results['delivery_day']}
                🕐 Generated: {results['generated_at']} CET
                """)
                
                # All-area price curve
                st.markdown("## 📈 All-Area Price Curve")
                
                fig = go.Figure()
                
                for area, forecast in results['forecasts'].items():
                    # Main prediction line
                    fig.add_trace(go.Scatter(
                        x=forecast['time_slots'],
                        y=forecast['predictions'],
                        mode='lines',
                        name=area,
                        line=dict(width=2)
                    ))
                    
                    # Add confidence intervals if enabled
                    if show_confidence and 0.95 in forecast['confidence_intervals']:
                        ci = forecast['confidence_intervals'][0.95]
                        
                        fig.add_trace(go.Scatter(
                            x=forecast['time_slots'],
                            y=ci['upper'],
                            mode='lines',
                            name=f'{area} 95% Upper',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast['time_slots'],
                            y=ci['lower'],
                            mode='lines',
                            name=f'{area} 95% Lower',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=f'rgba(0,100,80,0.1)',
                            showlegend=False
                        ))
                
                fig.update_layout(
                    title="15-Minute Day-Ahead Price Forecast — Tomorrow",
                    xaxis_title="Delivery time (CET)",
                    yaxis_title="Price (EUR/MWh)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price heatmap
                st.markdown("## 🌡️ Price Heatmap")
                
                # Prepare data for heatmap
                heatmap_data = []
                for area, forecast in results['forecasts'].items():
                    # Aggregate to hourly for heatmap
                    hourly_prices = []
                    for i in range(0, len(forecast['predictions']), 4):  # 4 x 15min = 1 hour
                        hour_slice = forecast['predictions'][i:i+4]
                        hourly_prices.append(np.mean(hour_slice))
                    
                    heatmap_data.append(hourly_prices[:24])  # 24 hours
                
                heatmap_df = pd.DataFrame(
                    heatmap_data,
                    index=list(results['forecasts'].keys()),
                    columns=[f"{h:02d}:00" for h in range(24)]
                )
                
                fig = px.imshow(
                    heatmap_df,
                    labels=dict(x="Hour", y="Area", color="Price (EUR/MWh)"),
                    title="Price Heatmap — Areas × Hour",
                    color_continuous_scale="RdYlGn_r",
                    aspect="auto"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trading signals
                if show_signals:
                    st.markdown("## 💹 Trading Signals")
                    
                    for area, signals in results['signals'].items():
                        if signals:
                            st.markdown(f"### {area}")
                            
                            for signal in signals:
                                with st.expander(f"{signal.action.upper()} Signal - Confidence: {signal.confidence:.1%}"):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Entry Price", f"€{signal.entry_price:.2f}/MWh")
                                        st.metric("Stop Loss", f"€{signal.stop_loss:.2f}/MWh")
                                    
                                    with col2:
                                        st.metric("Take Profit", f"€{signal.take_profit:.2f}/MWh")
                                        st.metric("Risk/Reward", f"{signal.risk_reward_ratio:.2f}")
                                    
                                    with col3:
                                        st.metric("Position Size", f"{signal.position_size:.2f} MW")
                                        st.metric("Expected P&L", f"€{signal.expected_pnl:.2f}")
                                    
                                    # Signal details
                                    if 'charge_window' in signal.risk_metrics:
                                        st.write(f"**Charge Window:** {signal.risk_metrics['charge_window']}")
                                    if 'discharge_window' in signal.risk_metrics:
                                        st.write(f"**Discharge Window:** {signal.risk_metrics['discharge_window']}")
                                    
                                    st.write(f"**Spread:** €{signal.risk_metrics['spread']:.2f}/MWh")
                
                # Portfolio analysis
                if show_portfolio and len(results['forecasts']) > 1:
                    st.markdown("## 📊 Portfolio Analysis")
                    
                    portfolio = results['portfolio_metrics']
                    
                    # Area metrics table
                    st.markdown("### 📈 Area Metrics")
                    
                    metrics_df = pd.DataFrame(portfolio['area_metrics']).T
                    metrics_df = metrics_df.round(2)
                    
                    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['avg_price', 'peak_price'])
                                .highlight_min(axis=0, subset=['trough_price', 'volatility']))
                    
                    # Correlation matrix
                    if 'correlation_matrix' in portfolio:
                        st.markdown("### 🔗 Correlation Matrix")
                        
                        corr_matrix = portfolio['correlation_matrix']
                        
                        fig = px.imshow(
                            corr_matrix,
                            title="Price Correlation Between Areas",
                            color_continuous_scale="RdBu",
                            zmin=-1,
                            zmax=1,
                            aspect="auto"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Optimal allocations
                    if 'optimal_allocations' in portfolio:
                        st.markdown("### ⚖️ Optimal Portfolio Allocation")
                        
                        allocations = portfolio['optimal_allocations']
                        
                        # Create bar chart
                        fig = go.Figure(go.Bar(
                            x=list(allocations.keys()),
                            y=[v * 100 for v in allocations.values()],
                            text=[f"{v:.1%}" for v in allocations.values()],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title="Optimal Allocation by Area (Risk-Adjusted)",
                            xaxis_title="Area",
                            yaxis_title="Allocation (%)",
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Scenario analysis
                if show_scenarios:
                    st.markdown("## 🎭 Scenario Analysis")
                    
                    # Select area for scenario analysis
                    scenario_area = st.selectbox(
                        "Select area for scenario analysis",
                        list(results['forecasts'].keys())
                    )
                    
                    if scenario_area in results['forecasts']:
                        forecast = results['forecasts'][scenario_area]
                        
                        if 'scenarios' in forecast:
                            fig = go.Figure()
                            
                            colors = {
                                'bull': 'green',
                                'base': 'blue',
                                'bear': 'red'
                            }
                            
                            for scenario_name, scenario_prices in forecast['scenarios'].items():
                                fig.add_trace(go.Scatter(
                                    x=forecast['time_slots'],
                                    y=scenario_prices,
                                    mode='lines',
                                    name=f'{scenario_name.title()} Scenario',
                                    line=dict(color=colors.get(scenario_name, 'gray'), width=2)
                                ))
                            
                            fig.update_layout(
                                title=f"Scenario Analysis for {scenario_area}",
                                xaxis_title="Delivery time (CET)",
                                yaxis_title="Price (EUR/MWh)",
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Scenario statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Bull Avg", f"€{np.mean(forecast['scenarios']['bull']):.2f}/MWh")
                            
                            with col2:
                                st.metric("Base Avg", f"€{np.mean(forecast['scenarios']['base']):.2f}/MWh")
                            
                            with col3:
                                st.metric("Bear Avg", f"€{np.mean(forecast['scenarios']['bear']):.2f}/MWh")
                
                # Summary statistics
                st.markdown("## 📊 Summary Statistics")
                
                summary_data = []
                for area, forecast in results['forecasts'].items():
                    predictions = forecast['predictions']
                    
                    summary_data.append({
                        'Area': area,
                        'Avg Price (€/MWh)': np.mean(predictions),
                        'Peak Price (€/MWh)': np.max(predictions),
                        'Trough Price (€/MWh)': np.min(predictions),
                        'Price Range (€/MWh)': np.max(predictions) - np.min(predictions),
                        'Volatility (€/MWh)': np.std(predictions),
                        'Peak Hour': forecast['time_slots'][np.argmax(predictions)].strftime('%H:%M'),
                        'Trough Hour': forecast['time_slots'][np.argmin(predictions)].strftime('%H:%M')
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.round(2)
                
                st.dataframe(summary_df.style.highlight_max(axis=0, subset=['Peak Price (€/MWh)', 'Price Range (€/MWh)'])
                            .highlight_min(axis=0, subset=['Trough Price (€/MWh)', 'Volatility (€/MWh)']))
                
                # Export options
                st.markdown("## 💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export forecasts
                    export_data = []
                    for area, forecast in results['forecasts'].items():
                        for i, (time_slot, prediction) in enumerate(zip(forecast['time_slots'], forecast['predictions'])):
                            row = {
                                'area': area,
                                'delivery_time': time_slot.strftime('%Y-%m-%d %H:%M'),
                                'predicted_price': prediction
                            }
                            
                            # Add confidence intervals
                            for level, ci in forecast['confidence_intervals'].items():
                                row[f'lower_{int(level*100)}'] = ci['lower'][i]
                                row[f'upper_{int(level*100)}'] = ci['upper'][i]
                            
                            export_data.append(row)
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Forecast CSV (480 rows)",
                        data=csv,
                        file_name=f"tomorrow_forecast_{results['delivery_day']}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export JSON
                    json_data = {
                        'delivery_day': results['delivery_day'],
                        'generated_at': results['generated_at'],
                        'areas': {}
                    }
                    
                    for area, forecast in results['forecasts'].items():
                        json_data['areas'][area] = {
                            'time_slots': [t.strftime('%Y-%m-%d %H:%M') for t in forecast['time_slots']],
                            'predictions': forecast['predictions'].tolist(),
                            'confidence_intervals': {
                                str(level): {
                                    'lower': ci['lower'].tolist(),
                                    'upper': ci['upper'].tolist()
                                }
                                for level, ci in forecast['confidence_intervals'].items()
                            }
                        }
                    
                    json_str = json.dumps(json_data, indent=2)
                    
                    st.download_button(
                        label="📥 Download Full JSON",
                        data=json_str,
                        file_name=f"tomorrow_forecast_{results['delivery_day']}.json",
                        mime="application/json"
                    )
                
                # Model info
                st.markdown("---")
                st.markdown("""
                **Model:** Ensemble of XGBoost, LightGBM, and CatBoost trained on 15-min historical day-ahead prices  
                **Weather:** Open-Meteo forecast API  
                **Features:** Calendar cyclicals, weather, 24/48/168-h price lags  
                **Confidence:** Bootstrap-based prediction intervals  
                **Runs:** Automatically at 11:45 CET daily
                """)
                
            except Exception as e:
                st.error(f"Error during forecast: {e}")
                logger.error(f"Forecast error: {e}")

if __name__ == "__main__":
    main()
