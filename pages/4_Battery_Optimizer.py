"""
Enhanced Battery Optimizer with Realistic Cost Modeling and Risk Management

This module provides advanced battery storage optimization with:
- Realistic round-trip efficiency modeling
- Battery degradation costs
- Transmission and curtailment costs
- Cross-border arbitrage optimization
- Risk management with VaR and stress testing
- Uncertainty quantification with Monte Carlo simulation
"""

import os
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

@dataclass
class BatteryConfig:
    """Battery configuration parameters."""
    capacity_mw: float = 1.0
    charge_hours: int = 4
    discharge_hours: int = 4
    max_cycles_per_day: int = 1
    round_trip_efficiency: float = 0.90  # 90% realistic efficiency
    degradation_cost_eur_mwh: float = 5.0  # €5/MWh discharged
    exchange_fee_eur_mwh: float = 0.5  # €0.5/MWh traded
    transmission_cost_eur_mwh: float = 2.0  # €2/MWh for cross-border
    maintenance_cost_eur_day: float = 10.0  # €10/day maintenance
    initial_investment_eur: float = 500000.0  # €500k initial investment
    battery_life_years: int = 10
    discount_rate: float = 0.08  # 8% discount rate

@dataclass
class MarketConfig:
    """Market configuration parameters."""
    areas: List[str] = None
    markets: List[str] = None
    train_split: float = 0.7
    start_date: str = "2026-01-17"
    end_date: str = "2026-04-18"
    
    def __post_init__(self):
        if self.areas is None:
            self.areas = ["AT", "BE", "FR", "GER", "NL"]
        if self.markets is None:
            self.markets = ["DayAhead", "IDA1"]

@dataclass
class RiskMetrics:
    """Risk metrics for strategy evaluation."""
    var_95: float = 0.0  # Value at Risk (95% confidence)
    cvar_95: float = 0.0  # Conditional Value at Risk
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

class EnhancedBatteryOptimizer:
    """Enhanced battery optimizer with realistic cost modeling and risk management."""
    
    def __init__(self, battery_config: BatteryConfig, market_config: MarketConfig):
        self.battery = battery_config
        self.market = market_config
        self.data = {}
        self.strategies = {}
        self.results = {}
        
    def load_data(self) -> bool:
        """Load price data from CSV files."""
        try:
            for market in self.market.markets:
                if market == "DayAhead":
                    filename = "dayahead_prices.csv"
                elif market == "IDA1":
                    filename = "ida1_prices.csv"
                elif market == "IDA2":
                    filename = "ida2_prices.csv"
                elif market == "IDA3":
                    filename = "ida3_prices.csv"
                else:
                    continue
                
                filepath = os.path.join(DATA_DIR, filename)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    df['deliveryStartCET'] = pd.to_datetime(df['deliveryStartCET'], utc=True)
                    df['date_cet'] = pd.to_datetime(df['date_cet'])
                    self.data[market] = df
                    logger.info(f"Loaded {len(df)} rows for {market}")
                else:
                    logger.warning(f"File not found: {filepath}")
            
            return len(self.data) > 0
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def calculate_realistic_costs(self, buy_price: float, sell_price: float, 
                                 volume_mwh: float, is_cross_border: bool = False) -> Dict:
        """Calculate realistic transaction costs."""
        # Round-trip efficiency loss
        efficiency_loss = volume_mwh * (1 - self.battery.round_trip_efficiency) * buy_price
        
        # Degradation cost
        degradation = volume_mwh * self.battery.degradation_cost_eur_mwh
        
        # Exchange fees
        exchange_fees = volume_mwh * self.battery.exchange_fee_eur_mwh * 2  # Buy and sell
        
        # Transmission costs (if cross-border)
        transmission = 0
        if is_cross_border:
            transmission = volume_mwh * self.battery.transmission_cost_eur_mwh
        
        # Total costs
        total_costs = efficiency_loss + degradation + exchange_fees + transmission
        
        # Gross profit
        gross_profit = (sell_price - buy_price) * volume_mwh
        
        # Net profit
        net_profit = gross_profit - total_costs
        
        return {
            'gross_profit': gross_profit,
            'efficiency_loss': efficiency_loss,
            'degradation_cost': degradation,
            'exchange_fees': exchange_fees,
            'transmission_cost': transmission,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'profit_margin': net_profit / gross_profit if gross_profit > 0 else 0
        }
    
    def optimize_single_area(self, area: str, market: str) -> Dict:
        """Optimize battery strategy for a single area and market."""
        if market not in self.data:
            return None
        
        df = self.data[market]
        area_data = df[df['area'] == area].copy()
        
        if area_data.empty:
            return None
        
        # Filter by date range
        area_data = area_data[
            (area_data['date_cet'] >= self.market.start_date) &
            (area_data['date_cet'] <= self.market.end_date)
        ]
        
        # Split into train/test
        split_idx = int(len(area_data) * self.market.train_split)
        train_data = area_data.iloc[:split_idx]
        test_data = area_data.iloc[split_idx:]
        
        # Get unique dates and hours
        train_dates = train_data['date_cet'].unique()
        
        best_strategy = None
        best_score = -np.inf
        
        # Grid search for optimal buy/sell hours
        for buy_start in range(24 - self.battery.charge_hours + 1):
            for sell_start in range(24 - self.battery.discharge_hours + 1):
                # Check if windows overlap
                buy_range = range(buy_start, buy_start + self.battery.charge_hours)
                sell_range = range(sell_start, sell_start + self.battery.discharge_hours)
                
                if set(buy_range) & set(sell_range):
                    continue
                
                # Calculate P&L for this strategy
                daily_pnl = []
                for date in train_dates:
                    day_data = train_data[train_data['date_cet'] == date]
                    
                    if len(day_data) < 24:
                        continue
                    
                    # Get buy and sell prices
                    buy_hours = [f"{h:02d}:00" for h in buy_range]
                    sell_hours = [f"{h:02d}:00" for h in sell_range]
                    
                    buy_prices = []
                    sell_prices = []
                    
                    for hour in buy_hours:
                        hour_data = day_data[day_data['deliveryStartCET'].dt.strftime('%H:%M') == hour]
                        if not hour_data.empty:
                            buy_prices.append(hour_data['price'].iloc[0])
                    
                    for hour in sell_hours:
                        hour_data = day_data[day_data['deliveryStartCET'].dt.strftime('%H:%M') == hour]
                        if not hour_data.empty:
                            sell_prices.append(hour_data['price'].iloc[0])
                    
                    if buy_prices and sell_prices:
                        avg_buy = np.mean(buy_prices)
                        avg_sell = np.mean(sell_prices)
                        
                        # Calculate with realistic costs
                        costs = self.calculate_realistic_costs(
                            avg_buy, avg_sell, 
                            self.battery.capacity_mw * self.battery.charge_hours
                        )
                        
                        daily_pnl.append(costs['net_profit'])
                
                if daily_pnl:
                    avg_pnl = np.mean(daily_pnl)
                    std_pnl = np.std(daily_pnl) if len(daily_pnl) > 1 else 1e-6
                    
                    # Sharpe-like score (risk-adjusted)
                    score = avg_pnl / std_pnl if std_pnl > 0 else avg_pnl
                    
                    if score > best_score:
                        best_score = score
                        best_strategy = {
                            'buy_start': buy_start,
                            'sell_start': sell_start,
                            'buy_hours': list(buy_range),
                            'sell_hours': list(sell_range),
                            'train_score': score,
                            'train_avg_pnl': avg_pnl,
                            'train_std_pnl': std_pnl,
                            'train_daily_pnl': daily_pnl
                        }
        
        if not best_strategy:
            return None
        
        # Evaluate on test data
        test_dates = test_data['date_cet'].unique()
        test_daily_pnl = []
        
        for date in test_dates:
            day_data = test_data[test_data['date_cet'] == date]
            
            if len(day_data) < 24:
                continue
            
            buy_range = best_strategy['buy_hours']
            sell_range = best_strategy['sell_hours']
            
            buy_hours = [f"{h:02d}:00" for h in buy_range]
            sell_hours = [f"{h:02d}:00" for h in sell_range]
            
            buy_prices = []
            sell_prices = []
            
            for hour in buy_hours:
                hour_data = day_data[day_data['deliveryStartCET'].dt.strftime('%H:%M') == hour]
                if not hour_data.empty:
                    buy_prices.append(hour_data['price'].iloc[0])
            
            for hour in sell_hours:
                hour_data = day_data[day_data['deliveryStartCET'].dt.strftime('%H:%M') == hour]
                if not hour_data.empty:
                    sell_prices.append(hour_data['price'].iloc[0])
            
            if buy_prices and sell_prices:
                avg_buy = np.mean(buy_prices)
                avg_sell = np.mean(sell_prices)
                
                costs = self.calculate_realistic_costs(
                    avg_buy, avg_sell, 
                    self.battery.capacity_mw * self.battery.charge_hours
                )
                
                test_daily_pnl.append(costs['net_profit'])
        
        # Calculate risk metrics
        all_daily_pnl = best_strategy['train_daily_pnl'] + test_daily_pnl
        risk_metrics = self.calculate_risk_metrics(all_daily_pnl)
        
        # Calculate annual P&L
        total_days = len(train_dates) + len(test_dates)
        annual_pnl = np.sum(all_daily_pnl) * (365 / total_days) if total_days > 0 else 0
        
        # Calculate NPV
        npv = self.calculate_npv(annual_pnl)
        
        return {
            'area': area,
            'market': market,
            'strategy': best_strategy,
            'train_metrics': {
                'days': len(train_dates),
                'total_pnl': np.sum(best_strategy['train_daily_pnl']),
                'avg_daily_pnl': best_strategy['train_avg_pnl'],
                'std_daily_pnl': best_strategy['train_std_pnl'],
                'sharpe': best_strategy['train_score']
            },
            'test_metrics': {
                'days': len(test_daily_pnl),
                'total_pnl': np.sum(test_daily_pnl),
                'avg_daily_pnl': np.mean(test_daily_pnl) if test_daily_pnl else 0,
                'std_daily_pnl': np.std(test_daily_pnl) if len(test_daily_pnl) > 1 else 0
            },
            'risk_metrics': risk_metrics,
            'annual_pnl': annual_pnl,
            'npv': npv,
            'all_daily_pnl': all_daily_pnl
        }
    
    def calculate_risk_metrics(self, daily_pnl: List[float]) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        if not daily_pnl:
            return RiskMetrics()
        
        pnl_array = np.array(daily_pnl)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(pnl_array, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = pnl_array[pnl_array <= var_95].mean() if len(pnl_array[pnl_array <= var_95]) > 0 else var_95
        
        # Maximum Drawdown
        cumulative = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe Ratio (annualized, assuming daily returns)
        avg_return = np.mean(pnl_array)
        std_return = np.std(pnl_array) if len(pnl_array) > 1 else 1e-6
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = pnl_array[pnl_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-6
        sortino_ratio = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar Ratio (return/max drawdown)
        total_return = np.sum(pnl_array)
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win Rate
        win_rate = len(pnl_array[pnl_array > 0]) / len(pnl_array) if len(pnl_array) > 0 else 0
        
        # Profit Factor
        gross_profit = np.sum(pnl_array[pnl_array > 0])
        gross_loss = abs(np.sum(pnl_array[pnl_array < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return RiskMetrics(
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
    
    def calculate_npv(self, annual_pnl: float) -> float:
        """Calculate Net Present Value of the battery investment."""
        years = self.battery.battery_life_years
        discount_rate = self.battery.discount_rate
        
        # Calculate annual cash flows (including maintenance)
        annual_cash_flow = annual_pnl - (self.battery.maintenance_cost_eur_day * 365)
        
        # Calculate NPV
        npv = -self.battery.initial_investment_eur
        for year in range(1, years + 1):
            npv += annual_cash_flow / ((1 + discount_rate) ** year)
        
        # Add salvage value (10% of initial investment)
        salvage_value = self.battery.initial_investment_eur * 0.1
        npv += salvage_value / ((1 + discount_rate) ** years)
        
        return npv
    
    def optimize_cross_border(self, area1: str, area2: str, market: str) -> Dict:
        """Optimize cross-border arbitrage between two areas."""
        if market not in self.data:
            return None
        
        df = self.data[market]
        area1_data = df[df['area'] == area1].copy()
        area2_data = df[df['area'] == area2].copy()
        
        if area1_data.empty or area2_data.empty:
            return None
        
        # Filter by date range
        area1_data = area1_data[
            (area1_data['date_cet'] >= self.market.start_date) &
            (area1_data['date_cet'] <= self.market.end_date)
        ]
        area2_data = area2_data[
            (area2_data['date_cet'] >= self.market.start_date) &
            (area2_data['date_cet'] <= self.market.end_date)
        ]
        
        # Merge data on date and hour
        merged = pd.merge(
            area1_data[['date_cet', 'deliveryStartCET', 'price']],
            area2_data[['date_cet', 'deliveryStartCET', 'price']],
            on=['date_cet', 'deliveryStartCET'],
            suffixes=(f'_{area1}', f'_{area2}')
        )
        
        if merged.empty:
            return None
        
        # Calculate spread
        merged['spread'] = merged[f'price_{area2}'] - merged[f'price_{area1}']
        
        # Find optimal trading hours based on spread
        merged['hour'] = pd.to_datetime(merged['deliveryStartCET']).dt.hour
        
        # Group by hour to find best hours for arbitrage
        hourly_spread = merged.groupby('hour')['spread'].agg(['mean', 'std', 'count'])
        hourly_spread = hourly_spread[hourly_spread['count'] >= 10]  # Minimum sample size
        
        if hourly_spread.empty:
            return None
        
        # Find hours with highest positive spread (buy in area1, sell in area2)
        best_hours = hourly_spread.nlargest(self.battery.discharge_hours, 'mean')
        
        # Calculate P&L for cross-border arbitrage
        daily_pnl = []
        for date in merged['date_cet'].unique():
            day_data = merged[merged['date_cet'] == date]
            
            if len(day_data) < 24:
                continue
            
            # Get prices for best hours
            best_hour_indices = best_hours.index.tolist()
            day_data_filtered = day_data[day_data['hour'].isin(best_hour_indices)]
            
            if not day_data_filtered.empty:
                avg_buy = day_data_filtered[f'price_{area1}'].mean()
                avg_sell = day_data_filtered[f'price_{area2}'].mean()
                
                # Calculate with realistic costs (including transmission)
                costs = self.calculate_realistic_costs(
                    avg_buy, avg_sell,
                    self.battery.capacity_mw * self.battery.discharge_hours,
                    is_cross_border=True
                )
                
                daily_pnl.append(costs['net_profit'])
        
        if not daily_pnl:
            return None
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(daily_pnl)
        
        # Calculate annual P&L
        total_days = len(merged['date_cet'].unique())
        annual_pnl = np.sum(daily_pnl) * (365 / total_days) if total_days > 0 else 0
        
        return {
            'area1': area1,
            'area2': area2,
            'market': market,
            'best_hours': best_hour_indices,
            'avg_spread': hourly_spread['mean'].mean(),
            'daily_pnl': daily_pnl,
            'risk_metrics': risk_metrics,
            'annual_pnl': annual_pnl,
            'total_days': total_days
        }
    
    def monte_carlo_simulation(self, strategy_result: Dict, num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation for uncertainty quantification."""
        if not strategy_result or 'all_daily_pnl' not in strategy_result:
            return None
        
        daily_pnl = strategy_result['all_daily_pnl']
        if not daily_pnl:
            return None
        
        # Fit distribution to daily P&L
        pnl_mean = np.mean(dale_pnl)
        pnl_std = np.std(daily_pnl) if len(daily_pnl) > 1 else 1e-6
        
        # Run simulations
        simulated_annual_pnl = []
        for _ in range(num_simulations):
            # Simulate 365 days
            simulated_days = np.random.normal(pnl_mean, pnl_std, 365)
            simulated_annual_pnl.append(np.sum(simulated_days))
        
        simulated_annual_pnl = np.array(simulated_annual_pnl)
        
        # Calculate statistics
        mean_annual = np.mean(simulated_annual_pnl)
        std_annual = np.std(simulated_annual_pnl)
        percentiles = np.percentile(simulated_annual_pnl, [5, 25, 50, 75, 95])
        
        # Probability of positive P&L
        prob_positive = np.sum(simulated_annual_pnl > 0) / num_simulations
        
        # Probability of exceeding target (e.g., €50k)
        target = 50000
        prob_target = np.sum(simulated_annual_pnl > target) / num_simulations
        
        return {
            'mean_annual_pnl': mean_annual,
            'std_annual_pnl': std_annual,
            'percentiles': percentiles,
            'prob_positive': prob_positive,
            'prob_target': prob_target,
            'simulated_values': simulated_annual_pnl
        }
    
    def optimize_all_areas(self) -> Dict[str, Dict]:
        """Optimize strategies for all areas and markets."""
        results = {}
        
        for area in self.market.areas:
            for market in self.market.markets:
                key = f"{area}_{market}"
                logger.info(f"Optimizing {key}")
                
                result = self.optimize_single_area(area, market)
                if result:
                    results[key] = result
        
        self.results = results
        return results
    
    def get_best_strategy(self) -> Optional[Dict]:
        """Get the best strategy across all areas and markets."""
        if not self.results:
            self.optimize_all_areas()
        
        if not self.results:
            return None
        
        # Find strategy with highest Sharpe ratio
        best_key = None
        best_sharpe = -np.inf
        
        for key, result in self.results.items():
            sharpe = result['risk_metrics'].sharpe_ratio
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_key = key
        
        return self.results.get(best_key)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Enhanced Battery Optimizer",
        page_icon="🔋",
        layout="wide"
    )
    
    st.markdown("""
    # 🔋 Enhanced Battery Optimizer
    Advanced battery storage optimization with realistic cost modeling and risk management
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Battery Settings")
    
    with st.sidebar.expander("🔋 Battery Configuration", expanded=True):
        capacity = st.number_input("Capacity (MW)", value=1.0, min_value=0.1, max_value=100.0, step=0.1)
        charge_hours = st.number_input("Charge (hours)", value=4, min_value=1, max_value=12)
        discharge_hours = st.number_input("Discharge (hours)", value=4, min_value=1, max_value=12)
        max_cycles = st.number_input("Max cycles/day", value=1, min_value=1, max_value=4)
        
        # Check for overlapping windows
        if charge_hours + discharge_hours > 24:
            st.error("Total charge + discharge hours cannot exceed 24")
            return
        
        st.markdown("---")
        
        round_trip_efficiency = st.slider(
            "Round-trip efficiency (%)", 
            min_value=70, max_value=95, value=90, step=1
        ) / 100.0
        
        degradation_cost = st.number_input(
            "Degradation (€/MWh discharged)", 
            value=5.0, min_value=0.0, max_value=20.0, step=0.5
        )
        
        exchange_fees = st.number_input(
            "Exchange fees (€/MWh traded)", 
            value=0.5, min_value=0.0, max_value=5.0, step=0.1
        )
        
        transmission_cost = st.number_input(
            "Transmission cost (€/MWh cross-border)", 
            value=2.0, min_value=0.0, max_value=10.0, step=0.5
        )
    
    with st.sidebar.expander("💰 Investment Parameters", expanded=False):
        initial_investment = st.number_input(
            "Initial investment (€)", 
            value=500000, min_value=10000, max_value=10000000, step=10000
        )
        
        battery_life = st.number_input(
            "Battery life (years)", 
            value=10, min_value=1, max_value=20
        )
        
        discount_rate = st.slider(
            "Discount rate (%)", 
            min_value=0, max_value=20, value=8, step=1
        ) / 100.0
    
    with st.sidebar.expander("🏪 Market Selection", expanded=True):
        markets = st.multiselect(
            "Markets to analyze",
            ["DayAhead", "IDA1", "IDA2", "IDA3"],
            default=["DayAhead", "IDA1"]
        )
        
        areas = st.multiselect(
            "Areas to analyze",
            ["AT", "BE", "FR", "GER", "NL"],
            default=["AT", "BE", "FR", "GER", "NL"]
        )
    
    with st.sidebar.expander("📅 Date Filter", expanded=False):
        day_type = st.selectbox("Day type", ["All", "Weekdays", "Weekends"])
        
        start_date = st.date_input("Start date", value=datetime(2026, 1, 17))
        end_date = st.date_input("End date", value=datetime(2026, 4, 18))
    
    # Create configurations
    battery_config = BatteryConfig(
        capacity_mw=capacity,
        charge_hours=charge_hours,
        discharge_hours=discharge_hours,
        max_cycles_per_day=max_cycles,
        round_trip_efficiency=round_trip_efficiency,
        degradation_cost_eur_mwh=degradation_cost,
        exchange_fee_eur_mwh=exchange_fees,
        transmission_cost_eur_mwh=transmission_cost,
        initial_investment_eur=initial_investment,
        battery_life_years=battery_life,
        discount_rate=discount_rate
    )
    
    market_config = MarketConfig(
        areas=areas,
        markets=markets,
        train_split=0.7,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # Initialize optimizer
    optimizer = EnhancedBatteryOptimizer(battery_config, market_config)
    
    # Run optimization button
    if st.sidebar.button("🚀 Optimize", type="primary"):
        with st.spinner("Loading data and optimizing strategies..."):
            # Load data
            if not optimizer.load_data():
                st.error("Failed to load data. Please check data directory.")
                return
            
            # Run optimization
            results = optimizer.optimize_all_areas()
            
            if not results:
                st.error("No valid strategies found.")
                return
            
            # Get best strategy
            best_strategy = optimizer.get_best_strategy()
            
            if not best_strategy:
                st.error("Could not determine best strategy.")
                return
            
            # Display results
            st.success("✅ Optimization complete!")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Best Strategy",
                    f"{best_strategy['area']} - {best_strategy['market']}"
                )
            
            with col2:
                st.metric(
                    "Annual P&L",
                    f"€{best_strategy['annual_pnl']:,.0f}"
                )
            
            with col3:
                st.metric(
                    "NPV (10 years)",
                    f"€{best_strategy['npv']:,.0f}"
                )
            
            with col4:
                st.metric(
                    "Sharpe Ratio",
                    f"{best_strategy['risk_metrics'].sharpe_ratio:.2f}"
                )
            
            # Detailed results
            st.markdown("---")
            st.markdown("## 📊 Performance Metrics")
            
            # Train vs Test comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟦 Train (in-sample)")
                st.write(f"**Days:** {best_strategy['train_metrics']['days']}")
                st.write(f"**Total P&L:** €{best_strategy['train_metrics']['total_pnl']:,.0f}")
                st.write(f"**Avg Daily:** €{best_strategy['train_metrics']['avg_daily_pnl']:,.0f}")
                st.write(f"**Sharpe:** {best_strategy['train_metrics']['sharpe']:.2f}")
            
            with col2:
                st.markdown("### 🟧 Test (out-of-sample)")
                st.write(f"**Days:** {best_strategy['test_metrics']['days']}")
                st.write(f"**Total P&L:** €{best_strategy['test_metrics']['total_pnl']:,.0f}")
                st.write(f"**Avg Daily:** €{best_strategy['test_metrics']['avg_daily_pnl']:,.0f}")
            
            # Risk metrics
            st.markdown("## ⚠️ Risk Metrics")
            
            risk = best_strategy['risk_metrics']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("VaR (95%)", f"€{risk.var_95:,.0f}")
                st.metric("Win Rate", f"{risk.win_rate:.1%}")
            
            with col2:
                st.metric("CVaR (95%)", f"€{risk.cvar_95:,.0f}")
                st.metric("Profit Factor", f"{risk.profit_factor:.2f}")
            
            with col3:
                st.metric("Max Drawdown", f"€{risk.max_drawdown:,.0f}")
                st.metric("Sortino Ratio", f"{risk.sortino_ratio:.2f}")
            
            with col4:
                st.metric("Calmar Ratio", f"{risk.calmar_ratio:.2f}")
                st.metric("Daily Volatility", f"€{best_strategy['train_metrics']['std_daily_pnl']:,.0f}")
            
            # Strategy details
            st.markdown("## 🎯 Strategy Details")
            
            strategy = best_strategy['strategy']
            st.write(f"**Buy hours:** {strategy['buy_hours']}")
            st.write(f"**Sell hours:** {strategy['sell_hours']}")
            st.write(f"**Charge duration:** {battery_config.charge_hours} hours")
            st.write(f"**Discharge duration:** {battery_config.discharge_hours} hours")
            st.write(f"**Round-trip efficiency:** {battery_config.round_trip_efficiency:.1%}")
            
            # Monte Carlo simulation
            st.markdown("## 🎲 Monte Carlo Simulation")
            
            with st.spinner("Running Monte Carlo simulation..."):
                mc_results = optimizer.monte_carlo_simulation(best_strategy)
                
                if mc_results:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Expected Annual P&L", f"€{mc_results['mean_annual_pnl']:,.0f}")
                        st.metric("Std Dev", f"€{mc_results['std_annual_pnl']:,.0f}")
                    
                    with col2:
                        st.metric("Prob. of Profit", f"{mc_results['prob_positive']:.1%}")
                        st.metric("Prob. > €50k", f"{mc_results['prob_target']:.1%}")
                    
                    with col3:
                        st.metric("5th Percentile", f"€{mc_results['percentiles'][0]:,.0f}")
                        st.metric("95th Percentile", f"€{mc_results['percentiles'][4]:,.0f}")
                    
                    # Distribution plot
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=mc_results['simulated_values'],
                        nbinsx=50,
                        name='Simulated Annual P&L',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title="Monte Carlo Simulation - Annual P&L Distribution",
                        xaxis_title="Annual P&L (€)",
                        yaxis_title="Frequency",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Cross-border analysis
            if len(areas) > 1:
                st.markdown("## 🌍 Cross-Border Arbitrage")
                
                cross_border_results = []
                for i, area1 in enumerate(areas):
                    for area2 in areas[i+1:]:
                        for market in markets:
                            result = optimizer.optimize_cross_border(area1, area2, market)
                            if result:
                                cross_border_results.append(result)
                
                if cross_border_results:
                    # Sort by annual P&L
                    cross_border_results.sort(key=lambda x: x['annual_pnl'], reverse=True)
                    
                    # Display top 5
                    for i, result in enumerate(cross_border_results[:5]):
                        with st.expander(f"#{i+1}: {result['area1']} → {result['area2']} ({result['market']})"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Annual P&L", f"€{result['annual_pnl']:,.0f}")
                                st.metric("Avg Spread", f"€{result['avg_spread']:.2f}/MWh")
                            
                            with col2:
                                st.metric("Sharpe Ratio", f"{result['risk_metrics'].sharpe_ratio:.2f}")
                                st.metric("Win Rate", f"{result['risk_metrics'].win_rate:.1%}")
                            
                            with col3:
                                st.metric("Max Drawdown", f"€{result['risk_metrics'].max_drawdown:,.0f}")
                                st.metric("Best Hours", str(result['best_hours']))
            
            # Export options
            st.markdown("## 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export all strategies
                export_data = []
                for key, result in results.items():
                    export_data.append({
                        'Strategy': key,
                        'Area': result['area'],
                        'Market': result['market'],
                        'Annual P&L (€)': result['annual_pnl'],
                        'Sharpe Ratio': result['risk_metrics'].sharpe_ratio,
                        'Win Rate': result['risk_metrics'].win_rate,
                        'Max Drawdown (€)': result['risk_metrics'].max_drawdown,
                        'NPV (€)': result['npv']
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download All Strategies (CSV)",
                    data=csv,
                    file_name=f"battery_strategies_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export summary report
                summary = {
                    'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'battery_config': {
                        'capacity_mw': capacity,
                        'charge_hours': charge_hours,
                        'discharge_hours': discharge_hours,
                        'round_trip_efficiency': round_trip_efficiency,
                        'degradation_cost': degradation_cost,
                        'exchange_fees': exchange_fees,
                        'transmission_cost': transmission_cost
                    },
                    'best_strategy': {
                        'area': best_strategy['area'],
                        'market': best_strategy['market'],
                        'annual_pnl': best_strategy['annual_pnl'],
                        'npv': best_strategy['npv'],
                        'sharpe': best_strategy['risk_metrics'].sharpe_ratio
                    },
                    'risk_metrics': {
                        'var_95': risk.var_95,
                        'cvar_95': risk.cvar_95,
                        'max_drawdown': risk.max_drawdown,
                        'win_rate': risk.win_rate,
                        'profit_factor': risk.profit_factor
                    }
                }
                
                import json
                json_str = json.dumps(summary, indent=2)
                
                st.download_button(
                    label="📊 Download Summary Report (JSON)",
                    data=json_str,
                    file_name=f"battery_summary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
