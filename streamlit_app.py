"""
Streamlit application for exploring Nord Pool auction and intraday price data.

This app reads the CSV files produced by the scraping pipeline (day‑ahead and
intraday auctions as well as continuous VWAP) and provides interactive
visualisations and summary tables.  Users can select a delivery area and
inspect day‑ahead vs intraday price trends, compare any day against a rolling
seven‑day average, and view potential arbitrage opportunities (the spread
between intraday and day‑ahead prices for matching delivery intervals).

To run locally:

    pip install streamlit pandas plotly
    streamlit run streamlit_app.py

To deploy publicly (e.g. on Streamlit Cloud), push this file along with your
data directory to a GitHub repository and follow the deployment instructions
at https://streamlit.io/sharing.  The app expects the following CSV files
inside a `data` folder in the working directory:

    - dayahead_prices.csv
    - ida1_prices.csv
    - ida2_prices.csv
    - ida3_prices.csv
    - intraday_continuous_vwap_qh.csv

If any file is missing, the corresponding analysis will be skipped.
"""

import os
import logging
from functools import lru_cache
from typing import Dict, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Daily prediction scheduler — starts once per server process
# Runs predict_tomorrow.run_predictions() every day at 11:45 CET
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _start_prediction_scheduler():
    """
    Start APScheduler in a background thread.
    @st.cache_resource ensures this runs exactly once per server process,
    not on every Streamlit rerun or new user session.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        def _daily_job():
            try:
                logging.info("[Scheduler] Starting daily prediction run at 11:45 CET")
                from predict_tomorrow import run_predictions
                result = run_predictions()
                logging.info(
                    f"[Scheduler] Predictions saved for {result.get('delivery_day', '?')}"
                )
            except Exception as exc:
                logging.error(f"[Scheduler] Prediction run failed: {exc}", exc_info=True)

        scheduler = BackgroundScheduler(timezone="Europe/Berlin")
        # Run every day at 11:45 CET
        scheduler.add_job(
            _daily_job,
            trigger=CronTrigger(hour=11, minute=45, timezone="Europe/Berlin"),
            id="daily_price_prediction",
            name="Day-ahead price prediction",
            replace_existing=True,
        )
        scheduler.start()
        logging.info("[Scheduler] APScheduler started — daily predictions at 11:45 CET")
        return scheduler
    except ImportError:
        logging.warning("[Scheduler] apscheduler not installed — skipping scheduled jobs")
        return None
    except Exception as exc:
        logging.error(f"[Scheduler] Failed to start scheduler: {exc}", exc_info=True)
        return None


# Start the scheduler when the Streamlit server loads this module
_start_prediction_scheduler()
DATA_DIR = os.path.join(BASE_DIR, "data")


@lru_cache(maxsize=None)
def load_csv(filename: str, price_col: str) -> Optional[pd.DataFrame]:
    """Load a CSV into a DataFrame and parse datetime columns.

    Returns None if the file does not exist or is missing required columns.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Standardise column names
    df.columns = [c.strip() for c in df.columns]
    required = {"date_cet", "area", "deliveryStartCET", price_col}
    missing = required - set(df.columns)
    if missing:
        return None
    df["deliveryStartCET"] = (
        pd.to_datetime(df["deliveryStartCET"], utc=True).dt.tz_convert("Europe/Paris")
    )
    return df


def compute_daily_average(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Compute the average price per date and area."""
    return (
        df.groupby(["date_cet", "area"])[price_col]
        .mean()
        .reset_index(name="avg_price")
    )


def compute_arbitrage(
    da_df: pd.DataFrame, other_df: pd.DataFrame, other_col: str
) -> pd.DataFrame:
    """Compute average spread (other - day ahead) per date and area."""
    if da_df is None or other_df is None:
        return pd.DataFrame()
    da_subset = da_df[["date_cet", "area", "deliveryStartCET", "price"]].rename(
        columns={"price": "da_price"}
    )
    merged = pd.merge(
        da_subset,
        other_df[["date_cet", "area", "deliveryStartCET", other_col]],
        on=["date_cet", "area", "deliveryStartCET"],
        how="inner",
    )
    merged["spread"] = merged[other_col] - merged["da_price"]
    return (
        merged.groupby(["date_cet", "area"])["spread"]
        .mean()
        .reset_index(name="avg_spread")
    )


def main() -> None:
    st.title("Nord Pool Price Explorer")
    st.markdown(
        """<style>
a.cc-btn{position:fixed;top:14px;right:60px;z-index:999999;
background:linear-gradient(135deg,#0D47A1,#1976D2);color:#fff!important;
padding:7px 16px;border-radius:20px;font-weight:600;font-size:13px;
text-decoration:none!important;box-shadow:0 2px 8px rgba(0,0,0,.25);
transition:all .2s ease;letter-spacing:.3px}
a.cc-btn:hover{background:linear-gradient(135deg,#1565C0,#1E88E5);
box-shadow:0 4px 14px rgba(0,0,0,.35);transform:translateY(-1px);color:#fff!important}
</style>
<a class="cc-btn" href="https://www.linkedin.com/in/bibek-agarwal" target="_blank">\u{1F4EC} Contact to know more</a>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        This Streamlit app visualises electricity prices from the Nord Pool day‑ahead
        and intraday markets.  Select a delivery area and explore price trends,
        compare any day against a seven‑day rolling average, and identify
        potential arbitrage opportunities.
        """
    )


    # Load datasets
    da = load_csv("dayahead_prices.csv", "price")
    ida1 = load_csv("ida1_prices.csv", "price")
    ida2 = load_csv("ida2_prices.csv", "price")
    ida3 = load_csv("ida3_prices.csv", "price")
    vwap = load_csv("intraday_continuous_vwap_qh.csv", "vwap")

    if da is None:
        st.error("Day‑ahead data not found.  Please ensure `dayahead_prices.csv` is in the data directory.")
        return

    # Determine available areas
    available_areas = sorted(da["area"].unique())
    area = st.selectbox("Select delivery area", available_areas)

    # Filter by area
    da_area = da[da["area"] == area]
    ida1_area = ida1[ida1["area"] == area] if ida1 is not None else None
    ida2_area = ida2[ida2["area"] == area] if ida2 is not None else None
    ida3_area = ida3[ida3["area"] == area] if ida3 is not None else None
    vwap_area = vwap[vwap["area"] == area] if vwap is not None else None

    # Compute daily averages
    da_avg = compute_daily_average(da_area, "price")
    ida1_avg = compute_daily_average(ida1_area, "price") if ida1_area is not None else None
    ida2_avg = compute_daily_average(ida2_area, "price") if ida2_area is not None else None
    ida3_avg = compute_daily_average(ida3_area, "price") if ida3_area is not None else None
    vwap_avg = compute_daily_average(vwap_area, "vwap") if vwap_area is not None else None

    # Date range selection
    dates = pd.to_datetime(da_avg["date_cet"]).sort_values().unique()
    if len(dates) > 1:
        start_date, end_date = st.select_slider(
            "Select date range",
            options=list(dates),
            value=(dates[0], dates[-1]),
            format_func=lambda x: x.strftime("%Y-%m-%d"),
        )
        da_avg = da_avg[
            (pd.to_datetime(da_avg["date_cet"]) >= start_date)
            & (pd.to_datetime(da_avg["date_cet"]) <= end_date)
        ]
        if ida1_avg is not None:
            ida1_avg = ida1_avg[
                (pd.to_datetime(ida1_avg["date_cet"]) >= start_date)
                & (pd.to_datetime(ida1_avg["date_cet"]) <= end_date)
            ]
        if ida2_avg is not None:
            ida2_avg = ida2_avg[
                (pd.to_datetime(ida2_avg["date_cet"]) >= start_date)
                & (pd.to_datetime(ida2_avg["date_cet"]) <= end_date)
            ]
        if ida3_avg is not None:
            ida3_avg = ida3_avg[
                (pd.to_datetime(ida3_avg["date_cet"]) >= start_date)
                & (pd.to_datetime(ida3_avg["date_cet"]) <= end_date)
            ]
        if vwap_avg is not None:
            vwap_avg = vwap_avg[
                (pd.to_datetime(vwap_avg["date_cet"]) >= start_date)
                & (pd.to_datetime(vwap_avg["date_cet"]) <= end_date)
            ]

    # Display daily average price line chart
    chart_df = da_avg.rename(columns={"avg_price": "DayAhead"})[["date_cet", "DayAhead"]].copy()
    if ida1_avg is not None and not ida1_avg.empty:
        chart_df = chart_df.merge(
            ida1_avg.rename(columns={"avg_price": "IDA1"})[["date_cet", "IDA1"]], on="date_cet", how="left"
        )
    if ida2_avg is not None and not ida2_avg.empty:
        chart_df = chart_df.merge(
            ida2_avg.rename(columns={"avg_price": "IDA2"})[["date_cet", "IDA2"]], on="date_cet", how="left"
        )
    if ida3_avg is not None and not ida3_avg.empty:
        chart_df = chart_df.merge(
            ida3_avg.rename(columns={"avg_price": "IDA3"})[["date_cet", "IDA3"]], on="date_cet", how="left"
        )
    if vwap_avg is not None and not vwap_avg.empty:
        chart_df = chart_df.merge(
            vwap_avg.rename(columns={"avg_price": "VWAP"})[["date_cet", "VWAP"]], on="date_cet", how="left"
        )

    chart_df = chart_df.sort_values("date_cet")
    if len(chart_df) > 1:
        fig = px.line(
            chart_df,
            x="date_cet",
            y=[c for c in chart_df.columns if c != "date_cet"],
            markers=True,
            labels={"value": "Average Price (EUR/MWh)", "date_cet": "Date", "variable": "Dataset"},
            title=f"Average prices for {area}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough dates to plot a trend.  Try loading more days of data.")

    # Compare each day to a 7‑day rolling average
    window = 7
    da_avg_sorted = da_avg.copy()
    da_avg_sorted["date_cet_dt"] = pd.to_datetime(da_avg_sorted["date_cet"])
    da_avg_sorted = da_avg_sorted.sort_values("date_cet_dt")
    da_avg_sorted["rolling_avg"] = da_avg_sorted["avg_price"].rolling(window, min_periods=1).mean()
    da_avg_sorted["diff_to_rolling"] = da_avg_sorted["avg_price"] - da_avg_sorted["rolling_avg"]
    st.subheader("Daily deviations from rolling 7‑day average (DayAhead)")
    st.dataframe(
        da_avg_sorted[["date_cet", "avg_price", "rolling_avg", "diff_to_rolling"]]
        .rename(
            columns={
                "date_cet": "Date",
                "avg_price": "DayAhead Avg",
                "rolling_avg": "7‑day Rolling Avg",
                "diff_to_rolling": "Diff to 7‑day Avg",
            }
        )
    )

    # Arbitrage opportunities: DayAhead vs IDA1
    arb = compute_arbitrage(da_area, ida1_area, "price")
    if not arb.empty:
        arb_area = arb[arb["area"] == area].copy()
        st.subheader("Intraday (IDA1) vs Day‑ahead average spread")
        st.dataframe(
            arb_area.rename(
                columns={"date_cet": "Date", "avg_spread": "Average Spread (EUR/MWh)"}
            )
        )
        # Chart for arbitrage spread
        fig_spread = px.bar(
            arb_area,
            x="date_cet",
            y="avg_spread",
            labels={"date_cet": "Date", "avg_spread": "IDA1 – DayAhead (EUR/MWh)"},
            title=f"Average intraday vs day‑ahead spread for {area}"
        )
        st.plotly_chart(fig_spread, use_container_width=True)
    else:
        st.info("Intraday auction 1 data not available; arbitrage spread cannot be computed.")


if __name__ == "__main__":
    main()
