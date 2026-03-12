"""
Flask web application for displaying energy auction and intraday price analytics.

This app reads the CSV files produced by the Nord Pool bot (`dayahead_prices.csv`,
`ida1_prices.csv`, `ida2_prices.csv`, `ida3_prices.csv`, and
`intraday_continuous_vwap_qh.csv`) from the `data/` directory and presents
interactive charts and summary tables on a simple website.

To run the application locally:

    # install dependencies
    pip install -r requirements.txt

    # set the FLASK_APP environment variable and run
    export FLASK_APP=app.py
    flask run --reload

Navigate to http://127.0.0.1:5000/ in your browser.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
from flask import Flask, abort, render_template


# Create the Flask application
app = Flask(__name__)

# Directory containing the CSV datasets.  By default this expects a sibling
# `data` directory relative to this file (i.e. in the repository root).  You
# can override this by setting the DATA_DIR environment variable when running
# the app.
DATA_DIR = Path(__file__).parent / "data"


def _load_csv(name: str) -> Optional[pd.DataFrame]:
    """Load a CSV file from the data directory if it exists, otherwise return None."""
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_all_data() -> Dict[str, pd.DataFrame]:
    """Load all known CSV files into a dictionary keyed by market name."""
    return {
        "dayahead": _load_csv("dayahead_prices.csv"),
        "ida1": _load_csv("ida1_prices.csv"),
        "ida2": _load_csv("ida2_prices.csv"),
        "ida3": _load_csv("ida3_prices.csv"),
        "vwap_qh": _load_csv("intraday_continuous_vwap_qh.csv"),
    }


def compute_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute a summary table showing average prices and spreads for the latest date
    in the day‑ahead auction.  The summary includes:

    - Average price for each market (day‑ahead and intraday auctions) on the
      latest date.
    - The spread between each intraday auction and the day‑ahead auction.
    - The max‑min price spread across all auctions.
    - The day‑to‑day change in the day‑ahead average price.
    - A simple arbitrage recommendation indicating whether buying on day‑ahead
      and selling on intraday would have been profitable.

    Returns a DataFrame indexed by area.
    """
    da = data.get("dayahead")
    if da is None or da.empty:
        return pd.DataFrame()

    # Ensure date column is datetime
    da["date_cet"] = pd.to_datetime(da["date_cet"])

    # Determine the latest date for which we have day‑ahead data
    latest_date = da["date_cet"].max()
    # Determine the previous date for day‑ahead data
    prev_da = da[da["date_cet"] < latest_date]
    prev_date: Optional[pd.Timestamp] = None
    if not prev_da.empty:
        prev_date = prev_da["date_cet"].max()

    # Helper to compute average price by area on latest date
    def avg_price(df: Optional[pd.DataFrame], market_name: str) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        temp = df.copy()
        if "date_cet" in temp.columns:
            temp["date_cet"] = pd.to_datetime(temp["date_cet"])
            temp = temp[temp["date_cet"] == latest_date]
        return temp.groupby("area")["price"].mean().rename(market_name)

    da_avg = avg_price(da, "da_avg")
    ida1_avg = avg_price(data.get("ida1"), "ida1_avg")
    ida2_avg = avg_price(data.get("ida2"), "ida2_avg")
    ida3_avg = avg_price(data.get("ida3"), "ida3_avg")

    # Combine all average price series into one DataFrame
    summary = pd.concat([da_avg, ida1_avg, ida2_avg, ida3_avg], axis=1)

    # Compute spreads between intraday auctions and day‑ahead
    for name in ["ida1_avg", "ida2_avg", "ida3_avg"]:
        if name in summary.columns:
            summary[f"spread_{name}_vs_da"] = summary[name] - summary["da_avg"]

    # Compute max‑min spread across all available markets for each area
    def row_spread(row):
        prices = [val for col, val in row.items() if col.endswith("_avg") and pd.notna(val)]
        return max(prices) - min(prices) if prices else float("nan")

    summary["max_min_spread"] = summary.apply(row_spread, axis=1)

    # Compute day‑to‑day change in day‑ahead average price
    if prev_date is not None:
        prev_da_df = da[da["date_cet"] == prev_date]
        prev_avg = prev_da_df.groupby("area")["price"].mean().rename("da_avg_prev")
        summary = summary.join(prev_avg, how="left")
        summary["da_change_day"] = summary["da_avg"] - summary["da_avg_prev"]
    else:
        summary["da_change_day"] = float("nan")

    # Compute a simple arbitrage recommendation: positive values indicate selling
    # on intraday (IDA1) after buying on day‑ahead would have made a profit
    if "spread_ida1_avg_vs_da" in summary.columns:
        summary["arbitrage_ida1"] = summary["spread_ida1_avg_vs_da"].apply(
            lambda x: "Sell IDA1" if pd.notna(x) and x > 0 else "Buy DA"
        )
    else:
        summary["arbitrage_ida1"] = "N/A"

    summary.index.name = "area"
    return summary


def build_price_chart(area: str, data: Dict[str, pd.DataFrame]) -> Optional[str]:
    """
    Build an interactive Plotly chart showing the price curves for the latest date
    across available auction markets for the specified area.  Returns HTML for
    embedding into a template or None if data is unavailable.
    """
    da = data.get("dayahead")
    if da is None or da.empty:
        return None
    # Filter data for the selected area and latest date
    da["date_cet"] = pd.to_datetime(da["date_cet"])
    latest_date = da["date_cet"].max()

    def prep_df(df: Optional[pd.DataFrame], market_label: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        temp = df.copy()
        if "date_cet" in temp.columns:
            temp["date_cet"] = pd.to_datetime(temp["date_cet"])
            temp = temp[(temp["date_cet"] == latest_date) & (temp["area"].str.upper() == area.upper())]
        if temp.empty:
            return temp
        temp = temp.sort_values("deliveryStartCET")
        temp["market"] = market_label
        return temp[["deliveryStartCET", "price", "market"]]

    frames: List[pd.DataFrame] = []
    frames.append(prep_df(da, "DayAhead"))
    frames.append(prep_df(data.get("ida1"), "IDA1"))
    frames.append(prep_df(data.get("ida2"), "IDA2"))
    frames.append(prep_df(data.get("ida3"), "IDA3"))
    chart_df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if chart_df.empty:
        return None

    fig = px.line(
        chart_df,
        x="deliveryStartCET",
        y="price",
        color="market",
        title=f"Price curves for {area.upper()} on {latest_date.date()}",
        labels={"deliveryStartCET": "Delivery Start (CET)", "price": "Price"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    # Return HTML snippet without a full document wrapper
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@app.route("/")
def index():
    """Home page showing a summary table for all areas."""
    data = load_all_data()
    summary = compute_summary(data)
    # Convert to JSON for easier handling in template
    table_json = summary.reset_index().to_dict(orient="records")
    columns = list(summary.reset_index().columns)
    return render_template(
        "index.html",
        table=table_json,
        columns=columns,
    )


@app.route("/market/<area>")
def market_view(area: str):
    """Detail page for a single market area showing an interactive chart."""
    data = load_all_data()
    chart_html = build_price_chart(area, data)
    if chart_html is None:
        abort(404, description=f"No data available for area {area}")
    return render_template(
        "market.html",
        area=area.upper(),
        chart_html=chart_html,
    )


if __name__ == "__main__":
    # When executed directly, run the Flask development server
    app.run(debug=True)
