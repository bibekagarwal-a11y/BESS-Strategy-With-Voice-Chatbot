"""
6_Tomorrow_Forecast.py
======================
Streamlit page that shows the automatic 15-minute day-ahead price forecast
for tomorrow across AT, BE, FR, GER, and NL.

The forecast is generated daily at 11:45 CET by the embedded APScheduler
(see streamlit_app.py).  Users can also trigger a manual re-run from this page.

Trader dashboard:
  - Prediction vs Actual overlay (once delivery day settles in the CSV)
  - % Delta bars  (orange = over-predict, blue = under-predict)
  - MAE / MAPE / Bias / Pearson r accuracy KPIs
  - Error-by-hour heatmap (when/where the model is systematically wrong)
  - Battery P&L simulation: predicted P&L vs actual P&L vs perfect-foresight
  - Spread & best charge/discharge windows per area
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Tomorrow's Forecast",
    page_icon="📅",
)
st.markdown(
    """
    <style>
    #watt-header{position:fixed;top:0;left:0;right:0;z-index:1000000;background:#fff;border-bottom:2px solid #e2e8f0;padding:0 20px 0 16px;height:52px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 2px 8px rgba(0,0,0,.06);}
    [data-testid="stMainBlockContainer"]{padding-top:60px!important;}
    [data-testid="stSidebarCollapseButton"]{display:none!important;}
    </style>
    <div id="watt-header">
      <div style="display:flex;align-items:center;gap:12px">
        <div style="background:linear-gradient(135deg,#1565C0,#0D47A1);border-radius:10px;width:38px;height:38px;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 2px 8px rgba(21,101,192,.3)">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="20" height="20"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        </div>
        <div>
          <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:1.2rem;font-weight:700;color:#0D1B3E;letter-spacing:-0.2px;line-height:1.2">Watt Happens</div>
          <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:0.7rem;color:#64748b;margin-top:1px">BESS Strategy &amp; Energy Intelligence Platform</div>
        </div>
      </div>
      <a href="https://www.linkedin.com/in/bibek-agarwal" target="_blank" style="display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,#0D47A1,#1976D2);color:#fff!important;padding:7px 16px;border-radius:20px;font-weight:600;font-size:13px;text-decoration:none!important;box-shadow:0 2px 8px rgba(0,0,0,.25);letter-spacing:.3px;white-space:nowrap"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="14" height="14" fill="white"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>Contact to know more</a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background:#ffffff;border-bottom:2px solid #e2e8f0;padding:14px 6rem;margin:-1rem -6rem 1.5rem -6rem;display:flex;align-items:center;gap:14px;box-shadow:0 2px 8px rgba(0,0,0,0.06)">
      <div style="background:linear-gradient(135deg,#1565C0,#0D47A1);border-radius:10px;width:46px;height:46px;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 2px 8px rgba(21,101,192,0.3)">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="24" height="24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
      </div>
      <div>
        <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:1.4rem;font-weight:700;color:#0D1B3E;letter-spacing:-0.2px;line-height:1.2">Watt Happens</div>
        <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:0.8rem;color:#64748b;margin-top:2px">BESS Strategy &amp; Energy Intelligence Platform</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "predictions_tomorrow.json")
PRICE_CSV   = os.path.join(BASE_DIR, "data", "dayahead_prices.csv")

AREA_NAMES = {
    "AT":  "Austria",
    "BE":  "Belgium",
    "FR":  "France",
    "GER": "Germany",
    "NL":  "Netherlands",
}

AREA_COLORS = {
    "AT":  "#1976D2",
    "BE":  "#E65100",
    "FR":  "#2E7D32",
    "GER": "#6A1B9A",
    "NL":  "#00838F",
}

CHART_CFG = {"height": 480, "margin": dict(l=40, r=20, t=50, b=40)}
CHARGE_SLOTS   = 16   # 4 hours × 4 slots/h
DISCHARGE_SLOTS = 16


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers — predictions
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions() -> dict | None:
    if not os.path.exists(OUTPUT_JSON):
        return None
    with open(OUTPUT_JSON) as fh:
        return json.load(fh)


def predictions_to_df(data: dict) -> pd.DataFrame:
    rows = []
    for area, slots in data["predictions"].items():
        for s in slots:
            rows.append({
                "area":          area,
                "datetime":      pd.to_datetime(s["datetime"]),
                "price_eur_mwh": s["price_eur_mwh"],
            })
    return pd.DataFrame(rows)


def run_now() -> dict:
    from predict_tomorrow import run_predictions
    with st.spinner("⏳ Training models and fetching weather forecast…  (30–90 s)"):
        result = run_predictions()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers — actuals (for accuracy check)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_actuals_for_day(delivery_day: str) -> dict[str, pd.DataFrame]:
    """
    Load actual DA prices for delivery_day from dayahead_prices.csv.
    Returns dict: area -> DataFrame[datetime (tz-naive, wall-clock CET), price_actual].
    Empty dict if the day is not yet in the CSV.
    """
    if not os.path.exists(PRICE_CSV):
        return {}
    try:
        df = pd.read_csv(PRICE_CSV)
        df["date_cet"] = df["date_cet"].astype(str)
        day_df = df[df["date_cet"] == delivery_day].copy()
        if day_df.empty:
            return {}
        # Convert to wall-clock CET (tz-naive) so it aligns with prediction datetimes
        day_df["datetime"] = (
            pd.to_datetime(day_df["deliveryStartCET"], utc=True)
            .dt.tz_convert("Europe/Paris")
            .dt.tz_localize(None)
        )
        result: dict[str, pd.DataFrame] = {}
        for area in AREA_NAMES:
            a_df = (
                day_df[day_df["area"] == area][["datetime", "price"]]
                .rename(columns={"price": "price_actual"})
                .sort_values("datetime")
                .reset_index(drop=True)
            )
            if not a_df.empty:
                result[area] = a_df
        return result
    except Exception:
        return {}


def merge_pred_actual(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join predicted and actual on datetime.
    Adds columns: delta, delta_pct, hour, abs_delta.
    """
    p = pred_df.copy()
    a = actual_df.copy()
    # Ensure tz-naive
    for frame in (p, a):
        if frame["datetime"].dt.tz is not None:
            frame["datetime"] = frame["datetime"].dt.tz_localize(None)

    m = pd.merge(p, a, on="datetime", how="inner")
    m["delta"]     = m["price_eur_mwh"] - m["price_actual"]
    # Clip denominator to avoid division by ~0 (negative prices happen!)
    m["delta_pct"] = m["delta"] / m["price_actual"].abs().clip(lower=1.0) * 100
    m["abs_delta"] = m["delta"].abs()
    m["hour"]      = m["datetime"].dt.hour
    return m


def accuracy_metrics(m: pd.DataFrame) -> dict:
    if m.empty:
        return {}
    return {
        "MAE": round(m["abs_delta"].mean(), 2),
        "RMSE": round(float(np.sqrt((m["delta"] ** 2).mean())), 2),
        "MAPE %": round(m["delta_pct"].abs().mean(), 1),
        "Bias": round(m["delta"].mean(), 2),
        "Max over": round(m["delta"].max(), 2),
        "Max under": round(m["delta"].min(), 2),
        "Pearson r": round(m[["price_eur_mwh", "price_actual"]].corr().iloc[0, 1], 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Battery strategy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_window(prices: pd.Series, n: int, mode: str = "min", exclude: tuple | None = None) -> tuple[int, int, float]:
    """Find the best consecutive n-slot window (min or max avg price)."""
    best_start, best_val = 0, (float("inf") if mode == "min" else float("-inf"))
    cmp = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
    for i in range(len(prices) - n + 1):
        end = i + n - 1
        if exclude:
            es, ee = exclude
            if not (end < es or i > ee):
                continue   # overlapping — skip
        val = prices.iloc[i:i + n].mean()
        if cmp(val, best_val):
            best_val, best_start = val, i
    return best_start, best_start + n - 1, round(best_val, 2)


def battery_pnl(
    pred_prices: pd.Series,
    n: int = CHARGE_SLOTS,
    actual_prices: pd.Series | None = None,
) -> dict:
    """
    Compute battery P&L for a 1-MW battery with n charge + n discharge slots.
    Returns dict with predicted_pnl, actual_pnl (if actuals given), optimal_pnl.
    All in EUR per MWh-equivalent cycle (i.e., EUR per MW capacity per cycle).
    """
    slot_h = 0.25   # each slot = 15 min = 0.25 h

    c_start, c_end, c_avg_pred = _best_window(pred_prices, n, "min")
    d_start, d_end, d_avg_pred = _best_window(pred_prices, n, "max", exclude=(c_start, c_end))

    pred_pnl = round((d_avg_pred - c_avg_pred) * n * slot_h, 2)

    result = {
        "charge_start":  c_start,
        "charge_end":    c_end,
        "discharge_start": d_start,
        "discharge_end": d_end,
        "avg_charge_pred": c_avg_pred,
        "avg_discharge_pred": d_avg_pred,
        "predicted_pnl": pred_pnl,
    }

    if actual_prices is not None and len(actual_prices) >= max(c_end, d_end) + 1:
        # Actual P&L using SAME slots as predicted strategy
        c_avg_act = round(float(actual_prices.iloc[c_start:c_end + 1].mean()), 2)
        d_avg_act = round(float(actual_prices.iloc[d_start:d_end + 1].mean()), 2)
        result["avg_charge_actual"]    = c_avg_act
        result["avg_discharge_actual"] = d_avg_act
        result["actual_pnl"] = round((d_avg_act - c_avg_act) * n * slot_h, 2)

        # Perfect-foresight P&L (best possible)
        opt_c_start, opt_c_end, opt_c_avg = _best_window(actual_prices, n, "min")
        opt_d_start, opt_d_end, opt_d_avg = _best_window(actual_prices, n, "max", exclude=(opt_c_start, opt_c_end))
        result["optimal_pnl"] = round((opt_d_avg - opt_c_avg) * n * slot_h, 2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Forecast charts (existing)
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_chart(df: pd.DataFrame, selected_areas: list[str]) -> go.Figure:
    fig = go.Figure()
    for area in selected_areas:
        adf = df[df["area"] == area].sort_values("datetime")
        fig.add_trace(go.Scatter(
            x=adf["datetime"],
            y=adf["price_eur_mwh"],
            mode="lines",
            name=f"{area} — {AREA_NAMES[area]}",
            line=dict(color=AREA_COLORS[area], width=2),
            hovertemplate=(
                f"<b>{AREA_NAMES[area]}</b><br>"
                "%{x|%H:%M}<br>%{y:.1f} EUR/MWh<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="15-Minute Day-Ahead Price Forecast — Tomorrow",
        xaxis_title="Delivery time (CET)",
        yaxis_title="Price (EUR/MWh)",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        **CHART_CFG,
    )
    fig.update_xaxes(tickformat="%H:%M", dtick=3600_000 * 2)
    return fig


def _area_profile_chart(df: pd.DataFrame, area: str, pnl: dict | None = None) -> go.Figure:
    adf  = df[df["area"] == area].sort_values("datetime").reset_index(drop=True)
    clr  = AREA_COLORS[area]
    fig  = go.Figure()
    # Area fill
    fig.add_trace(go.Scatter(
        x=adf["datetime"], y=adf["price_eur_mwh"],
        mode="lines", fill="tozeroy",
        fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.10)",
        line=dict(color=clr, width=2.5),
        name="Predicted", hovertemplate="%{x|%H:%M}  %{y:.1f} EUR/MWh<extra></extra>",
    ))
    # Peak / trough
    for idx_fn, sym, lbl in [(lambda d: d.idxmax(), "triangle-up", "Peak"),
                              (lambda d: d.idxmin(), "triangle-down", "Trough")]:
        idx = idx_fn(adf["price_eur_mwh"])
        row = adf.loc[idx]
        fig.add_trace(go.Scatter(
            x=[row["datetime"]], y=[row["price_eur_mwh"]],
            mode="markers+text",
            marker=dict(size=12, color=clr, symbol=sym),
            text=[f"{lbl}: {row['price_eur_mwh']:.1f}"],
            textposition="top center", showlegend=False,
            hovertemplate=f"{lbl}: %{{y:.1f}} EUR/MWh @ %{{x|%H:%M}}<extra></extra>",
        ))
    # Charge / discharge windows from battery strategy
    if pnl:
        def _add_window(start_i, end_i, color, label):
            window_df = adf.iloc[start_i:end_i + 1]
            fig.add_vrect(
                x0=window_df["datetime"].iloc[0],
                x1=window_df["datetime"].iloc[-1],
                fillcolor=color, opacity=0.15, line_width=0,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_size=11,
            )
        _add_window(pnl["charge_start"],    pnl["charge_end"],    "#1565C0", "⚡ Charge")
        _add_window(pnl["discharge_start"], pnl["discharge_end"], "#E65100", "💰 Discharge")

    fig.update_layout(
        title=f"{AREA_NAMES[area]} ({area}) — Hourly Profile",
        xaxis_title="Delivery time (CET)",
        yaxis_title="Price (EUR/MWh)",
        **CHART_CFG,
    )
    fig.update_xaxes(tickformat="%H:%M", dtick=3600_000 * 2)
    return fig


def _heatmap_chart(df: pd.DataFrame, selected_areas: list[str]) -> go.Figure:
    piv = (
        df[df["area"].isin(selected_areas)]
        .assign(hour=lambda d: d["datetime"].dt.strftime("%H:%M"))
        .pivot_table(index="area", columns="hour", values="price_eur_mwh", aggfunc="mean")
    )
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
        colorscale="RdYlGn_r", hoverongaps=False,
        hovertemplate="Area: %{y}<br>Time: %{x}<br>Price: %{z:.1f} EUR/MWh<extra></extra>",
        colorbar=dict(title="EUR/MWh"),
    ))
    fig.update_layout(
        title="Price Heatmap — Areas × Quarter-Hour",
        xaxis_title="Delivery time (CET)",
        xaxis=dict(tickangle=-45, nticks=24),
        height=280, margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Trader dashboard charts (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def _comparison_overlay_chart(merged: pd.DataFrame, area: str) -> go.Figure:
    """Predicted vs Actual line overlay with error fill."""
    clr = AREA_COLORS[area]
    fig = go.Figure()

    # Error fill band between curves
    fig.add_trace(go.Scatter(
        x=pd.concat([merged["datetime"], merged["datetime"].iloc[::-1]]),
        y=pd.concat([merged["price_actual"], merged["price_eur_mwh"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(200,200,200,0.30)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True,
        name="Error band",
        hoverinfo="skip",
    ))
    # Actual
    fig.add_trace(go.Scatter(
        x=merged["datetime"], y=merged["price_actual"],
        mode="lines", name="Actual",
        line=dict(color="#222", width=2.5),
        hovertemplate="%{x|%H:%M}  Actual: %{y:.1f} EUR/MWh<extra></extra>",
    ))
    # Predicted
    fig.add_trace(go.Scatter(
        x=merged["datetime"], y=merged["price_eur_mwh"],
        mode="lines", name="Predicted",
        line=dict(color=clr, width=2, dash="dash"),
        hovertemplate="%{x|%H:%M}  Predicted: %{y:.1f} EUR/MWh<extra></extra>",
    ))
    fig.update_layout(
        title=f"{area} — Predicted vs Actual (same delivery day)",
        xaxis_title="Delivery time (CET)",
        yaxis_title="Price (EUR/MWh)",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        **CHART_CFG,
    )
    fig.update_xaxes(tickformat="%H:%M", dtick=3600_000 * 2)
    return fig


def _delta_pct_chart(merged: pd.DataFrame, area: str) -> go.Figure:
    """% delta bar chart. Orange = over-predict, Blue = under-predict."""
    colors = merged["delta_pct"].apply(
        lambda x: "rgba(230,81,0,0.80)" if x >= 0 else "rgba(25,118,210,0.80)"
    )
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.70, 0.30],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["% Error per slot  (orange = over-predict, blue = under-predict)",
                        "Cumulative error (EUR/MWh)"],
    )
    # Delta % bars
    fig.add_trace(go.Bar(
        x=merged["datetime"], y=merged["delta_pct"],
        marker_color=colors, name="Δ%",
        hovertemplate="%{x|%H:%M}<br>Δ = %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="grey", row=1, col=1)
    # ±5% / ±10% reference lines
    for lvl, dash in [(5, "dot"), (10, "dash")]:
        for sign in (1, -1):
            fig.add_hline(y=sign * lvl, line_dash=dash,
                          line_color="rgba(150,150,150,0.5)", row=1, col=1)
    # Cumulative EUR error
    fig.add_trace(go.Scatter(
        x=merged["datetime"], y=merged["delta"].cumsum(),
        mode="lines", name="Cumul. error",
        line=dict(color="#6A1B9A", width=2),
        fill="tozeroy",
        fillcolor="rgba(106,27,154,0.10)",
        hovertemplate="%{x|%H:%M}<br>Cumul. error: %{y:.1f} EUR/MWh<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="grey", row=2, col=1)
    fig.update_xaxes(tickformat="%H:%M", dtick=3600_000 * 2)
    fig.update_yaxes(title_text="Error (%)", row=1, col=1)
    fig.update_yaxes(title_text="EUR/MWh", row=2, col=1)
    fig.update_layout(
        height=580, margin=dict(l=50, r=20, t=60, b=40),
        title=f"{area} — Prediction Error Analysis",
        showlegend=False,
    )
    return fig


def _error_by_hour_chart(merged_all: dict[str, pd.DataFrame]) -> go.Figure:
    """
    Average absolute % error by hour-of-day for each area.
    Shows when the model is systematically least reliable.
    """
    fig = go.Figure()
    for area, merged in merged_all.items():
        if merged.empty:
            continue
        hourly = (
            merged.groupby("hour")["delta_pct"]
            .apply(lambda s: s.abs().mean())
            .reset_index()
        )
        fig.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["delta_pct"],
            mode="lines+markers",
            name=f"{area} — {AREA_NAMES[area]}",
            line=dict(color=AREA_COLORS[area], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{area}</b><br>Hour: %{{x}}:00<br>Avg |Δ|: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        title="Average Absolute Error by Hour-of-Day",
        xaxis=dict(title="Hour (CET)", tickmode="linear", dtick=2, range=[0, 23]),
        yaxis_title="Mean |Δ%|",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        height=400, margin=dict(l=50, r=20, t=50, b=60),
    )
    return fig


def _pnl_comparison_chart(pnl_results: dict[str, dict]) -> go.Figure:
    """
    Bar chart: Predicted P&L vs Actual P&L vs Optimal (perfect-foresight) P&L
    per area. In EUR per MW capacity per day.
    """
    areas      = list(pnl_results.keys())
    pred_vals  = [pnl_results[a]["predicted_pnl"]  for a in areas]
    act_vals   = [pnl_results[a].get("actual_pnl",  None) for a in areas]
    opt_vals   = [pnl_results[a].get("optimal_pnl", None) for a in areas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Predicted P&L",
        x=areas, y=pred_vals,
        marker_color=[AREA_COLORS[a] for a in areas],
        hovertemplate="<b>%{x}</b><br>Predicted P&L: €%{y:.0f}/MW<extra></extra>",
    ))
    if any(v is not None for v in act_vals):
        fig.add_trace(go.Bar(
            name="Actual P&L",
            x=areas, y=act_vals,
            marker_color="rgba(50,50,50,0.70)",
            hovertemplate="<b>%{x}</b><br>Actual P&L: €%{y:.0f}/MW<extra></extra>",
        ))
    if any(v is not None for v in opt_vals):
        fig.add_trace(go.Bar(
            name="Perfect-foresight P&L",
            x=areas, y=opt_vals,
            marker_color="rgba(46,125,50,0.70)",
            hovertemplate="<b>%{x}</b><br>Optimal P&L: €%{y:.0f}/MW<extra></extra>",
        ))
    fig.update_layout(
        title="Simulated Battery P&L — 1 MW, 4h charge + 4h discharge cycle (EUR/MW/day)",
        barmode="group",
        xaxis_title="Area",
        yaxis_title="EUR / MW / day",
        legend=dict(orientation="h", y=-0.15),
        height=400, margin=dict(l=50, r=20, t=60, b=60),
    )
    return fig


def _spread_chart(df: pd.DataFrame, valid_areas: list[str]) -> go.Figure:
    """
    Horizontal bar chart of price spread (max–min) per area,
    with colour showing mean price level.
    """
    rows = []
    for area in valid_areas:
        adf = df[df["area"] == area]["price_eur_mwh"]
        rows.append({
            "area":   area,
            "spread": round(float(adf.max() - adf.min()), 2),
            "mean":   round(float(adf.mean()), 2),
            "min":    round(float(adf.min()), 2),
            "max":    round(float(adf.max()), 2),
        })
    rows.sort(key=lambda r: r["spread"], reverse=True)

    fig = go.Figure(go.Bar(
        y=[r["area"] for r in rows],
        x=[r["spread"] for r in rows],
        orientation="h",
        marker=dict(
            color=[r["mean"] for r in rows],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Mean EUR/MWh"),
        ),
        text=[f"€{r['spread']:.1f}  ({r['min']:.1f} → {r['max']:.1f})" for r in rows],
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Spread: €%{x:.1f}/MWh<extra></extra>",
    ))
    fig.update_layout(
        title="Intraday Price Spread per Area (max – min EUR/MWh)",
        xaxis_title="Spread (EUR/MWh)",
        yaxis=dict(autorange="reversed"),
        height=300, margin=dict(l=70, r=20, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("📅 Tomorrow's Day-Ahead Price Forecast")
st.caption(
    "15-minute predictions for the next delivery day, generated automatically "
    "at **11:45 CET** before the 12:00 auction closes."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚡ Tomorrow's Forecast")
    st.markdown("---")
    selected_areas = st.multiselect(
        "Show areas",
        options=list(AREA_NAMES.keys()),
        default=list(AREA_NAMES.keys()),
        format_func=lambda k: f"{k} — {AREA_NAMES[k]}",
    )
    st.markdown("---")
    run_btn = st.button("🔄 Run predictions now", use_container_width=True)
    st.caption(
        "The model trains on all historical 15-min data + today's "
        "weather forecast and predicts tomorrow's 96 delivery slots."
    )

if run_btn:
    result = run_now()
    st.success(f"✅ Predictions updated for {result['delivery_day']}")
    st.rerun()

# ── Load prediction data ──────────────────────────────────────────────────────
data = load_predictions()

if data is None:
    st.info(
        "No predictions found yet. Click **🔄 Run predictions now** in the sidebar "
        "to generate the first forecast, or wait until 11:45 CET."
    )
    st.stop()

if not data.get("predictions"):
    st.warning("Predictions file exists but contains no results. Try running again.")
    st.stop()

df           = predictions_to_df(data)
delivery_day = data.get("delivery_day", "unknown")
generated_at = data.get("generated_at", "")

# ── Try to load actuals (silently; only shows if available) ───────────────────
actuals_dict = load_actuals_for_day(delivery_day)
actuals_available = bool(actuals_dict)

# ── Header metrics ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📆 Delivery Day", delivery_day)
with col2:
    try:
        gen_dt = datetime.fromisoformat(generated_at)
        st.metric("🕐 Generated (CET)", gen_dt.strftime("%Y-%m-%d %H:%M"))
    except Exception:
        st.metric("🕐 Generated", generated_at[:16] if generated_at else "—")
with col3:
    areas_ready = list(data["predictions"].keys())
    st.metric("🌍 Areas predicted", f"{len(areas_ready)} / 5")
with col4:
    status = "✅ Actuals available" if actuals_available else "⏳ Awaiting settlement"
    st.metric("📊 Accuracy check", status)

st.markdown("---")

if not selected_areas:
    st.warning("Select at least one area in the sidebar.")
    st.stop()

valid_areas = [a for a in selected_areas if a in data["predictions"]]

# ── Main forecast chart ───────────────────────────────────────────────────────
st.subheader("📈 All-Area Price Curve")
if valid_areas:
    st.plotly_chart(_forecast_chart(df, valid_areas), use_container_width=True)
else:
    st.warning("No predictions available for the selected areas.")

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.subheader("🌡️ Price Heatmap")
if valid_areas:
    st.plotly_chart(_heatmap_chart(df, valid_areas), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# 💹  TRADER DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("💹 Trader Dashboard")

# ── Compute battery P&L for all valid areas ───────────────────────────────────
pnl_results: dict[str, dict] = {}
merged_all:  dict[str, pd.DataFrame] = {}

for area in valid_areas:
    adf_pred = df[df["area"] == area].sort_values("datetime").reset_index(drop=True)
    act_df   = actuals_dict.get(area)
    if act_df is not None:
        merged = merge_pred_actual(adf_pred, act_df)
        merged_all[area] = merged
        pnl_results[area] = battery_pnl(
            adf_pred["price_eur_mwh"],
            actual_prices=act_df["price_actual"].reset_index(drop=True)
            if len(act_df) == len(adf_pred)
            else None,
        )
    else:
        pnl_results[area] = battery_pnl(adf_pred["price_eur_mwh"])

# ── Section A: Accuracy (only when actuals are available) ─────────────────────
if actuals_available and merged_all:
    st.markdown("#### 🎯 Prediction Accuracy vs Actuals")

    # Per-area accuracy KPI row
    areas_with_actuals = [a for a in valid_areas if a in merged_all]
    kpi_tabs = st.tabs([f"{a} — {AREA_NAMES[a]}" for a in areas_with_actuals])

    for tab, area in zip(kpi_tabs, areas_with_actuals):
        with tab:
            m = merged_all[area]
            metrics = accuracy_metrics(m)

            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            with c1:
                st.metric("MAE (€/MWh)", metrics.get("MAE", "—"))
            with c2:
                st.metric("RMSE (€/MWh)", metrics.get("RMSE", "—"))
            with c3:
                mape = metrics.get("MAPE %", 0)
                st.metric("MAPE", f"{mape:.1f}%", delta=None)
            with c4:
                bias = metrics.get("Bias", 0)
                bias_label = "Over-pred" if bias > 0 else "Under-pred"
                st.metric("Bias (€/MWh)", f"{bias:+.2f}", help=bias_label)
            with c5:
                st.metric("Max over (€)", f"+{metrics.get('Max over', 0):.1f}")
            with c6:
                st.metric("Max under (€)", f"{metrics.get('Max under', 0):.1f}")
            with c7:
                r = metrics.get("Pearson r", 0)
                st.metric("Pearson r", f"{r:.3f}")

            # Comparison chart tabs
            chart_tab_a, chart_tab_b = st.tabs(
                ["📈 Predicted vs Actual", "📉 % Delta & Cumulative Error"]
            )
            with chart_tab_a:
                st.plotly_chart(_comparison_overlay_chart(m, area), use_container_width=True)
            with chart_tab_b:
                st.plotly_chart(_delta_pct_chart(m, area), use_container_width=True)

    # Error-by-hour across all areas
    st.markdown("##### ⏱️ Where does the model struggle? (Error by Hour-of-Day)")
    st.caption("Lower is better. Spikes show hours where the XGBoost model is systematically less reliable.")
    st.plotly_chart(_error_by_hour_chart(merged_all), use_container_width=True)

else:
    st.info(
        f"📌 **Actuals not yet available** for {delivery_day}. "
        "Once the delivery day settles and Nord Pool prices are ingested into the CSV "
        "(`dayahead_prices.csv`), this section will automatically show the accuracy check."
    )

# ── Section B: Battery P&L simulation (always visible) ───────────────────────
st.markdown("#### 🔋 Battery P&L Simulation (1 MW · 4h charge + 4h discharge)")
st.caption(
    "Strategy: charge during the cheapest predicted 4-hour window, discharge during "
    "the most expensive non-overlapping 4-hour window. "
    + ("Actual and perfect-foresight P&L shown for comparison." if actuals_available
       else "Actuals not yet available — predicted P&L only.")
)

if pnl_results:
    st.plotly_chart(_pnl_comparison_chart(pnl_results), use_container_width=True)

    # Detailed windows table
    window_rows = []
    slot_mins = 15
    for area in valid_areas:
        p = pnl_results.get(area, {})
        if not p:
            continue
        adf_s = df[df["area"] == area].sort_values("datetime").reset_index(drop=True)
        c_time = adf_s.iloc[p["charge_start"]]["datetime"].strftime("%H:%M")
        c_end  = adf_s.iloc[p["charge_end"]  ]["datetime"].strftime("%H:%M")
        d_time = adf_s.iloc[p["discharge_start"]]["datetime"].strftime("%H:%M")
        d_end  = adf_s.iloc[p["discharge_end"]  ]["datetime"].strftime("%H:%M")
        row = {
            "Area": area,
            "⚡ Charge window":  f"{c_time} – {c_end}",
            "Avg charge (€/MWh)": p["avg_charge_pred"],
            "💰 Discharge window": f"{d_time} – {d_end}",
            "Avg discharge (€/MWh)": p["avg_discharge_pred"],
            "Predicted P&L (€/MW)": p["predicted_pnl"],
        }
        if "actual_pnl" in p:
            row["Actual P&L (€/MW)"]    = p["actual_pnl"]
            row["Optimal P&L (€/MW)"]   = p.get("optimal_pnl", "—")
            eff = (p["actual_pnl"] / p["optimal_pnl"] * 100) if p.get("optimal_pnl", 0) != 0 else 0
            row["Strategy efficiency %"] = f"{eff:.0f}%"
        window_rows.append(row)

    st.dataframe(
        pd.DataFrame(window_rows).set_index("Area"),
        use_container_width=True,
    )

# ── Section C: Price spread per area ─────────────────────────────────────────
st.markdown("#### 📊 Intraday Spread Analysis")
st.caption(
    "Wider spread = higher arbitrage potential. "
    "The shade inside each bar reflects the mean price level (green = cheap, red = expensive)."
)
if valid_areas:
    st.plotly_chart(_spread_chart(df, valid_areas), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# Summary statistics (existing)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Summary Statistics")
stats = data.get("area_stats", {})
if stats:
    rows = []
    for area in valid_areas:
        if area not in stats:
            continue
        s = stats[area]
        peak_t   = df[df["area"] == area].sort_values("datetime").iloc[s["peak_period"]]["datetime"].strftime("%H:%M")
        trough_t = df[df["area"] == area].sort_values("datetime").iloc[s["trough_period"]]["datetime"].strftime("%H:%M")
        rows.append({
            "Area":             f"{area} — {AREA_NAMES[area]}",
            "Min (€/MWh)":  s["min"],
            "P25 (€/MWh)":  s["p25"],
            "Mean (€/MWh)": s["mean"],
            "P75 (€/MWh)":  s["p75"],
            "Max (€/MWh)":  s["max"],
            "Peak (CET)":   peak_t,
            "Trough (CET)": trough_t,
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("Area"),
        use_container_width=True,
    )

# ── Area deep-dive (existing, with charge/discharge overlay) ──────────────────
if valid_areas:
    st.markdown("---")
    st.subheader("🔍 Area Deep-Dive")
    tabs = st.tabs([f"{a} — {AREA_NAMES[a]}" for a in valid_areas])
    for tab, area in zip(tabs, valid_areas):
        with tab:
            if area not in data["predictions"]:
                st.warning(f"No predictions for {area}")
                continue
            pnl = pnl_results.get(area)
            st.plotly_chart(_area_profile_chart(df, area, pnl=pnl), use_container_width=True)
            adf = (
                df[df["area"] == area]
                .sort_values("datetime")
                .assign(time=lambda d: d["datetime"].dt.strftime("%H:%M"))
                [["time", "price_eur_mwh"]]
                .rename(columns={"time": "Delivery (CET)", "price_eur_mwh": "Price (€/MWh)"})
                .reset_index(drop=True)
            )
            with st.expander("Show all 96 quarter-hour slots"):
                cols = st.columns(4)
                chunk = 24
                for ci, col in enumerate(cols):
                    with col:
                        st.dataframe(adf.iloc[ci * chunk:(ci + 1) * chunk],
                                     hide_index=True, use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⬇️ Download Predictions")

dl_rows = []
for area, slots in data["predictions"].items():
    if area not in selected_areas:
        continue
    for s in slots:
        dl_rows.append({
            "delivery_day":   delivery_day,
            "area":           area,
            "area_name":      AREA_NAMES.get(area, area),
            "delivery_start": s["datetime"],
            "price_eur_mwh":  s["price_eur_mwh"],
        })

if dl_rows:
    dl_df      = pd.DataFrame(dl_rows)
    csv_bytes  = dl_df.to_csv(index=False).encode()
    json_bytes = json.dumps(data, indent=2).encode()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label=f"📥 Download forecast CSV ({len(dl_rows)} rows)",
            data=csv_bytes,
            file_name=f"da_forecast_{delivery_day}.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            label="📥 Download full JSON",
            data=json_bytes,
            file_name=f"da_forecast_{delivery_day}.json",
            mime="application/json",
        )

st.markdown("---")
st.caption(
    "Model: XGBoost trained on 15-min historical day-ahead prices · "
    "Weather: Open-Meteo forecast API · "
    "Features: calendar cyclicals, weather, 24/48/168-h price lags · "
    "Runs automatically at 11:45 CET daily"
)
