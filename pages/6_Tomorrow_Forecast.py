"""
6_Tomorrow_Forecast.py
======================
Streamlit page that shows the automatic 15-minute day-ahead price forecast
for tomorrow across AT, BE, FR, GER, and NL.

The forecast is generated daily at 11:45 CET by the embedded APScheduler
(see streamlit_app.py).  Users can also trigger a manual re-run from this page.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Tomorrow's Forecast",
    page_icon="📅",
)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "predictions_tomorrow.json")

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

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
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
    """Trigger a fresh prediction run and return the result."""
    from predict_tomorrow import run_predictions
    with st.spinner("⏳ Training models and fetching weather forecast…  (30–90 s)"):
        result = run_predictions()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Charts
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
                "%{x|%H:%M}<br>"
                "%{y:.1f} EUR/MWh<extra></extra>"
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
    fig.update_xaxes(tickformat="%H:%M", dtick=3600_000 * 2)  # tick every 2 h
    return fig


def _area_profile_chart(df: pd.DataFrame, area: str) -> go.Figure:
    adf = df[df["area"] == area].sort_values("datetime")
    color = AREA_COLORS[area]
    fig = go.Figure()
    # Shaded fill under the line
    fig.add_trace(go.Scatter(
        x=adf["datetime"],
        y=adf["price_eur_mwh"],
        mode="lines",
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.12)").replace("rgb(", "rgba("),
        line=dict(color=color, width=2.5),
        name="Price",
        hovertemplate="%{x|%H:%M}  %{y:.1f} EUR/MWh<extra></extra>",
    ))
    # Peak & trough markers
    peak_idx   = adf["price_eur_mwh"].idxmax()
    trough_idx = adf["price_eur_mwh"].idxmin()
    for idx, sym, label in [(peak_idx, "triangle-up", "Peak"),
                             (trough_idx, "triangle-down", "Trough")]:
        row = adf.loc[idx]
        fig.add_trace(go.Scatter(
            x=[row["datetime"]],
            y=[row["price_eur_mwh"]],
            mode="markers+text",
            marker=dict(size=12, color=color, symbol=sym),
            text=[f"{label}: {row['price_eur_mwh']:.1f}"],
            textposition="top center",
            showlegend=False,
            hovertemplate=f"{label}: %{{y:.1f}} EUR/MWh @ %{{x|%H:%M}}<extra></extra>",
        ))
    fig.update_layout(
        title=f"{AREA_NAMES[area]} ({area}) — Hourly Profile",
        xaxis_title="Delivery time (CET)",
        yaxis_title="Price (EUR/MWh)",
        **CHART_CFG,
    )
    fig.update_xaxes(tickformat="%H:%M", dtick=3600_000 * 2)
    return fig


def _heatmap_chart(df: pd.DataFrame, selected_areas: list[str]) -> go.Figure:
    """Price heatmap: areas × hour-of-day."""
    piv = (
        df[df["area"].isin(selected_areas)]
        .assign(hour=lambda d: d["datetime"].dt.strftime("%H:%M"))
        .pivot_table(index="area", columns="hour", values="price_eur_mwh", aggfunc="mean")
    )
    fig = go.Figure(go.Heatmap(
        z=piv.values,
        x=piv.columns.tolist(),
        y=piv.index.tolist(),
        colorscale="RdYlGn_r",
        hoverongaps=False,
        hovertemplate="Area: %{y}<br>Time: %{x}<br>Price: %{z:.1f} EUR/MWh<extra></extra>",
        colorbar=dict(title="EUR/MWh"),
    ))
    fig.update_layout(
        title="Price Heatmap — Areas × Quarter-Hour",
        xaxis_title="Delivery time (CET)",
        xaxis=dict(tickangle=-45, nticks=24),
        height=280,
        margin=dict(l=60, r=20, t=50, b=60),
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

# ── Load data ────────────────────────────────────────────────────────────────
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

df = predictions_to_df(data)
delivery_day = data.get("delivery_day", "unknown")
generated_at = data.get("generated_at", "")

# Header info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📆 Delivery Day", delivery_day)
with col2:
    try:
        gen_dt = datetime.fromisoformat(generated_at)
        st.metric("🕐 Generated at (CET)", gen_dt.strftime("%Y-%m-%d %H:%M"))
    except Exception:
        st.metric("🕐 Generated at", generated_at[:16] if generated_at else "—")
with col3:
    areas_ready = list(data["predictions"].keys())
    st.metric("🌍 Areas predicted", f"{len(areas_ready)} / 5")

st.markdown("---")

if not selected_areas:
    st.warning("Select at least one area in the sidebar.")
    st.stop()

# ── Main forecast chart ───────────────────────────────────────────────────────
st.subheader("📈 All-Area Price Curve")
valid_areas = [a for a in selected_areas if a in data["predictions"]]
if valid_areas:
    st.plotly_chart(_forecast_chart(df, valid_areas), use_container_width=True)
else:
    st.warning("No predictions available for the selected areas.")

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.subheader("🌡️ Price Heatmap")
if valid_areas:
    st.plotly_chart(_heatmap_chart(df, valid_areas), use_container_width=True)

# ── Summary table ─────────────────────────────────────────────────────────────
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
            "Area":      f"{area} — {AREA_NAMES[area]}",
            "Min (EUR/MWh)":  s["min"],
            "P25 (EUR/MWh)":  s["p25"],
            "Mean (EUR/MWh)": s["mean"],
            "P75 (EUR/MWh)":  s["p75"],
            "Max (EUR/MWh)":  s["max"],
            "Peak time (CET)":   peak_t,
            "Trough time (CET)": trough_t,
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("Area"),
        use_container_width=True,
    )

# ── Area-by-area deep dive ────────────────────────────────────────────────────
if len(valid_areas) > 0:
    st.subheader("🔍 Area Deep-Dive")
    tabs = st.tabs([f"{a} — {AREA_NAMES[a]}" for a in valid_areas])
    for tab, area in zip(tabs, valid_areas):
        with tab:
            if area not in data["predictions"]:
                st.warning(f"No predictions for {area}")
                continue
            st.plotly_chart(_area_profile_chart(df, area), use_container_width=True)
            # Raw table
            adf = (
                df[df["area"] == area]
                .sort_values("datetime")
                .assign(time=lambda d: d["datetime"].dt.strftime("%H:%M"))
                [["time", "price_eur_mwh"]]
                .rename(columns={"time": "Delivery (CET)", "price_eur_mwh": "Price (EUR/MWh)"})
                .reset_index(drop=True)
            )
            with st.expander("Show all 96 quarter-hour slots"):
                # Display in 4 columns of 24 rows
                cols = st.columns(4)
                chunk = 24
                for ci, col in enumerate(cols):
                    with col:
                        st.dataframe(
                            adf.iloc[ci * chunk: (ci + 1) * chunk],
                            hide_index=True,
                            use_container_width=True,
                        )

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⬇️ Download Predictions")

# Build a combined CSV
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
    dl_df = pd.DataFrame(dl_rows)
    csv_bytes = dl_df.to_csv(index=False).encode()
    st.download_button(
        label=f"📥 Download forecast CSV ({len(dl_rows)} rows)",
        data=csv_bytes,
        file_name=f"da_forecast_{delivery_day}.csv",
        mime="text/csv",
    )

    # JSON download
    json_bytes = json.dumps(data, indent=2).encode()
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
