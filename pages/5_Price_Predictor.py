"""
5_Price_Predictor.py — Day-Ahead Electricity Price Predictor
============================================================
Correlates hourly day-ahead prices with Open-Meteo weather signals
and calendar features to train XGBoost, LightGBM, and Ridge models.
Train/test split is chronological and user-selectable.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Price Predictor", page_icon="🔮")
st.markdown(
    """
    <style>
    #watt-header{position:fixed;top:52px;left:0;right:0;z-index:1000000;background:#fff;border-bottom:2px solid #e2e8f0;padding:10px 2.5rem;display:flex;align-items:center;gap:14px;box-shadow:0 2px 8px rgba(0,0,0,.06);}
    [data-testid="stMainBlockContainer"]{padding-top:80px!important;}
    </style>
    <div id="watt-header">
      <div style="background:linear-gradient(135deg,#1565C0,#0D47A1);border-radius:10px;width:42px;height:42px;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 2px 8px rgba(21,101,192,.3)">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="22" height="22"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
      </div>
      <div>
        <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:1.3rem;font-weight:700;color:#0D1B3E;letter-spacing:-0.2px;line-height:1.2">Watt Happens</div>
        <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:0.75rem;color:#64748b;margin-top:1px">BESS Strategy &amp; Energy Intelligence Platform</div>
      </div>
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


PALETTE = {
    "primary":  ["#0D47A1", "#1565C0", "#1976D2", "#1E88E5", "#42A5F5"],
    "accent":   ["#E65100", "#EF6C00", "#F57C00", "#FB8C00", "#FFA726"],
    "green":    ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A"],
    "test":     "#E65100",
    "train":    "#1976D2",
    "pred":     "#FFA726",
}

CHART_CFG = {"height": 400, "margin": dict(l=40, r=20, t=45, b=35)}

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
AREA_COORDS = {
    "AT":  {"lat": 48.21, "lon": 16.37, "name": "Austria"},
    "BE":  {"lat": 50.85, "lon":  4.35, "name": "Belgium"},
    "FR":  {"lat": 48.85, "lon":  2.35, "name": "France"},
    "GER": {"lat": 52.52, "lon": 13.41, "name": "Germany"},
    "NL":  {"lat": 52.37, "lon":  4.90, "name": "Netherlands"},
}

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "dayahead_prices.csv"
)

WEATHER_VARS = (
    "temperature_2m,windspeed_10m,windspeed_100m,"
    "shortwave_radiation,cloudcover,precipitation"
)

FEATURE_COLS = [
    # Calendar
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Weather
    "temperature_2m", "windspeed_10m", "windspeed_100m",
    "shortwave_radiation", "cloudcover", "precipitation",
    # Price lags
    "lag_24h", "lag_48h", "lag_168h", "rolling_mean_24h",
]

TARGET = "price_eur_mwh"

FEATURE_LABELS = {
    "hour_of_day": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
    "is_weekend": "Weekend",
    "hour_sin": "Hour (sin)",
    "hour_cos": "Hour (cos)",
    "dow_sin": "DoW (sin)",
    "dow_cos": "DoW (cos)",
    "temperature_2m": "Temperature (°C)",
    "windspeed_10m": "Wind 10m (km/h)",
    "windspeed_100m": "Wind 100m (km/h)",
    "shortwave_radiation": "Solar Radiation",
    "cloudcover": "Cloud Cover (%)",
    "precipitation": "Precipitation (mm)",
    "lag_24h": "Price Lag 24h",
    "lag_48h": "Price Lag 48h",
    "lag_168h": "Price Lag 168h (1 week)",
    "rolling_mean_24h": "Rolling Mean 24h",
}

MODEL_COLORS = {
    "XGBoost":  "#1976D2",
    "LightGBM": "#43A047",
    "Ridge":    "#E65100",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_hourly_prices() -> pd.DataFrame:
    """Load 15-min day-ahead prices and aggregate to hourly means."""
    df = pd.read_csv(DATA_PATH)
    df["dt"] = pd.to_datetime(df["deliveryStartCET"], utc=True).dt.tz_convert(None)
    df["hour"] = df["dt"].dt.floor("h")
    hourly = (
        df.groupby(["area", "hour"])["price"]
        .mean()
        .reset_index()
        .rename(columns={"hour": "datetime", "price": TARGET})
    )
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    return hourly


@st.cache_data(show_spinner=False)
def fetch_weather(area: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo historical archive (free, no API key)."""
    coords = AREA_COORDS[area]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_VARS,
        "timezone": "Europe/Berlin",
        "wind_speed_unit": "kmh",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    wdf = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "windspeed_10m": hourly["windspeed_10m"],
        "windspeed_100m": hourly["windspeed_100m"],
        "shortwave_radiation": hourly["shortwave_radiation"],
        "cloudcover": hourly["cloudcover"],
        "precipitation": hourly["precipitation"],
    })
    return wdf


def build_features(prices_df: pd.DataFrame, area: str) -> pd.DataFrame:
    """Merge prices with weather + add calendar & lag features."""
    df = (
        prices_df[prices_df["area"] == area]
        .copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if df.empty:
        return df

    start_str = df["datetime"].min().strftime("%Y-%m-%d")
    end_str   = df["datetime"].max().strftime("%Y-%m-%d")
    wdf = fetch_weather(area, start_str, end_str)

    df = df.merge(wdf, on="datetime", how="left")

    # Calendar
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Price lag features (valid for backtesting & day-ahead forecasting)
    df["lag_24h"]          = df[TARGET].shift(24)
    df["lag_48h"]          = df[TARGET].shift(48)
    df["lag_168h"]         = df[TARGET].shift(168)
    df["rolling_mean_24h"] = df[TARGET].shift(1).rolling(24).mean()

    return df.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE %": mape}


def train_and_evaluate(df: pd.DataFrame, train_pct: int, models_to_run: list) -> dict:
    """Train each model on chronological split and return evaluation results."""
    n_total = len(df)
    n_train = max(168, int(n_total * train_pct / 100))

    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df  = df.iloc[n_train:].reset_index(drop=True)

    X_tr = train_df[FEATURE_COLS].values
    y_tr = train_df[TARGET].values
    X_te = test_df[FEATURE_COLS].values
    y_te = test_df[TARGET].values

    # Scaler for Ridge
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    results = {}

    for mname in models_to_run:
        if mname == "XGBoost":
            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, gamma=0.1,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)
            raw_imp = model.feature_importances_
            feat_imp = {FEATURE_LABELS[f]: v for f, v in zip(FEATURE_COLS, raw_imp / raw_imp.sum())}

        elif mname == "LightGBM":
            model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1,
            )
            model.fit(X_tr, y_tr)
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)
            raw_imp = model.feature_importances_.astype(float)
            feat_imp = {FEATURE_LABELS[f]: v for f, v in zip(FEATURE_COLS, raw_imp / raw_imp.sum())}

        elif mname == "Ridge":
            model = Ridge(alpha=10.0)
            model.fit(X_tr_sc, y_tr)
            tr_pred = model.predict(X_tr_sc)
            te_pred = model.predict(X_te_sc)
            abs_c = np.abs(model.coef_)
            feat_imp = {FEATURE_LABELS[f]: v for f, v in zip(FEATURE_COLS, abs_c / abs_c.sum())}

        results[mname] = {
            "train_metrics": _metrics(y_tr, tr_pred),
            "test_metrics":  _metrics(y_te, te_pred),
            "train_pred": tr_pred,
            "test_pred":  te_pred,
            "train_df": train_df,
            "test_df":  test_df,
            "feature_importance": feat_imp,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def _actual_vs_pred(res: dict, mname: str, area: str) -> go.Figure:
    trd = res["train_df"].copy(); trd["pred"] = res["train_pred"]
    ted = res["test_df"].copy();  ted["pred"] = res["test_pred"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trd["datetime"], y=trd[TARGET],
        name="Actual (Train)", line=dict(color="#90CAF9", width=1), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=trd["datetime"], y=trd["pred"],
        name="Predicted (Train)", line=dict(color="#FFA726", width=1, dash="dot"), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=ted["datetime"], y=ted[TARGET],
        name="Actual (Test)", line=dict(color=PALETTE["train"], width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=ted["datetime"], y=ted["pred"],
        name="Predicted (Test)", line=dict(color=PALETTE["test"], width=1.8, dash="dash"),
    ))
    if len(trd) > 0:
        fig.add_vline(
            x=trd["datetime"].iloc[-1].isoformat(), line_dash="dot", line_color="#9E9E9E",
            annotation_text="← Train | Test →", annotation_position="top right",
        )
    fig.update_layout(
        title=f"{mname} — Actual vs Predicted ({area})",
        yaxis_title="€/MWh", **CHART_CFG,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _feature_importance(feat_imp: dict, mname: str) -> go.Figure:
    items = sorted(feat_imp.items(), key=lambda x: x[1])
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    n = len(labels)
    colors = [PALETTE["primary"][i % 5] for i in range(n)]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=f"{mname} — Feature Importance",
        xaxis_title="Relative Importance",
        height=max(350, n * 22 + 80),
        margin=dict(l=180, r=30, t=45, b=35),
    )
    return fig


def _scatter(res: dict, mname: str) -> go.Figure:
    ted = res["test_df"].copy(); ted["pred"] = res["test_pred"]
    lo = min(ted[TARGET].min(), ted["pred"].min())
    hi = max(ted[TARGET].max(), ted["pred"].max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ted[TARGET], y=ted["pred"], mode="markers",
        marker=dict(color=MODEL_COLORS.get(mname, "#999"), opacity=0.4, size=4),
        name="Test points",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#9E9E9E", dash="dash"), name="Perfect fit",
    ))
    r2 = res["test_metrics"]["R²"]
    fig.update_layout(
        title=f"{mname} — Predicted vs Actual (R²={r2:.3f})",
        xaxis_title="Actual (€/MWh)", yaxis_title="Predicted (€/MWh)",
        **CHART_CFG,
    )
    return fig


def _metrics_comparison(all_results: dict) -> go.Figure:
    models = list(all_results.keys())
    metric_keys = ["MAE", "RMSE", "MAPE %"]
    metric_labels = ["MAE (€/MWh)", "RMSE (€/MWh)", "MAPE (%)"]

    fig = go.Figure()
    for i, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        vals = [all_results[m]["test_metrics"][mk] for m in models]
        fig.add_trace(go.Bar(
            name=ml, x=models, y=vals,
            marker_color=PALETTE["primary"][i],
            text=[f"{v:.2f}" for v in vals], textposition="outside",
        ))
    fig.update_layout(
        barmode="group", title="Test-Set Metric Comparison",
        yaxis_title="Value", **CHART_CFG,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _error_by_hour(all_results: dict) -> go.Figure:
    fig = go.Figure()
    for mname, res in all_results.items():
        ted = res["test_df"].copy()
        ted["pred"] = res["test_pred"]
        ted["abs_err"] = (ted[TARGET] - ted["pred"]).abs()
        hourly_err = ted.groupby("hour_of_day")["abs_err"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=hourly_err["hour_of_day"], y=hourly_err["abs_err"],
            name=mname, mode="lines+markers",
            line=dict(color=MODEL_COLORS.get(mname, "#999")),
        ))
    fig.update_layout(
        title="Mean Absolute Error by Hour of Day",
        xaxis_title="Hour (CET)", yaxis_title="MAE (€/MWh)",
        **CHART_CFG,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Price Predictor")

all_prices = load_hourly_prices()
all_areas  = sorted(all_prices["area"].unique())

selected_area = st.sidebar.selectbox(
    "Bidding Area",
    options=all_areas,
    format_func=lambda a: f"{a} — {AREA_COORDS[a]['name']}",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Train / Test Split")
train_pct = st.sidebar.slider(
    "Training data (%)",
    min_value=50, max_value=90, value=80, step=5,
    help="Chronological split. First X% of hours train the model; remaining X% validate it out-of-sample.",
)
st.sidebar.caption(f"Test (out-of-sample): {100 - train_pct}%")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Models")
models_to_run = st.sidebar.multiselect(
    "Select models",
    options=["XGBoost", "LightGBM", "Ridge"],
    default=["XGBoost", "LightGBM", "Ridge"],
)

with st.sidebar.expander("⚙️ Advanced: Feature info"):
    st.markdown("""
**Weather** (Open-Meteo historical API):
- Temperature 2m (°C)
- Wind speed 10m & 100m (km/h)
- Shortwave solar radiation (W/m²)
- Cloud cover (%)
- Precipitation (mm)

**Calendar:**
- Hour, day-of-week, month
- Weekend flag
- Cyclical sin/cos encoding

**Price lags:**
- 24h, 48h, 168h prior price
- 24h rolling mean
""")

run_clicked = st.sidebar.button("🚀 Run Prediction", type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔮 Day-Ahead Price Predictor")
st.markdown(
    "Predicts hourly day-ahead electricity prices using **weather signals** "
    "(Open-Meteo) and **calendar features**, validated on a chronological out-of-sample test set."
)

if not run_clicked:
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1** — Select a bidding area")
    col2.info("**Step 2** — Set your train/test split and pick models")
    col3.info("**Step 3** — Click 🚀 Run Prediction")
    st.stop()

# ── Validation
if not models_to_run:
    st.error("Please select at least one model in the sidebar.")
    st.stop()

# ── Build features
with st.spinner(f"Fetching Open-Meteo weather for {AREA_COORDS[selected_area]['name']}…"):
    try:
        df = build_features(all_prices, selected_area)
    except Exception as e:
        st.error(f"Weather fetch failed: {e}")
        st.stop()

if df.empty or len(df) < 200:
    st.error("Not enough data after feature engineering. Try a wider date range.")
    st.stop()

# ── Data summary
n_total = len(df)
n_train = max(168, int(n_total * train_pct / 100))
n_test  = n_total - n_train
train_end   = df["datetime"].iloc[n_train - 1].strftime("%Y-%m-%d")
test_start  = df["datetime"].iloc[n_train].strftime("%Y-%m-%d")
date_min    = df["datetime"].min().strftime("%Y-%m-%d")
date_max    = df["datetime"].max().strftime("%Y-%m-%d")

st.subheader(f"📍 {AREA_COORDS[selected_area]['name']} ({selected_area})")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total hours",  f"{n_total:,}")
c2.metric("Train hours",  f"{n_train:,}  ({train_pct}%)")
c3.metric("Test hours",   f"{n_test:,}  ({100-train_pct}%)")
c4.metric("Avg price",    f"€{df[TARGET].mean():.1f}/MWh")

col_b1, col_b2 = st.columns(2)
with col_b1:
    st.info(f"**🏋 Train period:** {date_min} → {train_end}")
with col_b2:
    st.warning(f"**🧪 Test period:** {test_start} → {date_max}")

# ── Train models
with st.spinner(f"Training {', '.join(models_to_run)} on {n_train:,} hours…"):
    try:
        all_results = train_and_evaluate(df, train_pct, models_to_run)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Results — Metrics table
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Test-Set Performance")

rows = []
for mname, res in all_results.items():
    m = res["test_metrics"]
    rows.append({
        "Model": mname,
        "MAE (€/MWh)": round(m["MAE"], 2),
        "RMSE (€/MWh)": round(m["RMSE"], 2),
        "R²": round(m["R²"], 4),
        "MAPE (%)": round(m["MAPE %"], 2),
    })
metrics_df = pd.DataFrame(rows)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with st.expander("📋 Full metrics (train + test)"):
    rows2 = []
    for mname, res in all_results.items():
        for split, m in [("Train", res["train_metrics"]), ("Test", res["test_metrics"])]:
            rows2.append({
                "Model": mname, "Split": split,
                "MAE": round(m["MAE"], 2), "RMSE": round(m["RMSE"], 2),
                "R²": round(m["R²"], 4),  "MAPE %": round(m["MAPE %"], 2),
            })
    st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Comparison charts
# ─────────────────────────────────────────────────────────────────────────────
if len(all_results) > 1:
    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1:
        st.plotly_chart(_metrics_comparison(all_results), use_container_width=True)
    with col_cmp2:
        st.plotly_chart(_error_by_hour(all_results), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Per-model deep dive
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Per-Model Analysis")

tabs = st.tabs([f"📊 {m}" for m in models_to_run])
for tab, mname in zip(tabs, models_to_run):
    res = all_results[mname]
    with tab:
        # KPI row
        kc1, kc2, kc3, kc4 = st.columns(4)
        tm = res["test_metrics"]; trm = res["train_metrics"]
        kc1.metric("Test MAE",   f"{tm['MAE']:.2f} €/MWh",
                   delta=f"{tm['MAE'] - trm['MAE']:+.2f} vs train")
        kc2.metric("Test RMSE",  f"{tm['RMSE']:.2f} €/MWh",
                   delta=f"{tm['RMSE'] - trm['RMSE']:+.2f} vs train")
        kc3.metric("Test R²",    f"{tm['R²']:.4f}",
                   delta=f"{tm['R²'] - trm['R²']:+.4f} vs train")
        kc4.metric("Test MAPE",  f"{tm['MAPE %']:.2f}%",
                   delta=f"{tm['MAPE %'] - trm['MAPE %']:+.2f}pp vs train")

        # Actual vs Predicted (full period)
        st.plotly_chart(_actual_vs_pred(res, mname, selected_area), use_container_width=True)

        # Feature importance + scatter
        col_fi, col_sc = st.columns(2)
        with col_fi:
            st.plotly_chart(_feature_importance(res["feature_importance"], mname), use_container_width=True)
        with col_sc:
            st.plotly_chart(_scatter(res, mname), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Best model spotlight + download
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
best_model = min(all_results.keys(), key=lambda m: all_results[m]["test_metrics"]["MAE"])
best_res   = all_results[best_model]
st.subheader(f"🏆 Best Model on Test Set: {best_model}")
st.markdown(
    f"**{best_model}** achieves the lowest MAE of "
    f"**€{best_res['test_metrics']['MAE']:.2f}/MWh** on the test period "
    f"({test_start} → {date_max}), with R² = {best_res['test_metrics']['R²']:.4f}."
)

out_df = best_res["test_df"][["datetime", TARGET]].copy()
out_df["predicted_price"] = best_res["test_pred"].round(2)
out_df["area"] = selected_area
out_df["error_eur_mwh"] = (out_df[TARGET] - out_df["predicted_price"]).round(2)
csv_out = out_df.to_csv(index=False)

st.download_button(
    f"📥 Download {best_model} Test Predictions",
    csv_out,
    file_name=f"predictions_{selected_area}_{best_model.lower()}.csv",
    mime="text/csv",
)
