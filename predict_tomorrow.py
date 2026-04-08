"""
predict_tomorrow.py
===================
Daily 15-minute day-ahead price predictor for AT, BE, FR, GER, NL.

Trains one XGBoost model per bidding area on all available historical
15-min data, then predicts the 96 quarter-hour delivery slots for the
next calendar day.

Designed to run at 11:45 CET — 15 minutes before the 12:00 day-ahead
auction closes — giving traders a full-day price curve to bid on.

Usage (standalone):
    python predict_tomorrow.py

Or import and call:
    from predict_tomorrow import run_predictions
    run_predictions()

Output: data/predictions_tomorrow.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
PRICE_CSV   = os.path.join(DATA_DIR, "dayahead_prices.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "predictions_tomorrow.json")

AREAS = ["AT", "BE", "FR", "GER", "NL"]

AREA_COORDS = {
    "AT":  {"lat": 48.21, "lon": 16.37},
    "BE":  {"lat": 50.85, "lon":  4.35},
    "FR":  {"lat": 48.85, "lon":  2.35},
    "GER": {"lat": 52.52, "lon": 13.41},
    "NL":  {"lat": 52.37, "lon":  4.90},
}

# Open-Meteo archive uses legacy names; we rename to a consistent internal set
ARCHIVE_WEATHER_VARS = (
    "temperature_2m,windspeed_10m,windspeed_100m,"
    "shortwave_radiation,cloudcover,precipitation"
)
ARCHIVE_RENAME = {
    "windspeed_10m":  "wind_10m",
    "windspeed_100m": "wind_100m",
    "cloudcover":     "cloud_cover",
}

# Open-Meteo forecast API uses slightly different names
FORECAST_WEATHER_VARS = (
    "temperature_2m,wind_speed_10m,wind_speed_80m,"
    "shortwave_radiation,cloud_cover,precipitation"
)
FORECAST_RENAME = {
    "wind_speed_10m": "wind_10m",
    "wind_speed_80m": "wind_100m",
}

WEATHER_FEAT_COLS = [
    "temperature_2m", "wind_10m", "wind_100m",
    "shortwave_radiation", "cloud_cover", "precipitation",
]

# Lag periods (in 15-min units): 24 h = 96, 48 h = 192, 168 h = 672
LAG_MAP = {"lag_96": 96, "lag_192": 192, "lag_672": 672}
ROLL_WIN = 96  # 24-h rolling mean

FEATURE_COLS = [
    # Calendar
    "period_of_day", "hour_of_day", "quarter_of_hour",
    "day_of_week", "month", "is_weekend",
    "period_sin", "period_cos",
    "hour_sin",   "hour_cos",
    "dow_sin",    "dow_cos",
    # Weather
    *WEATHER_FEAT_COLS,
    # Lag
    "lag_96", "lag_192", "lag_672", "rolling_mean_96",
]

TARGET = "price"

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    tree_method="hist",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PREDICT] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Load all historical 15-min day-ahead prices; parse mixed-timezone CET/CEST."""
    df = pd.read_csv(PRICE_CSV)
    df["datetime"] = (
        pd.to_datetime(df["deliveryStartCET"], utc=True)
        .dt.tz_convert(None)
    )
    df = (
        df[df["area"].isin(AREAS)][["area", "datetime", "price"]]
        .copy()
        .sort_values(["area", "datetime"])
        .reset_index(drop=True)
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Weather helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_15min(wdf: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns and resample hourly weather to 15-min by forward-fill."""
    wdf = wdf.rename(columns=rename_map)
    if "datetime" in wdf.columns:
        wdf = wdf.set_index("datetime")
    wdf = wdf.resample("15min").ffill()
    wdf.index.name = "datetime"
    return wdf.reset_index()


def fetch_weather_archive(area: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical hourly weather from Open-Meteo archive → 15-min df."""
    coords = AREA_COORDS[area]
    params = {
        "latitude":        coords["lat"],
        "longitude":       coords["lon"],
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          ARCHIVE_WEATHER_VARS,
        "timezone":        "Europe/Berlin",
        "wind_speed_unit": "kmh",
    }
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params, timeout=60,
    )
    r.raise_for_status()
    raw = r.json()["hourly"]
    wdf = pd.DataFrame(raw)
    wdf["datetime"] = pd.to_datetime(wdf.pop("time"))
    return _to_15min(wdf, ARCHIVE_RENAME)


def fetch_weather_forecast(area: str, target_date: date) -> pd.DataFrame:
    """Fetch hourly weather forecast from Open-Meteo for `target_date` → 15-min df."""
    coords = AREA_COORDS[area]
    params = {
        "latitude":        coords["lat"],
        "longitude":       coords["lon"],
        "start_date":      str(target_date),
        "end_date":        str(target_date),
        "hourly":          FORECAST_WEATHER_VARS,
        "timezone":        "Europe/Berlin",
        "wind_speed_unit": "kmh",
    }
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params, timeout=30,
    )
    r.raise_for_status()
    raw = r.json()["hourly"]
    wdf = pd.DataFrame(raw)
    wdf["datetime"] = pd.to_datetime(wdf.pop("time"))
    return _to_15min(wdf, FORECAST_RENAME)


def _zero_weather(start: datetime) -> pd.DataFrame:
    """Fallback: 96-row weather df of zeros when API is unreachable."""
    idx = pd.date_range(start=start, periods=96, freq="15min")
    df = pd.DataFrame({"datetime": idx})
    for c in WEATHER_FEAT_COLS:
        df[c] = 0.0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    PI2 = 2 * np.pi
    df = df.copy()
    df["period_of_day"]   = dt.dt.hour * 4 + dt.dt.minute // 15
    df["hour_of_day"]     = dt.dt.hour
    df["quarter_of_hour"] = dt.dt.minute // 15
    df["day_of_week"]     = dt.dt.dayofweek
    df["month"]           = dt.dt.month
    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
    df["period_sin"]      = np.sin(PI2 * df["period_of_day"] / 96)
    df["period_cos"]      = np.cos(PI2 * df["period_of_day"] / 96)
    df["hour_sin"]        = np.sin(PI2 * df["hour_of_day"]   / 24)
    df["hour_cos"]        = np.cos(PI2 * df["hour_of_day"]   / 24)
    df["dow_sin"]         = np.sin(PI2 * df["day_of_week"]   / 7)
    df["dow_cos"]         = np.cos(PI2 * df["day_of_week"]   / 7)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add price lag features. Requires df sorted by datetime."""
    df = df.copy()
    for col, shift in LAG_MAP.items():
        df[col] = df[TARGET].shift(shift)
    df["rolling_mean_96"] = (
        df[TARGET].shift(1).rolling(ROLL_WIN, min_periods=1).mean()
    )
    return df


def _ensure_weather_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in WEATHER_FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Build inference feature matrix for tomorrow
# ──────────────────────────────────────────────────────────────────────────────

def build_inference_features(
    area_history: pd.DataFrame,     # full historical price df for this area
    target_date: date,
    forecast_weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the 96-row feature matrix for inference.

    Lag values are looked up from historical prices where available.
    Any missing lags are imputed using the period-of-day median from history.
    """
    start = datetime(target_date.year, target_date.month, target_date.day)
    slots = pd.date_range(start=start, periods=96, freq="15min")
    feat = pd.DataFrame({"datetime": slots})
    feat = add_calendar(feat)

    # ── Merge weather forecast ────────────────────────────────────────────────
    w_cols = [c for c in forecast_weather.columns if c != "datetime"]
    feat = feat.merge(forecast_weather[["datetime"] + w_cols], on="datetime", how="left")
    for c in WEATHER_FEAT_COLS:
        if c not in feat.columns:
            feat[c] = 0.0
        feat[c] = feat[c].ffill().bfill().fillna(0.0)

    # ── Lag features from history ─────────────────────────────────────────────
    hist_idx = area_history.set_index("datetime")[TARGET]

    for col, periods in LAG_MAP.items():
        delta = timedelta(minutes=periods * 15)
        feat[col] = feat["datetime"].apply(
            lambda ts, d=delta: hist_idx.get(ts - d, np.nan)
        )

    # ── Impute missing lags with period-of-day median ─────────────────────────
    # (fills gaps when today's/yesterday's data is not yet in the CSV)
    hist_with_cal = add_calendar(area_history.copy())
    period_median = (
        hist_with_cal.groupby("period_of_day")[TARGET].median().to_dict()
    )

    for col in LAG_MAP:
        mask = feat[col].isna()
        if mask.any():
            feat.loc[mask, col] = (
                feat.loc[mask, "period_of_day"].map(period_median)
            )
            feat[col] = feat[col].ffill().bfill().fillna(0.0)
            log.debug(f"Imputed {mask.sum()} NaN in {col} for {target_date}")

    # ── Rolling mean: period-of-day median from history ───────────────────────
    feat["rolling_mean_96"] = feat["period_of_day"].map(period_median).fillna(0.0)

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_area_model(
    area_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> xgb.XGBRegressor:
    """Merge weather, engineer features, train XGBoost, return fitted model."""
    # Merge historical weather
    w_cols = [c for c in weather_df.columns if c != "datetime"]
    df = area_df.merge(weather_df[["datetime"] + w_cols], on="datetime", how="left")
    df = _ensure_weather_cols(df)
    for c in WEATHER_FEAT_COLS:
        df[c] = df[c].ffill().bfill().fillna(0.0)

    df = add_calendar(df)
    df = add_lags(df)

    # Drop rows where target or lags are NaN (first ~672 rows of the series)
    needed = [TARGET] + list(LAG_MAP.keys())
    df = df.dropna(subset=needed)

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df[TARGET]

    log.info(f"  Training on {len(X)} rows ...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_predictions(delivery_date: date | None = None) -> dict:
    """
    Train one XGBoost model per area and predict 96 quarter-hourly prices
    for `delivery_date` (defaults to tomorrow in CET/Berlin time).

    Writes results to data/predictions_tomorrow.json and returns the dict.
    """
    try:
        from zoneinfo import ZoneInfo
        now_cet = datetime.now(tz=ZoneInfo("Europe/Berlin"))
    except ImportError:
        import pytz
        now_cet = datetime.now(tz=pytz.timezone("Europe/Berlin"))

    if delivery_date is None:
        delivery_date = (now_cet + timedelta(days=1)).date()

    log.info(f"=== Day-ahead prediction run ===")
    log.info(f"Delivery date : {delivery_date}")
    log.info(f"Generated at  : {now_cet.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    prices = load_prices()
    log.info(f"Loaded {len(prices):,} historical rows across {prices['area'].nunique()} areas")

    result = {
        "generated_at": now_cet.isoformat(),
        "delivery_day":  str(delivery_date),
        "predictions":   {},
        "area_stats":    {},
    }

    for area in AREAS:
        log.info(f"\n[{area}] ────────────────────────────────────────────────")
        area_df = (
            prices[prices["area"] == area]
            .copy()
            .reset_index(drop=True)
        )

        if len(area_df) < 300:
            log.warning(f"[{area}] Only {len(area_df)} rows — skipping (need ≥300)")
            continue

        start_str = str(area_df["datetime"].min().date())
        end_str   = str(area_df["datetime"].max().date())

        # ── Historical weather for training ──────────────────────────────────
        try:
            hist_weather = fetch_weather_archive(area, start_str, end_str)
            log.info(f"[{area}] Archive weather: {len(hist_weather)} rows")
        except Exception as exc:
            log.warning(f"[{area}] Archive weather failed ({exc}) — using zeros")
            hist_weather = _zero_weather(area_df["datetime"].min())

        # ── Train model ───────────────────────────────────────────────────────
        model = train_area_model(area_df, hist_weather)

        # ── Forecast weather for tomorrow ─────────────────────────────────────
        try:
            fcst_weather = fetch_weather_forecast(area, delivery_date)
            log.info(f"[{area}] Forecast weather: {len(fcst_weather)} rows")
        except Exception as exc:
            log.warning(f"[{area}] Forecast weather failed ({exc}) — using zeros")
            fcst_weather = _zero_weather(
                datetime(delivery_date.year, delivery_date.month, delivery_date.day)
            )

        # ── Build inference features ──────────────────────────────────────────
        feat = build_inference_features(area_df, delivery_date, fcst_weather)
        feat = _ensure_weather_cols(feat)
        X_pred = feat[FEATURE_COLS].fillna(feat[FEATURE_COLS].median())

        # ── Predict ───────────────────────────────────────────────────────────
        preds = np.maximum(model.predict(X_pred), -500.0)   # allow negatives ≥ -500

        slots = [
            {
                "datetime":      feat.iloc[i]["datetime"].strftime("%Y-%m-%dT%H:%M:%S"),
                "price_eur_mwh": round(float(preds[i]), 2),
            }
            for i in range(len(preds))
        ]
        result["predictions"][area] = slots
        result["area_stats"][area] = {
            "mean":  round(float(preds.mean()), 2),
            "min":   round(float(preds.min()),  2),
            "max":   round(float(preds.max()),  2),
            "p25":   round(float(np.percentile(preds, 25)), 2),
            "p75":   round(float(np.percentile(preds, 75)), 2),
            "peak_period": int(np.argmax(preds)),   # 0-95
            "trough_period": int(np.argmin(preds)),
        }
        log.info(
            f"[{area}] Predicted 96 slots | "
            f"min={preds.min():.1f}  mean={preds.mean():.1f}  max={preds.max():.1f} EUR/MWh"
        )

    # ── Persist ───────────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info(f"\nSaved → {OUTPUT_JSON}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_predictions()
