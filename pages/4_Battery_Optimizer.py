import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta
import io

st.set_page_config(layout="wide", page_title="Battery Optimizer", page_icon="🔋")

# Auto-run on first page visit (Austria with default params)
if "bo_first_load" not in st.session_state:
    st.session_state["bo_first_load"] = True

# Custom CSS for improved UI
st.markdown(
    """
    <style>
    /* Watt Happens Header Styling */
    #watt-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000000;
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 50%, #1976D2 100%);
        border-bottom: 2px solid #e2e8f0;
        padding: 0 20px;
        height: 60px;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.3);
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 70px !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #1976D2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .metric-positive {
        border-left-color: #2E7D32;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    .metric-negative {
        border-left-color: #E65100;
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
    }
    
    /* Section cards */
    .section-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
    }
    
    /* Info banner */
    .info-banner {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #1976D2;
        margin-bottom: 20px;
    }
    .warning-banner {
        background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #F57C00;
        margin-bottom: 20px;
    }
    
    /* Strategy highlight */
    .strategy-highlight {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #10b981;
        margin: 16px 0;
    }
    
    /* Train/Test split visual */
    .split-container {
        display: flex;
        height: 32px;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 12px 0;
    }
    .split-train {
        background: linear-gradient(90deg, #1976D2, #42A5F5);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 12px;
    }
    .split-test {
        background: linear-gradient(90deg, #F57C00, #FFA726);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 12px;
    }
    
    /* LinkedIn button */
    #watt-header a, #watt-header a:visited {
        color: #fff !important;
        background: rgba(255,255,255,0.2) !important;
        text-decoration: none !important;
        border-radius: 20px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        white-space: nowrap !important;
        letter-spacing: .3px !important;
        transition: all 0.3s ease !important;
    }
    #watt-header a:hover {
        background: rgba(255,255,255,0.3) !important;
        transform: translateY(-1px);
    }
    
    /* Battery indicator */
    .battery-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f1f5f9;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    
    /* Validation messages */
    .validation-error {
        background: #fee2e2;
        color: #dc2626;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 13px;
        margin-top: 8px;
    }
    .validation-success {
        background: #dcfce7;
        color: #16a34a;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 13px;
        margin-top: 8px;
    }
    </style>
    <div id="watt-header">
        <div style="flex:1"></div>
        <div style="display:flex;align-items:center;gap:12px">
            <div style="background:rgba(255,255,255,0.2);border-radius:10px;width:42px;height:42px;display:flex;align-items:center;justify-content:center;flex-shrink:0;border:1px solid rgba(255,255,255,0.3)">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="22" height="22"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
            </div>
            <div>
                <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:1.3rem;font-weight:700;color:#fff;letter-spacing:-0.2px;line-height:1.2">Watt Happens</div>
                <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:0.75rem;color:rgba(255,255,255,0.85);margin-top:1px">BESS Strategy & Energy Intelligence Platform</div>
            </div>
        </div>
        <div style="flex:1;display:flex;justify-content:flex-end;align-items:center">
            <a href="https://www.linkedin.com/in/bibek-agarwal" target="_blank" style="display:inline-flex;align-items:center;gap:8px;color:#fff!important;padding:8px 20px;border-radius:20px;font-weight:600;font-size:14px;text-decoration:none!important;white-space:nowrap">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="16" height="16"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.79M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
                Contact to know more
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Color palette
PALETTE = {
    "primary": ["#0D47A1", "#1565C0", "#1976D2", "#1E88E5", "#42A5F5"],
    "accent": ["#E65100", "#EF6C00", "#F57C00", "#FB8C00", "#FFA726"],
    "green": ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A"],
    "red": ["#B71C1C", "#C62828", "#D32F2F", "#E53935", "#EF5350"],
}

CHART_CONFIG = {
    "height": 380,
    "margin": dict(l=40, r=20, t=40, b=30),
}

DATASETS = {
    "DayAhead": ("dayahead_prices.csv", "price"),
    "IDA1": ("ida1_prices.csv", "price"),
    "IDA2": ("ida2_prices.csv", "price"),
    "IDA3": ("ida3_prices.csv", "price"),
    "VWAP": ("intraday_continuous_vwap_qh.csv", "vwap"),
}

AREAS = ["AT", "BE", "FR", "GER", "NL"]

@st.cache_data
def load_data(filename):
    """Load data from CSV file."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    df["deliveryStartCET"] = (
        pd.to_datetime(df["deliveryStartCET"], utc=True).dt.tz_convert("Europe/Paris")
    )
    df["hour"] = df["deliveryStartCET"].dt.hour
    df["day_of_week"] = df["deliveryStartCET"].dt.dayofweek
    df["date"] = df["deliveryStartCET"].dt.date
    return df

def filter_by_date_range(df, date_range, custom_start=None, custom_end=None):
    """Filter dataframe by date range."""
    if date_range == "All dates":
        return df
    if date_range == "Custom range":
        if custom_start is None or custom_end is None:
            return df
        start_ts = pd.Timestamp(custom_start, tz="Europe/Paris")
        end_ts = pd.Timestamp(custom_end, tz="Europe/Paris") + pd.Timedelta(days=1)
        return df[(df["deliveryStartCET"] >= start_ts) & (df["deliveryStartCET"] < end_ts)]
    max_date = df["deliveryStartCET"].max()
    if date_range == "Last 1 week":
        min_date = max_date - timedelta(days=7)
    elif date_range == "Last 2 weeks":
        min_date = max_date - timedelta(days=14)
    elif date_range == "Last 1 month":
        min_date = max_date - timedelta(days=30)
    elif date_range == "Last 2 months":
        min_date = max_date - timedelta(days=60)
    else:
        return df
    return df[df["deliveryStartCET"] >= min_date]

def filter_by_day_type(df, day_filter):
    """Filter by weekdays/weekends."""
    if day_filter == "All":
        return df
    elif day_filter == "Weekdays":
        return df[df["day_of_week"] < 5]
    elif day_filter == "Weekends":
        return df[df["day_of_week"] >= 5]
    return df

def _fmt_hour(h):
    h = int(h) % 48
    if h >= 24:
        return f"{h - 24:02d}:00 +1d"
    return f"{h:02d}:00"

def _fmt_window(start, length):
    end = int(start) + int(length)
    return f"{_fmt_hour(start)} → {_fmt_hour(end)}"

def _stats_from_daily(daily_pnl):
    """Compute summary stats from a daily P&L vector."""
    n = len(daily_pnl)
    total = float(daily_pnl.sum())
    avg = float(daily_pnl.mean())
    std = float(daily_pnl.std())
    win_rate = float((daily_pnl > 0).sum() / n)
    sharpe = (avg / std * np.sqrt(252)) if std > 0 else 0.0
    cumsum = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = float(abs((cumsum - running_max).min()))
    return {
        "total_pnl": total,
        "avg_daily_pnl": avg,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "num_days": n,
        "daily_pnl": daily_pnl,
        # Additional metrics
        "total_mwh_discharged": None,  # Will be calculated
        "avg_spread": None,  # Will be calculated
    }

def _format_schedule(cycles_list, charge_hours, discharge_hours):
    ch, dis = charge_hours, discharge_hours
    if len(cycles_list) == 1:
        b, s = cycles_list[0]
        return f"Buy {_fmt_window(b, ch)} | Sell {_fmt_window(s, dis)}"
    parts = []
    for i, (b, s) in enumerate(cycles_list, start=1):
        parts.append(f"C{i}: Buy {_fmt_window(b, ch)} / Sell {_fmt_window(s, dis)}")
    return " | ".join(parts)

@st.cache_data
def _build_price_matrix(df):
    """Build a (dates, D×24 price matrix) from a dataframe."""
    if len(df) == 0:
        return [], np.zeros((0, 24))
    pivot = df.groupby(["date", "hour"])["price"].mean().unstack()
    dates = list(pivot.index)
    mat = np.full((len(dates), 24), np.nan)
    for h in pivot.columns:
        if 0 <= int(h) < 24:
            mat[:, int(h)] = pivot[h].values
    return dates, mat

def apply_fixed_strategy(
    cycle_list, buy_dates, buy_mat, sell_dates, sell_mat,
    ch, dis, E_in, E_out, fixed_cost_per_cycle,
):
    """Apply a fixed schedule (known buy/sell windows) to price matrices.

    Returns (daily_pnl array, list_of_valid_dates, total_mwh_charged, total_mwh_discharged).
    Works for both same-day and overnight schedules.
    """
    buy_idx = {d: i for i, d in enumerate(buy_dates)}
    sell_idx = {d: i for i, d in enumerate(sell_dates)}
    common = sorted(set(buy_idx.keys()) & set(sell_idx.keys()))
    if not common:
        return np.array([]), [], 0, 0

    D = len(common)
    b_order = [buy_idx[d] for d in common]
    s_order = [sell_idx[d] for d in common]
    buy24 = buy_mat[b_order]
    sell24 = sell_mat[s_order]

    overnight = any(
        b >= 24 or s >= 24 or b + ch > 24 or s + dis > 24
        for b, s in cycle_list
    )

    if overnight and D >= 2:
        eff_D = D - 1
        buy_ext = np.full((eff_D, 48), np.nan)
        sell_ext = np.full((eff_D, 48), np.nan)
        buy_ext[:, :24] = buy24[:-1]
        buy_ext[:, 24:] = buy24[1:]
        sell_ext[:, :24] = sell24[:-1]
        sell_ext[:, 24:] = sell24[1:]
        mat_b, mat_s = buy_ext, sell_ext
        eff_dates = common[:-1]
    elif overnight and D < 2:
        return np.array([]), [], 0, 0
    else:
        mat_b, mat_s = buy24, sell24
        eff_D, eff_dates = D, common

    daily_pnl = np.zeros(eff_D)
    valid = np.ones(eff_D, dtype=bool)
    total_mwh_charged = 0
    total_mwh_discharged = 0

    for b, s in cycle_list:
        buy_win = mat_b[:, b: b + ch]
        sell_win = mat_s[:, s: s + dis]
        valid &= (~np.isnan(buy_win).any(axis=1)) & (~np.isnan(sell_win).any(axis=1))
        ab = np.nanmean(buy_win, axis=1)
        sv = np.nanmean(sell_win, axis=1)
        daily_pnl += sv * E_out - ab * E_in - fixed_cost_per_cycle
        total_mwh_charged += E_in * len([v for v in valid if v])
        total_mwh_discharged += E_out * len([v for v in valid if v])

    return daily_pnl[valid], [d for d, v in zip(eff_dates, valid) if v], total_mwh_charged, total_mwh_discharged

def _enumerate_schedules(
    buy_ext, sell_ext, max_hour, ch, dis, max_cycles,
    E_in, E_out, fixed_cost_per_cycle, top_per_k,
    midnight_mode, overnight_allow_k1_any_order,
):
    eff_days = buy_ext.shape[0]
    if eff_days == 0:
        return []

    buy_win_avg = {}
    for b in range(max_hour - ch + 1):
        sub = buy_ext[:, b:b + ch]
        buy_win_avg[b] = (~np.isnan(sub).any(axis=1), np.nanmean(sub, axis=1))
    sell_win_avg = {}
    for s in range(max_hour - dis + 1):
        sub = sell_ext[:, s:s + dis]
        sell_win_avg[s] = (~np.isnan(sub).any(axis=1), np.nanmean(sub, axis=1))

    def cycle_daily_pnl(ab, sv):
        return sv * E_out - ab * E_in - fixed_cost_per_cycle

    def crosses_midnight(cycles_list):
        for b, s in cycles_list:
            if b < 24 < b + ch:
                return True
            if s < 24 < s + dis:
                return True
            if (b + ch <= 24 and s >= 24) or (s + dis <= 24 and b >= 24):
                return True
        return False

    def all_on_day2(cycles_list):
        return all(b >= 24 and s >= 24 for b, s in cycles_list)

    def midnight_ok(cl):
        if midnight_mode == "none":
            return True
        return crosses_midnight(cl) and not all_on_day2(cl)

    out = []

    if max_cycles >= 1:
        k1 = []
        for b in range(max_hour - ch + 1):
            vb, ab = buy_win_avg[b]
            b_end = b + ch
            for s in range(max_hour - dis + 1):
                s_end = s + dis
                if not (s >= b_end or s_end <= b):
                    continue
                cl = [(b, s)]
                if not midnight_ok(cl):
                    continue
                vs, sv = sell_win_avg[s]
                valid = vb & vs
                if not valid.any():
                    continue
                daily_pnl = cycle_daily_pnl(ab[valid], sv[valid])
                # Calculate average buy and sell prices for spread
                avg_buy = np.nanmean(ab[valid])
                avg_sell = np.nanmean(sv[valid])
                stats = _stats_from_daily(daily_pnl)
                stats["avg_buy_price"] = avg_buy
                stats["avg_sell_price"] = avg_sell
                stats["avg_spread"] = avg_sell - avg_buy
                k1.append({
                    "cycles": 1,
                    "schedule": _format_schedule(cl, ch, dis),
                    "cycle_list": cl,
                    **stats,
                })
        k1.sort(key=lambda r: r["sharpe"], reverse=True)
        out.extend(k1[:top_per_k])

    def recurse(K_target, next_start, partial, partial_valid, partial_pnl, out_list):
        remaining = K_target - len(partial)
        if remaining == 0:
            if not partial_valid.any():
                return
            if not midnight_ok(partial):
                return
            daily = partial_pnl[partial_valid]
            stats = _stats_from_daily(daily)
            out_list.append({
                "cycles": K_target,
                "schedule": _format_schedule(partial, ch, dis),
                "cycle_list": list(partial),
                **stats,
            })
            return
        slack = (remaining - 1) * (ch + dis)
        for b in range(next_start, max_hour - ch - dis - slack + 1):
            vb, ab = buy_win_avg[b]
            nv1 = partial_valid & vb
            if not nv1.any():
                continue
            for s in range(b + ch, max_hour - dis - slack + 1):
                vs, sv = sell_win_avg[s]
                nv2 = nv1 & vs
                if not nv2.any():
                    continue
                cycle_pnl = cycle_daily_pnl(ab, sv)
                recurse(
                    K_target, s + dis,
                    partial + [(b, s)], nv2, partial_pnl + cycle_pnl, out_list,
                )

    for K in range(2, max_cycles + 1):
        if K * (ch + dis) > max_hour:
            continue
        kk = []
        init_valid = np.ones(eff_days, dtype=bool)
        init_pnl = np.zeros(eff_days)
        recurse(K, 0, [], init_valid, init_pnl, kk)
        kk.sort(key=lambda r: r["sharpe"], reverse=True)
        out.extend(kk[:top_per_k])

    return out

def optimize_battery(
    buy_dates, buy_mat, sellDates, sell_mat,
    capacity_mw, charge_hours, discharge_hours,
    max_cycles=1, allow_overnight=False,
    rte=1.0, degradation_eur_mwh=0.0, fee_eur_mwh=0.0, top_per_k=200,
):
    buy_idx = {d: i for i, d in enumerate(buy_dates)}
    sell_idx = {d: i for i, d in enumerate(sellDates)}
    common = sorted(set(buy_idx.keys()) & set(sell_idx.keys()))
    if len(common) == 0:
        return pd.DataFrame()
    b_order = [buy_idx[d] for d in common]
    s_order = [sell_idx[d] for d in common]
    buy24 = buy_mat[b_order]
    sell24 = sell_mat[s_order]
    D = buy24.shape[0]

    ch, dis = charge_hours, discharge_hours
    E_out = capacity_mw * dis
    E_in = E_out / rte if rte > 0 else E_out
    fixed_cost_per_cycle = degradation_eur_mwh * E_out + fee_eur_mwh * (E_in + E_out)

    all_results = []
    all_results.extend(
        _enumerate_schedules(
            buy24, sell24, 24, ch, dis, max_cycles,
            E_in, E_out, fixed_cost_per_cycle, top_per_k,
            midnight_mode="none", overnight_allow_k1_any_order=True,
        )
    )

    if allow_overnight and D >= 2:
        buy48 = np.full((D - 1, 48), np.nan)
        sell48 = np.full((D - 1, 48), np.nan)
        buy48[:, :24] = buy24[:-1]
        buy48[:, 24:] = buy24[1:]
        sell48[:, :24] = sell24[:-1]
        sell48[:, 24:] = sell24[1:]
        all_results.extend(
            _enumerate_schedules(
                buy48, sell48, 48, ch, dis, max_cycles,
                E_in, E_out, fixed_cost_per_cycle, top_per_k,
                midnight_mode="required", overnight_allow_k1_any_order=True,
            )
        )

    if not all_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    def _bs_from_cycle(row, idx, which):
        cl = row.get("cycle_list")
        if isinstance(cl, list) and len(cl) > idx:
            return cl[idx][0] if which == "b" else cl[idx][1]
        return np.nan

    results_df["buy_start"] = results_df.apply(lambda r: _bs_from_cycle(r, 0, "b"), axis=1)
    results_df["sell_start"] = results_df.apply(lambda r: _bs_from_cycle(r, 0, "s"), axis=1)
    return results_df.sort_values("sharpe", ascending=False).reset_index(drop=True)

# ── Chart helpers ─────────────────────────────────────────────────────────────

def create_sharpe_heatmap(results_df, title_suffix=""):
    if "cycles" in results_df.columns:
        one_cycle = results_df[results_df["cycles"] == 1]
    else:
        one_cycle = results_df
    one_cycle = one_cycle[(one_cycle["buy_start"] < 24) & (one_cycle["sell_start"] < 24)]

    heatmap_data = np.full((24, 24), np.nan)
    for _, row in one_cycle.iterrows():
        heatmap_data[int(row["buy_start"]), int(row["sell_start"])] = row["sharpe"]

    # Determine text color based on background
    zmin = np.nanmin(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 0
    zmax = np.nanmax(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 1
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"{h:02d}:00" for h in range(24)],
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale=[
            [0, "#B71C1C"],      # Dark red for negative
            [0.25, "#FF6B6B"],   # Light red
            [0.5, "#FFE66D"],    # Yellow for neutral
            [0.75, "#4ECDC4"],   # Teal
            [1, "#1B5E20"],      # Dark green for positive
        ],
        text=np.round(heatmap_data, 2),
        texttemplate="%{text:.2f}",
        textfont={"size": 9, "color": "white"},
        colorbar=dict(
            title="Sharpe",
            titleside="right",
            titlefont=dict(size=12),
        ),
        hoverongaps=False,
        hovertemplate="Buy: %{y}<br>Sell: %{x}<br>Sharpe: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"📊 Sharpe Heatmap: Buy vs Sell Hour {title_suffix}",
            font=dict(size=16),
        ),
        xaxis_title="Sell Start Hour",
        yaxis_title="Buy Start Hour",
        height=CHART_CONFIG["height"] + 40,
        margin=CHART_CONFIG["margin"],
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    return fig

def create_train_test_cumulative_chart(train_pnl, test_pnl, n_train_days, n_test_days):
    fig = go.Figure()

    train_cumsum = np.cumsum(train_pnl)
    fig.add_trace(go.Scatter(
        x=list(range(len(train_pnl))),
        y=train_cumsum,
        mode="lines",
        name=f"🟦 Train ({n_train_days}d — in-sample)",
        line=dict(width=3, color=PALETTE["primary"][2]),
        fill="tozeroy",
        fillcolor="rgba(25, 118, 210, 0.1)",
        hovertemplate="Day %{x}<br>Cumulative P&L: €%{y:,.0f}<extra>Train</extra>",
    ))

    if len(test_pnl) > 0:
        offset = train_cumsum[-1] if len(train_cumsum) > 0 else 0
        test_cumsum = np.cumsum(test_pnl) + offset
        test_x = list(range(len(train_pnl), len(train_pnl) + len(test_pnl)))
        fig.add_trace(go.Scatter(
            x=test_x, y=test_cumsum,
            mode="lines",
            name=f"🟧 Test ({n_test_days}d — out-of-sample)",
            line=dict(width=3, color=PALETTE["accent"][2]),
            fill="tozeroy",
            fillcolor="rgba(245, 124, 0, 0.1)",
            hovertemplate="Day %{x}<br>Cumulative P&L: €%{y:,.0f}<extra>Test</extra>",
        ))
        fig.add_vline(
            x=len(train_pnl) - 0.5,
            line=dict(color="#64748b", width=2, dash="dash"),
            annotation_text="⬅ Train | Test ➡",
            annotation_position="top",
            annotation_font_size=12,
            annotation_font_color="#64748b",
        )

    fig.update_layout(
        title=dict(
            text="📈 Cumulative P&L — Train vs Test (Best Strategy)",
            font=dict(size=16),
        ),
        yaxis_title="Cumulative P&L (EUR)",
        xaxis_title="Days",
        height=CHART_CONFIG["height"],
        margin=CHART_CONFIG["margin"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
        xaxis=dict(gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
    )
    return fig

def create_daily_pnl_chart(train_pnl, test_pnl=None):
    fig = go.Figure()

    # Color based on positive/negative with gradients
    train_colors = []
    for x in train_pnl:
        if x >= 0:
            train_colors.append(PALETTE["green"][2])
        else:
            train_colors.append(PALETTE["accent"][2])
    
    fig.add_trace(go.Bar(
        x=list(range(len(train_pnl))),
        y=train_pnl,
        marker_color=train_colors,
        name="🟦 Train",
        showlegend=(test_pnl is not None and len(test_pnl) > 0),
        hovertemplate="Day %{x}<br>P&L: €%{y:,.0f}<extra>Train</extra>",
    ))

    if test_pnl is not None and len(test_pnl) > 0:
        n_train = len(train_pnl)
        test_colors = []
        for x in test_pnl:
            if x >= 0:
                test_colors.append(PALETTE["green"][3])
            else:
                test_colors.append(PALETTE["accent"][3])
        
        fig.add_trace(go.Bar(
            x=list(range(n_train, n_train + len(test_pnl))),
            y=test_pnl,
            marker_color=test_colors,
            name="🟧 Test",
            opacity=0.85,
            hovertemplate="Day %{x}<br>P&L: €%{y:,.0f}<extra>Test</extra>",
        ))
        fig.add_vline(
            x=n_train - 0.5,
            line=dict(color="#64748b", width=2, dash="dash"),
            annotation_text="⬅ Train | Test ➡",
            annotation_position="top",
            annotation_font_size=12,
            annotation_font_color="#64748b",
        )

    # Add zero line
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dot"))

    fig.update_layout(
        title=dict(
            text="📊 Daily P&L — Best Strategy",
            font=dict(size=16),
        ),
        yaxis_title="P&L (EUR)",
        xaxis_title="Day",
        height=CHART_CONFIG["height"],
        margin=CHART_CONFIG["margin"],
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
        xaxis=dict(gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
    )
    return fig

def create_hourly_profile(df, price_column):
    hourly_stats = df.groupby("hour")[price_column].agg(['mean', 'min', 'max', 'std']).reset_index()
    
    fig = go.Figure()
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["mean"],
        mode='lines+markers',
        name='Average',
        line=dict(color=PALETTE["primary"][2], width=3),
        marker=dict(size=8),
        hovertemplate="Hour %{x}:00<br>Avg Price: €%{y:.2f}<extra></extra>",
    ))
    
    # Min/Max band
    fig.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["max"],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["min"],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(25, 118, 210, 0.1)',
        name='Min-Max Range',
        hovertemplate="Hour %{x}:00<br>Min: €%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="⏰ Average Price by Hour of Day (with Min-Max Range)",
            font=dict(size=16),
        ),
        xaxis_title="Hour of Day",
        yaxis_title="Price (EUR/MWh)",
        height=CHART_CONFIG["height"],
        margin=CHART_CONFIG["margin"],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.1)",
            gridwidth=1,
            tickmode='linear',
            tick0=0,
            dtick=2,
        ),
    )
    return fig

# ============================================================================
# SIDEBAR WITH IMPROVED ORGANIZATION
# ============================================================================

st.sidebar.markdown("## ⚙️ Battery Settings")

# Battery Configuration Card
with st.sidebar.expander("🔋 Battery Configuration", expanded=True):
    capacity_mw = st.number_input(
        "Capacity (MW)",
        min_value=0.1, max_value=100.0, value=1.0, step=0.1,
        help="Maximum power output of the battery system",
    )
    
    col_ch, col_dis = st.columns(2)
    with col_ch:
        charge_hours = st.number_input(
            "Charge (hrs)", min_value=1, max_value=8, value=4,
            help="Hours required to fully charge",
        )
    with col_dis:
        discharge_hours = st.number_input(
            "Discharge (hrs)", min_value=1, max_value=8, value=4,
            help="Hours required to fully discharge",
        )
    
    # Validation for charge/discharge
    max_cycles = st.slider(
        "Max cycles/day", min_value=1, max_value=6, value=1,
        help="Maximum complete charge/discharge cycles per day",
    )
    
    # Logic validation display
    total_cycle_time = max_cycles * (charge_hours + discharge_hours)
    if total_cycle_time > 24:
        st.markdown(
            f'<div class="validation-error">⚠️ {max_cycles} cycles × ({charge_hours}+{discharge_hours})h = {total_cycle_time}h > 24h</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="validation-success">✅ {max_cycles} cycles × ({charge_hours}+{discharge_hours})h = {total_cycle_time}h ≤ 24h</div>',
            unsafe_allow_html=True
        )
    
    allow_overnight = st.checkbox(
        "Allow overnight cycles",
        value=False,
        help="Enable cycles that cross midnight (e.g., buy 22:00→02:00, sell 06:00→10:00 next day)",
    )

# Advanced Settings Card
with st.sidebar.expander("⚡ Advanced: Costs & Efficiency", expanded=False):
    rte_pct = st.number_input(
        "Round-trip efficiency (%)",
        min_value=50.0, max_value=100.0, value=100.0, step=1.0,
        help="Energy delivered ÷ energy charged. 100% = no losses",
    )
    degradation_eur_mwh = st.number_input(
        "Degradation (€/MWh discharged)",
        min_value=0.0, max_value=100.0, value=0.0, step=0.5,
        help="Battery wear cost per MWh discharged. Typical: 2-10 €/MWh",
    )
    fee_eur_mwh = st.number_input(
        "Exchange fees (€/MWh traded)",
        min_value=0.0, max_value=20.0, value=0.0, step=0.05,
        help="Trading fees per MWh on both buy and sell legs",
    )
rte = rte_pct / 100.0

# Market Selection Card
with st.sidebar.expander("🏪 Market Selection", expanded=True):
    selected_markets = st.multiselect(
        "Markets to analyze",
        options=list(DATASETS.keys()),
        default=["DayAhead", "IDA1"],
        help="Select one or more Nord Pool markets for arbitrage",
    )

    area = st.selectbox(
        "Price Area",
        options=AREAS,
        index=0,
        help="Geographic pricing zone for electricity",
    )

# Date Filter Card
with st.sidebar.expander("📅 Date Filter", expanded=False):
    day_filter = st.selectbox(
        "Day type",
        options=["All", "Weekdays", "Weekends"],
        index=0,
        help="Filter by weekday/weekend patterns",
    )
    
    date_range = st.selectbox(
        "Date range",
        options=["All dates", "Last 1 week", "Last 2 weeks", "Last 1 month", "Last 2 months", "Custom range"],
        index=0,
    )

    custom_start = None
    custom_end = None
    if date_range == "Custom range":
        _peek_market = selected_markets[0] if selected_markets else list(DATASETS.keys())[0]
        try:
            _peek_df = load_data(DATASETS[_peek_market][0])
            _min_avail = _peek_df["date"].min()
            _max_avail = _peek_df["date"].max()
        except Exception:
            _min_avail = None
            _max_avail = None

        default_end = _max_avail if _max_avail is not None else datetime.today().date()
        default_start = _min_avail if _min_avail is not None else (default_end - timedelta(days=14))
        default_range_start = max(default_start, default_end - timedelta(days=14))

        picked = st.date_input(
            "Select range",
            value=(default_range_start, default_end),
            min_value=default_start,
            max_value=default_end,
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            custom_start, custom_end = picked
        else:
            custom_start = custom_end = picked

# Train/Test Split with Visual Indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Train / Test Split")

train_pct = st.sidebar.slider(
    "Training data (%)",
    min_value=50, max_value=90, value=70, step=5,
    help="First X% of data for training (in-sample), rest for testing (out-of-sample)",
)

# Visual split indicator
split_html = f'''
<div class="split-container" style="width: 100%;">
    <div class="split-train" style="width: {train_pct}%;">
        Train {train_pct}%
    </div>
    <div class="split-test" style="width: {100 - train_pct}%;">
        Test {100 - train_pct}%
    </div>
</div>
<p style="font-size: 11px; color: #64748b; margin-top: 4px;">
    🟦 In-sample (training) | 🟧 Out-of-sample (validation)
</p>
'''
st.sidebar.markdown(split_html, unsafe_allow_html=True)

# Reset button
if st.sidebar.button("🔄 Reset to Defaults", type="secondary"):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")

# Validate inputs before enabling optimize
validation_errors = []
if not selected_markets:
    validation_errors.append("Select at least one market")
if total_cycle_time > 24:
    validation_errors.append(f"Cycle configuration exceeds 24h ({total_cycle_time}h)")

if validation_errors:
    st.sidebar.markdown(
        '<div class="validation-error">' + '<br>'.join([f"• {e}" for e in validation_errors]) + '</div>',
        unsafe_allow_html=True
    )
    optimize_clicked = st.sidebar.button("🚀 Optimize", type="primary", disabled=True)
else:
    optimize_clicked = st.sidebar.button("🚀 Optimize", type="primary")

# Auto-run on first page visit
if st.session_state.get("bo_first_load", False):
    if not validation_errors:
        optimize_clicked = True
    st.session_state["bo_first_load"] = False

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Title with subtitle
st.markdown("""
<h1 style="margin-bottom: 0;">🔋 Battery Optimizer</h1>
<p style="color: #64748b; font-size: 1.1rem; margin-top: 8px;">
    Discover optimal buy/sell schedules across Nord Pool electricity markets using historical price arbitrage
</p>
""", unsafe_allow_html=True)

if optimize_clicked:
    if not selected_markets:
        st.error("Please select at least one market in the sidebar.")
        st.stop()

    # Progress indicator with steps
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
    with st.spinner(""):
        # Step 1: Loading data
        status_text.text("📥 Loading market data...")
        progress_bar.progress(10)
        
        # ── Load & filter ──────────────────────────────────────────────────
        market_dfs = {}
        for mkt in selected_markets:
            fname, pcol = DATASETS[mkt]
            df = load_data(fname)
            df = df[df["area"] == area].copy()
            df = filter_by_date_range(df, date_range, custom_start, custom_end)
            df = filter_by_day_type(df, day_filter)
            if pcol != "price":
                df = df.rename(columns={pcol: "price"})
            market_dfs[mkt] = df

        # Step 2: Processing
        status_text.text("⚙️ Processing price matrices...")
        progress_bar.progress(30)
        
        # ── Chronological train/test split on common dates ─────────────────
        common_dates_all = sorted(
            set.intersection(*[set(market_dfs[m]["date"].unique()) for m in selected_markets])
        )
        n_total = len(common_dates_all)
        if n_total < 10:
            st.error("Not enough data for a train/test split. Try a wider date range.")
            st.stop()

        n_train = max(5, int(round(n_total * train_pct / 100)))
        train_date_set = set(common_dates_all[:n_train])
        test_date_set = set(common_dates_all[n_train:])
        has_test = len(test_date_set) > 0

        train_start_d = min(train_date_set)
        train_end_d = max(train_date_set)
        test_start_d = min(test_date_set) if has_test else None
        test_end_d = max(test_date_set) if has_test else None

        # ── Build train / test price matrices per market ───────────────────
        train_mats = {}
        test_mats = {}
        for mkt in selected_markets:
            df = market_dfs[mkt]
            train_mats[mkt] = _build_price_matrix(df[df["date"].isin(train_date_set)])
            if has_test:
                test_mats[mkt] = _build_price_matrix(df[df["date"].isin(test_date_set)])

        # ── Cost constants ─────────────────────────────────────────────────
        E_out = capacity_mw * discharge_hours
        E_in = E_out / rte if rte > 0 else E_out
        fixed_cost_per_cycle = degradation_eur_mwh * E_out + fee_eur_mwh * (E_in + E_out)

        # Step 3: Optimizing
        status_text.text("🧬 Running optimization algorithms...")
        progress_bar.progress(50)
        
        # ── Optimise on TRAIN data ─────────────────────────────────────────
        combined_train = []
        total_market_combinations = len(selected_markets) * len(selected_markets)
        current_combination = 0
        
        for buy_market in selected_markets:
            b_dates, b_mat = train_mats[buy_market]
            for sell_market in selected_markets:
                current_combination += 1
                status_text.text(f"🧬 Optimizing {buy_market} → {sell_market} ({current_combination}/{total_market_combinations})...")
                progress_bar.progress(50 + int(30 * current_combination / total_market_combinations))
                
                s_dates, s_mat = train_mats[sell_market]
                part = optimize_battery(
                    b_dates, b_mat, s_dates, s_mat,
                    capacity_mw, charge_hours, discharge_hours,
                    max_cycles=max_cycles, allow_overnight=allow_overnight,
                    rte=rte, degradation_eur_mwh=degradation_eur_mwh, fee_eur_mwh=fee_eur_mwh,
                )
                if len(part) == 0:
                    continue
                part = part.copy()
                part["buy_market"] = buy_market
                part["sell_market"] = sell_market
                part["market_pair"] = (
                    buy_market if buy_market == sell_market
                    else f"{buy_market} → {sell_market}"
                )
                combined_train.append(part)

        if not combined_train:
            st.error("No valid optimisation results. Try adjusting parameters.")
            st.stop()

        results = (
            pd.concat(combined_train, ignore_index=True)
            .sort_values("sharpe", ascending=False)
            .reset_index(drop=True)
        )

        best = results.iloc[0]
        train_stats = _stats_from_daily(best["daily_pnl"])
        
        # Calculate additional metrics for train
        train_stats["capacity_factor"] = (max_cycles * discharge_hours / 24) * 100
        train_stats["total_mwh_discharged"] = max_cycles * E_out * train_stats["num_days"]
        train_stats["total_mwh_charged"] = max_cycles * E_in * train_stats["num_days"]

        # Step 4: Testing
        status_text.text("🧪 Validating on test data...")
        progress_bar.progress(90)
        
        # ── Apply best strategy to TEST data ─────────────────────────────
        test_stats = None
        test_daily_pnl = np.array([])
        if has_test:
            bt_dates, bt_mat = test_mats[best["buy_market"]]
            st_dates, st_mat = test_mats[best["sell_market"]]
            test_daily_pnl, _, test_mwh_charged, test_mwh_discharged = apply_fixed_strategy(
                best["cycle_list"],
                bt_dates, bt_mat, st_dates, st_mat,
                charge_hours, discharge_hours, E_in, E_out, fixed_cost_per_cycle,
            )
            if len(test_daily_pnl) > 0:
                test_stats = _stats_from_daily(test_daily_pnl)
                test_stats["capacity_factor"] = (max_cycles * discharge_hours / 24) * 100
                test_stats["total_mwh_discharged"] = test_mwh_discharged
                test_stats["total_mwh_charged"] = test_mwh_charged

        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Optimization complete!")
        
    # Clear progress indicators
    progress_container.empty()

    # ================================================================
    # PERIOD BANNERS
    # ================================================================
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown(
            f'<div class="info-banner">'
            f'<strong>🟦 Train (in-sample)</strong><br>'
            f'{train_start_d} → {train_end_d} ({n_train} days, {train_pct}%)'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_b2:
        if has_test and test_stats:
            n_test_used = int(test_stats["num_days"])
            st.markdown(
                f'<div class="warning-banner">'
                f'<strong>🟧 Test (out-of-sample)</strong><br>'
                f'{test_start_d} → {test_end_d} ({n_test_used} days, {100 - train_pct}%)'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background: #f1f5f9; border-radius: 12px; padding: 16px 20px; border-left: 4px solid #94a3b8; margin-bottom: 20px;">'
                f'<strong>⚪ No test data</strong><br>'
                f'Not enough data for validation split'
                f'</div>',
                unsafe_allow_html=True
            )

    # ===================================================================
    # BEST STRATEGY CARD (Highlighted)
    # ================================================================
    schedule_text = f"<strong>{best['market_pair']}</strong>: {best['schedule']}"
    annual_pnl = best["avg_daily_pnl"] * 365
    
    # Calculate avg spread captured
    avg_spread = best.get("avg_spread", 0)
    
    st.markdown(
        f'<div class="strategy-highlight">'
        f'<div style="font-size: 1.1rem; margin-bottom: 8px;">🎯 {schedule_text}</div>'
        f'<div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.95rem; color: #374151;">'
        f'<span><strong>Area:</strong> {area}</span>'
        f'<span><strong>Capacity:</strong> {capacity_mw} MW</span>'
        f'<span><strong>Est. Annual P&L:</strong> <span style="color: #059669; font-weight: 600;">€{annual_pnl:,.0f}</span></span>'
        f'<span><strong>Avg Spread:</strong> €{avg_spread:.2f}/MWh</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ================================================================
    # ENHANCED TRAIN vs TEST KPI COMPARISON
    # ================================================================
    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")
    
    # Additional metrics row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            "🔋 Total MWh Discharged (Train)",
            f"{train_stats.get('total_mwh_discharged', 0):,.1f}",
            help="Total energy discharged during training period"
        )
    with col_m2:
        st.metric(
            "⚡ Capacity Factor",
            f"{train_stats.get('capacity_factor', 0):.1f}%",
            help="Percentage of time battery is actively discharging"
        )
    with col_m3:
        rte_display = rte * 100
        st.metric(
            "🔄 Round-trip Efficiency",
            f"{rte_display:.0f}%",
            help="Energy output / Energy input ratio"
        )
    with col_m4:
        cycles_per_day = len(best.get("cycle_list", []))
        st.metric(
            "🔄 Cycles/Day",
            f"{cycles_per_day}",
            help="Number of charge/discharge cycles in best strategy"
        )

    if test_stats:
        st.markdown("#### Train vs Test Comparison")
        col_train, col_test = st.columns(2)

        with col_train:
            st.markdown('<div style="background: #eff6ff; border-radius: 12px; padding: 16px; border: 1px solid #bfdbfe;">', unsafe_allow_html=True)
            st.markdown("**🟦 Train (in-sample)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total P&L", f"€{train_stats['total_pnl']:,.0f}")
            c2.metric("Sharpe", f"{train_stats['sharpe']:.2f}")
            c3.metric("Win Rate", f"{train_stats['win_rate']*100:.1f}%")
            c4, c5, c6 = st.columns(3)
            c4.metric("Avg Daily", f"€{train_stats['avg_daily_pnl']:,.0f}")
            c5.metric("Max DD", f"€{train_stats['max_dd']:,.0f}")
            c6.metric("Days", f"{int(train_stats['num_days'])}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_test:
            st.markdown('<div style="background: #fff7ed; border-radius: 12px; padding: 16px; border: 1px solid #fed7aa;">', unsafe_allow_html=True)
            st.markdown("**🟧 Test (out-of-sample)**")
            sharpe_delta = test_stats["sharpe"] - train_stats["sharpe"]
            wr_delta = (test_stats["win_rate"] - train_stats["win_rate"]) * 100
            avg_delta = test_stats["avg_daily_pnl"] - train_stats["avg_daily_pnl"]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total P&L", f"€{test_stats['total_pnl']:,.0f}")
            c2.metric("Sharpe", f"{test_stats['sharpe']:.2f}", delta=f"{sharpe_delta:+.2f}", delta_color="inverse")
            c3.metric("Win Rate", f"{test_stats['win_rate']*100:.1f}%", delta=f"{wr_delta:+.1f}pp")
            c4, c5, c6 = st.columns(3)
            c4.metric("Avg Daily", f"€{test_stats['avg_daily_pnl']:,.0f}", delta=f"€{avg_delta:+,.0f}")
            c5.metric("Max DD", f"€{test_stats['max_dd']:,.0f}")
            c6.metric("Days", f"{int(test_stats['num_days'])}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Train-only KPIs with better styling
        st.markdown("#### Training Results")
        cols = st.columns(6)
        metrics = [
            ("Total P&L", f"€{train_stats['total_pnl']:,.0f}"),
            ("Sharpe", f"{train_stats['sharpe']:.2f}"),
            ("Max DD", f"€{train_stats['max_dd']:,.0f}"),
            ("Win Rate", f"{train_stats['win_rate']*100:.1f}%"),
            ("Avg Daily", f"€{train_stats['avg_daily_pnl']:,.0f}"),
            ("Days", f"{int(train_stats['num_days'])}"),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, value)

    # ================================================================
    # HEATMAP + CUMULATIVE P&L
    # ================================================================
    st.markdown("---")
    st.markdown("### 📈 Strategy Analysis")

    col_heatmap, col_cumulative = st.columns(2)
    with col_heatmap:
        st.plotly_chart(
            create_sharpe_heatmap(results, title_suffix="(Train)"),
            use_container_width=True,
        )
    with col_cumulative:
        n_test_used = int(test_stats["num_days"]) if test_stats else 0
        st.plotly_chart(
            create_train_test_cumulative_chart(
                best["daily_pnl"], test_daily_pnl, n_train, n_test_used
            ),
            use_container_width=True,
        )

    # ================================================================
    # TOP 10 TABLE (train) - ENHANCED
    # ================================================================
    st.markdown("---")
    st.markdown("### 🏆 Top 10 Strategies (Train)")
    
    top_10 = results.head(10).copy()
    top_10["Rank"] = range(1, len(top_10) + 1)
    top_10["Market"] = top_10["market_pair"]
    top_10["Schedule"] = top_10["schedule"]
    top_10["Cycles"] = top_10["cycles"].astype(int)
    top_10["Total P&L"] = top_10["total_pnl"].apply(lambda x: f"€{x:,.0f}")
    top_10["Sharpe"] = top_10["sharpe"].apply(lambda x: f"{x:.2f}")
    top_10["Win Rate"] = top_10["win_rate"].apply(lambda x: f"{x*100:.0f}%")
    top_10["Avg Daily"] = top_10["avg_daily_pnl"].apply(lambda x: f"€{x:,.0f}")
    top_10["Max DD"] = top_10["max_dd"].apply(lambda x: f"€{x:,.0f}")

    display_cols = ["Rank", "Market", "Cycles", "Schedule", "Total P&L", "Sharpe", "Win Rate", "Avg Daily", "Max DD"]
    
    # Styled dataframe
    st.dataframe(
        top_10[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", help="Strategy ranking by Sharpe ratio"),
            "Market": st.column_config.TextColumn("Market", help="Buy → Sell market pair"),
            "Cycles": st.column_config.NumberColumn("Cycles", help="Number of daily cycles"),
            "Sharpe": st.column_config.TextColumn("Sharpe", help="Risk-adjusted return ratio"),
            "Win Rate": st.column_config.TextColumn("Win %", help="Percentage of profitable days"),
        }
    )

    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        csv_top = top_10[display_cols].to_csv(index=False)
        st.download_button(
            "📥 Top 10 (CSV)",
            csv_top,
            file_name="top_strategies.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ================================================================
    # DAILY P&L CHART
    # ================================================================
    st.markdown("---")
    st.markdown("### 📊 Daily P&L — Best Strategy")

    st.plotly_chart(
        create_daily_pnl_chart(best["daily_pnl"], test_daily_pnl if has_test else None),
        use_container_width=True,
    )
    
    # Summary stats for daily P&L
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        best_day = best["daily_pnl"].max()
        st.metric("Best Day", f"€{best_day:,.0f}")
    with col_s2:
        worst_day = best["daily_pnl"].min()
        st.metric("Worst Day", f"€{worst_day:,.0f}")
    with col_s3:
        volatility = best["daily_pnl"].std()
        st.metric("Daily Volatility", f"€{volatility:,.0f}")
    with col_s4:
        profit_days = (best["daily_pnl"] > 0).sum()
        st.metric("Profit Days", f"{profit_days}/{len(best['daily_pnl'])}")

    # ================================================================
    # HOURLY PRICE PROFILE
    # ================================================================
    st.markdown("---")
    st.markdown("### ⏰ Hourly Price Profile")

    profile_market = st.selectbox(
        "Select market to view", options=selected_markets, index=0, key="_profile_market"
    )
    st.markdown(f"**Market:** {profile_market} | **Area:** {area}")
    st.plotly_chart(
        create_hourly_profile(market_dfs[profile_market], "price"),
        use_container_width=True,
    )
    
    # Price statistics
    price_df = market_dfs[profile_market]
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric("Avg Price", f"€{price_df['price'].mean():.2f}/MWh")
    with col_p2:
        st.metric("Min Price", f"€{price_df['price'].min():.2f}/MWh")
    with col_p3:
        st.metric("Max Price", f"€{price_df['price'].max():.2f}/MWh")
    with col_p4:
        st.metric("Price Std", f"€{price_df['price'].std():.2f}/MWh")

    # ================================================================
    # DOWNLOAD ALL RESULTS
    # ================================================================
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        download_cols = ["market_pair", "cycles", "schedule", "total_pnl", "sharpe",
                        "win_rate", "avg_daily_pnl", "max_dd", "num_days"]
        csv_all = results[download_cols].to_csv(index=False)
        st.download_button(
            "📥 Download All Strategies (CSV)",
            csv_all,
            file_name="all_strategies.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        # Create summary report
        summary_data = {
            'Metric': ['Area', 'Capacity (MW)', 'Charge Hours', 'Discharge Hours', 
                      'Max Cycles', 'Round-trip Efficiency', 'Best Strategy',
                      'Train Total P&L', 'Train Sharpe', 'Train Win Rate',
                      'Test Total P&L' if test_stats else 'Test Total P&L', 
                      'Annualized P&L (est)'],
            'Value': [area, f"{capacity_mw}", f"{charge_hours}", f"{discharge_hours}",
                     f"{max_cycles}", f"{rte*100:.0f}%", best['schedule'],
                     f"€{train_stats['total_pnl']:,.0f}", f"{train_stats['sharpe']:.2f}", 
                     f"{train_stats['win_rate']*100:.1f}%",
                     f"€{test_stats['total_pnl']:,.0f}" if test_stats else "N/A",
                     f"€{annual_pnl:,.0f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            "📊 Download Summary Report (CSV)",
            csv_summary,
            file_name="optimization_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Initial state with instructions
    st.markdown("""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                border-radius: 16px; padding: 40px; text-align: center; 
                border: 2px dashed #3b82f6; margin-top: 20px;">
        <div style="font-size: 3rem; margin-bottom: 16px;">🔋</div>
        <h3 style="color: #1e40af; margin-bottom: 12px;">Ready to Optimize</h3>
        <p style="color: #3b82f6; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
            Configure your battery parameters in the sidebar, then click 
            <strong>🚀 Optimize</strong> to discover profitable trading schedules
        </p>
        <div style="margin-top: 24px; display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;">
            <div style="background: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <span style="font-size: 1.5rem;">⚡</span><br>
                <small>Set capacity & duration</small>
            </div>
            <div style="background: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <span style="font-size: 1.5rem;">🏪</span><br>
                <small>Choose markets</small>
            </div>
            <div style="background: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <span style="font-size: 1.5rem;">📅</span><br>
                <small>Set date range</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
