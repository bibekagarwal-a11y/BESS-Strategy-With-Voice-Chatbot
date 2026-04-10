import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta
import io

st.set_page_config(layout="wide", page_title="Battery Optimizer", page_icon="🔋")

st.markdown(
    """
    <style>
    #watt-header{position:fixed;top:0;left:0;right:0;z-index:1000000;background:#fff;border-bottom:2px solid #e2e8f0;padding:0 20px;height:52px;display:flex;align-items:center;box-shadow:0 2px 8px rgba(0,0,0,.06);}
    [data-testid="stMainBlockContainer"]{padding-top:60px!important;}
    [data-testid="stSidebarCollapseButton"]{display:none!important;}
    </style>
    <div id="watt-header">
      <div style="flex:1"></div>
      <div style="display:flex;align-items:center;gap:12px">
        <div style="background:linear-gradient(135deg,#1565C0,#0D47A1);border-radius:10px;width:38px;height:38px;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 2px 8px rgba(21,101,192,.3)">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="20" height="20"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        </div>
        <div>
          <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:1.2rem;font-weight:700;color:#0D1B3E;letter-spacing:-0.2px;line-height:1.2">Watt Happens</div>
          <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:0.72rem;color:#64748b;margin-top:1px">BESS Strategy &amp; Energy Intelligence Platform</div>
        </div>
      </div>
      <div style="flex:1;display:flex;justify-content:flex-end;align-items:center">
        <a href="https://www.linkedin.com/in/bibek-agarwal" target="_blank" style="display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,#0D47A1,#1976D2);color:#fff!important;padding:8px 20px;border-radius:20px;font-weight:600;font-size:14px;text-decoration:none!important;box-shadow:0 2px 8px rgba(0,0,0,.25);letter-spacing:.3px;white-space:nowrap"><svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="white"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>Contact to know more</a>
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

# Color palette
PALETTE = {
    "primary": ["#0D47A1", "#1565C0", "#1976D2", "#1E88E5", "#42A5F5"],
    "accent": ["#E65100", "#EF6C00", "#F57C00", "#FB8C00", "#FFA726"],
    "green": ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A"],
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
    }


def _format_schedule(cycles_list, charge_hours, discharge_hours):
    ch, dis = charge_hours, discharge_hours
    if len(cycles_list) == 1:
        b, s = cycles_list[0]
        return f"Buy {_fmt_window(b, ch)} | Sell {_fmt_window(s, dis)}"
    parts = []
    for i, (b, s) in enumerate(cycles_list, start=1):
        parts.append(f"C{i}: Buy {_fmt_window(b, ch)} / Sell {_fmt_window(s, dis)}")
    return "   |   ".join(parts)


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

    Returns (daily_pnl array, list_of_valid_dates).
    Works for both same-day and overnight schedules.
    """
    buy_idx = {d: i for i, d in enumerate(buy_dates)}
    sell_idx = {d: i for i, d in enumerate(sell_dates)}
    common = sorted(set(buy_idx.keys()) & set(sell_idx.keys()))
    if not common:
        return np.array([]), []

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
        return np.array([]), []
    else:
        mat_b, mat_s = buy24, sell24
        eff_D, eff_dates = D, common

    daily_pnl = np.zeros(eff_D)
    valid = np.ones(eff_D, dtype=bool)

    for b, s in cycle_list:
        buy_win = mat_b[:, b: b + ch]
        sell_win = mat_s[:, s: s + dis]
        valid &= (~np.isnan(buy_win).any(axis=1)) & (~np.isnan(sell_win).any(axis=1))
        ab = np.nanmean(buy_win, axis=1)
        sv = np.nanmean(sell_win, axis=1)
        daily_pnl += sv * E_out - ab * E_in - fixed_cost_per_cycle

    return daily_pnl[valid], [d for d, v in zip(eff_dates, valid) if v]


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
                stats = _stats_from_daily(daily_pnl)
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
    buy_dates, buy_mat, sell_dates, sell_mat,
    capacity_mw, charge_hours, discharge_hours,
    max_cycles=1, allow_overnight=False,
    rte=1.0, degradation_eur_mwh=0.0, fee_eur_mwh=0.0, top_per_k=200,
):
    buy_idx = {d: i for i, d in enumerate(buy_dates)}
    sell_idx = {d: i for i, d in enumerate(sell_dates)}
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

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=list(range(24)), y=list(range(24)),
        colorscale="RdYlGn",
        text=np.round(heatmap_data, 2),
        texttemplate="%{text:.2f}",
        textfont={"size": 8},
        colorbar=dict(title="Sharpe"),
    ))
    fig.update_layout(
        title=f"Sharpe Heatmap: Buy vs Sell Hour{title_suffix}",
        xaxis_title="Sell Start Hour", yaxis_title="Buy Start Hour",
        height=CHART_CONFIG["height"], margin=CHART_CONFIG["margin"],
        hovermode="closest",
    )
    return fig


def create_train_test_cumulative_chart(train_pnl, test_pnl, n_train_days, n_test_days):
    fig = go.Figure()

    train_cumsum = np.cumsum(train_pnl)
    fig.add_trace(go.Scatter(
        x=list(range(len(train_pnl))),
        y=train_cumsum,
        mode="lines",
        name=f"Train ({n_train_days}d — in-sample)",
        line=dict(width=2.5, color=PALETTE["primary"][2]),
        fill="tozeroy",
        fillcolor="rgba(25, 118, 210, 0.08)",
    ))

    if len(test_pnl) > 0:
        offset = train_cumsum[-1] if len(train_cumsum) > 0 else 0
        test_cumsum = np.cumsum(test_pnl) + offset
        test_x = list(range(len(train_pnl), len(train_pnl) + len(test_pnl)))
        fig.add_trace(go.Scatter(
            x=test_x, y=test_cumsum,
            mode="lines",
            name=f"Test ({n_test_days}d — out-of-sample)",
            line=dict(width=2.5, color=PALETTE["accent"][2]),
            fill="tozeroy",
            fillcolor="rgba(245, 124, 0, 0.08)",
        ))
        fig.add_vline(
            x=len(train_pnl) - 0.5,
            line=dict(color="grey", width=1.5, dash="dash"),
            annotation_text="Train | Test",
            annotation_position="top",
            annotation_font_size=11,
        )

    fig.update_layout(
        title="Cumulative P&L — Train vs Test (Best Strategy)",
        yaxis_title="Cumulative P&L (EUR)", xaxis_title="Days",
        height=CHART_CONFIG["height"], margin=CHART_CONFIG["margin"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_daily_pnl_chart(train_pnl, test_pnl=None):
    fig = go.Figure()

    train_colors = [PALETTE["green"][2] if x >= 0 else PALETTE["accent"][2] for x in train_pnl]
    fig.add_trace(go.Bar(
        x=list(range(len(train_pnl))),
        y=train_pnl,
        marker_color=train_colors,
        name="Train",
        showlegend=(test_pnl is not None and len(test_pnl) > 0),
    ))

    if test_pnl is not None and len(test_pnl) > 0:
        n_train = len(train_pnl)
        test_colors = [PALETTE["green"][3] if x >= 0 else PALETTE["accent"][3] for x in test_pnl]
        fig.add_trace(go.Bar(
            x=list(range(n_train, n_train + len(test_pnl))),
            y=test_pnl,
            marker_color=test_colors,
            name="Test",
            opacity=0.75,
        ))
        fig.add_vline(
            x=n_train - 0.5,
            line=dict(color="grey", width=1.5, dash="dash"),
            annotation_text="Train | Test",
            annotation_position="top",
            annotation_font_size=11,
        )

    fig.update_layout(
        title="Daily P&L — Best Strategy",
        yaxis_title="P&L (EUR)", xaxis_title="Day",
        height=CHART_CONFIG["height"], margin=CHART_CONFIG["margin"],
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_hourly_profile(df, price_column):
    hourly_avg = df.groupby("hour")[price_column].mean().reset_index()
    fig = go.Figure(data=go.Bar(
        x=hourly_avg["hour"], y=hourly_avg[price_column],
        marker_color=PALETTE["primary"][2],
    ))
    fig.update_layout(
        title="Average Price by Hour of Day",
        xaxis_title="Hour of Day", yaxis_title="Price (EUR/MWh)",
        height=CHART_CONFIG["height"], margin=CHART_CONFIG["margin"],
        showlegend=False,
    )
    return fig


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("⚙️ Battery Optimizer Settings")

capacity_mw = st.sidebar.number_input(
    "Battery Capacity (MW)", min_value=0.1, max_value=100.0, value=1.0, step=0.1
)

charge_hours = st.sidebar.slider("Charge Duration (hours)", min_value=1, max_value=8, value=4)
discharge_hours = st.sidebar.slider("Discharge Duration (hours)", min_value=1, max_value=8, value=4)

max_cycles = st.sidebar.slider(
    "Max cycles per day", min_value=1, max_value=6, value=1,
    help="Maximum number of complete charge/discharge cycles per day.",
)

allow_overnight = st.sidebar.checkbox(
    "Allow cross-day (overnight) cycles", value=False,
    help="Cycles may straddle midnight — e.g. buy 22:00–02:00, sell 06:00–10:00 next day.",
)

with st.sidebar.expander("⚙️ Advanced: costs & efficiency", expanded=False):
    rte_pct = st.number_input(
        "Round-trip efficiency (%)", min_value=50.0, max_value=100.0, value=100.0, step=1.0,
        help="Energy delivered ÷ energy charged. Default 100% = no loss.",
    )
    degradation_eur_mwh = st.number_input(
        "Degradation cost (€/MWh discharged)", min_value=0.0, max_value=100.0, value=0.0, step=0.5,
        help="Wear-and-tear cost per MWh discharged. Common values 2–10 €/MWh.",
    )
    fee_eur_mwh = st.number_input(
        "Exchange fees (€/MWh traded)", min_value=0.0, max_value=20.0, value=0.0, step=0.05,
        help="Per-MWh exchange fee paid on both buy and sell legs.",
    )
rte = rte_pct / 100.0

selected_markets = st.sidebar.multiselect(
    "Markets to use",
    options=list(DATASETS.keys()),
    default=["DayAhead", "IDA1"],
    help="Pick one or more Nord Pool markets. The optimizer tries every buy→sell "
         "combination and ranks strategies together.",
)

if not selected_markets:
    st.sidebar.warning("Select at least one market to run the optimizer.")

area = st.sidebar.selectbox("Area", options=AREAS, index=4)

day_filter = st.sidebar.selectbox("Day Filter", options=["All", "Weekdays", "Weekends"], index=0)

date_range = st.sidebar.selectbox(
    "Date Range",
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

    picked = st.sidebar.date_input(
        "Select date range",
        value=(default_range_start, default_end),
        min_value=default_start,
        max_value=default_end,
    )
    if isinstance(picked, tuple) and len(picked) == 2:
        custom_start, custom_end = picked
    else:
        custom_start = custom_end = picked

# ── Train / Test Split ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Train / Test Split")
train_pct = st.sidebar.slider(
    "Training data (%)",
    min_value=50, max_value=90, value=70, step=5,
    help="Chronological split: the first X% of dates find the optimal schedule "
         "(in-sample training). The remaining (100−X)% validate it out-of-sample.",
)
st.sidebar.caption(f"Test (out-of-sample): {100 - train_pct}%")

optimize_clicked = st.sidebar.button("🚀 Optimize", type="primary")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🔋 Battery Optimizer")
st.markdown("Discover optimal buy/sell schedules across Nord Pool electricity markets")

if optimize_clicked:
    if not selected_markets:
        st.error("Please select at least one market in the sidebar.")
        st.stop()

    with st.spinner("Loading data and running optimisation…"):

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

        # ── Optimise on TRAIN data ─────────────────────────────────────────
        combined_train = []
        for buy_market in selected_markets:
            b_dates, b_mat = train_mats[buy_market]
            for sell_market in selected_markets:
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

        # ── Apply best strategy to TEST data ─────────────────────────────
        test_stats = None
        test_daily_pnl = np.array([])
        if has_test:
            bt_dates, bt_mat = test_mats[best["buy_market"]]
            st_dates, st_mat = test_mats[best["sell_market"]]
            test_daily_pnl, _ = apply_fixed_strategy(
                best["cycle_list"],
                bt_dates, bt_mat, st_dates, st_mat,
                charge_hours, discharge_hours, E_in, E_out, fixed_cost_per_cycle,
            )
            if len(test_daily_pnl) > 0:
                test_stats = _stats_from_daily(test_daily_pnl)

        # ================================================================
        # PERIOD BANNER
        # ================================================================
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.info(
                f"**🟦 Train (in-sample):** {train_start_d} → {train_end_d}  "
                f"({n_train} days, {train_pct}%)"
            )
        with col_b2:
            if has_test and test_stats:
                n_test_used = int(test_stats["num_days"])
                st.warning(
                    f"**🟧 Test (out-of-sample):** {test_start_d} → {test_end_d}  "
                    f"({n_test_used} days, {100 - train_pct}%)"
                )
            else:
                st.warning("No test data available for the selected range / split.")

        # ===================================================================
        # BEST STRATEGY CARD
        # ================================================================
        schedule_text = f"**{best['market_pair']}:** {best['schedule']}"
        annual_pnl = best["avg_daily_pnl"] * 365
        st.info(
            f"{schedule_text}\n\n"
            f"**Area:** {area} | **Capacity:** {capacity_mw} MW | "
            f"**Est. Annual P&L (train avg):** €{annual_pnl:,.0f}"
        )

        # ================================================================
        # TRAIN vs TEST KPI COMPARISON
        # ================================================================
        st.divider()
        st.subheader("Train vs Test Performance")

        if test_stats:
            col_train, col_test = st.columns(2)

            with col_train:
                st.markdown("### 🟦 Train (in-sample)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total P&L", f"€{train_stats['total_pnl']:,.0f}")
                c2.metric("Sharpe", f"{train_stats['sharpe']:.2f}")
                c3.metric("Win Rate", f"{train_stats['win_rate']*100:.1f}%")
                c4, c5, c6 = st.columns(3)
                c4.metric("Avg Daily", f"€{train_stats['avg_daily_pnl']:,.0f}")
                c5.metric("Max Drawdown", f"€{train_stats['max_dd']:,.0f}")
                c6.metric("# Days", f"{int(train_stats['num_days'])}")

            with col_test:
                st.markdown("### 🟧 Test (out-of-sample)")
                sharpe_delta = test_stats["sharpe"] - train_stats["sharpe"]
                wr_delta = (test_stats["win_rate"] - train_stats["win_rate"]) * 100
                avg_delta = test_stats["avg_daily_pnl"] - train_stats["avg_daily_pnl"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Total P&L", f"€{test_stats['total_pnl']:,.0f}")
                c2.metric("Sharpe", f"{test_stats['sharpe']:.2f}", delta=f"{sharpe_delta:+.2f}")
                c3.metric("Win Rate", f"{test_stats['win_rate']*100:.1f}%", delta=f"{wr_delta:+.1f}pp")
                c4, c5, c6 = st.columns(3)
                c4.metric("Avg Daily", f"€{test_stats['avg_daily_pnl']:,.0f}", delta=f"€{avg_delta:+,.0f}")
                c5.metric("Max Drawdown", f"€{test_stats['max_dd']:,.0f}")
                c6.metric("# Days", f"{int(test_stats['num_days'])}")
        else:
            # Train-only KPIs
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Total P&L", f"€{train_stats['total_pnl']:,.0f}")
            c2.metric("Sharpe", f"{train_stats['sharpe']:.2f}")
            c3.metric("Max Drawdown", f"€{train_stats['max_dd']:,.0f}")
            c4.metric("Win Rate", f"{train_stats['win_rate']*100:.1f}%")
            c5.metric("Avg Daily", f"€{train_stats['avg_daily_pnl']:,.0f}")
            c6.metric("# Days", f"{int(train_stats['num_days'])}")

        # ================================================================
        # HEATMAP + CUMULATIVE P&L
        # ================================================================
        st.divider()
        st.subheader("Strategy Analysis")

        col_heatmap, col_cumulative = st.columns(2)
        with col_heatmap:
            st.plotly_chart(
                create_sharpe_heatmap(results, title_suffix=" (Train)"),
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
        # TOP 10 TABLE (train)
        # ================================================================
        st.divider()
        st.subheader("Top 10 Strategies (Train)")

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
        st.dataframe(top_10[display_cols], use_container_width=True, hide_index=True)

        csv_top = top_10[display_cols].to_csv(index=False)
        st.download_button("📥 Download Top 10 Strategies", csv_top,
                           file_name="top_strategies.csv", mime="text/csv")

        # ================================================================
        # DAILY P&L CHART
        # ================================================================
        st.divider()
        st.subheader("Daily P&L — Best Strategy")

        st.plotly_chart(
            create_daily_pnl_chart(best["daily_pnl"], test_daily_pnl if has_test else None),
            use_container_width=True,
        )

        # ================================================================
        # HOURLY PRICE PROFILE
        # ================================================================
        st.divider()
        st.subheader("Hourly Price Profile")

        profile_market = st.selectbox(
            "Market to show", options=selected_markets, index=0, key="_profile_market"
        )
        st.markdown(f"**Market:** {profile_market} | **Area:** {area}")
        st.plotly_chart(
            create_hourly_profile(market_dfs[profile_market], "price"),
            use_container_width=True,
        )

        # ================================================================
        # DOWNLOAD ALL RESULTS
        # ================================================================
        st.divider()
        download_cols = ["market_pair", "cycles", "schedule", "total_pnl", "sharpe",
                         "win_rate", "avg_daily_pnl", "max_dd", "num_days"]
        csv_all = results[download_cols].to_csv(index=False)
        st.download_button("📥 Download All Results", csv_all,
                           file_name="all_strategies.csv", mime="text/csv")

else:
    st.info("👈 Configure settings in the sidebar and click **Optimize** to discover your optimal battery schedule.")


