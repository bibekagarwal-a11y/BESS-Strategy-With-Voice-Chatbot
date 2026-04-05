import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta
import io

st.set_page_config(layout="wide", page_title="Battery Optimizer", page_icon="🔋")

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
    # Parse as UTC first (handles mixed +01:00/+02:00 offsets around DST),
    # then convert to Europe/Paris so hour/date reflect local CET/CEST wall-clock time.
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
        # include end date fully (add 1 day)
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
    """Format an absolute hour index (0..47) as HH:00, with '+1d' if past midnight."""
    h = int(h) % 48
    if h >= 24:
        return f"{h - 24:02d}:00 +1d"
    return f"{h:02d}:00"


def _fmt_window(start, length):
    """Format a time window like '22:00 → 02:00 +1d'."""
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
    """Format a list of (buy_start, sell_start) cycles into a human label."""
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
    """Build a (dates, D×24 price matrix) from a dataframe with date/hour/price cols.

    Cached by Streamlit so repeated calls with the same underlying df do not
    re-run the groupby/pivot.
    """
    if len(df) == 0:
        return [], np.zeros((0, 24))
    pivot = df.groupby(["date", "hour"])["price"].mean().unstack()
    dates = list(pivot.index)
    mat = np.full((len(dates), 24), np.nan)
    for h in pivot.columns:
        if 0 <= int(h) < 24:
            mat[:, int(h)] = pivot[h].values
    return dates, mat


def _enumerate_schedules(
    buy_ext, sell_ext, max_hour, ch, dis, max_cycles,
    E_in, E_out, fixed_cost_per_cycle, top_per_k,
    midnight_mode, overnight_allow_k1_any_order,
):
    """Pure-numpy enumeration of K-cycle schedules for a prebuilt (eff_days × max_hour)
    pair of buy/sell matrices. No recursion into optimize_battery.

    midnight_mode:
        "none"     → no filter (same-day schedules on a 24h matrix)
        "required" → at least one window must straddle hour 24 and not all on day 2
                     (used when enumerating on a 48h extended matrix)
    """
    eff_days = buy_ext.shape[0]
    if eff_days == 0:
        return []

    # Precompute window averages once
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

    # ---- K = 1: either-order non-overlapping (supports sell-then-buy) ----
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

    # ---- K >= 2: chronological enumeration with pruning ----
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
    buy_dates,
    buy_mat,
    sell_dates,
    sell_mat,
    capacity_mw,
    charge_hours,
    discharge_hours,
    max_cycles=1,
    allow_overnight=False,
    rte=1.0,
    degradation_eur_mwh=0.0,
    fee_eur_mwh=0.0,
    top_per_k=200,
):
    """
    Optimize K = 1..max_cycles battery schedules from pre-built price matrices.

    buy_mat / sell_mat: (D × 24) numpy arrays indexed by buy_dates / sell_dates.
    """
    # Align on common dates
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

    # Same-day enumeration on 24h matrix (always run)
    all_results.extend(
        _enumerate_schedules(
            buy24, sell24, 24, ch, dis, max_cycles,
            E_in, E_out, fixed_cost_per_cycle, top_per_k,
            midnight_mode="none",
            overnight_allow_k1_any_order=True,
        )
    )

    # Overnight enumeration on 48h extended matrix (added on top)
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
                midnight_mode="required",
                overnight_allow_k1_any_order=True,
            )
        )

    if not all_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    # Back-compat columns for the Sharpe heatmap (K=1 rows only)
    def _bs_from_cycle(row, idx, which):
        cl = row.get("cycle_list")
        if isinstance(cl, list) and len(cl) > idx:
            return cl[idx][0] if which == "b" else cl[idx][1]
        return np.nan
    results_df["buy_start"] = results_df.apply(lambda r: _bs_from_cycle(r, 0, "b"), axis=1)
    results_df["sell_start"] = results_df.apply(lambda r: _bs_from_cycle(r, 0, "s"), axis=1)
    return results_df.sort_values("sharpe", ascending=False).reset_index(drop=True)


def create_sharpe_heatmap(results_df):
    """Create heatmap of Sharpe ratios across buy/sell hours (1-cycle, same-day only)."""
    # Only makes sense for 1-cycle strategies with starts inside a single day
    if "cycles" in results_df.columns:
        one_cycle = results_df[results_df["cycles"] == 1]
    else:
        one_cycle = results_df
    one_cycle = one_cycle[(one_cycle["buy_start"] < 24) & (one_cycle["sell_start"] < 24)]

    heatmap_data = np.full((24, 24), np.nan)
    for _, row in one_cycle.iterrows():
        buy_start = int(row["buy_start"])
        sell_start = int(row["sell_start"])
        heatmap_data[buy_start, sell_start] = row["sharpe"]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=list(range(24)),
            y=list(range(24)),
            colorscale="RdYlGn",
            text=np.round(heatmap_data, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 8},
            colorbar=dict(title="Sharpe Ratio"),
        )
    )

    fig.update_layout(
        title="Sharpe Ratio Heatmap: Buy Hour vs Sell Hour",
        xaxis_title="Sell Start Hour",
        yaxis_title="Buy Start Hour",
        height=CHART_CONFIG["height"],
        margin=CHART_CONFIG["margin"],
        hovermode="closest",
    )

    return fig


def create_cumulative_pnl_chart(results_df, top_n=3):
    """Create cumulative P&L chart for top strategies."""
    fig = go.Figure()

    for idx in range(min(top_n, len(results_df))):
        row = results_df.iloc[idx]
        daily_pnl = row["daily_pnl"]
        cumulative_pnl = np.cumsum(daily_pnl)

        label = row.get("schedule", f"Strategy {idx+1}")

        fig.add_trace(
            go.Scatter(
                y=cumulative_pnl,
                mode="lines",
                name=f"#{idx+1}: {label}",
                line=dict(width=2, color=PALETTE["primary"][idx]),
            )
        )

    fig.update_layout(
        title="Cumulative P&L: Top 3 Strategies",
        yaxis_title="Cumulative P&L (EUR)",
        xaxis_title="Days",
        height=CHART_CONFIG["height"],
        margin=CHART_CONFIG["margin"],
        hovermode="x unified",
    )

    return fig


def create_daily_pnl_chart(daily_pnl):
    """Create bar chart of daily P&L."""
    colors = [
        PALETTE["green"][2] if x >= 0 else PALETTE["accent"][2] for x in daily_pnl
    ]

    fig = go.Figure(
        data=go.Bar(y=daily_pnl, marker_color=colors, text=np.round(daily_pnl, 0))
    )

    fig.update_layout(
        title="Daily P&L: Optimal Strategy #1",
        yaxis_title="P&L (EUR)",
        xaxis_title="Day",
        height=CHART_CONFIG["height"],
        margin=CHART_CONFIG["margin"],
        showlegend=False,
    )

    return fig


def create_hourly_profile(df, price_column):
    """Create average hourly price profile."""
    hourly_avg = df.groupby("hour")[price_column].mean().reset_index()

    fig = go.Figure(
        data=go.Bar(
            x=hourly_avg["hour"],
            y=hourly_avg[price_column],
            marker_color=PALETTE["primary"][2],
        )
    )

    fig.update_layout(
        title="Average Price by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Price (EUR/MWh)",
        height=CHART_CONFIG["height"],
        margin=CHART_CONFIG["margin"],
        showlegend=False,
    )

    return fig


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.title("⚙️ Battery Optimizer Settings")

capacity_mw = st.sidebar.number_input(
    "Battery Capacity (MW)", min_value=0.1, max_value=100.0, value=1.0, step=0.1
)

charge_hours = st.sidebar.slider(
    "Charge Duration (hours)", min_value=1, max_value=8, value=4
)

discharge_hours = st.sidebar.slider(
    "Discharge Duration (hours)", min_value=1, max_value=8, value=4
)

max_cycles = st.sidebar.slider(
    "Max cycles per day",
    min_value=1,
    max_value=6,
    value=1,
    help="Maximum number of complete charge/discharge cycles per day. The "
         "optimizer searches all schedules with 1..max cycles and ranks them "
         "together so you can compare single vs. multi-cycle strategies.",
)

allow_overnight = st.sidebar.checkbox(
    "Allow cross-day (overnight) cycles",
    value=False,
    help="If enabled, cycles may straddle midnight — e.g. buy 22:00–02:00, "
         "sell 06:00–10:00 next day.",
)

with st.sidebar.expander("⚙️ Advanced: costs & efficiency", expanded=False):
    rte_pct = st.number_input(
        "Round-trip efficiency (%)",
        min_value=50.0,
        max_value=100.0,
        value=100.0,
        step=1.0,
        help="Energy delivered ÷ energy charged. A 1 MWh charge delivers RTE×1 MWh "
             "on discharge. Default 100% = no loss.",
    )
    degradation_eur_mwh = st.number_input(
        "Degradation cost (€ / MWh discharged)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,
        help="Wear-and-tear cost charged per MWh of discharged energy. "
             "Common values 2–10 €/MWh. Default 0.",
    )
    fee_eur_mwh = st.number_input(
        "Exchange fees (€ / MWh traded)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.05,
        help="Per-MWh exchange fee paid on BOTH the buy and sell legs. "
             "Default 0.",
    )
rte = rte_pct / 100.0

selected_markets = st.sidebar.multiselect(
    "Markets to use",
    options=list(DATASETS.keys()),
    default=["DayAhead", "IDA1"],
    help="Pick one or more Nord Pool markets. The optimizer will try every "
         "combination of buy-market → sell-market across the selected set "
         "(including buying and selling in the same market) and rank the best "
         "strategies together. Select just one market to search intra-market "
         "arbitrage; select multiple to search cross-market arbitrage.",
)

if not selected_markets:
    st.sidebar.warning("Select at least one market to run the optimizer.")

area = st.sidebar.selectbox("Area", options=AREAS, index=4)  # GER = index 3

day_filter = st.sidebar.selectbox(
    "Day Filter", options=["All", "Weekdays", "Weekends"], index=0
)

date_range = st.sidebar.selectbox(
    "Date Range",
    options=[
        "All dates",
        "Last 1 week",
        "Last 2 weeks",
        "Last 1 month",
        "Last 2 months",
        "Custom range",
    ],
    index=0,
)

custom_start = None
custom_end = None
if date_range == "Custom range":
    # Peek at available dates from the first selected market to set sensible bounds
    _peek_market = selected_markets[0] if selected_markets else list(DATASETS.keys())[0]
    try:
        _peek_df = load_data(DATASETS[_peek_market][0])
        _min_avail = _peek_df["date"].min()
        _max_avail = _peek_df["date"].max()
    except Exception:
        _min_avail = None
        _max_avail = None

    default_end = _max_avail if _max_avail is not None else datetime.today().date()
    default_start = (
        _min_avail
        if _min_avail is not None
        else (default_end - timedelta(days=14))
    )
    # Default selection: last 2 weeks of available data
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
        custom_start = picked
        custom_end = picked

optimize_clicked = st.sidebar.button("🚀 Optimize", type="primary")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🔋 Battery Optimizer")
st.markdown(
    "Discover optimal buy/sell schedules across Nord Pool electricity markets"
)

if optimize_clicked:
    if not selected_markets:
        st.error("Please select at least one market in the sidebar.")
        st.stop()

    with st.spinner("Loading data and optimizing across markets..."):
        # Load and prepare each selected market once, then build its price matrix once
        market_dfs = {}
        market_mats = {}
        for mkt in selected_markets:
            fname, pcol = DATASETS[mkt]
            df = load_data(fname)
            df = df[df["area"] == area].copy()
            df = filter_by_date_range(df, date_range, custom_start, custom_end)
            df = filter_by_day_type(df, day_filter)
            if pcol != "price":
                df = df.rename(columns={pcol: "price"})
            market_dfs[mkt] = df
            market_mats[mkt] = _build_price_matrix(df)

        # Run optimization for every (buy_market, sell_market) ordered pair
        # (same market on both sides is allowed — intra-market arbitrage)
        combined = []
        for buy_market in selected_markets:
            b_dates, b_mat = market_mats[buy_market]
            for sell_market in selected_markets:
                s_dates, s_mat = market_mats[sell_market]
                part = optimize_battery(
                    b_dates, b_mat,
                    s_dates, s_mat,
                    capacity_mw,
                    charge_hours,
                    discharge_hours,
                    max_cycles=max_cycles,
                    allow_overnight=allow_overnight,
                    rte=rte,
                    degradation_eur_mwh=degradation_eur_mwh,
                    fee_eur_mwh=fee_eur_mwh,
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
                combined.append(part)

        if combined:
            results = (
                pd.concat(combined, ignore_index=True)
                .sort_values("sharpe", ascending=False)
                .reset_index(drop=True)
            )
        else:
            results = pd.DataFrame()

        if len(results) == 0:
            st.error("No valid optimization results. Try adjusting parameters.")
        else:
            best_strategy = results.iloc[0]

            # ================================================================
            # TOP KPIs ROW
            # ================================================================
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric(
                    "Total P&L",
                    f"€{best_strategy['total_pnl']:,.0f}",
                )

            with col2:
                st.metric(
                    "Sharpe Ratio",
                    f"{best_strategy['sharpe']:.2f}",
                )

            with col3:
                st.metric(
                    "Max Drawdown",
                    f"€{best_strategy['max_dd']:,.0f}",
                )

            with col4:
                st.metric(
                    "Win Rate",
                    f"{best_strategy['win_rate']*100:.1f}%",
                )

            with col5:
                st.metric(
                    "Avg Daily P&L",
                    f"€{best_strategy['avg_daily_pnl']:,.0f}",
                )

            with col6:
                st.metric(
                    "# Days",
                    f"{int(best_strategy['num_days'])}",
                )

            # ================================================================
            # OPTIMAL SCHEDULE CARD
            # ================================================================
            st.divider()

            schedule_text = f"**{best_strategy['market_pair']}:** {best_strategy['schedule']}"

            annual_pnl = best_strategy["avg_daily_pnl"] * 365

            schedule_info = f"""
            {schedule_text}

            **Area:** {area} | **Capacity:** {capacity_mw} MW | **Est. Annual P&L:** €{annual_pnl:,.0f}
            """

            st.info(schedule_info)

            # ================================================================
            # TWO COLUMNS: HEATMAP + CUMULATIVE P&L
            # ================================================================
            st.divider()
            st.subheader("Strategy Analysis")

            col_heatmap, col_cumulative = st.columns(2)

            with col_heatmap:
                heatmap_fig = create_sharpe_heatmap(results)
                st.plotly_chart(heatmap_fig, use_container_width=True)

            with col_cumulative:
                cumulative_fig = create_cumulative_pnl_chart(results, top_n=3)
                st.plotly_chart(cumulative_fig, use_container_width=True)

            # ================================================================
            # TOP 10 STRATEGIES TABLE
            # ================================================================
            st.divider()
            st.subheader("Top 10 Strategies")

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

            display_cols = [
                "Rank",
                "Market",
                "Cycles",
                "Schedule",
                "Total P&L",
                "Sharpe",
                "Win Rate",
                "Avg Daily",
                "Max DD",
            ]

            st.dataframe(top_10[display_cols], use_container_width=True, hide_index=True)

            # Download button for top strategies
            csv_top = top_10[display_cols].to_csv(index=False)
            st.download_button(
                "📥 Download Top 10 Strategies",
                csv_top,
                file_name="top_strategies.csv",
                mime="text/csv",
            )

            # ================================================================
            # DAILY P&L CHART
            # ================================================================
            st.divider()
            st.subheader("Daily P&L: Strategy #1")

            daily_pnl_fig = create_daily_pnl_chart(best_strategy["daily_pnl"])
            st.plotly_chart(daily_pnl_fig, use_container_width=True)

            # ================================================================
            # HOURLY PRICE PROFILE
            # ================================================================
            st.divider()
            st.subheader("Hourly Price Profile")

            profile_market = st.selectbox(
                "Market to show",
                options=selected_markets,
                index=0,
                key="_profile_market",
            )
            st.markdown(f"**Market:** {profile_market} | **Area:** {area}")
            hourly_fig = create_hourly_profile(market_dfs[profile_market], "price")
            st.plotly_chart(hourly_fig, use_container_width=True)

            # ================================================================
            # DOWNLOAD ALL RESULTS
            # ================================================================
            st.divider()

            # Prepare full results for download
            download_df = results.copy()

            download_cols = [
                "market_pair",
                "cycles",
                "schedule",
                "total_pnl",
                "sharpe",
                "win_rate",
                "avg_daily_pnl",
                "max_dd",
                "num_days",
            ]

            csv_all = download_df[download_cols].to_csv(index=False)
            st.download_button(
                "📥 Download All Results",
                csv_all,
                file_name="all_strategies.csv",
                mime="text/csv",
            )
else:
    st.info(
        "👈 Configure settings in the sidebar and click **Optimize** to discover your optimal battery schedule."
    )
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           