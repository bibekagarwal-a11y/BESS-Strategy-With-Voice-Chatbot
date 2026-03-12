import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("data")
charts_dir = Path("charts")
reports_dir = Path("reports")

charts_dir.mkdir(exist_ok=True)
reports_dir.mkdir(exist_ok=True)


def load_csv(name):
    path = data_dir / name
    if path.exists():
        df = pd.read_csv(path)
        df["source"] = name
        return df
    return None


def find_price_col(df):
    candidates = [c for c in df.columns if "price" in c.lower() or "vwap" in c.lower()]
    if not candidates:
        raise ValueError(f"No price-like column found in columns: {list(df.columns)}")
    return candidates[0]


def find_area_col(df):
    candidates = [c for c in df.columns if c.lower() in ["area", "country", "market_area", "bidding_zone"]]
    if candidates:
        return candidates[0]
    return None


def normalize_dataset(df, market_name):
    if df is None:
        return None

    price_col = find_price_col(df)
    area_col = find_area_col(df)

    out = df.copy()
    out["market"] = market_name
    out["price_value"] = pd.to_numeric(out[price_col], errors="coerce")

    if area_col:
        out["area_name"] = out[area_col].astype(str)
    else:
        out["area_name"] = "ALL"

    if "date_cet" in out.columns:
        out["date_cet"] = pd.to_datetime(out["date_cet"], errors="coerce")
    else:
        out["date_cet"] = pd.NaT

    return out


dayahead = load_csv("dayahead_prices.csv")
ida1 = load_csv("ida1_prices.csv")
ida2 = load_csv("ida2_prices.csv")
ida3 = load_csv("ida3_prices.csv")
intraday = load_csv("intraday_continuous_vwap_qh.csv")

datasets = {
    "DayAhead": normalize_dataset(dayahead, "DayAhead"),
    "IDA1": normalize_dataset(ida1, "IDA1"),
    "IDA2": normalize_dataset(ida2, "IDA2"),
    "IDA3": normalize_dataset(ida3, "IDA3"),
    "IntradayVWAP": normalize_dataset(intraday, "IntradayVWAP"),
}

available = [df for df in datasets.values() if df is not None]
if not available:
    raise ValueError("No CSV files found in data/")

all_prices = pd.concat(available, ignore_index=True)

# -------------------------
# Global summary
# -------------------------
summary = (
    all_prices.groupby(["area_name", "market"])["price_value"]
    .agg(["mean", "min", "max", "std"])
    .reset_index()
)

summary["spread_max_min"] = summary["max"] - summary["min"]
summary.to_csv(reports_dir / "summary_by_area_market.csv", index=False)

# -------------------------
# Daily averages by area + market
# -------------------------
daily = (
    all_prices.dropna(subset=["date_cet"])
    .groupby(["area_name", "date_cet", "market"])["price_value"]
    .mean()
    .reset_index()
    .sort_values(["area_name", "market", "date_cet"])
)

daily.to_csv(reports_dir / "daily_market_averages_by_area.csv", index=False)

areas = sorted(daily["area_name"].dropna().unique())

all_yesterday_today_rows = []
all_anomaly_rows = []
all_arbitrage_rows = []
all_arbitrage_summary_rows = []

# -------------------------
# Separate charts per area
# -------------------------
for area in areas:
    area_safe = str(area).replace("/", "_").replace("\\", "_").replace(" ", "_")
    area_chart_dir = charts_dir / area_safe
    area_report_dir = reports_dir / area_safe
    area_chart_dir.mkdir(exist_ok=True)
    area_report_dir.mkdir(exist_ok=True)

    area_df = all_prices[all_prices["area_name"] == area].copy()
    area_daily = daily[daily["area_name"] == area].copy()

    # 1. Price trends per area
    plt.figure(figsize=(10, 6))
    for market, group in area_df.groupby("market"):
        plt.plot(group["price_value"].reset_index(drop=True), label=market)

    plt.legend()
    plt.title(f"Price Comparison - {area}")
    plt.ylabel("€/MWh")
    plt.xlabel("Observation")
    plt.tight_layout()
    plt.savefig(area_chart_dir / "price_trends.png")
    plt.close()

    # 2. Max-min spread per area
    area_summary = (
        area_df.groupby("market")["price_value"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )
    area_summary["spread_max_min"] = area_summary["max"] - area_summary["min"]
    area_summary.to_csv(area_report_dir / "summary.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(area_summary["market"], area_summary["spread_max_min"])
    plt.title(f"Max-Min Spread by Market - {area}")
    plt.ylabel("Spread €/MWh")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(area_chart_dir / "spreads.png")
    plt.close()

    # 3. Yesterday vs today per area
    area_changes = []
    for market, group in area_daily.groupby("market"):
        group = group.sort_values("date_cet").copy()
        if len(group) >= 2:
            yday_row = group.iloc[-2]
            today_row = group.iloc[-1]
            change_abs = today_row["price_value"] - yday_row["price_value"]
            change_pct = (change_abs / yday_row["price_value"] * 100) if yday_row["price_value"] != 0 else None

            row = {
                "area_name": area,
                "market": market,
                "yesterday": yday_row["date_cet"],
                "today": today_row["date_cet"],
                "yesterday_avg": yday_row["price_value"],
                "today_avg": today_row["price_value"],
                "change_abs": change_abs,
                "change_pct": change_pct,
            }
            area_changes.append(row)
            all_yesterday_today_rows.append(row)

    area_changes_df = pd.DataFrame(area_changes)
    if not area_changes_df.empty:
        area_changes_df.to_csv(area_report_dir / "yesterday_vs_today.csv", index=False)

        plt.figure(figsize=(10, 6))
        plt.bar(area_changes_df["market"], area_changes_df["change_abs"])
        plt.title(f"Yesterday vs Today - {area}")
        plt.ylabel("Change €/MWh")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(area_chart_dir / "yesterday_vs_today.png")
        plt.close()

    # 4. 7-day anomaly detection per area
    area_anomalies = []
    for market, group in area_daily.groupby("market"):
        group = group.sort_values("date_cet").copy()
        group["rolling_7d_avg"] = group["price_value"].rolling(7, min_periods=2).mean()
        group["rolling_7d_std"] = group["price_value"].rolling(7, min_periods=2).std()
        group["zscore_like"] = (group["price_value"] - group["rolling_7d_avg"]) / group["rolling_7d_std"]
        latest = group.iloc[-1]

        row = {
            "area_name": area,
            "market": market,
            "date": latest["date_cet"],
            "price": latest["price_value"],
            "rolling_7d_avg": latest["rolling_7d_avg"],
            "rolling_7d_std": latest["rolling_7d_std"],
            "zscore_like": latest["zscore_like"],
            "is_anomaly_gt_2std": pd.notna(latest["zscore_like"]) and abs(latest["zscore_like"]) > 2,
        }
        area_anomalies.append(row)
        all_anomaly_rows.append(row)

    area_anomaly_df = pd.DataFrame(area_anomalies)
    if not area_anomaly_df.empty:
        area_anomaly_df.to_csv(area_report_dir / "anomaly_detection.csv", index=False)

    # 5. Arbitrage per area: buy DA, sell intraday
    area_arbitrage_rows = []

    area_dayahead = area_daily[area_daily["market"] == "DayAhead"][["date_cet", "price_value"]].rename(
        columns={"price_value": "dayahead_avg"}
    )

    for intraday_market in ["IDA1", "IDA2", "IDA3", "IntradayVWAP"]:
        intraday_daily = area_daily[area_daily["market"] == intraday_market][["date_cet", "price_value"]].rename(
            columns={"price_value": "intraday_avg"}
        )
        merged = area_dayahead.merge(intraday_daily, on="date_cet", how="inner")
        if not merged.empty:
            merged["area_name"] = area
            merged["market_pair"] = f"DayAhead_to_{intraday_market}"
            merged["profit_if_buy_DA_sell_intraday"] = merged["intraday_avg"] - merged["dayahead_avg"]
            area_arbitrage_rows.append(merged)

    if area_arbitrage_rows:
        area_arbitrage_df = pd.concat(area_arbitrage_rows, ignore_index=True)
        area_arbitrage_df.to_csv(area_report_dir / "arbitrage_opportunities.csv", index=False)
        all_arbitrage_rows.append(area_arbitrage_df)

        area_arb_summary = (
            area_arbitrage_df.groupby("market_pair")["profit_if_buy_DA_sell_intraday"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        area_arb_summary["area_name"] = area
        area_arb_summary.to_csv(area_report_dir / "arbitrage_summary.csv", index=False)
        all_arbitrage_summary_rows.append(area_arb_summary)

        plt.figure(figsize=(10, 6))
        plt.bar(area_arb_summary["market_pair"], area_arb_summary["mean"])
        plt.title(f"Arbitrage Summary - {area}")
        plt.ylabel("Average Profit €/MWh")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(area_chart_dir / "arbitrage_summary.png")
        plt.close()

    # 6. Markdown report per area
    area_report_lines = [f"# Daily Electricity Market Analysis - {area}", ""]

    if not area_summary.empty:
        top_market = area_summary.sort_values("mean", ascending=False).iloc[0]
        spread_market = area_summary.sort_values("spread_max_min", ascending=False).iloc[0]

        area_report_lines += [
            "## Highest average price",
            f"{top_market['market']} — {top_market['mean']:.2f} €/MWh",
            "",
            "## Largest spread",
            f"{spread_market['market']} — {spread_market['spread_max_min']:.2f} €/MWh",
            "",
        ]

    if not area_changes_df.empty:
        biggest_move = area_changes_df.iloc[area_changes_df["change_abs"].abs().idxmax()]
        area_report_lines += [
            "## Biggest yesterday-vs-today move",
            f"{biggest_move['market']} — {biggest_move['change_abs']:.2f} €/MWh ({biggest_move['change_pct']:.2f}%)",
            "",
        ]

    if not area_anomaly_df.empty:
        flagged = area_anomaly_df[area_anomaly_df["is_anomaly_gt_2std"] == True]
        area_report_lines += ["## 7-day anomalies"]
        if flagged.empty:
            area_report_lines += ["No strong anomalies detected (> 2 rolling std).", ""]
        else:
            for _, row in flagged.iterrows():
                area_report_lines.append(f"{row['market']} on {pd.to_datetime(row['date']).date()} flagged as anomaly.")
            area_report_lines.append("")

    (area_report_dir / "daily_report.md").write_text("\n".join(area_report_lines))

# -------------------------
# Save merged outputs too
# -------------------------
if all_yesterday_today_rows:
    pd.DataFrame(all_yesterday_today_rows).to_csv(reports_dir / "yesterday_vs_today_all_areas.csv", index=False)

if all_anomaly_rows:
    pd.DataFrame(all_anomaly_rows).to_csv(reports_dir / "anomaly_detection_all_areas.csv", index=False)

if all_arbitrage_rows:
    pd.concat(all_arbitrage_rows, ignore_index=True).to_csv(reports_dir / "arbitrage_opportunities_all_areas.csv", index=False)

if all_arbitrage_summary_rows:
    pd.concat(all_arbitrage_summary_rows, ignore_index=True).to_csv(reports_dir / "arbitrage_summary_all_areas.csv", index=False)

print("Analysis complete. Separate charts created for each area.")
