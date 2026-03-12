import pandas as pd
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone

DATA_DIR = Path("data")
OUT_DIR = Path("docs/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scheduler run date in UTC, used so the UI can filter by the day the workflow ran.
RUN_DATE = datetime.now(timezone.utc).date().isoformat()


def load_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        print(f"Missing file: {path}")
        return None
    print(f"Loading: {path}")
    return pd.read_csv(path)


def find_first(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def find_price_col(df):
    for c in df.columns:
        lc = c.lower()
        if "price" in lc or "vwap" in lc:
            return c
    raise ValueError(f"No price/vwap column found in columns: {list(df.columns)}")


def normalize(df, market_name):
    if df is None:
        return None

    out = df.copy()
    out["market"] = market_name

    price_col = find_price_col(out)
    out["price_value"] = pd.to_numeric(out[price_col], errors="coerce")

    area_col = find_first(out, ["area", "country", "market_area", "bidding_zone"])
    out["area"] = out[area_col].astype(str).str.strip() if area_col else "ALL"

    start_col = find_first(out, ["deliveryStartCET", "delivery_start_cet", "start_cet"])
    end_col = find_first(out, ["deliveryEndCET", "delivery_end_cet", "end_cet"])
    date_col = find_first(out, ["date_cet", "date"])

    if start_col:
      out["start"] = pd.to_datetime(out[start_col], errors="coerce")
    else:
      out["start"] = pd.NaT

    if end_col:
      out["end"] = pd.to_datetime(out[end_col], errors="coerce")
    elif start_col:
      out["end"] = out["start"] + pd.Timedelta(minutes=15)
    else:
      out["end"] = pd.NaT

    if start_col:
        out["date"] = out["start"].dt.date.astype(str)
        out["contract"] = out["start"].dt.strftime("%H:%M") + "-" + out["end"].dt.strftime("%H:%M")
        out["contract_sort"] = out["start"].dt.hour * 60 + out["start"].dt.minute
    else:
        if date_col:
            out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.date.astype(str)
        else:
            out["date"] = "unknown"
        out["row_num"] = out.groupby(["date", "area"]).cumcount() + 1
        out["contract"] = out["row_num"].apply(lambda x: f"Q{x:02d}")
        out["contract_sort"] = out["row_num"]

    # Add scheduler run date so the UI can filter by "the run from 11 March"
    out["run_date"] = RUN_DATE

    return out[
        [
            "date",
            "run_date",
            "area",
            "market",
            "price_value",
            "start",
            "end",
            "contract",
            "contract_sort",
        ]
    ]


def expand_to_quarters(df):
    if df is None or df.empty:
        return df

    rows = []

    for _, row in df.iterrows():
        start = row["start"]
        end = row["end"]

        if pd.isna(start) or pd.isna(end):
            rows.append(row.to_dict())
            continue

        duration_mins = (end - start).total_seconds() / 60

        if duration_mins <= 15:
            rows.append(row.to_dict())
            continue

        current = start
        while current < end:
            next_q = min(current + pd.Timedelta(minutes=15), end)
            new_row = row.to_dict()
            new_row["start"] = current
            new_row["end"] = next_q
            new_row["contract"] = current.strftime("%H:%M") + "-" + next_q.strftime("%H:%M")
            new_row["contract_sort"] = current.hour * 60 + current.minute
            new_row["date"] = str(current.date())
            new_row["run_date"] = RUN_DATE
            rows.append(new_row)
            current = next_q

    return pd.DataFrame(rows)


# Load datasets
market_frames = {
    "DA": normalize(load_csv("dayahead_prices.csv"), "DA"),
    "IDA1": normalize(load_csv("ida1_prices.csv"), "IDA1"),
    "IDA2": normalize(load_csv("ida2_prices.csv"), "IDA2"),
    "IDA3": normalize(load_csv("ida3_prices.csv"), "IDA3"),
    "VWAP": normalize(load_csv("intraday_continuous_vwap_qh.csv"), "VWAP"),
}

# Expand longer contracts (especially DA hourly) into quarter-hours
for key in list(market_frames.keys()):
    market_frames[key] = expand_to_quarters(market_frames[key])

available_markets = [k for k, v in market_frames.items() if v is not None and not v.empty]
if not available_markets:
    raise ValueError("No datasets loaded from data/")

base = pd.concat([market_frames[k] for k in available_markets], ignore_index=True)

# Clean up merge keys to reduce avoidable mismatches
base["date"] = base["date"].astype(str).str.strip()
base["run_date"] = base["run_date"].astype(str).str.strip()
base["area"] = base["area"].astype(str).str.strip()
base["contract"] = base["contract"].astype(str).str.strip()

# Build every unordered pair once; UI direction selector will handle forward/reverse
pairs = list(combinations(available_markets, 2))

profit_rows = []

for left_market, right_market in pairs:
    left_df = base[base["market"] == left_market].copy().rename(columns={"price_value": "left_price"})
    right_df = base[base["market"] == right_market].copy().rename(columns={"price_value": "right_price"})

    merged = left_df.merge(
        right_df,
        on=["date", "run_date", "area", "contract"],
        how="inner",
        suffixes=("_left", "_right"),
    )

    if merged.empty:
        print(f"No matches for pair: {left_market}_{right_market}")
        continue

    merged["rule"] = f"{left_market}_{right_market}"
    merged["buy_price"] = pd.to_numeric(merged["left_price"], errors="coerce")
    merged["sell_price"] = pd.to_numeric(merged["right_price"], errors="coerce")
    merged["profit"] = merged["sell_price"] - merged["buy_price"]

    if "contract_sort_left" in merged.columns:
        merged["contract_sort"] = merged["contract_sort_left"]
    elif "contract_sort_right" in merged.columns:
        merged["contract_sort"] = merged["contract_sort_right"]
    else:
        merged["contract_sort"] = 0

    merged = merged.dropna(subset=["buy_price", "sell_price", "profit"])

    if merged.empty:
        print(f"All rows dropped after numeric cleanup for pair: {left_market}_{right_market}")
        continue

    profit_rows.append(
        merged[
            [
                "date",
                "run_date",
                "area",
                "contract",
                "contract_sort",
                "rule",
                "buy_price",
                "sell_price",
                "profit",
            ]
        ]
    )

csv_path = OUT_DIR / "contract_profits.csv"
json_path = OUT_DIR / "contract_profits.json"

if not profit_rows:
    print("No matched rows were created. Writing empty output files.")
    empty_df = pd.DataFrame(
        columns=[
            "date",
            "run_date",
            "area",
            "contract",
            "contract_sort",
            "rule",
            "buy_price",
            "sell_price",
            "profit",
        ]
    )
    empty_df.to_csv(csv_path, index=False)
    json_path.write_text("[]", encoding="utf-8")
    print(f"Wrote empty CSV: {csv_path}")
    print(f"Wrote empty JSON: {json_path}")
    raise ValueError(
        "No matched rows were created. Check date/area/contract alignment across files."
    )

result = pd.concat(profit_rows, ignore_index=True).sort_values(
    ["run_date", "date", "area", "rule", "contract_sort", "contract"]
)

result.to_csv(csv_path, index=False)
json_path.write_text(result.to_json(orient="records"), encoding="utf-8")

print(f"Wrote CSV: {csv_path}")
print(f"Wrote JSON: {json_path}")
print(f"Rows written: {len(result)}")
print(f"Run date written: {RUN_DATE}")
print(f"Rules generated: {sorted(result['rule'].unique().tolist())}")
print(f"Market dates generated: {sorted(result['date'].unique().tolist())[:5]} ...")
