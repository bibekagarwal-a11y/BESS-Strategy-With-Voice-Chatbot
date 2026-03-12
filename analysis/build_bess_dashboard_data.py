from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from bess_optimizer import BatteryConfig, optimize_day_sequential

DATA_DIR = Path("docs/data")
OUT_DIR = Path("docs/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_contract_profits() -> pd.DataFrame:
    path = DATA_DIR / "contract_profits.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    return df


def infer_market_code_from_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert rule-based rows into market-specific rows for the BESS engine.
    For now, use the strategy pair to recover market codes.
    """
    out = df.copy()

    # Example rule strings: DA_IDA1, IDA1_VWAP, etc.
    out["left_market"] = out["rule"].str.split("_").str[0]
    out["right_market"] = out["rule"].str.split("_").str[1]

    # Build market-specific price rows from buy/sell columns
    left_rows = out[["date", "area", "contract", "contract_sort", "left_market", "buy_price"]].copy()
    left_rows.columns = ["date", "area", "contract", "contract_sort", "market_code", "price"]

    right_rows = out[["date", "area", "contract", "contract_sort", "right_market", "sell_price"]].copy()
    right_rows.columns = ["date", "area", "contract", "contract_sort", "market_code", "price"]

    merged = pd.concat([left_rows, right_rows], ignore_index=True).drop_duplicates(
        subset=["date", "area", "contract", "contract_sort", "market_code"]
    )

    return merged


def main() -> None:
    df = load_contract_profits()
    market_prices = infer_market_code_from_rule(df)

    config = BatteryConfig(
        capacity_mwh=1.0,
        max_charge_mw=1.0,
        max_discharge_mw=1.0,
        roundtrip_efficiency=0.90,
        initial_soc_mwh=0.0,
        final_soc_target_mwh=0.0,
    )

    summary_rows = []
    trade_rows = []
    schedule_rows = []

    grouped = market_prices.groupby(["area", "date"], dropna=False)

    for (area, date), g in grouped:
        results = optimize_day_sequential(g, config)

        total_pnl = 0.0
        for stage_name, stage_result in results.items():
            total_pnl += stage_result.pnl_eur

            summary_rows.append({
                "area": area,
                "date": date,
                "stage": stage_name,
                "stage_pnl_eur": stage_result.pnl_eur,
            })

            if not stage_result.trades.empty:
                tr = stage_result.trades.copy()
                tr["area"] = area
                tr["date"] = date
                trade_rows.append(tr)

            if not stage_result.schedule.empty:
                sc = stage_result.schedule.copy()
                sc["area"] = area
                sc["date"] = date
                sc["stage"] = stage_name
                schedule_rows.append(sc)

        summary_rows.append({
            "area": area,
            "date": date,
            "stage": "TOTAL",
            "stage_pnl_eur": total_pnl,
        })

    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    schedule_df = pd.concat(schedule_rows, ignore_index=True) if schedule_rows else pd.DataFrame()

    summary_df.to_csv(OUT_DIR / "bess_summary.csv", index=False)
    trades_df.to_csv(OUT_DIR / "bess_trades.csv", index=False)
    schedule_df.to_csv(OUT_DIR / "bess_schedule.csv", index=False)

    (OUT_DIR / "bess_summary.json").write_text(summary_df.to_json(orient="records"))
    (OUT_DIR / "bess_trades.json").write_text(trades_df.to_json(orient="records"))
    (OUT_DIR / "bess_schedule.json").write_text(schedule_df.to_json(orient="records"))

    print("Wrote BESS dashboard data:")
    print(OUT_DIR / "bess_summary.csv")
    print(OUT_DIR / "bess_trades.csv")
    print(OUT_DIR / "bess_schedule.csv")


if __name__ == "__main__":
    main()
