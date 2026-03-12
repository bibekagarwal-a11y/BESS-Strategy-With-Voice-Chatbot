from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass
class BatteryConfig:
    capacity_mwh: float = 1.0
    max_charge_mw: float = 1.0
    max_discharge_mw: float = 1.0
    roundtrip_efficiency: float = 0.90
    initial_soc_mwh: float = 0.0
    final_soc_target_mwh: float = 0.0


@dataclass
class StageResult:
    stage_name: str
    pnl_eur: float
    trades: pd.DataFrame
    schedule: pd.DataFrame


def _duration_hours(contract_label: str) -> float:
    if not contract_label or "-" not in contract_label:
        return 0.25

    start_str, end_str = contract_label.split("-")
    sh, sm = map(int, start_str.split(":"))
    eh, em = map(int, end_str.split(":"))

    start = sh * 60 + sm
    end = eh * 60 + em

    if end < start:
        end += 24 * 60

    return (end - start) / 60


def _prepare_day_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe safely before optimization.
    Ensures required columns exist and sorts correctly.
    """

    out = df.copy()

    # Ensure rule column exists
    if "rule" not in out.columns:
        out["rule"] = "unknown"

    # Ensure contract_sort exists
    if "contract_sort" not in out.columns:
        out["contract_sort"] = range(len(out))

    # Safe sorting
    sort_cols = [c for c in ["contract_sort", "rule"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def _simple_single_market_schedule(
    prices: pd.Series,
    contracts: pd.Series,
    contract_sort: pd.Series,
    config: BatteryConfig,
    stage_name: str,
) -> StageResult:
    """
    Very practical heuristic:
    - charge on relatively low-price intervals
    - discharge on relatively high-price intervals
    - enforce SOC, power, capacity
    - no simultaneous charge/discharge
    """

    df = pd.DataFrame({
        "contract": contracts.values,
        "contract_sort": contract_sort.values,
        "price": prices.values,
    }).sort_values("contract_sort").reset_index(drop=True)

    df["duration_h"] = df["contract"].apply(_duration_hours)
    df["charge_mwh"] = 0.0
    df["discharge_mwh"] = 0.0
    df["soc_mwh"] = 0.0
    df["cashflow_eur"] = 0.0

    low_threshold = df["price"].quantile(0.30)
    high_threshold = df["price"].quantile(0.70)

    soc = config.initial_soc_mwh
    eta_charge = config.roundtrip_efficiency ** 0.5
    eta_discharge = config.roundtrip_efficiency ** 0.5

    trades = []

    for i, row in df.iterrows():

        duration_h = row["duration_h"]
        max_charge_mwh = config.max_charge_mw * duration_h
        max_discharge_mwh = config.max_discharge_mw * duration_h
        price = row["price"]

        charge = 0.0
        discharge = 0.0

        if price <= low_threshold:

            room = max(config.capacity_mwh - soc, 0.0)
            charge = min(max_charge_mwh * eta_charge, room)

            soc += charge

            cashflow = -(charge / eta_charge) * price if eta_charge > 0 else -charge * price

            trades.append({
                "stage": stage_name,
                "contract": row["contract"],
                "side": "charge",
                "energy_mwh": charge,
                "price_eur_mwh": price,
                "cashflow_eur": cashflow,
            })

            df.at[i, "charge_mwh"] = charge
            df.at[i, "cashflow_eur"] += cashflow

        elif price >= high_threshold:

            available = max(soc, 0.0)
            raw_discharge = min(max_discharge_mwh / max(eta_discharge, 1e-9), available)

            discharge = raw_discharge * eta_discharge
            soc -= raw_discharge

            cashflow = discharge * price

            trades.append({
                "stage": stage_name,
                "contract": row["contract"],
                "side": "discharge",
                "energy_mwh": discharge,
                "price_eur_mwh": price,
                "cashflow_eur": cashflow,
            })

            df.at[i, "discharge_mwh"] = discharge
            df.at[i, "cashflow_eur"] += cashflow

        df.at[i, "soc_mwh"] = soc

    # End-of-day SOC correction
    if soc > config.final_soc_target_mwh and len(df) > 0:

        last_idx = df.index[-1]
        excess = soc - config.final_soc_target_mwh
        price = df.at[last_idx, "price"]

        eta_discharge = config.roundtrip_efficiency ** 0.5
        delivered = excess * eta_discharge
        cashflow = delivered * price

        df.at[last_idx, "discharge_mwh"] += delivered
        df.at[last_idx, "cashflow_eur"] += cashflow

        soc = config.final_soc_target_mwh
        df.at[last_idx, "soc_mwh"] = soc

        trades.append({
            "stage": stage_name,
            "contract": df.at[last_idx, "contract"],
            "side": "discharge_eod",
            "energy_mwh": delivered,
            "price_eur_mwh": price,
            "cashflow_eur": cashflow,
        })

    pnl = float(df["cashflow_eur"].sum())

    return StageResult(
        stage_name=stage_name,
        pnl_eur=pnl,
        trades=pd.DataFrame(trades),
        schedule=df,
    )


def optimize_day_sequential(
    area_day_df: pd.DataFrame,
    config: BatteryConfig,
) -> Dict[str, StageResult]:

    """
    Sequential battery strategy:
    1. DA baseline
    2. IDA1 uplift
    3. IDA2 uplift
    4. IDA3 uplift
    5. Continuous uplift (VWAP)
    """

    area_day_df = _prepare_day_frame(area_day_df)

    stage_market_map = {
        "DA": "DA",
        "IDA1": "IDA1",
        "IDA2": "IDA2",
        "IDA3": "IDA3",
        "VWAP": "VWAP",
    }

    results: Dict[str, StageResult] = {}

    for stage_name, market_code in stage_market_map.items():

        market_rows = area_day_df[area_day_df["market_code"] == market_code].copy()

        if market_rows.empty:
            continue

        result = _simple_single_market_schedule(
            prices=market_rows["price"],
            contracts=market_rows["contract"],
            contract_sort=market_rows["contract_sort"],
            config=config,
            stage_name=stage_name,
        )

        results[stage_name] = result

    return results
