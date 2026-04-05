import os
import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

# Set the timezone to Europe/Paris
TZ = ZoneInfo("Europe/Paris")

# Public Nord Pool dataportal APIs (no auth)
PRICES_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
INTRADAY_STATS_URL = "https://dataportal-api.nordpoolgroup.com/api/IntradayMarketStatistics"

# Configuration from environment variables
AREAS = os.environ.get("AREAS", "FR,GER" )
CURRENCY = os.environ.get("CURRENCY", "EUR")
START_DATE = os.environ.get("START_DATE", "2026-03-01")

# Permanent datasets directory
DATA_DIR = "data"

# Raw JSON directory for debugging (GitHub Actions artifacts)
ARTIFACTS_RAW_DIR = os.path.join("artifacts", "raw")


def paris_now() -> datetime:
    """Returns the current time in Europe/Paris timezone."""
    return datetime.now(tz=TZ)


def daterange(d0: date, d1: date):
    """Generates a range of dates from d0 to d1 inclusive."""
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def ensure_dirs():
    """Ensures that the necessary directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_RAW_DIR, exist_ok=True)


def write_raw(prefix: str, market: str, d: date, payload: object):
    """Saves raw JSON payloads for debugging."""
    ensure_dirs()
    path = os.path.join(ARTIFACTS_RAW_DIR, f"{prefix}_{market}_{d.isoformat()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_utc_iso_to_paris(iso_utc: Optional[str]) -> Optional[str]:
    """Converts UTC ISO string to Europe/Paris ISO string."""
    if not iso_utc:
        return None
    s = iso_utc
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt.astimezone(TZ).isoformat()


def fetch_prices(d: date, market: str) -> object:
    """Fetches auction prices from Nord Pool."""
    print(f"Fetching {market} prices for {d.isoformat()}...")
    r = requests.get(
        PRICES_URL,
        params={
            "date": d.isoformat(),
            "market": market,
            "deliveryArea": AREAS,
            "currency": CURRENCY,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def extract_auction_rows(payload: Any) -> List[Dict[str, Any]]:
    """Extracts relevant rows from the auction payload."""
    target_areas = [a.strip().upper() for a in AREAS.split(",") if a.strip()]
    delivery_date_cet = payload.get("deliveryDateCET")
    currency = payload.get("currency", CURRENCY)
    market = payload.get("market", "Unknown")

    rows: List[Dict[str, Any]] = []
    for e in payload.get("multiAreaEntries", []) or []:
        start_utc = e.get("deliveryStart")
        end_utc = e.get("deliveryEnd")
        per_area = e.get("entryPerArea", {}) or {}
        start_cet = parse_utc_iso_to_paris(start_utc)
        end_cet = parse_utc_iso_to_paris(end_utc)

        for area in target_areas:
            if area in per_area:
                rows.append({
                    "market": market,
                    "date_cet": delivery_date_cet,
                    "area": area,
                    "deliveryStartUTC": start_utc,
                    "deliveryEndUTC": end_utc,
                    "deliveryStartCET": start_cet,
                    "deliveryEndCET": end_cet,
                    "price": per_area[area],
                    "currency": currency,
                })
    return rows


def fetch_intraday_stats(d: date, area: str) -> object:
    """Fetches intraday market statistics."""
    print(f"Fetching intraday stats for {area} on {d.isoformat()}...")
    r = requests.get(
        INTRADAY_STATS_URL,
        params={"date": d.isoformat(), "deliveryArea": area},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def extract_vwap_qh_rows(payload: Any, area: str) -> List[Dict[str, Any]]:
    """Extracts VWAP rows for quarter-hour contracts."""
    rows: List[Dict[str, Any]] = []
    delivery_date_cet = payload.get("deliveryDateCET", None)
    area_payload = (payload.get("deliveryArea") or area).strip().upper()

    for c in payload.get("contracts", []) or []:
        name = c.get("contractName")
        if not name or not name.startswith("QH-"):
            continue
        start_utc = c.get("deliveryStart")
        end_utc = c.get("deliveryEnd")
        vwap = c.get("averagePrice")

        if not (start_utc and end_utc) or not isinstance(vwap, (int, float)):
            continue

        rows.append({
            "market": "IntradayContinuousVWAP",
            "date_cet": delivery_date_cet,
            "area": area_payload,
            "contractName": name,
            "contractId": c.get("contractId"),
            "deliveryStartUTC": start_utc,
            "deliveryEndUTC": end_utc,
            "deliveryStartCET": parse_utc_iso_to_paris(start_utc),
            "deliveryEndCET": parse_utc_iso_to_paris(end_utc),
            "vwap": float(vwap),
            "volume": c.get("volume"),
            "priceUnit": payload.get("priceUnit"),
            "volumeUnit": payload.get("volumeUnit"),
        })
    return rows


def upsert_csv(rows: List[Dict[str, Any]], filename: str, sort_cols: List[str]):
    """Merges new rows into CSV, drops duplicates, and sorts."""
    if not rows:
        return
    ensure_dirs()
    path = os.path.join(DATA_DIR, filename)
    new_df = pd.DataFrame(rows)

    if os.path.exists(path):
        old_df = pd.read_csv(path)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    df = df.drop_duplicates()
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    if existing_sort_cols:
        df = df.sort_values(existing_sort_cols)
    df.to_csv(path, index=False)


def run(backfill: bool):
    """Main execution logic for fetching data."""
    ensure_dirs()
    start = date.fromisoformat(START_DATE)
    today = paris_now().date()
    if backfill:
        dates = list(daterange(start, today))
    else:
        # Daily run: refresh yesterday (the "day before") AND today.
        # Yesterday's numbers are final; today's day-ahead has been
        # published by ~13:00 Paris time so it is available at 15:00.
        yesterday = today - timedelta(days=1)
        dates = [yesterday, today]

    all_dayahead, all_ida1, all_ida2, all_ida3, all_vwap_qh = [], [], [], [], []
    auction_markets = [
        ("DayAhead", all_dayahead),
        ("SIDC_IntradayAuction1", all_ida1),
        ("SIDC_IntradayAuction2", all_ida2),
        ("SIDC_IntradayAuction3", all_ida3),
    ]
    areas = [a.strip().upper() for a in AREAS.split(",") if a.strip()]

    for d in dates:
        for market, bucket in auction_markets:
            try:
                payload = fetch_prices(d, market)
                write_raw("prices", market, d, payload)
                bucket.extend(extract_auction_rows(payload))
            except Exception as e:
                print(f"Error fetching {market} for {d}: {e}")

        for area in areas:
            try:
                stats = fetch_intraday_stats(d, area)
                write_raw("intraday_stats", area, d, stats)
                all_vwap_qh.extend(extract_vwap_qh_rows(stats, area))
            except Exception as e:
                print(f"Error fetching intraday stats for {area} on {d}: {e}")

    upsert_csv(all_dayahead, "dayahead_prices.csv", ["date_cet", "area", "deliveryStartCET"])
    upsert_csv(all_ida1, "ida1_prices.csv", ["date_cet", "area", "deliveryStartCET"])
    upsert_csv(all_ida2, "ida2_prices.csv", ["date_cet", "area", "deliveryStartCET"])
    upsert_csv(all_ida3, "ida3_prices.csv", ["date_cet", "area", "deliveryStartCET"])
    upsert_csv(all_vwap_qh, "intraday_continuous_vwap_qh.csv", ["date_cet", "area", "deliveryStartCET"])


def main():
    """Entry point for the script."""
    # Configuration for scheduled run (Targeting 15:00 Paris time)
    TARGET_HOUR = int(os.environ.get("TARGET_HOUR", "15"))
    enforce_time = os.environ.get("ENFORCE_TIME_PARIS", "1") == "1"
    backfill = os.environ.get("BACKFILL", "0") == "1"

    if enforce_time and not backfill:
        current_hour = paris_now().hour
        if current_hour < TARGET_HOUR:
            print(f"Current time {paris_now().strftime('%H:%M')} is before {TARGET_HOUR}:00 Paris time — exiting")
            return
        else:
            print(f"Current time {paris_now().strftime('%H:%M')} is after {TARGET_HOUR}:00 Paris time — proceeding")

    run(backfill=backfill)
    print("Done")


if __name__ == "__main__":
    main()
