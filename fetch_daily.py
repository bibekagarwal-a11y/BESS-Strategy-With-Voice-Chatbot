"""Daily data fetcher for Nord Pool BESS bot.

Run once per day (at ~15:00 Europe/Paris) to fetch prices and intraday
stats for YESTERDAY and TODAY. Idempotent — existing rows are upserted,
not duplicated.

Usage:
    python fetch_daily.py           # fetch yesterday + today
    python fetch_daily.py --days 5  # fetch the last 5 days (for gap-fill)
"""
import argparse
from datetime import timedelta

import nordpool_bot as bot


def fetch_range(start_date, end_date):
    """Fetch all markets/areas for every day in [start_date, end_date]."""
    bot.ensure_dirs()
    dates = list(bot.daterange(start_date, end_date))
    print(f"Fetching {len(dates)} day(s): {start_date} → {end_date}")

    all_dayahead, all_ida1, all_ida2, all_ida3, all_vwap_qh = [], [], [], [], []
    auction_markets = [
        ("DayAhead", all_dayahead),
        ("SIDC_IntradayAuction1", all_ida1),
        ("SIDC_IntradayAuction2", all_ida2),
        ("SIDC_IntradayAuction3", all_ida3),
    ]
    areas = [a.strip().upper() for a in bot.AREAS.split(",") if a.strip()]

    for d in dates:
        for market, bucket in auction_markets:
            try:
                payload = bot.fetch_prices(d, market)
                bot.write_raw("prices", market, d, payload)
                bucket.extend(bot.extract_auction_rows(payload))
            except Exception as e:
                print(f"  {market} {d}: {e}")
        for area in areas:
            try:
                stats = bot.fetch_intraday_stats(d, area)
                bot.write_raw("intraday_stats", area, d, stats)
                all_vwap_qh.extend(bot.extract_vwap_qh_rows(stats, area))
            except Exception as e:
                print(f"  intraday {area} {d}: {e}")

    bot.upsert_csv(all_dayahead, "dayahead_prices.csv",
                   ["date_cet", "area", "deliveryStartCET"])
    bot.upsert_csv(all_ida1, "ida1_prices.csv",
                   ["date_cet", "area", "deliveryStartCET"])
    bot.upsert_csv(all_ida2, "ida2_prices.csv",
                   ["date_cet", "area", "deliveryStartCET"])
    bot.upsert_csv(all_ida3, "ida3_prices.csv",
                   ["date_cet", "area", "deliveryStartCET"])
    bot.upsert_csv(all_vwap_qh, "intraday_continuous_vwap_qh.csv",
                   ["date_cet", "area", "deliveryStartCET"])
    print("Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=2,
                        help="How many days back from today to fetch (default 2 "
                             "= yesterday + today).")
    parser.add_argument("--start", type=str, default=None,
                        help="Optional explicit start date (YYYY-MM-DD). "
                             "Overrides --days.")
    parser.add_argument("--end", type=str, default=None,
                        help="Optional explicit end date (YYYY-MM-DD). "
                             "Defaults to today (Europe/Paris).")
    args = parser.parse_args()

    today = bot.paris_now().date()
    end = bot.date.fromisoformat(args.end) if args.end else today
    if args.start:
        start = bot.date.fromisoformat(args.start)
    else:
        start = end - timedelta(days=max(0, args.days - 1))

    fetch_range(start, end)


if __name__ == "__main__":
    main()
