# Nordpool Electricity Market Analysis

Automated GitHub-based electricity market analysis using the CSV files in `data/`.

## Separate country/area outputs

Each area gets its own folder under:

- `charts/<AREA>/`
- `reports/<AREA>/`

Examples:
- `charts/FR/price_trends.png`
- `charts/FR/spreads.png`
- `charts/FR/yesterday_vs_today.png`
- `charts/FR/arbitrage_summary.png`

- `reports/FR/summary.csv`
- `reports/FR/yesterday_vs_today.csv`
- `reports/FR/anomaly_detection.csv`
- `reports/FR/arbitrage_opportunities.csv`
- `reports/FR/arbitrage_summary.csv`
- `reports/FR/daily_report.md`

## Combined outputs

- `reports/summary_by_area_market.csv`
- `reports/daily_market_averages_by_area.csv`
- `reports/yesterday_vs_today_all_areas.csv`
- `reports/anomaly_detection_all_areas.csv`
- `reports/arbitrage_opportunities_all_areas.csv`
- `reports/arbitrage_summary_all_areas.csv`
