import json
from playwright.sync_api import sync_playwright

URLS = [
    # IDA pages (your links)
    "https://data.nordpoolgroup.com/auction/intraday-auction-1/prices?deliveryDate=latest&currency=EUR&aggregation=DeliveryPeriod&deliveryAreas=FR,GER",
    "https://data.nordpoolgroup.com/auction/intraday-auction-2/prices?deliveryDate=latest&currency=EUR&aggregation=DeliveryPeriod&deliveryAreas=FR,GER",
    "https://data.nordpoolgroup.com/auction/intraday-auction-3/prices?deliveryDate=latest&currency=EUR&aggregation=DeliveryPeriod&deliveryAreas=FR,GER",
    # Continuous stats pages (your links)
    "https://data.nordpoolgroup.com/intraday/intraday-market-statistics?deliveryDate=latest&deliveryArea=GER",
    "https://data.nordpoolgroup.com/intraday/intraday-market-statistics?deliveryDate=latest&deliveryArea=FR",
]

def main():
    hits = []

    def record_response(resp):
        try:
            url = resp.url
            status = resp.status
            ct = (resp.headers.get("content-type") or "").lower()

            # We care mainly about JSON/XHR calls
            if "application/json" in ct or "json" in ct or "/api/" in url:
                hits.append({"url": url, "status": status, "content_type": ct})
        except Exception:
            pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("response", record_response)

        for u in URLS:
            print(f"\n=== Visiting: {u}")
            page.goto(u, wait_until="networkidle", timeout=60000)

        browser.close()

    # De-duplicate by (url,status)
    uniq = {}
    for h in hits:
        uniq[(h["url"], h["status"])] = h

    hits = sorted(uniq.values(), key=lambda x: (x["status"], x["url"]))

    print("\n\n=== Candidate API calls (JSON / /api/):")
    print(json.dumps(hits, indent=2))

if __name__ == "__main__":
    main()
