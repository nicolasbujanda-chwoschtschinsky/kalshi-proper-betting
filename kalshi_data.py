"""
kalshi_data.py - Download past (settled) Kalshi markets into a CSV file.

The big idea:
    To test a strategy on the past, we need three things for every market:
      1. the price you could have bought at (the YES ask and the NO ask),
      2. the time you would have bought it,
      3. how the market ended (YES = 1, NO = 0).
    Kalshi's public API gives all three for free, with no account or API key.

Kalshi keeps recent markets in its "live" database and moves older ones to a
"historical" database. So this script asks both places and combines them.

Example (run this on your own computer):
    python kalshi_data.py --series KXHIGHNY --hours-before 24 --out data/KXHIGHNY.csv
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
PAUSE_SECONDS = 0.15  # small pause between requests so we stay under rate limits


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def get_json(path, params=None, retries=4):
    """GET a Kalshi endpoint and return the JSON. Retries if we get rate-limited."""
    url = BASE_URL + path
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:  # "too many requests" -> wait and try again
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        time.sleep(PAUSE_SECONDS)
        return resp.json()
    raise RuntimeError(f"Kept getting rate-limited on {url}")


def to_dollars(value):
    """
    Kalshi prices come back as dollar strings like "0.5600".
    Older responses used whole cents like 56. This turns either into 0.56.
    Returns None if there is no price.
    """
    if value is None or value == "":
        return None
    x = float(value)
    if x > 1.0:  # must be cents
        x = x / 100.0
    return x


def read_close(ohlc):
    """Pull the closing price out of one OHLC block (works for old and new field names)."""
    if not ohlc:
        return None
    for key in ("close", "close_dollars"):
        if key in ohlc:
            return to_dollars(ohlc[key])
    return None


def parse_time(text):
    """Turn Kalshi's '2026-03-01T15:00:00Z' into a Unix timestamp (seconds)."""
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


# ---------------------------------------------------------------------------
# Step 1: list every settled market in a series
# ---------------------------------------------------------------------------

def list_markets(series_ticker):
    """
    Return a list of (market_dict, where) pairs.
    'where' is "live" or "historical" so we know which candlestick endpoint to use.
    """
    found = {}

    # Live database: settled markets that have not been archived yet.
    cursor = ""
    while True:
        data = get_json("/markets", {"series_ticker": series_ticker,
                                     "status": "settled", "limit": 1000,
                                     "cursor": cursor})
        if not data:
            break
        for m in data.get("markets", []):
            found[m["ticker"]] = (m, "live")
        cursor = data.get("cursor") or ""
        if not cursor:
            break

    # Historical database: older settled markets.
    cursor = ""
    while True:
        data = get_json("/historical/markets", {"series_ticker": series_ticker,
                                                "limit": 1000, "cursor": cursor})
        if not data:
            break
        for m in data.get("markets", []):
            found.setdefault(m["ticker"], (m, "historical"))
        cursor = data.get("cursor") or ""
        if not cursor:
            break

    return list(found.values())


# ---------------------------------------------------------------------------
# Step 2: find the price at our chosen entry time
# ---------------------------------------------------------------------------

def price_at(market, where, series_ticker, entry_ts):
    """
    Look at the hourly candles that end just before entry_ts and return the
    last one that had both a bid and an ask. That is "the price you could
    have traded at" when you decided to bet.
    """
    ticker = market["ticker"]
    params = {"start_ts": entry_ts - 6 * 3600, "end_ts": entry_ts,
              "period_interval": 60}
    if where == "historical":
        data = get_json(f"/historical/markets/{ticker}/candlesticks", params)
    else:
        data = get_json(f"/series/{series_ticker}/markets/{ticker}/candlesticks", params)
    if not data:
        return None

    for candle in reversed(data.get("candlesticks", [])):
        yes_bid = read_close(candle.get("yes_bid"))
        yes_ask = read_close(candle.get("yes_ask"))
        # A real two-sided market needs 0 < bid < ask < 1.
        if yes_bid and yes_ask and 0 < yes_bid < yes_ask < 1:
            volume = candle.get("volume") or candle.get("volume_fp") or 0
            return {"yes_bid": yes_bid, "yes_ask": yes_ask,
                    "candle_ts": candle["end_period_ts"], "candle_volume": float(volume)}
    return None


# ---------------------------------------------------------------------------
# Step 3: put it all together
# ---------------------------------------------------------------------------

FIELDS = ["ticker", "event_ticker", "series", "close_time", "entry_time",
          "yes_bid", "yes_ask", "no_ask", "mid", "outcome", "candle_volume"]


def download(series_ticker, hours_before, out_path, max_markets=None):
    markets = list_markets(series_ticker)
    print(f"Found {len(markets)} settled markets in {series_ticker}.")

    rows, skipped = [], 0
    for i, (m, where) in enumerate(markets):
        if max_markets and i >= max_markets:
            break
        # Only plain YES/NO markets that actually settled.
        if m.get("market_type", "binary") != "binary" or m.get("result") not in ("yes", "no"):
            skipped += 1
            continue

        close_ts = parse_time(m["close_time"])
        entry_ts = close_ts - int(hours_before * 3600)
        quote = price_at(m, where, series_ticker, entry_ts)
        if quote is None:
            skipped += 1
            continue

        yes_bid, yes_ask = quote["yes_bid"], quote["yes_ask"]
        rows.append({
            "ticker": m["ticker"],
            "event_ticker": m["event_ticker"],
            "series": series_ticker,
            "close_time": m["close_time"],
            "entry_time": datetime.fromtimestamp(quote["candle_ts"], timezone.utc).isoformat(),
            "yes_bid": round(yes_bid, 4),
            "yes_ask": round(yes_ask, 4),
            # Buying NO is the same as selling YES, so the NO ask = 1 - YES bid.
            "no_ask": round(1 - yes_bid, 4),
            "mid": round((yes_bid + yes_ask) / 2, 4),
            "outcome": 1 if m["result"] == "yes" else 0,
            "candle_volume": quote["candle_volume"],
        })
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1} markets checked, {len(rows)} usable")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} markets to {out_path} (skipped {skipped} with no usable price).")


def main(argv=None):
    p = argparse.ArgumentParser(description="Download settled Kalshi markets for backtesting.")
    p.add_argument("--series", required=True,
                   help="Kalshi series ticker, e.g. KXHIGHNY (NYC high temp) or KXFED.")
    p.add_argument("--hours-before", type=float, default=24,
                   help="How many hours before close we pretend to place the bet (default 24).")
    p.add_argument("--out", default=None, help="Output CSV path (default data/<SERIES>.csv).")
    p.add_argument("--max-markets", type=int, default=None,
                   help="Stop after this many markets (handy for a quick test).")
    args = p.parse_args(argv)
    out = args.out or os.path.join("data", f"{args.series}.csv")
    try:
        download(args.series, args.hours_before, out, args.max_markets)
    except requests.RequestException as e:
        sys.exit(f"Could not reach Kalshi: {e}")


if __name__ == "__main__":
    main()
