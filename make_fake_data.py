"""
make_fake_data.py - Build a fake Kalshi-like dataset so you can test the code
without internet.

How the fake world is built:
  1. Every market gets a TRUE chance t of happening.
  2. Traders over-weight small chances. Following Burgi, Deng & Whelan (2026),
     the market's belief is pulled toward 50% by a factor beta = 0.09:
         market mid = (1 - beta) * t + beta * 0.5
     Example: t = 0.20 -> mid = 0.91 * 0.20 + 0.045 = 0.227 (about 23 cents).
  3. Add a bid-ask spread of 1 to 4 cents and round to whole cents.
  4. Flip a weighted coin with chance t to decide the outcome.
  5. Also make a 'p_forecast' column: a noisy guess of t (noise = 2 points by
     default), like a good model would make, so you can try --use-csv-forecast.
     Try --noise 0.04: the forecaster becomes LESS accurate than the market
     and the backtest shows it losing money. Accuracy comes first.

Because we KNOW the bias is in this data, the backtest should find it. That is
how we check the code works before trusting it on real data.
"""

import argparse
import os

import numpy as np
import pandas as pd


def make(n_events=1500, beta=0.09, forecast_noise=0.02, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []
    for e in range(n_events):
        close = start + pd.Timedelta(hours=6 * e)
        for k in range(rng.integers(1, 4)):          # 1-3 markets per event
            t = rng.beta(0.6, 0.6)                    # many cheap and expensive contracts
            mid = (1 - beta) * t + beta * 0.5
            spread = rng.integers(1, 5) / 100
            yes_bid = np.clip(np.round(mid - spread / 2, 2), 0.01, 0.98)
            yes_ask = np.clip(np.round(mid + spread / 2, 2), yes_bid + 0.01, 0.99)
            outcome = int(rng.random() < t)
            p_forecast = float(np.clip(t + rng.normal(0, forecast_noise), 0.01, 0.99))
            rows.append({
                "ticker": f"FAKE-{e:05d}-{k}",
                "event_ticker": f"FAKE-{e:05d}",
                "series": "FAKE",
                "close_time": close.isoformat(),
                "entry_time": (close - pd.Timedelta(hours=24)).isoformat(),
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_ask": round(1 - yes_bid, 2),
                "mid": round((yes_bid + yes_ask) / 2, 3),
                "outcome": outcome,
                "candle_volume": int(rng.integers(10, 5000)),
                "p_forecast": round(p_forecast, 4),
                "true_p": round(t, 4),  # only fake data has this!
            })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make a fake Kalshi-style dataset.")
    ap.add_argument("--events", type=int, default=1500)
    ap.add_argument("--beta", type=float, default=0.09,
                    help="Longshot bias strength (0 = honest market)")
    ap.add_argument("--noise", type=float, default=0.02,
                    help="Noise in the fake p_forecast column (bigger = worse forecaster)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join("data", "fake_markets.csv"))
    args = ap.parse_args(argv)
    df = make(args.events, args.beta, args.noise, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} fake markets ({args.events} events) to {args.out}")


if __name__ == "__main__":
    main()
