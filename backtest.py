"""
backtest.py - Test the betting strategies on past Kalshi markets.

How the test works, one step at a time:

  1. Sort markets by close time. We must never use the future to make a past bet.
  2. Split them: the first part is "training" (we learn from it), the last part is
     "testing" (we pretend to trade on it). Because of step 1, every test market
     closes after every training market.
  3. Get a probability p for each test market (from the calibration curve
     learned on training data, or from your own p_forecast column).
  4. For each sizing rule, decide side and size, pay the ask price and the
     taker fee, then collect $1 per winning contract.
  5. Report ROI = profit / money spent, plus a 95% range for the ROI.
     Markets in the same event move together (if one weather bucket wins, its
     neighbors lose), so the range is computed by resampling whole EVENTS.

Example:
    python backtest.py --data data/KXHIGHNY.csv
    python backtest.py --data data/fake_markets.csv --budget 1000
"""

import argparse
import os

import numpy as np
import pandas as pd

from forecasters import CalibrationForecaster
from strategies import raw_size, taker_fee
import strategies

FIXED_BUDGET_RULES = ["flat", "brier", "log"]


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def load_markets(path):
    df = pd.read_csv(path)
    needed = {"ticker", "close_time", "yes_ask", "no_ask", "outcome"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "event_ticker" not in df.columns:
        df["event_ticker"] = df["ticker"]
    if "mid" not in df.columns:
        df["mid"] = (df["yes_ask"] + (1 - df["no_ask"])) / 2
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df.sort_values("close_time").reset_index(drop=True)


def time_split(df, train_frac):
    """Split by EVENT so all markets of one event land on the same side."""
    events = df.drop_duplicates("event_ticker")["event_ticker"].tolist()
    cut = int(len(events) * train_frac)
    train_events = set(events[:cut])
    is_train = df["event_ticker"].isin(train_events)
    return df[is_train].copy(), df[~is_train].copy()


# ---------------------------------------------------------------------------
# Running one strategy
# ---------------------------------------------------------------------------

def apply_min_edge(test, min_edge):
    """
    Skip markets where our edge over the price we pay is smaller than min_edge.
    We do this by moving p back to the edge of the no-bet zone, so the
    sizing rules see zero edge there and place no bet.
    """
    if min_edge <= 0:
        return test
    test = test.copy()
    yes_edge = test["p"] - test["yes_ask"]
    no_edge = (1 - test["p"]) - test["no_ask"]
    small = (yes_edge < min_edge) & (no_edge < min_edge)
    test.loc[small, "p"] = test.loc[small, "yes_ask"]  # p == ask -> no bet
    return test


def run_fixed_budget(test, rule, budget, fees=True):
    """
    Proper betting says bet sizes should be PROPORTIONAL to the edge. So we
    compute every market's raw size first, then scale all of them by the same
    number so the total money spent equals the budget.
    """
    sides, sizes, prices = [], [], []
    for p, ya, na in zip(test["p"], test["yes_ask"], test["no_ask"]):
        side, size, price = raw_size(rule, p, ya, na)
        sides.append(side)
        sizes.append(size)
        prices.append(price if price is not None else 0.0)
    sizes = np.array(sizes)
    prices = np.array(prices)

    cost_per_unit = (sizes * prices).sum()
    if cost_per_unit == 0:
        return empty_trades(test, rule)
    scale = budget / cost_per_unit

    trades = test[["ticker", "event_ticker", "close_time", "p", "mid",
                   "yes_ask", "no_ask", "outcome"]].copy()
    trades["rule"] = rule
    trades["side"] = sides
    trades["price"] = prices
    trades["contracts"] = np.round(sizes * scale, 2)  # Kalshi allows 0.01-contract steps
    return settle(trades, fees)


def run_kelly(test, budget, fraction=1.0, fees=True):
    """
    Kelly bets a FRACTION of your current money, so it must go in time order.
    Simplification: each market settles before the next bet is placed.
    """
    bankroll = budget
    rows = []
    for _, m in test.iterrows():
        side, f_per_contract, price = raw_size("kelly", m["p"], m["yes_ask"], m["no_ask"])
        contracts = 0.0
        if side is not None and bankroll > 0:
            contracts = np.floor(fraction * f_per_contract * bankroll * 100) / 100
            contracts = min(contracts, np.floor(bankroll / price * 100) / 100)
        row = m[["ticker", "event_ticker", "close_time", "p", "mid",
                 "yes_ask", "no_ask", "outcome"]].to_dict()
        row.update(rule="kelly" if fraction == 1 else f"kelly_x{fraction:g}",
                   side=side, price=price or 0.0, contracts=contracts)
        one = settle(pd.DataFrame([row]), fees)
        bankroll += one["profit"].iloc[0]
        rows.append(one.iloc[0])
    return pd.DataFrame(rows)


def settle(trades, fees=True):
    """Work out cost, fee, payout and profit for each trade."""
    won = np.where(trades["side"] == "YES", trades["outcome"] == 1,
                   np.where(trades["side"] == "NO", trades["outcome"] == 0, False))
    trades["cost"] = trades["contracts"] * trades["price"]
    trades["fee"] = [taker_fee(c, pr) if (fees and c > 0) else 0.0
                     for c, pr in zip(trades["contracts"], trades["price"])]
    trades["payout"] = np.where(won, trades["contracts"], 0.0)
    trades["profit"] = trades["payout"] - trades["cost"] - trades["fee"]
    trades["won"] = won
    return trades


def empty_trades(test, rule):
    t = test[["ticker", "event_ticker", "close_time", "p", "mid",
              "yes_ask", "no_ask", "outcome"]].copy()
    t["rule"], t["side"], t["price"], t["contracts"] = rule, None, 0.0, 0.0
    return settle(t)


# ---------------------------------------------------------------------------
# Scoring the results
# ---------------------------------------------------------------------------

def roi(trades):
    spent = (trades["cost"] + trades["fee"]).sum()
    return trades["profit"].sum() / spent if spent > 0 else 0.0


def event_bootstrap_roi(trades, n_boot=2000, seed=0):
    """95% range for ROI, resampling whole events (see step 5 at the top)."""
    placed = trades[trades["contracts"] > 0]
    if placed.empty:
        return (0.0, 0.0)
    by_event = placed.groupby("event_ticker").agg(
        profit=("profit", "sum"), spent=("cost", "sum"), fee=("fee", "sum"))
    profit = by_event["profit"].to_numpy()
    spent = (by_event["spent"] + by_event["fee"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(profit), size=(n_boot, len(profit)))
    rois = profit[idx].sum(1) / spent[idx].sum(1)
    return tuple(np.percentile(rois, [2.5, 97.5]))


def summarize(trades, budget):
    placed = trades[trades["contracts"] > 0]
    lo, hi = event_bootstrap_roi(trades)
    return {
        "rule": trades["rule"].iloc[0],
        "bets": len(placed),
        "spent": round((placed["cost"] + placed["fee"]).sum(), 2),
        "fees": round(placed["fee"].sum(), 2),
        "profit": round(placed["profit"].sum(), 2),
        "roi_%": round(100 * roi(placed), 2),
        "roi_95%_low": round(100 * lo, 2),
        "roi_95%_high": round(100 * hi, 2),
        "win_rate_%": round(100 * placed["won"].mean(), 1) if len(placed) else 0.0,
        "end_money": round(budget + placed["profit"].sum(), 2),
    }


def brier(p, outcome):
    return float(np.mean((np.asarray(p) - np.asarray(outcome)) ** 2))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(train, forecaster, all_trades, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: calibration curve. A perfectly honest market sits on the dashed line.
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, 1, 21)
    which = np.digitize(train["mid"], bins) - 1
    xs, ys, ns = [], [], []
    for b in range(20):
        sel = which == b
        if sel.sum() >= 5:
            xs.append(train["mid"][sel].mean())
            ys.append(train["outcome"][sel].mean())
            ns.append(sel.sum())
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Honest price (price = win rate)")
    ax.scatter(xs, ys, s=np.sqrt(ns) * 6, alpha=0.6, label="Training markets (bins)")
    if forecaster is not None:
        grid = np.linspace(0.01, 0.99, 99)
        ax.plot(grid, forecaster.predict(grid), color="C3", label="Learned curve (our p)")
    ax.set_xlabel("Market price (mid)")
    ax.set_ylabel("How often YES actually happened")
    ax.set_title("Is the market price honest?")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration.png"), dpi=150)
    plt.close(fig)

    # Plot 2: cumulative profit as a percent of money spent, for each rule.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rule, t in all_trades.items():
        t = t[t["contracts"] > 0].sort_values("close_time")
        if t.empty:
            continue
        spent = (t["cost"] + t["fee"]).sum()
        ax.plot(t["close_time"], 100 * t["profit"].cumsum() / spent, label=rule)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Cumulative profit (% of total spent)")
    ax.set_title("Test period: profit over time")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_profit.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data, budget=1000.0, train_frac=0.6, fees=True, fee_rate=0.07,
        use_csv_forecast=False, out_dir="results", plots=True, quiet=False, min_edge=0.0):
    strategies.TAKER_FEE_RATE = fee_rate
    df = load_markets(data)
    train, test = time_split(df, train_frac)
    if len(train) < 50 or len(test) < 20:
        raise ValueError(f"Not enough data: {len(train)} train / {len(test)} test markets.")

    forecaster = None
    if use_csv_forecast:
        if "p_forecast" not in df.columns:
            raise ValueError("--use-csv-forecast needs a 'p_forecast' column in the CSV.")
        test["p"] = test["p_forecast"].clip(0.01, 0.99)
    else:
        forecaster = CalibrationForecaster().fit(train["mid"], train["outcome"])
        test["p"] = forecaster.predict(test["mid"])

    test = apply_min_edge(test, min_edge)
    all_trades = {r: run_fixed_budget(test, r, budget, fees) for r in FIXED_BUDGET_RULES}
    all_trades["kelly"] = run_kelly(test, budget, 1.0, fees)
    all_trades["kelly_x0.25"] = run_kelly(test, budget, 0.25, fees)

    summary = pd.DataFrame([summarize(t, budget) for t in all_trades.values()])

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.concat(all_trades.values()).to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    if plots:
        make_plots(train, forecaster, all_trades, out_dir)

    if not quiet:
        print(f"\nData: {data}")
        print(f"Train: {len(train)} markets   Test: {len(test)} markets   "
              f"Fees: {'on' if fees else 'off'}")
        print(f"Brier score on test (lower = better):  market {brier(test['mid'], test['outcome']):.4f}"
              f"   our p {brier(test['p'], test['outcome']):.4f}")
        print("\n" + summary.to_string(index=False))
        print(f"\nSaved results to {out_dir}/")
    return summary, all_trades


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest proper betting on past Kalshi markets.")
    ap.add_argument("--data", required=True, help="CSV from kalshi_data.py or make_fake_data.py")
    ap.add_argument("--budget", type=float, default=1000.0, help="Dollars to spend in the test period")
    ap.add_argument("--train-frac", type=float, default=0.6, help="Share of events used for training")
    ap.add_argument("--no-fees", action="store_true", help="Turn off Kalshi taker fees")
    ap.add_argument("--fee-rate", type=float, default=0.07, help="Taker fee rate (default 0.07)")
    ap.add_argument("--use-csv-forecast", action="store_true",
                    help="Use your own 'p_forecast' column instead of the calibration curve")
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="Only bet when p beats the price you pay by at least this much (e.g. 0.02)")
    ap.add_argument("--out", default="results", help="Folder for results and plots")
    args = ap.parse_args(argv)
    run(args.data, args.budget, args.train_frac, not args.no_fees, args.fee_rate,
        args.use_csv_forecast, args.out, min_edge=args.min_edge)


if __name__ == "__main__":
    main()
