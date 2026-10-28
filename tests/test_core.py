"""Run with:  python -m pytest -q"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import decide_side, raw_size, taker_fee            # noqa: E402
from forecasters import CalibrationForecaster, pool_adjacent_violators  # noqa: E402
import make_fake_data                                                # noqa: E402
import backtest                                                      # noqa: E402


# --- strategies -------------------------------------------------------------

def test_buy_yes_when_p_above_yes_ask():
    assert decide_side(0.60, yes_ask=0.52, no_ask=0.50) == ("YES", 0.52)


def test_buy_no_when_p_below_one_minus_no_ask():
    # 1 - no_ask = 0.40, and p = 0.30 is below it -> buy NO
    assert decide_side(0.30, yes_ask=0.62, no_ask=0.60) == ("NO", 0.60)


def test_no_bet_inside_the_spread():
    # no-bet zone is 1 - 0.50 = 0.50 up to 0.52
    assert decide_side(0.51, yes_ask=0.52, no_ask=0.50) == (None, None)


def test_brier_size_is_edge_against_price_paid():
    side, size, price = raw_size("brier", 0.70, yes_ask=0.60, no_ask=0.42)
    assert side == "YES" and price == 0.60
    assert size == pytest.approx(0.10)
    side, size, price = raw_size("brier", 0.20, yes_ask=0.32, no_ask=0.70)
    assert side == "NO" and size == pytest.approx(0.10)   # ref = 1 - 0.70 = 0.30


def test_bigger_edge_means_bigger_brier_bet():
    _, small, _ = raw_size("brier", 0.55, 0.50, 0.52)
    _, big, _ = raw_size("brier", 0.75, 0.50, 0.52)
    assert big > small > 0


def test_kalshi_fee_example():
    # 0.07 * 100 * 0.5 * 0.5 = 1.75 exactly
    assert taker_fee(100, 0.50) == pytest.approx(1.75)
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> rounds UP to 0.02
    assert taker_fee(1, 0.50) == pytest.approx(0.02)


# --- forecaster -------------------------------------------------------------

def test_pav_output_never_goes_down():
    out = pool_adjacent_violators([0.1, 0.3, 0.2, 0.5, 0.4], [1, 1, 1, 1, 1])
    assert np.all(np.diff(out) >= -1e-12)


def test_calibration_learns_the_longshot_bias():
    df = make_fake_data.make(n_events=3000, beta=0.09, seed=1)
    f = CalibrationForecaster().fit(df["mid"], df["outcome"])
    # A 10-cent contract should be judged LESS likely than 10%,
    # a 90-cent one MORE likely than 90%.
    assert f.predict([0.10])[0] < 0.10
    assert f.predict([0.90])[0] > 0.90


# --- end to end -------------------------------------------------------------

def test_backtest_runs_and_spends_budget(tmp_path):
    data = tmp_path / "fake.csv"
    make_fake_data.make(n_events=600, seed=3).to_csv(data, index=False)
    summary, trades = backtest.run(str(data), budget=500, out_dir=str(tmp_path / "res"),
                                   plots=False, quiet=True)
    brier_row = summary[summary["rule"] == "brier"].iloc[0]
    assert brier_row["spent"] == pytest.approx(500, rel=0.05)
    assert set(summary["rule"]) >= {"flat", "brier", "log", "kelly"}


def test_no_peeking_at_the_future(tmp_path):
    data = tmp_path / "fake.csv"
    make_fake_data.make(n_events=400, seed=4).to_csv(data, index=False)
    df = backtest.load_markets(str(data))
    train, test = backtest.time_split(df, 0.6)
    # Every test market must close after every training market.
    assert test["close_time"].min() >= train["close_time"].max()
    # And no event is split between the two sides.
    assert not set(train["event_ticker"]) & set(test["event_ticker"])
