"""
forecasters.py - Where your probability p comes from.

The betting rule needs YOUR probability p. This file gives two ways to get one.

1. CalibrationForecaster  (uses the favorite-longshot bias)
   - We know from Burgi, Deng & Whelan (2026) that Kalshi prices are a little
     off at the ends: cheap contracts win LESS often than their price says,
     expensive contracts win a bit MORE often.
   - So if we look at past markets, we can measure "when the price was X,
     how often did YES actually happen?" That curve is called a calibration curve.
   - Because the bias is a steady pattern, the curve from OLD markets should
     still help on NEW markets. So we fit it on old data and use it as p.
   - We force the curve to only go up (a higher price never means a lower
     chance). The method that does this is called isotonic regression, using
     the "pool adjacent violators" algorithm below.

2. Your own forecasts (a CSV column called p_forecast)
   - For example probabilities from an AI model or your own model.
   - The backtest just uses them as p.
"""

import numpy as np


def pool_adjacent_violators(y, w):
    """
    Isotonic regression: find the closest non-decreasing sequence to y.
    Idea: walk left to right; whenever a value is smaller than the one before
    it, merge the two into their weighted average and check again.
    """
    blocks = []  # each block: [mean, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            wt = w1 + w2
            blocks.append([(m1 * w1 + m2 * w2) / wt, wt, c1 + c2])
    out = []
    for mean, _, count in blocks:
        out.extend([mean] * count)
    return np.array(out)


class CalibrationForecaster:
    """Learns 'price -> real win rate' from past markets."""

    def __init__(self, n_bins=20, prior_strength=10, clip=(0.01, 0.99)):
        self.n_bins = n_bins
        # prior_strength pulls each bin toward its own price when the bin has few
        # markets, so one lucky week does not fool us (a "shrinkage" estimate).
        self.prior_strength = prior_strength
        self.clip = clip
        self.x = None
        self.y = None

    def fit(self, price, outcome):
        price = np.asarray(price, float)
        outcome = np.asarray(outcome, float)
        edges = np.linspace(0, 1, self.n_bins + 1)
        centers, rates, counts = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (price >= lo) & (price < hi) if hi < 1 else (price >= lo)
            n = mask.sum()
            if n == 0:
                continue
            avg_price = price[mask].mean()
            wins = outcome[mask].sum()
            # Shrink toward the average price: (wins + k*price) / (n + k)
            rate = (wins + self.prior_strength * avg_price) / (n + self.prior_strength)
            centers.append(avg_price)
            rates.append(rate)
            counts.append(n)
        self.x = np.array(centers)
        self.y = pool_adjacent_violators(rates, counts)
        return self

    def predict(self, price):
        if self.x is None:
            raise RuntimeError("Call fit() first.")
        p = np.interp(np.asarray(price, float), self.x, self.y)
        return np.clip(p, *self.clip)
