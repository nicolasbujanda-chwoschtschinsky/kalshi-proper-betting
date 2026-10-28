"""
strategies.py - How many contracts to buy, given your probability and the prices.

Build-up of the idea (from Gu, Kagan, Sun, Wu & Xu 2026, arXiv:2607.06166):

  1. A YES contract pays $1 if the event happens. So if you think the chance is
     p and you can buy YES at price a, your expected profit per contract is
     p - a. That is only positive when p > a.

  2. Buying NO at price b pays $1 if the event does NOT happen. Your expected
     profit per NO contract is (1 - p) - b. That is only positive when
     p < 1 - b.

  3. Because of 1 and 2, there is a "do nothing" zone: 1 - b <= p <= a.
     Inside it, the spread eats any edge you think you have.

  4. Outside that zone, the paper proves the safe way to size the bet is the
     "proper betting" rule: shares = grad G(p) - grad G(reference price),
     where G comes from a scoring rule.
       - Brier rule:  shares are proportional to (p - reference)
       - Log rule:    shares are proportional to (logit(p) - logit(reference))
     The reference price is the YES ask when buying YES and (1 - NO ask) when
     buying NO, so you always measure your edge against the price you really pay.

  5. Kelly is included only so you can see it lose. Kelly assumes your p is
     exactly right; real forecasts are not, so Kelly bets too big.
"""

import math

TAKER_FEE_RATE = 0.07  # Kalshi general taker fee: 0.07 * contracts * P * (1 - P)


def logit(x):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def decide_side(p, yes_ask, no_ask):
    """
    Return ("YES", price), ("NO", price) or (None, None).
    Uses rules 1-3 above.
    """
    if p > yes_ask:
        return "YES", yes_ask
    if p < 1 - no_ask:
        return "NO", no_ask
    return None, None


def raw_size(rule, p, yes_ask, no_ask):
    """
    Un-scaled number of contracts for one market under a sizing rule.
    Always >= 0. The side comes from decide_side().
    """
    side, price = decide_side(p, yes_ask, no_ask)
    if side is None:
        return None, 0.0, None

    # Reference price measured on the YES scale (rule 4).
    ref = yes_ask if side == "YES" else 1 - no_ask

    if rule == "flat":
        size = 1.0                              # same bet everywhere
    elif rule == "brier":
        size = abs(p - ref)                     # grows in a straight line with edge
    elif rule == "log":
        size = abs(logit(p) - logit(ref))       # grows slower for huge disagreements
    elif rule == "kelly":
        # Kelly fraction of bankroll: (win prob - price) / (1 - price), on the side we buy.
        win_prob = p if side == "YES" else 1 - p
        size = (win_prob - price) / (1 - price) / price  # fraction -> contracts per $1 bankroll
    else:
        raise ValueError(f"Unknown rule: {rule}")
    return side, size, price


def taker_fee(contracts, price):
    """
    Kalshi general taker fee, rounded UP to the next cent.
    Example: 100 contracts at 50 cents -> 0.07 * 100 * 0.5 * 0.5 = $1.75
    """
    fee = TAKER_FEE_RATE * contracts * price * (1 - price)
    return math.ceil(round(fee * 100, 6)) / 100
