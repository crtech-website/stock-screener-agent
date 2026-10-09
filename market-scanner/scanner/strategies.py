"""Strategy library.

Each builder returns two dates x assets frames: a boolean signal and a ranking score.
The same frames drive the live scan (last row) and the backtest (every row), so what
gets alerted is exactly what was tested.

Sources for the rules:
- connors_rsi2: Larry Connors & Cesar Alvarez, "Short Term Trading Strategies That Work" (2008).
- ibs_reversion: internal bar strength mean reversion, as studied for US equities and ETFs
  (e.g. Pagonidis, "The IBS Effect", 2013).
- trend_breakout: Mark Minervini's trend template plus a 52-week-high breakout for stocks;
  for crypto the Turtle-style 55-day channel breakout (Richard Dennis' System 2).
- momentum: cross-sectional 12-1 month momentum (Jegadeesh & Titman 1993); for crypto the
  shorter-horizon momentum documented by Liu & Tsyvinski (2021).
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .indicators import ibs, pct_change, rsi, sma


@dataclass
class Strategy:
    name: str
    label: str
    build: Callable
    min_bars: Callable          # params -> bars of history needed before the first signal
    needs_high_low: bool = False


def oversold_bounce(p, prm):
    r = rsi(p.close, 14)
    drop = pct_change(p.close, prm["drop_bars"])
    sig = (r < prm["rsi_below"]) & (drop <= -prm["drop_pct"])
    return sig, (prm["rsi_below"] - r) + (-drop)


def connors_rsi2(p, prm):
    r2 = rsi(p.close, 2)
    sig = (p.close > sma(p.close, 200)) & (r2 < prm["rsi2_below"])
    return sig, prm["rsi2_below"] - r2


def ibs_reversion(p, prm):
    v = ibs(p.high, p.low, p.close)
    sig = (v < prm["ibs_below"]) & (p.close > sma(p.close, 200)) & (p.close < p.close.shift(1))
    return sig, prm["ibs_below"] - v


def trend_breakout(p, prm):
    c = p.close
    fast, mid, slow = sma(c, prm["fast"]), sma(c, prm["mid"]), sma(c, prm["slow"])
    template = (c > fast) & (fast > mid) & (mid > slow) & (slow > slow.shift(20))
    high = c.rolling(prm["high_window"], min_periods=prm["high_window"]).max()
    # First close at a new high, so a stock sitting at highs for weeks fires once, not daily.
    fresh_high = (c >= high) & (c.shift(1) < high.shift(1))
    vol_ratio = p.volume / sma(p.volume, 50)
    sig = template & fresh_high & (vol_ratio >= prm["volume_mult"])
    return sig, vol_ratio


def momentum(p, prm):
    c = p.close
    mom = c.shift(prm["skip"]) / c.shift(prm["lookback"]) - 1
    pct_rank = mom.where(p.tradable).rank(axis=1, pct=True)
    dates = pd.to_datetime(c.index)
    if prm["rebalance"] == "weekly":
        period = dates.isocalendar().week.values
    else:
        period = dates.month.values
    first_of_period = pd.Series(period, index=c.index).diff().fillna(0).ne(0)
    sig = (pct_rank >= 1 - prm["top_pct"]) & (c > sma(c, 50))
    sig = sig.mul(first_of_period, axis=0).astype(bool)
    return sig, pct_rank


REGISTRY = {
    s.name: s for s in [
        Strategy("oversold_bounce", "Oversold bounce", oversold_bounce, lambda prm: 30),
        Strategy("connors_rsi2", "Connors RSI(2) pullback", connors_rsi2, lambda prm: 210),
        Strategy("ibs_reversion", "IBS reversion", ibs_reversion, lambda prm: 210, needs_high_low=True),
        Strategy("trend_breakout", "Trend breakout", trend_breakout,
                 lambda prm: max(prm["slow"] + 20, prm["high_window"])),
        Strategy("momentum", "Momentum leaders", momentum, lambda prm: prm["lookback"] + 1),
    ]
}


def params_for(cfg, name, market):
    raw = cfg["strategies"][name]
    base = {k: v for k, v in raw.items() if k not in ("stocks", "crypto")}
    return {**base, **raw.get(market, {})}


def active_strategies(cfg, panel):
    """Strategies enabled for this market that have enough history to run."""
    out = []
    for name, strat in REGISTRY.items():
        prm = params_for(cfg, name, panel.market)
        if not prm.get("enabled") or panel.market not in prm["markets"]:
            continue
        if strat.needs_high_low and panel.high is None:
            continue
        if len(panel.close) < strat.min_bars(prm):
            continue
        out.append((strat, prm))
    return out


def loading(cfg, panel):
    """Labels of strategies enabled for this market that don't have enough history yet."""
    out = []
    for name, strat in REGISTRY.items():
        prm = params_for(cfg, name, panel.market)
        if not prm.get("enabled") or panel.market not in prm["markets"]:
            continue
        if strat.needs_high_low and panel.high is None:
            continue
        need = strat.min_bars(prm)
        if len(panel.close) < need:
            out.append(f"{strat.label} (has {len(panel.close)} of {need} days)")
    return out


def signals(strat, prm, panel, use_regime=None):
    sig, score = strat.build(panel, prm)
    sig = sig.fillna(False).astype(bool) & panel.tradable
    if prm["regime"] if use_regime is None else use_regime:
        sig = sig.mul(panel.regime, axis=0).astype(bool)
    return sig, score


def _days(n):
    return {252: "52 weeks (1 year)", 55: "55 days", 90: "90 days", 21: "month", 7: "week"}.get(n, f"{n} days")


def describe(name, market, prm):
    """What the strategy does, for someone who has never traded."""
    thing = "stock" if market == "stocks" else "coin"
    if name == "oversold_bounce":
        return (f"Buys a {thing} right after a sharp, fast drop, betting on a quick rebound. "
                "This style usually wins more often than it loses, but each win is small.")
    if name == "connors_rsi2":
        return (f"Buys a short dip in a {thing} that has been rising for months, betting the dip is temporary "
                "and the price bounces back within days. From trader Larry Connors.")
    if name == "ibs_reversion":
        return ("Buys a stock that is in an uptrend but closed near the low of the day, betting on a "
                "bounce over the next few days.")
    if name == "trend_breakout":
        return (f"Buys a {thing} that is already rising and just hit its highest price in {_days(prm['high_window'])} "
                "with heavy buying. The bet is that strength keeps going. It wins less often, "
                "but the wins are meant to be about twice the size of the losses.")
    if name == "momentum":
        period = "year" if prm["lookback"] >= 250 else f"{prm['lookback']} days"
        return (f"Buys the biggest winners of the past {period}. Things that have been going up "
                f"tend to keep going up for a while. The list is refreshed every {_days(prm['hold'])}.")
    return ""


def explain(name, panel, prm, asset, score):
    """Why this asset was picked today, in plain words."""
    c = panel.close[asset]
    if name == "oversold_bounce":
        drop = (c.iloc[-1] / c.iloc[-1 - prm["drop_bars"]] - 1) * 100
        return (f"Fell {abs(drop):.0f}% in the last {prm['drop_bars']} days. Its RSI is "
                f"{rsi(c, 14).iloc[-1]:.0f} out of 100; under 30 means it has been sold very hard.")
    if name == "connors_rsi2":
        move = (c.iloc[-1] / c.iloc[-4] - 1) * 100
        r2 = rsi(c, 2).iloc[-1]
        return (f"Still in a long-term uptrend (above its 200-day average), but "
                f"{'dropped' if move < 0 else 'moved'} {abs(move):.1f}% over the last 3 days. "
                f"Its short-term RSI is {'under 1' if r2 < 1 else f'{r2:.0f}'} out of 100, which is extremely low.")
    if name == "ibs_reversion":
        v = ibs(panel.high[asset], panel.low[asset], c).iloc[-1]
        return (f"In an uptrend, but closed in the bottom {v * 100:.0f}% of today's price range, "
                "a sign of a short-term overreaction.")
    if name == "trend_breakout":
        return (f"Hit its highest price in {_days(prm['high_window'])} today, on {score:.1f}x its normal "
                "trading volume, and its moving averages all point up.")
    if name == "momentum":
        ret = (c.iloc[-1 - prm["skip"]] / c.iloc[-1 - prm["lookback"]] - 1) * 100
        return (f"One of the strongest performers: up {ret:.0f}% over the past {prm['lookback']} days "
                f"(not counting the last {prm['skip']}), and still above its 50-day average.")
    return ""
