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


def signals(strat, prm, panel, use_regime=None):
    sig, score = strat.build(panel, prm)
    sig = sig.fillna(False).astype(bool) & panel.tradable
    if prm["regime"] if use_regime is None else use_regime:
        sig = sig.mul(panel.regime, axis=0).astype(bool)
    return sig, score
