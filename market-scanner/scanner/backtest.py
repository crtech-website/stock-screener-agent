"""Event-study backtest: what happened after every past signal, net of costs, versus the market.

This answers "did buying these signals beat buying everything?" It does not simulate a
portfolio, position sizing or exits other than a fixed holding period.
"""

import math

import numpy as np
import pandas as pd

from .strategies import active_strategies, signals


def forward_returns(close, h):
    return close.shift(-h) / close - 1


def event_stats(panel, sig, h, cost):
    fwd = forward_returns(panel.close, h)
    # Equal-weight average of every liquid asset that day: the "just buy the market" baseline.
    bench = fwd.where(panel.tradable).mean(axis=1)
    trade = fwd.where(sig) - cost
    pooled = trade.stack().dropna()
    if pooled.empty:
        return {"n": 0}
    # Bad prints and unadjusted corporate actions produce absurd returns; trim the tails.
    lo, hi = pooled.quantile([0.005, 0.995])
    trade = trade.clip(lower=lo, upper=hi, axis=None)
    excess = trade.sub(bench, axis=0)

    per_date = excess.mean(axis=1).dropna()
    # Signals within the same h-day block share most of their return window, so the
    # t-stat is computed over block averages rather than over overlapping trades.
    pos = pd.Series(np.arange(len(panel.close)), index=panel.close.index).reindex(per_date.index)
    blocks = per_date.groupby(pos // h).mean()
    t = blocks.mean() / (blocks.std(ddof=1) / math.sqrt(len(blocks))) if len(blocks) > 2 else float("nan")

    trades = trade.stack().dropna()
    return {
        "n": int(len(trades)),
        "days": int(len(per_date)),
        "avg_ret": float(trades.mean()),
        "median_ret": float(trades.median()),
        "win_rate": float((trades > 0).mean()),
        "avg_excess": float(excess.stack().dropna().mean()),
        "t_excess": float(t),
    }


def run(panel, cfg):
    cost = cfg[panel.market]["cost_bps"] / 10_000
    report = {"market": panel.market, "start": panel.close.index[0], "end": panel.close.index[-1],
              "assets": int((panel.tradable.sum() >= 30).sum()), "strategies": {}}
    for strat, prm in active_strategies(cfg, panel):
        entry = {"label": strat.label, "hold": prm["hold"], "regime": prm["regime"], "variants": {}}
        for variant, use_regime in (("as_configured", None), ("no_regime", False), ("with_regime", True)):
            sig, _ = signals(strat, prm, panel, use_regime=use_regime)
            entry["variants"][variant] = {h: event_stats(panel, sig, h, cost)
                                          for h in sorted(set(cfg["backtest"]["horizons"]) | {prm["hold"]})}
        entry["headline"] = entry["variants"]["as_configured"][prm["hold"]]
        report["strategies"][strat.name] = entry
    return report


def has_edge(entry, cfg):
    h = entry["headline"]
    return h.get("n", 0) > 0 and h["avg_excess"] > 0 and h["t_excess"] >= cfg["backtest"]["min_t_stat"]


def format_report(report):
    lines = [f"{report['market'].upper()} backtest {report['start']} to {report['end']}, "
             f"{report['assets']} assets with 30+ liquid days",
             "Returns are net of costs. Excess = trade minus equal-weight market on the same days.", ""]
    for name, e in report["strategies"].items():
        lines.append(f"{e['label']} (hold {e['hold']} bars, regime filter {'on' if e['regime'] else 'off'})")
        for variant in ("no_regime", "with_regime"):
            s = e["variants"][variant][e["hold"]]
            if not s.get("n"):
                lines.append(f"  {variant:12} no signals")
                continue
            lines.append(f"  {variant:12} n={s['n']:<6} avg {s['avg_ret']:+.2%}  median {s['median_ret']:+.2%}  "
                         f"win {s['win_rate']:.0%}  excess {s['avg_excess']:+.2%}  t={s['t_excess']:.1f}")
        lines.append("")
    return "\n".join(lines)
