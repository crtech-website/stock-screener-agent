import logging

import numpy as np

from . import plan
from .backtest import has_edge
from .indicators import pct_change, rsi, sma
from .strategies import active_strategies, describe, explain, signals

log = logging.getLogger(__name__)

HOLD_BACK = {"losing", "lagging"}


def candidates(panel, cfg, report):
    """Today's signals from every active strategy, each with a full trade plan."""
    c = panel.close
    last = c.iloc[-1]
    atr_now = plan.atr(panel, cfg["trading"]["atr_period"]).iloc[-1]
    rsi14 = rsi(c, 14).iloc[-1]
    chg = {n: pct_change(c, n).iloc[-1] for n in (1, 5, 20)}
    dollar_vol = sma((c * panel.volume).fillna(0), 20).iloc[-1]
    mcap = panel.market_cap.iloc[-1] if panel.market_cap is not None else None
    regime_now = bool(panel.regime.iloc[-1])

    out = []
    for strat, prm in active_strategies(cfg, panel):
        entry = report["strategies"].get(strat.name, {})
        if cfg["backtest"]["require_edge"] and not (entry and has_edge(entry)):
            log.info("%s skipped: backtest shows no edge", strat.name)
            continue
        if cfg["alerts"]["skip_losing_strategies"] and entry and entry["headline"].get("verdict") in HOLD_BACK:
            log.info("%s skipped: lost money in the backtest", strat.name)
            continue
        sig, score = signals(strat, prm, panel)
        today = sig.iloc[-1]
        hits = score.iloc[-1][today[today].index].sort_values(ascending=False)
        log.info("%s: %d signals today", strat.name, len(hits))
        sma_now = sma(c, prm["target_sma"]).iloc[-1] if prm.get("target_sma") else None
        for asset, sc in hits.head(cfg["checks"]["candidates_per_strategy"]).items():
            price, a = float(last[asset]), float(atr_now[asset])
            if not (np.isfinite(a) and a > 0):
                continue
            stop, target = plan.levels(price, a, float(sma_now[asset]) if sma_now is not None else None, prm, cfg)
            # You buy at the ask, above the market price; a limit at the bare close would rarely fill.
            limit = price * (1 + cfg[panel.market].get("half_spread_pct", 0) / 100)
            out.append({
                "strategy": strat.name,
                "strategy_label": strat.label,
                "strategy_text": describe(strat.name, panel.market, prm),
                "reason": explain(strat.name, panel, prm, asset, sc),
                "hold": prm["hold"],
                "as_of": panel.last_date,
                "id": asset,
                "symbol": panel.symbols.get(asset, asset),
                "name": panel.names.get(asset, ""),
                "name_only": panel.names.get(asset, asset),
                "price": price,
                "live_price": panel.live.get(asset),
                "entry": limit,
                "stop": float(stop),
                "target": float(target),
                "position_pct": plan.position_pct(limit, float(stop), cfg),
                "atr": a,
                "score": float(sc),
                "rsi14": round(float(rsi14[asset]), 1),
                "change_1d_pct": round(float(chg[1][asset]), 1),
                "change_5d_pct": round(float(chg[5][asset]), 1),
                "change_20d_pct": round(float(chg[20][asset]), 1),
                "avg_dollar_volume": float(dollar_vol[asset]),
                "market_cap": float(mcap[asset]) if mcap is not None else None,
                "regime_on": regime_now,
                "backtest": entry.get("headline"),
            })
    return out


def pick(cands, cfg, extra=0):
    """Top picks per strategy after checks, dropping anything the LLM called structural."""
    by_strategy = {}
    for c in cands:
        if c.get("llm", {}).get("verdict") == "structural":
            continue
        by_strategy.setdefault(c["strategy"], [])
        if len(by_strategy[c["strategy"]]) < cfg["alerts"]["picks_per_strategy"] + extra:
            by_strategy[c["strategy"]].append(c)
    return by_strategy


def held_back(report, cfg):
    """Labels of strategies whose picks are withheld because they lost money in testing."""
    if not cfg["alerts"]["skip_losing_strategies"]:
        return []
    return [e["label"] for e in report["strategies"].values() if e["headline"].get("verdict") in HOLD_BACK]
