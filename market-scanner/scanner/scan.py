import logging

from .backtest import has_edge
from .indicators import pct_change, rsi, sma
from .strategies import active_strategies, signals

log = logging.getLogger(__name__)


def candidates(panel, cfg, report):
    """Today's signals from every active strategy, best-scored first within each strategy."""
    c = panel.close
    rsi14 = rsi(c, 14).iloc[-1]
    sma200 = sma(c, 200).iloc[-1]
    chg = {n: pct_change(c, n).iloc[-1] for n in (1, 5, 20)}
    dollar_vol = sma((c * panel.volume).fillna(0), 20).iloc[-1]
    mcap = panel.market_cap.iloc[-1] if panel.market_cap is not None else None
    regime_now = bool(panel.regime.iloc[-1])

    out = []
    for strat, prm in active_strategies(cfg, panel):
        entry = report["strategies"].get(strat.name, {})
        if cfg["backtest"]["require_edge"] and not (entry and has_edge(entry, cfg)):
            log.info("%s skipped: backtest shows no edge", strat.name)
            continue
        sig, score = signals(strat, prm, panel)
        today = sig.iloc[-1]
        hits = score.iloc[-1][today[today].index].sort_values(ascending=False)
        log.info("%s: %d signals today", strat.name, len(hits))
        for asset, sc in hits.head(cfg["checks"]["candidates_per_strategy"]).items():
            price = c[asset].iloc[-1]
            out.append({
                "strategy": strat.name,
                "strategy_label": strat.label,
                "hold": prm["hold"],
                "as_of": panel.last_date,
                "id": asset,
                "symbol": panel.symbols.get(asset, asset),
                "name": panel.names.get(asset, ""),
                "name_only": panel.names.get(asset, asset),
                "price": float(price),
                "score": float(sc),
                "rsi14": round(float(rsi14[asset]), 1),
                "pct_from_sma200": round(float((price / sma200[asset] - 1) * 100), 1) if sma200[asset] == sma200[asset] else None,
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
