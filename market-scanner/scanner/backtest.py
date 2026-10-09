"""Backtest each strategy's full trade plan (entry, take profit, stop loss, time limit) on cached history."""

from . import plan
from .strategies import active_strategies, signals

VERDICT_TEXT = {
    "edge": "Beat the market in testing",
    "unproven": "Not proven: results could be luck",
    "losing": "Lost money in testing",
    "too_few": "Not enough past trades to judge",
}


def run(panel, cfg):
    arrays = plan.plan_arrays(panel, cfg)
    mkt = plan.market_index(panel)
    report = {"market": panel.market, "start": panel.close.index[0], "end": panel.close.index[-1],
              "days": len(panel.close), "assets": int((panel.tradable.sum() >= 30).sum()), "strategies": {}}
    for strat, prm in active_strategies(cfg, panel):
        entry = {"label": strat.label, "hold": prm["hold"], "regime": prm["regime"], "variants": {}}
        for variant, use_regime in (("no_regime", False), ("with_regime", True)):
            sig, _ = signals(strat, prm, panel, use_regime=use_regime)
            trades = plan.backtest_signals(panel, sig, prm, cfg, arrays=arrays, mkt=mkt)
            entry["variants"][variant] = plan.summarize(trades, prm["hold"], cfg)
        entry["headline"] = entry["variants"]["with_regime" if prm["regime"] else "no_regime"]
        report["strategies"][strat.name] = entry
    return report


def has_edge(entry):
    return entry["headline"].get("verdict") == "edge"


def format_report(report):
    lines = [f"{report['market'].upper()} backtest, {report['start']} to {report['end']} "
             f"({report['days']} days, {report['assets']} assets)",
             "Each trade: buy at the close, sell at the take-profit, the stop loss or the time limit,",
             "whichever comes first, after trading costs. 'vs market' = trade minus buying everything.", ""]
    for e in report["strategies"].values():
        lines.append(f"{e['label']} (max {e['hold']} days, market filter {'on' if e['regime'] else 'off'})")
        for variant, name in (("no_regime", "without market filter"), ("with_regime", "with market filter")):
            s = e["variants"][variant]
            if not s.get("n"):
                lines.append(f"  {name:22} no trades")
                continue
            lines.append(
                f"  {name:22} {s['n']:>5} trades  won {s['win_rate']:.0%}  avg {s['avg_ret']:+.2%}  "
                f"win {s['avg_win']:+.1%} / loss {s['avg_loss']:+.1%}  vs market {s['excess']:+.2%}  "
                f"t={s['t']:.1f}  [{VERDICT_TEXT[s['verdict']]}]")
            lines.append(f"  {'':22} exits: {s['pct_target']:.0%} take-profit, {s['pct_stop']:.0%} stop loss, "
                         f"{s['pct_time']:.0%} time limit")
        lines.append("")
    return "\n".join(lines)
