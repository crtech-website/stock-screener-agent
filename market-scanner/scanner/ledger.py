"""Forward tracking: every alerted pick is logged with its plan, then graded once the plan has played out.

The backtest says how a strategy did historically. The ledger says how the alerts you
actually received did, which is the number that matters.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import plan
from .strategies import REGISTRY

LABELS = {k: v.label for k, v in REGISTRY.items()}

COLUMNS = ["as_of", "logged_utc", "strategy", "id", "symbol", "entry", "stop", "target", "hold",
           "llm_verdict", "n_flags"]


def ledger_file(market):
    return Path(os.environ.get("SCANNER_LEDGER", "ledger")) / f"{market}.csv"


def append(market, picks_by_strategy):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows = [{"as_of": p["as_of"], "logged_utc": now, "strategy": p["strategy"], "id": p["id"],
             "symbol": p["symbol"], "entry": p["entry"], "stop": p["stop"], "target": p["target"],
             "hold": p["hold"], "llm_verdict": p.get("llm", {}).get("verdict", ""),
             "n_flags": len(p.get("flags", []))}
            for picks in picks_by_strategy.values() for p in picks]
    if not rows:
        return
    f = ledger_file(market)
    f.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(f, dtype={"as_of": str}) if f.exists() else pd.DataFrame(columns=COLUMNS)
    merged = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    # A re-run on the same day shouldn't count the same pick twice.
    merged.drop_duplicates(["as_of", "strategy", "id"], keep="first").to_csv(f, index=False)


def grade(market, panel, cfg):
    """Play out every logged pick with the stop, target and time limit it was sent with."""
    f = ledger_file(market)
    if not f.exists():
        return pd.DataFrame(), 0
    led = pd.read_csv(f, dtype={"as_of": str})
    # Rows from before trade plans existed have no stop/target and can't be graded.
    led = led.dropna(subset=[c for c in ("stop", "target", "hold") if c in led]) if "stop" in led else led.iloc[0:0]
    pos = {d: i for i, d in enumerate(panel.close.index)}
    col = {a: j for j, a in enumerate(panel.close.columns)}
    led = led[led["as_of"].isin(pos) & led["id"].isin(col)]
    c, h, l, _ = plan.plan_arrays(panel, cfg)
    cost = cfg[market]["cost_bps"] / 10_000
    graded = []
    for hold, g in led.groupby("hold"):
        ti = g["as_of"].map(pos).to_numpy()
        aj = g["id"].map(col).to_numpy()
        t = plan.simulate(c, h, l, ti, aj, g["stop"].to_numpy(float), g["target"].to_numpy(float), int(hold), cost)
        if t.empty:
            continue
        t["mkt"] = plan.market_returns(panel, t["ti"].to_numpy(), t["exit_day"].to_numpy())
        key = dict(zip(zip(ti, aj), g["strategy"]))
        t["strategy"] = [key[(a, b)] for a, b in zip(t["ti"], t["aj"])]
        graded.append(t)
    total = int(pd.read_csv(f).shape[0])
    return (pd.concat(graded, ignore_index=True) if graded else pd.DataFrame()), total


def scorecard(market, panel, cfg):
    graded, total = grade(market, panel, cfg)
    title = f"WEEKLY SCORECARD: {'stocks' if market == 'stocks' else 'crypto'} alerts you received"
    if total == 0:
        return f"{title}\nNo alerts logged yet."
    if graded.empty:
        return f"{title}\n{total} alerts logged. None has finished its plan yet."
    lines = [title, f"{len(graded)} of {total} alerts have finished (hit the take profit, the stop loss, "
             "or the time limit).", ""]
    for name, g in graded.groupby("strategy"):
        wins = int((g["ret"] > 0).sum())
        lines.append(f"{LABELS.get(name, name)}: "
                     f"{len(g)} trades, {wins} won, {len(g) - wins} lost. "
                     f"Average {g['ret'].mean():+.1%} per trade; the whole market did {g['mkt'].mean():+.1%} "
                     f"over the same days.")
        lines.append(f"  Exits: {(g['exit_reason'] == 'target').sum()} take profit, "
                     f"{(g['exit_reason'] == 'stop').sum()} stop loss, {(g['exit_reason'] == 'time').sum()} time limit.")
    lines.append("")
    lines.append("Results assume you followed each plan exactly, after trading costs.")
    return "\n".join(lines)


def write_latest(market, picks_by_strategy, report, when, cfg, regime_on, held_back=(), loading=()):
    """The latest scan's full report in machine-readable form, read by the Claude agent that checks Robinhood.

    It carries everything the Telegram alert shows, so the agent can give the complete
    plain-English report together with live Robinhood prices.
    """
    import json

    from .alerts import close_time, ny_time

    def pct(x):
        return None if x is None else round(x * 100, 2)

    strategies = {}
    for name, e in report["strategies"].items():
        h = e["headline"]
        strategies[name] = {"label": e["label"], "hold_days": e["hold"], "market_filter": e["regime"],
                            "verdict": h.get("verdict"), "past_trades": h.get("n", 0),
                            "win_rate_pct": pct(h.get("win_rate")), "avg_win_pct": pct(h.get("avg_win")),
                            "avg_loss_pct": pct(h.get("avg_loss")), "avg_per_trade_pct": pct(h.get("avg_ret")),
                            "market_same_days_pct": pct(h.get("avg_mkt"))}
    picks = []
    for group in picks_by_strategy.values():
        for p in group:
            picks.append({
                "strategy": p["strategy"], "strategy_label": p["strategy_label"],
                "what_it_does": p.get("strategy_text", ""), "why_picked": p.get("reason", ""),
                "verdict": (p.get("backtest") or {}).get("verdict"),
                "symbol": p["symbol"], "id": p["id"], "name": p.get("name", ""),
                "market_cap": p.get("market_cap"),
                "change_1d_pct": p.get("change_1d_pct"), "change_5d_pct": p.get("change_5d_pct"),
                "change_20d_pct": p.get("change_20d_pct"),
                "as_of": p["as_of"], "close": round(p["price"], 8), "limit_buy": round(p["entry"], 8),
                "stop": round(p["stop"], 8), "target": round(p["target"], 8), "hold_days": int(p["hold"]),
                "position_pct": round(p["position_pct"], 1),
                "warnings": p.get("flags", [])[:4],
                "latest_headline": (p.get("headlines") or [{}])[0].get("title"),
            })
    bench = "S&P 500 (SPY)" if market == "stocks" else "Bitcoin"
    out = {
        "market": market,
        "generated_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_new_york": ny_time(when),
        "as_of": report["end"],
        "prices_from": close_time(market, report["end"]),
        "market_mood": (f"{bench} is above its 200-day average: the overall market is in an uptrend." if regime_on
                        else f"{bench} is below its 200-day average: the market is weak; trend strategies are paused."),
        "held_back": list(held_back),
        "loading": list(loading),
        "risk_per_trade_pct": cfg["trading"]["risk_per_trade_pct"],
        "strategies": strategies,
        "picks": picks,
    }
    f = ledger_file(market).parent / f"latest-{market}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1, default=str))
