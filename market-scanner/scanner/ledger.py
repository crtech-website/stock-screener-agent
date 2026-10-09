"""Forward-tracking: every alerted pick is logged, then graded once its holding period has passed.

The backtest says how a strategy did historically. The ledger says how the alerts you
actually received did, which is the number that matters.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .strategies import params_for

COLUMNS = ["as_of", "logged_utc", "strategy", "id", "symbol", "price", "score", "llm_verdict", "n_flags"]


def ledger_file(market):
    return Path(os.environ.get("SCANNER_LEDGER", "ledger")) / f"{market}.csv"


def append(market, picks_by_strategy):
    rows = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    for picks in picks_by_strategy.values():
        for p in picks:
            rows.append({"as_of": p["as_of"], "logged_utc": now, "strategy": p["strategy"], "id": p["id"],
                         "symbol": p["symbol"], "price": p["price"], "score": round(p["score"], 4),
                         "llm_verdict": p.get("llm", {}).get("verdict", ""), "n_flags": len(p.get("flags", []))})
    if not rows:
        return
    f = ledger_file(market)
    f.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=COLUMNS)
    merged = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    # A re-run on the same bar shouldn't count the same pick twice.
    merged.drop_duplicates(["as_of", "strategy", "id"], keep="first").to_csv(f, index=False)


def scorecard(market, panel, cfg):
    f = ledger_file(market)
    if not f.exists():
        return f"{market.upper()} scorecard: no alerts logged yet."
    led = pd.read_csv(f, dtype={"as_of": str})
    close = panel.close
    pos = {d: i for i, d in enumerate(close.index)}
    rows = []
    for _, r in led.iterrows():
        h = params_for(cfg, r["strategy"], market)["hold"] if r["strategy"] in cfg["strategies"] else 5
        i = pos.get(r["as_of"])
        if i is None or i + h >= len(close) or r["id"] not in close:
            continue
        entry, exit_ = close[r["id"]].iloc[i], close[r["id"]].iloc[i + h]
        if not (entry == entry and exit_ == exit_):
            continue
        universe = (close.iloc[i + h] / close.iloc[i] - 1).where(panel.tradable.iloc[i]).mean()
        ret = exit_ / entry - 1
        rows.append({"strategy": r["strategy"], "ret": ret, "excess": ret - universe})
    pending = len(led) - len(rows)
    if not rows:
        return f"{market.upper()} scorecard: {len(led)} alerts logged, none past their holding period yet."
    df = pd.DataFrame(rows)
    lines = [f"{market.upper()} scorecard: {len(df)} graded alerts, {pending} still open or ungradable"]
    for name, g in df.groupby("strategy"):
        lines.append(f"  {name:16} n={len(g):<4} avg {g['ret'].mean():+.2%}  win {(g['ret'] > 0).mean():.0%}  "
                     f"vs market {g['excess'].mean():+.2%}")
    lines.append("Returns are close to close over each strategy's holding period, before costs.")
    return "\n".join(lines)
