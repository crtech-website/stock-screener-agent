"""Watches every coin with an open crypto plan and reports what EMA, Bollinger Bands, RSI and MACD
show on 4-hour candles, plus when a plan reaches its take profit, stop loss or time limit.

The readings are context for a plan that is already running. They were not part of the backtest,
so the plan's stop, take profit and time limit still decide the trade.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from . import alerts, ledger
from .alerts import fmt_price, ny_time
from .indicators import bollinger, ema, macd, rsi
from .strategies import REGISTRY

log = logging.getLogger(__name__)

FOOTER = ("These readings are extra context, not part of the tested plan. The plan's take profit, "
          "stop loss and time limit still decide the trade. Not investment advice.")


def state_file():
    return Path(os.environ.get("SCANNER_CACHE", ".cache")) / "monitor" / "state.json"


def open_plans(now, max_coins):
    """Crypto plans still running, plus ones that ended in the last day so their result gets reported."""
    f = ledger.ledger_file("crypto")
    if not f.exists():
        return []
    led = pd.read_csv(f, dtype={"as_of": str})
    if "stop" not in led:
        return []
    led = led.dropna(subset=["entry", "stop", "target", "hold"])
    plans = []
    for r in led.itertuples():
        # A plan is bought at the signal day's close, which is 00:00 UTC the next day.
        start = pd.Timestamp(r.as_of, tz="UTC") + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=int(r.hold))
        if end > now - timedelta(days=1):
            plans.append({"key": f"{r.as_of}:{r.strategy}:{r.id}", "id": r.id, "symbol": r.symbol,
                          "strategy": r.strategy, "as_of": r.as_of, "entry": float(r.entry),
                          "stop": float(r.stop), "target": float(r.target), "start": start, "end": end})
    # Newest plan per coin, newest coins first, so the API budget goes to the freshest picks.
    plans.sort(key=lambda p: p["as_of"], reverse=True)
    seen, out = set(), []
    for p in plans:
        if p["id"] not in seen:
            seen.add(p["id"])
            out.append(p)
    return out[:max_coins]


def hourly_prices(client, coin, days):
    # Without an interval, CoinGecko returns hourly points for 2 to 90 days.
    data = client.get(f"/coins/{coin}/market_chart", {"vs_currency": "usd", "days": days})
    pts = data.get("prices", [])
    s = pd.Series([p for _, p in pts], index=pd.to_datetime([t for t, _ in pts], unit="ms", utc=True))
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def candles(prices, now, hours):
    """Closing price of each finished candle, labelled by the time it closed.

    The candle still in progress is left out: its readings would flip back and forth until it closes.
    """
    closes = prices.resample(f"{hours}h", closed="left", label="right").last()
    return closes[closes.index <= now].dropna()


def readings(c):
    e20, e50 = ema(c, 20), ema(c, 50)
    mid, up, lo = bollinger(c)
    line, sig, _ = macd(c)
    return pd.DataFrame({"close": c, "e20": e20, "e50": e50, "rsi": rsi(c, 14), "up": up, "lo": lo,
                         "macd": line, "sig": sig}).dropna()


def changes(r):
    """What changed between the last two finished candles, in plain words."""
    a, b = r.iloc[-2], r.iloc[-1]
    out = []

    def crossed_up(x, y):
        return a[x] <= a[y] and b[x] > b[y]

    def crossed_down(x, y):
        return a[x] >= a[y] and b[x] < b[y]

    if crossed_up("macd", "sig"):
        out.append("MACD turned up: upward momentum is picking up.")
    if crossed_down("macd", "sig"):
        out.append("MACD turned down: upward momentum is fading.")
    if a["rsi"] < 70 <= b["rsi"]:
        out.append(f"RSI rose above 70 (now {b['rsi']:.0f}): overbought. The price has risen fast, "
                   "and it often pauses or pulls back after this.")
    if a["rsi"] >= 70 > b["rsi"]:
        out.append(f"RSI fell back below 70 (now {b['rsi']:.0f}): the fast run-up is cooling off.")
    if a["rsi"] > 30 >= b["rsi"]:
        out.append(f"RSI fell below 30 (now {b['rsi']:.0f}): oversold. The price has dropped fast; "
                   "a bounce often follows, but not always.")
    if a["rsi"] <= 30 < b["rsi"]:
        out.append(f"RSI climbed back above 30 (now {b['rsi']:.0f}): the selling is easing.")
    if crossed_up("close", "up"):
        out.append("Price closed above the upper Bollinger Band: a strong push up, and stretched. "
                   "Moves this far from the average often give part of it back.")
    if crossed_down("close", "lo"):
        out.append("Price closed below the lower Bollinger Band: a sharp drop, stretched to the downside.")
    if crossed_down("close", "e20"):
        out.append("Price fell below its 20-candle average (EMA20): the short-term uptrend is weakening.")
    if crossed_up("close", "e20"):
        out.append("Price rose above its 20-candle average (EMA20): the short-term trend turned up.")
    if crossed_down("e20", "e50"):
        out.append("EMA20 crossed below EMA50: the trend of the last week or so has turned down.")
    if crossed_up("e20", "e50"):
        out.append("EMA20 crossed above EMA50: the trend of the last week or so has turned up.")
    return out


def snapshot(r):
    b = r.iloc[-1]
    if b["close"] > b["e20"] and b["close"] > b["e50"]:
        trend = "above its EMA20 and EMA50, an uptrend"
    elif b["close"] < b["e20"] and b["close"] < b["e50"]:
        trend = "below its EMA20 and EMA50, a downtrend"
    else:
        trend = "between its EMA20 and EMA50, no clear trend"
    zone = "overbought" if b["rsi"] >= 70 else "oversold" if b["rsi"] <= 30 else "neutral"
    pos = (b["close"] - b["lo"]) / (b["up"] - b["lo"]) if b["up"] > b["lo"] else 0.5
    band = ("above the upper band" if pos > 1 else "near the upper band" if pos >= 0.8 else
            "below the lower band" if pos < 0 else "near the lower band" if pos <= 0.2 else "mid-range")
    mom = "above its signal line, momentum up" if b["macd"] > b["sig"] else "below its signal line, momentum down"
    return [f"  Trend: price is {trend}.",
            f"  RSI: {b['rsi']:.0f} out of 100, {zone}.",
            f"  Bollinger Bands: {band}.",
            f"  MACD: {mom}."]


def plan_status(p, prices, now, half_spread):
    """Play the plan out on hourly prices: first take profit or stop wins, else the time limit."""
    after = prices[prices.index >= p["start"]]
    if after.empty:
        return None
    # The limit buy lasts one day and fills at Robinhood's ask (above the middle price) or better.
    day_one = after[after.index < p["start"] + timedelta(days=1)]
    can_fill = day_one[day_one * (1 + half_spread) <= p["entry"]]
    if can_fill.empty:
        if now >= p["start"] + timedelta(days=1):
            return {"reason": "unfilled", "when": p["start"] + timedelta(days=1), "result": None}
        return None
    filled_at = can_fill.index[0]
    if can_fill.iloc[0] <= p["stop"]:
        return {"reason": "below_stop", "when": filled_at, "result": None}
    entry = min(p["entry"], can_fill.iloc[0] * (1 + half_spread))
    after = after[after.index >= filled_at]
    hits = []
    for level, reason in ((p["target"], "target"), (p["stop"], "stop")):
        touched = after[after >= level] if reason == "target" else after[after <= level]
        if not touched.empty:
            hits.append((touched.index[0], reason, level))
    if hits:
        when, reason, level = min(hits)
    elif now >= p["end"] and not after.empty:
        when, reason, level = p["end"], "time", float(after[after.index <= p["end"]].iloc[-1])
    else:
        return None
    # Robinhood sells below the middle price, so the result is what you'd actually have received.
    result = level * (1 - half_spread) / entry - 1
    return {"reason": reason, "when": when, "result": result}


def plan_line(p, price, now):
    days_left = max(0, (p["end"] - now).days)
    label = REGISTRY[p["strategy"]].label if p["strategy"] in REGISTRY else p["strategy"]
    to_target = (p["target"] / price - 1) * 100
    to_stop = (1 - p["stop"] / price) * 100
    return (f"{p['symbol']}, about {fmt_price(price)}. {label} plan from {pd.Timestamp(p['as_of']):%b %-d}: "
            f"take profit {fmt_price(p['target'])} ({to_target:+.1f}% away), "
            f"stop loss {fmt_price(p['stop'])} ({to_stop:.1f}% below), {days_left} days left.")


def finished_text(p, done):
    when = f"{done['when'].astimezone(alerts.NY):%b %-d, %-I %p}"
    if done["reason"] == "unfilled":
        return (f"{p['symbol']}: PLAN NEVER STARTED. The price stayed above the limit buy for the whole first day, "
                "so the order would not have filled. No longer watching this one.")
    if done["reason"] == "below_stop":
        return (f"{p['symbol']}: PLAN SKIPPED. By the time it could be bought ({when}) the price was already "
                "below the stop loss. No longer watching this one.")
    what = {"target": "reached its take profit", "stop": "hit its stop loss",
            "time": "reached its time limit (sell at the market)"}[done["reason"]]
    return (f"{p['symbol']}: PLAN FINISHED. It {what} on {when}. "
            f"Result if you followed the plan: {done['result']:+.1%} after Robinhood's spread. "
            "No longer watching this one.")


def run(cfg, client, now=None):
    m = cfg["monitor"]
    now = now or datetime.now(timezone.utc)
    half_spread = cfg["crypto"]["half_spread_pct"] / 100
    f = state_file()
    state = json.loads(f.read_text()) if f.exists() else {"plans": {}}
    plans = open_plans(now, m["max_coins"])
    blocks = []
    for p in plans:
        st = state["plans"].get(p["key"], {})
        if st.get("finished"):
            continue
        try:
            prices = hourly_prices(client, p["id"], m["history_days"])
        except requests.RequestException as e:
            log.warning("monitor: prices for %s failed (%s); will retry next run", p["symbol"], e)
            continue
        done = plan_status(p, prices, now, half_spread)
        if done:
            blocks.append(finished_text(p, done))
            state["plans"][p["key"]] = {"finished": True}
            continue
        r = readings(candles(prices, now, m["candle_hours"]))
        if len(r) < 2:
            log.info("monitor: not enough candles yet for %s", p["symbol"])
            continue
        last_bar = r.index[-1].isoformat()
        first_look = "last_bar" not in st
        events = [] if first_look or st["last_bar"] == last_bar else changes(r)
        if first_look:
            blocks.append("\n".join([plan_line(p, prices.iloc[-1], now), "  Now watching. Readings now:",
                                     *snapshot(r)]))
        elif events:
            blocks.append("\n".join([plan_line(p, prices.iloc[-1], now), "  What changed:",
                                     *[f"  - {e}" for e in events], "  Readings now:", *snapshot(r)]))
        state["plans"][p["key"]] = {"last_bar": last_bar}
    # Forget plans that dropped off the list, so the state file stays small.
    keep = {p["key"] for p in plans}
    state["plans"] = {k: v for k, v in state["plans"].items() if k in keep}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state))
    if not blocks:
        log.info("monitor: %d plans watched, nothing new", len(plans))
        return None
    text = "\n\n".join([f"CRYPTO WATCH\n{ny_time(now)}\nReadings use {m['candle_hours']}-hour candles "
                        f"(each candle is {m['candle_hours']} hours of price moves).", *blocks, FOOTER])
    print(text)
    alerts.send(text)
    return text
