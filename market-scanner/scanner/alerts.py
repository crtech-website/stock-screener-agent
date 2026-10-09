import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from .secrets import secret

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
RULE = "-" * 34

VERDICT_LINE = {
    "edge": "Verdict: BEAT THE MARKET in testing. Past results are no guarantee.",
    "unproven": "Verdict: NOT PROVEN. The results could be luck. Use small amounts or practice on paper.",
    "losing": "Verdict: LOST MONEY in testing. Consider skipping these picks.",
    "lagging": "Verdict: DID CLEARLY WORSE than just holding the market in testing. Consider skipping.",
    "too_few": "Verdict: NOT ENOUGH HISTORY yet to judge this strategy.",
}

GLOSSARY = """HOW TO PLACE THE ORDERS
- Limit buy: you only buy at your price or lower.
- Take profit: a limit sell. It sells automatically once the price rises to your target.
- Stop loss: a stop order. It sells automatically if the price falls to your stop, so a bad trade can't keep losing.
- Most brokers let you set the take profit and stop loss together, called a bracket or OCO order (one cancels the other).
- A stop loss can fill below your price if the price jumps down overnight.
Automated screen, not investment advice."""


def ny_time(when):
    return when.astimezone(NY).strftime("%A, %b %-d, %Y, %-I:%M %p") + " New York time"


def fmt_money(x):
    if x is None or x != x:
        return "n/a"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"${x / div:.1f}{unit}"
    return f"${x:,.0f}"


def fmt_price(x):
    if x >= 1000:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.4g}"


def track_record(bt, hold, months):
    if not bt or not bt.get("n"):
        return ["Track record: not enough history yet."]
    period = f"the last {months} months" if months else "the available history"
    return [
        f"Track record over {period}, using this exact plan ({bt['n']} past trades):",
        f"  Won {bt['win_rate']:.0%} of trades. Average win {bt['avg_win']:+.1%}, average loss {bt['avg_loss']:+.1%}.",
        f"  Average per trade: {bt['avg_ret']:+.1%}. Buying the whole market on the same days instead "
        f"averaged {bt.get('avg_mkt', bt['avg_ret'] - bt['excess']):+.1%}.",
        f"  {VERDICT_LINE[bt['verdict']]}",
    ]


def trade_plan(p, cfg):
    entry, stop, target = p["entry"], p["stop"], p["target"]
    stop_pct = (stop / entry - 1) * 100
    target_pct = (target / entry - 1) * 100
    rr = target_pct / -stop_pct if stop_pct < 0 else 0
    example = cfg["trading"]["example_account"]
    pos = p["position_pct"]
    return [
        "The plan:",
        f"  1. Buy: limit order at {fmt_price(entry)}. If it hasn't filled by the next close, cancel it."
        + (" This is about 1% above the close because Robinhood's buy price runs that much above the market price."
           if entry > p["price"] * 1.005 else ""),
        f"  2. Take profit: limit sell at {fmt_price(target)} ({target_pct:+.1f}%).",
        f"  3. Stop loss: stop sell at {fmt_price(stop)} ({stop_pct:+.1f}%).",
        f"  4. Time limit: if neither is hit within {p['hold']} days, sell at the market.",
        f"  Possible gain vs possible loss: {rr:.1f} to 1."
        + (" Below 1 is normal for dip-buying strategies, which aim to win often." if rr < 1 else ""),
        f"  Size: put at most {pos:.0f}% of your account in this trade "
        f"(${example * pos / 100:,.0f} of a ${example:,} account). If the stop hits, you lose about "
        f"{cfg['trading']['risk_per_trade_pct']}% of the account.",
    ]


def format_pick(p, cfg):
    name = f" ({p['name']})" if p.get("name") and p["name"] != p["symbol"] else ""
    lines = [f">> {p['symbol']}{name}, closed at {fmt_price(p['price'])}, market value {fmt_money(p.get('market_cap'))}"]
    if p.get("live_price"):
        move = (p["live_price"] / p["price"] - 1) * 100
        lines.append(f"Price now: {fmt_price(p['live_price'])} ({move:+.1f}% since the close)."
                     + (" That's above the buy price, so the limit order may not fill. Don't chase it."
                        if move > 2 else ""))
    lines += [
             f"Why it was picked: {p['reason']}",
             f"Recent moves: {p['change_1d_pct']:+.1f}% today, {p['change_5d_pct']:+.1f}% this week, "
             f"{p['change_20d_pct']:+.1f}% this month."]
    lines += trade_plan(p, cfg)
    if p.get("flags"):
        lines.append("Warnings: " + " | ".join(p["flags"][:4]))
    if p.get("llm"):
        lines.append(f"AI read: {p['llm'].get('summary', '')} ({p['llm'].get('verdict')})")
    if p.get("headlines"):
        h = p["headlines"][0]
        lines.append(f"Latest news ({h['date']}): {h['title']}")
    return "\n".join(lines)


def close_time(market, as_of):
    """When the bar dated as_of closed, in New York time."""
    d = datetime.strptime(as_of, "%Y-%m-%d")
    if market == "stocks":
        return f"the {d:%b %-d} market close (4:00 PM New York time)"
    closed = datetime(d.year, d.month, d.day, tzinfo=ZoneInfo("UTC")) + timedelta(days=1)
    return f"the {d:%b %-d} daily close ({closed.astimezone(NY):%b %-d, %-I:%M %p} New York time)"


def format_alert(market, picks_by_strategy, regime_on, when, cfg, months=None, as_of=None, held_back=(),
                 loading=()):
    bench = "The S&P 500 (SPY)" if market == "stocks" else "Bitcoin"
    mood = (f"{bench} is above its 200-day average, so the overall market is in an uptrend."
            if regime_on else
            f"{bench} is below its 200-day average, so the market is weak. "
            "Trend strategies are paused; be extra careful with the rest.")
    head = [f"{'STOCK' if market == 'stocks' else 'CRYPTO'} SCAN",
            ny_time(when),
            f"Signals and plans use {close_time(market, as_of)}." if as_of else "",
            f"Market mood: {mood}",
            f"Not shown because they lost money or trailed the market in testing: {', '.join(held_back)}." if held_back else "",
            f"Still loading price history, not active yet: {', '.join(loading)}." if loading else ""]
    sections = []
    for picks in picks_by_strategy.values():
        if not picks:
            continue
        first = picks[0]
        block = [RULE, f"STRATEGY: {first['strategy_label']}", f"What it does: {first['strategy_text']}"]
        block += track_record(first.get("backtest"), first["hold"], months)
        if len(picks) > 1:
            risk = cfg["trading"]["risk_per_trade_pct"]
            block.append(f"Note: picks from the same strategy tend to rise and fall together. Taking all "
                         f"{len(picks)} means about {risk * len(picks):g}% of your account is at risk at once.")
        sections.append("\n".join(block))
        sections += [format_pick(p, cfg) for p in picks]
    if not sections:
        # A quiet night gets a short message; the order guide only matters when there's a trade.
        return ("\n".join(x for x in head if x) + "\n\nNo trade ideas passed the filters today. "
                "Nothing to do; the scanner ran normally.\nAutomated screen, not investment advice.")
    return "\n".join(x for x in head if x) + "\n\n" + "\n\n".join(sections) + f"\n\n{RULE}\n" + GLOSSARY


def chunks(text, limit):
    parts, current = [], ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        while len(block) > limit:
            parts.append(block[:limit])
            block = block[limit:]
        current = block
    if current:
        parts.append(current)
    return parts


def send(text):
    try:
        return _send(text)
    except requests.RequestException as e:
        # The ledger and picks file are already written; a messaging hiccup must not fail the run.
        log.warning("alert delivery failed: %s", e)
        return False


def _send(text):
    sent = False
    token, chat = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHAT_ID")
    if token and chat:
        for part in chunks(text, 4000):
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": part, "disable_web_page_preview": True},
                          timeout=30).raise_for_status()
        sent = True
    webhook = secret("DISCORD_WEBHOOK_URL")
    if webhook:
        for part in chunks(text, 1900):
            requests.post(webhook, json={"content": part}, timeout=30).raise_for_status()
        sent = True
    if not sent:
        log.warning("No TELEGRAM_* or DISCORD_WEBHOOK_URL set; alert printed only.")
    return sent
