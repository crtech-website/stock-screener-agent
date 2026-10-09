import logging
import os

import requests

log = logging.getLogger(__name__)


def fmt_money(x):
    if x is None or x != x:
        return "n/a"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"${x / div:.1f}{unit}"
    return f"${x:,.0f}"


def fmt_price(x):
    return f"${x:,.2f}" if x >= 1 else f"${x:.6g}"


def backtest_line(bt, hold):
    if not bt or not bt.get("n"):
        return "   Backtest: not enough history yet"
    return (f"   Backtest ({hold}-bar hold): avg {bt['avg_ret']:+.1%}, win {bt['win_rate']:.0%}, "
            f"vs market {bt['avg_excess']:+.1%} (t={bt['t_excess']:.1f}, n={bt['n']})")


def format_alert(market, picks_by_strategy, regime_on, when):
    head = [f"{market.upper()} scan {when:%Y-%m-%d %H:%M} UTC",
            f"Market regime: {'benchmark above 200-day avg' if regime_on else 'benchmark BELOW 200-day avg'}"]
    sections = []
    for picks in picks_by_strategy.values():
        if not picks:
            continue
        first = picks[0]
        lines = [f"== {first['strategy_label']} ==", backtest_line(first.get("backtest"), first["hold"])]
        for p in picks:
            name = f" {p['name']}" if p.get("name") else ""
            lines.append(f"{p['symbol']}{name}  {fmt_price(p['price'])}  mcap {fmt_money(p.get('market_cap'))}")
            lines.append(f"   1d {p['change_1d_pct']:+.1f}%  5d {p['change_5d_pct']:+.1f}%  "
                         f"20d {p['change_20d_pct']:+.1f}%  RSI14 {p['rsi14']:.0f}")
            for flag in p.get("flags", [])[:4]:
                lines.append(f"   ! {flag}")
            if p.get("llm"):
                lines.append(f"   AI: {p['llm'].get('verdict')} ({p['llm'].get('confidence')}/10) {p['llm'].get('summary', '')}")
            elif p.get("headlines"):
                lines.append(f"   Latest: {p['headlines'][0]['title']}")
        sections.append("\n".join(lines))
    body = "\n\n".join(sections) if sections else "No signals passed the filters today."
    return "\n".join(head) + "\n\n" + body + "\n\nAutomated screen output, not investment advice."


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
    sent = False
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        for part in chunks(text, 4000):
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": part, "disable_web_page_preview": True},
                          timeout=30).raise_for_status()
        sent = True
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if webhook:
        for part in chunks(text, 1900):
            requests.post(webhook, json={"content": part}, timeout=30).raise_for_status()
        sent = True
    if not sent:
        log.warning("No TELEGRAM_* or DISCORD_WEBHOOK_URL set; alert printed only.")
    return sent
