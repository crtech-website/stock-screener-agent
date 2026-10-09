"""Optional LLM read of each finalist, using any free OpenAI-compatible API.

The model only sees facts the code already gathered (prices, filings, insider trades,
headlines), so no paid web search is needed. It writes commentary and can veto a pick it
judges structurally broken; it never reorders the quant ranking, because that ranking is
what the backtest measured.
"""

import json
import logging
import os
import re

import requests

from .http import RateLimitedClient

log = logging.getLogger(__name__)

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

SYSTEM = """You are a skeptical analyst. A rules-based screen flagged this asset. Using ONLY the facts
provided, judge whether the setup looks like a temporary move or a structural problem.
Do not invent facts. If the facts don't explain the move, say so and answer "unclear"."""

OUTPUT_SPEC = """Reply with exactly one block and nothing after it:
<verdict>
{"verdict": "temporary" | "structural" | "unclear",
 "confidence": <integer 0-10>,
 "summary": "<two sentences max: what is going on and the main risk>"}
</verdict>"""

VERDICT_RE = re.compile(r"<verdict>\s*(\{.*?\})\s*</verdict>", re.S)

FACT_KEYS = ["strategy_label", "symbol", "name", "price", "change_1d_pct", "change_5d_pct", "change_20d_pct",
             "rsi14", "pct_from_sma200", "market_cap", "avg_dollar_volume", "flags", "severe", "headlines", "tvl"]


def parse_verdict(text):
    found = VERDICT_RE.findall(text or "")
    return json.loads(found[-1]) if found else None


def build_prompt(c):
    facts = {k: c[k] for k in FACT_KEYS if c.get(k) not in (None, [], {})}
    if c.get("sec"):
        facts["recent_filings"] = c["sec"]["filings"]
        facts["insider_open_market"] = {k: c["sec"][k] for k in
                                        ("insider_buys", "insider_buy_usd", "insider_sells", "insider_sell_usd")}
    return f"```json\n{json.dumps(facts, indent=1, default=str)}\n```\n\n{OUTPUT_SPEC}"


def review(candidates, cfg):
    lc = cfg["llm"]
    if lc["provider"] == "none":
        return candidates
    base, key_env = PROVIDERS[lc["provider"]]
    if not os.environ.get(key_env):
        log.warning("llm.provider is %s but %s is not set; skipping LLM review", lc["provider"], key_env)
        return candidates
    client = RateLimitedClient(base, lc["calls_per_minute"],
                               headers={"Authorization": f"Bearer {os.environ[key_env]}"})
    for c in candidates[: lc["max_reviews"]]:
        try:
            resp = client.post("/chat/completions", {
                "model": lc["model"],
                "temperature": 0.2,
                "max_tokens": 700,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": build_prompt(c)}],
            })
            verdict = parse_verdict(resp["choices"][0]["message"]["content"])
        except (requests.HTTPError, KeyError, json.JSONDecodeError) as e:
            # A free tier running out mid-run shouldn't cost the whole alert.
            log.warning("LLM review failed for %s: %s", c["symbol"], e)
            verdict = None
        if verdict:
            c["llm"] = verdict
    return candidates
