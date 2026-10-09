"""Free Tier 2: hard facts about each candidate from SEC EDGAR, Google News and DefiLlama.

These replace the paid web-search step. They catch the drops that are not overreactions:
bankruptcies, delisting notices, restatements, dilution, hacks.
"""

import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

from .http import RateLimitedClient

log = logging.getLogger(__name__)

SEVERE_8K_ITEMS = {"1.03": "bankruptcy filing", "3.01": "delisting notice", "4.02": "financials can't be relied on"}
WARN_8K_ITEMS = {"4.01": "auditor change", "5.02": "officer/director departure"}
DILUTION_FORMS = re.compile(r"^(S-1|S-3|S-3ASR|F-1|F-3|424B\d)$")

SEVERE_NEWS = re.compile(
    r"bankrupt|chapter 11|chapter 7|delist|going concern|fraud|sec charges|indicted|trading halt|"
    r"halted|exploit|hacked|\bhack\b|rug ?pull|insolven|depeg|restat", re.I)
WARN_NEWS = re.compile(
    r"downgrade|offering|dilut|investigation|probe|lawsuit|recall|cuts? guidance|lowers? guidance|"
    r"misses|unlock|layoffs|resigns|steps down|short seller|subpoena", re.I)


class SecChecks:
    def __init__(self):
        # SEC rejects requests without a descriptive User-Agent that includes contact info.
        ua = os.environ.get("SEC_USER_AGENT", "market-scanner research contact@example.com")
        self.client = RateLimitedClient(calls_per_minute=300, headers={"User-Agent": ua})
        self._ciks = None

    def cik_for(self, ticker):
        if self._ciks is None:
            data = self.client.get("https://www.sec.gov/files/company_tickers.json")
            self._ciks = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
        return self._ciks.get(ticker.upper())

    def review(self, ticker, days=90, max_form4=12):
        out = {"severe": [], "warnings": [], "filings": [], "insider_buys": 0, "insider_buy_usd": 0.0,
               "insider_sells": 0, "insider_sell_usd": 0.0}
        cik = self.cik_for(ticker)
        if not cik:
            out["warnings"].append("not found in SEC company list")
            return out
        recent = self.client.get(f"https://data.sec.gov/submissions/CIK{cik}.json")["filings"]["recent"]
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        dilution_cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        form4_docs = []
        n = len(recent["form"])
        for i in range(n):
            form, filed = recent["form"][i], recent["filingDate"][i]
            if filed < cutoff:
                break
            items = (recent.get("items") or [""] * n)[i] or ""
            if form == "4":
                form4_docs.append((recent["accessionNumber"][i], recent["primaryDocument"][i]))
                continue
            if form in ("NT 10-K", "NT 10-Q"):
                out["severe"].append(f"late filing ({form}) {filed}")
            if form == "8-K":
                for item in items.split(","):
                    item = item.strip()
                    if item in SEVERE_8K_ITEMS:
                        out["severe"].append(f"8-K {item} {SEVERE_8K_ITEMS[item]} {filed}")
                    elif item in WARN_8K_ITEMS:
                        out["warnings"].append(f"8-K {item} {WARN_8K_ITEMS[item]} {filed}")
            if DILUTION_FORMS.match(form) and filed >= dilution_cutoff:
                out["warnings"].append(f"dilution: {form} {filed}")
                out["dilution"] = True
            if len(out["filings"]) < 8 and form in ("8-K", "10-Q", "10-K", "S-1", "S-3", "424B5", "NT 10-Q", "NT 10-K"):
                out["filings"].append(f"{form} {filed} {items}".strip())

        for accession, doc in form4_docs[:max_form4]:
            self._tally_form4(cik, accession, doc, out)
        return out

    def _tally_form4(self, cik, accession, doc, out):
        # primaryDocument points at the XSL-rendered HTML; the raw XML sits beside it.
        name = doc.split("/")[-1]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{name}"
        try:
            root = ET.fromstring(self.client.get_text(url))
        except (requests.HTTPError, ET.ParseError):
            return
        for tx in root.iter("nonDerivativeTransaction"):
            code = tx.findtext("transactionCoding/transactionCode")
            shares = float(tx.findtext("transactionAmounts/transactionShares/value") or 0)
            price = float(tx.findtext("transactionAmounts/transactionPricePerShare/value") or 0)
            # Only open-market trades count. Grants, option exercises and tax withholding say little.
            if code == "P":
                out["insider_buys"] += 1
                out["insider_buy_usd"] += shares * price
            elif code == "S":
                out["insider_sells"] += 1
                out["insider_sell_usd"] += shares * price


class NewsChecks:
    def __init__(self):
        self.client = RateLimitedClient(calls_per_minute=30, headers={"User-Agent": "Mozilla/5.0"})

    def headlines(self, query, limit=5, days=7):
        url = (f"https://news.google.com/rss/search?q={quote_plus(query)}+when:{days}d"
               "&hl=en-US&gl=US&ceid=US:en")
        try:
            root = ET.fromstring(self.client.get_text(url))
        except (requests.HTTPError, ET.ParseError):
            return []
        items = []
        for it in root.iter("item"):
            try:
                when = parsedate_to_datetime(it.findtext("pubDate"))
            except (TypeError, ValueError):
                when = datetime(1970, 1, 1, tzinfo=timezone.utc)
            items.append((when, it.findtext("title") or ""))
        # Google ranks by relevance; the newest headlines are the ones that explain today's move.
        items.sort(key=lambda x: x[0], reverse=True)
        return [{"title": t, "date": w.strftime("%m-%d")} for w, t in items[:limit]]

    def review(self, query, limit):
        heads = self.headlines(query, limit=limit)
        severe = [h["title"] for h in heads if SEVERE_NEWS.search(h["title"])]
        warn = [h["title"] for h in heads if WARN_NEWS.search(h["title"]) and h["title"] not in severe]
        return {"headlines": heads, "news_severe": severe, "news_warnings": warn}


class TvlIndex:
    def __init__(self):
        self.client = RateLimitedClient(calls_per_minute=30)
        self._tvl = None

    def get(self, coin_id):
        if self._tvl is None:
            self._tvl = self._build()
        return self._tvl.get(coin_id)

    def _build(self):
        protocols = self.client.get("https://api.llama.fi/protocols")
        # Versions of a protocol (Aave V2, V3, V4) share one token, but DefiLlama usually puts
        # the gecko_id on only one of them, so resolve it through the parent and sum them.
        parent_gid = {p["parentProtocol"]: p["gecko_id"] for p in protocols
                      if p.get("parentProtocol") and p.get("gecko_id")}
        totals = {}
        for p in protocols:
            parent = p.get("parentProtocol")
            # Some parents (Uniswap) have no gecko_id anywhere; their slug usually equals the CoinGecko id.
            gid = p.get("gecko_id") or parent_gid.get(parent) or (parent or "").replace("parent#", "") or None
            tvl = p.get("tvl") or 0
            if not gid or tvl <= 0:
                continue
            t = totals.setdefault(gid, {"tvl": 0.0, "prev_7d": 0.0})
            t["tvl"] += tvl
            t["prev_7d"] += tvl / (1 + (p.get("change_7d") or 0) / 100)
        return {gid: {"tvl_usd": t["tvl"], "tvl_change_7d_pct": (t["tvl"] / t["prev_7d"] - 1) * 100}
                for gid, t in totals.items()}


def review_candidates(candidates, market, cfg):
    """Attach facts and red flags to each candidate dict in place; return the ones that survive."""
    ck = cfg["checks"]
    news = NewsChecks()
    sec = SecChecks() if market == "stocks" else None
    tvl = TvlIndex() if market == "crypto" else None
    memo = {}  # the same asset can be flagged by several strategies
    kept = []
    for c in candidates:
        if c["id"] not in memo:
            memo[c["id"]] = _facts(c, sec, news, tvl, ck)
        c.update({k: (list(v) if isinstance(v, list) else v) for k, v in memo[c["id"]].items()})
        if c["severe"] and ck["exclude_sec_severe"]:
            log.info("%s %s excluded: %s", c["strategy"], c["symbol"], c["severe"])
            continue
        kept.append(c)
    return kept


def _facts(c, sec, news, tvl, ck):
    f = {"flags": [], "severe": []}
    if sec:
        s = sec.review(c["symbol"])
        f["sec"] = s
        f["severe"] += s["severe"]
        f["flags"] += s["warnings"]
        if s["insider_buys"]:
            f["flags"].append(f"insider buying: {s['insider_buys']} trades, ${s['insider_buy_usd']:,.0f}")
        if s.get("dilution") and ck["exclude_dilution"]:
            f["severe"].append("recent offering paperwork")
        query = f'"{c["symbol"]}" stock'
    else:
        query = f'"{c["name_only"]}" crypto'
        t = tvl.get(c["id"])
        if t:
            f["tvl"] = t
            if t["tvl_change_7d_pct"] <= -25:
                f["flags"].append(f"TVL down {t['tvl_change_7d_pct']:.0f}% in 7d")
    n = news.review(query, ck["news_headlines"])
    f.update(n)
    f["flags"] += [f"news: {h}" for h in n["news_warnings"][:2]]
    f["flags"] += [f"NEWS RED FLAG: {h}" for h in n["news_severe"][:2]]
    return f
