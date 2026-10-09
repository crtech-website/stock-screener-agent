import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .http import RateLimitedClient
from .panel import Panel

log = logging.getLogger(__name__)

COINGECKO = "https://api.coingecko.com/api/v3"
STABLE_SYMBOLS = {
    "usdt", "usdc", "dai", "fdusd", "tusd", "usdd", "pyusd", "usde", "usds", "frax", "lusd",
    "gusd", "busd", "usdp", "eurc", "eurt", "xaut", "paxg", "usd0", "rlusd", "usdtb", "usdg",
}
DERIVATIVE_NAME = re.compile(r"\b(wrapped|staked|bridged|liquid staking|restaked)\b|\busd\b", re.I)


def cache_dir():
    return Path(os.environ.get("SCANNER_CACHE", ".cache")) / "crypto"


def client_from_env(cfg):
    headers = {"accept": "application/json"}
    rate = cfg["crypto"]["calls_per_minute"]
    if os.environ.get("COINGECKO_API_KEY"):
        headers["x-cg-demo-api-key"] = os.environ["COINGECKO_API_KEY"]
    else:
        # Keyless access works for trying things out but is throttled far below the Demo plan.
        rate = min(rate, 4)
    return RateLimitedClient(COINGECKO, rate, headers=headers)


def fetch_markets(client, min_market_cap, max_pages=60):
    """Page through /coins/markets by market cap and stop at the floor.

    Sorting by market cap means the long tail of 10k+ dead tokens is never downloaded.
    """
    rows = []
    for page in range(1, max_pages + 1):
        batch = client.get("/coins/markets", {"vs_currency": "usd", "order": "market_cap_desc",
                                              "per_page": 250, "page": page})
        if not batch:
            break
        rows.extend(batch)
        if (batch[-1].get("market_cap") or 0) < min_market_cap:
            break
    return pd.DataFrame(rows)


def is_stable_or_derivative(symbol, name, price=None, flat=False):
    if str(symbol).lower() in STABLE_SYMBOLS or DERIVATIVE_NAME.search(str(name)):
        return True
    return price is not None and 0.97 <= price <= 1.03 and flat


def daily_history(client, coin_id, days):
    """Daily closes, labelled by the day they close.

    CoinGecko stamps each daily point at 00:00 UTC, which is the close of the previous day.
    The final point is the live price, so the current UTC day is dropped as incomplete.
    """
    data = client.get(f"/coins/{coin_id}/market_chart",
                      {"vs_currency": "usd", "days": days, "interval": "daily"})
    frames = []
    for key, col in (("prices", "close"), ("market_caps", "market_cap"), ("total_volumes", "volume")):
        f = pd.DataFrame(data.get(key, []), columns=["ts", col])
        f["date"] = (pd.to_datetime(f["ts"], unit="ms", utc=True) - pd.Timedelta(seconds=1)).dt.strftime("%Y-%m-%d")
        frames.append(f.drop(columns="ts").drop_duplicates("date", keep="last").set_index("date"))
    out = pd.concat(frames, axis=1).reset_index()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return out[out["date"] < today].assign(id=coin_id)


def update_history(client, cfg, now=None):
    c = cfg["crypto"]
    now = now or datetime.now(timezone.utc)
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    hist_file, meta_file = d / "history.pkl", d / "meta.json"
    hist = pd.read_pickle(hist_file) if hist_file.exists() else pd.DataFrame(
        columns=["date", "id", "close", "market_cap", "volume"])
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    markets = fetch_markets(client, c["min_market_cap"])
    for _, m in markets.iterrows():
        meta[m["id"]] = {"symbol": m["symbol"].upper(), "name": m["name"]}
    liquid = markets[(markets["market_cap"].fillna(0) >= c["min_market_cap"])
                     & (markets["total_volume"].fillna(0) >= c["min_volume_24h"])]
    liquid = liquid[~liquid.apply(lambda r: is_stable_or_derivative(r["symbol"], r["name"]), axis=1)]

    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    last_seen = hist.groupby("id")["date"].max() if not hist.empty else pd.Series(dtype=str)
    counts = hist.groupby("id").size() if not hist.empty else pd.Series(dtype=int)

    # New coins get a full history; coins with a gap get just enough to close it.
    jobs = []
    for coin in liquid["id"]:
        if counts.get(coin, 0) < 30:
            jobs.append((coin, c["history_days"]))
        else:
            gap = (now.date() - pd.Timestamp(last_seen[coin]).date()).days
            if gap > 2:
                jobs.append((coin, min(c["history_days"], gap + 2)))
    jobs = jobs[: c["max_backfill_per_run"]]
    log.info("crypto: %d coins above floor, %d liquid, %d history fetches", len(markets), len(liquid), len(jobs))

    fetched = [daily_history(client, coin, days) for coin, days in jobs]

    # A run just after 00:00 UTC sees yesterday's close in /coins/markets, so it can be stored
    # without a per-coin call. Runs at other hours get a provisional bar that isn't saved.
    snap = liquid[["id", "current_price", "market_cap", "total_volume"]].rename(
        columns={"current_price": "close", "total_volume": "volume"})
    parts = list(fetched)
    provisional = None
    if now.hour < 3:
        parts.append(snap.assign(date=yesterday))
    else:
        provisional = snap.assign(date=now.strftime("%Y-%m-%d"))

    # Fresh rows come before stored ones so they win on duplicates.
    hist = pd.concat([*parts, hist], ignore_index=True).drop_duplicates(["date", "id"], keep="first")
    cutoff = (now - timedelta(days=c["history_days"] + 5)).strftime("%Y-%m-%d")
    hist = hist[hist["date"] >= cutoff]
    hist.to_pickle(hist_file)
    meta_file.write_text(json.dumps(meta))

    if provisional is not None:
        hist = pd.concat([hist, provisional], ignore_index=True).drop_duplicates(["date", "id"], keep="last")
    return hist, meta


def build_panel(hist, meta, cfg):
    c = cfg["crypto"]

    def wide(col):
        return hist.pivot(index="date", columns="id", values=col).sort_index().astype("float64")

    close, mcap, volume = wide("close"), wide("market_cap"), wide("volume")
    bench = close[c["benchmark"]] if c["benchmark"] in close else pd.Series(dtype=float)
    excluded = [i for i in close.columns
                if is_stable_or_derivative(meta.get(i, {}).get("symbol", ""), meta.get(i, {}).get("name", ""))]
    keep = [i for i in close.columns if i not in excluded]
    close, mcap, volume = close[keep], mcap[keep], volume[keep]
    tradable = close.notna() & (mcap >= c["min_market_cap"]) & (volume >= c["min_volume_24h"])
    names = {i: meta.get(i, {}).get("name", i) for i in keep}
    symbols = {i: meta.get(i, {}).get("symbol", i) for i in keep}
    return Panel("crypto", close, volume, tradable, bench, market_cap=mcap, names=names, symbols=symbols)


def load(cfg, refresh=True, client=None):
    client = client or client_from_env(cfg)
    if refresh:
        hist, meta = update_history(client, cfg)
    else:
        d = cache_dir()
        hist, meta = pd.read_pickle(d / "history.pkl"), json.loads((d / "meta.json").read_text())
    return build_panel(hist, meta, cfg), client
