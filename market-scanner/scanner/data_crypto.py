import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .http import RateLimitedClient
from .secrets import secret
from .panel import Panel

log = logging.getLogger(__name__)

COINGECKO = "https://api.coingecko.com/api/v3"
STABLE_SYMBOLS = {
    "usdt", "usdc", "dai", "fdusd", "tusd", "usdd", "pyusd", "usde", "usds", "frax", "lusd",
    "gusd", "busd", "usdp", "eurc", "eurt", "xaut", "paxg", "usd0", "rlusd", "usdtb", "usdg",
}
DERIVATIVE_NAME = re.compile(
    r"\b(wrapped|staked|bridged|liquid staking|restaked|heloc|treasury|treasuries|t-bill|tokenized|"
    r"money market|yield fund|gold)\b|\busd\b", re.I)


def cache_dir():
    return Path(os.environ.get("SCANNER_CACHE", ".cache")) / "crypto"


def client_from_env(cfg):
    headers = {"accept": "application/json"}
    rate = cfg["crypto"]["calls_per_minute"]
    if secret("COINGECKO_API_KEY"):
        headers["x-cg-demo-api-key"] = secret("COINGECKO_API_KEY")
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


def venue_symbols(cfg):
    """Symbols the trading platform sells, or None to scan everything."""
    name = cfg["crypto"].get("only_symbols_file")
    if not name:
        return None
    path = Path(name)
    if not path.exists():
        log.warning("crypto.only_symbols_file %s not found; scanning every coin", name)
        return None
    return {line.strip().upper() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def keep_venue_coins(markets, symbols):
    """Restrict to coins the platform sells. When several coins share a ticker, the largest one wins,
    since that is almost always the one a broker lists."""
    if symbols is None:
        return markets
    listed = markets[markets["symbol"].str.upper().isin(symbols)]
    return listed.sort_values("market_cap", ascending=False).drop_duplicates("symbol", keep="first")


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
    liquid = keep_venue_coins(liquid, venue_symbols(cfg))

    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    last_seen = hist.groupby("id")["date"].max() if not hist.empty else pd.Series(dtype=str)
    counts = hist.groupby("id").size() if not hist.empty else pd.Series(dtype=int)

    # New coins get a full history; coins with a gap get just enough to close it.
    jobs = []
    for coin in liquid["id"]:
        if counts.get(coin, 0) < 30:
            jobs.append((coin, c["history_days"]))
        else:
            # gap 1 = yesterday's close is stored. CoinGecko publishes it about 35 minutes after
            # 00:00 UTC, which is why the nightly scan runs at 00:45 UTC.
            gap = (now.date() - pd.Timestamp(last_seen[coin]).date()).days
            if gap >= 2:
                jobs.append((coin, min(c["history_days"], gap + 1)))
    jobs = jobs[: c["max_backfill_per_run"]]
    log.info("crypto: %d coins above floor, %d liquid, %d history fetches", len(markets), len(liquid), len(jobs))

    fetched = []
    for coin, days in jobs:
        try:
            fetched.append(daily_history(client, coin, days))
        except requests.RequestException as e:
            # One coin failing (rate limit, delisting) shouldn't sink the night; it's retried next run.
            log.warning("crypto: history for %s failed (%s); will retry next run", coin, e)

    # Fallback for a coin whose history call didn't include yesterday yet: a run soon after
    # 00:00 UTC stores the current price as yesterday's close. Fetched closes win over it.
    # Runs at other hours keep the current price only for display.
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
    meta_file.write_text(json.dumps({k: v for k, v in meta.items() if k != "_live"}))

    if provisional is not None:
        meta["_live"] = {"prices": provisional.set_index("id")["close"].dropna().to_dict(),
                         "at": now.isoformat()}
    return hist, meta


def remove_spikes(close, factor=3.0):
    """Blank out one-day bad prints: a price 3x away from both the day before and the day after."""
    up = (close / close.shift(1) > factor) & (close / close.shift(-1) > factor)
    down = (close / close.shift(1) < 1 / factor) & (close / close.shift(-1) < 1 / factor)
    return close.mask(up | down)


def build_panel(hist, meta, cfg):
    c = cfg["crypto"]

    def wide(col):
        return hist.pivot(index="date", columns="id", values=col).sort_index().astype("float64")

    # Only completed UTC days. A day still in progress would make signals differ from what was
    # backtested, since every backtest bar is a full day.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hist = hist[hist["date"] < today]
    close, mcap, volume = wide("close"), wide("market_cap"), wide("volume")
    bench = close[c["benchmark"]] if c["benchmark"] in close else pd.Series(dtype=float)
    excluded = [i for i in close.columns
                if is_stable_or_derivative(meta.get(i, {}).get("symbol", ""), meta.get(i, {}).get("name", ""))]
    # Pegged tokens (stablecoins, tokenized loans and treasuries) that the name check missed:
    # they sit within 10% of $1 nearly every day and barely move.
    recent = close.tail(90)
    near_one = recent.apply(lambda s: s.dropna().between(0.9, 1.1).mean() if s.notna().any() else 0)
    typical_move = recent.pct_change(fill_method=None).abs().median()
    excluded += list(close.columns[(near_one >= 0.85) & (typical_move < 0.02)])
    allowed = venue_symbols(cfg)
    if allowed is not None:
        # Coins cached before the list was applied, or that share a ticker with a listed coin.
        ids = pd.DataFrame({"id": close.columns,
                            "symbol": [meta.get(i, {}).get("symbol", "").upper() for i in close.columns],
                            "cap": mcap.ffill().iloc[-1].reindex(close.columns).fillna(0).to_numpy()})
        ids = ids[ids["symbol"].isin(allowed)].sort_values("cap", ascending=False).drop_duplicates("symbol")
        excluded += [i for i in close.columns if i not in set(ids["id"])]
    keep = [i for i in close.columns if i not in excluded]
    close, mcap, volume = remove_spikes(close[keep]), mcap[keep], volume[keep]
    tradable = close.notna() & (mcap >= c["min_market_cap"]) & (volume >= c["min_volume_24h"])
    names = {i: meta.get(i, {}).get("name", i) for i in keep}
    symbols = {i: meta.get(i, {}).get("symbol", i) for i in keep}
    live = meta.get("_live", {})
    return Panel("crypto", close, volume, tradable, bench, market_cap=mcap, names=names, symbols=symbols,
                 live=live.get("prices", {}), live_at=live.get("at"))


def load(cfg, refresh=True, client=None):
    client = client or client_from_env(cfg)
    if refresh:
        hist, meta = update_history(client, cfg)
    else:
        d = cache_dir()
        hist, meta = pd.read_pickle(d / "history.pkl"), json.loads((d / "meta.json").read_text())
    return build_panel(hist, meta, cfg), client
