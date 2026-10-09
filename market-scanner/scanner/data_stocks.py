import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .http import RateLimitedClient
from .secrets import secret
from .indicators import sma
from .panel import Panel

log = logging.getLogger(__name__)


def cache_dir():
    return Path(os.environ.get("SCANNER_CACHE", ".cache")) / "stocks"


def client_from_env(cfg):
    s = cfg["stocks"]
    key = secret("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is not set. Add it under Settings > Secrets and variables > Actions.")
    return RateLimitedClient(s["api_base"], s["calls_per_minute"], params={"apiKey": key})


def _cached(path, max_age_days):
    return path.exists() and (datetime.now().timestamp() - path.stat().st_mtime) < max_age_days * 86400


def _ticker_pages(client, active):
    page = client.get("/v3/reference/tickers", {"market": "stocks", "type": "CS",
                                                "active": "true" if active else "false", "limit": 1000})
    rows = page.get("results", [])
    while page.get("next_url"):
        page = client.get(page["next_url"])
        rows.extend(page.get("results", []))
    return pd.DataFrame(rows, columns=["ticker", "name"]).assign(active=active)


def load_universe(client, cfg):
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    active_file, delisted_file = d / "universe_active.pkl", d / "universe_delisted.pkl"
    if not _cached(active_file, 7):
        _ticker_pages(client, True).to_pickle(active_file)
    frames = [pd.read_pickle(active_file)]
    if cfg["stocks"]["include_delisted"]:
        if not _cached(delisted_file, 30):
            _ticker_pages(client, False).to_pickle(delisted_file)
        frames.append(pd.read_pickle(delisted_file))
    # A ticker reused by a new company shows up in both lists; the active row wins.
    return pd.concat(frames).drop_duplicates("ticker", keep="first")


def trading_days_wanted(history_days, today):
    start = today - timedelta(days=int(history_days * 1.46))
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, today)]


def _load_state(d):
    f = d / "state.json"
    return json.loads(f.read_text()) if f.exists() else {"fetched": {}, "splits_checked": None}


def update_bars(client, cfg, keep, today=None):
    """Maintain a rolling store of daily bars for the whole US market.

    Grouped-daily returns every stock for one date in a single call, so a full-market
    history costs one call per trading day, and each run after the backfill needs one call.
    """
    s = cfg["stocks"]
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    state = _load_state(d)
    bars_file = d / "bars.pkl"
    bars = pd.read_pickle(bars_file) if bars_file.exists() else pd.DataFrame()

    wanted = trading_days_wanted(s["history_days"], today)
    # Newest first, so a partial backfill still gives a contiguous recent window to scan.
    missing = [day for day in reversed(wanted) if day not in state["fetched"]][: s["max_backfill_per_run"]]
    log.info("stocks: %d of %d trading days cached, fetching %d",
             len(set(state["fetched"]) & set(wanted)), len(wanted), len(missing))

    keep = set(keep) | {s["benchmark"]}
    frames = []
    for day in missing:
        try:
            data = client.get(f"/v2/aggs/grouped/locale/us/market/stocks/{day}", {"adjusted": "true"})
        except requests.HTTPError as e:
            log.info("stocks: %s not available (%s)", day, e.response.status_code)
            continue
        results = data.get("results") or []
        if results:
            f = pd.DataFrame(results).rename(columns={"T": "ticker"})
            f = f[f["ticker"].isin(keep)]
            f["date"] = day
            frames.append(f[["date", "ticker", "h", "l", "c", "v"]])
            state["fetched"][day] = today.isoformat()
        elif (today - date.fromisoformat(day)).days > 3:
            # Old empty dates are market holidays. Recent empty ones may not be published yet.
            state["fetched"][day] = today.isoformat()

    if frames:
        new = pd.concat(frames, ignore_index=True)
        new[["h", "l", "c", "v"]] = new[["h", "l", "c", "v"]].astype("float32")
        bars = pd.concat([bars, new], ignore_index=True)

    bars = apply_new_splits(client, bars, state, today)
    bars = bars[bars["date"] >= wanted[0]]
    state["fetched"] = {k: v for k, v in state["fetched"].items() if k >= wanted[0]}
    bars.to_pickle(bars_file)
    (d / "state.json").write_text(json.dumps(state))
    return bars


def apply_new_splits(client, bars, state, today):
    """Adjust cached bars for splits that happened after those bars were downloaded.

    Bars fetched after a split are already adjusted by the API; only older downloads need
    fixing. Without this, every split looks like a crash or a moonshot to the strategies.
    """
    since = state.get("splits_checked")
    state["splits_checked"] = today.isoformat()
    if since is None or bars.empty:
        return bars
    page = client.get("/v3/reference/splits", {"execution_date.gt": since,
                                              "execution_date.lte": today.isoformat(), "limit": 1000})
    splits = page.get("results", [])
    while page.get("next_url"):
        page = client.get(page["next_url"])
        splits.extend(page.get("results", []))

    fetched_on = bars["date"].map(state["fetched"]).fillna(today.isoformat())
    for sp in splits:
        if not sp.get("split_from") or not sp.get("split_to"):
            continue
        ex = sp["execution_date"]
        factor = sp["split_from"] / sp["split_to"]
        rows = (bars["ticker"] == sp["ticker"]) & (bars["date"] < ex) & (fetched_on < ex)
        if rows.any():
            bars.loc[rows, ["h", "l", "c"]] *= factor
            bars.loc[rows, "v"] /= factor
            log.info("stocks: adjusted %s for %s:%s split on %s", sp["ticker"], sp["split_to"], sp["split_from"], ex)
    return bars


def build_panel(bars, universe, cfg):
    s = cfg["stocks"]
    # A day fetched twice (e.g. a cache restored without its state file) must not crash the pivot.
    bars = bars.drop_duplicates(["date", "ticker"], keep="last")
    bench = bars[bars["ticker"] == s["benchmark"]].set_index("date")["c"].sort_index()
    bars = bars[bars["ticker"].isin(set(universe["ticker"]))]

    def wide(col):
        return bars.pivot(index="date", columns="ticker", values=col).sort_index().astype("float64")

    close, volume, high, low = wide("c"), wide("v"), wide("h"), wide("l")
    dollar_vol = sma((close * volume).fillna(0), 20)
    tradable = close.notna() & (close >= s["min_price"]) & (dollar_vol >= s["min_avg_dollar_volume"])
    names = universe.set_index("ticker")["name"].to_dict()
    return Panel("stocks", close, volume, tradable, bench, high=high, low=low, names=names)


def market_caps(client, tickers, max_age_days=7):
    """Market cap isn't in the bulk feed, so it's looked up only for candidates and cached a week."""
    f = cache_dir() / "market_caps.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
    out = {}
    for t in tickers:
        hit = cache.get(t)
        if hit and hit["on"] >= cutoff:
            out[t] = hit["cap"]
            continue
        try:
            out[t] = client.get(f"/v3/reference/tickers/{t}").get("results", {}).get("market_cap")
        except requests.HTTPError:
            out[t] = None
        cache[t] = {"cap": out[t], "on": date.today().isoformat()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cache))
    return out


def load(cfg, refresh=True, client=None):
    client = client or client_from_env(cfg)
    universe = load_universe(client, cfg)
    if refresh:
        bars = update_bars(client, cfg, universe["ticker"])
    else:
        bars = pd.read_pickle(cache_dir() / "bars.pkl")
    return build_panel(bars, universe, cfg), client
