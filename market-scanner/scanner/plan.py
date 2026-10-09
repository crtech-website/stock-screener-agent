"""Trade plans: where to buy, take profit and stop out, and how each plan played out in the past.

The same `levels` and `simulate` functions drive the alert, the backtest and the scorecard,
so the track record shown next to a pick is the record of the exact orders it recommends.
"""

import math

import numpy as np
import pandas as pd

from .indicators import sma


def atr(panel, n=14):
    """Wilder ATR: the typical daily price move. Uses close-to-close moves when highs/lows are missing."""
    c = panel.close
    if panel.high is not None:
        prev = c.shift(1)
        tr = np.fmax(np.fmax(panel.high - panel.low, (panel.high - prev).abs()), (panel.low - prev).abs())
    else:
        tr = c.diff().abs()
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def levels(entry, atr_v, sma_v, prm, cfg):
    """Stop and target for an entry price. Works on scalars and numpy arrays."""
    floor = entry * (1 - cfg["trading"]["max_stop_pct"] / 100)
    stop = np.maximum(entry - prm["stop_atr"] * atr_v, floor)
    target = entry + prm["target_atr"] * atr_v
    if sma_v is not None:
        target = np.maximum(target, sma_v)
    return stop, target


def position_pct(entry, stop, cfg):
    """Share of the account to put in so that hitting the stop loses `risk_per_trade_pct`."""
    t = cfg["trading"]
    stop_pct = (entry - stop) / entry * 100
    return min(t["risk_per_trade_pct"] / stop_pct * 100, t["max_position_pct"]) if stop_pct > 0 else 0.0


def market_returns(panel, ti, exit_day, cache=None):
    """What buying every liquid asset equally on day ti and selling on exit_day returned.

    Buy-and-hold over each trade's own window, dropping the top and bottom 2% so a few
    bad prints can't move the baseline.
    """
    c = panel.close.to_numpy(dtype="float64")
    trad = panel.tradable.to_numpy()
    cache = {} if cache is None else cache
    out = np.empty(len(ti))
    for i, (a, b) in enumerate(zip(ti, exit_day)):
        key = (int(a), int(b))
        if key not in cache:
            r = c[b][trad[a]] / c[a][trad[a]] - 1
            r = np.sort(r[np.isfinite(r)])
            trim = max(1, int(len(r) * 0.02)) if len(r) >= 20 else 0
            cache[key] = float(r[trim:len(r) - trim].mean()) if len(r) > 2 * trim else 0.0
        out[i] = cache[key]
    return out


def simulate(close, high, low, ti, aj, stop, target, hold, cost):
    """Play out bracket trades bought at close[ti, aj].

    Each later day: a low at or under the stop sells at the stop (or that day's high if the
    price gapped below it), a high at or over the target sells at the target, and if neither
    happens within `hold` days the position is sold at that day's close. When both are touched
    on the same day the stop is assumed to have hit first, since daily bars can't tell.
    Trades whose holding window runs past the data are dropped.
    """
    T = close.shape[0]
    ok = ti + hold < T
    ti, aj, stop, target = ti[ok], aj[ok], stop[ok], target[ok]
    if len(ti) == 0:
        return pd.DataFrame({"ti": pd.Series(dtype=int), "aj": pd.Series(dtype=int), "ret": pd.Series(dtype=float),
                             "exit_day": pd.Series(dtype=int), "exit_reason": pd.Series(dtype=str)})
    idx = ti[:, None] + np.arange(1, hold + 1)
    col = aj[:, None]
    hit_stop = low[idx, col] <= stop[:, None]
    hit_target = high[idx, col] >= target[:, None]
    never = hold + 1
    first_stop = np.where(hit_stop.any(1), hit_stop.argmax(1), never)
    first_target = np.where(hit_target.any(1), hit_target.argmax(1), never)

    stopped = (first_stop <= first_target) & (first_stop < never)
    took_profit = ~stopped & (first_target < never)
    exit_day = np.where(stopped, ti + first_stop + 1, np.where(took_profit, ti + first_target + 1, ti + hold))
    stop_fill = np.minimum(stop, high[exit_day, aj])
    exit_price = np.where(stopped, stop_fill, np.where(took_profit, target, close[exit_day, aj]))

    entry = close[ti, aj]
    ret = exit_price / entry - 1 - cost
    good = np.isfinite(ret)
    reason = np.where(stopped, "stop", np.where(took_profit, "target", "time"))
    return pd.DataFrame({"ti": ti[good], "aj": aj[good], "ret": ret[good],
                         "exit_day": exit_day[good], "exit_reason": reason[good]})


def plan_arrays(panel, cfg):
    c = panel.close.to_numpy(dtype="float64")
    h = panel.high.to_numpy(dtype="float64") if panel.high is not None else c
    l = panel.low.to_numpy(dtype="float64") if panel.low is not None else c
    a = atr(panel, cfg["trading"]["atr_period"]).to_numpy(dtype="float64")
    return c, h, l, a


def backtest_signals(panel, sig, prm, cfg, arrays=None, mkt_cache=None):
    """Simulate the strategy's plan on every past signal."""
    c, h, l, a = arrays or plan_arrays(panel, cfg)
    ti, aj = np.nonzero(sig.to_numpy())
    entry, atr_v = c[ti, aj], a[ti, aj]
    sma_v = sma(panel.close, prm["target_sma"]).to_numpy()[ti, aj] if prm.get("target_sma") else None
    keep = np.isfinite(entry) & np.isfinite(atr_v) & (atr_v > 0)
    if sma_v is not None:
        sma_v = sma_v[keep]
    ti, aj, entry, atr_v = ti[keep], aj[keep], entry[keep], atr_v[keep]
    stop, target = levels(entry, atr_v, sma_v, prm, cfg)
    cost = cfg[panel.market]["cost_bps"] / 10_000
    trades = simulate(c, h, l, ti, aj, stop, target, prm["hold"], cost)
    trades["mkt"] = market_returns(panel, trades["ti"].to_numpy(), trades["exit_day"].to_numpy(), mkt_cache)
    return trades


def summarize(trades, hold, cfg):
    n = len(trades)
    if n == 0:
        return {"n": 0, "verdict": "too_few"}
    ret = trades["ret"]
    # Bad prints and corporate actions the data missed produce absurd returns; trim the tails.
    lo, hi = ret.quantile([0.005, 0.995])
    ret = ret.clip(lo, hi)
    excess = ret - trades["mkt"]
    wins, losses = ret[ret > 0], ret[ret <= 0]

    # Trades opened in the same few days share most of their market exposure, so the t-stat
    # is computed over block averages by entry day, not over individual overlapping trades.
    per_day = excess.groupby(trades["ti"]).mean()
    blocks = per_day.groupby(per_day.index // hold).mean()
    t = blocks.mean() / (blocks.std(ddof=1) / math.sqrt(len(blocks))) if len(blocks) > 2 else float("nan")
    out = {
        "n": n,
        "win_rate": float((ret > 0).mean()),
        "avg_ret": float(ret.mean()),
        "avg_mkt": float(trades["mkt"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "pct_target": float((trades["exit_reason"] == "target").mean()),
        "pct_stop": float((trades["exit_reason"] == "stop").mean()),
        "pct_time": float((trades["exit_reason"] == "time").mean()),
        # Same per-trade weighting as avg_ret, so "average per trade" and "vs market" add up.
        "excess": float(excess.mean()),
        "t": float(t),
    }
    b = cfg["backtest"]
    if n < b["min_trades"]:
        out["verdict"] = "too_few"
    elif out["excess"] > 0 and t >= b["min_t_stat"]:
        out["verdict"] = "edge"
    elif out["avg_ret"] < 0:
        out["verdict"] = "losing"
    elif out["excess"] < 0 and t <= -b["min_t_stat"]:
        out["verdict"] = "lagging"
    else:
        out["verdict"] = "unproven"
    return out
