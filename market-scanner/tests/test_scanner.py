import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from scanner import alerts, backtest, checks, data_crypto, data_stocks, ledger, llm, plan, scan
from scanner.indicators import ibs, rsi
from scanner.panel import Panel
from scanner.strategies import REGISTRY, active_strategies, params_for, signals

CFG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())


# ---------- helpers ----------

def random_panel(n_days=320, n_assets=60, seed=0, market="stocks"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days).strftime("%Y-%m-%d")
    cols = [f"A{i}" for i in range(n_assets)]
    close = pd.DataFrame(50 * np.exp(rng.normal(0.0004, 0.02, (n_days, n_assets)).cumsum(0)), index=dates, columns=cols)
    high = close * (1 + rng.uniform(0, 0.02, close.shape))
    low = close * (1 - rng.uniform(0, 0.02, close.shape))
    volume = pd.DataFrame(rng.integers(200_000, 400_000, close.shape), index=dates, columns=cols).astype(float)
    bench = pd.Series(100 * np.exp(rng.normal(0.0005, 0.01, n_days).cumsum()), index=dates)
    tradable = close.notna()
    return Panel(market, close, volume, tradable, bench, high=high, low=low,
                 names={c: f"Co {c}" for c in cols})


def wilder_reference(prices, period=14):
    gains = losses = None
    out = None
    for i in range(1, len(prices)):
        d = prices[i] - prices[i - 1]
        g, l = max(d, 0), max(-d, 0)
        if gains is None:
            gains, losses = g, l
        else:
            gains += (g - gains) / period
            losses += (l - losses) / period
        out = 100 - 100 / (1 + gains / losses) if losses else 100.0
    return out


# ---------- indicators ----------

def test_rsi_matches_reference_loop():
    prices = list(100 + np.random.default_rng(0).normal(0, 2, 80).cumsum())
    assert rsi(pd.Series(prices)).iloc[-1] == pytest.approx(wilder_reference(prices), rel=1e-9)
    assert rsi(pd.Series(prices), 2).iloc[-1] == pytest.approx(wilder_reference(prices, 2), rel=1e-9)


def test_ibs():
    assert ibs(pd.Series([12.0]), pd.Series([10.0]), pd.Series([10.5])).iloc[0] == pytest.approx(0.25)


# ---------- strategies ----------

def test_strategies_have_no_lookahead():
    """A signal on day t must not change when later days are added."""
    full = random_panel(n_days=320)
    cut = 280
    part = Panel("stocks", full.close.iloc[:cut], full.volume.iloc[:cut], full.tradable.iloc[:cut],
                 full.benchmark.iloc[:cut], high=full.high.iloc[:cut], low=full.low.iloc[:cut])
    for strat, prm in active_strategies(CFG, part):
        a, _ = signals(strat, prm, part)
        b, _ = signals(strat, prm, full)
        pd.testing.assert_frame_equal(a, b.iloc[:cut], obj=strat.name)


def test_each_strategy_fires_on_its_pattern():
    p = random_panel(n_days=320, n_assets=40, seed=4)
    c = p.close
    # A0: steady uptrend then a sharp 2-day crash (oversold bounce)
    c["A0"] = np.linspace(40, 80, 320)
    c.iloc[-2:, 0] = [62, 50]
    # A1: uptrend with a mild 3-day dip (Connors RSI2 above the 200-day)
    c["A1"] = np.linspace(40, 80, 320)
    c.iloc[-3:, 1] = [79.0, 78.0, 77.0]
    # A2: long uptrend, flat for a while, then a fresh high on heavy volume (breakout)
    c["A2"] = np.concatenate([np.linspace(30, 70, 280), np.full(39, 69.0), [75.0]])
    p.volume.iloc[-1, 2] = 2_000_000
    p.high, p.low = c * 1.01, c * 0.99
    fired = {}
    for strat, prm in active_strategies(CFG, p):
        sig, _ = signals(strat, prm, p)
        fired[strat.name] = set(sig.columns[sig.iloc[-1]])
    assert "A0" in fired["oversold_bounce"]
    assert "A1" in fired["connors_rsi2"]
    assert "A2" in fired["trend_breakout"]


def test_momentum_fires_only_on_rebalance_days():
    p = random_panel(n_days=320)
    strat = REGISTRY["momentum"]
    prm = params_for(CFG, "momentum", "stocks")
    sig, _ = signals(strat, prm, p, use_regime=False)
    days = sig.index[sig.any(axis=1)]
    months = pd.to_datetime(pd.Series(days)).dt.to_period("M")
    assert months.is_unique and len(days) > 0


def test_regime_filter_blocks_signals_in_downtrend():
    p = random_panel()
    p.benchmark = pd.Series(np.linspace(200, 100, len(p.close)), index=p.close.index)
    strat = REGISTRY["oversold_bounce"]
    prm = {**params_for(CFG, "oversold_bounce", "stocks"), "rsi_below": 60, "drop_pct": 0}
    on, _ = signals(strat, prm, p, use_regime=True)
    off, _ = signals(strat, prm, p, use_regime=False)
    assert off.iloc[220:].values.any() and not on.iloc[220:].values.any()


# ---------- backtest ----------

def _bars(rows):
    a = np.array(rows, dtype=float)
    return a[:, None]  # (days, 1 asset)


def test_simulate_exits_target_stop_time_and_gap():
    # day: close, high, low. Entry at day 0 close = 100, stop 90, target 110, hold 3.
    c = _bars([100, 104, 108, 109])
    h = _bars([100, 106, 111, 110])
    l = _bars([100, 98, 105, 100])
    t = plan.simulate(c, h, l, np.array([0]), np.array([0]),
                      np.array([90.0]), np.array([110.0]), 3, 0.0)
    assert t["exit_reason"][0] == "target" and t["ret"][0] == pytest.approx(0.10) and t["exit_day"][0] == 2

    # Both touched on day 1: the stop is assumed first.
    h2, l2 = _bars([100, 112, 0, 0]), _bars([100, 89, 0, 0])
    t = plan.simulate(c, h2, l2, np.array([0]), np.array([0]),
                      np.array([90.0]), np.array([110.0]), 3, 0.0)
    assert t["exit_reason"][0] == "stop" and t["ret"][0] == pytest.approx(-0.10)

    # Gap down: the whole day trades below the stop, so the fill is that day's high, not the stop.
    h3, l3 = _bars([100, 85, 0, 0]), _bars([100, 80, 0, 0])
    t = plan.simulate(c, h3, l3, np.array([0]), np.array([0]),
                      np.array([90.0]), np.array([110.0]), 3, 0.0)
    assert t["ret"][0] == pytest.approx(-0.15)

    # Neither: sold at the close after 3 days.
    flat = _bars([100, 101, 102, 103])
    t = plan.simulate(flat, flat, flat, np.array([0]), np.array([0]),
                      np.array([90.0]), np.array([110.0]), 3, 0.001)
    assert t["exit_reason"][0] == "time" and t["ret"][0] == pytest.approx(0.029)

    # Not enough days left to finish the plan: dropped.
    t = plan.simulate(flat, flat, flat, np.array([2]), np.array([0]),
                      np.array([90.0]), np.array([110.0]), 3, 0.0)
    assert t.empty


def test_levels_and_position_size():
    prm = {"stop_atr": 2.0, "target_atr": 4.0}
    stop, target = plan.levels(100.0, 5.0, None, prm, CFG)
    assert (stop, target) == (90.0, 120.0)
    # A huge ATR is capped by max_stop_pct.
    stop, _ = plan.levels(100.0, 40.0, None, prm, CFG)
    assert stop == pytest.approx(100 * (1 - CFG["trading"]["max_stop_pct"] / 100))
    # Connors-style target: the moving average when it's further away than the ATR target.
    _, target = plan.levels(100.0, 2.0, 106.0, {"stop_atr": 2.5, "target_atr": 0.5}, CFG)
    assert target == 106.0
    # 10% stop and 1% risk -> 10% of the account.
    assert plan.position_pct(100.0, 90.0, CFG) == pytest.approx(10.0)
    assert plan.position_pct(100.0, 99.0, CFG) == CFG["trading"]["max_position_pct"]


def test_summarize_verdicts():
    rng = np.random.default_rng(0)
    n = 400
    trades = pd.DataFrame({"ti": np.arange(n), "aj": 0, "ret": 0.02 + rng.normal(0, 0.01, n),
                           "mkt": 0.0, "exit_day": np.arange(n) + 1,
                           "exit_reason": ["target"] * n})
    s = plan.summarize(trades, 5, CFG)
    assert s["verdict"] == "edge" and s["win_rate"] > 0.9
    trades["ret"] = -0.01 + rng.normal(0, 0.01, n)
    assert plan.summarize(trades, 5, CFG)["verdict"] == "losing"
    assert plan.summarize(trades.head(5), 5, CFG)["verdict"] == "too_few"


def test_backtest_runs_all_strategies():
    report = backtest.run(random_panel(n_days=320, n_assets=80), CFG)
    assert set(report["strategies"]) == {"oversold_bounce", "connors_rsi2", "ibs_reversion", "trend_breakout", "momentum"}
    text = backtest.format_report(report)
    assert "Connors" in text and "take-profit" in text
    for e in report["strategies"].values():
        assert e["headline"]["verdict"] in backtest.VERDICT_TEXT


# ---------- stock data ----------

def test_update_bars_backfill_and_split_adjustment(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANNER_CACHE", str(tmp_path))
    prices = {"AAA": 100.0, "SPY": 500.0}
    calls = []

    class Fake:
        split = []

        def get(self, path, params=None):
            calls.append(path)
            if "splits" in path:
                return {"results": self.split}
            day = path.rsplit("/", 1)[-1]
            if day == "2026-09-07":
                return {"resultsCount": 0}
            return {"results": [{"T": t, "h": p, "l": p, "c": p, "v": 1000} for t, p in prices.items()]}

    cfg = {**CFG, "stocks": {**CFG["stocks"], "history_days": 20, "max_backfill_per_run": 100}}
    fake = Fake()
    bars = data_stocks.update_bars(fake, cfg, keep=["AAA"], today=date(2026, 10, 8))
    assert "2026-09-07" not in set(bars["date"])
    n = len(calls)

    # A 2-for-1 split executes on 10-09; the API now reports the new, halved price.
    prices["AAA"] = 50.0
    fake.split = [{"ticker": "AAA", "execution_date": "2026-10-09", "split_from": 1, "split_to": 2}]
    bars = data_stocks.update_bars(fake, cfg, keep=["AAA"], today=date(2026, 10, 9))
    assert len(calls) == n + 2  # one new day plus the splits lookup
    aaa = bars[bars["ticker"] == "AAA"]
    assert aaa["c"].max() == pytest.approx(50.0)
    assert aaa[aaa["date"] == "2026-10-08"]["v"].iloc[0] == pytest.approx(2000)


def test_build_stock_panel_liquidity_and_benchmark():
    rows = []
    for d in pd.bdate_range("2026-01-01", periods=30).strftime("%Y-%m-%d"):
        rows += [{"date": d, "ticker": "LIQ", "h": 21, "l": 19, "c": 20, "v": 1e6},
                 {"date": d, "ticker": "THIN", "h": 21, "l": 19, "c": 20, "v": 100},
                 {"date": d, "ticker": "SPY", "h": 500, "l": 500, "c": 500, "v": 1e8}]
    universe = pd.DataFrame({"ticker": ["LIQ", "THIN"], "name": ["Liquid", "Thin"], "active": True})
    p = data_stocks.build_panel(pd.DataFrame(rows), universe, CFG)
    assert p.tradable["LIQ"].iloc[-1] and not p.tradable["THIN"].iloc[-1]
    assert "SPY" not in p.close and len(p.benchmark) == 30


# ---------- crypto data ----------

def test_crypto_history_backfill_snapshot_and_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANNER_CACHE", str(tmp_path))
    now = datetime(2026, 10, 8, 0, 20, tzinfo=timezone.utc)
    markets = [
        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "current_price": 100.0, "market_cap": 2e12, "total_volume": 5e10},
        {"id": "tether", "symbol": "usdt", "name": "Tether", "current_price": 1.0, "market_cap": 1e11, "total_volume": 5e10},
        {"id": "dust", "symbol": "dust", "name": "Dust", "current_price": 0.1, "market_cap": 1e6, "total_volume": 1e3},
    ]
    chart_calls = []

    class Fake:
        def get(self, path, params=None):
            if path == "/coins/markets":
                return markets if params["page"] == 1 else []
            chart_calls.append(path)
            days = pd.date_range("2026-09-01", "2026-10-07", freq="D", tz="UTC")
            pts = [[int(t.timestamp() * 1000), 90.0] for t in days] + [[int(now.timestamp() * 1000), 99.0]]
            return {"prices": pts, "market_caps": [[t, 1e12] for t, _ in pts], "total_volumes": [[t, 1e10] for t, _ in pts]}

    monkeypatch.setattr(data_crypto, "datetime", SimpleNamespace(now=lambda tz=None: now))
    hist, meta = data_crypto.update_history(Fake(), CFG, now=now)
    btc = hist[hist["id"] == "bitcoin"].set_index("date")["close"]
    # The 00:00 UTC point on 10-07 is the 10-06 close; 10-07's close comes from the snapshot.
    assert "2026-10-08" not in btc.index
    assert btc["2026-10-06"] == 90.0 and btc["2026-10-07"] == 100.0
    assert chart_calls == ["/coins/bitcoin/market_chart"]
    panel = data_crypto.build_panel(hist, meta, CFG)
    assert list(panel.close.columns) == ["bitcoin"] and panel.symbols["bitcoin"] == "BTC"


# ---------- checks ----------

FORM4 = """<?xml version="1.0"?><ownershipDocument>
<nonDerivativeTable>
 <nonDerivativeTransaction><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
  <transactionAmounts><transactionShares><value>1000</value></transactionShares>
  <transactionPricePerShare><value>12.5</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
 <nonDerivativeTransaction><transactionCoding><transactionCode>M</transactionCode></transactionCoding>
  <transactionAmounts><transactionShares><value>99999</value></transactionShares>
  <transactionPricePerShare><value>1</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
</nonDerivativeTable></ownershipDocument>"""


def test_sec_review_flags_and_insider_buys():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    submissions = {"filings": {"recent": {
        "form": ["4", "8-K", "NT 10-Q", "424B5", "10-K"],
        "filingDate": [today] * 4 + ["2001-01-01"],
        "items": ["", "3.01,9.01", "", "", ""],
        "accessionNumber": ["0001-26-000001", "a", "b", "c", "d"],
        "primaryDocument": ["xslF345X05/form4.xml", "x", "x", "x", "x"],
    }}}

    class Fake:
        def get(self, url, params=None):
            if "company_tickers" in url:
                return {"0": {"ticker": "ABC", "cik_str": 123}}
            return submissions

        def get_text(self, url, params=None):
            assert url.endswith("/123/0001260000 01".replace(" ", "") + "/form4.xml")
            return FORM4

    sec = checks.SecChecks()
    sec.client = Fake()
    r = sec.review("ABC")
    assert any("delisting" in s for s in r["severe"]) and any("late filing" in s for s in r["severe"])
    assert r["dilution"] and r["insider_buys"] == 1 and r["insider_buy_usd"] == pytest.approx(12500)


def test_news_flags_from_rss():
    rss = """<rss><channel>
      <item><title>Acme files for Chapter 11 protection - Reuters</title><pubDate>Thu, 08 Oct 2026 10:00:00 GMT</pubDate></item>
      <item><title>Analyst downgrade hits Acme - CNBC</title><pubDate>Wed, 07 Oct 2026 10:00:00 GMT</pubDate></item>
      <item><title>Acme launches new widget - PR</title><pubDate>Tue, 06 Oct 2026 10:00:00 GMT</pubDate></item>
    </channel></rss>"""
    n = checks.NewsChecks()
    n.client = SimpleNamespace(get_text=lambda url, params=None: rss)
    r = n.review('"ACME" stock', 5)
    assert len(r["headlines"]) == 3 and len(r["news_severe"]) == 1 and len(r["news_warnings"]) == 1


def test_review_candidates_excludes_severe(monkeypatch):
    class FakeSec:
        def review(self, t):
            return {"severe": ["8-K 1.03 bankruptcy filing"] if t == "BAD" else [], "warnings": [],
                    "filings": [], "insider_buys": 0, "insider_buy_usd": 0, "insider_sells": 0, "insider_sell_usd": 0}

    class FakeNews:
        def review(self, q, limit, must_mention=()):
            return {"headlines": [], "news_severe": [], "news_warnings": []}

    monkeypatch.setattr(checks, "SecChecks", FakeSec)
    monkeypatch.setattr(checks, "NewsChecks", FakeNews)
    cands = [{"id": t, "symbol": t, "strategy": "s"} for t in ("BAD", "OK", "OK")]
    kept = checks.review_candidates(cands, "stocks", CFG)
    assert [c["symbol"] for c in kept] == ["OK", "OK"]


# ---------- llm ----------

def test_llm_review_parses_and_survives_failures(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = {**CFG, "llm": {**CFG["llm"], "provider": "groq", "calls_per_minute": 6000}}
    replies = iter([
        {"choices": [{"message": {"content": 'ok <verdict>{"verdict": "structural", "confidence": 8, "summary": "s"}</verdict>'}}]},
        {"oops": True},
    ])
    monkeypatch.setattr(llm.RateLimitedClient, "post", lambda self, path, payload: next(replies))
    cands = [{"symbol": "A", "flags": []}, {"symbol": "B", "flags": []}]
    out = llm.review(cands, cfg)
    assert out[0]["llm"]["verdict"] == "structural" and "llm" not in out[1]
    assert "symbol" in llm.build_prompt(cands[0])


# ---------- scan, ledger, alerts ----------

def test_scan_candidates_and_pick():
    p = random_panel(n_days=320, n_assets=40, seed=4)
    p.close["A0"] = np.linspace(40, 80, 320)
    p.close.iloc[-2:, 0] = [62, 50]
    report = backtest.run(p, CFG)
    cands = scan.candidates(p, CFG, report)
    ob = [c for c in cands if c["strategy"] == "oversold_bounce"]
    assert ob and ob[0]["id"] == "A0" and ob[0]["backtest"] is not None
    assert ob[0]["stop"] < ob[0]["entry"] < ob[0]["target"]
    assert "Fell" in ob[0]["reason"] and "rebound" in ob[0]["strategy_text"]
    ob[0]["llm"] = {"verdict": "structural"}
    picks = scan.pick(cands, CFG)
    assert "A0" not in [c["id"] for c in picks.get("oversold_bounce", [])]


def test_ledger_roundtrip_and_scorecard(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANNER_LEDGER", str(tmp_path))
    p = random_panel(n_days=60, n_assets=5)
    p.close["A0"] = np.linspace(10, 20, 60)
    p.high["A0"], p.low["A0"] = p.close["A0"], p.close["A0"]
    d = p.close.index[30]
    entry = float(p.close["A0"].iloc[30])
    pick = {"as_of": d, "strategy": "trend_breakout", "id": "A0", "symbol": "A0",
            "entry": entry, "stop": entry * 0.9, "target": entry * 1.05, "hold": 20}
    ledger.append("crypto", {"trend_breakout": [pick]})
    ledger.append("crypto", {"trend_breakout": [pick]})
    assert len(pd.read_csv(tmp_path / "crypto.csv")) == 1
    report = {"end": d, "strategies": {"trend_breakout": {
        "label": "Trend breakout", "hold": 20, "regime": True,
        "headline": {"verdict": "unproven", "n": 40, "win_rate": 0.5, "avg_win": 0.1, "avg_loss": -0.08,
                     "avg_ret": 0.01, "avg_mkt": 0.005}}}}
    ledger.write_latest("crypto", {"trend_breakout": [{**pick, "strategy_label": "Trend breakout", "price": entry,
                                                      "position_pct": 9.0, "backtest": {"verdict": "unproven"},
                                                      "strategy_text": "Buys strength.", "reason": "New high."}]},
                        report, datetime(2026, 10, 9, 1, 0, tzinfo=timezone.utc), CFG, True, held_back=["Momentum leaders"])
    latest = json.loads((tmp_path / "latest-crypto.json").read_text())
    pk = latest["picks"][0]
    assert pk["limit_buy"] == round(entry, 8) and pk["hold_days"] == 20 and pk["why_picked"] == "New high."
    assert latest["strategies"]["trend_breakout"]["win_rate_pct"] == 50.0
    assert latest["held_back"] == ["Momentum leaders"] and "New York time" in latest["generated_new_york"]
    assert "uptrend" in latest["market_mood"]
    card = ledger.scorecard("crypto", p, CFG)
    assert "1 of 1 alerts have finished" in card and "1 take profit" in card and "Trend breakout" in card


def test_alert_format_and_chunking():
    pick = {"strategy": "connors_rsi2", "strategy_label": "Connors RSI(2) pullback",
            "strategy_text": "Buys a short dip.", "reason": "Dropped 4% over 3 days.", "hold": 5,
            "symbol": "AAA", "name": "Alpha", "price": 50.0, "entry": 50.0, "stop": 45.0, "target": 53.0,
            "position_pct": 10.0, "market_cap": 2.3e9, "change_1d_pct": -3.0, "change_5d_pct": -6.0,
            "change_20d_pct": 4.0, "rsi14": 38, "flags": ["insider buying: 2 trades, $250,000"],
            "headlines": [{"title": "Alpha dips on rate fears", "date": "10-08"}],
            "backtest": {"n": 812, "win_rate": 0.58, "avg_ret": 0.011, "avg_win": 0.04, "avg_loss": -0.03,
                         "excess": 0.006, "t": 2.4, "verdict": "edge"}}
    when = datetime(2026, 10, 9, 3, 49, tzinfo=timezone.utc)
    text = alerts.format_alert("stocks", {"connors_rsi2": [pick]}, True, when, CFG, months=23, as_of="2026-10-08",
                               held_back=["Momentum leaders"])
    assert "Oct 8 market close (4:00 PM New York time)" in text and "in testing: Momentum leaders" in text
    assert "Below 1 is normal" in text
    assert "Thursday, Oct 8, 2026, 11:49 PM New York time" in text
    assert "limit order at $50.00" in text and "stop sell at $45.00 (-10.0%)" in text
    assert "limit sell at $53.00 (+6.0%)" in text and "0.6 to 1" in text
    assert "$100 of a $1,000 account" in text and "BEAT THE MARKET" in text and "HOW TO PLACE" in text
    parts = alerts.chunks("\n\n".join(["x" * 900] * 5), 1900)
    assert all(len(p) <= 1900 for p in parts) and len(parts) == 3


def test_crypto_alert_close_time_live_price_and_combined_risk():
    base = {"strategy": "trend_breakout", "strategy_label": "Trend breakout", "strategy_text": "x", "reason": "y",
            "hold": 20, "name": "Starknet", "price": 0.0679, "entry": 0.0679, "stop": 0.0607, "target": 0.0824,
            "position_pct": 9.0, "market_cap": 5e8, "change_1d_pct": 22.0, "change_5d_pct": 15.0,
            "change_20d_pct": 52.0, "rsi14": 70, "backtest": None}
    a = {**base, "symbol": "STRK", "live_price": 0.0720}
    b = {**base, "symbol": "RAY", "live_price": None}
    when = datetime(2026, 10, 9, 4, 28, tzinfo=timezone.utc)
    text = alerts.format_alert("crypto", {"trend_breakout": [a, b]}, True, when, CFG, as_of="2026-10-08")
    assert "Oct 8 daily close (Oct 8, 8:00 PM New York time)" in text
    assert "Price now: $0.072 (+6.0% since the close)" in text and "Don't chase it" in text
    assert "about 2% of your account is at risk at once" in text


def test_crypto_panel_uses_only_completed_days(monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    days = list(pd.date_range(end=pd.Timestamp(today), periods=5).strftime("%Y-%m-%d"))
    hist = pd.DataFrame({"date": days, "id": "bitcoin", "close": [1.0, 2, 3, 4, 99],
                         "market_cap": 1e12, "volume": 1e10})
    meta = {"bitcoin": {"symbol": "BTC", "name": "Bitcoin"}, "_live": {"prices": {"bitcoin": 99.0}, "at": "x"}}
    p = data_crypto.build_panel(hist, meta, CFG)
    assert p.last_date == days[-2] and p.close["bitcoin"].iloc[-1] == 4.0 and p.live["bitcoin"] == 99.0


def test_market_returns_is_buy_and_hold_and_trimmed():
    dates = [f"2026-01-{d:02d}" for d in range(1, 6)]
    cols = [f"A{i}" for i in range(50)]
    close = pd.DataFrame(10.0, index=dates, columns=cols)
    close.iloc[2:, :] = 11.0           # everything +10% from day 2
    close.iloc[4, 0] = 10_000.0        # one bad print
    p = Panel("crypto", close, close, close.notna(), pd.Series(dtype=float))
    r = plan.market_returns(p, np.array([0, 0, 0]), np.array([1, 4, 1]))
    assert r[0] == pytest.approx(0.0) and r[1] == pytest.approx(0.10, rel=0.05) and r[2] == pytest.approx(0.0)


def test_lagging_verdict():
    rng = np.random.default_rng(1)
    n = 400
    trades = pd.DataFrame({"ti": np.arange(n), "aj": 0, "ret": 0.01 + rng.normal(0, 0.01, n),
                           "mkt": 0.05, "exit_day": np.arange(n) + 1, "exit_reason": ["time"] * n})
    s = plan.summarize(trades, 5, CFG)
    assert s["verdict"] == "lagging" and s["excess"] == pytest.approx(s["avg_ret"] - s["avg_mkt"])


def test_crypto_panel_drops_pegged_tokens_and_bad_prints():
    days = pd.date_range("2026-01-01", periods=60).strftime("%Y-%m-%d")
    rng = np.random.default_rng(2)
    real = 10 * np.exp(rng.normal(0, 0.03, 60).cumsum())
    real[30] = real[29] * 0.2            # one-day bad print
    pegged = 1 + rng.normal(0, 0.001, 60)
    hist = pd.concat([pd.DataFrame({"date": days, "id": "realcoin", "close": real}),
                      pd.DataFrame({"date": days, "id": "loan-token", "close": pegged})]).assign(
        market_cap=1e9, volume=1e8)
    meta = {"realcoin": {"symbol": "REAL", "name": "Real"}, "loan-token": {"symbol": "LOAN", "name": "Loan"}}
    cfg = {**CFG, "crypto": {**CFG["crypto"], "only_symbols_file": None}}
    p = data_crypto.build_panel(hist, meta, cfg)
    assert list(p.close.columns) == ["realcoin"] and np.isnan(p.close["realcoin"].iloc[30])


def test_crypto_universe_limited_to_platform_coins(tmp_path):
    f = tmp_path / "coins.txt"
    f.write_text("# comment\nSTRK\nBTC\n")
    cfg = {**CFG, "crypto": {**CFG["crypto"], "only_symbols_file": str(f)}}
    markets = pd.DataFrame({"id": ["starknet", "strike-fake", "bitcoin", "solana"],
                            "symbol": ["strk", "strk", "btc", "sol"], "market_cap": [5e8, 1e7, 2e12, 5e10]})
    kept = data_crypto.keep_venue_coins(markets, data_crypto.venue_symbols(cfg))
    assert sorted(kept["id"]) == ["bitcoin", "starknet"]
    assert data_crypto.venue_symbols({**CFG, "crypto": {**CFG["crypto"], "only_symbols_file": None}}) is None
    assert "STRK" in data_crypto.venue_symbols(CFG) and len(data_crypto.venue_symbols(CFG)) == 92


def test_crypto_refetches_when_yesterday_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANNER_CACHE", str(tmp_path))
    markets = [{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "current_price": 100.0,
                "market_cap": 2e12, "total_volume": 5e10}]
    stored = pd.DataFrame({"date": pd.date_range("2026-08-01", "2026-10-07").strftime("%Y-%m-%d"),
                           "id": "bitcoin", "close": 1.0, "market_cap": 1e12, "volume": 1e10})
    (tmp_path / "crypto").mkdir()
    stored.to_pickle(tmp_path / "crypto" / "history.pkl")
    calls = []

    class Fake:
        def get(self, path, params=None):
            if path == "/coins/markets":
                return markets if params["page"] == 1 else []
            calls.append(params["days"])
            return {"prices": [], "market_caps": [], "total_volumes": []}

    # Oct 9, 14:00 UTC: Oct 8's close is missing (gap 2) and must be fetched.
    data_crypto.update_history(Fake(), CFG, now=datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc))
    assert calls == [3]
    # Oct 8, 14:00 UTC: Oct 7 is stored (gap 1), nothing to fetch.
    calls.clear()
    data_crypto.update_history(Fake(), CFG, now=datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc))
    assert calls == []


def test_stock_panel_survives_duplicate_rows():
    rows = []
    for d in pd.bdate_range("2026-01-01", periods=25).strftime("%Y-%m-%d"):
        rows += [{"date": d, "ticker": "AAA", "h": 11, "l": 9, "c": 10, "v": 1e6},
                 {"date": d, "ticker": "SPY", "h": 500, "l": 500, "c": 500, "v": 1e8}]
    rows.append({"date": rows[0]["date"], "ticker": "AAA", "h": 11, "l": 9, "c": 10.5, "v": 1e6})
    universe = pd.DataFrame({"ticker": ["AAA"], "name": ["A"], "active": True})
    p = data_stocks.build_panel(pd.DataFrame(rows), universe, CFG)
    assert p.close["AAA"].iloc[0] == 10.5


def test_loading_lists_strategies_short_of_history():
    from scanner.strategies import loading
    short = random_panel(n_days=150)
    names = " ".join(loading(CFG, short))
    assert "Connors" in names and "Momentum" in names and "Oversold" not in names
    assert loading(CFG, random_panel(n_days=320)) == []
    text = alerts.format_alert("stocks", {}, True, datetime(2026, 10, 9, 22, 40, tzinfo=timezone.utc), CFG,
                               as_of="2026-10-09", loading=loading(CFG, short))
    assert "Still loading price history" in text


def test_one_failing_coin_does_not_sink_the_crypto_scan(tmp_path, monkeypatch):
    import requests
    monkeypatch.setenv("SCANNER_CACHE", str(tmp_path))
    markets = [{"id": c, "symbol": s, "name": n, "current_price": 1.5, "market_cap": 2e9, "total_volume": 5e8}
               for c, s, n in (("bitcoin", "btc", "Bitcoin"), ("ethereum", "eth", "Ethereum"))]

    class Fake:
        def get(self, path, params=None):
            if path == "/coins/markets":
                return markets if params["page"] == 1 else []
            if "ethereum" in path:
                raise requests.HTTPError("429 Too Many Requests")
            days = pd.date_range("2026-08-01", "2026-10-08", freq="D", tz="UTC")
            pts = [[int(t.timestamp() * 1000), 100.0] for t in days]
            return {"prices": pts, "market_caps": [[t, 1e12] for t, _ in pts], "total_volumes": [[t, 1e10] for t, _ in pts]}

    hist, _ = data_crypto.update_history(Fake(), CFG, now=datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc))
    assert set(hist["id"]) == {"bitcoin"}


def test_checks_and_alerts_survive_outages(monkeypatch):
    import requests

    class DownSec:
        def review(self, t):
            raise requests.ConnectionError("SEC down")

    monkeypatch.setattr(checks, "SecChecks", DownSec)
    kept = checks.review_candidates([{"id": "A", "symbol": "A", "strategy": "s", "name_only": "A Co"}], "stocks", CFG)
    assert kept and "background checks unavailable tonight" in kept[0]["flags"]

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")

    def boom(*a, **k):
        raise requests.ConnectionError("telegram down")

    monkeypatch.setattr(alerts.requests, "post", boom)
    assert alerts.send("hello") is False
