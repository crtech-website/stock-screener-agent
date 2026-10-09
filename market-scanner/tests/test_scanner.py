import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from scanner import alerts, backtest, checks, data_crypto, data_stocks, ledger, llm, scan
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

def test_event_stats_numbers():
    dates = [f"2026-01-{d:02d}" for d in range(1, 11)]
    close = pd.DataFrame({"X": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19.0],
                          "Y": [10.0] * 10}, index=dates)
    sig = pd.DataFrame(False, index=dates, columns=close.columns)
    sig.loc["2026-01-01", "X"] = True
    p = Panel("stocks", close, close * 0 + 1e6, close.notna(), pd.Series(dtype=float))
    s = backtest.event_stats(p, sig, 1, cost=0.0)
    assert s["n"] == 1
    assert s["avg_ret"] == pytest.approx(0.1)
    assert s["avg_excess"] == pytest.approx(0.1 - 0.05)  # market = mean(X +10%, Y 0%)


def test_backtest_runs_all_strategies():
    report = backtest.run(random_panel(n_days=320, n_assets=80), CFG)
    assert set(report["strategies"]) == {"oversold_bounce", "connors_rsi2", "ibs_reversion", "trend_breakout", "momentum"}
    assert "Connors" in backtest.format_report(report)


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
        def review(self, q, limit):
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
    ob[0]["llm"] = {"verdict": "structural"}
    picks = scan.pick(cands, CFG)
    assert "A0" not in [c["id"] for c in picks.get("oversold_bounce", [])]


def test_ledger_roundtrip_and_scorecard(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANNER_LEDGER", str(tmp_path))
    p = random_panel(n_days=60, n_assets=5)
    p.close["A0"] = np.linspace(10, 20, 60)
    d = p.close.index[30]
    pick = {"as_of": d, "strategy": "oversold_bounce", "id": "A0", "symbol": "A0", "price": 1.0, "score": 1.0}
    ledger.append("stocks", {"oversold_bounce": [pick]})
    ledger.append("stocks", {"oversold_bounce": [pick]})
    assert len(pd.read_csv(tmp_path / "stocks.csv")) == 1
    card = ledger.scorecard("stocks", p, CFG)
    assert "n=1" in card and "+" in card


def test_alert_format_and_chunking():
    pick = {"strategy": "connors_rsi2", "strategy_label": "Connors RSI(2) pullback", "hold": 5, "symbol": "AAA",
            "name": "Alpha", "price": 12.5, "market_cap": 2.3e9, "change_1d_pct": -3.0, "change_5d_pct": -6.0,
            "change_20d_pct": 4.0, "rsi14": 38, "flags": ["insider buying: 2 trades, $250,000"],
            "headlines": [{"title": "Alpha dips on rate fears", "date": "10-08"}],
            "backtest": {"n": 812, "avg_ret": 0.011, "win_rate": 0.58, "avg_excess": 0.006, "t_excess": 2.4}}
    text = alerts.format_alert("stocks", {"connors_rsi2": [pick]}, True, datetime(2026, 10, 8, 22, 40))
    assert "Connors" in text and "t=2.4" in text and "$2.3B" in text and "insider buying" in text
    parts = alerts.chunks("\n\n".join(["x" * 900] * 5), 1900)
    assert all(len(p) <= 1900 for p in parts) and len(parts) == 3
