import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import alerts, backtest, data_crypto, data_stocks, ledger, scan

log = logging.getLogger("scanner")
DATA = {"stocks": data_stocks, "crypto": data_crypto}


def cmd_scan(args, cfg, out):
    data = DATA[args.market]
    panel, client = data.load(cfg)
    now = datetime.now(timezone.utc)

    report = backtest.run(panel, cfg)
    (out / f"{args.market}-backtest.json").write_text(json.dumps(report, indent=1, default=str))
    (out / f"{args.market}-backtest.txt").write_text(backtest.format_report(report))

    cands = scan.candidates(panel, cfg, report)
    if args.market == "stocks" and cands:
        caps = data_stocks.market_caps(client, sorted({c["id"] for c in cands}))
        for c in cands:
            c["market_cap"] = caps.get(c["id"])
        floor = cfg["stocks"]["min_market_cap"]
        cands = [c for c in cands if c["market_cap"] is None or c["market_cap"] >= floor]

    if not args.no_checks:
        from .checks import review_candidates
        from .llm import review as llm_review

        cands = review_candidates(cands, args.market, cfg)
        # Only likely finalists go to the LLM (one spare per strategy in case one is vetoed),
        # so a small free quota is spread across every strategy instead of used up on the first.
        finalists = [c for group in scan.pick(cands, cfg, extra=1).values() for c in group]
        llm_review(finalists, cfg)

    picks = scan.pick(cands, cfg)
    (out / f"{args.market}-candidates.json").write_text(json.dumps(cands, indent=1, default=str))
    months = round(len(panel.close) / (21 if args.market == "stocks" else 30.4))
    text = alerts.format_alert(args.market, picks, bool(panel.regime.iloc[-1]), now, cfg,
                               months=months, as_of=panel.last_date, held_back=scan.held_back(report, cfg))
    print(text)

    if not args.no_ledger:
        ledger.append(args.market, picks)
    if not args.no_alert and (any(picks.values()) or cfg["alerts"]["send_when_empty"]):
        alerts.send(text)


def cmd_backtest(args, cfg, out):
    panel, _ = DATA[args.market].load(cfg, refresh=args.refresh)
    report = backtest.run(panel, cfg)
    text = backtest.format_report(report)
    (out / f"{args.market}-backtest.txt").write_text(text)
    print(text)


def cmd_scorecard(args, cfg, out):
    panel, _ = DATA[args.market].load(cfg, refresh=False)
    text = ledger.scorecard(args.market, panel, cfg)
    print(text)
    if args.send:
        alerts.send(text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="scanner", description="Free two-tier market scanner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="update data, screen, check, alert")
    p.add_argument("market", choices=DATA)
    p.add_argument("--no-alert", action="store_true", help="print the alert instead of sending it")
    p.add_argument("--no-checks", action="store_true", help="skip SEC/news/LLM checks")
    p.add_argument("--no-ledger", action="store_true", help="don't record picks in the ledger")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("backtest", help="score every strategy on cached history")
    p.add_argument("market", choices=DATA)
    p.add_argument("--refresh", action="store_true", help="update data first")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("scorecard", help="grade past alerts from the ledger")
    p.add_argument("market", choices=DATA)
    p.add_argument("--send", action="store_true")
    p.set_defaults(func=cmd_scorecard)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.func(args, cfg, out)


if __name__ == "__main__":
    main()
