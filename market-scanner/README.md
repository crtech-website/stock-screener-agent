# Market Scanner

A free daily scanner for US stocks and crypto. It screens the whole market with five documented strategies, backtests each one on two years of your own cached data, checks the finalists against SEC filings, insider trades, news and DeFi data, and sends the top picks to Telegram or Discord. Every alerted pick is logged and graded later, so you can see whether the alerts are any good.

Running cost: $0. You need free accounts with Massive (stocks data), CoinGecko (crypto data) and Telegram or Discord. An LLM review is optional and runs on a free tier.

```
Tier 1: whole market, pandas            Tier 2: finalists only                     Alert
stocks: every US common stock           SEC: bankruptcy, delisting, restatement,   per strategy:
crypto: every coin above $50M cap         late filings, offerings (dilution)       backtest stats
-> liquidity filter                     Form 4: insider open-market buys/sells     top 3 picks
-> 5 strategies + market regime         Google News: last 7 days of headlines      flags + headline
-> backtest of each strategy            DefiLlama: TVL trend (crypto)              -> Telegram / Discord
-> ranked candidates                    optional free LLM verdict                  -> ledger (graded weekly)
```

## Strategies

| Strategy | Rule | Holding period | Source |
|---|---|---|---|
| Oversold bounce | RSI(14) < 30 and down 15%+ over 2 days | 5 bars | The original idea from this project |
| Connors RSI(2) pullback | Above 200-day average and RSI(2) < 5 | 5 bars | Connors & Alvarez, *Short Term Trading Strategies That Work* |
| IBS reversion (stocks) | Above 200-day average, down day, close in bottom 15% of the day's range | 3 bars | Internal bar strength studies on US equities |
| Trend breakout | Stocks: Minervini trend template (price > 50 > 150 > 200-day, 200-day rising) and first close at a 52-week high on 1.5x volume. Crypto: same with 20/50/100-day averages and a 55-day high (Turtle System 2) | 20 bars | Minervini; Dennis' Turtle rules |
| Momentum leaders | Stocks: top 5% by 12-month return skipping the last month, picked on the first trading day of each month. Crypto: top 10% by 90-day return skipping the last week, picked weekly | 21 / 7 bars | Jegadeesh & Titman (1993); Liu & Tsyvinski (2021) |

The market regime filter (benchmark above its 200-day average; SPY for stocks, Bitcoin for crypto) is on by default for the two trend strategies and off for the three mean-reversion ones. The backtest reports every strategy both ways, so you can see whether the filter helps and flip it in `config.yaml`.

Each alert section starts with that strategy's backtest line. This one is real, from a test run on 25 large coins:

```
== Connors RSI(2) pullback ==
   Backtest (5-bar hold): avg -0.9%, win 37%, vs market -2.7% (t=-1.2, n=41)
```

That line is the point of the backtest: on that sample, buying crypto RSI(2) dips lost money and trailed the market.

"vs market" is the average return of the signals minus the average return of all liquid assets over the same days, after costs. A t-stat under 2 means the edge could easily be noise. Set `backtest.require_edge: true` to silence strategies that don't clear `min_t_stat`.

## The trade plan in each alert

Every pick comes with four orders, and the backtest tests exactly these orders:

1. **Buy** with a limit order at the last close. Cancel it if it hasn't filled by the next close.
2. **Take profit**: a limit sell above the buy price.
3. **Stop loss**: a stop sell below the buy price.
4. **Time limit**: sell at the market if neither price is hit within the strategy's holding period.

The distances come from ATR, the average daily price move over the last 14 days, so a jumpy coin gets wider levels than a calm stock. Each strategy's settings are in `config.yaml` (`stop_atr`, `target_atr`, `hold`). The stop is never more than `max_stop_pct` (25%) below the buy price.

Position size is set so that hitting the stop loses about 1% of your account (`risk_per_trade_pct`), capped at 20% of the account in one trade.

Strategies whose plan lost money in the backtest are held back from alerts (`skip_losing_strategies`), and the alert says which ones.

Backtest limits to know: crypto data has daily closes only, so stops are checked against closes. A real stop order can be triggered by an intraday dip that recovers by the close, so live crypto stop-outs will happen somewhat more often than the backtest shows. When a day touches both the stop and the target, the backtest assumes the stop hit first.

## Robinhood

The scanner is set up for trading on Robinhood:

- **Costs:** the backtest charges what Robinhood really costs. That's about 1.9% per crypto round trip, because Robinhood buys about 0.95% above the market price and sells about 0.95% below it, measured Oct 9, 2026. Stocks are charged 0.1%.
- **Coins:** crypto is limited to the 92 coins in `robinhood_crypto.txt`. Edit that file when Robinhood adds or removes a coin, or delete it to scan everything.
- **Buy price:** the buy limit is set about 1% above the close for crypto, so the order can actually fill at Robinhood's buy price.
- **Picks file:** each scan saves tonight's picks to `ledger/latest-crypto.json` and `ledger/latest-stocks.json`.
- **Market scanner agent (Claude scheduled task, 7:56 AM New York time, or Run now any time):** reads those files and checks each pick against live Robinhood prices. It skips picks that aren't on Robinhood, whose price has already run more than 2% past the buy price, that are already below the stop, or whose buy/sell gap is too wide. For the rest, it adds them to a Robinhood watchlist called "Scanner picks" and sets price alerts at the stop and take-profit. When a plan's time limit passes it turns those alerts off (it never deletes anything). It never places orders, and it never touches your other alerts or watchlists.

## Setup

1. Push this folder to a GitHub repo. A public repo gets unlimited Actions minutes; a private one gets 2,000 a month, about 10x what this uses. Note that the `ledger/` folder of picks is visible in a public repo.
2. Get the free keys:
   - Massive (formerly Polygon.io), Basic plan: massive.com
   - CoinGecko Demo: coingecko.com/en/developers/dashboard
   - Telegram: create a bot with @BotFather for the token, send it one message, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat id. Or use a Discord channel webhook.
3. Settings > Secrets and variables > Actions, add:

| Secret | Needed for |
|---|---|
| `MASSIVE_API_KEY` | stocks |
| `SEC_USER_AGENT` | stocks. Your name and email, e.g. `Jane Doe jane@mail.com`. SEC blocks requests without it. |
| `COINGECKO_API_KEY` | crypto |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, or `DISCORD_WEBHOOK_URL` | alerts |
| `GROQ_API_KEY` or `GEMINI_API_KEY` or `OPENROUTER_API_KEY` | only if you turn on the LLM review |

4. Actions tab > Stock scan > Run workflow. Do this three or four times on day one: each run backfills about 120 trading days (about 25 minutes, because the free plan allows 5 calls a minute), and the 200-day and 52-week strategies switch on once there's enough history. After that, a daily run needs one data call and takes a few minutes. Crypto backfills 100 coins per run and fills in after two or three runs.

Schedules: stocks at 08:37 UTC on weekday mornings (the free Massive plan only publishes a day's closing prices the next morning), crypto at 00:45 UTC daily (CoinGecko publishes each day's official close about 35 minutes after midnight UTC), the Robinhood agent at 7:56 AM New York time, before the market opens, scorecard on Saturdays. Until enough history is loaded, the alert lists which strategies are still waiting for data. Each run also uploads the full backtest report and every candidate's details as a run artifact.

## Optional LLM review

Set `llm.provider` in `config.yaml` to `groq`, `gemini` or `openrouter` and add that key as a secret. The model only sees the facts the code already gathered (prices, filings, insider trades, headlines), so no paid web search is involved. It writes a two-sentence summary per pick and can drop a pick it calls structurally broken. It never reorders picks, because the order is what the backtest measured.

The default model is Groq's free `openai/gpt-oss-120b` (about 1,000 requests and 200k tokens a day). For Gemini, copy a current Flash model id from Google AI Studio into `llm.model`. Free-tier limits change often, so check the provider's console.

## Commands

```bash
pip install -r requirements.txt
export MASSIVE_API_KEY=...  COINGECKO_API_KEY=...  SEC_USER_AGENT="Name you@mail.com"

python -m scanner scan crypto --no-alert     # full run, prints the alert instead of sending it
python -m scanner scan stocks --no-checks    # skip SEC/news/LLM
python -m scanner backtest stocks            # report from cached data, no API calls
python -m scanner scorecard crypto           # grade past alerts from ledger/crypto.csv
python -m pytest -q                          # offline tests
```

## Free-tier usage

| Service | This project uses | Free limit |
|---|---|---|
| Massive | ~1 call/day plus a one-time ~500-call backfill | 5 calls/min, 2 years of daily history |
| CoinGecko | ~5 calls/day plus a one-time ~250-call backfill | 10,000 calls/month |
| SEC EDGAR | ~15 calls per candidate | 10 requests/second |
| GitHub Actions | ~200 min/month after the backfill | unlimited (public) / 2,000 min (private) |
| Groq (optional) | up to 12 requests per run | ~1,000 requests/day |

## What the backtest can and can't tell you

- Entry is the signal day's close and exit the close N bars later. Real fills will be a bit worse; `cost_bps` covers a round trip (20 bps stocks, 40 bps crypto by default).
- Delisted stocks are included, which removes most survivorship bias, but a stock that stops trading before the exit drops out of the sample instead of counting as a total loss. Mean-reversion results are slightly flattered by that.
- Two years is one or two market regimes. A strategy that only worked in one of them will look better than it is.
- Testing many settings on the same two years and keeping the best one overfits. Change parameters sparingly and trust the ledger scorecard over the backtest once it has a few months of data.

Crypto data is provided by CoinGecko (their free plan requires this attribution).
