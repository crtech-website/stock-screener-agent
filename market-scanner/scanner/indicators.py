"""Indicators that work on a Series or a wide DataFrame (dates x assets) alike."""


def sma(x, n):
    return x.rolling(n, min_periods=n).mean()


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing (alpha = 1/period) is what TradingView and most brokers show as RSI.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def ibs(high, low, close):
    """Internal bar strength: where the close sits in the day's range, 0 = at the low."""
    rng = high - low
    return ((close - low) / rng).where(rng > 0)


def pct_change(close, bars):
    return (close / close.shift(bars) - 1) * 100
