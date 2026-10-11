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


def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def macd(close, fast=12, slow=26, signal=9):
    """MACD line, signal line and histogram, with the standard 12/26/9 settings."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close, n=20, k=2.0):
    """Middle band (n-bar average) and the bands k standard deviations above and below it."""
    mid = sma(close, n)
    # Population standard deviation, as TradingView and most charting tools draw the bands.
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd
