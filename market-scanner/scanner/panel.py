from dataclasses import dataclass, field

import pandas as pd

from .indicators import sma


@dataclass
class Panel:
    """Wide daily matrices (dates x assets) shared by live scans, backtests and the scorecard."""

    market: str
    close: pd.DataFrame
    volume: pd.DataFrame
    tradable: pd.DataFrame            # point-in-time liquidity filter, same shape as close
    benchmark: pd.Series              # benchmark closes, for the regime filter
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    market_cap: pd.DataFrame | None = None
    names: dict = field(default_factory=dict)
    symbols: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)      # latest prices, if newer than the last close
    live_at: str | None = None

    @property
    def regime(self):
        """True on days the benchmark closed above its 200-day average."""
        b = self.benchmark.reindex(self.close.index).ffill()
        ok = b > sma(b, 200)
        # Without 200 days of benchmark history there's no regime to judge, so don't block anything.
        return ok.where(sma(b, 200).notna(), True)

    @property
    def last_date(self):
        return self.close.index[-1]
