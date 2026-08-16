"""Data download, cleaning, and alignment for ETF pairs.

Design notes
------------
* Prices are cached under ``data/raw`` so re-runs do not hammer Yahoo.
* Pair panels are *inner-joined* on the trading calendar: both legs must
  print a valid Close on the same date (no forward-fill across holidays
  of only one venue — these are all US-listed ETFs, so this is mild).
* We work with *adjusted* closes from yfinance (``Auto Adjust=True``)
  so splits/dividends do not invent fake cointegration breaks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .config import DEFAULT_END, DEFAULT_START, PROJECT_ROOT


def raw_cache_path(ticker: str) -> Path:
    """Return the on-disk parquet path for a single ticker's OHLCV history."""
    safe = ticker.replace("/", "_").upper()
    return PROJECT_ROOT / "data" / "raw" / f"{safe}.parquet"


def processed_pair_path(y_ticker: str, x_ticker: str) -> Path:
    """Return the on-disk parquet path for an aligned two-leg panel."""
    name = f"{y_ticker.upper()}_{x_ticker.upper()}_panel.parquet"
    return PROJECT_ROOT / "data" / "processed" / name


def download_price_history(
    ticker: str,
    *,
    start: str = DEFAULT_START,
    end: str | None = DEFAULT_END,
    force: bool = False,
) -> pd.DataFrame:
    """Download (or load-cached) daily OHLCV for ``ticker``.

    Parameters
    ----------
    ticker:
        Yahoo Finance symbol, e.g. ``\"EWA\"``.
    start, end:
        Inclusive ISO dates. ``end=None`` means \"through the latest bar\".
    force:
        If True, ignore any existing cache and re-download.

    Returns
    -------
    DataFrame indexed by timezone-naive timestamps with columns
    ``Open, High, Low, Close, Volume`` (adjusted).
    """
    cache = raw_cache_path(ticker)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not force:
        # Fast path: local parquet beats a network round-trip.
        frame = pd.read_parquet(cache)
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return frame.sort_index()

    # yfinance returns a MultiIndex column layout for some versions when
    # a single ticker is requested; flatten defensively.
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"No price data returned for {ticker!r} ({start} → {end}).")

    if isinstance(raw.columns, pd.MultiIndex):
        # Keep the field level (Open/High/...) and drop the ticker level.
        raw.columns = raw.columns.get_level_values(0)

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    frame = raw.loc[:, keep].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index().dropna(how="all")

    frame.to_parquet(cache)
    return frame


def load_close_series(
    ticker: str,
    *,
    start: str = DEFAULT_START,
    end: str | None = DEFAULT_END,
    force: bool = False,
) -> pd.Series:
    """Convenience wrapper: adjusted Close as a named Series."""
    hist = download_price_history(ticker, start=start, end=end, force=force)
    close = hist["Close"].astype(float).rename(ticker.upper())
    return close.dropna()


def align_pair(
    y_ticker: str,
    x_ticker: str,
    *,
    start: str = DEFAULT_START,
    end: str | None = DEFAULT_END,
    force: bool = False,
    persist: bool = True,
) -> pd.DataFrame:
    """Build an inner-joined price panel for one pair.

    Columns
    -------
    ``y``, ``x``:
        Adjusted closes for the dependent / independent legs.
    ``y_ret``, ``x_ret``:
        Simple one-day returns (useful later for P&L; not used in EG levels).

    The Engle–Granger regression is run on *levels* (``y`` on ``x``), not
    returns — returns would destroy the I(1) / cointegration structure.
    """
    y = load_close_series(y_ticker, start=start, end=end, force=force)
    x = load_close_series(x_ticker, start=start, end=end, force=force)

    panel = pd.concat(
        {
            "y": y.rename("y"),
            "x": x.rename("x"),
        },
        axis=1,
        join="inner",
    ).dropna()

    if panel.shape[0] < 252:
        # Rough sanity floor: less than one liquid year is rarely enough
        # for EG critical values to be meaningful in a project report.
        raise ValueError(
            f"Aligned panel for {y_ticker}/{x_ticker} has only {panel.shape[0]} rows; "
            "check tickers or widen the sample."
        )

    panel["y_ret"] = panel["y"].pct_change()
    panel["x_ret"] = panel["x"].pct_change()

    # Attach metadata as attrs (pandas-friendly; survives parquet round-trip
    # only partially, so we also write a sidecar comment in the filename).
    panel.attrs["y_ticker"] = y_ticker.upper()
    panel.attrs["x_ticker"] = x_ticker.upper()
    panel.attrs["start"] = str(panel.index.min().date())
    panel.attrs["end"] = str(panel.index.max().date())

    if persist:
        out = processed_pair_path(y_ticker, x_ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out)

    return panel


def load_all_pairs(
    pairs: Iterable[tuple[str, str, str, str]] | None = None,
    *,
    start: str = DEFAULT_START,
    end: str | None = DEFAULT_END,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download and align every configured pair.

    Returns
    -------
    dict keyed by ``\"Y_X\"`` (e..g. ``\"EWA_EWC\"``) → aligned panel.
    """
    from .config import PAIRS

    selected = list(pairs) if pairs is not None else list(PAIRS)
    out: dict[str, pd.DataFrame] = {}
    for y_ticker, x_ticker, _role, _story in selected:
        key = f"{y_ticker.upper()}_{x_ticker.upper()}"
        out[key] = align_pair(
            y_ticker,
            x_ticker,
            start=start,
            end=end,
            force=force,
            persist=True,
        )
    return out
