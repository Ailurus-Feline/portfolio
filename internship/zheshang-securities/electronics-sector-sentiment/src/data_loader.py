"""Load and align raw CSV inputs for the sentiment pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    CALENDAR_FILE,
    HISTORY_FILE,
    INDEX_FILE,
    MIN_HISTORY_DAYS,
    PRICE_FILE_PATTERN,
    RAW_DATA_DIR,
    TRADING_STATUS_OK,
)


def _parse_numeric(series: pd.Series) -> pd.Series:
    """Coerce Wind-exported numeric strings (commas allowed) to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def load_trading_calendar(path: Path = CALENDAR_FILE) -> pd.DatetimeIndex:
    """Return sorted trading dates."""
    calendar = pd.read_csv(path, parse_dates=["date"])
    return pd.DatetimeIndex(calendar["date"].sort_values().unique())


def load_constituents_history(path: Path = HISTORY_FILE) -> pd.DataFrame:
    """Load constituent membership periods."""
    history = pd.read_csv(path, dtype=str).fillna("")
    history["in_date"] = _parse_dates(history["in_date"])
    history["out_date"] = _parse_dates(history["out_date"])
    return history


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed Wind export date formats into normalized timestamps."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series.astype(str), format="mixed", errors="coerce")


def load_index(path: Path = INDEX_FILE) -> pd.DataFrame:
    """Load Shenwan electronics index levels."""
    index_df = pd.read_csv(path)
    date_col = index_df.columns[0]
    index_df = index_df.rename(columns={date_col: "date"})
    index_df["date"] = _parse_dates(index_df["date"])
    index_df["close"] = _parse_numeric(index_df["close"])
    return index_df.loc[:, ["date", "close"]].sort_values("date").reset_index(drop=True)


def load_prices_long(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Concatenate split daily price files into one long table."""
    files = sorted(raw_dir.glob(PRICE_FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching {PRICE_FILE_PATTERN} in {raw_dir}")

    frames: list[pd.DataFrame] = []
    usecols = ["date", "symbol", "close", "is_trading"]
    for file_path in files:
        chunk = pd.read_csv(file_path, usecols=usecols, dtype={"date": str, "symbol": str, "is_trading": str})
        chunk["date"] = _parse_dates(chunk["date"])
        chunk["close"] = _parse_numeric(chunk["close"])
        frames.append(chunk)

    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_membership_matrix(
    dates: pd.DatetimeIndex,
    history: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    """Boolean matrix: True if a symbol belongs to the sector on that date."""
    membership = pd.DataFrame(False, index=dates, columns=symbols, dtype=bool)
    start = dates.min()
    end = dates.max()

    for row in history.itertuples(index=False):
        in_date = row.in_date if pd.notna(row.in_date) else start
        out_date = row.out_date if pd.notna(row.out_date) else end
        if row.symbol not in membership.columns:
            continue
        mask = (membership.index >= in_date) & (membership.index <= out_date)
        membership.loc[mask, row.symbol] = True

    return membership


def build_price_panels(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pivot long prices into wide panels and derive the daily validity mask.

    Returns
    -------
    close_panel : close prices, indexed by date
    trading_panel : raw trading-status strings
    valid_panel : True when a stock counts toward breadth on that day
    """
    symbols = sorted(prices["symbol"].unique())
    close_panel = prices.pivot(index="date", columns="symbol", values="close")
    trading_panel = prices.pivot(index="date", columns="symbol", values="is_trading")

    close_panel = close_panel.reindex(calendar)
    trading_panel = trading_panel.reindex(calendar)

    membership = build_membership_matrix(calendar, history, symbols)
    membership = membership.reindex(columns=close_panel.columns, fill_value=False)

    has_price = close_panel.notna() & (close_panel > 0)
    is_trading = trading_panel.eq(TRADING_STATUS_OK)
    enough_history = close_panel.rolling(MIN_HISTORY_DAYS, min_periods=MIN_HISTORY_DAYS).count() >= MIN_HISTORY_DAYS

    valid_panel = membership & has_price & is_trading & enough_history
    return close_panel, trading_panel, valid_panel


def load_market_data() -> dict[str, object]:
    """Convenience loader returning all objects needed by the v1 pipeline."""
    calendar = load_trading_calendar()
    history = load_constituents_history()
    index_df = load_index()
    prices = load_prices_long()
    close_panel, trading_panel, valid_panel = build_price_panels(prices, calendar, history)

    return {
        "calendar": calendar,
        "history": history,
        "index": index_df,
        "prices": prices,
        "close_panel": close_panel,
        "trading_panel": trading_panel,
        "valid_panel": valid_panel,
    }
