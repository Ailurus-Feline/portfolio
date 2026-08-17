"""Tests for time-split train/test pairs validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.walkforward import run_train_test_backtest, time_split_index


def _synthetic_panel(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    x = 50 + np.cumsum(rng.normal(scale=0.5, size=n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.9 * e[t - 1] + rng.normal(scale=0.2)
    y = 2.0 + 0.5 * x + e
    panel = pd.DataFrame({"y": y, "x": x}, index=idx)
    panel.attrs["y_ticker"] = "Y"
    panel.attrs["x_ticker"] = "X"
    return panel


def test_time_split_index_respects_frac():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    split = time_split_index(idx, train_frac=0.7)
    assert split == idx[70]


def test_train_test_runs_and_freezes_beta():
    bundle = run_train_test_backtest(_synthetic_panel(), train_frac=0.7, cost_bps=0.0)
    assert bundle.train.beta == bundle.test.beta == bundle.beta
    assert bundle.test.equity.index.min() >= bundle.split_date
    assert bundle.train.equity.index.max() < bundle.split_date
    assert not bundle.z_scan_table.empty
