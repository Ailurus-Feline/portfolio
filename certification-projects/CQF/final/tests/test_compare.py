"""Smoke tests for comparison / stress-test helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_pairs.compare import cost_sensitivity, z_grid_is_oos


def _panel(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    x = 40 + np.cumsum(rng.normal(scale=0.3, size=n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.85 * e[t - 1] + rng.normal(scale=0.2)
    y = 1.0 + 0.6 * x + e
    panel = pd.DataFrame({"y": y, "x": x}, index=idx)
    panel.attrs["y_ticker"] = "Y"
    panel.attrs["x_ticker"] = "X"
    return panel


def test_z_grid_has_train_and_test():
    tab = z_grid_is_oos(_panel(), grid=[1.0, 1.5], cost_bps=0.0)
    assert set(tab["segment"]) == {"train", "test"}
    assert tab["Z"].nunique() == 2


def test_cost_sensitivity_monotonic_on_full_if_trades():
    tab = cost_sensitivity(_panel(), costs=(0.0, 20.0))
    full = tab[tab["segment"] == "full"].set_index("cost_bps")
    # Higher costs cannot raise full-sample total return.
    assert full.loc[20.0, "total_return"] <= full.loc[0.0, "total_return"] + 1e-9
