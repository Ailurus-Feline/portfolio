"""Pairs-trading research package."""

from __future__ import annotations

from .backtest import BacktestResult, run_pairs_backtest
from .cointegration import EngleGrangerResult, engle_granger, engle_granger_from_panel
from .config import PAIRS, PROJECT_ROOT
from .data import align_pair, load_all_pairs
from .johansen import JohansenResult, johansen_from_panel, johansen_pair
from .ou_process import OUFit, fit_ou_ar1, zscore_series
from .signals import ZStarScan, scan_z_grid
from .walkforward import TrainTestBundle, run_train_test_backtest

__version__ = "0.1.0"

__all__ = [
    "PAIRS",
    "PROJECT_ROOT",
    "align_pair",
    "load_all_pairs",
    "engle_granger",
    "engle_granger_from_panel",
    "EngleGrangerResult",
    "johansen_pair",
    "johansen_from_panel",
    "JohansenResult",
    "fit_ou_ar1",
    "zscore_series",
    "OUFit",
    "scan_z_grid",
    "ZStarScan",
    "run_pairs_backtest",
    "BacktestResult",
    "run_train_test_backtest",
    "TrainTestBundle",
    "__version__",
]
