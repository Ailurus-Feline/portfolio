"""Pairs-trading research package."""

from __future__ import annotations

from .cointegration import EngleGrangerResult, engle_granger, engle_granger_from_panel
from .config import PAIRS, PROJECT_ROOT
from .data import align_pair, load_all_pairs

__version__ = "0.1.0"

__all__ = [
    "PAIRS",
    "PROJECT_ROOT",
    "align_pair",
    "load_all_pairs",
    "engle_granger",
    "engle_granger_from_panel",
    "EngleGrangerResult",
    "__version__",
]
