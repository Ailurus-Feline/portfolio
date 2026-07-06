"""Position rules for v3 long-only backtests."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from src.config import (
    CONTINUOUS_COLD_Z,
    CONTINUOUS_HOT_Z,
    HYSTERESIS_BUY_Z,
    HYSTERESIS_SELL_Z,
    Z_OVERCOOL,
    Z_OVERHEAT,
)

if TYPE_CHECKING:
    pass

PositionRule = Callable[[float, float], float]


def make_baseline_target(buy_z: float, sell_z: float) -> PositionRule:
    """Overcooled -> long; overheated -> flat; otherwise maintain. Starts flat."""

    def target(z: float, current_position: float) -> float:
        if z < buy_z:
            return 1.0
        if z > sell_z:
            return 0.0
        return current_position

    return target


def make_hysteresis_target(buy_z: float, sell_z: float) -> PositionRule:
    """Enter below buy_z; exit above sell_z; otherwise maintain."""

    def target(z: float, current_position: float) -> float:
        if current_position >= 0.5:
            return 0.0 if z > sell_z else 1.0
        return 1.0 if z < buy_z else 0.0

    return target


def make_default_long_target(buy_z: float, sell_z: float) -> PositionRule:
    """Default fully invested; exit on overheated; re-enter on overcooled."""

    def target(z: float, current_position: float) -> float:
        if z > sell_z:
            return 0.0
        if z < buy_z:
            return 1.0
        return current_position

    return target


def make_continuous_target(cold_z: float, hot_z: float) -> PositionRule:
    """Scale exposure when cold; flat when hot; maintain otherwise."""

    def target(z: float, current_position: float) -> float:
        if z > hot_z:
            return 0.0
        if z <= cold_z:
            depth = (-z - abs(cold_z)) / 2.0
            return float(np.clip(depth, 0.0, 1.0))
        return current_position

    return target


def make_dual_signal_target(
    buy_z: float,
    sell_z: float,
    mode: str,
) -> Callable[[float, float, float, float], float]:
    """
    Two-alpha position rule.

    mode:
      and_entry_or_exit — both cold to enter, either hot to exit
      min_position — min(target1, target2) from independent baseline rules
      avg_position — average of independent baseline targets
    """
    single = make_baseline_target(buy_z, sell_z)

    def target(z1: float, z2: float, current_position: float) -> float:
        if mode == "and_entry_or_exit":
            if z1 > sell_z or z2 > sell_z:
                return 0.0
            if z1 < buy_z and z2 < buy_z:
                return 1.0
            return current_position
        t1 = single(z1, current_position)
        t2 = single(z2, current_position)
        if mode == "min_position":
            return min(t1, t2)
        if mode == "avg_position":
            return (t1 + t2) / 2.0
        raise ValueError(f"Unknown dual mode: {mode!r}")

    return target


def baseline_target(z: float, current_position: float) -> float:
    """
    Overcooled -> full long; overheated -> flat; otherwise maintain.

    After buying on overcooled, hold through neutral until overheated.
    """
    if z < Z_OVERCOOL:
        return 1.0
    if z > Z_OVERHEAT:
        return 0.0
    return current_position


def hysteresis_target(z: float, current_position: float) -> float:
    """Enter long only below a colder threshold; exit only above a sell threshold."""
    if current_position >= 0.5:
        return 0.0 if z > HYSTERESIS_SELL_Z else 1.0
    return 1.0 if z < HYSTERESIS_BUY_Z else 0.0


def continuous_target(z: float, current_position: float) -> float:
    """
    Scale exposure by sentiment depth when cold; flat when hot; maintain otherwise.

    For z <= COLD_Z: position = clip((-z - |COLD_Z|) / 2, 0, 1)
    For z >= HOT_Z: flat
    Between bands: maintain previous position
    """
    if z > CONTINUOUS_HOT_Z:
        return 0.0
    if z <= CONTINUOUS_COLD_Z:
        depth = (-z - abs(CONTINUOUS_COLD_Z)) / 2.0
        return float(np.clip(depth, 0.0, 1.0))
    return current_position


RULES: dict[str, PositionRule] = {
    "baseline": baseline_target,
    "hysteresis": hysteresis_target,
    "continuous": continuous_target,
}

RULE_DESCRIPTIONS: dict[str, str] = {
    "baseline": (
        f"Buy when z<{Z_OVERCOOL:g}, sell when z>{Z_OVERHEAT:g}, "
        "maintain between (hold after buy until overheated)"
    ),
    "hysteresis": (
        f"Flat->long when z<{HYSTERESIS_BUY_Z:g}; "
        f"long->flat when z>{HYSTERESIS_SELL_Z:g}; else maintain"
    ),
    "continuous": (
        f"Scale long when z<={CONTINUOUS_COLD_Z:g}, flat when z>{CONTINUOUS_HOT_Z:g}, else maintain"
    ),
}
