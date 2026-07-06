"""v3 parameter grid search — maximize chance of beating buy & hold."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from src.alpha_signals import (
    ALPHA_NAMES,
    build_breadth_alpha_signal,
    build_market_base,
    load_sentiment_composite,
)
from src.backtest import (
    load_backtest_config,
    run_backtest,
    run_dual_backtest,
    summarize_performance,
)
from src.config import (
    BACKTEST_CONFIG_FILE,
    V2_BEST_EMA,
    V3_OPT_DIR,
    V3_OPT_FIGURES_DIR,
    V3_OPT_GRID_RESULTS,
    V3_OPT_REPORT,
    V3_OPT_TOP_CONFIGS,
)
from src.plot import plot_equity_curve
from src.strategy_rules import (
    make_baseline_target,
    make_continuous_target,
    make_default_long_target,
    make_dual_signal_target,
    make_hysteresis_target,
)

BUY_Z_GRID = (-2.0, -1.5, -1.2, -1.0, -0.8, -0.5, -0.3)
SELL_Z_GRID = (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
COLD_Z_GRID = (-1.5, -1.2, -1.0, -0.8, -0.5)
HOT_Z_GRID = (0.5, 0.8, 1.0, 1.2, 1.5)
EMA_GRID = (40, 60, 90)

BREADTH_ALPHAS = ("new_high_low_net", "above_ma", "positive_return", "advance_decline", "index_momentum")
DUAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("advance_decline", "positive_return"),
    ("advance_decline", "sentiment_composite"),
    ("positive_return", "new_high_low_net"),
    ("advance_decline", "index_momentum"),
)
DUAL_MODES = ("and_entry_or_exit", "min_position", "avg_position")


def _threshold_pairs() -> list[tuple[float, float]]:
    return [(buy_z, sell_z) for buy_z in BUY_Z_GRID for sell_z in SELL_Z_GRID if buy_z < sell_z]


def _build_signal_catalog() -> dict[tuple[str, int], pd.DataFrame]:
    """Key = (alpha_name, ema_span). sentiment_composite only uses v2 EMA."""
    catalog: dict[tuple[str, int], pd.DataFrame] = {}
    catalog[("sentiment_composite", V2_BEST_EMA)] = load_sentiment_composite()

    base = build_market_base()
    for alpha in BREADTH_ALPHAS:
        for ema_span in EMA_GRID:
            catalog[(alpha, ema_span)] = build_breadth_alpha_signal(base, alpha, ema_span=ema_span)
    return catalog


def _merge_dual_signals(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.rename(columns={"signal_z": "signal_z"}).merge(
        right.loc[:, ["date", "signal_z"]].rename(columns={"signal_z": "signal_z_2"}),
        on="date",
        how="inner",
    )
    keep = ["date", "split", "index_close", "signal_z", "signal_z_2"]
    return merged.loc[:, keep].dropna().reset_index(drop=True)


def _evaluate(
    daily: pd.DataFrame,
    *,
    alpha: str,
    rule_family: str,
    ema_span: int,
    buy_z: float | None,
    sell_z: float | None,
    cold_z: float | None,
    hot_z: float | None,
    alpha_2: str | None = None,
    dual_mode: str | None = None,
    ema_span_2: int | None = None,
) -> dict[str, float | int | str]:
    full = summarize_performance(daily)
    valid = summarize_performance(daily, "valid")
    final = summarize_performance(daily, "final")
    train = summarize_performance(daily, "train")

    config_id = (
        f"{alpha}__ema{ema_span}__{rule_family}"
        f"__buy{buy_z:g}__sell{sell_z:g}"
        if buy_z is not None and sell_z is not None
        else f"{alpha}__ema{ema_span}__{rule_family}__cold{cold_z:g}__hot{hot_z:g}"
    )
    if alpha_2 is not None:
        config_id = (
            f"{alpha}+{alpha_2}__ema{ema_span}+{ema_span_2}"
            f"__{dual_mode}__buy{buy_z:g}__sell{sell_z:g}"
        )

    return {
        "config_id": config_id,
        "alpha": alpha,
        "alpha_2": alpha_2 or "",
        "rule_family": rule_family,
        "dual_mode": dual_mode or "",
        "ema_span": ema_span,
        "ema_span_2": ema_span_2 if ema_span_2 is not None else "",
        "buy_z": buy_z if buy_z is not None else "",
        "sell_z": sell_z if sell_z is not None else "",
        "cold_z": cold_z if cold_z is not None else "",
        "hot_z": hot_z if hot_z is not None else "",
        "total_return_all": full.total_return,
        "excess_all": full.excess_return_vs_benchmark,
        "total_return_valid": valid.total_return,
        "excess_valid": valid.excess_return_vs_benchmark,
        "total_return_final": final.total_return,
        "excess_final": final.excess_return_vs_benchmark,
        "total_return_train": train.total_return,
        "excess_train": train.excess_return_vs_benchmark,
        "max_drawdown_all": full.max_drawdown,
        "sharpe_all": full.sharpe,
        "n_trades_all": full.n_trades,
        "avg_position_all": full.avg_position,
        "beats_bh_all": int(full.excess_return_vs_benchmark > 0),
        "beats_bh_valid": int(valid.excess_return_vs_benchmark > 0),
        "beats_bh_final": int(final.excess_return_vs_benchmark > 0),
    }


def _run_single_alpha_grid(
    catalog: dict[tuple[str, int], pd.DataFrame],
    config,
    rows: list[dict[str, float | int | str]],
) -> None:
    threshold_pairs = _threshold_pairs()

    for alpha in ALPHA_NAMES:
        ema_values = (V2_BEST_EMA,) if alpha == "sentiment_composite" else EMA_GRID
        for ema_span in ema_values:
            signals = catalog[(alpha, ema_span)]

            for buy_z, sell_z in threshold_pairs:
                for rule_family, rule_fn, initial_pos in (
                    ("overcooled_entry", make_baseline_target(buy_z, sell_z), 0.0),
                    ("default_long", make_default_long_target(buy_z, sell_z), 1.0),
                    ("hysteresis", make_hysteresis_target(buy_z, sell_z), 0.0),
                ):
                    daily = run_backtest(signals, config, rule_fn, initial_position=initial_pos)
                    rows.append(
                        _evaluate(
                            daily,
                            alpha=alpha,
                            rule_family=rule_family,
                            ema_span=ema_span,
                            buy_z=buy_z,
                            sell_z=sell_z,
                            cold_z=None,
                            hot_z=None,
                        )
                    )

            for cold_z, hot_z in itertools.product(COLD_Z_GRID, HOT_Z_GRID):
                rule_fn = make_continuous_target(cold_z, hot_z)
                daily = run_backtest(signals, config, rule_fn, initial_position=0.0)
                rows.append(
                    _evaluate(
                        daily,
                        alpha=alpha,
                        rule_family="continuous",
                        ema_span=ema_span,
                        buy_z=None,
                        sell_z=None,
                        cold_z=cold_z,
                        hot_z=hot_z,
                    )
                )


def _run_dual_alpha_grid(
    catalog: dict[tuple[str, int], pd.DataFrame],
    config,
    rows: list[dict[str, float | int | str]],
) -> None:
    threshold_pairs = _threshold_pairs()

    for alpha_a, alpha_b in DUAL_PAIRS:
        ema_a_values = (V2_BEST_EMA,) if alpha_a == "sentiment_composite" else (60, 90)
        ema_b_values = (V2_BEST_EMA,) if alpha_b == "sentiment_composite" else (60, 90)

        for ema_a, ema_b in itertools.product(ema_a_values, ema_b_values):
            dual_signals = _merge_dual_signals(catalog[(alpha_a, ema_a)], catalog[(alpha_b, ema_b)])

            for buy_z, sell_z in threshold_pairs:
                for dual_mode in DUAL_MODES:
                    rule_fn = make_dual_signal_target(buy_z, sell_z, dual_mode)
                    daily = run_dual_backtest(dual_signals, config, rule_fn, initial_position=0.0)
                    rows.append(
                        _evaluate(
                            daily,
                            alpha=alpha_a,
                            rule_family="dual",
                            ema_span=ema_a,
                            buy_z=buy_z,
                            sell_z=sell_z,
                            cold_z=None,
                            hot_z=None,
                            alpha_2=alpha_b,
                            dual_mode=dual_mode,
                            ema_span_2=ema_b,
                        )
                    )


def _write_report(path: Path, results: pd.DataFrame, benchmark_return: float) -> None:
    lines = [
        "Electronics Sector Sentiment — v3 Optimization Report",
        "=" * 56,
        "",
        f"Grid size: {len(results)} configurations",
        f"Benchmark (buy & hold) full-sample return: {benchmark_return * 100:.2f}%",
        f"Configurations beating B&H (full sample): {int(results['beats_bh_all'].sum())}",
        f"Configurations beating B&H (valid only): {int(results['beats_bh_valid'].sum())}",
        f"Configurations beating B&H (final only): {int(results['beats_bh_final'].sum())}",
        "",
        "Top 10 by full-sample excess vs benchmark:",
    ]
    top_all = results.sort_values("excess_all", ascending=False).head(10)
    for _, row in top_all.iterrows():
        lines.append(
            f"  {row['config_id']}: "
            f"return={row['total_return_all'] * 100:.1f}%, "
            f"excess={row['excess_all'] * 100:.1f}%, "
            f"trades={int(row['n_trades_all'])}, "
            f"avg_pos={row['avg_position_all']:.2f}"
        )

    lines.extend(["", "Top 10 by valid-split excess (selection metric):"])
    top_valid = results.sort_values("excess_valid", ascending=False).head(10)
    for _, row in top_valid.iterrows():
        lines.append(
            f"  {row['config_id']}: "
            f"valid excess={row['excess_valid'] * 100:.1f}%, "
            f"full excess={row['excess_all'] * 100:.1f}%"
        )

    winners = results.loc[results["beats_bh_all"] == 1].sort_values("excess_all", ascending=False)
    lines.extend(["", "All configurations beating B&H on full sample:"])
    if winners.empty:
        lines.append("  (none)")
    else:
        for _, row in winners.iterrows():
            lines.append(
                f"  {row['config_id']}: "
                f"return={row['total_return_all'] * 100:.1f}%, "
                f"excess={row['excess_all'] * 100:.1f}%"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _archive_top_configs(
    results: pd.DataFrame,
    catalog: dict[tuple[str, int], pd.DataFrame],
    config,
    top_n: int = 10,
    figure_n: int = 5,
) -> None:
    """Save top-N summary table and equity curves for the best few configs only."""
    V3_OPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = results.sort_values("excess_all", ascending=False).head(top_n)
    top.to_csv(V3_OPT_TOP_CONFIGS, index=False)

    for _, row in top.head(figure_n).iterrows():
        safe_id = str(row["config_id"]).replace("/", "_")
        if row["rule_family"] == "dual":
            dual_signals = _merge_dual_signals(
                catalog[(row["alpha"], int(row["ema_span"]))],
                catalog[(row["alpha_2"], int(row["ema_span_2"]))],
            )
            rule_fn = make_dual_signal_target(float(row["buy_z"]), float(row["sell_z"]), row["dual_mode"])
            daily = run_dual_backtest(dual_signals, config, rule_fn, initial_position=0.0)
        elif row["rule_family"] == "continuous":
            signals = catalog[(row["alpha"], int(row["ema_span"]))]
            rule_fn = make_continuous_target(float(row["cold_z"]), float(row["hot_z"]))
            daily = run_backtest(signals, config, rule_fn, initial_position=0.0)
        else:
            signals = catalog[(row["alpha"], int(row["ema_span"]))]
            buy_z, sell_z = float(row["buy_z"]), float(row["sell_z"])
            initial_pos = 1.0 if row["rule_family"] == "default_long" else 0.0
            makers = {
                "overcooled_entry": make_baseline_target,
                "default_long": make_default_long_target,
                "hysteresis": make_hysteresis_target,
            }
            daily = run_backtest(
                signals,
                config,
                makers[row["rule_family"]](buy_z, sell_z),
                initial_position=initial_pos,
            )

        plot_equity_curve(
            daily,
            V3_OPT_FIGURES_DIR / f"{safe_id}.png",
            title=f"{row['config_id']} vs Buy & Hold",
        )


def run_v3_optimization() -> pd.DataFrame:
    """Run the full v3 parameter grid and write outputs."""
    V3_OPT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = _build_signal_catalog()
    config = load_backtest_config(BACKTEST_CONFIG_FILE)

    rows: list[dict[str, float | int | str]] = []
    print("Running single-alpha grid...")
    _run_single_alpha_grid(catalog, config, rows)
    print(f"  completed {len(rows)} runs")

    print("Running dual-alpha grid...")
    start = len(rows)
    _run_dual_alpha_grid(catalog, config, rows)
    print(f"  completed {len(rows) - start} dual runs")

    results = pd.DataFrame(rows).sort_values("excess_all", ascending=False).reset_index(drop=True)
    results.to_csv(V3_OPT_GRID_RESULTS, index=False)

    benchmark_return = float(
        catalog[("advance_decline", 60)]["index_close"].iloc[-1]
        / catalog[("advance_decline", 60)]["index_close"].iloc[0]
        - 1.0
    )
    _write_report(V3_OPT_REPORT, results, benchmark_return)
    _archive_top_configs(results, catalog, config)

    n_beats = int(results["beats_bh_all"].sum())
    best = results.iloc[0]
    print(f"\nWrote {V3_OPT_GRID_RESULTS} ({len(results)} configs)")
    print(f"Beat B&H on full sample: {n_beats}")
    print(
        f"Best full-sample excess: {best['config_id']} "
        f"({best['excess_all'] * 100:.1f}%)"
    )
    return results


def main() -> None:
    run_v3_optimization()


if __name__ == "__main__":
    main()
