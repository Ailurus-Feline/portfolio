"""v4 alpha combination pipeline: equal-weight, IC-weighted, Ridge, Decision Tree."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from src.alpha_combination import (
    CombinationSpec,
    build_composite_signal,
    build_v4_alpha_matrix,
    spec_cache_key,
)
from src.alpha_signals import build_breadth_alpha_signal, build_market_base
from src.backtest import (
    load_backtest_config,
    performance_summary_table,
    run_backtest,
    summarize_performance,
)
from src.config import (
    BACKTEST_CONFIG_FILE,
    V3_CHAMPION_ALPHA,
    V3_CHAMPION_BUY_Z,
    V3_CHAMPION_DIR,
    V3_CHAMPION_EMA,
    V3_CHAMPION_SELL_Z,
    V4_ALPHA_MATRIX,
    V4_BEST_CONFIG,
    V4_BUY_Z_GRID,
    V4_COMBINATION_RESULTS,
    V4_IC_HORIZONS,
    V4_METHODS_DIR,
    V4_OUTPUT_DIR,
    V4_REPORT,
    V4_RIDGE_LAMBDAS,
    V4_SELL_Z_GRID,
    V4_SENSITIVITY_DIR,
    V4_SENSITIVITY_RESULTS,
    V4_TREE_MAX_DEPTHS,
    V4_TREE_MIN_SAMPLES_LEAF,
)
from src.plot import plot_equity_curve
from src.strategy_rules import make_baseline_target


def _load_champion_daily(config) -> pd.DataFrame:
    champion_path = V3_CHAMPION_DIR / "backtest_daily.csv"
    if champion_path.exists():
        return pd.read_csv(champion_path, parse_dates=["date"])

    base = build_market_base()
    signals = build_breadth_alpha_signal(base, V3_CHAMPION_ALPHA, ema_span=V3_CHAMPION_EMA)
    rule_fn = make_baseline_target(V3_CHAMPION_BUY_Z, V3_CHAMPION_SELL_Z)
    return run_backtest(signals, config, rule_fn, initial_position=0.0)


def _segment_return(daily: pd.DataFrame, split: str | None) -> float:
    summary = summarize_performance(daily, split)
    return summary.total_return


def _evaluate_combo(
    signal_frame: pd.DataFrame,
    config,
    champion_daily: pd.DataFrame,
    *,
    method: str,
    buy_z: float,
    sell_z: float,
    ridge_lambda: float | None,
    ic_horizon: int | None,
    tree_max_depth: int | None,
    tree_min_samples_leaf: int | None,
    weights_label: str,
) -> dict[str, float | int | str]:
    daily = run_backtest(
        signal_frame,
        config,
        make_baseline_target(buy_z, sell_z),
        initial_position=0.0,
    )
    full = summarize_performance(daily)
    valid = summarize_performance(daily, "valid")
    final = summarize_performance(daily, "final")

    champ_all = _segment_return(champion_daily, None)
    champ_valid = _segment_return(champion_daily, "valid")
    champ_final = _segment_return(champion_daily, "final")

    if method == "ridge":
        config_id = f"{method}__lam{ridge_lambda:g}__buy{buy_z:g}__sell{sell_z:g}"
    elif method == "ic_weighted":
        config_id = f"{method}__h{ic_horizon}__buy{buy_z:g}__sell{sell_z:g}"
    elif method == "decision_tree":
        config_id = (
            f"{method}__d{tree_max_depth}__leaf{tree_min_samples_leaf}"
            f"__buy{buy_z:g}__sell{sell_z:g}"
        )
    else:
        config_id = f"{method}__buy{buy_z:g}__sell{sell_z:g}"

    return {
        "config_id": config_id,
        "method": method,
        "ridge_lambda": ridge_lambda if ridge_lambda is not None else "",
        "ic_horizon": ic_horizon if ic_horizon is not None else "",
        "tree_max_depth": tree_max_depth if tree_max_depth is not None else "",
        "tree_min_samples_leaf": tree_min_samples_leaf if tree_min_samples_leaf is not None else "",
        "buy_z": buy_z,
        "sell_z": sell_z,
        "weights": weights_label,
        "total_return_all": full.total_return,
        "excess_all": full.excess_return_vs_benchmark,
        "excess_vs_champion_all": full.total_return - champ_all,
        "total_return_valid": valid.total_return,
        "excess_valid": valid.excess_return_vs_benchmark,
        "excess_vs_champion_valid": valid.total_return - champ_valid,
        "total_return_final": final.total_return,
        "excess_final": final.excess_return_vs_benchmark,
        "excess_vs_champion_final": final.total_return - champ_final,
        "max_drawdown_all": full.max_drawdown,
        "sharpe_all": full.sharpe,
        "n_trades_all": full.n_trades,
        "avg_position_all": full.avg_position,
        "beats_bh_all": int(full.excess_return_vs_benchmark > 0),
        "beats_bh_final": int(final.excess_return_vs_benchmark > 0),
        "beats_champion_all": int(full.total_return > champ_all),
        "beats_champion_final": int(final.total_return > champ_final),
        "_daily": daily,
    }


def _method_specs() -> list[CombinationSpec]:
    specs = [CombinationSpec(method="equal_weight")]
    for horizon in V4_IC_HORIZONS:
        specs.append(CombinationSpec(method="ic_weighted", ic_horizon=horizon))
    for lam in V4_RIDGE_LAMBDAS:
        specs.append(CombinationSpec(method="ridge", ridge_lambda=lam, ic_horizon=20))
    for depth, leaf in itertools.product(V4_TREE_MAX_DEPTHS, V4_TREE_MIN_SAMPLES_LEAF):
        specs.append(
            CombinationSpec(
                method="decision_tree",
                ic_horizon=20,
                tree_max_depth=depth,
                tree_min_samples_leaf=leaf,
            )
        )
    return specs


def _weights_label(spec: CombinationSpec) -> str:
    if spec.weights is None:
        return ""
    return ",".join(f"{weight:.4f}" for weight in spec.weights)


def _write_best_config(path: Path, row: pd.Series) -> None:
    lines = [
        "# v4 best config (selected on validation excess vs v3 champion)",
        f"config_id: {row['config_id']}",
        f"method: {row['method']}",
        f"ridge_lambda: {row['ridge_lambda']}",
        f"ic_horizon: {row['ic_horizon']}",
        f"tree_max_depth: {row.get('tree_max_depth', '')}",
        f"tree_min_samples_leaf: {row.get('tree_min_samples_leaf', '')}",
        f"buy_z: {row['buy_z']}",
        f"sell_z: {row['sell_z']}",
        f"weights: {row['weights']}",
        "",
        f"excess_all: {row['excess_all']}",
        f"excess_vs_champion_all: {row['excess_vs_champion_all']}",
        f"excess_valid: {row['excess_valid']}",
        f"excess_vs_champion_valid: {row['excess_vs_champion_valid']}",
        f"excess_final: {row['excess_final']}",
        f"excess_vs_champion_final: {row['excess_vs_champion_final']}",
        f"beats_champion_all: {int(row['beats_champion_all'])}",
        f"beats_champion_final: {int(row['beats_champion_final'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, results: pd.DataFrame, champion_row: pd.Series, best: pd.Series) -> None:
    lines = [
        "Electronics Sector Sentiment — v4 Combination Report",
        "=" * 56,
        "",
        "Methods: equal-weight, IC-weighted, Ridge, Decision Tree (train fit)",
        "Selection metric: validation excess vs v3 champion",
        "Final split: evaluate once, no re-tuning",
        "",
        "v3 champion reference:",
        f"  return_all={champion_row['total_return_all'] * 100:.1f}%, "
        f"excess_all={champion_row['excess_all'] * 100:.1f}%",
        f"  return_final={champion_row['total_return_final'] * 100:.1f}%, "
        f"excess_final={champion_row['excess_final'] * 100:.1f}%",
        "",
        "Best v4 combination (valid-selected):",
        f"  {best['config_id']}",
        f"  all: return={best['total_return_all'] * 100:.1f}%, "
        f"excess={best['excess_all'] * 100:.1f}%, "
        f"vs champion={best['excess_vs_champion_all'] * 100:.1f}%",
        f"  final: return={best['total_return_final'] * 100:.1f}%, "
        f"excess={best['excess_final'] * 100:.1f}%, "
        f"vs champion={best['excess_vs_champion_final'] * 100:.1f}%",
        "",
        "Top 5 by validation excess vs champion:",
    ]
    top = results.sort_values("excess_vs_champion_valid", ascending=False).head(5)
    for _, row in top.iterrows():
        lines.append(
            f"  {row['config_id']}: valid_vs_champion={row['excess_vs_champion_valid'] * 100:.1f}%, "
            f"final_vs_champion={row['excess_vs_champion_final'] * 100:.1f}%"
        )

    lines.extend(["", "Best per method (valid-selected):"])
    for method in ("equal_weight", "ic_weighted", "ridge", "decision_tree"):
        subset = results.loc[results["method"] == method]
        if subset.empty:
            continue
        row = subset.sort_values("excess_vs_champion_valid", ascending=False).iloc[0]
        lines.append(
            f"  {method}: {row['config_id']} | all={row['total_return_all'] * 100:.1f}% "
            f"vs champion={row['excess_vs_champion_all'] * 100:.1f}%"
        )

    beats = results.loc[(results["beats_champion_all"] == 1) & (results["beats_champion_final"] == 1)]
    lines.extend(
        [
            "",
            "Configs beating v3 champion on BOTH full sample and final:",
        ]
    )
    if beats.empty:
        lines.append("  (none) — deploy v3 champion (see FINAL_STRATEGY.yaml)")
    else:
        for _, row in beats.sort_values("excess_vs_champion_all", ascending=False).iterrows():
            lines.append(f"  {row['config_id']}: vs champion all={row['excess_vs_champion_all'] * 100:.1f}%")

    lines.extend(
        [
            "",
            f"Alpha matrix: {V4_ALPHA_MATRIX.name}",
            f"Full grid: {V4_COMBINATION_RESULTS.name}",
            f"Sensitivity: {V4_SENSITIVITY_RESULTS.name}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _archive_method_outputs(method: str, daily: pd.DataFrame, row: pd.Series) -> None:
    slug = str(row["config_id"]).replace("/", "_").replace("+", "_")
    out_dir = V4_METHODS_DIR / method.replace("_", "-") / slug
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily.to_csv(out_dir / "backtest_daily.csv", index=False)
    performance_summary_table(daily).to_csv(out_dir / "performance_summary.csv", index=False)
    plot_equity_curve(
        daily,
        fig_dir / "equity_curve.png",
        title=f"v4 {row['config_id']} vs Buy & Hold",
    )


def _replay_best(
    combo_only: pd.DataFrame,
    signal_cache: dict[str, pd.DataFrame],
    config,
    method: str,
) -> tuple[pd.Series, pd.DataFrame]:
    subset = combo_only.loc[combo_only["method"] == method]
    row = subset.sort_values(["excess_vs_champion_valid", "excess_valid"], ascending=False).iloc[0]
    key = _row_cache_key(row)
    daily = run_backtest(
        signal_cache[key],
        config,
        make_baseline_target(float(row["buy_z"]), float(row["sell_z"])),
        initial_position=0.0,
    )
    return row, daily


def _row_cache_key(row: pd.Series) -> str:
    ridge_lambda = row["ridge_lambda"] if row["ridge_lambda"] != "" else None
    ic_horizon = int(row["ic_horizon"]) if row["ic_horizon"] != "" else None
    tree_max_depth = int(row["tree_max_depth"]) if row.get("tree_max_depth", "") != "" else None
    tree_min_samples_leaf = (
        int(row["tree_min_samples_leaf"]) if row.get("tree_min_samples_leaf", "") != "" else None
    )
    return spec_cache_key(
        CombinationSpec(
            method=row["method"],
            ridge_lambda=ridge_lambda,
            ic_horizon=ic_horizon,
            tree_max_depth=tree_max_depth,
            tree_min_samples_leaf=tree_min_samples_leaf,
        )
    )


def run_v4_pipeline() -> pd.DataFrame:
    """Run v4 combination grid, select on validation, archive outputs."""
    V4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    V4_SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)

    alpha_matrix = build_v4_alpha_matrix()
    alpha_matrix.to_csv(V4_ALPHA_MATRIX, index=False)

    config = load_backtest_config(BACKTEST_CONFIG_FILE)
    champion_daily = _load_champion_daily(config)
    champion_full = summarize_performance(champion_daily)
    champion_valid = summarize_performance(champion_daily, "valid")
    champion_final = summarize_performance(champion_daily, "final")

    champion_row = {
        "config_id": "v3_champion_reference",
        "method": "v3_champion",
        "ridge_lambda": "",
        "ic_horizon": "",
        "tree_max_depth": "",
        "tree_min_samples_leaf": "",
        "buy_z": V3_CHAMPION_BUY_Z,
        "sell_z": V3_CHAMPION_SELL_Z,
        "weights": V3_CHAMPION_ALPHA,
        "total_return_all": champion_full.total_return,
        "excess_all": champion_full.excess_return_vs_benchmark,
        "excess_vs_champion_all": 0.0,
        "total_return_valid": champion_valid.total_return,
        "excess_valid": champion_valid.excess_return_vs_benchmark,
        "excess_vs_champion_valid": 0.0,
        "total_return_final": champion_final.total_return,
        "excess_final": champion_final.excess_return_vs_benchmark,
        "excess_vs_champion_final": 0.0,
        "max_drawdown_all": champion_full.max_drawdown,
        "sharpe_all": champion_full.sharpe,
        "n_trades_all": champion_full.n_trades,
        "avg_position_all": champion_full.avg_position,
        "beats_bh_all": int(champion_full.excess_return_vs_benchmark > 0),
        "beats_bh_final": int(champion_final.excess_return_vs_benchmark > 0),
        "beats_champion_all": 0,
        "beats_champion_final": 0,
    }

    rows: list[dict[str, float | int | str]] = [champion_row]
    best_daily: pd.DataFrame | None = None
    signal_cache: dict[str, pd.DataFrame] = {}

    for spec in _method_specs():
        signal_frame, fitted = build_composite_signal(alpha_matrix, spec)
        cache_key = spec_cache_key(fitted)
        signal_cache[cache_key] = signal_frame
        weights_label = _weights_label(fitted)

        for buy_z, sell_z in itertools.product(V4_BUY_Z_GRID, V4_SELL_Z_GRID):
            if buy_z >= sell_z:
                continue
            result = _evaluate_combo(
                signal_frame,
                config,
                champion_daily,
                method=fitted.method,
                buy_z=buy_z,
                sell_z=sell_z,
                ridge_lambda=fitted.ridge_lambda,
                ic_horizon=fitted.ic_horizon,
                tree_max_depth=fitted.tree_max_depth,
                tree_min_samples_leaf=fitted.tree_min_samples_leaf,
                weights_label=weights_label,
            )
            result.pop("_daily", None)
            rows.append(result)

    results = pd.DataFrame(rows)
    combo_only = results.loc[results["method"] != "v3_champion"].copy()
    best = combo_only.sort_values(
        ["excess_vs_champion_valid", "excess_valid"],
        ascending=False,
    ).iloc[0]

    best_key = _row_cache_key(best)
    best_signal = signal_cache[best_key]
    _ = run_backtest(
        best_signal,
        config,
        make_baseline_target(float(best["buy_z"]), float(best["sell_z"])),
        initial_position=0.0,
    )

    export_cols = [col for col in results.columns]
    results[export_cols].to_csv(V4_COMBINATION_RESULTS, index=False)
    results[export_cols].to_csv(V4_SENSITIVITY_RESULTS, index=False)
    _write_best_config(V4_BEST_CONFIG, best)
    _write_report(V4_REPORT, combo_only, pd.Series(champion_row), best)

    for method in ("equal_weight", "ic_weighted", "ridge", "decision_tree"):
        method_row, method_daily = _replay_best(combo_only, signal_cache, config, method)
        _archive_method_outputs(method, method_daily, method_row)

    print(f"[v3 champion] return={champion_full.total_return * 100:.1f}% excess={champion_full.excess_return_vs_benchmark * 100:.1f}%")
    print(
        f"[v4 best] {best['config_id']} return={best['total_return_all'] * 100:.1f}% "
        f"vs champion={best['excess_vs_champion_all'] * 100:.1f}% "
        f"final vs champion={best['excess_vs_champion_final'] * 100:.1f}%"
    )
    print(f"\nWrote {V4_ALPHA_MATRIX}")
    print(f"Wrote {V4_COMBINATION_RESULTS}")
    print(f"Wrote {V4_REPORT}")
    return results


def main() -> None:
    run_v4_pipeline()


if __name__ == "__main__":
    main()
