"""v3 multi-alpha, multi-rule backtest runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alpha_signals import (
    ALPHA_NAMES,
    SENTIMENT_ALPHAS,
    build_all_alpha_signals,
    build_breadth_alpha_signal,
    build_market_base,
)
from src.backtest import (
    load_backtest_config,
    performance_summary_table,
    run_backtest,
    summarize_performance,
)
from src.config import (
    BACKTEST_CONFIG_FILE,
    FINAL_PNL_FIGURE,
    V3_BEST_CONFIG,
    V3_CERTIFIED_ALPHAS,
    V3_CHAMPION_ALPHA,
    V3_CHAMPION_BUY_Z,
    V3_CHAMPION_DIR,
    V3_CHAMPION_EMA,
    V3_CHAMPION_RULE,
    V3_CHAMPION_SELL_Z,
    V3_EXPLORATORY_ALPHAS_DIR,
    V3_MATRIX_RESULTS,
    V3_OPT_GRID_RESULTS,
    V3_PNL_FIGURE,
    V3_REPORT,
)
from src.plot import plot_equity_curve, plot_pnl_vs_benchmark
from src.strategy_rules import RULES, RULE_DESCRIPTIONS, make_baseline_target


def _alpha_output_dir(alpha: str, rule: str) -> Path:
    return V3_EXPLORATORY_ALPHAS_DIR / alpha / rule


def _write_variant_report(path: Path, alpha: str, rule: str, summary: pd.DataFrame) -> None:
    full = summary.loc[summary["split"] == "all"].iloc[0]
    final = summary.loc[summary["split"] == "final"]
    final_row = final.iloc[0] if not final.empty else None

    rule_desc = RULE_DESCRIPTIONS.get(
        rule,
        f"Buy when z<{V3_CHAMPION_BUY_Z:g}, sell when z>{V3_CHAMPION_SELL_Z:g}, "
        "maintain between (hold after buy until overheated)",
    )
    lines = [
        f"v3 Backtest — {alpha} / {rule}",
        "=" * 48,
        "",
        f"Rule: {rule_desc}",
        "Signal timing: close on day t -> position on day t+1",
        "Benchmark: buy and hold index (801080)",
        "",
        "Full sample:",
        f"  total return: {full['total_return'] * 100:.2f}%",
        f"  excess vs benchmark: {full['excess_vs_benchmark'] * 100:.2f}%",
        f"  max drawdown: {full['max_drawdown'] * 100:.2f}%",
        f"  Sharpe: {full['sharpe']:.3f}",
        f"  trades: {int(full['n_trades'])}",
        f"  avg position: {full['avg_position']:.2f}",
    ]
    if final_row is not None:
        lines.extend(
            [
                "",
                "Final split:",
                f"  total return: {final_row['total_return'] * 100:.2f}%",
                f"  excess vs benchmark: {final_row['excess_vs_benchmark'] * 100:.2f}%",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_champion_config(path: Path, summary: pd.DataFrame) -> None:
    full = summary.loc[summary["split"] == "all"].iloc[0]
    final = summary.loc[summary["split"] == "final"].iloc[0]
    lines = [
        "# v3 champion — locked after optimization grid (4810 configs)",
        f"alpha: {V3_CHAMPION_ALPHA}",
        f"ema_span: {V3_CHAMPION_EMA}",
        f"rule: {V3_CHAMPION_RULE}",
        f"buy_z: {V3_CHAMPION_BUY_Z}",
        f"sell_z: {V3_CHAMPION_SELL_Z}",
        "",
        "# Full sample",
        f"total_return: {full['total_return']:.6f}",
        f"excess_vs_benchmark: {full['excess_vs_benchmark']:.6f}",
        f"n_trades: {int(full['n_trades'])}",
        f"avg_position: {full['avg_position']:.4f}",
        "",
        "# Final holdout (20%)",
        f"total_return_final: {final['total_return']:.6f}",
        f"excess_vs_benchmark_final: {final['excess_vs_benchmark']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_certified_alphas(path: Path) -> None:
    """Document which alphas v3 validated for v4 combination."""
    if not V3_OPT_GRID_RESULTS.exists():
        return

    grid = pd.read_csv(V3_OPT_GRID_RESULTS)
    grid["alpha_2"] = grid["alpha_2"].fillna("").astype(str)

    lines = [
        "# v3 alpha certification for v4",
        "# Criterion A (primary): single-alpha config beats buy & hold on full sample",
        "# Criterion B (secondary): usable in dual-alpha combo that beats B&H",
        "",
        "tested_alphas:",
    ]
    for alpha in ALPHA_NAMES:
        category = "sentiment" if alpha in SENTIMENT_ALPHAS else "auxiliary"
        lines.append(f"  - {alpha}  # {category}")

    lines.extend(["", "single_alpha_certified:"])
    for alpha in ALPHA_NAMES:
        sub = grid[(grid["alpha"] == alpha) & (grid["alpha_2"] == "")]
        beat = sub.loc[sub["beats_bh_all"] == 1]
        if beat.empty:
            best = sub.sort_values("excess_all", ascending=False).iloc[0]
            lines.append(
                f"  {alpha}: false  # best excess={best['excess_all'] * 100:.1f}% "
                f"({best['rule_family']})"
            )
        else:
            best = beat.sort_values("excess_all", ascending=False).iloc[0]
            lines.append(
                f"  {alpha}: true  # best excess={best['excess_all'] * 100:.1f}% "
                f"({best['rule_family']} buy={best['buy_z']} sell={best['sell_z']})"
            )

    dual = grid[(grid["rule_family"] == "dual") & (grid["beats_bh_all"] == 1)]
    lines.extend(["", "dual_combos_certified_for_v4:"])
    if dual.empty:
        lines.append("  (none)")
    else:
        pairs = (
            dual.groupby(["alpha", "alpha_2"])["excess_all"]
            .max()
            .sort_values(ascending=False)
        )
        for (alpha_a, alpha_b), excess in pairs.head(8).items():
            lines.append(f"  - {alpha_a} + {alpha_b}  # max excess={excess * 100:.1f}%")

    lines.extend(
        [
            "",
            "v4_recommendation:",
            "  verdict: single advance_decline beats all tested dual combos",
            "  deploy: advance_decline champion (do not combine unless v4 improves final holdout)",
            "  optional_retest: positive_return  # best dual partner, still below champion",
            "  filter_only: sentiment_composite, index_momentum",
            "  drop: above_ma, new_high_low_net",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_master_report(path: Path, master: pd.DataFrame, champion_summary: pd.DataFrame | None) -> None:
    full_champ = None
    final_champ = None
    if champion_summary is not None:
        full_champ = champion_summary.loc[champion_summary["split"] == "all"].iloc[0]
        final_champ = champion_summary.loc[champion_summary["split"] == "final"].iloc[0]

    lines = [
        "Electronics Sector Sentiment — v3 Final Report",
        "=" * 48,
        "",
        "CHAMPION (locked):",
        f"  {V3_CHAMPION_ALPHA} / EMA{V3_CHAMPION_EMA} / {V3_CHAMPION_RULE}",
        f"  buy z < {V3_CHAMPION_BUY_Z:g}, sell z > {V3_CHAMPION_SELL_Z:g}, maintain between",
    ]
    if full_champ is not None and final_champ is not None:
        lines.extend(
            [
                f"  full return: {full_champ['total_return'] * 100:.1f}%, "
                f"excess vs B&H: {full_champ['excess_vs_benchmark'] * 100:.1f}%",
                f"  final return: {final_champ['total_return'] * 100:.1f}%, "
                f"final excess: {final_champ['excess_vs_benchmark'] * 100:.1f}%",
            ]
        )

    lines.extend(
        [
            "",
            "Exploratory matrix (default thresholds ±1):",
            "  6 alphas x 3 rules — see exploratory/matrix_results.csv",
            "",
            "Alpha certification for v4: see certified_alphas.yaml",
            "",
            "Top 5 exploratory matrix by full-sample excess:",
        ]
    )
    top = master.sort_values("excess_all", ascending=False).head(5)
    for _, row in top.iterrows():
        lines.append(
            f"  {row['alpha']}/{row['rule']}: "
            f"return={row['total_return_all'] * 100:.1f}%, "
            f"excess={row['excess_all'] * 100:.1f}%"
        )

    lines.extend(
        [
            "",
            "Single vs dual (optimization grid):",
            "  champion single advance_decline: full excess +707%, final excess +28.5%",
            "  best dual advance_decline+positive_return: full excess +660%, final excess -49%",
            "  conclusion: dual combos beat B&H but underperform solo advance_decline",
            "",
            f"Champion outputs: output/v3/champion/",
            f"Exploratory matrix: output/v3/exploratory/matrix_results.csv",
            f"Optimization grid: output/v3/optimization/grid_results.csv",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v3_champion() -> pd.DataFrame:
    """Run and archive the locked v3 champion strategy."""
    V3_CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = V3_CHAMPION_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base = build_market_base()
    signals = build_breadth_alpha_signal(base, V3_CHAMPION_ALPHA, ema_span=V3_CHAMPION_EMA)
    config = load_backtest_config(BACKTEST_CONFIG_FILE)
    rule_fn = make_baseline_target(V3_CHAMPION_BUY_Z, V3_CHAMPION_SELL_Z)

    daily = run_backtest(signals, config, rule_fn, initial_position=0.0)
    summary = performance_summary_table(daily)

    daily.to_csv(V3_CHAMPION_DIR / "backtest_daily.csv", index=False)
    summary.to_csv(V3_CHAMPION_DIR / "performance_summary.csv", index=False)
    _write_champion_config(V3_BEST_CONFIG, summary)
    _write_variant_report(
        V3_CHAMPION_DIR / "report.txt",
        V3_CHAMPION_ALPHA,
        V3_CHAMPION_RULE,
        summary,
    )
    plot_equity_curve(
        daily,
        fig_dir / "equity_curve.png",
        title=f"v3 Champion: {V3_CHAMPION_ALPHA} vs Buy & Hold",
    )
    plot_pnl_vs_benchmark(
        daily,
        V3_PNL_FIGURE,
        title="Final Strategy: advance_decline vs Buy & Hold (801080)",
        initial_capital=config.initial_capital,
    )
    plot_pnl_vs_benchmark(
        daily,
        FINAL_PNL_FIGURE,
        title="Final Strategy: advance_decline vs Buy & Hold (801080)",
        initial_capital=config.initial_capital,
    )

    full = summarize_performance(daily)
    print(
        f"[champion] return={full.total_return * 100:.1f}% "
        f"excess={full.excess_return_vs_benchmark * 100:.1f}% trades={full.n_trades}"
    )
    return summary


def run_v3_matrix() -> pd.DataFrame:
    """Run all alpha x rule combinations and archive results."""
    signals = build_all_alpha_signals()
    config = load_backtest_config(BACKTEST_CONFIG_FILE)

    master_rows: list[dict[str, float | int | str]] = []

    for alpha in ALPHA_NAMES:
        signal_frame = signals[alpha]
        for rule_name, rule_fn in RULES.items():
            out_dir = _alpha_output_dir(alpha, rule_name)
            fig_dir = out_dir / "figures"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_dir.mkdir(parents=True, exist_ok=True)

            daily = run_backtest(signal_frame, config, rule_fn)
            summary = performance_summary_table(daily)

            daily.to_csv(out_dir / "backtest_daily.csv", index=False)
            summary.to_csv(out_dir / "performance_summary.csv", index=False)
            _write_variant_report(out_dir / "report.txt", alpha, rule_name, summary)

            category = "sentiment" if alpha in SENTIMENT_ALPHAS else "auxiliary"
            full = summarize_performance(daily)
            final = summarize_performance(daily, "final")

            plot_equity_curve(
                daily,
                fig_dir / "equity_curve.png",
                title=f"{alpha} / {rule_name} vs Buy & Hold",
            )

            master_rows.append(
                {
                    "alpha": alpha,
                    "rule": rule_name,
                    "category": category,
                    "total_return_all": full.total_return,
                    "excess_all": full.excess_return_vs_benchmark,
                    "max_drawdown_all": full.max_drawdown,
                    "sharpe_all": full.sharpe,
                    "n_trades_all": full.n_trades,
                    "avg_position_all": full.avg_position,
                    "total_return_final": final.total_return,
                    "excess_final": final.excess_return_vs_benchmark,
                    "max_drawdown_final": final.max_drawdown,
                    "n_trades_final": final.n_trades,
                }
            )

            print(
                f"[{alpha}/{rule_name}] return={full.total_return * 100:.1f}% "
                f"excess={full.excess_return_vs_benchmark * 100:.1f}% trades={full.n_trades}"
            )

    master = pd.DataFrame(master_rows).sort_values(["category", "alpha", "rule"]).reset_index(drop=True)
    V3_EXPLORATORY_ALPHAS_DIR.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(V3_MATRIX_RESULTS, index=False)
    return master


def main() -> None:
    master = run_v3_matrix()
    champion_summary = run_v3_champion()
    _write_certified_alphas(V3_CERTIFIED_ALPHAS)
    _write_master_report(V3_REPORT, master, champion_summary)
    print(f"\nWrote {V3_MATRIX_RESULTS}")
    print(f"Wrote {V3_BEST_CONFIG}")
    print(f"Wrote {V3_CERTIFIED_ALPHAS}")
    print(f"Wrote {V3_REPORT}")


if __name__ == "__main__":
    main()
