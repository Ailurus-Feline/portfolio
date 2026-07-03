"""v2 grid search over sentiment construction parameters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    IC_HORIZONS,
    IC_PRIMARY_HORIZON,
    TOP_K_TRAIN,
    V1_OUTPUT_CSV,
    V2_BEST_CONFIG,
    V2_BEST_SENTIMENT_CSV,
    V2_BEST_SENTIMENT_FIGURE,
    V2_EVENT_STUDY_FIGURE,
    V2_FIGURES_DIR,
    V2_FINAL_REPORT,
    V2_GRID_RESULTS,
    V2_GRID_TOP_FIGURE,
    V2_SELECTION_FIGURE,
    V2_IC_BY_SPLIT_FIGURE,
    V2_OUTPUT_DIR,
    V2_QUINTILE_FIGURE,
    V2_SUMMARY_FIGURE,
    V2_V1_V2_COMPARISON_FIGURE,
    WEIGHT_PRESETS,
    EMA_GRID,
)
from src.data_loader import load_market_data
from src.diagnostics import (
    add_forward_returns,
    attach_splits,
    evaluate_segment,
    ic_score,
    plot_event_study,
    plot_grid_top_results,
    plot_ic_by_split,
    plot_quintile_returns,
    plot_selection_candidates,
    plot_v2_summary,
    split_date_range,
    summarize_ic,
)
from src.indicators import compute_breadth_indicators
from src.plot import plot_sentiment_vs_index, plot_v1_v2_zscore_comparison
from src.sentiment import (
    SentimentParams,
    composite_from_percentiles,
    composite_sentiment,
    merge_with_index,
    prepare_percentile_features,
    select_analysis_rows,
)


def build_parameter_grid() -> list[tuple[str, SentimentParams]]:
    """Return all weight-preset and EMA combinations."""
    grid: list[tuple[str, SentimentParams]] = []
    for preset_name, weights in WEIGHT_PRESETS.items():
        for ema_span in EMA_GRID:
            config_id = f"{preset_name}__ema{ema_span}"
            grid.append((config_id, SentimentParams(weights=weights, ema_span=ema_span)))
    return grid


def _build_evaluation_frame(market: dict[str, object]) -> pd.DataFrame:
    indicators = compute_breadth_indicators(
        close_panel=market["close_panel"],
        valid_panel=market["valid_panel"],
    ).reset_index(names="date")
    features = prepare_percentile_features(indicators)

    timeline = select_analysis_rows(merge_with_index(composite_sentiment(indicators), market["index"]))
    timeline = attach_splits(timeline)

    base = merge_with_index(features, market["index"])
    base = base.merge(timeline[["date", "split"]], on="date", how="inner")
    base = base.sort_values("date").reset_index(drop=True)
    return add_forward_returns(base)


def _evaluate_config(
    base: pd.DataFrame,
    config_id: str,
    params: SentimentParams,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    sentiment = composite_from_percentiles(base, params)
    eval_frame = sentiment.loc[sentiment["sentiment_z"].notna(), :].loc[
        :,
        [
            "date",
            "split",
            "sentiment_z",
            "regime",
            "index_close",
            *[f"fwd_ret_{horizon}d" for horizon in IC_HORIZONS],
        ],
    ].copy()

    row: dict[str, float | int | str] = {
        "config_id": config_id,
        "weight_preset": config_id.rsplit("__", maxsplit=1)[0],
        "w_new_high_low": params.weights[0],
        "w_above_ma": params.weights[1],
        "w_positive_return": params.weights[2],
        "ema_span": params.ema_span,
    }

    for split in ("train", "valid", "final"):
        for horizon in IC_HORIZONS:
            summary = summarize_ic(eval_frame, horizon, split=split)
            prefix = f"{split}_"
            row[f"{prefix}rank_ic_{horizon}d"] = summary.rank_ic
            row[f"{prefix}ic_ir_{horizon}d"] = summary.ic_ir
            row[f"{prefix}n_obs_{horizon}d"] = summary.n_obs
        split_frame = eval_frame.loc[eval_frame["split"] == split]
        row[f"{split}_score"] = ic_score({horizon: summarize_ic(split_frame, horizon) for horizon in IC_HORIZONS})

    return eval_frame, row


def _select_best_candidate(results: pd.DataFrame) -> pd.Series:
    train_ranked = results.sort_values("train_score", ascending=True).head(TOP_K_TRAIN)
    return train_ranked.sort_values(["valid_score", "train_score"], ascending=[True, True]).iloc[0]


def _format_ic(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.4f}"


def _write_best_config(path: Path, winner: pd.Series) -> None:
    content = f"""# v2 selected sentiment parameters (locked before final evaluation)

weight_preset: {winner['weight_preset']}
weights:
  new_high_low_net: {winner['w_new_high_low']:.4f}
  above_ma: {winner['w_above_ma']:.4f}
  positive_return: {winner['w_positive_return']:.4f}
ema_span: {int(winner['ema_span'])}

selection_rule: top {TOP_K_TRAIN} by train IC score, then best valid IC score
"""
    path.write_text(content, encoding="utf-8")


def _write_final_report(
    path: Path,
    results: pd.DataFrame,
    winner: pd.Series,
    eval_frame: pd.DataFrame,
    final_diag,
) -> None:
    train_start, train_end = split_date_range(eval_frame, "train")
    valid_start, valid_end = split_date_range(eval_frame, "valid")
    final_start, final_end = split_date_range(eval_frame, "final")

    lines = [
        "Electronics Sector Sentiment — v2 Final Report",
        "=" * 48,
        "",
        "Split ranges (chronological 6-2-2):",
        f"  train: {train_start.date()} -> {train_end.date()}",
        f"  valid: {valid_start.date()} -> {valid_end.date()}",
        f"  final: {final_start.date()} -> {final_end.date()}",
        "",
        "Selected parameters:",
        f"  config_id: {winner['config_id']}",
        f"  weights: ({winner['w_new_high_low']:.2f}, {winner['w_above_ma']:.2f}, {winner['w_positive_return']:.2f})",
        f"  ema_span: {int(winner['ema_span'])}",
        "",
        "IC summary (Rank IC; negative is desirable for contrarian sentiment):",
        f"  train score: {_format_ic(winner['train_score'])}",
        f"  valid score: {_format_ic(winner['valid_score'])}",
        f"  final score: {_format_ic(winner['final_score'])}",
        "",
        f"Final rank IC ({IC_PRIMARY_HORIZON}d): {_format_ic(winner[f'final_rank_ic_{IC_PRIMARY_HORIZON}d'])}",
        f"Final rank IC ({IC_HORIZONS[1]}d): {_format_ic(winner[f'final_rank_ic_{IC_HORIZONS[1]}d'])}",
        "",
        f"Final event study ({IC_PRIMARY_HORIZON}d mean forward return):",
    ]

    for _, event_row in final_diag.event_study.iterrows():
        lines.append(
            f"  {event_row['regime']}: {event_row['mean_forward_return'] * 100:.2f}% "
            f"(n={int(event_row['count'])})"
        )

    lines.extend(
        [
            "",
            f"Grid size: {len(results)} combinations",
            f"Top-{TOP_K_TRAIN} train candidates reviewed on valid before final lock.",
            "",
            "Figures (see output/v2/figures/):",
            "  v2_summary.png              — four-panel overview (start here)",
            "  ic_by_split.png             — Rank IC on train / valid / final",
            "  event_study.png             — forward return by overcooled/neutral/overheated",
            "  quintile_forward_returns.png — forward return by sentiment quintile",
            "  grid_top_configs.png        — top configs by valid IC score",
            "  selection_candidates.png    — train top-5 pool and final pick",
            "  sentiment_vs_index_best.png — optimized sentiment vs index chart",
            "  v1_v2_zscore_comparison.png  — overlay showing v1 vs v2 z-score difference",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v2_grid_search() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Run the full v2 grid search and write outputs."""
    market = load_market_data()
    base = _build_evaluation_frame(market)

    result_rows: list[dict[str, float | int | str]] = []
    for config_id, params in build_parameter_grid():
        _, row = _evaluate_config(base, config_id, params)
        result_rows.append(row)

    results = pd.DataFrame(result_rows).sort_values("train_score", ascending=True).reset_index(drop=True)
    winner = _select_best_candidate(results)

    winner_params = SentimentParams(
        weights=(winner["w_new_high_low"], winner["w_above_ma"], winner["w_positive_return"]),
        ema_span=int(winner["ema_span"]),
    )
    best_eval_frame = composite_from_percentiles(base, winner_params)
    best_eval_frame = select_analysis_rows(best_eval_frame)
    best_eval_frame = best_eval_frame.loc[
        :,
        [
            "date",
            "split",
            "sentiment_raw",
            "sentiment_slow",
            "sentiment_z",
            "regime",
            "index_close",
            "index_log",
            *[f"fwd_ret_{horizon}d" for horizon in IC_HORIZONS],
        ],
    ].copy()

    final_diag = evaluate_segment(best_eval_frame, "final")

    V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    V2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(V2_GRID_RESULTS, index=False)
    best_eval_frame.to_csv(V2_BEST_SENTIMENT_CSV, index=False)
    _write_best_config(V2_BEST_CONFIG, winner)
    _write_final_report(V2_FINAL_REPORT, results, winner, best_eval_frame, final_diag)

    plot_v2_summary(winner, results, final_diag, V2_SUMMARY_FIGURE)
    plot_ic_by_split(winner, V2_IC_BY_SPLIT_FIGURE)
    plot_event_study(final_diag.event_study, V2_EVENT_STUDY_FIGURE, IC_PRIMARY_HORIZON, "final")
    plot_quintile_returns(final_diag.quintile_returns, V2_QUINTILE_FIGURE, IC_PRIMARY_HORIZON, "final")
    plot_grid_top_results(results, V2_GRID_TOP_FIGURE, winner)
    plot_selection_candidates(results, V2_SELECTION_FIGURE, winner, top_k_train=TOP_K_TRAIN)
    plot_sentiment_vs_index(
        best_eval_frame,
        V2_BEST_SENTIMENT_FIGURE,
        title=(
            "Electronics Sector Sentiment (v2 Best) vs Shenwan Electronics Index\n"
            f"weights=0.25/0.25/0.50, EMA=60  |  black line = sentiment z-score"
        ),
    )

    v1_frame = pd.read_csv(V1_OUTPUT_CSV, parse_dates=["date"]) if V1_OUTPUT_CSV.exists() else None
    if v1_frame is not None:
        plot_v1_v2_zscore_comparison(v1_frame, best_eval_frame, V2_V1_V2_COMPARISON_FIGURE)

    figure_paths = [
        V2_SUMMARY_FIGURE,
        V2_IC_BY_SPLIT_FIGURE,
        V2_EVENT_STUDY_FIGURE,
        V2_QUINTILE_FIGURE,
        V2_GRID_TOP_FIGURE,
        V2_SELECTION_FIGURE,
        V2_BEST_SENTIMENT_FIGURE,
    ]
    if v1_frame is not None:
        figure_paths.append(V2_V1_V2_COMPARISON_FIGURE)
    for path in [V2_GRID_RESULTS, V2_BEST_CONFIG, V2_FINAL_REPORT, V2_BEST_SENTIMENT_CSV, *figure_paths]:
        print(f"Wrote {path}")

    print(f"Selected config: {winner['config_id']}")
    print(
        f"Train score: {winner['train_score']:.4f}; "
        f"Valid score: {winner['valid_score']:.4f}; "
        f"Final score: {winner['final_score']:.4f}"
    )

    return results, winner, best_eval_frame


def main() -> None:
    run_v2_grid_search()


if __name__ == "__main__":
    main()
