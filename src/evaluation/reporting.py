from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_dataframe


def build_model_cost_summary(config: dict) -> pd.DataFrame:
    """Merge available cost tables and save an aggregated model-level summary."""
    cost_paths = [
        config["paths"]["baseline_costs"],
        config["paths"]["transformer_costs"],
        config["paths"]["frozen_llm_costs"],
    ]
    frames = [pd.read_csv(path) for path in cost_paths if Path(path).exists()]
    if not frames:
        return pd.DataFrame()

    split_level_costs = pd.concat(frames, ignore_index=True)
    model_cost_summary = split_level_costs.groupby("model_name", as_index=False).mean(numeric_only=True)
    save_dataframe(model_cost_summary, config["paths"]["model_cost_summary"], index=False)
    return model_cost_summary


def create_results_tables(
    aggregated_metrics: pd.DataFrame,
    cost_summary: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create main benchmark and mechanism comparison tables."""
    if cost_summary.empty:
        merged = aggregated_metrics.copy()
    else:
        merged = aggregated_metrics.merge(cost_summary, on="model_name", how="left")

    main_columns = [
        "model_name",
        "overall_mae",
        "overall_mse",
        "overall_direction_accuracy",
        "vol_low_mae",
        "vol_mid_mae",
        "vol_high_mae",
        "shock_mae",
        "tail_bottom_10_mae",
        "wcs_error",
        "wcs_acc",
        "training_time_seconds",
        "inference_time_seconds",
        "total_params",
        "trainable_params",
        "peak_gpu_memory_mb",
    ]
    main_results = merged[[column for column in main_columns if column in merged.columns]].copy()
    save_dataframe(main_results, config["paths"]["main_results_table"], index=False)
    save_dataframe(main_results, config["paths"]["main_results_with_frozen_llm"], index=False)

    mechanism_columns = [
        "model_name",
        "overall_mae",
        "overall_direction_accuracy",
        "vol_high_mae",
        "vol_high_direction_accuracy",
        "shock_mae",
        "shock_direction_accuracy",
        "tail_bottom_10_mae",
        "wcs_error",
        "wcs_acc",
        "training_time_seconds",
        "inference_time_seconds",
        "total_params",
        "trainable_params",
        "peak_gpu_memory_mb",
    ]
    mechanism_results = merged[merged["model_name"].isin(["lightgbm", "small_transformer"])].copy()
    mechanism_results = mechanism_results[[column for column in mechanism_columns if column in mechanism_results.columns]]
    save_dataframe(mechanism_results, config["paths"]["mechanism_comparison_table"], index=False)
    pretraining_results = merged[merged["model_name"].isin(["small_transformer", "frozen_llm"])].copy()
    pretraining_results = pretraining_results[
        [column for column in mechanism_columns if column in pretraining_results.columns]
    ]
    save_dataframe(pretraining_results, config["paths"]["mechanism_comparison_pretraining"], index=False)

    return main_results, mechanism_results, main_results, pretraining_results


def create_frozen_llm_full_run_summary(
    aggregated_metrics: pd.DataFrame,
    cost_summary: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Create a concise frozen LLM summary row with environment and comparison deltas."""
    frozen_llm_row = aggregated_metrics[aggregated_metrics["model_name"] == "frozen_llm"].copy()
    if frozen_llm_row.empty:
        return pd.DataFrame()

    small_transformer_row = aggregated_metrics[aggregated_metrics["model_name"] == "small_transformer"].copy()
    cost_row = cost_summary[cost_summary["model_name"] == "frozen_llm"].copy() if not cost_summary.empty else pd.DataFrame()
    context_path = Path(config["paths"]["frozen_llm_run_context"])
    context_row = pd.read_csv(context_path) if context_path.exists() else pd.DataFrame()

    summary = frozen_llm_row.copy()
    if not cost_row.empty:
        summary = summary.merge(cost_row, on="model_name", how="left")
    if not context_row.empty:
        summary = summary.merge(context_row, on="model_name", how="left")

    if not small_transformer_row.empty:
        small_row = small_transformer_row.iloc[0]
        summary["delta_vs_small_transformer_overall_mae"] = summary["overall_mae"] - float(small_row["overall_mae"])
        summary["delta_vs_small_transformer_vol_high_mae"] = summary["vol_high_mae"] - float(small_row["vol_high_mae"])
        summary["delta_vs_small_transformer_shock_mae"] = summary["shock_mae"] - float(small_row["shock_mae"])
        summary["delta_vs_small_transformer_wcs_error"] = summary["wcs_error"] - float(small_row["wcs_error"])
        summary["delta_vs_small_transformer_wcs_acc"] = summary["wcs_acc"] - float(small_row["wcs_acc"])

    save_dataframe(summary, config["paths"]["frozen_llm_full_run_summary"], index=False)
    return summary
