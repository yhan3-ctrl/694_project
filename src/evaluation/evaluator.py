from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import save_dataframe, save_json


def _compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"mae": np.nan, "mse": np.nan, "direction_accuracy": np.nan}

    errors = df["y_true"] - df["y_pred"]
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "direction_accuracy": float(np.mean(np.sign(df["y_true"]) == np.sign(df["y_pred"]))),
    }


def evaluate_predictions(prediction_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute split-level and aggregated metrics from prediction data."""
    metric_rows: list[dict[str, Any]] = []

    for (model_name, split_id), split_df in prediction_df.groupby(["model_name", "split_id"], sort=True):
        overall = _compute_metrics(split_df)
        vol_low = _compute_metrics(split_df[split_df["vol_regime"] == "vol_low"])
        vol_mid = _compute_metrics(split_df[split_df["vol_regime"] == "vol_mid"])
        vol_high = _compute_metrics(split_df[split_df["vol_regime"] == "vol_high"])
        shock = _compute_metrics(split_df[split_df["is_shock"]])

        tail_cutoff = split_df["y_true"].quantile(0.10)
        tail_df = split_df[split_df["y_true"] <= tail_cutoff]
        tail_metrics = _compute_metrics(tail_df)

        metric_rows.append(
            {
                "model_name": model_name,
                "split_id": split_id,
                "overall_mae": overall["mae"],
                "overall_mse": overall["mse"],
                "overall_direction_accuracy": overall["direction_accuracy"],
                "vol_low_mae": vol_low["mae"],
                "vol_low_mse": vol_low["mse"],
                "vol_low_direction_accuracy": vol_low["direction_accuracy"],
                "vol_mid_mae": vol_mid["mae"],
                "vol_mid_mse": vol_mid["mse"],
                "vol_mid_direction_accuracy": vol_mid["direction_accuracy"],
                "vol_high_mae": vol_high["mae"],
                "vol_high_mse": vol_high["mse"],
                "vol_high_direction_accuracy": vol_high["direction_accuracy"],
                "shock_mae": shock["mae"],
                "shock_mse": shock["mse"],
                "shock_direction_accuracy": shock["direction_accuracy"],
                "tail_bottom_10_mae": tail_metrics["mae"],
                "wcs_error": float(np.nanmax([vol_high["mae"], shock["mae"]])),
                "wcs_acc": float(np.nanmin([vol_high["direction_accuracy"], shock["direction_accuracy"]])),
            }
        )

    split_metrics = pd.DataFrame(metric_rows).sort_values(["model_name", "split_id"]).reset_index(drop=True)
    aggregated = split_metrics.groupby("model_name", as_index=False).mean(numeric_only=True)
    return split_metrics, aggregated


def save_metrics(
    split_metrics: pd.DataFrame,
    aggregated_metrics: pd.DataFrame,
    split_csv_path: str,
    split_json_path: str,
    aggregated_csv_path: str,
    aggregated_json_path: str,
) -> None:
    """Persist evaluation metrics in CSV and JSON formats."""
    save_dataframe(split_metrics, split_csv_path, index=False)
    save_dataframe(aggregated_metrics, aggregated_csv_path, index=False)
    save_json(split_metrics.to_dict(orient="records"), split_json_path)
    save_json(aggregated_metrics.to_dict(orient="records"), aggregated_json_path)

