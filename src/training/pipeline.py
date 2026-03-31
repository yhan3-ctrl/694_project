from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.arima_model import ARIMABaseline
from src.models.lightgbm_model import LightGBMBaseline
from src.regimes.slicing import build_test_regime_frame, summarize_regimes
from src.utils.io import ensure_dir, save_dataframe


def load_split_indices(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _make_model(model_name: str, config: dict[str, Any]):
    if model_name == "arima":
        return ARIMABaseline(order=tuple(config["models"]["arima"]["order"]))
    if model_name == "lightgbm":
        return LightGBMBaseline(params=config["models"]["lightgbm"])
    raise ValueError(f"Unsupported model: {model_name}")


def _select_split_frames(feature_df: pd.DataFrame, split_payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = feature_df.loc[split_payload["train_indices"]].copy()
    val_df = feature_df.loc[split_payload["val_indices"]].copy()
    test_df = feature_df.loc[split_payload["test_indices"]].copy()

    for frame in (train_df, val_df, test_df):
        frame.rename(columns={"target": "y_true"}, inplace=True)
        frame["date"] = pd.to_datetime(frame["date"])

    return train_df, val_df, test_df


def run_baseline_training(
    feature_df: pd.DataFrame,
    split_indices: dict[str, Any],
    config: dict[str, Any],
    logger,
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train all configured baselines and save predictions."""
    prediction_frames: list[pd.DataFrame] = []
    regime_summary_frames: list[pd.DataFrame] = []
    cost_rows: list[dict[str, object]] = []
    enabled_models = model_names or config["models"]["baselines_enabled"]

    for split_id, split_payload in split_indices.items():
        train_df, val_df, test_df = _select_split_frames(feature_df, split_payload)
        logger.info(
            "Running split %s with %s train rows, %s val rows, %s test rows.",
            split_id,
            len(train_df),
            len(val_df),
            len(test_df),
        )

        for model_name in enabled_models:
            logger.info("Training %s on split %s.", model_name, split_id)
            model = _make_model(model_name, config)
            train_start_time = time.perf_counter()
            model.fit(train_df, val_df)
            training_time_seconds = float(time.perf_counter() - train_start_time)
            inference_start_time = time.perf_counter()
            prediction_df = model.predict(test_df)
            inference_time_seconds = float(time.perf_counter() - inference_start_time)

            prediction_df, date_regimes = build_test_regime_frame(
                prediction_df,
                volatility_window=config["regimes"]["volatility_window"],
                volatility_quantiles=config["regimes"]["volatility_quantiles"],
                shock_top_fraction=config["regimes"]["shock_top_fraction"],
                shock_window_radius=config["regimes"]["shock_window_radius"],
            )
            prediction_df["split_id"] = split_id
            prediction_df["model_name"] = model_name
            prediction_df = prediction_df[
                [
                    "date",
                    "ticker",
                    "y_true",
                    "y_pred",
                    "split_id",
                    "model_name",
                    "time_regime",
                    "vol_regime",
                    "is_shock",
                    "shock_window_id",
                    "rolling_vol_20",
                ]
            ]
            prediction_frames.append(prediction_df)

            model_dir = ensure_dir(Path("outputs/predictions") / model_name)
            save_dataframe(prediction_df, model_dir / f"{split_id}.csv", index=False)

            regime_summary = summarize_regimes(prediction_df, split_id)
            regime_summary["model_name"] = model_name
            regime_summary_frames.append(regime_summary)
            save_dataframe(
                date_regimes,
                Path("outputs/tables") / f"regime_dates_{split_id}_{model_name}.csv",
                index=False,
            )
            parameter_counts = model.count_parameters() if hasattr(model, "count_parameters") else {}
            cost_rows.append(
                {
                    "split_id": split_id,
                    "model_name": model_name,
                    "training_time_seconds": training_time_seconds,
                    "inference_time_seconds": inference_time_seconds,
                    "total_params": parameter_counts.get("total_params", 0),
                    "trainable_params": parameter_counts.get("trainable_params", 0),
                    "peak_gpu_memory_mb": None,
                }
            )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_regime_summaries = pd.concat(regime_summary_frames, ignore_index=True)
    all_costs = pd.DataFrame(cost_rows).sort_values(["model_name", "split_id"]).reset_index(drop=True)
    save_dataframe(all_regime_summaries, Path("outputs/tables") / "regime_summary.csv", index=False)
    save_dataframe(all_costs, config["paths"]["baseline_costs"], index=False)
    return all_predictions, all_regime_summaries, all_costs
