from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.sequence_dataset import SequenceDatasetBuilder
from src.models.frozen_llm import FrozenPatchLLMRegressor
from src.regimes.slicing import build_test_regime_frame, summarize_regimes
from src.utils.io import ensure_dir, save_dataframe
from src.utils.runtime import select_torch_device


def _limit_indices(indices: list[int], limit: int | None) -> list[int]:
    if limit is None or limit <= 0 or len(indices) <= limit:
        return indices
    return indices[-limit:]


def _prepare_frozen_llm_run(
    feature_df: pd.DataFrame,
    split_indices: dict[str, dict[str, Any]],
    config: dict[str, Any],
    smoke_test: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    llm_cfg = copy.deepcopy(config["models"]["frozen_llm"])
    adjusted_split_indices = copy.deepcopy(split_indices)

    if smoke_test:
        smoke_cfg = llm_cfg["smoke_test"]
        llm_cfg["backbone_name"] = smoke_cfg["backbone_name"]
        llm_cfg["batch_size"] = smoke_cfg["batch_size"]
        llm_cfg["num_epochs"] = smoke_cfg["num_epochs"]
        llm_cfg["max_train_samples"] = smoke_cfg["max_train_samples"]
        llm_cfg["max_val_samples"] = smoke_cfg["max_val_samples"]
        llm_cfg["max_test_samples"] = smoke_cfg["max_test_samples"]

        allowed_row_ids = set(feature_df.index[feature_df["ticker"].isin(smoke_cfg["tickers"])].tolist())
        adjusted_split_indices = {
            split_id: payload
            for split_id, payload in adjusted_split_indices.items()
            if split_id in set(smoke_cfg["split_ids"])
        }

        for split_id, payload in adjusted_split_indices.items():
            payload["train_indices"] = [idx for idx in payload["train_indices"] if idx in allowed_row_ids]
            payload["val_indices"] = [idx for idx in payload["val_indices"] if idx in allowed_row_ids]
            payload["test_indices"] = [idx for idx in payload["test_indices"] if idx in allowed_row_ids]

    for payload in adjusted_split_indices.values():
        payload["train_indices"] = _limit_indices(payload["train_indices"], llm_cfg["max_train_samples"])
        payload["val_indices"] = _limit_indices(payload["val_indices"], llm_cfg["max_val_samples"])
        payload["test_indices"] = _limit_indices(payload["test_indices"], llm_cfg["max_test_samples"])

    if not adjusted_split_indices:
        raise RuntimeError("No valid splits remain for the frozen LLM run after applying filters.")

    return feature_df, adjusted_split_indices, llm_cfg


def run_frozen_llm_training(
    feature_df: pd.DataFrame,
    split_indices: dict[str, dict[str, Any]],
    config: dict[str, Any],
    logger,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train the frozen LLM regressor across selected splits and save all outputs."""
    feature_df, split_indices, llm_cfg = _prepare_frozen_llm_run(feature_df, split_indices, config, smoke_test=smoke_test)
    feature_columns = config["features"]["feature_columns"]
    builder = SequenceDatasetBuilder(
        feature_df=feature_df,
        feature_columns=feature_columns,
        lookback_window=llm_cfg["lookback_window"],
    )

    prediction_frames: list[pd.DataFrame] = []
    regime_summary_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    cost_rows: list[dict[str, object]] = []

    for split_id, split_payload in split_indices.items():
        logger.info("Training frozen_llm on split %s.", split_id)
        train_loader, train_meta = builder.build_dataloader(
            split_payload["train_indices"],
            batch_size=llm_cfg["batch_size"],
            shuffle=True,
        )
        val_loader, _ = builder.build_dataloader(
            split_payload["val_indices"],
            batch_size=llm_cfg["batch_size"],
            shuffle=False,
        )
        test_loader, _ = builder.build_dataloader(
            split_payload["test_indices"],
            batch_size=llm_cfg["batch_size"],
            shuffle=False,
        )

        logger.info(
            "Frozen LLM split %s metadata: train=%s samples, val=%s, test=%s, backbone=%s, smoke_test=%s.",
            split_id,
            train_meta.num_samples,
            len(val_loader.dataset),
            len(test_loader.dataset),
            llm_cfg["backbone_name"],
            smoke_test,
        )

        checkpoint_dir = ensure_dir(config["paths"]["frozen_llm_checkpoints_dir"])
        checkpoint_path = Path(checkpoint_dir) / f"{split_id}.pt"
        device = select_torch_device() if llm_cfg["device"] == "auto" else torch.device(llm_cfg["device"])
        model = FrozenPatchLLMRegressor(
            input_dim=train_meta.num_features,
            lookback_window=llm_cfg["lookback_window"],
            patch_size=llm_cfg["patch_size"],
            backbone_name=llm_cfg["backbone_name"],
            projection_hidden_size=llm_cfg["projection_hidden_size"],
            regression_hidden_size=llm_cfg["regression_hidden_size"],
            learning_rate=llm_cfg["learning_rate"],
            num_epochs=llm_cfg["num_epochs"],
            early_stopping_patience=llm_cfg["early_stopping_patience"],
            weight_decay=llm_cfg["weight_decay"],
            dropout=llm_cfg["dropout"],
            seed=config["project"]["random_seed"],
            checkpoint_path=checkpoint_path,
            trust_remote_code=llm_cfg["trust_remote_code"],
            hidden_size=llm_cfg["hidden_size"],
            device=device,
        )
        model.fit(train_loader, val_loader)
        prediction_df = model.predict(test_loader)
        prediction_df, date_regimes = build_test_regime_frame(
            prediction_df,
            volatility_window=config["regimes"]["volatility_window"],
            volatility_quantiles=config["regimes"]["volatility_quantiles"],
            shock_top_fraction=config["regimes"]["shock_top_fraction"],
            shock_window_radius=config["regimes"]["shock_window_radius"],
        )
        prediction_df["split_id"] = split_id
        prediction_df["model_name"] = model.model_name
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

        save_dataframe(prediction_df, Path("outputs/predictions") / model.model_name / f"{split_id}.csv", index=False)
        save_dataframe(date_regimes, Path("outputs/tables") / f"regime_dates_{split_id}_{model.model_name}.csv", index=False)

        regime_summary = summarize_regimes(prediction_df, split_id)
        regime_summary["model_name"] = model.model_name
        regime_summary_frames.append(regime_summary)

        history_df = pd.DataFrame(model.history)
        history_df["split_id"] = split_id
        history_df["model_name"] = model.model_name
        history_df["backbone_name"] = llm_cfg["backbone_name"]
        history_frames.append(history_df)

        cost_rows.append({"split_id": split_id, "model_name": model.model_name, **model.get_cost_summary()})

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_regime_summaries = pd.concat(regime_summary_frames, ignore_index=True)
    all_histories = pd.concat(history_frames, ignore_index=True)
    all_costs = pd.DataFrame(cost_rows).sort_values(["model_name", "split_id"]).reset_index(drop=True)

    save_dataframe(all_regime_summaries, Path("outputs/tables") / "regime_summary_frozen_llm.csv", index=False)
    save_dataframe(all_histories, config["paths"]["frozen_llm_history"], index=False)
    save_dataframe(all_costs, config["paths"]["frozen_llm_costs"], index=False)
    return all_predictions, all_regime_summaries, all_histories, all_costs
