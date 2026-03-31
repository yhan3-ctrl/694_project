from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.sequence_dataset import SequenceDatasetBuilder
from src.models.small_transformer import SmallTransformerRegressor
from src.regimes.slicing import build_test_regime_frame, summarize_regimes
from src.training.pipeline import load_split_indices
from src.utils.io import ensure_dir, save_dataframe


def run_transformer_training(
    feature_df: pd.DataFrame,
    split_indices: dict[str, dict],
    config: dict,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train the small transformer across all splits and save predictions and checkpoints."""
    transformer_cfg = config["models"]["small_transformer"]
    feature_columns = config["features"]["feature_columns"]
    builder = SequenceDatasetBuilder(
        feature_df=feature_df,
        feature_columns=feature_columns,
        lookback_window=transformer_cfg["lookback_window"],
    )

    prediction_frames: list[pd.DataFrame] = []
    regime_summary_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    cost_rows: list[dict[str, object]] = []

    for split_id, split_payload in split_indices.items():
        logger.info("Training small_transformer on split %s.", split_id)
        train_loader, train_meta = builder.build_dataloader(
            split_payload["train_indices"],
            batch_size=transformer_cfg["batch_size"],
            shuffle=True,
        )
        val_loader, val_meta = builder.build_dataloader(
            split_payload["val_indices"],
            batch_size=transformer_cfg["batch_size"],
            shuffle=False,
        )
        test_loader, test_meta = builder.build_dataloader(
            split_payload["test_indices"],
            batch_size=transformer_cfg["batch_size"],
            shuffle=False,
        )
        logger.info(
            "Sequence split %s metadata: train=%s val=%s test=%s samples with %s features.",
            split_id,
            train_meta.num_samples,
            val_meta.num_samples,
            test_meta.num_samples,
            train_meta.num_features,
        )

        checkpoint_dir = ensure_dir(config["paths"]["transformer_checkpoints_dir"])
        checkpoint_path = Path(checkpoint_dir) / f"{split_id}.pt"
        model = SmallTransformerRegressor(
            input_dim=train_meta.num_features,
            lookback_window=transformer_cfg["lookback_window"],
            d_model=transformer_cfg["d_model"],
            nhead=transformer_cfg["nhead"],
            num_layers=transformer_cfg["num_layers"],
            dropout=transformer_cfg["dropout"],
            dim_feedforward=transformer_cfg["dim_feedforward"],
            learning_rate=transformer_cfg["learning_rate"],
            num_epochs=transformer_cfg["num_epochs"],
            early_stopping_patience=transformer_cfg["early_stopping_patience"],
            weight_decay=transformer_cfg["weight_decay"],
            seed=config["project"]["random_seed"],
            checkpoint_path=checkpoint_path,
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

        save_dataframe(
            prediction_df,
            Path("outputs/predictions") / model.model_name / f"{split_id}.csv",
            index=False,
        )
        save_dataframe(
            date_regimes,
            Path("outputs/tables") / f"regime_dates_{split_id}_{model.model_name}.csv",
            index=False,
        )

        regime_summary = summarize_regimes(prediction_df, split_id)
        regime_summary["model_name"] = model.model_name
        regime_summary_frames.append(regime_summary)

        history_df = pd.DataFrame(model.history)
        history_df["split_id"] = split_id
        history_df["model_name"] = model.model_name
        history_frames.append(history_df)

        cost_summary = model.get_cost_summary()
        cost_rows.append(
            {
                "split_id": split_id,
                "model_name": model.model_name,
                **cost_summary,
            }
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_regime_summaries = pd.concat(regime_summary_frames, ignore_index=True)
    all_histories = pd.concat(history_frames, ignore_index=True)
    all_costs = pd.DataFrame(cost_rows).sort_values(["model_name", "split_id"]).reset_index(drop=True)

    save_dataframe(all_regime_summaries, Path("outputs/tables") / "regime_summary_transformer.csv", index=False)
    save_dataframe(all_histories, config["paths"]["transformer_history"], index=False)
    save_dataframe(all_costs, config["paths"]["transformer_costs"], index=False)
    return all_predictions, all_regime_summaries, all_histories, all_costs
