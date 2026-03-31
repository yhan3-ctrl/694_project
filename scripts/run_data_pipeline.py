from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.yahoo import clean_and_align_prices, download_yahoo_ohlcv
from src.features.engineering import build_feature_dataset
from src.splits.time_splits import create_rolling_splits, create_single_time_split, materialize_split_indices
from src.utils.config import load_config
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger
from src.utils.runtime import get_runtime_context


def run_data_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    logger = get_logger("run_data_pipeline")
    logger.info("Runtime context: %s", get_runtime_context())
    logger.info("Downloading Yahoo Finance data.")

    raw_df = download_yahoo_ohlcv(
        tickers=config["data"]["tickers"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    save_dataframe(raw_df, config["paths"]["raw_prices"], index=False)

    cleaned_df, dataset_summary = clean_and_align_prices(raw_df)
    save_dataframe(cleaned_df, config["paths"]["processed_prices"], index=False)
    save_dataframe(dataset_summary, config["paths"]["dataset_summary"], index=False)
    logger.info("Saved raw and cleaned price data.")

    feature_df = build_feature_dataset(
        cleaned_df,
        lookback_window=config["features"]["lookback_window"],
        return_windows=config["features"]["return_windows"],
        rolling_windows=config["features"]["rolling_windows"],
        volatility_window=config["features"]["volatility_window"],
    )
    save_dataframe(feature_df, config["paths"]["feature_dataset"], index=False)
    logger.info("Saved feature dataset with %s rows.", len(feature_df))

    unique_dates = feature_df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    single_splits = create_single_time_split(
        unique_dates=unique_dates,
        train_frac=config["splits"]["single"]["train_frac"],
        val_frac=config["splits"]["single"]["val_frac"],
    )
    rolling_splits = create_rolling_splits(
        unique_dates=unique_dates,
        train_days=config["splits"]["rolling"]["train_days"],
        val_days=config["splits"]["rolling"]["val_days"],
        test_days=config["splits"]["rolling"]["test_days"],
        step_days=config["splits"]["rolling"]["step_days"],
        num_splits=config["splits"]["rolling"]["num_splits"],
    )
    split_summary, split_indices = materialize_split_indices(feature_df, single_splits + rolling_splits)
    save_dataframe(split_summary, config["paths"]["split_summary"], index=False)
    save_json(split_indices, config["paths"]["split_indices"])
    logger.info("Saved split summary and split indices.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    args = parser.parse_args()
    run_data_pipeline(args.config)


if __name__ == "__main__":
    main()

