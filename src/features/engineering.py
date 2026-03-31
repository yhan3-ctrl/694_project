from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_dataset(
    price_df: pd.DataFrame,
    lookback_window: int,
    return_windows: list[int],
    rolling_windows: list[int],
    volatility_window: int,
) -> pd.DataFrame:
    """Build a leakage-safe next-day return modeling table."""
    frames: list[pd.DataFrame] = []

    for ticker, group in price_df.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True).copy()

        group["log_return_1"] = np.log(group["close"] / group["close"].shift(1))
        for window in return_windows:
            group[f"log_return_{window}"] = np.log(group["close"] / group["close"].shift(window))

        for window in rolling_windows:
            group[f"rolling_mean_return_{window}"] = group["log_return_1"].rolling(window).mean()
            group[f"rolling_std_return_{window}"] = group["log_return_1"].rolling(window).std()
            group[f"rolling_mean_volume_{window}"] = group["volume"].rolling(window).mean()

        group["volatility_20"] = group["log_return_1"].rolling(volatility_window).std()
        group["volume_log"] = np.log1p(group["volume"])
        group["volume_change_1"] = np.log1p(group["volume"]).diff()
        group["volume_ratio_20"] = group["volume"] / group["rolling_mean_volume_20"]
        group["close_to_open"] = group["close"] / group["open"] - 1.0
        group["high_to_low"] = group["high"] / group["low"] - 1.0
        group["close_to_high"] = group["close"] / group["high"]
        group["close_to_low"] = group["close"] / group["low"]
        group["adj_close_ratio"] = group["adj_close"] / group["close"]
        group["range_normalized"] = (group["high"] - group["low"]) / group["close"]
        group["target"] = np.log(group["close"].shift(-1) / group["close"])
        group["ticker"] = ticker

        frames.append(group)

    feature_df = pd.concat(frames, ignore_index=True)
    feature_df = feature_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_columns = [
        column
        for column in feature_df.columns
        if column
        not in {
            "target",
            "date",
            "ticker",
        }
    ]
    feature_df = feature_df.dropna(subset=feature_columns + ["target"]).reset_index(drop=True)
    feature_df = feature_df[feature_df.groupby("ticker").cumcount() >= max(0, lookback_window - 1)].reset_index(
        drop=True
    )
    feature_df["sample_id"] = np.arange(len(feature_df))
    return feature_df

