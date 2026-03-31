from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged: list[list[int]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def build_test_regime_frame(
    test_df: pd.DataFrame,
    volatility_window: int,
    volatility_quantiles: list[float],
    shock_top_fraction: float,
    shock_window_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reusable date-level regime labels for a test segment."""
    date_frame = (
        test_df.groupby("date", as_index=False)
        .agg(
            market_return=("y_true", "mean"),
            market_abs_return=("y_true", lambda values: np.abs(values).mean()),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    date_frame["rolling_vol_20"] = date_frame["market_return"].rolling(volatility_window).std()

    valid_vol = date_frame["rolling_vol_20"].dropna()
    if valid_vol.empty:
        low_q, high_q = 0.0, 0.0
    else:
        low_q, high_q = valid_vol.quantile(volatility_quantiles).tolist()

    date_frame["vol_regime"] = "vol_mid"
    date_frame.loc[date_frame["rolling_vol_20"] <= low_q, "vol_regime"] = "vol_low"
    date_frame.loc[date_frame["rolling_vol_20"] >= high_q, "vol_regime"] = "vol_high"
    date_frame.loc[date_frame["rolling_vol_20"].isna(), "vol_regime"] = "vol_mid"

    num_dates = len(date_frame)
    num_shocks = max(1, math.ceil(num_dates * shock_top_fraction))
    shock_positions = date_frame["market_abs_return"].nlargest(num_shocks).index.tolist()
    merged_intervals = _merge_intervals(
        [
            (max(0, position - shock_window_radius), min(num_dates - 1, position + shock_window_radius))
            for position in shock_positions
        ]
    )

    date_frame["is_shock"] = False
    date_frame["shock_window_id"] = None
    for interval_id, (start, end) in enumerate(merged_intervals):
        date_frame.loc[start:end, "is_shock"] = True
        date_frame.loc[start:end, "shock_window_id"] = f"shock_window_{interval_id}"

    date_frame["time_regime"] = "future_test"

    labeled_test = test_df.merge(
        date_frame[["date", "time_regime", "vol_regime", "is_shock", "shock_window_id", "rolling_vol_20"]],
        on="date",
        how="left",
    )
    labeled_test["is_shock"] = labeled_test["is_shock"].fillna(False)

    return labeled_test, date_frame


def summarize_regimes(labeled_test_df: pd.DataFrame, split_id: str) -> pd.DataFrame:
    """Create a compact regime coverage summary for a split."""
    summary = pd.DataFrame(
        [
            {"split_id": split_id, "regime": "overall", "num_rows": int(len(labeled_test_df))},
            {
                "split_id": split_id,
                "regime": "vol_low",
                "num_rows": int((labeled_test_df["vol_regime"] == "vol_low").sum()),
            },
            {
                "split_id": split_id,
                "regime": "vol_mid",
                "num_rows": int((labeled_test_df["vol_regime"] == "vol_mid").sum()),
            },
            {
                "split_id": split_id,
                "regime": "vol_high",
                "num_rows": int((labeled_test_df["vol_regime"] == "vol_high").sum()),
            },
            {
                "split_id": split_id,
                "regime": "shock",
                "num_rows": int(labeled_test_df["is_shock"].sum()),
            },
        ]
    )
    return summary

