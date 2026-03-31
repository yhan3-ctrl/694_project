from __future__ import annotations

from typing import Any

import pandas as pd


def _build_split_record(
    split_id: str,
    split_kind: str,
    train_dates: pd.Index,
    val_dates: pd.Index,
    test_dates: pd.Index,
) -> dict[str, Any]:
    return {
        "split_id": split_id,
        "split_kind": split_kind,
        "train_start": train_dates.min().date().isoformat(),
        "train_end": train_dates.max().date().isoformat(),
        "val_start": val_dates.min().date().isoformat(),
        "val_end": val_dates.max().date().isoformat(),
        "test_start": test_dates.min().date().isoformat(),
        "test_end": test_dates.max().date().isoformat(),
        "train_num_dates": int(len(train_dates)),
        "val_num_dates": int(len(val_dates)),
        "test_num_dates": int(len(test_dates)),
    }


def create_single_time_split(
    unique_dates: pd.Index,
    train_frac: float,
    val_frac: float,
) -> list[dict[str, Any]]:
    """Create a single chronological train/val/test split."""
    n_dates = len(unique_dates)
    train_end = int(n_dates * train_frac)
    val_end = train_end + int(n_dates * val_frac)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    return [_build_split_record("single_0", "single", train_dates, val_dates, test_dates)]


def create_rolling_splits(
    unique_dates: pd.Index,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    num_splits: int,
) -> list[dict[str, Any]]:
    """Create rolling walk-forward splits over ordered dates."""
    splits: list[dict[str, Any]] = []
    total_window = train_days + val_days + test_days
    max_start = len(unique_dates) - total_window

    start_positions = [idx * step_days for idx in range(num_splits + 10)]
    for position in start_positions:
        if position > max_start or len(splits) >= num_splits:
            break

        train_dates = unique_dates[position : position + train_days]
        val_dates = unique_dates[position + train_days : position + train_days + val_days]
        test_dates = unique_dates[position + train_days + val_days : position + total_window]

        splits.append(
            _build_split_record(
                split_id=f"rolling_{len(splits)}",
                split_kind="rolling",
                train_dates=train_dates,
                val_dates=val_dates,
                test_dates=test_dates,
            )
        )

    if len(splits) < num_splits:
        raise RuntimeError("Unable to create the requested number of rolling splits from the dataset.")

    return splits


def materialize_split_indices(
    feature_df: pd.DataFrame,
    split_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate row indices and row counts for each split."""
    summary_rows: list[dict[str, Any]] = []
    split_indices: dict[str, Any] = {}

    date_series = pd.to_datetime(feature_df["date"])

    for record in split_records:
        split_id = record["split_id"]
        train_mask = (date_series >= record["train_start"]) & (date_series <= record["train_end"])
        val_mask = (date_series >= record["val_start"]) & (date_series <= record["val_end"])
        test_mask = (date_series >= record["test_start"]) & (date_series <= record["test_end"])

        train_idx = feature_df.index[train_mask].tolist()
        val_idx = feature_df.index[val_mask].tolist()
        test_idx = feature_df.index[test_mask].tolist()

        split_indices[split_id] = {
            **record,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        }

        summary_rows.append(
            {
                **record,
                "train_num_rows": int(len(train_idx)),
                "val_num_rows": int(len(val_idx)),
                "test_num_rows": int(len(test_idx)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["split_kind", "split_id"]).reset_index(drop=True)
    return summary_df, split_indices

