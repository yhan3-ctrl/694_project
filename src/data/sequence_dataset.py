from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SequenceDatasetMetadata:
    """Container for sequence dataset summary fields."""

    num_samples: int
    lookback_window: int
    num_features: int


class SequenceForecastDataset(Dataset):
    """PyTorch dataset for rolling sequence forecasting samples."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        dates: list[str],
        tickers: list[str],
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.dates = dates
        self.tickers = tickers

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "inputs": self.sequences[index],
            "target": self.targets[index],
            "date": self.dates[index],
            "ticker": self.tickers[index],
        }


class SequenceDatasetBuilder:
    """Build leakage-safe sequence datasets from the engineered feature table."""

    def __init__(
        self,
        feature_df: pd.DataFrame,
        feature_columns: list[str],
        lookback_window: int,
    ) -> None:
        self.feature_columns = feature_columns
        self.lookback_window = lookback_window
        self.feature_df = feature_df.copy()
        self.feature_df["date"] = pd.to_datetime(self.feature_df["date"])
        self._group_cache: dict[str, dict[str, np.ndarray]] = {}
        self._row_lookup: dict[int, tuple[str, int]] = {}
        self._prepare_group_cache()

    def _prepare_group_cache(self) -> None:
        for ticker, group in self.feature_df.groupby("ticker", sort=True):
            sorted_group = group.sort_values("date").copy()
            row_ids = sorted_group.index.to_numpy(dtype=int)
            features = sorted_group[self.feature_columns].to_numpy(dtype=np.float32)
            targets = sorted_group["target"].to_numpy(dtype=np.float32)
            dates = sorted_group["date"].dt.strftime("%Y-%m-%d").tolist()

            self._group_cache[ticker] = {
                "row_ids": row_ids,
                "features": features,
                "targets": targets,
                "dates": np.asarray(dates, dtype=object),
            }
            for position, row_id in enumerate(row_ids):
                self._row_lookup[int(row_id)] = (ticker, position)

    def build_dataset(self, indices: list[int]) -> tuple[SequenceForecastDataset, SequenceDatasetMetadata]:
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        dates: list[str] = []
        tickers: list[str] = []

        for row_id in indices:
            ticker, position = self._row_lookup[int(row_id)]
            cache = self._group_cache[ticker]
            start = position - self.lookback_window + 1
            if start < 0:
                continue

            sequence = cache["features"][start : position + 1]
            if sequence.shape[0] != self.lookback_window:
                continue

            sequences.append(sequence)
            targets.append(float(cache["targets"][position]))
            dates.append(str(cache["dates"][position]))
            tickers.append(ticker)

        if not sequences:
            raise RuntimeError("No sequence samples were created for the requested indices.")

        stacked_sequences = np.stack(sequences).astype(np.float32)
        stacked_targets = np.asarray(targets, dtype=np.float32)
        dataset = SequenceForecastDataset(
            sequences=stacked_sequences,
            targets=stacked_targets,
            dates=dates,
            tickers=tickers,
        )
        metadata = SequenceDatasetMetadata(
            num_samples=len(dataset),
            lookback_window=self.lookback_window,
            num_features=stacked_sequences.shape[-1],
        )
        return dataset, metadata

    def build_dataloader(
        self,
        indices: list[int],
        batch_size: int,
        shuffle: bool,
    ) -> tuple[DataLoader, SequenceDatasetMetadata]:
        dataset, metadata = self.build_dataset(indices)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )
        return loader, metadata
