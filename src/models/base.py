from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseForecastModel(ABC):
    """Abstract interface for all baseline models."""

    model_name: str

    @abstractmethod
    def fit(self, train_data: Any, val_data: Any | None = None) -> None:
        """Fit the model on train data."""

    @abstractmethod
    def predict(self, test_data: Any) -> pd.DataFrame:
        """Return test predictions with at least date, ticker, and y_pred."""

    def save(self, path: str | Path) -> None:
        """Persist model state when supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save().")

    def load(self, path: str | Path) -> None:
        """Load model state when supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement load().")

    def count_parameters(self) -> dict[str, int]:
        """Return total and trainable parameter counts when meaningful."""
        return {"total_params": 0, "trainable_params": 0}
