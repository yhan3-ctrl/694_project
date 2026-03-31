from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.models.base import BaseForecastModel


class ARIMABaseline(BaseForecastModel):
    """Per-ticker ARIMA baseline over the target series."""

    model_name = "arima"

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)) -> None:
        self.order = order
        self.train_targets_: dict[str, pd.Series] = {}
        self.fallback_means_: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        combined = train_df if val_df is None else pd.concat([train_df, val_df], ignore_index=True)
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.train_targets_.clear()
        self.fallback_means_.clear()

        for ticker, group in combined.groupby("ticker", sort=True):
            target_series = group["y_true"].astype(float).reset_index(drop=True)
            self.train_targets_[ticker] = target_series
            self.fallback_means_[ticker] = float(target_series.mean())

    def _forecast_one_ticker(self, ticker: str, steps: int) -> np.ndarray:
        history = self.train_targets_[ticker]
        if steps == 0:
            return np.array([], dtype=float)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(history, order=self.order).fit()
            return np.asarray(fitted.forecast(steps=steps), dtype=float)
        except Exception:
            return np.full(steps, self.fallback_means_[ticker], dtype=float)

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        outputs: list[pd.DataFrame] = []

        for ticker, group in test_df.groupby("ticker", sort=True):
            group = group.sort_values("date").copy()
            predictions = self._forecast_one_ticker(ticker, len(group))
            output = group[["date", "ticker", "y_true"]].copy()
            output["y_pred"] = predictions
            outputs.append(output)

        return pd.concat(outputs, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

