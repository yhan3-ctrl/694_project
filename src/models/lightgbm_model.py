from __future__ import annotations

import pandas as pd
import lightgbm as lgb

from src.models.base import BaseForecastModel


NON_FEATURE_COLUMNS = {"date", "ticker", "y_true", "sample_id"}


class LightGBMBaseline(BaseForecastModel):
    """Cross-sectional LightGBM regressor using engineered features."""

    model_name = "lightgbm"

    def __init__(self, params: dict[str, float | int]) -> None:
        self.params = params
        self.model: lgb.LGBMRegressor | None = None
        self.feature_columns: list[str] = []
        self.ticker_to_id: dict[str, int] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        prepared["ticker_id"] = prepared["ticker"].map(self.ticker_to_id).fillna(-1).astype(int)
        return prepared

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        all_tickers = sorted(train_df["ticker"].unique().tolist())
        self.ticker_to_id = {ticker: idx for idx, ticker in enumerate(all_tickers)}

        train_prepared = self._prepare_features(train_df)
        self.feature_columns = [
            column for column in train_prepared.columns if column not in NON_FEATURE_COLUMNS and column != "target"
        ]

        self.model = lgb.LGBMRegressor(
            objective="regression",
            random_state=42,
            **self.params,
        )

        fit_kwargs = {}
        if val_df is not None and not val_df.empty:
            val_prepared = self._prepare_features(val_df)
            fit_kwargs = {
                "eval_set": [(val_prepared[self.feature_columns], val_prepared["y_true"])],
                "eval_metric": "l2",
                "callbacks": [lgb.early_stopping(stopping_rounds=30, verbose=False)],
            }

        self.model.fit(train_prepared[self.feature_columns], train_prepared["y_true"], **fit_kwargs)

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("LightGBMBaseline must be fit before calling predict.")

        prepared = self._prepare_features(test_df)
        output = prepared[["date", "ticker", "y_true"]].copy()
        output["y_pred"] = self.model.predict(prepared[self.feature_columns])
        return output.sort_values(["date", "ticker"]).reset_index(drop=True)

