from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from src.models.base import BaseForecastModel
from src.utils.io import ensure_parent
from src.utils.runtime import get_peak_gpu_memory_mb, select_torch_device, set_global_seed


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence inputs."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderRegressor(nn.Module):
    """Encoder-only transformer with mean pooling and a regression head."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position_encoding(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


class SmallTransformerRegressor(BaseForecastModel):
    """Small transformer wrapper with training, prediction, checkpointing, and cost tracking."""

    model_name = "small_transformer"

    def __init__(
        self,
        input_dim: int,
        lookback_window: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        dim_feedforward: int,
        learning_rate: float,
        num_epochs: int,
        early_stopping_patience: int,
        weight_decay: float,
        seed: int,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
    ) -> None:
        set_global_seed(seed)
        self.input_dim = input_dim
        self.lookback_window = lookback_window
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device or select_torch_device()
        self.checkpoint_path = Path(checkpoint_path)

        self.network = TransformerEncoderRegressor(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        ).to(self.device)
        self.feature_mean_: torch.Tensor | None = None
        self.feature_std_: torch.Tensor | None = None
        self.history: list[dict[str, float | int]] = []
        self.training_time_seconds = 0.0
        self.inference_time_seconds = 0.0
        self.peak_gpu_memory_mb: float | None = None

    def _normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("Feature scaler statistics are not initialized. Call fit() before predict().")

        mean = self.feature_mean_.to(inputs.device).view(1, 1, -1)
        std = self.feature_std_.to(inputs.device).view(1, 1, -1)
        return (inputs - mean) / std

    def _fit_feature_scaler(self, train_loader: Any) -> None:
        train_sequences = train_loader.dataset.sequences
        self.feature_mean_ = train_sequences.mean(dim=(0, 1))
        self.feature_std_ = train_sequences.std(dim=(0, 1)).clamp_min(1e-6)

    def _run_epoch(
        self,
        loader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        is_training = optimizer is not None
        self.network.train(mode=is_training)
        total_loss = 0.0
        num_examples = 0

        for batch in loader:
            inputs = self._normalize_inputs(batch["inputs"].to(self.device))
            targets = batch["target"].to(self.device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predictions = self.network(inputs)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += float(loss.item()) * batch_size
            num_examples += int(batch_size)

        return total_loss / max(1, num_examples)

    def fit(self, train_data: Any, val_data: Any | None = None) -> None:
        set_global_seed(self.seed)
        self.history = []
        full_history: list[dict[str, float | int]] = []
        self._fit_feature_scaler(train_data)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_epoch(train_data, criterion=criterion, optimizer=optimizer)
            val_loss = train_loss if val_data is None else self._run_epoch(val_data, criterion=criterion, optimizer=None)
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            full_history = list(self.history)

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                patience_counter = 0
                self.save(self.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        self.training_time_seconds = float(time.perf_counter() - start_time)
        self.peak_gpu_memory_mb = get_peak_gpu_memory_mb(self.device)
        self.load(self.checkpoint_path)
        self.history = full_history
        self.training_time_seconds = float(time.perf_counter() - start_time)
        self.peak_gpu_memory_mb = get_peak_gpu_memory_mb(self.device)
        self.save(self.checkpoint_path)

    def predict(self, test_data: Any) -> pd.DataFrame:
        self.network.eval()
        start_time = time.perf_counter()

        outputs: list[pd.DataFrame] = []
        with torch.no_grad():
            for batch in test_data:
                inputs = self._normalize_inputs(batch["inputs"].to(self.device))
                targets = batch["target"].cpu().numpy()
                predictions = self.network(inputs).cpu().numpy()

                batch_df = pd.DataFrame(
                    {
                        "date": pd.to_datetime(batch["date"]),
                        "ticker": list(batch["ticker"]),
                        "y_true": targets,
                        "y_pred": predictions,
                    }
                )
                outputs.append(batch_df)

        self.inference_time_seconds = float(time.perf_counter() - start_time)
        return pd.concat(outputs, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        file_path = ensure_parent(path)
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "model_config": {
                    "input_dim": self.input_dim,
                    "lookback_window": self.lookback_window,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "dim_feedforward": self.dim_feedforward,
                },
                "feature_mean": None if self.feature_mean_ is None else self.feature_mean_.cpu(),
                "feature_std": None if self.feature_std_ is None else self.feature_std_.cpu(),
                "history": self.history,
                "training_time_seconds": self.training_time_seconds,
                "inference_time_seconds": self.inference_time_seconds,
                "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            },
            file_path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.network.load_state_dict(payload["state_dict"])
        self.network.to(self.device)
        self.feature_mean_ = payload["feature_mean"]
        self.feature_std_ = payload["feature_std"]
        self.history = payload.get("history", [])
        self.training_time_seconds = float(payload.get("training_time_seconds", self.training_time_seconds))
        self.inference_time_seconds = float(payload.get("inference_time_seconds", self.inference_time_seconds))
        self.peak_gpu_memory_mb = payload.get("peak_gpu_memory_mb")

    def count_parameters(self) -> dict[str, int]:
        total_params = sum(parameter.numel() for parameter in self.network.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.network.parameters() if parameter.requires_grad)
        return {"total_params": int(total_params), "trainable_params": int(trainable_params)}

    def get_cost_summary(self) -> dict[str, float | int | None]:
        counts = self.count_parameters()
        return {
            "training_time_seconds": self.training_time_seconds,
            "inference_time_seconds": self.inference_time_seconds,
            "total_params": counts["total_params"],
            "trainable_params": counts["trainable_params"],
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }
