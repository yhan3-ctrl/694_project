from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from src.models.base import BaseForecastModel
from src.utils.io import ensure_parent
from src.utils.runtime import get_peak_gpu_memory_mb, select_torch_device, set_global_seed

LOGGER = logging.getLogger(__name__)


class FrozenPatchLLMRegressor(BaseForecastModel):
    """Patch-based frozen causal LM regressor for time-series forecasting."""

    model_name = "frozen_llm"

    def __init__(
        self,
        input_dim: int,
        lookback_window: int,
        patch_size: int,
        backbone_name: str,
        projection_hidden_size: int,
        regression_hidden_size: int,
        learning_rate: float,
        num_epochs: int,
        early_stopping_patience: int,
        weight_decay: float,
        dropout: float,
        seed: int,
        checkpoint_path: str | Path,
        trust_remote_code: bool = False,
        hidden_size: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        set_global_seed(seed)
        self.input_dim = input_dim
        self.lookback_window = lookback_window
        self.patch_size = patch_size
        self.backbone_name = backbone_name
        self.projection_hidden_size = projection_hidden_size
        self.regression_hidden_size = regression_hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.dropout_rate = dropout
        self.seed = seed
        self.device = device or select_torch_device()
        self.checkpoint_path = Path(checkpoint_path)
        self.trust_remote_code = trust_remote_code

        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            trust_remote_code=trust_remote_code,
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone_dtype = self._module_floating_dtype(self.backbone)

        backbone_hidden_size = hidden_size or getattr(self.backbone.config, "hidden_size", None)
        if backbone_hidden_size is None:
            backbone_hidden_size = getattr(self.backbone.config, "n_embd", None)
        if backbone_hidden_size is None:
            backbone_hidden_size = self.backbone.get_input_embeddings().embedding_dim
        self.hidden_size = int(backbone_hidden_size)

        self.num_patches = math.ceil(self.lookback_window / self.patch_size)
        patch_input_dim = self.patch_size * self.input_dim
        self.patch_projector = nn.Sequential(
            nn.Linear(patch_input_dim, projection_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        ).to(self.device)
        self.patch_position_embeddings = nn.Embedding(self.num_patches, self.hidden_size).to(self.device)
        self.pooler = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, regression_hidden_size),
            nn.Tanh(),
        ).to(self.device)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_size, 1),
        ).to(self.device)
        self.trainable_dtype = self._module_floating_dtype(self.patch_projector)

        self.feature_mean_: torch.Tensor | None = None
        self.feature_std_: torch.Tensor | None = None
        self.history: list[dict[str, float | int]] = []
        self.training_time_seconds = 0.0
        self.inference_time_seconds = 0.0
        self.peak_gpu_memory_mb: float | None = None
        self._dtype_debug_logged = False
        LOGGER.info(
            "Initialized frozen_llm with backbone dtype=%s, trainable dtype=%s, device=%s.",
            self.backbone_dtype,
            self.trainable_dtype,
            self.device,
        )

    @staticmethod
    def _module_floating_dtype(module: nn.Module) -> torch.dtype:
        for parameter in module.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        for buffer in module.buffers():
            if buffer.is_floating_point():
                return buffer.dtype
        return torch.float32

    def _trainable_modules(self) -> list[nn.Module]:
        return [self.patch_projector, self.patch_position_embeddings, self.pooler, self.regression_head]

    def _fit_feature_scaler(self, train_loader: Any) -> None:
        train_sequences = train_loader.dataset.sequences
        self.feature_mean_ = train_sequences.mean(dim=(0, 1))
        self.feature_std_ = train_sequences.std(dim=(0, 1)).clamp_min(1e-6)

    def _normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("Feature scaler statistics are not initialized. Call fit() before predict().")

        mean = self.feature_mean_.to(inputs.device).view(1, 1, -1)
        std = self.feature_std_.to(inputs.device).view(1, 1, -1)
        return (inputs - mean) / std

    def _extract_patches(self, inputs: torch.Tensor) -> torch.Tensor:
        """Left-pad with zeros if needed, then flatten consecutive time patches."""
        batch_size, seq_len, feature_dim = inputs.shape
        padded_len = self.num_patches * self.patch_size
        if padded_len > seq_len:
            pad_steps = padded_len - seq_len
            pad_tensor = torch.zeros(batch_size, pad_steps, feature_dim, device=inputs.device, dtype=inputs.dtype)
            inputs = torch.cat([pad_tensor, inputs], dim=1)

        patches = inputs.reshape(batch_size, self.num_patches, self.patch_size * feature_dim)
        return patches

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_inputs(inputs)
        patch_tokens = self._extract_patches(normalized)
        projected = self.patch_projector(patch_tokens.to(self.trainable_dtype))

        positions = torch.arange(self.num_patches, device=inputs.device).unsqueeze(0)
        llm_inputs = projected + self.patch_position_embeddings(positions)
        llm_inputs = llm_inputs.to(dtype=self.backbone_dtype)
        attention_mask = torch.ones(llm_inputs.shape[:2], dtype=torch.long, device=inputs.device)

        if not self._dtype_debug_logged:
            LOGGER.info(
                "Frozen LLM dtype check: backbone=%s, projection_output=%s, inputs_embeds=%s.",
                self.backbone_dtype,
                projected.dtype,
                llm_inputs.dtype,
            )
            self._dtype_debug_logged = True

        outputs = self.backbone(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        pooler_dtype = self._module_floating_dtype(self.pooler)
        last_hidden = outputs.hidden_states[-1][:, -1, :].to(dtype=pooler_dtype)
        pooled = self.pooler(last_hidden)
        predictions = self.regression_head(pooled).squeeze(-1)
        return predictions.to(dtype=torch.float32)

    def _optimizer(self) -> torch.optim.Optimizer:
        trainable_params: list[torch.nn.Parameter] = []
        for module in self._trainable_modules():
            trainable_params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
        return torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def _run_epoch(
        self,
        loader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        is_training = optimizer is not None
        for module in self._trainable_modules():
            module.train(mode=is_training)
        self.backbone.eval()

        total_loss = 0.0
        num_examples = 0
        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            targets = batch["target"].to(self.device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predictions = self._forward(inputs)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for module in self._trainable_modules() for param in module.parameters() if param.requires_grad],
                    max_norm=1.0,
                )
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
        optimizer = self._optimizer()
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
                self.training_time_seconds = float(time.perf_counter() - start_time)
                self.peak_gpu_memory_mb = get_peak_gpu_memory_mb(self.device)
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
        self.backbone.eval()
        for module in self._trainable_modules():
            module.eval()

        outputs: list[pd.DataFrame] = []
        start_time = time.perf_counter()
        with torch.no_grad():
            for batch in test_data:
                inputs = batch["inputs"].to(self.device)
                predictions = self._forward(inputs).cpu().numpy()
                batch_df = pd.DataFrame(
                    {
                        "date": pd.to_datetime(batch["date"]),
                        "ticker": list(batch["ticker"]),
                        "y_true": batch["target"].cpu().numpy(),
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
                "backbone_name": self.backbone_name,
                "state_patch_projector": self.patch_projector.state_dict(),
                "state_patch_position_embeddings": self.patch_position_embeddings.state_dict(),
                "state_pooler": self.pooler.state_dict(),
                "state_regression_head": self.regression_head.state_dict(),
                "feature_mean": None if self.feature_mean_ is None else self.feature_mean_.cpu(),
                "feature_std": None if self.feature_std_ is None else self.feature_std_.cpu(),
                "history": self.history,
                "training_time_seconds": self.training_time_seconds,
                "inference_time_seconds": self.inference_time_seconds,
                "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
                "model_config": {
                    "input_dim": self.input_dim,
                    "lookback_window": self.lookback_window,
                    "patch_size": self.patch_size,
                    "hidden_size": self.hidden_size,
                    "projection_hidden_size": self.projection_hidden_size,
                    "regression_hidden_size": self.regression_hidden_size,
                },
            },
            file_path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.patch_projector.load_state_dict(payload["state_patch_projector"])
        self.patch_position_embeddings.load_state_dict(payload["state_patch_position_embeddings"])
        self.pooler.load_state_dict(payload["state_pooler"])
        self.regression_head.load_state_dict(payload["state_regression_head"])
        self.feature_mean_ = payload.get("feature_mean")
        self.feature_std_ = payload.get("feature_std")
        self.history = payload.get("history", [])
        self.training_time_seconds = float(payload.get("training_time_seconds", self.training_time_seconds))
        self.inference_time_seconds = float(payload.get("inference_time_seconds", self.inference_time_seconds))
        self.peak_gpu_memory_mb = payload.get("peak_gpu_memory_mb")

    def count_parameters(self) -> dict[str, int]:
        total_params = sum(parameter.numel() for parameter in self.backbone.parameters())
        total_params += sum(parameter.numel() for module in self._trainable_modules() for parameter in module.parameters())
        trainable_params = sum(
            parameter.numel() for module in self._trainable_modules() for parameter in module.parameters() if parameter.requires_grad
        )
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
