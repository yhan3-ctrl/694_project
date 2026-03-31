from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "outputs" / ".matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "outputs" / ".cache").resolve()))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.utils.config import load_config
from src.utils.logging_utils import get_logger


def _check_baseline_outputs() -> list[str]:
    required_dirs = [
        Path("outputs/predictions/arima"),
        Path("outputs/predictions/lightgbm"),
        Path("outputs/predictions/small_transformer"),
    ]
    missing = [str(path) for path in required_dirs if not path.exists() or not any(path.glob("*.csv"))]
    return missing


def run_preflight(config_path: str) -> None:
    logger = get_logger("preflight_frozen_llm_colab")
    config = load_config(config_path)
    llm_cfg = config["models"]["frozen_llm"]

    logger.info("Preflight config path: %s", config_path)
    logger.info("Expected Frozen LLM config: backbone=%s", llm_cfg["backbone_name"])
    logger.info("Expected Frozen LLM config: device=%s", llm_cfg["device"])
    logger.info("Expected Frozen LLM config: batch_size=%s", llm_cfg["batch_size"])
    logger.info("Expected Frozen LLM config: num_epochs=%s", llm_cfg["num_epochs"])
    logger.info("Expected Frozen LLM config: learning_rate=%s", llm_cfg["learning_rate"])
    logger.info("Expected Frozen LLM config: patch_size=%s", llm_cfg["patch_size"])
    logger.info("Expected Frozen LLM config: lookback_window=%s", llm_cfg["lookback_window"])
    logger.info("Expected Frozen LLM config: mixed_precision=%s", llm_cfg["mixed_precision"])

    if not torch.cuda.is_available():
        raise RuntimeError("Preflight failed: CUDA is not available. This formal run must be executed on Colab A100.")

    device_name = torch.cuda.get_device_name(0)
    logger.info("Detected torch device: cuda")
    logger.info("Detected GPU name: %s", device_name)
    logger.info("Detected CUDA version: %s", torch.version.cuda)

    if "A100" not in device_name:
        raise RuntimeError(f"Preflight failed: expected an A100 GPU, but found '{device_name}'.")

    if str(llm_cfg["device"]).lower() != "cuda":
        raise RuntimeError(
            f"Preflight failed: configs/colab_a100_frozen_full.yaml should use device='cuda', found '{llm_cfg['device']}'."
        )

    missing_baselines = _check_baseline_outputs()
    if missing_baselines:
        raise RuntimeError(
            "Preflight failed: baseline prediction outputs are missing. Missing directories: "
            + ", ".join(missing_baselines)
        )

    logger.info("Confirmed baseline predictions exist for arima, lightgbm, and small_transformer.")
    logger.info("Preflight passed. Environment and inputs are ready for Frozen LLM full run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for Colab A100 Frozen LLM full run.")
    parser.add_argument("--config", default="configs/colab_a100_frozen_full.yaml", help="Path to the YAML config.")
    args = parser.parse_args()
    run_preflight(args.config)


if __name__ == "__main__":
    main()

