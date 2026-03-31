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

from src.training.frozen_llm_pipeline import run_frozen_llm_training
from src.training.pipeline import load_split_indices
from src.utils.config import load_config
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger
from src.utils.runtime import get_runtime_context, select_torch_device


def run_frozen_llm(config_path: str, smoke_test: bool = False) -> None:
    config = load_config(config_path)
    llm_cfg = config["models"]["frozen_llm"]
    logger = get_logger("run_frozen_llm")
    runtime_context = get_runtime_context()
    selected_device = select_torch_device()
    logger.info("Runtime context: %s", runtime_context)
    logger.info("Selected torch device: %s", selected_device)
    logger.info("Smoke test mode: %s", smoke_test)

    feature_df = pd.read_csv(config["paths"]["feature_dataset"], parse_dates=["date"])
    split_indices = load_split_indices(config["paths"]["split_indices"])
    predictions, _, histories, costs = run_frozen_llm_training(
        feature_df=feature_df,
        split_indices=split_indices,
        config=config,
        logger=logger,
        smoke_test=smoke_test,
    )
    run_mode = "smoke" if smoke_test else "full"
    active_backbone = llm_cfg["smoke_test"]["backbone_name"] if smoke_test else llm_cfg["backbone_name"]
    active_batch_size = llm_cfg["smoke_test"]["batch_size"] if smoke_test else llm_cfg["batch_size"]
    active_num_epochs = llm_cfg["smoke_test"]["num_epochs"] if smoke_test else llm_cfg["num_epochs"]
    context_df = pd.DataFrame(
        [
            {
                "model_name": "frozen_llm",
                "run_mode": run_mode,
                "device": str(selected_device),
                "backbone_name": active_backbone,
                "batch_size": active_batch_size,
                "num_epochs": active_num_epochs,
                "mixed_precision": bool(llm_cfg["mixed_precision"]),
                "nvidia_smi_available": runtime_context["nvidia_smi_available"],
                "gpu_name": runtime_context["gpu_name"],
                "driver_version": runtime_context["driver_version"],
            }
        ]
    )
    save_dataframe(context_df, config["paths"]["frozen_llm_run_context"], index=False)
    logger.info("Saved frozen LLM prediction rows: %s.", len(predictions))
    logger.info("Saved frozen LLM history rows: %s and cost rows: %s.", len(histories), len(costs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen LLM baseline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Enable smoke test mode for quick validation.")
    args = parser.parse_args()
    run_frozen_llm(args.config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
