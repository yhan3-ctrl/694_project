from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "outputs" / ".matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "outputs" / ".cache").resolve()))

import pandas as pd

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.pipeline import load_split_indices, run_baseline_training
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.runtime import get_runtime_context


def run_baselines(config_path: str) -> None:
    config = load_config(config_path)
    logger = get_logger("run_baselines")
    logger.info("Runtime context: %s", get_runtime_context())

    feature_df = pd.read_csv(config["paths"]["feature_dataset"], parse_dates=["date"])
    split_indices = load_split_indices(config["paths"]["split_indices"])
    predictions, _, costs = run_baseline_training(
        feature_df,
        split_indices,
        config,
        logger,
        model_names=config["models"]["baselines_enabled"],
    )
    logger.info("Saved %s prediction rows.", len(predictions))
    logger.info("Saved baseline cost rows: %s.", len(costs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline models.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    args = parser.parse_args()
    run_baselines(args.config)


if __name__ == "__main__":
    main()
