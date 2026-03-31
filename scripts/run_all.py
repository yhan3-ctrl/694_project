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

from scripts.run_baselines import run_baselines
from scripts.run_data_pipeline import run_data_pipeline
from scripts.run_evaluation import run_evaluation
from scripts.run_frozen_llm import run_frozen_llm
from scripts.run_plotting import run_plotting
from scripts.run_transformer import run_transformer
from src.utils.logging_utils import get_logger


def run_all(config_path: str, frozen_llm_smoke_test: bool = False) -> None:
    logger = get_logger("run_all")
    logger.info("Starting full end-to-end pipeline.")
    run_data_pipeline(config_path)
    run_baselines(config_path)
    run_transformer(config_path)
    run_frozen_llm(config_path, smoke_test=frozen_llm_smoke_test)
    run_evaluation(config_path)
    run_plotting(config_path)
    logger.info("Full pipeline completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full benchmark pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    parser.add_argument(
        "--frozen-llm-smoke-test",
        action="store_true",
        help="Run the Frozen LLM stage in smoke test mode.",
    )
    args = parser.parse_args()
    run_all(args.config, frozen_llm_smoke_test=args.frozen_llm_smoke_test)


if __name__ == "__main__":
    main()
