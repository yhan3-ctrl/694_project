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

from src.utils.config import load_config
from src.utils.io import save_json
from src.utils.logging_utils import get_logger


def verify_outputs(config_path: str) -> None:
    logger = get_logger("verify_frozen_llm_full_outputs")
    config = load_config(config_path)

    required_files = [
        Path("outputs/metrics/frozen_llm_training_history.csv"),
        Path("outputs/metrics/aggregated_metrics.csv"),
        Path("outputs/metrics/split_level_metrics.csv"),
        Path("outputs/tables/frozen_llm_costs.csv"),
        Path("outputs/tables/main_results_with_frozen_llm.csv"),
        Path("outputs/tables/mechanism_comparison_pretraining.csv"),
        Path("outputs/figures/training_curve_frozen_llm.png"),
        Path("outputs/figures/pretraining_vs_architecture.png"),
        Path("outputs/figures/overall_performance.png"),
        Path("outputs/figures/per_regime_performance.png"),
        Path("outputs/figures/worst_case_stability.png"),
        Path(config["paths"]["frozen_llm_full_run_summary"]),
        Path(config["paths"]["frozen_llm_run_context"]),
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    prediction_files = sorted(Path("outputs/predictions/frozen_llm").glob("*.csv"))
    if not prediction_files:
        missing.append("outputs/predictions/frozen_llm/*.csv")

    if missing:
        raise RuntimeError("Frozen LLM full-run verification failed. Missing outputs: " + ", ".join(missing))

    aggregated = pd.read_csv("outputs/metrics/aggregated_metrics.csv")
    frozen_metrics = aggregated[aggregated["model_name"] == "frozen_llm"]
    if frozen_metrics.empty:
        raise RuntimeError("Frozen LLM full-run verification failed: frozen_llm is missing from aggregated_metrics.csv.")

    cost_df = pd.read_csv("outputs/tables/frozen_llm_costs.csv")
    if cost_df.empty:
        raise RuntimeError("Frozen LLM full-run verification failed: frozen_llm_costs.csv is empty.")

    context_df = pd.read_csv(config["paths"]["frozen_llm_run_context"])
    if context_df.empty or str(context_df.iloc[0]["run_mode"]) != "full":
        raise RuntimeError("Frozen LLM full-run verification failed: run context is missing or not marked as full.")

    summary_text = [
        "Frozen LLM full-run verification passed.",
        f"Predictions files: {len(prediction_files)}",
        f"Run mode: {context_df.iloc[0]['run_mode']}",
        f"Device: {context_df.iloc[0]['device']}",
        f"Backbone: {context_df.iloc[0]['backbone_name']}",
        f"Batch size: {context_df.iloc[0]['batch_size']}",
        f"Epochs: {context_df.iloc[0]['num_epochs']}",
    ]
    summary_path = Path("outputs/logs/frozen_llm_full_run_summary.txt")
    summary_path.write_text("\n".join(summary_text), encoding="utf-8")
    logger.info("Verification passed for Frozen LLM full run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Frozen LLM full-run outputs.")
    parser.add_argument("--config", default="configs/colab_a100_frozen_full.yaml", help="Path to the YAML config.")
    args = parser.parse_args()
    verify_outputs(args.config)


if __name__ == "__main__":
    main()

