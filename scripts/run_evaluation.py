from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_predictions, save_metrics
from src.evaluation.reporting import build_model_cost_summary, create_frozen_llm_full_run_summary, create_results_tables
from src.utils.config import load_config
from src.utils.logging_utils import get_logger


def run_evaluation(config_path: str) -> None:
    config = load_config(config_path)
    logger = get_logger("run_evaluation")

    prediction_files = sorted(Path("outputs/predictions").glob("*/*.csv"))
    if not prediction_files:
        raise RuntimeError("No prediction files found. Run scripts/run_baselines.py first.")

    prediction_df = pd.concat(
        [pd.read_csv(path, parse_dates=["date"]) for path in prediction_files],
        ignore_index=True,
    )
    split_metrics, aggregated_metrics = evaluate_predictions(prediction_df)
    save_metrics(
        split_metrics=split_metrics,
        aggregated_metrics=aggregated_metrics,
        split_csv_path=config["paths"]["metrics_split"],
        split_json_path=config["paths"]["metrics_split_json"],
        aggregated_csv_path=config["paths"]["metrics_aggregated"],
        aggregated_json_path=config["paths"]["metrics_aggregated_json"],
    )
    cost_summary = build_model_cost_summary(config)
    create_results_tables(aggregated_metrics, cost_summary, config)
    create_frozen_llm_full_run_summary(aggregated_metrics, cost_summary, config)
    logger.info("Saved evaluation outputs for %s models.", aggregated_metrics.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation metrics.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    args = parser.parse_args()
    run_evaluation(args.config)


if __name__ == "__main__":
    main()
