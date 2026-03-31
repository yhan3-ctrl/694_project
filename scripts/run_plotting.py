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

from src.plotting.plots import (
    plot_data_overview,
    plot_overall_performance,
    plot_per_regime_performance,
    plot_pretraining_vs_architecture,
    plot_regime_slicing,
    plot_rolling_splits_transformer,
    plot_training_curve_frozen_llm,
    plot_training_curve_transformer,
    plot_worst_case_stability,
)
from src.utils.config import load_config
from src.utils.logging_utils import get_logger


def run_plotting(config_path: str) -> None:
    config = load_config(config_path)
    logger = get_logger("run_plotting")

    feature_df = pd.read_csv(config["paths"]["feature_dataset"], parse_dates=["date"])
    aggregated_metrics = pd.read_csv(config["paths"]["metrics_aggregated"])
    split_metrics = pd.read_csv(config["paths"]["metrics_split"])
    prediction_files = sorted(Path("outputs/predictions").glob("*/*.csv"))
    prediction_df = pd.concat([pd.read_csv(path, parse_dates=["date"]) for path in prediction_files], ignore_index=True)
    history_path = Path(config["paths"]["transformer_history"])
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    frozen_llm_history_path = Path(config["paths"]["frozen_llm_history"])
    frozen_llm_history_df = pd.read_csv(frozen_llm_history_path) if frozen_llm_history_path.exists() else pd.DataFrame()
    run_context_path = Path(config["paths"]["frozen_llm_run_context"])
    run_context_df = pd.read_csv(run_context_path) if run_context_path.exists() else pd.DataFrame()
    frozen_llm_run_label = None
    if not run_context_df.empty and "run_mode" in run_context_df.columns:
        frozen_llm_run_label = str(run_context_df.iloc[0]["run_mode"])

    output_dir = Path("outputs/figures")
    plot_data_overview(feature_df, output_dir)
    plot_regime_slicing(prediction_df, output_dir)
    plot_overall_performance(aggregated_metrics, output_dir, frozen_llm_run_label=frozen_llm_run_label)
    plot_per_regime_performance(aggregated_metrics, output_dir, frozen_llm_run_label=frozen_llm_run_label)
    plot_worst_case_stability(aggregated_metrics, output_dir, frozen_llm_run_label=frozen_llm_run_label)
    plot_training_curve_transformer(history_df, output_dir)
    plot_training_curve_frozen_llm(
        frozen_llm_history_df,
        output_dir,
        frozen_llm_run_label=frozen_llm_run_label,
    )
    plot_rolling_splits_transformer(split_metrics, output_dir)
    plot_pretraining_vs_architecture(
        aggregated_metrics,
        output_dir,
        frozen_llm_run_label=frozen_llm_run_label,
    )
    logger.info("Saved plotting outputs to %s.", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plotting for benchmark outputs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    args = parser.parse_args()
    run_plotting(args.config)


if __name__ == "__main__":
    main()
