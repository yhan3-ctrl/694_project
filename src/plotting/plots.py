from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def _model_colors(models: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {model: cmap(idx % 10) for idx, model in enumerate(models)}


def plot_data_overview(feature_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Plot close price and daily return series for a few representative tickers."""
    output_dir = ensure_dir(output_dir)
    tickers = ["SPY", "AAPL", "NVDA"]
    plot_df = feature_df[feature_df["ticker"].isin(tickers)].copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for ticker in tickers:
        ticker_df = plot_df[plot_df["ticker"] == ticker]
        axes[0].plot(ticker_df["date"], ticker_df["close"], label=ticker)
        axes[1].plot(ticker_df["date"], ticker_df["log_return_1"], label=ticker)

    axes[0].set_title("Data Overview: Close Price Time Series")
    axes[0].set_ylabel("Close Price")
    axes[0].legend()

    axes[1].set_title("Data Overview: Daily Log Return Time Series")
    axes[1].set_ylabel("Log Return")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "data_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regime_slicing(prediction_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Visualize rolling volatility and shock windows for one representative split."""
    output_dir = ensure_dir(output_dir)
    split_id = sorted(prediction_df["split_id"].unique())[0]
    split_df = prediction_df[prediction_df["split_id"] == split_id].copy()
    date_df = (
        split_df.groupby("date", as_index=False)
        .agg(rolling_vol_20=("rolling_vol_20", "first"), is_shock=("is_shock", "max"))
        .sort_values("date")
    )
    date_df["date"] = pd.to_datetime(date_df["date"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(date_df["date"], date_df["rolling_vol_20"], color="steelblue", label="20-day rolling volatility")
    shock_dates = date_df.loc[date_df["is_shock"], "date"]
    for idx, shock_date in enumerate(shock_dates):
        ax.axvline(shock_date, color="crimson", alpha=0.15, linewidth=2, label="shock window" if idx == 0 else None)

    ax.set_title(f"Regime Slicing on Test Segment: {split_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "regime_slicing.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overall_performance(aggregated_metrics: pd.DataFrame, output_dir: str | Path, frozen_llm_run_label: str | None = None) -> None:
    """Compare overall MAE and direction accuracy across models."""
    output_dir = ensure_dir(output_dir)
    models = aggregated_metrics["model_name"].tolist()
    colors = _model_colors(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(aggregated_metrics["model_name"], aggregated_metrics["overall_mae"], color=[colors[model] for model in models])
    title_suffix = f" | Frozen LLM: {frozen_llm_run_label}" if frozen_llm_run_label else ""
    axes[0].set_title(f"Overall MAE{title_suffix}")
    axes[0].set_ylabel("MAE")

    axes[1].bar(
        aggregated_metrics["model_name"],
        aggregated_metrics["overall_direction_accuracy"],
        color=[colors[model] for model in models],
    )
    axes[1].set_title(f"Overall Direction Accuracy{title_suffix}")
    axes[1].set_ylabel("Accuracy")

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "overall_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_regime_performance(
    aggregated_metrics: pd.DataFrame,
    output_dir: str | Path,
    frozen_llm_run_label: str | None = None,
) -> None:
    """Compare MAE by regime across models."""
    output_dir = ensure_dir(output_dir)
    regimes = ["vol_low", "vol_mid", "vol_high", "shock"]
    models = aggregated_metrics["model_name"].tolist()
    colors = _model_colors(models)
    x = list(range(len(regimes)))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, model_name in enumerate(models):
        row = aggregated_metrics[aggregated_metrics["model_name"] == model_name].iloc[0]
        values = [row[f"{regime}_mae"] for regime in regimes]
        offset = idx - (len(models) - 1) / 2
        positions = [value + offset * width for value in x]
        ax.bar(positions, values, width=width, label=model_name, color=colors[model_name])

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("MAE")
    title_suffix = f" | Frozen LLM: {frozen_llm_run_label}" if frozen_llm_run_label else ""
    ax.set_title(f"Per-Regime Performance Comparison{title_suffix}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "per_regime_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_worst_case_stability(
    aggregated_metrics: pd.DataFrame,
    output_dir: str | Path,
    frozen_llm_run_label: str | None = None,
) -> None:
    """Compare worst-case stability metrics across models."""
    output_dir = ensure_dir(output_dir)
    models = aggregated_metrics["model_name"].tolist()
    colors = _model_colors(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(aggregated_metrics["model_name"], aggregated_metrics["wcs_error"], color=[colors[model] for model in models])
    title_suffix = f" | Frozen LLM: {frozen_llm_run_label}" if frozen_llm_run_label else ""
    axes[0].set_title(f"Worst-Case Stability Error{title_suffix}")
    axes[0].set_ylabel("WCS Error")

    axes[1].bar(aggregated_metrics["model_name"], aggregated_metrics["wcs_acc"], color=[colors[model] for model in models])
    axes[1].set_title(f"Worst-Case Stability Accuracy{title_suffix}")
    axes[1].set_ylabel("WCS Accuracy")

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "worst_case_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve_transformer(history_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Plot average transformer train and validation losses across epochs."""
    if history_df.empty:
        return

    output_dir = ensure_dir(output_dir)
    averaged = history_df.groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(averaged["epoch"], averaged["train_loss"], marker="o", label="train loss", color="#4C78A8")
    ax.plot(averaged["epoch"], averaged["val_loss"], marker="o", label="val loss", color="#F58518")
    ax.set_title("Small Transformer Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curve_transformer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve_frozen_llm(
    history_df: pd.DataFrame,
    output_dir: str | Path,
    frozen_llm_run_label: str | None = None,
) -> None:
    """Plot average frozen LLM train and validation losses across epochs."""
    if history_df.empty:
        return

    output_dir = ensure_dir(output_dir)
    averaged = history_df.groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(averaged["epoch"], averaged["train_loss"], marker="o", label="train loss", color="#72B7B2")
    ax.plot(averaged["epoch"], averaged["val_loss"], marker="o", label="val loss", color="#E45756")
    label_suffix = f" ({frozen_llm_run_label})" if frozen_llm_run_label else ""
    ax.set_title(f"Frozen LLM Training Curve{label_suffix}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curve_frozen_llm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_splits_transformer(split_metrics: pd.DataFrame, output_dir: str | Path) -> None:
    """Show how the small transformer varies across rolling splits."""
    output_dir = ensure_dir(output_dir)
    transformer_rows = split_metrics[
        (split_metrics["model_name"] == "small_transformer") & (split_metrics["split_id"].str.startswith("rolling_"))
    ].copy()
    if transformer_rows.empty:
        return

    transformer_rows = transformer_rows.sort_values("split_id").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        transformer_rows["split_id"],
        transformer_rows["overall_mae"],
        marker="o",
        color="#54A24B",
    )
    axes[0].set_title("Small Transformer Across Rolling Splits: Overall MAE")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("MAE")

    axes[1].plot(
        transformer_rows["split_id"],
        transformer_rows["wcs_error"],
        marker="o",
        color="#E45756",
    )
    axes[1].set_title("Small Transformer Across Rolling Splits: WCS Error")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("WCS Error")

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "rolling_splits_transformer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pretraining_vs_architecture(
    aggregated_metrics: pd.DataFrame,
    output_dir: str | Path,
    frozen_llm_run_label: str | None = None,
) -> None:
    """Compare the small transformer and frozen LLM across overall and stress metrics."""
    output_dir = ensure_dir(output_dir)
    compare_df = aggregated_metrics[aggregated_metrics["model_name"].isin(["small_transformer", "frozen_llm"])].copy()
    if compare_df.empty:
        return

    metrics = ["overall_mae", "vol_high_mae", "shock_mae", "wcs_error"]
    titles = ["Overall MAE", "Vol High MAE", "Shock MAE", "WCS Error"]
    colors = _model_colors(compare_df["model_name"].tolist())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric_name, title in zip(axes.flatten(), metrics, titles):
        ax.bar(
            compare_df["model_name"],
            compare_df[metric_name],
            color=[colors[model] for model in compare_df["model_name"]],
        )
        ax.set_title(title)
        ax.set_ylabel(metric_name)

    label_suffix = f" ({frozen_llm_run_label})" if frozen_llm_run_label else ""
    fig.suptitle(f"Pretraining vs Architecture: Small Transformer vs Frozen LLM{label_suffix}", y=1.02)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "pretraining_vs_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
