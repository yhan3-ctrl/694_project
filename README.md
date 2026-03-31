# Financial Forecasting Stress-Test Benchmark

This repository now includes Phase 1, Phase 2, and Phase 3 for a reproducible financial time-series forecasting benchmark focused on regime robustness under distribution shift. The current scope builds the full baseline-ready pipeline for next-day return forecasting, including data ingestion, leakage-safe feature engineering, time-ordered splits, regime slicing, evaluation, two classical baselines (`ARIMA`, `LightGBM`), a sequence-based `Small Transformer` control model, and a `Frozen LLM` pretraining test baseline.

## Project Goals

- Forecast next-day log return for a fixed universe of liquid US assets.
- Evaluate not only overall average performance, but also performance in stress regimes.
- Build a reusable benchmark system that can later host stronger models such as frozen LLM and LoRA variants.

## Directory Layout

```text
configs/                 YAML configuration
data/raw/                Raw Yahoo Finance downloads
data/processed/          Cleaned aligned price data and features
src/data/                Download and cleaning logic
src/features/            Feature engineering
src/splits/              Time-based split generation
src/regimes/             Stress-regime slicing
src/models/              Baseline model interfaces and implementations
src/training/            End-to-end training and prediction pipeline
src/evaluation/          Metric computation and aggregation
src/plotting/            Plot generation
src/utils/               Shared config, I/O, logging, runtime helpers
outputs/datasets/        Saved datasets and split metadata
outputs/predictions/     Model predictions by split
outputs/metrics/         Split-level and aggregated metrics
outputs/tables/          Summary tables
outputs/figures/         Generated figures
outputs/checkpoints/     Saved model checkpoints
outputs/logs/            Run logs
scripts/                 Executable entry points
```

## Environment Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you are running inside Google Colab, the same `requirements.txt` can be installed there as well. The current `Small Transformer` and `Frozen LLM` baselines support automatic CPU / GPU device selection. In this project stage, the Frozen LLM pipeline includes a local smoke test path and can later be moved to Colab A100 for larger backbones and full experiments.

For the formal Frozen LLM run on Colab A100, use:

- config: [`configs/colab_a100_frozen_full.yaml`](/Users/unnh/Desktop/694%20project/configs/colab_a100_frozen_full.yaml)
- wrapper: [`scripts/run_frozen_llm_full_colab.sh`](/Users/unnh/Desktop/694%20project/scripts/run_frozen_llm_full_colab.sh)
- runbook: [`docs/colab_a100_frozen_full.md`](/Users/unnh/Desktop/694%20project/docs/colab_a100_frozen_full.md)
- preflight: [`scripts/preflight_frozen_llm_colab.py`](/Users/unnh/Desktop/694%20project/scripts/preflight_frozen_llm_colab.py)
- verifier: [`scripts/verify_frozen_llm_full_outputs.py`](/Users/unnh/Desktop/694%20project/scripts/verify_frozen_llm_full_outputs.py)

## Default Dataset

- Tickers: `SPY`, `QQQ`, `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `NVDA`, `TSLA`, `JPM`, `XOM`, `JNJ`
- Source: Yahoo Finance
- Date range: `2015-01-01` to `2025-12-31`
- Target: next-day log return

## Run Order

### 1. Data pipeline

```bash
python scripts/run_data_pipeline.py --config configs/default.yaml
```

Outputs:
- `data/raw/yahoo_ohlcv_raw.csv`
- `data/processed/daily_ohlcv_processed.csv`
- `data/processed/model_features.csv`
- `outputs/tables/dataset_summary.csv`
- `outputs/tables/split_summary.csv`
- `outputs/datasets/split_indices.json`

### 2. Train classical baselines and save predictions

```bash
python scripts/run_baselines.py --config configs/default.yaml
```

Outputs:
- `outputs/predictions/arima/*.csv`
- `outputs/predictions/lightgbm/*.csv`

Each prediction file includes:
- `date`
- `ticker`
- `y_true`
- `y_pred`
- `split_id`
- regime labels

### 3. Train the Small Transformer

```bash
python scripts/run_transformer.py --config configs/default.yaml
```

Outputs:
- `outputs/predictions/small_transformer/*.csv`
- `outputs/checkpoints/small_transformer/*.pt`
- `outputs/metrics/small_transformer_training_history.csv`
- `outputs/tables/small_transformer_costs.csv`

The transformer uses:
- sequence input with lookback window `L=60`
- encoder-only transformer
- batch training
- early stopping with best-checkpoint restore
- automatic device selection

### 4. Train the Frozen LLM baseline

Smoke test:

```bash
python scripts/run_frozen_llm.py --config configs/default.yaml --smoke-test
```

Full run:

```bash
python scripts/run_frozen_llm.py --config configs/default.yaml
```

Formal Colab A100 full run:

```bash
python scripts/run_frozen_llm.py --config configs/colab_a100_frozen_full.yaml
```

Outputs:
- `outputs/predictions/frozen_llm/*.csv`
- `outputs/checkpoints/frozen_llm/*.pt`
- `outputs/metrics/frozen_llm_training_history.csv`
- `outputs/tables/frozen_llm_costs.csv`

Frozen LLM design:
- sequence input with lookback window `L=60`
- patch extraction over numeric features
- trainable reprogramming projection into the frozen causal LM hidden space
- fully frozen Hugging Face causal LM backbone
- trainable pooling and regression head

Important config keys under `models.frozen_llm`:
- `backbone_name`
- `patch_size`
- `hidden_size`
- `projection_hidden_size`
- `batch_size`
- `learning_rate`
- `num_epochs`
- `early_stopping_patience`
- `device`
- `smoke_test`

Smoke test mode supports:
- lightweight backbone override
- fewer epochs
- reduced sample counts
- limited ticker set
- limited split set

### 5. Evaluate metrics

```bash
python scripts/run_evaluation.py --config configs/default.yaml
```

Outputs:
- `outputs/metrics/split_level_metrics.csv`
- `outputs/metrics/split_level_metrics.json`
- `outputs/metrics/aggregated_metrics.csv`
- `outputs/metrics/aggregated_metrics.json`
- `outputs/tables/model_cost_summary.csv`
- `outputs/tables/main_results_with_transformer.csv`
- `outputs/tables/mechanism_comparison_initial.csv`
- `outputs/tables/main_results_with_frozen_llm.csv`
- `outputs/tables/mechanism_comparison_pretraining.csv`

### 6. Generate figures

```bash
python scripts/run_plotting.py --config configs/default.yaml
```

Outputs:
- data overview figure
- regime slicing figure
- overall performance comparison
- per-regime performance comparison
- worst-case stability comparison
- transformer training curve
- transformer rolling-split variation figure
- frozen LLM training curve
- pretraining vs architecture comparison

### 7. Run everything end to end

Standard full pipeline:

```bash
python scripts/run_all.py --config configs/default.yaml
```

Full pipeline with the Frozen LLM stage in smoke-test mode:

```bash
python scripts/run_all.py --config configs/default.yaml --frozen-llm-smoke-test
```

## Implementation Notes

- All splits are strictly time ordered; there is no shuffling.
- Rolling features only use current or historical information.
- Split metadata saves both date boundaries and row indices.
- Regime slicing is applied on the test segment only.
- Evaluation reports overall metrics, per-regime metrics, a tail-risk metric, and worst-case stability metrics.
- The Small Transformer consumes sequence tensors built from the same engineered feature table without changing the Phase 1 baseline pipeline.
- Feature normalization for the transformer is fit on training data only.
- Transformer checkpoints and training histories are saved automatically.
- The Frozen LLM baseline reuses the same sequence interface and applies numeric patching before feeding frozen backbone embeddings.
- If `lookback_window` is not divisible by `patch_size`, the Frozen LLM pipeline uses left zero padding so the most recent observations are preserved.
- Frozen LLM smoke tests are intended for interface validation on local hardware before scaling up to larger backbones on Colab A100.

## Main Outputs

- Dataset summary table for coverage and missingness
- Split summary table with train/val/test boundaries
- Prediction files for each model and split
- Split-level and aggregated metrics in CSV and JSON
- Transformer checkpoints and cost summaries
- Frozen LLM checkpoints, cost summaries, and smoke-test outputs
- Figures under `outputs/figures/`

## Current Model Set

- `ARIMA`: per-ticker univariate classical baseline
- `LightGBM`: tabular cross-sectional baseline over engineered features
- `Small Transformer`: sequence-based architecture control baseline for later frozen-LLM comparison
- `Frozen LLM`: patch-based numeric reprogramming baseline with a frozen causal LM backbone

## Next Steps After Phase 3

- Add stronger deep-learning and frozen-LLM baselines
- Add LoRA-based adaptation experiments
- Expand stress-regime definitions and statistical tests
