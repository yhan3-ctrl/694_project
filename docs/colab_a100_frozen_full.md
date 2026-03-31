# Colab A100 Frozen LLM Full Run

This runbook is for the formal Frozen LLM experiment on Google Colab with an A100 GPU. It is intended to replace local full-run attempts and keep the smoke test local-only.

## Recommended Config

Use:

```bash
configs/colab_a100_frozen_full.yaml
```

Current formal Colab settings:

- `backbone_name: Qwen/Qwen2.5-0.5B-Instruct`
- `device: cuda`
- `batch_size: 8`
- `num_epochs: 4`
- `mixed_precision: false`
- `patch_size: 5`
- `lookback_window: 60`

## Before You Run

1. Open Colab and switch the runtime to `A100 GPU`.
2. Verify the device with [sanity_check.ipynb](/Users/unnh/Desktop/694%20project/sanity_check.ipynb).
3. Make sure the Colab workspace contains this repository and the prediction outputs for:
   - `arima`
   - `lightgbm`
   - `small_transformer`

The evaluation pipeline compares all models by reading `outputs/predictions/*/*.csv`, so those earlier prediction files must be present in the Colab workspace.

## Minimal Migration Path

If you already have the repository locally and only want to run Frozen LLM on Colab:

1. Copy or sync the full repository to Colab.
2. Preserve the existing `outputs/predictions/arima/`, `outputs/predictions/lightgbm/`, and `outputs/predictions/small_transformer/` directories.
3. Then run only the Frozen LLM stage plus evaluation and plotting.

## Colab Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run formal Frozen LLM:

```bash
python scripts/run_frozen_llm.py --config configs/colab_a100_frozen_full.yaml
```

Refresh metrics and figures:

```bash
python scripts/run_evaluation.py --config configs/colab_a100_frozen_full.yaml
python scripts/run_plotting.py --config configs/colab_a100_frozen_full.yaml
```

Or run the convenience wrapper:

```bash
bash scripts/run_frozen_llm_full_colab.sh
```

The wrapper now performs:

1. Colab A100 preflight validation
2. Frozen LLM formal training
3. Evaluation refresh
4. Plot refresh
5. Formal output verification

## Expected Formal Outputs

- `outputs/predictions/frozen_llm/*.csv`
- `outputs/checkpoints/frozen_llm/*.pt`
- `outputs/metrics/frozen_llm_training_history.csv`
- `outputs/tables/frozen_llm_costs.csv`
- `outputs/tables/frozen_llm_run_context.csv`
- `outputs/tables/frozen_llm_full_run_summary.csv`
- `outputs/logs/frozen_llm_full_run_summary.txt`
- `outputs/tables/main_results_with_frozen_llm.csv`
- `outputs/tables/mechanism_comparison_pretraining.csv`
- `outputs/metrics/aggregated_metrics.csv`
- `outputs/figures/overall_performance.png`
- `outputs/figures/per_regime_performance.png`
- `outputs/figures/worst_case_stability.png`
- `outputs/figures/pretraining_vs_architecture.png`
- `outputs/figures/training_curve_frozen_llm.png`

## Notes

- Do not use `--smoke-test` for the formal Colab run.
- Canonical Frozen LLM outputs should be treated as the formal result after the Colab run finishes.
- If you want to keep local smoke outputs, preserve the `_smoke` copies before syncing back.
