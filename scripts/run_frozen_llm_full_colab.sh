#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/colab_a100_frozen_full.yaml}"

python scripts/preflight_frozen_llm_colab.py --config "${CONFIG_PATH}"
python scripts/run_frozen_llm.py --config "${CONFIG_PATH}"
python scripts/run_evaluation.py --config "${CONFIG_PATH}"
python scripts/run_plotting.py --config "${CONFIG_PATH}"
python scripts/verify_frozen_llm_full_outputs.py --config "${CONFIG_PATH}"
