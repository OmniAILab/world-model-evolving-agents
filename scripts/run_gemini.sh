#!/usr/bin/env bash
set -euo pipefail
ENV_CFG="${1:?usage: bash scripts/run_gemini.sh <env-config> [split]}"
SPLIT="${2:-test}"
ENV_NAME="$(basename "$ENV_CFG" .yaml)"
for m in react free_text_pred weva_f weva_doc; do
  echo "== gemini_${ENV_NAME}_${m}"
  weva run --config "configs/methods/${m}.yaml" configs/backbones/gemini.yaml \
    --env-config "$ENV_CFG" --split "$SPLIT" --name "gemini_${ENV_NAME}_${m}"
done
python scripts/summarise_runs.py outputs
