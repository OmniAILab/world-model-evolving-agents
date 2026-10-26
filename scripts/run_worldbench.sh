#!/usr/bin/env bash
set -euo pipefail
: "${WORLDBENCH_ROOT:?set WORLDBENCH_ROOT to a clone of https://github.com/OmniAILab/WorldBench}"
: "${GEMINI_API_KEY:?set GEMINI_API_KEY}"
ENV_CFG="${ENV_CFG:-configs/envs/worldbench.yaml}"
weva run --config configs/methods/weva_doc.yaml --env-config "$ENV_CFG" --name "${NAME:-worldbench_weva_doc}" "$@"
