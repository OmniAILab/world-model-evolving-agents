#!/usr/bin/env bash
set -euo pipefail
ENV_CFG="${1:-configs/envs/toy-ledger.yaml}"
for cfg in configs/methods/*.yaml; do
  name="$(basename "$cfg" .yaml)"
  echo "== $name"
  python -m weva.cli run --config "$cfg" configs/toy.yaml --env-config "$ENV_CFG" --name "$name" \
    > /dev/null || echo "   (skipped: $name needs a backend or adapter not available offline)"
done
python scripts/summarise_runs.py
