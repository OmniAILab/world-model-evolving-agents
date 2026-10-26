#!/bin/bash
#SBATCH --job-name=weva-phase1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase1-%j.out
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"
mkdir -p logs outputs
weva train-controller --config configs/methods/weva.yaml --env-config "${ENV_CONFIG:?set ENV_CONFIG}" \
  --backend torch --training configs/training/phase1.yaml --out outputs/controller.json
