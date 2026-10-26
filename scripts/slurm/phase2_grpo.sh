#!/bin/bash
#SBATCH --job-name=weva-phase2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/phase2-%j.out
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"
mkdir -p logs
weva train-phase2 --config configs/methods/weva.yaml --env-config "${ENV_CONFIG:?set ENV_CONFIG}" \
  --backend torch --training configs/training/phase2.yaml --controller outputs/controller.json
