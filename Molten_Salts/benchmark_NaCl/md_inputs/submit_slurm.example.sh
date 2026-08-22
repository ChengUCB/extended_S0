#!/bin/bash
#SBATCH --job-name=nacl-npt
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=out.%j
#SBATCH --error=err.%j

set -euo pipefail

: "${INITIAL_XYZ:?Set INITIAL_XYZ}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

extra_args=()
if [[ "${DIRECT_REPLICA:-0}" == "1" ]]; then
    extra_args+=(--direct-replica)
fi

python npt_md.py "$INITIAL_XYZ" "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" "${extra_args[@]}"
