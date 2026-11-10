#!/bin/bash
# Task 4 (Section 13.5): eigenvalues of DB on one already allocated 4-GPU node.
#
# This script does not call sbatch, salloc, srun, or torchrun.
# It starts one Python process. That process gives each worker one GPU.
#
#   bash eval/run_task4_interactive.sh smoke
#   bash eval/run_task4_interactive.sh production

set -euo pipefail

MODE="${1:-production}"
REPO="/global/homes/d/divyam/TACSLAB/Hierarchical-Implicit-Neural-Emulators"
CKPT="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth"
ROOT="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task4_db"

cd "${REPO}"

if ! python -c "import torch" >/dev/null 2>&1; then
  module load conda
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate hine
fi

NGPU="$(python -c "import torch; print(torch.cuda.device_count())")"
if [[ "${NGPU}" -ne 4 ]]; then
  echo "Need exactly 4 visible GPUs, found ${NGPU}." >&2
  echo "Allocate one 4-GPU interactive node first. This script does not request one." >&2
  exit 1
fi

case "${MODE}" in
  smoke)
    N_STATES=4
    K=8
    NCV=24
    SAVE_DIR="${ROOT}/smoke"
    ;;
  production|n20)
    N_STATES=20
    K=12
    NCV=32
    SAVE_DIR="${ROOT}/n20"
    ;;
  *)
    echo "usage: bash eval/run_task4_interactive.sh {smoke|production}" >&2
    exit 2
    ;;
esac

mkdir -p "${SAVE_DIR}"
echo "task4 mode=${MODE} states=${N_STATES} k=${K} gpus=${NGPU} save_dir=${SAVE_DIR}"

exec python -u eval/task3_parasitic_arnoldi.py \
  --db-only \
  --n-gpus 4 \
  --n-states "${N_STATES}" \
  --k "${K}" \
  --ncv "${NCV}" \
  --maxiter 120 \
  --checkpoint "${CKPT}" \
  --save-dir "${SAVE_DIR}"
