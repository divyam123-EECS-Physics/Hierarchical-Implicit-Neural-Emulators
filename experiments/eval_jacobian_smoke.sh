#!/bin/bash
# Online Jacobian / spectral-radius analysis of the paper-smoke checkpoint
# on the eval .mat set. One GPU, debug queue (same pattern as train_paper_smoke.sh).
#
#SBATCH -A m5326
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hine_jac_smoke
#SBATCH -o /pscratch/sd/d/divyam/hine_runs/logs/jacobian_smoke-%j.out
#SBATCH -e /pscratch/sd/d/divyam/hine_runs/logs/jacobian_smoke-%j.err

set -euo pipefail

module load conda
eval "$(conda shell.bash hook)"
conda activate hine

REPO="/global/homes/d/divyam/TACSLAB/Hierarchical-Implicit-Neural-Emulators"
CKPT="/pscratch/sd/d/divyam/hine_runs/paper_smoke_1gpu/checkpoint-last.pth"
EVAL_DIR="/global/homes/d/divyam/TACSLAB/eval"
OUTPUT_DIR="/pscratch/sd/d/divyam/hine_runs/jacobian_smoke"

mkdir -p /pscratch/sd/d/divyam/hine_runs/logs
mkdir -p "${OUTPUT_DIR}"

cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0

echo "host=$(hostname) date=$(date) cwd=$(pwd)"
echo "python=$(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: missing checkpoint: ${CKPT}" >&2
  exit 1
fi
if ! compgen -G "${EVAL_DIR}/*.mat" > /dev/null; then
  echo "ERROR: no .mat files under ${EVAL_DIR}" >&2
  exit 1
fi

echo "checkpoint=${CKPT}"
echo "eval_dir=${EVAL_DIR}"
echo "output_dir=${OUTPUT_DIR}"

python eval_hine_jacobian.py \
  --checkpoint "${CKPT}" \
  --eval-dir "${EVAL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-starts 4 \
  --horizon 20 \
  --power-iters 15 \
  --device cuda

echo "JACOBIAN_SMOKE_DONE date=$(date)"
ls -lh "${OUTPUT_DIR}/summary.txt" "${OUTPUT_DIR}/trajectories.npz"
