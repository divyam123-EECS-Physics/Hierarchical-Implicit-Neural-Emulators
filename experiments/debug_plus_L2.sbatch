#!/bin/bash
#SBATCH --job-name=hine-plus-L2
#SBATCH --account=m5326_g
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/debug_plus_L2-%j.out
#SBATCH --mail-user=divyam123@berkeley.edu
#SBATCH --mail-type=FAIL

# FutureINNPlus L=2: hierarchy_L=2, ds_factors=8 -> latent_res 32 (3 frames).
# One epoch smoke on 1 GPU.

set -euo pipefail
cd /global/homes/d/divyam/TACSLAB/Hierarchical-Implicit-Neural-Emulators
mkdir -p logs output/output_debug_plus_L2

export PYTHONUNBUFFERED=1
while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${port}
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

/global/homes/d/divyam/.conda/envs/hine/bin/torchrun \
  --nnodes=1 --nproc_per_node=1 --master_port="${MASTER_PORT}" \
  main_plus.py \
  --img_size 256 \
  --batch_size 8 \
  --epochs 1 \
  --eval_freq 1 \
  --save_last_freq 1 \
  --hierarchy_L 2 \
  --ds_factors 8 \
  --num_workers 4 \
  --output_dir output/output_debug_plus_L2 \
  --resume output/output_debug_plus_L2

echo "PLUS_L2_DEBUG_DONE"
