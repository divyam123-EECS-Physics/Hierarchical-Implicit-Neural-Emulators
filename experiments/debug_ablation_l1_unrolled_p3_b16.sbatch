#!/bin/bash
#SBATCH --job-name=hine-abl-p3b16
#SBATCH --account=m5326_g
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/debug_ablation_l1_unrolled_p3_b16-%j.out
#SBATCH --mail-user=divyam123@berkeley.edu
#SBATCH --mail-type=FAIL

# L=1 unrolled training P=3, batch 16 (1 epoch memory smoke).

set -euo pipefail
cd /global/homes/d/divyam/TACSLAB/Hierarchical-Implicit-Neural-Emulators
mkdir -p logs output/output_debug_ablation_l1_unrolled_p3_b16

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
  main_ablation.py \
  --ablation l1_unrolled \
  --train_unroll 3 \
  --img_size 256 \
  --batch_size 16 \
  --epochs 1 \
  --eval_freq 1 \
  --save_last_freq 1 \
  --num_workers 4 \
  --output_dir output/output_debug_ablation_l1_unrolled_p3_b16 \
  --resume output/output_debug_ablation_l1_unrolled_p3_b16

echo "ABLATION_L1_UNROLLED_P3_B16_DEBUG_DONE"
