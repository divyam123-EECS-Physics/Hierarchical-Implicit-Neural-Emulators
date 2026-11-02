#!/bin/bash
#SBATCH --job-name=hine-dbg-32-8
#SBATCH --account=m5326_g
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/debug_256_32_8-%j.out
#SBATCH --mail-user=divyam123@berkeley.edu
#SBATCH --mail-type=FAIL

# Default L=3 hierarchy: 256 -> mid 32 -> low 8
# ds_factor=8, ds_factor_low=32. One GPU so the loader stops after 4000 frames.

set -euo pipefail
cd /global/homes/d/divyam/TACSLAB/Hierarchical-Implicit-Neural-Emulators
mkdir -p logs output/output_debug_256_32_8

export PYTHONUNBUFFERED=1
/global/homes/d/divyam/.conda/envs/hine/bin/torchrun \
  --nnodes=1 --nproc_per_node=1 \
  main.py \
  --img_size 256 \
  --batch_size 8 \
  --epochs 1 \
  --num_of_frame 4 \
  --ds_factor 8 \
  --ds_factor_low 32 \
  --num_workers 4 \
  --output_dir output/output_debug_256_32_8 \
  --resume output/output_debug_256_32_8
