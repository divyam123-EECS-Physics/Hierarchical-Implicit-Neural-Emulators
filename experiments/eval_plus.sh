#!/bin/bash
# Evaluate a FutureINNPlus checkpoint. Override LATENT_RES / OUTPUT_DIR / RESUME_EPOCH.

while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
echo "NODELIST=${SLURM_NODELIST}"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR=$MASTER_ADDR"

btz=${BATCH_SIZE:-32}
LATENT_RES=${LATENT_RES:-"64 32 8"}
output_dir=${OUTPUT_DIR:-output/output_plus_L4_m64_32_8}
resume_epoch=${RESUME_EPOCH:-700}

torchrun --nnodes=1 --nproc_per_node=1 --master_port=${port} \
  main_plus.py --img_size 256 --batch_size $btz \
    --resume $output_dir --output_dir $output_dir \
    --latent_res $LATENT_RES \
    --evaluate --resume_epoch $resume_epoch
