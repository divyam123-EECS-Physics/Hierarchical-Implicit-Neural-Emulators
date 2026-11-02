#!/bin/bash
# Launch FutureINNPlus (L=4 example). Override LATENT_RES / OUTPUT_DIR as needed.
# L=4: LATENT_RES="64 32 8"
# L=5: LATENT_RES="128 64 32 8"
# L=3 paper-shaped: LATENT_RES="32 8"

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
img_res=${IMG_SIZE:-256}
LATENT_RES=${LATENT_RES:-"64 32 8"}
output_dir=${OUTPUT_DIR:-output/output_plus_L4_m64_32_8}
nproc=${NPROC:-4}

torchrun --nnodes=1 --nproc_per_node=$nproc --master_port=${port} \
  main_plus.py --img_size $img_res --batch_size $btz \
    --resume $output_dir --output_dir $output_dir \
    --latent_res $LATENT_RES
