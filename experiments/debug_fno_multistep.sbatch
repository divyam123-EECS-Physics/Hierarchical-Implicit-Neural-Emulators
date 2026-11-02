#!/bin/bash
#SBATCH --job-name=fno-ms-dbg
#SBATCH --account=m5326_g
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/d/divyam/hine_runs/logs/debug_fno_multistep-%j.out
#SBATCH --error=/pscratch/sd/d/divyam/hine_runs/logs/debug_fno_multistep-%j.err
#SBATCH --mail-user=divyam123@berkeley.edu
#SBATCH --mail-type=FAIL

# Single debug job: FNO 2-to-2, then 3-to-3, then 1-to-1 (1 epoch each, no resume).

set -euo pipefail
cd /global/homes/d/divyam/TACSLAB/Hierarchical-Implicit-Neural-Emulators
mkdir -p /pscratch/sd/d/divyam/hine_runs/logs

TORCHRUN=/global/homes/d/divyam/.conda/envs/hine/bin/torchrun
export PYTHONUNBUFFERED=1

# Pick a free port for torchrun (same pattern as experiments/srun.sh).
while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${port}
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

run_one () {
  local in_steps=$1
  local out_steps=$2
  local num_of_frame=$3
  local out_dir=$4
  mkdir -p "${out_dir}"
  echo "=== FNO ${in_steps}-to-${out_steps} -> ${out_dir} ==="
  "${TORCHRUN}" --nnodes=1 --nproc_per_node=1 --master_port="${MASTER_PORT}" \
    train_fno_baseline.py \
    --img_size 256 \
    --in_steps "${in_steps}" \
    --out_steps "${out_steps}" \
    --num_of_frame "${num_of_frame}" \
    --modes 96 \
    --width 32 \
    --n_layers 3 \
    --batch_size 8 \
    --epochs 1 \
    --eval_freq 1 \
    --save_last_freq 1 \
    --num_workers 4 \
    --output_dir "${out_dir}"
  echo "=== done ${in_steps}-to-${out_steps} ==="
}

run_one 2 2 4 /pscratch/sd/d/divyam/hine_runs/fno_2to2_debug
run_one 3 3 6 /pscratch/sd/d/divyam/hine_runs/fno_3to3_debug
run_one 1 1 2 /pscratch/sd/d/divyam/hine_runs/fno_1to1_debug

echo "FNO_MULTISTEP_DEBUG_DONE"
