#!/bin/bash
# Create / update the named conda env `hine` on NERSC (scratch-backed via ~/.conda).
# Daily use after this:  module load conda && conda activate hine
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

module load conda

if conda env list | awk '{print $1}' | grep -qx 'hine'; then
  echo "Updating existing env: hine"
  conda env update -n hine -f environment.yml --prune
else
  echo "Creating env: hine"
  conda env create -f environment.yml
fi

# Activate without requiring conda init in ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate hine

# Force CUDA wheels: conda env create may already have pulled a torch dep via torch_fidelity.
echo "Installing CUDA PyTorch wheels (cu128)..."
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "Installing remaining pip packages..."
pip install opencv-python torch_fidelity

echo "--- verification ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import numpy, scipy, einops, matplotlib, cv2, pandas, tqdm, torch_fidelity; print('deps ok')"

echo
echo "Done. Activate with:"
echo "  module load conda"
echo "  conda activate hine"
echo
echo "Env prefix: $(conda info --base)/../envs/hine  (or: conda env list | grep hine)"
echo "Note: \$HOME/.conda -> \$PSCRATCH/conda2; scratch can be purged — re-run this script to rebuild."
