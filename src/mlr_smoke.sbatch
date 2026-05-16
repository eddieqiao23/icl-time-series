#!/bin/bash
#SBATCH -J mlr_smoke
#SBATCH -p mit_normal_gpu
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o /home/eqiao/icl-time-series/src/mlr_smoke.out
#SBATCH -e /home/eqiao/icl-time-series/src/mlr_smoke.out

set -eo pipefail

module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate in-context-learning

cd /home/eqiao/icl-time-series/src

echo "===== MLR smoke test ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo

time srun python train.py --config conf/mlr_test.yaml
