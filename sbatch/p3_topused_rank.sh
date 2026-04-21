#!/bin/bash
#SBATCH -J p3_rank
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail
module purge
module load anaconda3/2023.09-0-aqbc
eval "$(conda shell.bash hook)" || true
conda activate phatgoose 2>/dev/null || source activate phatgoose

export PYTHONNOUSERSITE=1
export PYTHONPATH=/users/$USER/phatgoose:${PYTHONPATH:-}

cd /users/$USER/phatgoose

python top_used_from_pickles.py \
  --routing_dir "exp_out/p3_gen_base/routing_distribution" \
  --out "exp_out/p3_gen_base/top_used.json"

echo "Wrote exp_out/p3_gen_base/top_used.json"