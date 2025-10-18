#!/bin/bash
#SBATCH -J bigbench_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail
module purge
module load anaconda/2023.09-0-7nso27y
eval "$(conda shell.bash hook)" || true
conda activate phatgoose 2>/dev/null || source activate phatgoose

export PYTHONNOUSERSITE=1
export PYTHONPATH=/users/$USER/phatgoose:${PYTHONPATH:-}
cd /users/$USER/phatgoose
mkdir -p logs

python - <<'PY' || { echo "[FATAL] env check failed"; exit 2; }
import sys
print("Python:", sys.version)
assert sys.version.startswith("3.9"), f"Expect Python 3.9, got {sys.version}"
import sentence_transformers
print("sentence_transformers OK")
PY

bash colm/experiments/bash_scripts/eval_multitask.sh \
  -exp_name flan_t5_xl_gen \
  -gin_file colm/datasets/bigbench.gen.gin \
  -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL","D/BBCAUSALJUDGEMENT/EVAL","D/BBDATEUNDERSTANDING/EVAL","D/BBDISAMBIGUATIONQA/EVAL","D/BBFORMALFALLACIES/EVAL","D/BBGEOMETRICSHAPES/EVAL","D/BBHYPERBATON/EVAL","D/BBLOGICALDEDUCTION/EVAL","D/BBMOVIERECOMMENDATION/EVAL","D/BBMULTISTEPARITHMETICTWO/EVAL","D/BBNAVIGATE/EVAL","D/BBOBJECTCOUNTING/EVAL","D/BBPENGUINSINATABLE/EVAL","D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL","D/BBRUINNAMES/EVAL","D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL","D/BBSNARKS/EVAL","D/BBSPORTSUNDERSTANDING/EVAL","D/BBTEMPORALSEQUENCES/EVAL","D/BBTRACKINGSHUFFLEDOBJECTS/EVAL","D/BBWEBOFLIES/EVAL","D/BBWORDSORTING/EVAL"] P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/flan_t5_xl_gen/output_text" M/MODEL/hf_torch_model.model_name_or_path="google/flan-t5-xl" M/MODEL/Model.init_moma_calls=[]'

echo "Done. Check exp_out/flan_t5_xl_gen/ and logs/*.out, *.err"