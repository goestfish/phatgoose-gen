#!/bin/bash
#SBATCH -J p3_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail
module purge
module load anaconda3/2023.09-0-aqbc
eval "$(conda shell.bash hook)" || true
conda activate phatgoose 2>/dev/null || source activate phatgoose

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/users/$USER/phatgoose:${PYTHONPATH:-}
export PHATGOOSE_CAST_DTYPE=bfloat16

cd /users/$USER/phatgoose

DATASETS='["D/BBBOOLEANEXPRESSIONS/EVAL","D/BBCAUSALJUDGEMENT/EVAL","D/BBDATEUNDERSTANDING/EVAL","D/BBDISAMBIGUATIONQA/EVAL","D/BBFORMALFALLACIES/EVAL","D/BBGEOMETRICSHAPES/EVAL","D/BBHYPERBATON/EVAL","D/BBLOGICALDEDUCTION/EVAL","D/BBMOVIERECOMMENDATION/EVAL","D/BBMULTISTEPARITHMETICTWO/EVAL","D/BBNAVIGATE/EVAL","D/BBOBJECTCOUNTING/EVAL","D/BBPENGUINSINATABLE/EVAL","D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL","D/BBRUINNAMES/EVAL","D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL","D/BBSNARKS/EVAL","D/BBSPORTSUNDERSTANDING/EVAL","D/BBTEMPORALSEQUENCES/EVAL","D/BBTRACKINGSHUFFLEDOBJECTS/EVAL","D/BBWEBOFLIES/EVAL","D/BBWORDSORTING/EVAL"]'

EXP_NAME="p3_gen_base"
OUT_BASE="exp_out/${EXP_NAME}"

EXTRA_BINDINGS="
P/EVALUATE/Evaluator.datasets=${DATASETS}
M/MODEL/FFNExperts.topk_value=2
M/MODEL/FFNExperts.normalize_topk=True
M/MODEL/ENCODER/ExposeHidden.reduction_method=None
M/MODEL/DECODER/ExposeHidden.reduction_method=None
P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()]
WriteOutputText.save_dir=\"${OUT_BASE}/output_text\"
RoutingDistribution.save_dir=\"${OUT_BASE}/routing_distribution\"
M/MODEL/load_weights.weight_path=\"exp_out/P3_Phatgoose/best.pt\"
M/MODEL/hf_torch_model.model_name_or_path=\"google/flan-t5-xl\"
M/MODEL/hf_torch_model.model_class=\"seq2seq_lm\"
M/MODEL/hf_torch_model.from_pretrained_kwargs={\"torch_dtype\":\"bfloat16\",\"low_cpu_mem_usage\":True}
"

bash colm/experiments/bash_scripts/eval_gen.sh -exp_name "${EXP_NAME}" -extra_bindings "${EXTRA_BINDINGS}"
echo "Done baseline: ${OUT_BASE}"