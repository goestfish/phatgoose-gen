#!/bin/bash
#SBATCH -J p3_drop1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-(N-1)%2

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

TOP_JSON="exp_out/p3_gen_base/top_used.json"
CKPT="exp_out/P3_Phatgoose/best.pt"

NUM_EXPERTS=$(python - <<'PY'
import json
d = json.load(open("exp_out/p3_gen_base/top_used.json"))
print(len(d["ranked_experts"]))
PY
)

DROP_IDX=${SLURM_ARRAY_TASK_ID}

if [ "$DROP_IDX" -ge "$NUM_EXPERTS" ]; then
  echo "DROP_IDX=$DROP_IDX out of range NUM_EXPERTS=$NUM_EXPERTS"
  exit 1
fi

KEEP=$(python - <<PY
n=${NUM_EXPERTS}
drop=${DROP_IDX}
keep=[i for i in range(n) if i != drop]
print(",".join(map(str, keep)))
PY
)

REMOVED="${DROP_IDX}"

EXP_NAME="p3_gen_drop1_e${DROP_IDX}"
OUT_BASE="exp_out/${EXP_NAME}"

mkdir -p "${OUT_BASE}"
mkdir -p logs

python - <<PY
import json, os
meta = {
    "num_experts_before": ${NUM_EXPERTS},
    "removed_expert": ${DROP_IDX},
    "num_experts_after": ${NUM_EXPERTS} - 1,
    "kept_experts": [int(x) for x in "${KEEP}".split(",")] if "${KEEP}" else [],
}
os.makedirs("${OUT_BASE}", exist_ok=True)
with open("${OUT_BASE}/run_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
PY

EXTRA_BINDINGS="
P/EVALUATE/Evaluator.datasets=${DATASETS}
M/MODEL/FFNExperts.topk_value=2
M/MODEL/FFNExperts.normalize_topk=True
M/MODEL/ENCODER/ExposeHidden.reduction_method=None
M/MODEL/DECODER/ExposeHidden.reduction_method=None
P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()]
WriteOutputText.save_dir=\"${OUT_BASE}/output_text\"
RoutingDistribution.save_dir=\"${OUT_BASE}/routing_distribution\"
M/MODEL/load_weights.weight_path=\"${CKPT}\"
M/MODEL/hf_torch_model.model_name_or_path=\"google/flan-t5-xl\"
M/MODEL/hf_torch_model.model_class=\"seq2seq_lm\"
M/MODEL/hf_torch_model.from_pretrained_kwargs={\"torch_dtype\":\"bfloat16\",\"low_cpu_mem_usage\":True}
M/MODEL/Router.removed_experts_str=\"${REMOVED}\"
"

echo "Running ${EXP_NAME}"
echo "NUM_EXPERTS=${NUM_EXPERTS}"
echo "REMOVED=${REMOVED}"
echo "KEEP=${KEEP}"

bash colm/experiments/bash_scripts/eval_gen.sh \
  -exp_name "${EXP_NAME}" \
  -extra_bindings "${EXTRA_BINDINGS}"