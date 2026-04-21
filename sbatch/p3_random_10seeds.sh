#!/bin/bash
#SBATCH -J p3_randdrop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-39%2

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

TASK_ID=${SLURM_ARRAY_TASK_ID}

# 0-9 -> M=5
# 10-19 -> M=10
# 20-29 -> M=20
# 30-39 -> M=30
if [ "$TASK_ID" -lt 10 ]; then
  M=5
  SEED=$TASK_ID
elif [ "$TASK_ID" -lt 20 ]; then
  M=10
  SEED=$((TASK_ID - 10))
elif [ "$TASK_ID" -lt 30 ]; then
  M=20
  SEED=$((TASK_ID - 20))
else
  M=30
  SEED=$((TASK_ID - 30))
fi

if [ "$M" -gt "$NUM_EXPERTS" ]; then
  echo "Error: M=$M > NUM_EXPERTS=$NUM_EXPERTS"
  exit 1
fi

REMOVED=$(python - <<PY
import random
n=${NUM_EXPERTS}
m=${M}
seed=${SEED}
rng = random.Random(seed)
removed = sorted(rng.sample(range(n), m))
print(",".join(map(str, removed)))
PY
)

KEEP=$(python - <<PY
n=${NUM_EXPERTS}
removed = set(map(int, "${REMOVED}".split(","))) if "${REMOVED}" else set()
keep = [i for i in range(n) if i not in removed]
print(",".join(map(str, keep)))
PY
)

EXP_NAME="p3_random_5_10_20_30/p3_gen_rand_m${M}_s${SEED}"
OUT_BASE="exp_out/${EXP_NAME}"

mkdir -p "${OUT_BASE}"
mkdir -p logs

python - <<PY
import json, os
removed = [int(x) for x in "${REMOVED}".split(",")] if "${REMOVED}" else []
keep = [int(x) for x in "${KEEP}".split(",")] if "${KEEP}" else []
meta = {
    "experiment_type": "random_remove_m",
    "seed": ${SEED},
    "num_experts_before": ${NUM_EXPERTS},
    "num_removed": ${M},
    "removed_experts": removed,
    "num_experts_after": ${NUM_EXPERTS} - ${M},
    "kept_experts": keep,
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
echo "TASK_ID=${TASK_ID}"
echo "NUM_EXPERTS=${NUM_EXPERTS}"
echo "M=${M}"
echo "SEED=${SEED}"
echo "REMOVED=${REMOVED}"
echo "KEEP=${KEEP}"

bash colm/experiments/bash_scripts/eval_gen.sh \
  -exp_name "${EXP_NAME}" \
  -extra_bindings "${EXTRA_BINDINGS}"