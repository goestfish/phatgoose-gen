#!/bin/bash
#SBATCH -J bbboolean_gen_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail
module purge
module load anaconda/2023.09-0-7nso27y
eval "$(conda shell.bash hook)" || true
conda activate phatgoose 2>/dev/null || source activate phatgoose

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/users/$USER/phatgoose:${PYTHONPATH:-}
export PHATGOOSE_CAST_DTYPE=bfloat16

cd /users/$USER/phatgoose
mkdir -p logs

echo "===== Running quick GEN test on D_BBBOOLEANEXPRESSIONS_EVAL ====="

bash colm/experiments/bash_scripts/eval_single_task_loralinear.sh \
  -exp_name test_bbboolean_gen \
  -dataset BBBOOLEANEXPRESSIONS \
  -extra_bindings '
P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL"]
M/MODEL/FFNExperts.topk_value=2
M/MODEL/FFNExperts.normalize_topk=True
M/MODEL/ENCODER/ExposeHidden.reduction_method=None
M/MODEL/DECODER/ExposeHidden.reduction_method=None
P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()]
WriteOutputText.save_dir="exp_out/p3_phatgoose_gen/output_text"
RoutingDistribution.save_dir="exp_out/p3_phatgoose_gen/routing_distribution"
M/MODEL/load_weights.weight_path="exp_out/P3_Phatgoose/best.pt"
M/MODEL/hf_torch_model.model_name_or_path="google/flan-t5-xl"
M/MODEL/hf_torch_model.model_class="seq2seq_lm"
M/MODEL/hf_torch_model.from_pretrained_kwargs={"torch_dtype":"bfloat16","low_cpu_mem_usage":True}
'

echo "Done. Check exp_out/test_bbboolean_gen/output_text/ and logs/*.out, *.err"