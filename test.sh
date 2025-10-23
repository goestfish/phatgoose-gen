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

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/users/$USER/phatgoose:${PYTHONPATH:-}
export PHATGOOSE_CAST_DTYPE=bfloat16
cd /users/$USER/phatgoose
mkdir -p logs

python - <<'PY'
import os, gin, torch
from src.launch_single_process import main as _entry_main  # 注册 gin 的 main
from src.utils.gin import get_scope_defined_objects

ROOT = "/users/bkou1/phatgoose"
os.chdir(ROOT)
gin.add_config_file_search_path(ROOT)
gin.add_config_file_search_path(os.path.join(ROOT, "colm"))
gin.add_config_file_search_path(os.path.join(ROOT, "colm/models"))
gin.add_config_file_search_path(os.path.join(ROOT, "colm/models/t5xl"))
gin.add_config_file_search_path(os.path.join(ROOT, "colm/datasets"))
gin.add_config_file_search_path(os.path.join(ROOT, "colm/experiments"))

gin_files = [
  os.path.join(ROOT, "colm/datasets/bigbench.gen.gin"),
  os.path.join(ROOT, "colm/models/t5xl/t5.gin"),
  os.path.join(ROOT, "colm/models/t5xl/moe_lora_rank16.gin"),
  os.path.join(ROOT, "colm/experiments/eval.gin"),
]

gin_bindings = [
  'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL"]',
  'P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()]',
  'WriteOutputText.save_dir="exp_out/p3_phatgoose_gen/output_text"',
  'M/MODEL/hf_torch_model.model_name_or_path="google/flan-t5-xl"',
  'M/MODEL/hf_torch_model.model_class="seq2seq_lm"',
  'M/MODEL/hf_torch_model.from_pretrained_kwargs={"torch_dtype":"bfloat16","low_cpu_mem_usage":True}',
  'M/MODEL/Model.init_moma_calls=[@M/MODEL/ENCODER/watch_hiddens,@M/MODEL/DECODER/watch_hiddens,@M/MODEL/ENCODER/make_moe,@M/MODEL/DECODER/make_moe,@M/MODEL/load_weights]',
  'M/MODEL/load_weights.weight_path="exp_out/P3_Phatgoose/best.pt"',
  'M/MODEL/set_trainable_params.trainable_params="^$"',
  'M/MODEL/set_trainable_params.mix_precision="bf16"',
  'M/MODEL/FFNExperts.topk_value=2',
  'M/MODEL/FFNExperts.normalize_topk=True',
]

gin.parse_config_files_and_bindings(gin_files, gin_bindings)

wrapper = get_scope_defined_objects("M/MODEL")

def find_core_module(obj):
  for name in ("torch_model","model","hf_model","backbone"):
    if hasattr(obj, name):
      x = getattr(obj, name)

      if hasattr(x, "module"): x = x.module
      if hasattr(x, "parameters"):
        return x

  for name in dir(obj):
    if name.startswith("_"): continue
    try:
      x = getattr(obj, name)
    except Exception:
      continue
    if hasattr(x, "parameters"):
      return x
  raise RuntimeError("No nn.Module-like attribute found on M/MODEL")

core = find_core_module(wrapper)
first_param = next(core.parameters())

print("=== MODEL CHECK (before eval) ===")
print("[Wrapper class]", type(wrapper))
print("[Core class]", type(core))
print("[First param dtype]", first_param.dtype)
print("[First param device]", first_param.device)
try:
  print("[HF model_type]", getattr(getattr(core, "config", None), "model_type", None))
except Exception:
  pass
print("===============================")

if str(first_param.dtype) != "torch.bfloat16":
  import sys
  print("[FAIL] Model is NOT bfloat16, got", first_param.dtype)
  sys.exit(2)
PY


bash colm/experiments/bash_scripts/eval_multitask.sh \
    -exp_name P3_Phatgoose \
    --gin_files \
    colm/datasets/bigbench.gen.gin \
    colm/models/t5xl/t5.gin \
    colm/models/t5xl/moe_lora_rank16.gin \
    colm/experiments/eval.gin \
    -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL","D/BBCAUSALJUDGEMENT/EVAL","D/BBDATEUNDERSTANDING/EVAL","D/BBDISAMBIGUATIONQA/EVAL","D/BBFORMALFALLACIES/EVAL","D/BBGEOMETRICSHAPES/EVAL","D/BBHYPERBATON/EVAL","D/BBLOGICALDEDUCTION/EVAL","D/BBMOVIERECOMMENDATION/EVAL","D/BBMULTISTEPARITHMETICTWO/EVAL","D/BBNAVIGATE/EVAL","D/BBOBJECTCOUNTING/EVAL","D/BBPENGUINSINATABLE/EVAL","D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL","D/BBRUINNAMES/EVAL","D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL","D/BBSNARKS/EVAL","D/BBSPORTSUNDERSTANDING/EVAL","D/BBTEMPORALSEQUENCES/EVAL","D/BBTRACKINGSHUFFLEDOBJECTS/EVAL","D/BBWEBOFLIES/EVAL","D/BBWORDSORTING/EVAL"] P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/p3_phatgoose_gen/output_text" M/MODEL/hf_torch_model.model_name_or_path="google/flan-t5-xl" M/MODEL/hf_torch_model.model_class="seq2seq_lm" M/MODEL/hf_torch_model.from_pretrained_kwargs={"torch_dtype":"bfloat16","low_cpu_mem_usage":True} M/MODEL/Model.init_moma_calls=[@M/MODEL/ENCODER/watch_hiddens,@M/MODEL/DECODER/watch_hiddens,@M/MODEL/ENCODER/make_moe,@M/MODEL/DECODER/make_moe,@M/MODEL/load_weights] M/MODEL/load_weights.weight_path="exp_out/P3_Phatgoose/best.pt" M/MODEL/set_trainable_params.trainable_params="^$" M/MODEL/set_trainable_params.mix_precision="bf16" M/MODEL/FFNExperts.topk_value=2 M/MODEL/FFNExperts.normalize_topk=True'

echo "Done. Check exp_out/p3_phatgoose_gen/ and logs/*.out, *.err"