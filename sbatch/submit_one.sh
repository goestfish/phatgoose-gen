#!/bin/bash
set -eo pipefail

cd /users/$USER/phatgoose

NUM_EXPERTS=$(python - <<'PY'
import json
d = json.load(open("exp_out/p3_gen_base/top_used.json"))
print(len(d["ranked_experts"]))
PY
)

echo "Total experts: ${NUM_EXPERTS}"
echo "Submitting leave-one-out jobs: 0..$((NUM_EXPERTS - 1))"

sbatch --array=0-$((NUM_EXPERTS - 1))%2 sbatch/remove_one_expert.sh