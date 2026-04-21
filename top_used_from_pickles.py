import os
import glob
import pickle
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing_dir", type=str, required=True,
                    help="e.g. exp_out/p3_gen/routing_distribution")
    ap.add_argument("--out", type=str, required=True,
                    help="output json path")
    args = ap.parse_args()

    routing_dir = os.path.expandvars(args.routing_dir)
    files = sorted(glob.glob(os.path.join(routing_dir, "*.pickle")))
    if not files:
        raise FileNotFoundError(f"No .pickle files found in {routing_dir}")

    sums = None
    count = 0

    for fp in files:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        for k, arr in d.items():
            arr = np.asarray(arr, dtype=np.float64)
            if sums is None:
                sums = np.zeros_like(arr)
            sums += arr
            count += 1

    if sums is None or count == 0:
        raise RuntimeError("No routing arrays found inside pickles.")

    avg = sums / count
    ranked = [int(x) for x in np.argsort(-avg)]
    out_path = os.path.expandvars(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump({"ranked_experts": ranked, "avg_usage": avg.tolist()}, f, indent=2)

    print(f"Wrote: {out_path}")
    print("Top 10 experts:", ranked[:10])

if __name__ == "__main__":
    main()