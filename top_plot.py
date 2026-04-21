import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", default="top_used_bar.png")

    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    ranked = data["ranked_experts"]
    usage = np.array(data["avg_usage"])

    topk = args.topk
    top_ids = ranked[:topk]
    top_vals = usage[top_ids] * 100

    plt.figure(figsize=(8,4))

    plt.bar([str(i) for i in top_ids], top_vals)

    plt.xlabel("Expert ID")
    plt.ylabel("Router Usage (%)")
    plt.title(f"Top {topk} Most Used Experts (Baseline Routing)")

    plt.tight_layout()

    plt.savefig(args.out, dpi=300)

    print("Saved figure:", args.out)

if __name__ == "__main__":
    main()