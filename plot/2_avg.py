#!/usr/bin/env python
import re
import ast
import json
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("exp_out/rand_10seeds")
META_DIR = Path("exp_out/p3_random_5_10_20_30")
OUT_DIR = Path("rand_10seeds_plots/rand_10seeds_pair_expert_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PAIR_COUNT = 3


def parse_result_txt(path: Path):
    text = path.read_text(errors="ignore").strip()

    if "Final results:" in text:
        text = text.split("Final results:", 1)[1].strip()

    text = re.sub(r"np\.float64\(([^)]+)\)", r"\1", text)

    try:
        return ast.literal_eval(text)
    except Exception as e:
        print(f"Failed to parse {path}: {e}")
        print(text[:500])
        return None


def dataset_score(v):
    if not isinstance(v, dict):
        return None
    for key in ["score", "exact_match", "exact_match_multiple_ans", "accuracy"]:
        if key in v and isinstance(v[key], (int, float)):
            return float(v[key])
    return None


def parse_m_seed(exp_name: str):
    m = re.search(r"_m(\d+)_s(\d+)", exp_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_scores():
    rows = []

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"RESULTS_DIR not found: {RESULTS_DIR.resolve()}")

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        result_file = exp_dir / "results.txt"
        if not result_file.exists():
            continue

        result_dict = parse_result_txt(result_file)
        M, seed = parse_m_seed(exp_dir.name)

        if not isinstance(result_dict, dict):
            continue

        dataset_scores = {}
        for ds, v in result_dict.items():
            s = dataset_score(v)
            if s is not None:
                dataset_scores[ds] = s

        mean_score = None
        if dataset_scores:
            mean_score = sum(dataset_scores.values()) / len(dataset_scores)

        rows.append({
            "exp_name": exp_dir.name,
            "M": M,
            "seed": seed,
            "mean_score": mean_score,
            "dataset_scores": dataset_scores,
        })

    df = pd.DataFrame(rows).sort_values(["M", "seed"]).reset_index(drop=True)
    return df


def load_removed_meta():
    rows = []

    if not META_DIR.exists():
        raise FileNotFoundError(f"META_DIR not found: {META_DIR.resolve()}")

    for exp_dir in sorted(META_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        meta_file = exp_dir / "run_meta.json"
        if not meta_file.exists():
            continue

        with open(meta_file, "r") as f:
            meta = json.load(f)

        M, seed = parse_m_seed(exp_dir.name)
        removed = meta.get("removed_experts", [])
        kept = meta.get("kept_experts", [])

        rows.append({
            "exp_name": exp_dir.name,
            "M": M,
            "seed": seed,
            "removed_experts": removed,
            "kept_experts": kept,
        })

    df = pd.DataFrame(rows).sort_values(["M", "seed"]).reset_index(drop=True)
    return df


def infer_num_experts(meta_df: pd.DataFrame):
    mx = -1
    for col in ["removed_experts", "kept_experts"]:
        for xs in meta_df[col]:
            if xs:
                mx = max(mx, max(xs))
    return mx + 1 if mx >= 0 else 0


def build_run_table():
    score_df = load_scores()
    meta_df = load_removed_meta()

    df = pd.merge(
        score_df,
        meta_df,
        on=["exp_name", "M", "seed"],
        how="inner",
        validate="one_to_one",
    )

    if df.empty:
        raise ValueError("Merged run table is empty. Check directory paths and experiment names.")

    return df.sort_values(["M", "seed"]).reset_index(drop=True)


def compute_pair_stats_all(run_df: pd.DataFrame):
    meta_df = run_df[["removed_experts", "kept_experts"]].copy()
    num_experts = infer_num_experts(meta_df)

    rows = []

    for e1, e2 in combinations(range(num_experts), 2):
        both_removed_mask = run_df["removed_experts"].apply(lambda xs: e1 in xs and e2 in xs)
        both_kept_mask = run_df["kept_experts"].apply(lambda xs: e1 in xs and e2 in xs)

        both_removed_scores = run_df.loc[both_removed_mask, "mean_score"].dropna().tolist()
        both_kept_scores = run_df.loc[both_kept_mask, "mean_score"].dropna().tolist()

        both_removed_mean = (
            sum(both_removed_scores) / len(both_removed_scores)
            if both_removed_scores else float("nan")
        )
        both_kept_mean = (
            sum(both_kept_scores) / len(both_kept_scores)
            if both_kept_scores else float("nan")
        )

        delta = float("nan")
        if pd.notna(both_kept_mean) and pd.notna(both_removed_mean):
            delta = both_kept_mean - both_removed_mean

        rows.append({
            "expert_1": int(e1),
            "expert_2": int(e2),
            "both_removed_count": len(both_removed_scores),
            "both_kept_count": len(both_kept_scores),
            "both_removed_mean_score": both_removed_mean,
            "both_kept_mean_score": both_kept_mean,
            "delta_both_kept_minus_both_removed": delta,
        })

    return pd.DataFrame(rows).sort_values(
        ["expert_1", "expert_2"]
    ).reset_index(drop=True)


def filter_pair_stats_df(stats_df: pd.DataFrame):
    filtered = stats_df[
        (stats_df["both_removed_count"] >= MIN_PAIR_COUNT) &
        (stats_df["both_kept_count"] >= MIN_PAIR_COUNT) &
        (pd.notna(stats_df["both_removed_mean_score"])) &
        (pd.notna(stats_df["both_kept_mean_score"])) &
        (pd.notna(stats_df["delta_both_kept_minus_both_removed"]))
    ].copy()

    filtered = filtered[
        filtered["delta_both_kept_minus_both_removed"].apply(
            lambda x: pd.notna(x) and float("-inf") < x < float("inf")
        )
    ].copy()

    return filtered.sort_values(
        "delta_both_kept_minus_both_removed",
        ascending=False
    ).reset_index(drop=True)


def save_rank_text(stats_df: pd.DataFrame):
    out_path = OUT_DIR / "pair_expert_rankings_all.txt"
    with open(out_path, "w") as f:
        f.write(f"=== ALL M combined pair stats (counts >= {MIN_PAIR_COUNT}) ===\n")
        for i, row in stats_df.iterrows():
            f.write(
                f"{i+1:>3}. pair ({int(row['expert_1'])}, {int(row['expert_2'])}): "
                f"delta={row['delta_both_kept_minus_both_removed']:.6f}, "
                f"both_kept_mean={row['both_kept_mean_score']:.6f}, "
                f"both_removed_mean={row['both_removed_mean_score']:.6f}, "
                f"both_kept_count={int(row['both_kept_count'])}, "
                f"both_removed_count={int(row['both_removed_count'])}\n"
            )

def plot_pair_deltas_sorted(stats_df: pd.DataFrame):
    sub = stats_df.copy().sort_values(
        "delta_both_kept_minus_both_removed",
        ascending=False
    ).reset_index(drop=True)

    if sub.empty:
        return

    sub["rank"] = range(1, len(sub) + 1)

    plt.figure(figsize=(18, 8))
    plt.plot(
        sub["rank"],
        sub["delta_both_kept_minus_both_removed"],
        marker="o",
        markersize=3
    )
    plt.axhline(0.0, linestyle="--")

    plt.xlabel("Pair rank (sorted by both_kept_mean - both_removed_mean)")
    plt.ylabel("Delta = both_kept_mean_score - both_removed_mean_score")
    plt.title(f"Pair expert importance ranking across ALL M (counts >= {MIN_PAIR_COUNT})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pair_expert_delta_sorted_ALL.png", dpi=220)
    plt.close()

def plot_pair_top_bottom_bar(stats_df: pd.DataFrame, top_k=15):
    sub = stats_df.copy().sort_values(
        "delta_both_kept_minus_both_removed",
        ascending=False
    ).reset_index(drop=True)

    if sub.empty:
        return

    top = sub.head(top_k)
    bottom = sub.tail(top_k).sort_values(
        "delta_both_kept_minus_both_removed",
        ascending=True
    )

    show = pd.concat([top, bottom], axis=0).drop_duplicates(
        subset=["expert_1", "expert_2"]
    ).reset_index(drop=True)

    labels = [f"({int(r.expert_1)},{int(r.expert_2)})" for _, r in show.iterrows()]
    values = show["delta_both_kept_minus_both_removed"].tolist()

    plt.figure(figsize=(16, 7))
    plt.bar(range(len(show)), values)
    plt.axhline(0.0, linestyle="--")
    plt.xticks(range(len(show)), labels, rotation=45, ha="right")
    plt.xlabel("Expert pair")
    plt.ylabel("Delta = both_kept_mean_score - both_removed_mean_score")
    plt.title(f"Top and bottom expert pairs across ALL M (counts >= {MIN_PAIR_COUNT})")

    for i, v in enumerate(values):
        if pd.isna(v) or not (float("-inf") < v < float("inf")):
            continue
        plt.text(
            i,
            v,
            f"{v:.4f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=7
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "pair_expert_delta_top_bottom_ALL.png", dpi=220)
    plt.close()

def main():
    run_df = build_run_table()
    run_df.to_csv(OUT_DIR / "merged_runs.csv", index=False)

    print("Merged runs:", len(run_df))
    print(run_df[["exp_name", "M", "seed", "mean_score"]].head())

    raw_pair_stats_df = compute_pair_stats_all(run_df)
    raw_pair_stats_df.to_csv(OUT_DIR / "pair_expert_kept_vs_removed_all_raw.csv", index=False)

    filtered_pair_stats_df = filter_pair_stats_df(raw_pair_stats_df)
    filtered_pair_stats_df.to_csv(OUT_DIR / "pair_expert_kept_vs_removed_all_filtered.csv", index=False)

    save_rank_text(filtered_pair_stats_df)
    plot_pair_deltas_sorted(filtered_pair_stats_df)
    plot_pair_top_bottom_bar(filtered_pair_stats_df, top_k=15)

    print(f"\n=== ALL M combined pair stats (both_removed_count >= {MIN_PAIR_COUNT}, both_kept_count >= {MIN_PAIR_COUNT}) ===")
    print(filtered_pair_stats_df.head(20)[[
        "expert_1",
        "expert_2",
        "both_removed_mean_score",
        "both_kept_mean_score",
        "delta_both_kept_minus_both_removed",
        "both_removed_count",
        "both_kept_count",
    ]])

    print(f"\nSaved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()