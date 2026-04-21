#!/usr/bin/env python
import re
import ast
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("exp_out/rand_10seeds")
META_DIR = Path("exp_out/p3_random_5_10_20_30")
OUT_DIR = Path("rand_10seeds_plots/rand_10seeds_expert_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COUNT = 2
MIN_COUNT_ALL = 3


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


def compute_expert_stats(run_df: pd.DataFrame):
    num_experts = infer_num_experts(run_df)
    rows = []

    for M in sorted(run_df["M"].dropna().unique()):
        sub = run_df[run_df["M"] == M].copy()

        for e in range(num_experts):
            removed_mask = sub["removed_experts"].apply(lambda xs: e in xs)
            kept_mask = sub["kept_experts"].apply(lambda xs: e in xs)

            removed_scores = sub.loc[removed_mask, "mean_score"].dropna().tolist()
            kept_scores = sub.loc[kept_mask, "mean_score"].dropna().tolist()

            removed_mean = sum(removed_scores) / len(removed_scores) if removed_scores else float("nan")
            kept_mean = sum(kept_scores) / len(kept_scores) if kept_scores else float("nan")

            delta = float("nan")
            if pd.notna(kept_mean) and pd.notna(removed_mean):
                delta = kept_mean - removed_mean

            rows.append({
                "M": int(M),
                "expert_id": int(e),
                "removed_count": len(removed_scores),
                "kept_count": len(kept_scores),
                "removed_mean_score": removed_mean,
                "kept_mean_score": kept_mean,
                "delta_kept_minus_removed": delta,
            })

    stats_df = pd.DataFrame(rows).sort_values(["M", "expert_id"]).reset_index(drop=True)
    return stats_df

def compute_expert_stats_all(run_df: pd.DataFrame):
    num_experts = infer_num_experts(run_df)
    rows = []

    for e in range(num_experts):
        removed_mask = run_df["removed_experts"].apply(lambda xs: e in xs)
        kept_mask = run_df["kept_experts"].apply(lambda xs: e in xs)

        removed_scores = run_df.loc[removed_mask, "mean_score"].dropna().tolist()
        kept_scores = run_df.loc[kept_mask, "mean_score"].dropna().tolist()

        removed_mean = sum(removed_scores) / len(removed_scores) if removed_scores else float("nan")
        kept_mean = sum(kept_scores) / len(kept_scores) if kept_scores else float("nan")

        delta = float("nan")
        if pd.notna(kept_mean) and pd.notna(removed_mean):
            delta = kept_mean - removed_mean

        rows.append({
            "group": "ALL",
            "expert_id": int(e),
            "removed_count": len(removed_scores),
            "kept_count": len(kept_scores),
            "removed_mean_score": removed_mean,
            "kept_mean_score": kept_mean,
            "delta_kept_minus_removed": delta,
        })

    stats_df = pd.DataFrame(rows).sort_values(["expert_id"]).reset_index(drop=True)
    return stats_df

def filter_stats_df(stats_df: pd.DataFrame):
    filtered = stats_df[
        (stats_df["removed_count"] >= MIN_COUNT) &
        (stats_df["kept_count"] >= MIN_COUNT) &
        (pd.notna(stats_df["removed_mean_score"])) &
        (pd.notna(stats_df["kept_mean_score"])) &
        (pd.notna(stats_df["delta_kept_minus_removed"]))
    ].copy()

    filtered = filtered[
        filtered["delta_kept_minus_removed"].apply(lambda x: pd.notna(x) and float("-inf") < x < float("inf"))
    ].copy()

    return filtered.sort_values(["M", "expert_id"]).reset_index(drop=True)

def filter_stats_df_all(stats_df: pd.DataFrame):
    filtered = stats_df[
        (stats_df["removed_count"] >= MIN_COUNT_ALL) &
        (stats_df["kept_count"] >= MIN_COUNT_ALL) &
        (pd.notna(stats_df["removed_mean_score"])) &
        (pd.notna(stats_df["kept_mean_score"])) &
        (pd.notna(stats_df["delta_kept_minus_removed"]))
    ].copy()

    filtered = filtered[
        filtered["delta_kept_minus_removed"].apply(
            lambda x: pd.notna(x) and float("-inf") < x < float("inf")
        )
    ].copy()

    return filtered.sort_values(["expert_id"]).reset_index(drop=True)


def save_sorted_tables(stats_df: pd.DataFrame):
    for M in sorted(stats_df["M"].unique()):
        sub = stats_df[stats_df["M"] == M].copy()
        sub = sub.sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)
        sub["rank_by_delta"] = range(1, len(sub) + 1)

        ordered_cols = [
            "rank_by_delta",
            "expert_id",
            "removed_count",
            "kept_count",
            "removed_mean_score",
            "kept_mean_score",
            "delta_kept_minus_removed",
        ]
        sub = sub[ordered_cols]
        sub.to_csv(OUT_DIR / f"expert_kept_vs_removed_sorted_M{M}.csv", index=False)

def save_all_sorted_table(stats_df: pd.DataFrame):
    sub = stats_df.copy().sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)
    sub["rank_by_delta"] = range(1, len(sub) + 1)

    ordered_cols = [
        "rank_by_delta",
        "expert_id",
        "removed_count",
        "kept_count",
        "removed_mean_score",
        "kept_mean_score",
        "delta_kept_minus_removed",
    ]
    sub = sub[ordered_cols]
    sub.to_csv(OUT_DIR / "expert_kept_vs_removed_sorted_ALL.csv", index=False)

def save_all_rank_text(stats_df: pd.DataFrame):
    out_path = OUT_DIR / "expert_rankings_all.txt"
    sub = stats_df.copy().sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)

    with open(out_path, "w") as f:
        f.write(f"=== ALL M combined (counts >= {MIN_COUNT_ALL}) ===\n")
        for i, row in sub.iterrows():
            f.write(
                f"{i+1:>2}. expert {int(row['expert_id'])}: "
                f"delta={row['delta_kept_minus_removed']:.6f}, "
                f"kept_mean={row['kept_mean_score']:.6f}, "
                f"removed_mean={row['removed_mean_score']:.6f}, "
                f"kept_count={int(row['kept_count'])}, "
                f"removed_count={int(row['removed_count'])}\n"
            )

def plot_sorted_expert_deltas(stats_df: pd.DataFrame):
    for M in sorted(stats_df["M"].unique()):
        sub = stats_df[stats_df["M"] == M].copy()
        sub = sub.sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)

        if sub.empty:
            continue

        sub["rank"] = range(1, len(sub) + 1)

        plt.figure(figsize=(14, 6))
        plt.plot(sub["rank"], sub["delta_kept_minus_removed"], marker="o")
        plt.axhline(0.0, linestyle="--")

        for _, row in sub.iterrows():
            x = row["rank"]
            y = row["delta_kept_minus_removed"]

            if pd.isna(x) or pd.isna(y):
                continue
            if not (float("-inf") < y < float("inf")):
                continue

            label = f"e{int(row['expert_id'])}: {y:.4f}"
            plt.text(x, y, label, fontsize=8, rotation=45, ha="left", va="bottom")

        plt.xlabel("Expert rank (sorted by kept_mean - removed_mean)")
        plt.ylabel("Delta = kept_mean_score - removed_mean_score")
        plt.title(f"Expert importance ranking at M={M} (counts >= {MIN_COUNT})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"expert_delta_sorted_M{M}.png", dpi=220)
        plt.close()


def plot_top_bottom_bar(stats_df: pd.DataFrame, top_k=10):
    for M in sorted(stats_df["M"].unique()):
        sub = stats_df[stats_df["M"] == M].copy()
        sub = sub.sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)

        if sub.empty:
            continue

        top = sub.head(top_k)
        bottom = sub.tail(top_k).sort_values("delta_kept_minus_removed", ascending=True)
        show = pd.concat([top, bottom], axis=0).drop_duplicates(subset=["expert_id"]).reset_index(drop=True)

        labels = [f"e{int(e)}" for e in show["expert_id"]]
        values = show["delta_kept_minus_removed"].tolist()

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(show)), values)
        plt.axhline(0.0, linestyle="--")
        plt.xticks(range(len(show)), labels, rotation=45, ha="right")
        plt.xlabel("Expert ID")
        plt.ylabel("Delta = kept_mean_score - removed_mean_score")
        plt.title(f"Top and bottom experts by delta at M={M} (counts >= {MIN_COUNT})")

        for i, v in enumerate(values):
            if pd.isna(v) or not (float("-inf") < v < float("inf")):
                continue
            plt.text(
                i,
                v,
                f"{v:.4f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8
            )

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"expert_delta_top_bottom_M{M}.png", dpi=220)
        plt.close()

def plot_all_expert_deltas(stats_df: pd.DataFrame):
    sub = stats_df.copy().sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)

    if sub.empty:
        return

    sub["rank"] = range(1, len(sub) + 1)

    plt.figure(figsize=(14, 6))
    plt.plot(sub["rank"], sub["delta_kept_minus_removed"], marker="o")
    plt.axhline(0.0, linestyle="--")

    for _, row in sub.iterrows():
        x = row["rank"]
        y = row["delta_kept_minus_removed"]

        if pd.isna(x) or pd.isna(y):
            continue
        if not (float("-inf") < y < float("inf")):
            continue

        label = f"e{int(row['expert_id'])}: {y:.4f}"
        plt.text(x, y, label, fontsize=8, rotation=45, ha="left", va="bottom")

    plt.xlabel("Expert rank (sorted by kept_mean - removed_mean)")
    plt.ylabel("Delta = kept_mean_score - removed_mean_score")
    plt.title(f"Expert importance ranking across ALL M (counts >= {MIN_COUNT_ALL})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "expert_delta_sorted_ALL.png", dpi=220)
    plt.close()

def plot_all_top_bottom_bar(stats_df: pd.DataFrame, top_k=10):
    sub = stats_df.copy().sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)

    if sub.empty:
        return

    top = sub.head(top_k)
    bottom = sub.tail(top_k).sort_values("delta_kept_minus_removed", ascending=True)
    show = pd.concat([top, bottom], axis=0).drop_duplicates(subset=["expert_id"]).reset_index(drop=True)

    labels = [f"e{int(e)}" for e in show["expert_id"]]
    values = show["delta_kept_minus_removed"].tolist()

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(show)), values)
    plt.axhline(0.0, linestyle="--")
    plt.xticks(range(len(show)), labels, rotation=45, ha="right")
    plt.xlabel("Expert ID")
    plt.ylabel("Delta = kept_mean_score - removed_mean_score")
    plt.title(f"Top and bottom experts across ALL M (counts >= {MIN_COUNT_ALL})")

    for i, v in enumerate(values):
        if pd.isna(v) or not (float("-inf") < v < float("inf")):
            continue
        plt.text(
            i,
            v,
            f"{v:.4f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "expert_delta_top_bottom_ALL.png", dpi=220)
    plt.close()

def save_rank_text(stats_df: pd.DataFrame):
    out_path = OUT_DIR / "expert_rankings.txt"
    with open(out_path, "w") as f:
        for M in sorted(stats_df["M"].unique()):
            sub = stats_df[stats_df["M"] == M].copy()
            sub = sub.sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)

            f.write(f"=== M={M} (counts >= {MIN_COUNT}) ===\n")
            for i, row in sub.iterrows():
                f.write(
                    f"{i+1:>2}. expert {int(row['expert_id'])}: "
                    f"delta={row['delta_kept_minus_removed']:.6f}, "
                    f"kept_mean={row['kept_mean_score']:.6f}, "
                    f"removed_mean={row['removed_mean_score']:.6f}, "
                    f"kept_count={int(row['kept_count'])}, "
                    f"removed_count={int(row['removed_count'])}\n"
                )
            f.write("\n")


def main():
    run_df = build_run_table()
    run_df.to_csv(OUT_DIR / "merged_runs.csv", index=False)

    print("Merged runs:", len(run_df))
    print(run_df[["exp_name", "M", "seed", "mean_score"]].head())

    raw_stats_df = compute_expert_stats(run_df)
    raw_stats_df.to_csv(OUT_DIR / "expert_kept_vs_removed_summary_raw.csv", index=False)

    filtered_stats_df = filter_stats_df(raw_stats_df)
    filtered_stats_df.to_csv(OUT_DIR / "expert_kept_vs_removed_summary_filtered.csv", index=False)

    save_sorted_tables(filtered_stats_df)
    save_rank_text(filtered_stats_df)
    plot_sorted_expert_deltas(filtered_stats_df)
    plot_top_bottom_bar(filtered_stats_df, top_k=10)

    print(f"\nFiltered experts with removed_count >= {MIN_COUNT} and kept_count >= {MIN_COUNT}:")
    for M in sorted(filtered_stats_df["M"].unique()):
        sub = filtered_stats_df[filtered_stats_df["M"] == M].copy()
        sub = sub.sort_values("delta_kept_minus_removed", ascending=False).reset_index(drop=True)
        print(f"\n=== M={M} ===")
        print(sub[[
            "expert_id",
            "removed_mean_score",
            "kept_mean_score",
            "delta_kept_minus_removed",
            "removed_count",
            "kept_count",
        ]].head(10))

    all_stats_df = compute_expert_stats_all(run_df)
    all_stats_df.to_csv(OUT_DIR / "expert_kept_vs_removed_summary_all_raw.csv", index=False)

    filtered_all_stats_df = filter_stats_df_all(all_stats_df)
    filtered_all_stats_df.to_csv(OUT_DIR / "expert_kept_vs_removed_summary_all_filtered.csv", index=False)

    save_all_sorted_table(filtered_all_stats_df)
    save_all_rank_text(filtered_all_stats_df)
    plot_all_expert_deltas(filtered_all_stats_df)
    plot_all_top_bottom_bar(filtered_all_stats_df, top_k=10)

    print(f"\n=== ALL M combined (removed_count >= {MIN_COUNT_ALL}, kept_count >= {MIN_COUNT_ALL}) ===")
    print(
        filtered_all_stats_df
        .sort_values("delta_kept_minus_removed", ascending=False)
        .reset_index(drop=True)[[
            "expert_id",
            "removed_mean_score",
            "kept_mean_score",
            "delta_kept_minus_removed",
            "removed_count",
            "kept_count",
        ]].head(15)
    )

    print(f"\nSaved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()