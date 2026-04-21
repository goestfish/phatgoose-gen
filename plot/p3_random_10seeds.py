#!/usr/bin/env python
import re
import ast
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


RAND_DIR = Path("exp_out/rand_10seeds")
OUT_DIR = Path("rand_10seeds_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def collect_runs():
    rows = []

    if not RAND_DIR.exists():
        raise FileNotFoundError(f"RAND_DIR not found: {RAND_DIR.resolve()}")

    for exp_dir in sorted(RAND_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        result_file = exp_dir / "results.txt"
        if not result_file.exists():
            print(f"Skipping {exp_dir.name}: no results.txt")
            continue

        result_dict = parse_result_txt(result_file)
        M, seed = parse_m_seed(exp_dir.name)

        dataset_scores = {}
        avg_from_file = None
        conf_from_file = None

        if isinstance(result_dict, dict):
            avg_raw = result_dict.get("average_score")
            conf_raw = result_dict.get("confidence")

            if isinstance(avg_raw, (int, float)):
                avg_from_file = float(avg_raw)
            if isinstance(conf_raw, (int, float)):
                conf_from_file = float(conf_raw)

            for ds, v in result_dict.items():
                s = dataset_score(v)
                if s is not None:
                    dataset_scores[ds] = s

        computed_mean = None
        if dataset_scores:
            computed_mean = sum(dataset_scores.values()) / len(dataset_scores)

        rows.append({
            "exp_name": exp_dir.name,
            "M": M,
            "seed": seed,
            "mean_score": computed_mean,
            "average_score_from_file": avg_from_file,
            "confidence_from_file": conf_from_file,
            "num_dataset_scores_found": len(dataset_scores),
            "dataset_scores": dataset_scores,
            "result_file": str(result_file),
        })

    return pd.DataFrame(rows)


def plot_mean_with_std(df: pd.DataFrame):
    g = (
        df.groupby("M")["mean_score"]
        .agg(["mean", "std", "var", "min", "max", "count"])
        .reset_index()
        .sort_values("M")
    )

    plt.figure(figsize=(7, 5))
    plt.errorbar(g["M"], g["mean"], yerr=g["std"], marker="o", capsize=4)
    plt.xlabel("Number of removed experts (M)")
    plt.ylabel("Mean BBH score")
    plt.title("Random expert removal: mean performance ± std")
    plt.xticks(g["M"])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "random_remove_mean_std.png", dpi=200)
    plt.close()

    g.to_csv(OUT_DIR / "random_remove_mean_std.csv", index=False)


def plot_scatter(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    Ms = sorted(df["M"].dropna().unique())

    for m in Ms:
        sub = df[df["M"] == m]
        plt.scatter([m] * len(sub), sub["mean_score"], alpha=0.8)

    plt.xlabel("Number of removed experts (M)")
    plt.ylabel("Mean BBH score")
    plt.title("Random expert removal: all seeds")
    plt.xticks(Ms)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "random_remove_scatter.png", dpi=200)
    plt.close()


def plot_boxplot(df: pd.DataFrame):
    Ms = sorted(df["M"].dropna().unique())
    data = [df[df["M"] == m]["mean_score"].dropna().tolist() for m in Ms]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=Ms)
    plt.xlabel("Number of removed experts (M)")
    plt.ylabel("Mean BBH score")
    plt.title("Random expert removal: score distribution across seeds")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "random_remove_boxplot.png", dpi=200)
    plt.close()


def plot_per_dataset(df: pd.DataFrame):
    dataset_to_scores = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        M = row["M"]
        for ds, score in row["dataset_scores"].items():
            dataset_to_scores[ds][M].append(score)

    datasets = sorted(dataset_to_scores.keys())
    summary_rows = []

    plt.figure(figsize=(10, 8))
    for ds in datasets:
        Ms = sorted(dataset_to_scores[ds].keys())
        means = []
        stds = []
        for m in Ms:
            vals = dataset_to_scores[ds][m]
            mean_val = sum(vals) / len(vals)
            std_val = pd.Series(vals).std(ddof=1) if len(vals) > 1 else 0.0
            means.append(mean_val)
            stds.append(std_val)
            summary_rows.append({
                "dataset": ds,
                "M": m,
                "mean_score": mean_val,
                "std": std_val,
                "var": std_val ** 2,
                "n": len(vals),
            })
        plt.plot(Ms, means, marker="o", label=ds)

    plt.xlabel("Number of removed experts (M)")
    plt.ylabel("Dataset score")
    plt.title("Per-dataset sensitivity to random expert removal")
    plt.xticks(sorted(df["M"].dropna().unique()))
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_dataset_lines.png", dpi=200)
    plt.close()

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "per_dataset_summary.csv", index=False)


def plot_dataset_heatmap(df: pd.DataFrame):
    dataset_to_scores = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        M = row["M"]
        for ds, score in row["dataset_scores"].items():
            dataset_to_scores[ds][M].append(score)

    datasets = sorted(dataset_to_scores.keys())
    Ms = sorted(df["M"].dropna().unique())

    mat = []
    for ds in datasets:
        row = []
        for m in Ms:
            vals = dataset_to_scores[ds].get(m, [])
            row.append(sum(vals) / len(vals) if vals else float("nan"))
        mat.append(row)

    heatmap_df = pd.DataFrame(mat, index=datasets, columns=Ms)
    heatmap_df.to_csv(OUT_DIR / "dataset_heatmap_values.csv")

    plt.figure(figsize=(8, max(6, 0.35 * len(datasets))))
    plt.imshow(heatmap_df.values, aspect="auto")
    plt.colorbar(label="Mean score")
    plt.yticks(range(len(datasets)), datasets)
    plt.xticks(range(len(Ms)), Ms)
    plt.xlabel("Number of removed experts (M)")
    plt.title("Per-dataset mean score heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dataset_heatmap.png", dpi=200)
    plt.close()


def main():
    df = collect_runs()

    if df.empty:
        print("No runs found.")
        return

    df = df.sort_values(["M", "seed"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "random_runs_raw.csv", index=False)

    print("Found runs:", len(df))
    print(df[[
        "exp_name",
        "M",
        "seed",
        "mean_score",
        "average_score_from_file",
        "confidence_from_file",
        "num_dataset_scores_found"
    ]])

    summary = (
        df.groupby("M")["mean_score"]
        .agg(["mean", "std", "var", "min", "max", "count"])
        .reset_index()
        .sort_values("M")
    )
    print("\nSummary by M:")
    print(summary)
    summary.to_csv(OUT_DIR / "random_remove_variance_summary.csv", index=False)

    plot_mean_with_std(df)
    plot_scatter(df)
    plot_boxplot(df)
    plot_per_dataset(df)
    plot_dataset_heatmap(df)

    print(f"Saved plots to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()