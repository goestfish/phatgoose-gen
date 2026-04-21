import ast
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = "exp_out/p3_single_keep1"
BASELINE_DIR = "exp_out/p3_gen_base"
OUT_DIR = "plots_keep1"

TOP_BOTTOM_N = 10


EXPERT_DATASET_MAP = {
    0: "P3AGNEWS",
    1: "P3AMAZONPOLARITY",
    2: "P3COSMOSQA",
    3: "P3SAMSUM",
    4: "P3QUARTZ",
    5: "P3ROPES",
    6: "P3WIKIBIO",
    7: "P3PAWS",
    8: "P3WIKIQA",
    9: "P3SOCIALIQA",
    10: "P3QASC",
    11: "P3QUAIL",
    12: "P3DREAM",
    13: "P3WIQA",
    14: "P3QUAREL",
    15: "P3SCIQ",
    16: "P3QUOREF",
    17: "P3DUORC",
    18: "P3ROTTENTOMATOES",
    19: "P3YELP",
    20: "P3COMMONGEN",
    21: "P3GIGAWORD",
    22: "P3XSUM",
    23: "P3MRPC",
    24: "P3QQP",
    25: "P3COMMONSENSEQA",
    26: "P3COSE",
    27: "P3WIKIHOP",
    28: "P3HOTPOTQA",
    29: "P3APPREVIEWS",
    30: "P3TREC",
    31: "P3MULTINEWS",
    32: "P3IMDB",
    33: "P3ADVERSARIALQA",
    34: "P3CNNDAILYMAIL",
    35: "P3DBPEDIA14",
}


def parse_result_payload(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore").strip()

    if "Final results:" in text:
        text = text.split("Final results:", 1)[1].strip()

    text = re.sub(r"np\.float64\(([^)]+)\)", r"\1", text)
    text = re.sub(r"np\.float32\(([^)]+)\)", r"\1", text)

    try:
        data = ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Failed to parse {path}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Parsed payload is not a dict in {path}")
    return data


def find_result_file(path_like: Path):
    path_like = Path(path_like)

    if path_like.is_file():
        return path_like

    direct_candidates = [
        path_like / "result.txt",
        path_like / "results.txt",
    ]
    for c in direct_candidates:
        if c.exists():
            return c

    for pattern in ("result.txt", "results.txt"):
        matches = sorted(path_like.rglob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not find result.txt or results.txt under {path_like}")


def task_score(entry):
    if isinstance(entry, dict):
        if "score" in entry:
            return float(entry["score"])
        for key in ("exact_match", "exact_match_multiple_ans", "accuracy"):
            if key in entry:
                return float(entry[key])
    raise ValueError(f"Could not extract score from entry: {entry}")


def clean_task_name(task: str) -> str:
    return task.replace("D/BB", "").replace("/EVAL", "")


def short_dataset_name(name: str) -> str:
    if not name:
        return "UNKNOWN"
    if name.startswith("P3"):
        return name[2:]
    return name


def expert_dataset(expert_id: int) -> str:
    return EXPERT_DATASET_MAP.get(expert_id, "UNKNOWN")


def expert_label(expert_id: int) -> str:
    return f"e{expert_id}:{short_dataset_name(expert_dataset(expert_id))}"


def parse_expert_id(exp_dir: Path):
    meta_path = exp_dir / "run_meta.json"
    if meta_path.exists():
        try:
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "expert_id" in meta:
                return int(meta["expert_id"]), meta
        except Exception:
            pass

    m = re.search(r"_e(\d+)$", exp_dir.name)
    if not m:
        raise ValueError(f"Could not infer expert_id from {exp_dir}")
    return int(m.group(1)), None


def collect_runs(base_dir: Path):
    runs = []

    for exp_dir in sorted(base_dir.glob("p3_gen_keep1_e*")):
        if not exp_dir.is_dir():
            continue

        try:
            result_file = find_result_file(exp_dir)
        except FileNotFoundError:
            print(f"[WARN] skipping {exp_dir}: no result file found")
            continue

        result = parse_result_payload(result_file)
        expert_id, meta = parse_expert_id(exp_dir)

        run = {
            "exp_dir": exp_dir,
            "exp_name": exp_dir.name,
            "expert_id": expert_id,
            "dataset_name": expert_dataset(expert_id),
            "dataset_short": short_dataset_name(expert_dataset(expert_id)),
            "label": expert_label(expert_id),
            "average_score": float(result.get("average_score", float("nan"))),
            "confidence": float(result.get("confidence", float("nan"))) if "confidence" in result else float("nan"),
            "tasks": {},
            "meta": meta,
        }

        for key, value in result.items():
            if key in {"average_score", "confidence"}:
                continue
            if isinstance(value, dict):
                run["tasks"][key] = task_score(value)

        runs.append(run)

    runs.sort(key=lambda x: x["expert_id"])
    return runs


def load_baseline(baseline_dir: Path):
    baseline_file = find_result_file(baseline_dir)
    baseline_raw = parse_result_payload(baseline_file)

    baseline_avg = float(baseline_raw["average_score"])
    baseline_tasks = {
        key: task_score(value)
        for key, value in baseline_raw.items()
        if key not in {"average_score", "confidence"} and isinstance(value, dict)
    }
    return baseline_avg, baseline_tasks


def save_summary_csv(runs, baseline_avg, out_dir: Path):
    out_path = out_dir / "single_keep1_summary.csv"
    rows = []
    for run in runs:
        rows.append({
            "expert_id": run["expert_id"],
            "dataset_name": run["dataset_name"],
            "dataset_short": run["dataset_short"],
            "label": run["label"],
            "average_score": run["average_score"],
            "confidence": run["confidence"],
            "delta_vs_baseline": run["average_score"] - baseline_avg,
            "exp_name": run["exp_name"],
            "exp_dir": str(run["exp_dir"]),
        })

    df = pd.DataFrame(rows).sort_values(["average_score", "expert_id"], ascending=[False, True])
    df.to_csv(out_path, index=False)
    return df, out_path


def save_task_csv(runs, baseline_tasks, out_dir: Path):
    out_path = out_dir / "single_keep1_task_scores.csv"
    all_tasks = sorted({t for run in runs for t in run["tasks"].keys()})

    rows = []
    for run in runs:
        for task in all_tasks:
            score = run["tasks"].get(task, float("nan"))
            baseline = baseline_tasks.get(task, float("nan"))
            delta = score - baseline if not (math.isnan(score) or math.isnan(baseline)) else float("nan")
            rows.append({
                "expert_id": run["expert_id"],
                "dataset_name": run["dataset_name"],
                "dataset_short": run["dataset_short"],
                "label": run["label"],
                "task": task,
                "task_clean": clean_task_name(task),
                "score": score,
                "baseline_score": baseline,
                "delta_vs_baseline": delta,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df, out_path


def plot_average_score_sorted(summary_df: pd.DataFrame, baseline_avg: float, out_dir: Path):
    df = summary_df.sort_values(["average_score", "expert_id"], ascending=[False, True]).reset_index(drop=True)

    y = np.arange(len(df))
    labels = df["label"].tolist()
    values = df["average_score"].tolist()

    plt.figure(figsize=(11, max(8, 0.32 * len(df) + 2)))
    plt.barh(y, values)
    plt.axvline(baseline_avg, linestyle="--")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Average BBH score")
    plt.title("Single-expert BBH average score (sorted)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out = out_dir / "single_keep1_average_score_sorted.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_delta_vs_baseline_sorted(summary_df: pd.DataFrame, out_dir: Path):
    df = summary_df.sort_values(["delta_vs_baseline", "expert_id"], ascending=[False, True]).reset_index(drop=True)

    y = np.arange(len(df))
    labels = df["label"].tolist()
    values = df["delta_vs_baseline"].tolist()

    plt.figure(figsize=(11, max(8, 0.32 * len(df) + 2)))
    plt.barh(y, values)
    plt.axvline(0.0, linestyle="--")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Average BBH score delta vs baseline")
    plt.title("Single-expert performance delta vs baseline (sorted)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out = out_dir / "single_keep1_vs_baseline_sorted.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_top_bottom(summary_df: pd.DataFrame, out_dir: Path, n_each: int = 10):
    df = summary_df.sort_values(["average_score", "expert_id"], ascending=[True, True]).reset_index(drop=True)
    bottom = df.head(n_each)
    top = df.tail(n_each)
    selected = pd.concat([bottom, top], axis=0).reset_index(drop=True)

    y = np.arange(len(selected))
    labels = selected["label"].tolist()
    values = selected["average_score"].tolist()

    plt.figure(figsize=(11, max(6, 0.35 * len(selected) + 2)))
    plt.barh(y, values)
    plt.yticks(y, labels, fontsize=9)
    plt.xlabel("Average BBH score")
    plt.title(f"Bottom {n_each} and top {n_each} single experts")
    plt.tight_layout()

    out = out_dir / "single_keep1_top_bottom_average_score.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_task_heatmap(task_df: pd.DataFrame, out_dir: Path):
    pivot = task_df.pivot(index="label", columns="task_clean", values="score")

    avg_map = (
        task_df[["label", "score"]]
        .groupby("label")
        .mean()
        .rename(columns={"score": "mean_score"})
    )
    label_order = avg_map.sort_values("mean_score", ascending=False).index.tolist()
    pivot = pivot.loc[label_order]

    plt.figure(figsize=(max(12, 0.45 * len(pivot.columns) + 4), max(10, 0.28 * len(pivot.index) + 3)))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Task score")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=8)
    plt.xlabel("BBH task")
    plt.ylabel("Single kept expert")
    plt.title("Single-expert task score heatmap")
    plt.tight_layout()

    out = out_dir / "single_keep1_task_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    pivot.to_csv(out_dir / "single_keep1_task_heatmap_values.csv")
    return out


def plot_top_expert_per_task(task_df: pd.DataFrame, out_dir: Path):
    rows = []
    for task, sub in task_df.groupby("task_clean"):
        best = sub.sort_values(["score", "expert_id"], ascending=[False, True]).iloc[0]
        rows.append({
            "task_clean": task,
            "best_expert_id": int(best["expert_id"]),
            "best_label": best["label"],
            "best_dataset": best["dataset_name"],
            "best_score": float(best["score"]),
        })

    best_df = pd.DataFrame(rows).sort_values(["best_score", "task_clean"], ascending=[False, True]).reset_index(drop=True)
    best_df.to_csv(out_dir / "single_keep1_best_expert_per_task.csv", index=False)

    top_show = best_df.head(min(15, len(best_df))).copy()
    y = np.arange(len(top_show))

    plt.figure(figsize=(11, max(6, 0.35 * len(top_show) + 2)))
    plt.barh(y, top_show["best_score"].tolist())
    plt.yticks(y, top_show["task_clean"].tolist(), fontsize=9)
    plt.xlabel("Best single-expert task score")
    plt.title("Best single-expert score for selected BBH tasks")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out = out_dir / "single_keep1_best_expert_per_task.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out, best_df


def write_summary_txt(summary_df: pd.DataFrame, best_df: pd.DataFrame, baseline_avg: float, out_dir: Path):
    top_df = summary_df.sort_values(["average_score", "expert_id"], ascending=[False, True]).head(10)
    bottom_df = summary_df.sort_values(["average_score", "expert_id"], ascending=[True, True]).head(10)

    lines = []
    lines.append(f"Baseline average_score: {baseline_avg:.6f}")
    lines.append("")
    lines.append("Top 10 single experts by average score:")
    for _, row in top_df.iterrows():
        lines.append(
            f"  e{int(row['expert_id'])} ({row['dataset_name']}): "
            f"avg={row['average_score']:.6f}, "
            f"delta={row['delta_vs_baseline']:+.6f}, "
            f"confidence={row['confidence']:.6f}"
        )

    lines.append("")
    lines.append("Bottom 10 single experts by average score:")
    for _, row in bottom_df.iterrows():
        lines.append(
            f"  e{int(row['expert_id'])} ({row['dataset_name']}): "
            f"avg={row['average_score']:.6f}, "
            f"delta={row['delta_vs_baseline']:+.6f}, "
            f"confidence={row['confidence']:.6f}"
        )

    lines.append("")
    lines.append("Best single expert per task:")
    for _, row in best_df.iterrows():
        lines.append(
            f"  {row['task_clean']}: e{int(row['best_expert_id'])} "
            f"({row['best_dataset']}) score={row['best_score']:.6f}"
        )

    out = out_dir / "single_keep1_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    base_dir = Path(BASE_DIR)
    baseline_dir = Path(BASELINE_DIR)
    out_dir = Path(OUT_DIR)

    if not base_dir.exists():
        raise FileNotFoundError(f"BASE_DIR does not exist: {base_dir}")
    if not baseline_dir.exists():
        raise FileNotFoundError(f"BASELINE_DIR does not exist: {baseline_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_avg, baseline_tasks = load_baseline(baseline_dir)
    runs = collect_runs(base_dir)

    if not runs:
        raise RuntimeError(f"No runs found under {base_dir}")

    summary_df, summary_csv = save_summary_csv(runs, baseline_avg, out_dir)
    task_df, task_csv = save_task_csv(runs, baseline_tasks, out_dir)

    files = []
    files.append(summary_csv)
    files.append(task_csv)
    files.append(plot_average_score_sorted(summary_df, baseline_avg, out_dir))
    files.append(plot_delta_vs_baseline_sorted(summary_df, out_dir))
    files.append(plot_top_bottom(summary_df, out_dir, n_each=TOP_BOTTOM_N))
    files.append(plot_task_heatmap(task_df, out_dir))
    best_plot, best_df = plot_top_expert_per_task(task_df, out_dir)
    files.append(best_plot)
    files.append(write_summary_txt(summary_df, best_df, baseline_avg, out_dir))

    print("Done. Saved files:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()