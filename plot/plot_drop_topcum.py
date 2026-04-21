import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = "exp_out/drop_1-10"
BASELINE_DIR = "exp_out/p3_gen_base"
OUT_DIR = "plots_topcum"

TOP_N_TASK_CURVES = 8

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
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No dict payload found in {path}")

    payload = text[start:]
    data = eval(payload, {"__builtins__": {}}, {"np": np})  # noqa: S307
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


def parse_k(exp_dir: Path):
    meta_path = exp_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "num_removed" in meta:
                return int(meta["num_removed"]), meta
        except Exception:
            pass

    m = re.search(r"topcum(\d+)", exp_dir.name)
    if not m:
        raise ValueError(f"Could not infer K from {exp_dir}")
    return int(m.group(1)), None


def get_dataset_for_expert(expert_id):
    return EXPERT_DATASET_MAP.get(expert_id, "UNKNOWN")


def format_removed_experts_with_datasets(expert_list):
    if not expert_list:
        return ""
    parts = []
    for eid in expert_list:
        parts.append(f"{eid}:{get_dataset_for_expert(eid)}")
    return "; ".join(parts)

def short_dataset_name(name):
    if not name or name == "UNKNOWN":
        return "UNKNOWN"

    name = str(name)
    if name.startswith("P3"):
        name = name[2:]
    return name


def point_label_for_run(run, max_items=3):
    removed = run["removed_experts"] if run["removed_experts"] is not None else []
    if not removed:
        return f"K={run['k']}"

    parts = []
    for eid in removed[:max_items]:
        ds = short_dataset_name(get_dataset_for_expert(eid))
        parts.append(f"{eid}:{ds}")

    if len(removed) > max_items:
        parts.append("...")

    return "\n".join(parts)


def collect_runs(base_dir: Path):
    runs = []

    for exp_dir in sorted(base_dir.glob("p3_gen_drop_topcum*")):
        if not exp_dir.is_dir():
            continue

        try:
            result_file = find_result_file(exp_dir)
        except FileNotFoundError:
            print(f"[WARN] skipping {exp_dir}: no result file found")
            continue

        result = parse_result_payload(result_file)
        k, meta = parse_k(exp_dir)

        run = {
            "exp_dir": exp_dir,
            "exp_name": exp_dir.name,
            "k": k,
            "average_score": float(result.get("average_score", float("nan"))),
            "confidence": float(result.get("confidence", float("nan"))) if "confidence" in result else float("nan"),
            "tasks": {},
            "removed_experts": None,
            "kept_experts": None,
        }

        if meta:
            run["removed_experts"] = meta.get("removed_experts")
            run["kept_experts"] = meta.get("kept_experts")

        for key, value in result.items():
            if key in {"average_score", "confidence"}:
                continue
            if isinstance(value, dict):
                run["tasks"][key] = task_score(value)

        runs.append(run)

    runs.sort(key=lambda x: x["k"])
    return runs


def save_summary_csv(runs, baseline_avg, out_dir: Path):
    out_path = out_dir / "summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_name",
            "k",
            "average_score",
            "confidence",
            "delta_vs_baseline",
            "removed_experts",
            "removed_expert_datasets",
            "removed_experts_with_datasets",
            "kept_experts",
            "exp_dir",
        ])
        for run in runs:
            removed = run["removed_experts"] if run["removed_experts"] is not None else []
            removed_datasets = [get_dataset_for_expert(eid) for eid in removed]

            writer.writerow([
                run["exp_name"],
                run["k"],
                run["average_score"],
                run["confidence"],
                run["average_score"] - baseline_avg,
                json.dumps(removed),
                json.dumps(removed_datasets, ensure_ascii=False),
                format_removed_experts_with_datasets(removed),
                json.dumps(run["kept_experts"]) if run["kept_experts"] is not None else "",
                str(run["exp_dir"]),
            ])
    return out_path


def save_removed_experts_csv(runs, out_dir: Path):
    out_path = out_dir / "removed_experts_by_k.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k",
            "exp_name",
            "removed_rank",
            "expert_id",
            "training_dataset",
        ])

        for run in runs:
            removed = run["removed_experts"] if run["removed_experts"] is not None else []
            for rank, expert_id in enumerate(removed, start=1):
                writer.writerow([
                    run["k"],
                    run["exp_name"],
                    rank,
                    expert_id,
                    get_dataset_for_expert(expert_id),
                ])
    return out_path


def save_task_csv(runs, baseline_tasks, out_dir: Path):
    out_path = out_dir / "task_scores_and_deltas.csv"
    all_tasks = sorted({t for run in runs for t in run["tasks"].keys()})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "task", "task_clean", "score", "baseline_score", "delta_vs_baseline"])
        for run in runs:
            for task in all_tasks:
                score = run["tasks"].get(task, float("nan"))
                baseline = baseline_tasks.get(task, float("nan"))
                delta = score - baseline if not (math.isnan(score) or math.isnan(baseline)) else float("nan")
                writer.writerow([run["k"], task, clean_task_name(task), score, baseline, delta])

    return out_path


def plot_average_score(runs, baseline_avg, out_dir: Path):
    ks = [r["k"] for r in runs]
    ys = [r["average_score"] for r in runs]

    plt.figure(figsize=(11, 6))
    plt.plot(ks, ys, marker="o")
    plt.axhline(baseline_avg, linestyle="--", label=f"baseline={baseline_avg:.4f}")
    plt.xticks(ks)
    plt.xlabel("Number of top-used experts removed (M)")
    plt.ylabel("Average score")
    plt.title("Average BBH score vs cumulative top-expert removal")

    for i, run in enumerate(runs):
        label = point_label_for_run(run, max_items=3)
        offset_y = 10 if i % 2 == 0 else -28

        plt.annotate(
            label,
            (run["k"], run["average_score"]),
            textcoords="offset points",
            xytext=(0, offset_y),
            ha="center",
            fontsize=8,
        )

    plt.legend()
    plt.tight_layout()

    out = out_dir / "average_score_vs_k.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out

def plot_delta_vs_baseline(runs, baseline_avg, out_dir: Path):
    ks = [r["k"] for r in runs]
    deltas = [r["average_score"] - baseline_avg for r in runs]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, deltas, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xticks(ks)
    plt.xlabel("Number of top-used experts removed (K)")
    plt.ylabel("Average score delta vs baseline")
    plt.title("Change in BBH average score vs baseline")
    plt.tight_layout()

    out = out_dir / "delta_vs_baseline_vs_k.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_selected_task_curves(runs, baseline_tasks, out_dir: Path, top_n: int = 8):
    all_tasks = sorted({t for run in runs for t in run["tasks"].keys() if t in baseline_tasks})
    if not all_tasks:
        return None

    task_strength = {}
    for task in all_tasks:
        deltas = []
        for run in runs:
            if task in run["tasks"]:
                deltas.append(run["tasks"][task] - baseline_tasks[task])
        valid = [abs(x) for x in deltas if not np.isnan(x)]
        task_strength[task] = max(valid) if valid else -1

    selected = sorted(all_tasks, key=lambda t: task_strength[t], reverse=True)[:top_n]

    plt.figure(figsize=(10, 6))
    for task in selected:
        ks = []
        ds = []
        for run in runs:
            if task not in run["tasks"]:
                continue
            ks.append(run["k"])
            ds.append(run["tasks"][task] - baseline_tasks[task])
        plt.plot(ks, ds, marker="o", label=clean_task_name(task))

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Number of top-used experts removed (K)")
    plt.ylabel("Task score delta vs baseline")
    plt.title(f"Task deltas for top {len(selected)} most affected BBH tasks")
    plt.xticks([r["k"] for r in runs])
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = out_dir / "selected_task_delta_vs_k.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_final_top_bottom_tasks(runs, baseline_tasks, out_dir: Path, n_each: int = 10):
    final_run = max(runs, key=lambda r: r["k"])

    rows = []
    for task, baseline in baseline_tasks.items():
        if task not in final_run["tasks"]:
            continue
        delta = final_run["tasks"][task] - baseline
        rows.append((task, delta))

    if not rows:
        return None

    rows_sorted = sorted(rows, key=lambda x: x[1])
    bottom = rows_sorted[:n_each]
    top = rows_sorted[-n_each:]
    selected = bottom + top

    labels = [clean_task_name(t) for t, _ in selected]
    values = [d for _, d in selected]
    y = np.arange(len(selected))

    plt.figure(figsize=(10, max(6, 0.35 * len(selected) + 2)))
    plt.barh(y, values)
    plt.axvline(0.0, linestyle="--")
    plt.yticks(y, labels, fontsize=9)
    plt.xlabel(f"Task score delta vs baseline at K={final_run['k']}")
    plt.title(f"Top {n_each} drops and top {n_each} gains at K={final_run['k']}")
    plt.tight_layout()

    out = out_dir / f"final_k{final_run['k']}_top_bottom_tasks.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def write_text_summary(runs, baseline_avg, baseline_tasks, out_dir: Path):
    final_run = max(runs, key=lambda r: r["k"])

    lines = []
    lines.append(f"Baseline average_score: {baseline_avg:.6f}")
    lines.append("")
    lines.append("Per-K average scores:")
    for run in runs:
        lines.append(
            f"  K={run['k']:>2}: avg={run['average_score']:.6f}, "
            f"delta={run['average_score'] - baseline_avg:+.6f}, "
            f"confidence={run['confidence']:.6f}"
        )
        removed = run["removed_experts"] if run["removed_experts"] is not None else []
        if removed:
            lines.append(f"      removed experts: {format_removed_experts_with_datasets(removed)}")

    lines.append("")
    lines.append(
        f"Largest K run: K={final_run['k']}, avg={final_run['average_score']:.6f}, "
        f"delta={final_run['average_score'] - baseline_avg:+.6f}"
    )

    final_removed = final_run["removed_experts"] if final_run["removed_experts"] is not None else []
    if final_removed:
        lines.append(f"Removed experts at largest K: {format_removed_experts_with_datasets(final_removed)}")

    rows = []
    for task, base in baseline_tasks.items():
        if task in final_run["tasks"]:
            rows.append((task, final_run["tasks"][task] - base))
    rows.sort(key=lambda x: x[1])

    lines.append("Worst 5 task deltas at largest K:")
    for task, delta in rows[:5]:
        lines.append(f"  {clean_task_name(task)}: {delta:+.6f}")

    lines.append("Best 5 task deltas at largest K:")
    for task, delta in rows[-5:]:
        lines.append(f"  {clean_task_name(task)}: {delta:+.6f}")

    out = out_dir / "summary.txt"
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

    baseline_file = find_result_file(baseline_dir)
    baseline_raw = parse_result_payload(baseline_file)

    baseline_avg = float(baseline_raw["average_score"])
    baseline_tasks = {
        key: task_score(value)
        for key, value in baseline_raw.items()
        if key not in {"average_score", "confidence"} and isinstance(value, dict)
    }

    runs = collect_runs(base_dir)
    if not runs:
        raise RuntimeError(f"No runs found under {base_dir}")

    files = []
    files.append(save_summary_csv(runs, baseline_avg, out_dir))
    files.append(save_removed_experts_csv(runs, out_dir))
    files.append(save_task_csv(runs, baseline_tasks, out_dir))
    files.append(plot_average_score(runs, baseline_avg, out_dir))
    files.append(plot_delta_vs_baseline(runs, baseline_avg, out_dir))

    maybe = plot_selected_task_curves(
        runs,
        baseline_tasks,
        out_dir,
        top_n=TOP_N_TASK_CURVES,
    )
    if maybe is not None:
        files.append(maybe)

    maybe = plot_final_top_bottom_tasks(
        runs,
        baseline_tasks,
        out_dir,
        n_each=TOP_BOTTOM_N,
    )
    if maybe is not None:
        files.append(maybe)

    files.append(write_text_summary(runs, baseline_avg, baseline_tasks, out_dir))

    print("Done. Saved files:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()