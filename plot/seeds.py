#!/usr/bin/env python
import json
import re
from pathlib import Path

import pandas as pd


META_DIR = Path("exp_out/p3_random_5_10_20_30")
OUT_DIR = Path("rand_10seeds_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_m_seed(exp_name: str):
    m = re.search(r"_m(\d+)_s(\d+)", exp_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def main():
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
        removed_str = ",".join(map(str, removed))

        rows.append({
            "seed": seed,
            "M": M,
            "removed_experts": removed_str,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No runs found.")
        return

    table = (
        df.pivot(index="seed", columns="M", values="removed_experts")
          .sort_index(axis=0)
          .sort_index(axis=1)
    )

    table.columns = [f"M={c}" for c in table.columns]
    table.index.name = "seed"

    csv_path = OUT_DIR / "removed_experts_one_table.csv"
    md_path = OUT_DIR / "removed_experts_one_table.md"

    table.to_csv(csv_path)

    with open(md_path, "w") as f:
        cols = ["seed"] + list(table.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for seed, row in table.iterrows():
            vals = [str(seed)] + [str(row[c]) if pd.notna(row[c]) else "" for c in table.columns]
            f.write("| " + " | ".join(vals) + " |\n")

    print(table)
    print(f"\nSaved to:\n{csv_path.resolve()}\n{md_path.resolve()}")


if __name__ == "__main__":
    main()