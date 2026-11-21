"""Step 4: compare robustness gaps across all models you have run.   python compare_models.py"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def size_b(name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", name)
    return float(m.group(1)) if m else float("nan")


def collect(kind):
    rows = []
    for f in sorted(Path("results").glob(f"*/summary_{kind}.json")):
        s = json.loads(f.read_text())
        if "_dbg" in s["model"]:
            continue
        rows.append(s)
    return rows


out = Path("results/_comparison")
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, kind, key, title in [(axes[0], "mmlu", "gap_cyclic", "MMLU: option-order robustness gap"),
                             (axes[1], "gsm8k", "gap_rename", "GSM8K: rename robustness gap")]:
    rows = collect(kind)
    if not rows:
        ax.axis("off"); continue
    df = pd.DataFrame(rows)
    df["size_b"] = df["model"].map(size_b)
    df = df.sort_values(["size_b", "model"])
    df.to_csv(out / f"compare_{kind}.csv", index=False)
    print(f"\n=== {kind.upper()} ===")
    print(df[["model", "n", "acc_orig", key]].round(3).to_string(index=False))
    gaps = df[key] * 100
    lo = gaps - df[key + "_ci"].map(lambda c: c[0]) * 100
    hi = df[key + "_ci"].map(lambda c: c[1]) * 100 - gaps
    ax.errorbar(range(len(df)), gaps, yerr=[lo.abs(), hi.abs()], fmt="o", capsize=4, color="#C44E52")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Gap (accuracy points): original minus perturbed"); ax.set_title(title, fontsize=11)
fig.tight_layout()
fig.savefig(out / "fig_compare_models.png", dpi=200)
fig.savefig(out / "fig_compare_models.pdf")
print(f"\nSaved to {out}/")
