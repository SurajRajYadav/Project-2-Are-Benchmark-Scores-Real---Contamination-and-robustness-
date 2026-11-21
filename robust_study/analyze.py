"""Step 3: robustness gaps, position-bias checks, and contamination-signal analysis for one model.

    python analyze.py --run results/Qwen2.5-1.5B
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robust.data import read_jsonl
from robust.metrics import bootstrap_ci, extract_number, is_correct_number, paired_counts, spearman_ci

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
CHANCE = 0.25


def pct(x):
    return f"{100 * x:+.1f}"


def quartile_plot(ax, signal, d, title, xlabel):
    """Mean 'original-version advantage' per quartile of a contamination signal (with 95% CIs)."""
    s = pd.Series(np.asarray(signal, dtype=float))
    q = pd.qcut(s.rank(method="first"), 4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
    means, los, his = [], [], []
    for lab in q.cat.categories:
        vals = np.asarray(d)[(q == lab).values]
        lo, hi = bootstrap_ci(vals)
        means.append(vals.mean()); los.append(vals.mean() - lo); his.append(hi - vals.mean())
    ax.bar(range(4), means, yerr=[los, his], capsize=4, color="#4C72B0")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(q.cat.categories)
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel); ax.set_ylabel("Accuracy on original minus perturbed")


#  MMLU
def analyze_mmlu(run_dir):
    rows = []
    for f in sorted(run_dir.glob("raw_mmlu_*.jsonl")):
        rows += read_jsonl(f)
    long = pd.DataFrame(rows)
    long["correct"] = (long["pred"] == long["gold"]).astype(float)
    wide = long.pivot(index="id", columns="cond", values="correct").dropna()
    order = ["orig"] + sorted([c for c in wide if c.startswith("shift")]) + sorted([c for c in wide if c.startswith("rand")])
    wide = wide[order]
    cyc = [c for c in order if c.startswith("shift")]
    rnd = [c for c in order if c.startswith("rand")]
    n = len(wide)
    print(f"\n==================== MMLU: {n} items, conditions {order} ====================")

    # 1) accuracy per condition
    acc = pd.DataFrame([{"condition": c, "accuracy": wide[c].mean(), "ci_lo": bootstrap_ci(wide[c])[0],
                         "ci_hi": bootstrap_ci(wide[c])[1]} for c in order])
    print(acc.round(3).to_string(index=False))
    acc.round(4).to_csv(run_dir / "table_mmlu_accuracy.csv", index=False)
    acc_orig = wide["orig"].mean()
    if acc_orig < CHANCE + 0.05:
        print(f"\n!! Original accuracy {acc_orig:.3f} is near chance ({CHANCE}). Gaps are uninterpretable "
              f"for near-chance models (use as a NEGATIVE CONTROL only).")

    # 2) robustness gap = accuracy(original order) - accuracy(perturbed orders), paired over items
    summary = {"model": run_dir.name, "n": int(n), "acc_orig": float(acc_orig)}
    for name, cols in [("cyclic", cyc), ("random", rnd)]:
        if not cols:
            continue
        d = (wide["orig"] - wide[cols].mean(axis=1)).values       # per-item advantage of the original order
        lo, hi = bootstrap_ci(d)
        summary[f"gap_{name}"] = float(d.mean()); summary[f"gap_{name}_ci"] = [lo, hi]
        summary[f"acc_{name}"] = float(wide[cols].mean(axis=1).mean())
        print(f"\nRobustness gap ({name} reorderings): {pct(d.mean())} points "
              f"[95% CI {pct(lo)}, {pct(hi)}]   relative drop {100 * d.mean() / max(acc_orig, 1e-9):.1f}%")
        if name == "cyclic":
            d_cyc = d

    # 3) consistency: does the model pick the same ORIGINAL option after reordering?
    po = long.pivot(index="id", columns="cond", values="pred_orig").loc[wide.index]
    cons = {c: float((po[c] == po["orig"]).mean()) for c in order if c != "orig"}
    summary["consistency"] = float(np.mean(list(cons.values())))
    print("\nPrediction consistency with the original-order answer:", {k: round(v, 3) for k, v in cons.items()})

    # 4) position bias (a confound!): letter preference and accuracy by gold position, pooled over conditions
    bias = pd.DataFrame({"predicted_share": long["pred"].value_counts(normalize=True).sort_index(),
                         "accuracy_when_gold_here": long.groupby("gold")["correct"].mean()})
    bias.index = list("ABCD")
    print("\nPosition bias (pooled over all conditions):\n", bias.round(3))
    bias.round(4).to_csv(run_dir / "table_mmlu_position_bias.csv")

    # 5) contamination signals vs per-item original-order advantage
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.bar(range(len(order)), acc["accuracy"], yerr=[acc["accuracy"] - acc["ci_lo"], acc["ci_hi"] - acc["accuracy"]],
           capsize=3, color=["#C44E52"] + ["#4C72B0"] * (len(order) - 1))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45)
    ax.set_ylim(max(0, acc["ci_lo"].min() - 0.05), min(1, acc["ci_hi"].max() + 0.05))
    ax.set_ylabel("Accuracy"); ax.set_title("MMLU accuracy by option order")
    sig_path = run_dir / "signals_mmlu.jsonl"
    if sig_path.exists() and cyc:
        sig = pd.DataFrame(read_jsonl(sig_path)).set_index("id").loc[wide.index]
        d_series = pd.Series(d_cyc, index=wide.index)
        print("\nContamination-signal correlations with original-order advantage (Spearman, 95% bootstrap CI):")
        for k, (col, label) in enumerate([("order_gap", "Order-likelihood gap"), ("min_k", "Min-K% Prob")]):
            rho, lo, hi, p = spearman_ci(sig[col], d_series)
            summary[f"spearman_{col}"] = rho; summary[f"spearman_{col}_ci"] = [lo, hi]
            print(f"  {label:22s} rho={rho:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.3g}")
            quartile_plot(axes[k + 1], sig[col], d_series.values, f"{label}  (rho={rho:+.2f})", f"{col} quartile")
    else:
        for a in axes[1:]:
            a.axis("off")
    fig.tight_layout()
    (run_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(run_dir / "figures" / "fig_mmlu.png", dpi=200)
    fig.savefig(run_dir / "figures" / "fig_mmlu.pdf")
    (run_dir / "summary_mmlu.json").write_text(json.dumps(summary, indent=2))


#  GSM8K
def analyze_gsm8k(run_dir):
    recs = {}
    for f in sorted(run_dir.glob("raw_gsm8k_*.jsonl")):
        for r in read_jsonl(f):
            correct = float(is_correct_number(extract_number(r["raw"]), r["gold"]))
            recs.setdefault(r["id"], {"id": r["id"]})[f"c_{r['cond']}"] = correct
            recs[r["id"]][f"ch_{r['cond']}"] = r["changed"]
    df = pd.DataFrame(recs.values())
    conds = [c for c in ["orig", "rename", "distractor"] if f"c_{c}" in df]
    df = df.dropna(subset=[f"c_{c}" for c in conds])
    print(f"\n==================== GSM8K: {len(df)} items, conditions {conds} ====================")
    summary = {"model": run_dir.name, "n": int(len(df))}
    table = []
    for c in conds:
        lo, hi = bootstrap_ci(df[f"c_{c}"])
        table.append({"condition": c, "accuracy": df[f"c_{c}"].mean(), "ci_lo": lo, "ci_hi": hi})
    print(pd.DataFrame(table).round(3).to_string(index=False))
    pd.DataFrame(table).round(4).to_csv(run_dir / "table_gsm8k_accuracy.csv", index=False)
    summary["acc_orig"] = float(df["c_orig"].mean())

    for c in [x for x in conds if x != "orig"]:
        sub = df[df[f"ch_{c}"] == True] if c == "rename" else df          # rename: only items with names
        d = (sub["c_orig"] - sub[f"c_{c}"]).values
        lo, hi = bootstrap_ci(d)
        summary[f"gap_{c}"] = float(d.mean()); summary[f"gap_{c}_ci"] = [lo, hi]; summary[f"n_{c}"] = int(len(sub))
        pc = paired_counts(sub["c_orig"], sub[f"c_{c}"])
        print(f"\n[{c}] items={len(sub)}  gap={pct(d.mean())} pts [{pct(lo)}, {pct(hi)}]  paired={pc}")

    sig_path = run_dir / "signals_gsm8k.jsonl"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(range(len(conds)), [t["accuracy"] for t in table],
                yerr=[[t["accuracy"] - t["ci_lo"] for t in table], [t["ci_hi"] - t["accuracy"] for t in table]],
                capsize=4, color=["#C44E52"] + ["#4C72B0"] * (len(conds) - 1))
    axes[0].set_xticks(range(len(conds))); axes[0].set_xticklabels(conds)
    axes[0].set_ylabel("Accuracy"); axes[0].set_title("GSM8K accuracy by variant")
    if sig_path.exists() and "rename" in conds:
        sig = pd.DataFrame(read_jsonl(sig_path)).set_index("id")
        sub = df[df["ch_rename"] == True].set_index("id").join(sig, how="inner")
        d = (sub["c_orig"] - sub["c_rename"]).values
        print("\nSignal correlations with (original - renamed) correctness, items with names:")
        for k, (col, label) in enumerate([("rename_gap", "Rename log-prob gap"), ("min_k", "Min-K% Prob")]):
            rho, lo, hi, p = spearman_ci(sub[col], d)
            summary[f"spearman_{col}"] = rho; summary[f"spearman_{col}_ci"] = [lo, hi]
            print(f"  {label:22s} rho={rho:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.3g}")
            quartile_plot(axes[k + 1], sub[col], d, f"{label} (rho={rho:+.2f})", f"{col} quartile")
    else:
        for a in axes[1:]:
            a.axis("off")
    fig.tight_layout()
    (run_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(run_dir / "figures" / "fig_gsm8k.png", dpi=200)
    fig.savefig(run_dir / "figures" / "fig_gsm8k.pdf")
    (run_dir / "summary_gsm8k.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run_dir = Path(ap.parse_args().run)
    if (run_dir / "raw_mmlu_orig.jsonl").exists():
        analyze_mmlu(run_dir)
    if (run_dir / "raw_gsm8k_orig.jsonl").exists():
        analyze_gsm8k(run_dir)
    print(f"\nSaved tables, figures and summary JSON in {run_dir}/")


if __name__ == "__main__":
    main()
