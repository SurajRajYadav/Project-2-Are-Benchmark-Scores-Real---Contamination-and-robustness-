"""Step 2a: MMLU under reordered options + contamination signals.

    python run_mmlu.py --model Qwen/Qwen2.5-1.5B --limit 50     # debug
    python run_mmlu.py --model Qwen/Qwen2.5-1.5B                # full

Conditions: orig | shift1..shiftS (cyclic) | rand1..randR (random shuffles).
Each raw record stores the reordering (perm) so we can map predictions back to the ORIGINAL options.
Resumable: re-run the same command after a disconnect.
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from robust.data import read_jsonl, write_jsonl
from robust.model import LM, min_k_prob
from robust.perturb import apply_perm, make_perm, mmlu_body, mmlu_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="data")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--shifts", type=int, default=3, help="cyclic shifts (max 3)")
ap.add_argument("--random-shuffles", type=int, default=2)
ap.add_argument("--signal-perms", type=int, default=4, help="shuffled orders for the order-likelihood signal")
ap.add_argument("--skip-signals", action="store_true")
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--load-4bit", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

items = read_jsonl(f"{args.data}/mmlu.jsonl")
if args.limit:
    items = items[: args.limit]
slug = args.model.split("/")[-1] + (f"_dbg{args.limit}" if args.limit else "")
run_dir = Path("results") / slug
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config_mmlu.json").write_text(json.dumps(vars(args), indent=2))

lm = LM(args.model, load_4bit=args.load_4bit)
conds = (["orig"] + [f"shift{i}" for i in range(1, min(args.shifts, 3) + 1)]
         + [f"rand{i}" for i in range(1, args.random_shuffles + 1)])

# ----------------------------------------------------------------------- accuracy under reordering
for cond in conds:
    path = run_dir / f"raw_mmlu_{cond}.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    jobs = []
    for it in todo:
        perm = make_perm(cond, it["id"], args.seed)
        ch, gold = apply_perm(it["choices"], it["answer"], perm)
        jobs.append((it, perm, gold, mmlu_prompt(it["subject"], it["question"], ch)))
    jobs.sort(key=lambda j: len(j[3]))                      # similar lengths per batch
    for i in tqdm(range(0, len(jobs), args.batch_size), desc=f"mmlu [{cond}]"):
        batch = jobs[i : i + args.batch_size]
        lps = lm.choice_logprobs([b[3] for b in batch])
        rows = []
        for (it, perm, gold, _), lp in zip(batch, lps):
            pred = max(range(4), key=lambda k: lp[k])
            rows.append({"id": it["id"], "subject": it["subject"], "cond": cond, "perm": perm,
                         "gold": gold, "pred": pred, "pred_orig": perm[pred],
                         "gold_orig": it["answer"], "logp": lp})
        write_jsonl(path, rows, mode="a")

# ----------------------------------------------------------------------- contamination signals
if not args.skip_signals:
    path = run_dir / "signals_mmlu.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    buf = []
    for it in tqdm([x for x in items if x["id"] not in done], desc="signals"):
        texts = [mmlu_body(it["question"], it["choices"])]
        for p in range(1, args.signal_perms + 1):
            perm = make_perm(f"sig{p}", it["id"], args.seed)
            ch, _ = apply_perm(it["choices"], it["answer"], perm)
            texts.append(mmlu_body(it["question"], ch))
        lps = lm.token_logprobs(texts)
        totals = [sum(x) for x in lps]
        buf.append({"id": it["id"],
                    # >0 means the model finds the CANONICAL option order more likely than shuffles
                    "order_gap": totals[0] - sum(totals[1:]) / (len(totals) - 1),
                    "min_k": min_k_prob(lps[0], 0.2),
                    "mean_lp": sum(lps[0]) / max(1, len(lps[0]))})
        if len(buf) >= 25:
            write_jsonl(path, buf, mode="a"); buf = []
    if buf:
        write_jsonl(path, buf, mode="a")
print(f"Done. Results in {run_dir}/")
