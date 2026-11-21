"""Step 2b: GSM8K under answer-preserving perturbations (use an INSTRUCT model).

    python run_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 20     # debug
    python run_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct

Conditions: orig | rename (swap first names) | distractor (insert an irrelevant sentence).
Signal: rename_gap = mean token log-prob of the ORIGINAL text minus that of the RENAMED text.
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from robust.data import read_jsonl, write_jsonl
from robust.model import LM, min_k_prob
from robust.perturb import gsm_prompt_text, gsm_variant

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="data")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--conditions", nargs="+", default=["orig", "rename", "distractor"])
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--skip-signals", action="store_true")
ap.add_argument("--load-4bit", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

items = read_jsonl(f"{args.data}/gsm8k.jsonl")
if args.limit:
    items = items[: args.limit]
slug = args.model.split("/")[-1] + (f"_dbg{args.limit}" if args.limit else "")
run_dir = Path("results") / slug
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config_gsm8k.json").write_text(json.dumps(vars(args), indent=2))
lm = LM(args.model, load_4bit=args.load_4bit)

for cond in args.conditions:
    path = run_dir / f"raw_gsm8k_{cond}.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    jobs = []
    for it in items:
        if it["id"] in done:
            continue
        text, changed = gsm_variant(cond, it["id"], it["question"], args.seed)
        jobs.append((it, changed, lm.chat(gsm_prompt_text(text))))
    jobs.sort(key=lambda j: len(j[2]))
    for i in tqdm(range(0, len(jobs), args.batch_size), desc=f"gsm8k [{cond}]"):
        batch = jobs[i : i + args.batch_size]
        outs = lm.generate([b[2] for b in batch], args.max_new_tokens)
        write_jsonl(path, [{"id": it["id"], "cond": cond, "raw": o, "gold": it["gold"], "changed": ch}
                           for (it, ch, _), o in zip(batch, outs)], mode="a")

if not args.skip_signals:
    path = run_dir / "signals_gsm8k.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    buf = []
    for it in tqdm([x for x in items if x["id"] not in done], desc="signals"):
        renamed, changed = gsm_variant("rename", it["id"], it["question"], args.seed)
        if not changed:
            continue
        lp_o, lp_r = lm.token_logprobs([it["question"], renamed])
        buf.append({"id": it["id"], "min_k": min_k_prob(lp_o, 0.2),
                    "rename_gap": sum(lp_o) / len(lp_o) - sum(lp_r) / len(lp_r)})
        if len(buf) >= 25:
            write_jsonl(path, buf, mode="a"); buf = []
    if buf:
        write_jsonl(path, buf, mode="a")
print(f"Done. Results in {run_dir}/")
