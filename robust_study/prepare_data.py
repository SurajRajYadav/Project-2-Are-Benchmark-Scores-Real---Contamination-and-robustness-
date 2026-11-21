"""Step 1: sample benchmark items.   python prepare_data.py --mmlu-n 2000 --gsm-n 500"""
import argparse

from robust.data import load_gsm8k, load_mmlu, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--mmlu-n", type=int, default=2000)
ap.add_argument("--gsm-n", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data")
args = ap.parse_args()

if args.mmlu_n > 0:
    items = load_mmlu(args.mmlu_n, args.seed)
    write_jsonl(f"{args.out}/mmlu.jsonl", items)
    print(f"Saved {len(items)} MMLU items")
if args.gsm_n > 0:
    items = load_gsm8k(args.gsm_n, args.seed)
    write_jsonl(f"{args.out}/gsm8k.jsonl", items)
    print(f"Saved {len(items)} GSM8K items")
