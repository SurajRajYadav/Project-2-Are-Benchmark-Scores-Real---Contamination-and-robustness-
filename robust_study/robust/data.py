"""Benchmark loading (MMLU, GSM8K) and jsonl helpers."""
import json
import random
import re
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows, mode="w"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Options such as "All of the above" or "Both A and B" refer to other options, so they become
# invalid once options are reordered. We drop those items to keep perturbations answer-preserving.
_POSITION_DEPENDENT = re.compile(
    r"\b(all|none|both|neither)\b[^.]{0,20}\b(above|below|these|the following)\b"
    r"|\b[A-D]\b\s*(and|or|,)\s*\b[A-D]\b",
    re.I,
)


def is_position_dependent(choices):
    return any(_POSITION_DEPENDENT.search(c) for c in choices)


def load_mmlu(n, seed=0):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    ok = [i for i in range(len(ds)) if not is_position_dependent(ds[i]["choices"])]
    print(f"MMLU: kept {len(ok)} of {len(ds)} items (dropped position-dependent options)")
    idxs = random.Random(seed).sample(ok, min(n, len(ok)))
    items = []
    for i in idxs:
        ex = ds[i]
        items.append({"id": f"mmlu-{i}", "subject": ex["subject"], "question": ex["question"],
                      "choices": list(ex["choices"]), "answer": int(ex["answer"])})
    return items


def load_gsm8k(n, seed=0):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    items = []
    for i in idxs:
        ex = ds[i]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsm-{i}", "question": ex["question"], "gold": float(gold)})
    return items
