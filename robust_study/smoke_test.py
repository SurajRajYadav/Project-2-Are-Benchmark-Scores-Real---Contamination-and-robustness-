"""Smoke test: unit-checks the perturbations, then fabricates results for 3 fake models and runs the
whole analysis (no GPU, no downloads).   python smoke_test.py"""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from robust.data import is_position_dependent, write_jsonl
from robust.metrics import extract_number
from robust.perturb import add_distractor, apply_perm, make_perm, rename_people

#  unit checks
choices, answer = ["w", "x", "y", "z"], 2          # correct text is "y"
for cond in ["orig", "shift1", "shift2", "shift3", "rand1", "rand2", "sig1"]:
    perm = make_perm(cond, "item-1")
    new_choices, new_gold = apply_perm(choices, answer, perm)
    assert sorted(perm) == [0, 1, 2, 3]
    assert new_choices[new_gold] == "y", cond       # the correct option text must follow the answer
    assert perm[new_gold] == answer                  # perm maps shown position -> original index
assert make_perm("rand1", "a") == make_perm("rand1", "a")      # deterministic
assert make_perm("sig1", "a") != [0, 1, 2, 3]
assert is_position_dependent(["All of the above", "x", "y", "z"])
assert is_position_dependent(["Both A and B", "x", "y", "z"])
assert not is_position_dependent(["Paris", "London", "Rome", "Berlin"])
assert extract_number("so 3+4=7.\nAnswer: 1,234.") == 1234.0
assert extract_number("it costs $18") == 18.0
assert extract_number("no numbers") is None
text, changed = rename_people("Janet has 3 ducks. Janet sells 2 to Mark.", random.Random(0))
assert changed and "Janet" not in text and "Mark" not in text and text.count(text.split()[0]) == 2
text, _ = add_distractor("Tom has 5 apples. He eats 2. How many are left?", random.Random(0))
assert text.endswith("How many are left?") and text.startswith("Tom has 5 apples.")
print("unit checks OK")

#  fake results
rng = np.random.default_rng(0)
FAKE = {"FakeModel-0.5B": (0.40, 0.00), "FakeModel-1.5B": (0.55, 0.10), "FakeModel-3B": (0.65, 0.18)}
for model, (base_acc, p_mem) in FAKE.items():
    run = Path("results") / model
    run.mkdir(parents=True, exist_ok=True)
    conds = ["orig", "shift1", "shift2", "shift3", "rand1", "rand2"]
    raw = {c: [] for c in conds}
    sigs = []
    for i in range(1500):
        iid, gold_orig = f"mmlu-{i}", int(rng.integers(0, 4))
        mem = rng.random() < p_mem                         # fake 'memorized' item
        for c in conds:
            perm = make_perm(c, iid)
            gold = perm.index(gold_orig)
            ok = (c == "orig" and mem) or rng.random() < (0.5 if mem else base_acc)
            pred = gold if ok else int((gold + rng.integers(1, 4)) % 4)
            raw[c].append({"id": iid, "subject": "fake", "cond": c, "perm": perm, "gold": gold, "pred": pred,
                           "pred_orig": perm[pred], "gold_orig": gold_orig, "logp": [0, 0, 0, 0]})
        sigs.append({"id": iid, "order_gap": float(rng.normal(3 if mem else 0, 2)),
                     "min_k": float(rng.normal(-3 if mem else -3.6, 0.5)), "mean_lp": -3.0})
    for c in conds:
        write_jsonl(run / f"raw_mmlu_{c}.jsonl", raw[c])
    write_jsonl(run / "signals_mmlu.jsonl", sigs)

    g, gs = {c: [] for c in ["orig", "rename", "distractor"]}, []
    for i in range(400):
        iid = f"gsm-{i}"
        mem = rng.random() < p_mem
        changed = rng.random() < 0.7
        for c in g:
            ok = rng.random() < (0.9 if (mem and c == "orig") else base_acc * (0.9 if c == "distractor" else 0.95))
            g[c].append({"id": iid, "cond": c, "raw": f"Answer: {10 if ok else 11}", "gold": 10.0,
                         "changed": changed if c == "rename" else True})
        if changed:
            gs.append({"id": iid, "min_k": float(rng.normal(-3 if mem else -3.6, 0.5)),
                       "rename_gap": float(rng.normal(0.3 if mem else 0.0, 0.2))})
    for c in g:
        write_jsonl(run / f"raw_gsm8k_{c}.jsonl", g[c])
    write_jsonl(run / "signals_gsm8k.jsonl", gs)
    subprocess.run([sys.executable, "analyze.py", "--run", str(run)], check=True)

subprocess.run([sys.executable, "compare_models.py"], check=True)
print("\nSMOKE TEST PASSED")
