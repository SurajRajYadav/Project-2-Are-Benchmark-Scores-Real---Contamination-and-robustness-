"""Statistics helpers and GSM8K answer extraction."""
import re

import numpy as np
from scipy.stats import binomtest, spearmanr

_NUM = r"-?\d[\d,]*\.?\d*"


def extract_number(text):
    """Prefer the number after the last 'Answer:'; otherwise the last number in the text."""
    m = re.findall(r"answer\s*[:\-]\s*\$?\s*(" + _NUM + ")", text, flags=re.I)
    cand = m[-1] if m else (re.findall(_NUM, text) or [None])[-1]
    if cand is None:
        return None
    try:
        return float(cand.replace(",", "").rstrip("."))
    except ValueError:
        return None


def is_correct_number(pred, gold, tol=1e-6):
    return pred is not None and abs(pred - gold) < tol


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """95% bootstrap CI for the mean of `values` (resampling items)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def spearman_ci(x, y, n_boot=1000, seed=0):
    """Spearman rho with a bootstrap CI over items. Returns (rho, lo, hi, p)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    try:
        rho, p = spearmanr(x, y)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx])[0]
        if not np.isnan(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return float(rho), float(lo), float(hi), float(p)


def paired_counts(ca, cb):
    """Sign test on discordant items: wrong under a / right under b (helped) vs the reverse (hurt)."""
    ca, cb = np.asarray(ca), np.asarray(cb)
    helped = int(((ca == 0) & (cb == 1)).sum())
    hurt = int(((ca == 1) & (cb == 0)).sum())
    p = binomtest(helped, helped + hurt, 0.5).pvalue if helped + hurt else 1.0
    return {"both_right": int(((ca == 1) & (cb == 1)).sum()), "both_wrong": int(((ca == 0) & (cb == 0)).sum()),
            "b_right_a_wrong": helped, "a_right_b_wrong": hurt, "sign_test_p": float(p)}
