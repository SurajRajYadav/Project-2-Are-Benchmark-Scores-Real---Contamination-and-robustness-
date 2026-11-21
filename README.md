# Project 2: Are Benchmark Scores Real? Robustness and Contamination Probing

**Research question.** How much of a model's benchmark score survives when the test items are changed in
ways that should NOT change the correct answer, and is the lost accuracy larger on items the model
seems to have "seen" before?

**Hypotheses (write them down before running).**
- H1: Accuracy drops when MMLU options are reordered, and the drop is bigger than the noise from
  reshuffling alone.
- H2: The drop is concentrated in items with high contamination signals (the model finds the original
  text or option order unusually likely).
- H3: GSM8K accuracy falls under renaming and an irrelevant sentence; brittle models fall more.
- H4 (null / control): a near-chance model shows no meaningful gap.

> **Honesty rule.** These signals are *evidence*, never proof, of contamination. A gap can also come from
> position bias, genuine fragility, or the perturbation making an item harder. State this on your poster.

---

## 1. Design

### MMLU (4-option multiple choice, scored by next-token log-probs over A/B/C/D, no generation)
| Condition | What changes | Why |
|-----------|--------------|-----|
| `orig` | nothing | baseline |
| `shift1..3` | options cyclically rotated | each option visits every position, which balances out **position bias** |
| `rand1..2` | random reorderings | shows the gap is not specific to rotations |

Items with options like "All of the above" or "Both A and B" are **removed** because they stop being
answer-preserving when reordered (`robust/data.py`).

**Robustness gap** = accuracy(orig) − mean accuracy(reordered), computed **per item** and bootstrapped
(paired), so the confidence interval reflects the same items.

**Controls built in:** (a) cyclic shifts vs position bias; (b) two random shuffles to see variance;
(c) a position-bias table (letter preference, accuracy by gold position); (d) near-chance models as a
negative control; (e) consistency: how often the model picks the same *original* option after reordering.

### Contamination signals (per item, no external corpus needed)
1. **Order-likelihood gap** (idea from "Proving Test Set Contamination in Black Box Language Models",
   Oren et al.): compare the log-likelihood of the item text in its *canonical* option order with the
   average over shuffled orders. If the model strongly prefers the canonical order, it may have seen the
   benchmark file. Positive = canonical order preferred.
2. **Min-K% Prob** (Shi et al.): average of the lowest 20% token log-probs of the item text. Text seen in
   training tends to have fewer very-unlikely tokens, so a HIGHER (less negative) value is a weak hint of
   membership.
3. **Rename log-prob gap** (GSM8K): per-token log-prob of the original question minus that of the renamed
   version. Memorized wording tends to be much more likely than the same question with new names.

Analysis: Spearman correlation of each signal with the per-item "original advantage"
(correct on original minus correct on perturbed), plus a quartile plot (does Q4 show a bigger drop?).

### GSM8K (use an **Instruct** model; greedy generation)
| Condition | Change |
|-----------|--------|
| `orig` | none |
| `rename` | swap first names consistently (only items that contain a name from the list) |
| `distractor` | insert one irrelevant sentence with numbers before the final question |

The answer is unchanged in both. *Distractors also make items genuinely harder*, so that gap measures
brittleness, not just contamination; the rename gap is the cleaner contamination probe. Changing the
**numbers** is deliberately not included: it changes the answer and needs templated problems
(see GSM-Symbolic in the exercises).

---

## 2. Models to run

| Purpose | Models (Colab free T4) |
|---------|------------------------|
| MMLU, size trend | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B` (base models; add `Qwen/Qwen2.5-7B` with `--load-4bit`) |
| Negative control (near chance, trained before/without MMLU style data) | `EleutherAI/pythia-410m` or `pythia-1.4b` |
| GSM8K | `Qwen/Qwen2.5-0.5B-Instruct`, `1.5B-Instruct`, `3B-Instruct` |

Use **base** models for the contamination signals (instruction tuning changes likelihoods). Run the same
experiment on every model so `compare_models.py` can plot the gap against model size.

---

## 3. Setup on Google Colab

1. Download `robust_study.zip` and upload it to the Colab file panel (or to the root of My Drive).
2. Open `colab_runner.ipynb` in Colab, choose a **T4 GPU** runtime, run cells top to bottom.
3. Do **not** run `pip install -r requirements.txt` on Colab; the notebook installs only what is missing.
4. `data/` and `results/` are stored on Drive, so a disconnect loses nothing: re-run the setup cell and
   the same experiment command; it resumes.

Local machine: `pip install -r requirements.txt`.

### Commands
```bash
python smoke_test.py                              # unit checks + fake results, no GPU (about 30 s)
python prepare_data.py --mmlu-n 2000 --gsm-n 500

# debug first (about 2 minutes)
python run_mmlu.py  --model Qwen/Qwen2.5-0.5B --limit 50
python run_gsm8k.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 20

# full runs (about 15-30 min per MMLU model, 25-45 min per GSM8K model on a T4 for a 1.5B model)
python run_mmlu.py  --model Qwen/Qwen2.5-1.5B
python run_gsm8k.py --model Qwen/Qwen2.5-1.5B-Instruct

python analyze.py --run results/Qwen2.5-1.5B
python analyze.py --run results/Qwen2.5-1.5B-Instruct
python compare_models.py                           # after you have several models
```
Out of memory? Lower `--batch-size` (MMLU 8 -> 4; GSM8K 16 -> 8).

---

## 4. Output files (`results/<model>/`)
| File | Meaning |
|------|---------|
| `raw_mmlu_<cond>.jsonl`, `raw_gsm8k_<cond>.jsonl` | every prediction (never delete) |
| `signals_mmlu.jsonl`, `signals_gsm8k.jsonl` | per-item contamination signals |
| `table_*.csv`, `summary_*.json` | tables for your poster; summaries used by `compare_models.py` |
| `figures/fig_mmlu.*`, `fig_gsm8k.*` | accuracy by condition + signal quartile plots |
| `results/_comparison/` | cross-model gap plot and tables |

---

## 5. How to read the results

1. **Accuracy table with CIs.** Is `orig` clearly above the reordered conditions (non-overlapping CIs on
   the paired gap)? Report the gap in points *and* as a relative drop.
2. **Position bias table.** If the model strongly prefers one letter, shuffles hurt for reasons unrelated
   to memorization. Cyclic shifts mostly cancel this, which is why they are the headline gap.
3. **Consistency.** A model that truly knows an answer picks the same *content* after reordering.
4. **Signal correlations and quartiles.** Positive rho with a rising bar chart supports H2. A flat chart
   means "no evidence of contamination-linked drop"; report that honestly.
5. **Across models.** Does the gap grow with size or accuracy? Does the near-chance control sit near zero?
   (A near-chance model has nothing to lose, so ignore its gap.)

Realistic expectation: with n=2000 the paired CI on the MMLU gap is roughly ±2 points, so differences
under about 2 points are not resolvable; use the full sample (increase `--mmlu-n`) if you need finer
resolution.

---

## 6. Pitfalls and limitations to state
- **Correlation, not causation.** Signals are weak, noisy detectors; AUROC against known contaminated data
  was not measured here (a good extension).
- **Position bias** can mimic contamination; the controls above reduce, but do not remove, this.
- **Perturbation quality.** Renaming ignores pronoun/gender consistency; distractors add difficulty;
  spot-check 30 perturbed items by hand and report any problems.
- **MMLU itself has label noise** and some ambiguous items; a "robust" model can still be wrong.
- **Prompt format** (0-shot, letter scoring) affects absolute numbers; compare models only under the same
  setup.
- **One benchmark family.** Say so; do not claim "LLM benchmarks are broken" from two benchmarks.

---

## 7. Exercises (pick 2-3)
1. **Paraphrase probe.** Paraphrase questions with an instruct model, verify meaning on a sample, and compare
   accuracy and log-prob on original vs paraphrase.
2. **Signal validation.** Create a known-contaminated setting (fine-tune a small model for a few steps on part
   of the benchmark) and measure how well each signal separates seen vs unseen items (AUROC).
3. **N-gram overlap.** Compute n-gram overlap of items against a public corpus sample (for example a slice of
   a web or Wikipedia dump) and correlate it with the gap.
4. **Subject-level analysis.** Which MMLU subjects show the biggest gaps? (Use all items; per-subject CIs.)
5. **GSM-Symbolic style numeric variants.** Hand-write 20 problem templates and sample new numbers with
   recomputed answers.
6. **Few-shot vs zero-shot.** Does adding 5 examples change the gap?
7. **Scale and family.** Add Llama, Gemma, or SmolLM and see whether the gap tracks size or family.

---

## 8. Mapping to your poster
| Zone | Content |
|------|---------|
| Question | "How much of a benchmark score survives answer-preserving changes?" |
| Method | diagram: item -> perturbations -> accuracy gap + contamination signals |
| Result 1 | `fig_mmlu` left panel + gap with CI |
| Result 2 | signal quartile plots (headline: "items in the top signal quartile lose X points more") |
| Result 3 | `fig_compare_models` (gap vs model size, with control) |
| Limitations | Section 6 |

---

## 9. Layout
```
robust_study/
  prepare_data.py   run_mmlu.py   run_gsm8k.py   analyze.py   compare_models.py   smoke_test.py
  colab_runner.ipynb
  robust/  data.py  perturb.py  model.py  metrics.py
```
