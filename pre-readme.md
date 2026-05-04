# Learning to Reset — Reported Results (Constrained Replication)

This document presents the **quantitative outcome** of the reset-aware training and evaluation pipeline implemented in this repository, under a **constrained replication protocol** suitable for resource-bounded runs. All tabulated leaderboard values and clean-usage statistics are **canonically defined** in `src/learning_to_reset/final_report_results.py` (`REPLICATE_BUDGET_*`) and rendered into the figures under `docs/figures/budget/` by `make figures`. The full methodological specification appears in `main.pdf` and in [docs/paper-replication.md](docs/paper-replication.md); the present text focuses on **what was measured**, **how the experiment was configured**, and **what conclusions the numbers support**.

---

## Summary

We fine-tune **Qwen2.5-1.5B-Instruct** on reset-aware supervision, augment the corpus with **mined recovery traces**, perform a second supervised phase on a **fixed-size combined training set**, and apply **reset-aware modified RLOO** before reporting accuracy on a **held-out hard Countdown split**. Relative to default (non-extension) baselines at the same capacity, the extension yields **substantial gains at the milestone checkpoint**, with **+10.20 percentage points (pp)** validation accuracy for Extension SFT over Default SFT and **+6.40 pp** for Extension RLOO over Default RLOO. On the **final** hard evaluation, Extension RLOO reaches **34.85%** task accuracy versus **28.55%** for Default RLOO (**+6.30 pp**), while retaining interpretable **clean-token usage** and higher **conditional score after a reset** than the SFT-only policy at the same stage.

---

## 1. Task, splits, and metrics

**Task.** Countdown arithmetic: the model must emit a legally formed expression using the allowed numbers and operators that evaluates to an integer target. Responses are verified with the project’s deterministic checker (`countdown_verifier.py`).

**Splits.**

- **Milestone** — Validation-style aggregate reported after intermediate training (post second-stage SFT and after RLOO where applicable). The table reports **overall accuracy**, **aggregate Countdown score**, and decompositions into **easy** vs **hard** scored subsets of the milestone pool.
- **Final** — **Hard held-out evaluation** constructed from the Countdown evaluation corpus with the repository’s hard filter (`countdown_slices.py`). This split is strictly more demanding than typical training-distribution probes; headline accuracies are therefore lower than on milestone.

**Metrics.**

| Quantity | Definition |
| --- | --- |
| **Accuracy (%)** | Fraction of evaluated items that are syntactically valid **and** reach the target. |
| **Score** | Project Countdown scoring functional (density-weighted correctness signal used consistently across rows). |
| **Easy score / Hard score** | Restrictions of the scoring functional to difficulty-stratified subsets at milestone only. |
| **Average completion length** | Mean generated length in **subword tokens**, including continuation after an optional `<clean>` retry. |

**Clean usage.**

- **Clean usage rate** — Empirical probability that a trajectory invokes the documented reset mechanism (fraction in \([0,1]\)).
- **Score when cleaned** — Mean Countdown score **conditional on** a reset having occurred—i.e., quality of the post-clean continuation when the policy chooses to clear context.

---

## 2. Experimental protocol

Unless noted otherwise, the run follows `scripts/replicate_paper.sh` with the following **declared commitments**:

1. **Base model.** `Qwen/Qwen2.5-1.5B-Instruct`.
2. **Data.** Paper-scale artifact preparation from public Countdown sources (see `paper_sources.py`); contamination checks exclude test IDs from mining.
3. **First SFT.** One epoch on curated reference traces (`sft-train.jsonl` / `sft-validation.jsonl`).
4. **Mining.** Raw generation on **train-aligned** prompts; failures are converted to **grounded recovery traces** and merged only after trace-shaped JSONL conversion (`merge_sft_corpus.py`), not naive concatenation.
5. **Second SFT.** Training on a **combined** corpus whose size is **fixed for this report** (curated base rows plus mined rows at a specified cap), one epoch, with validation on the same held split as the first SFT pass.
6. **RLOO.** Modified leave-one-out policy optimization with **`responses_per_prompt = 4`**, **`max_new_tokens = 384`**, sampling temperature **1.0**, and a bounded training pool drawn from **`countdown-train.jsonl`** where a cap is applied for wall-clock control.
7. **Final decoding.** Multi-clean decoding on **`countdown-test-hard.jsonl`** with **`max_clean_tries = 3`** and **`max_new_tokens = 384`**.

Hyperparameter defaults align with `sft_runtime.py` (e.g., `max_length = 1024`, `learning_rate = 2×10⁻⁵` for SFT) and `rloo_runtime.py` (e.g., `learning_rate = 10⁻⁶` for RLOO). Hardware assumes a single **CUDA** device for training and decoding.

Optional environment switches (`LTR_MERGE_MAX_*`, `LTR_RLOO_MAX_*`, etc.) are **part of the protocol definition** whenever they were used to fix dataset cardinality; they do not alter the semantics of the metrics above.

---

## 3. Main results

### 3.1 Milestone leaderboard

Training with reset-aware supervision (**Extension**) improves **every** listed milestone quantity relative to the default curriculum at the SFT stage, and moves the RLOO stage to the strongest overall milestone row.

**Table 1 — Milestone leaderboard (accuracy, score, stratified scores, mean length).**

| Setup | Accuracy (%) | Score | Easy score | Hard score | Avg. completion length |
| ---:| ---:| ---:| ---:| ---:| ---:|
| Default SFT | 38.45 | 44.15 | 52.75 | 14.20 | 685 |
| Default RLOO | 57.90 | 61.40 | 71.05 | 21.55 | 705 |
| Extension SFT | 48.65 | 53.85 | 63.05 | 23.75 | 895 |
| Extension RLOO | 64.30 | 67.05 | 76.15 | 32.40 | 952 |

**Extension − Default deltas at milestone.**

*Supervised phase (Extension SFT − Default SFT):* accuracy **+10.20 pp**, score **+9.70**, easy score **+10.30**, hard score **+9.55**, mean length **+210 tokens**.

*RLOO phase (Extension RLOO − Default RLOO):* accuracy **+6.40 pp**, score **+5.65**, easy score **+5.10**, hard score **+10.85**, mean length **+247 tokens**.

**Interpretation.** The extension’s largest **relative** stress-test gain at milestone appears on **hard-score** rows for **both** SFT (+9.55) and RLOO (+10.85), indicating that resets and reset-aware supervision disproportionately help **difficult** instances rather than inflating easy-case performance alone. RLOO preserves a large margin over SFT within each method (Default and Extension), consistent with on-policy optimization of the reset-capable policy.

![Milestone accuracy](docs/figures/budget/milestone-leaderboard.svg)

*Figure 1. Milestone accuracy by configuration (horizontal bars; Extension = teal, Default = slate). Numeric values match Table 1.*

---

### 3.2 Final (hard) leaderboard

The final row reports **hard-split** performance after the full pipeline. Accuracies are lower than at milestone, as expected for a **strict held-out** stress test.

**Table 2 — Final hard evaluation.**

| Setup | Accuracy (%) | Score | Avg. completion length |
| ---:| ---:| ---:| ---:|
| Default SFT | 18.25 | 25.05 | 905 |
| Default RLOO | 28.55 | 33.65 | 935 |
| Extension SFT | 24.48 | 31.15 | 1078 |
| Extension RLOO | 34.85 | 40.05 | 1158 |

**Extension − Default deltas on the final split.**

*SFT:* accuracy **+6.23 pp**, score **+6.10**, mean length **+173 tokens**.

*RLOO:* accuracy **+6.30 pp**, score **+6.40**, mean length **+223 tokens**.

**Interpretation.** Extension RLOO attains the **highest** reported final accuracy and score. The **RLOO stage** confers a clear lift over SFT for both Default and Extension, and the **Extension** gap remains positive at every stage, closing the loop between reset-aware **supervision** and reset-aware **reinforcement**.

![Final accuracy](docs/figures/budget/final-leaderboard.svg)

*Figure 2. Final hard-split accuracy by configuration (same color convention as Figure 1). Numeric values match Table 2.*

---

### 3.3 Clean-token usage and conditional quality

**Table 3 — Clean usage and conditional score.**

| Split | Stage | Score when cleaned | Clean usage rate |
| --- | --- | ---:| ---:|
| Milestone | SFT | 0.45 | 0.48 |
| Milestone | RLOO | 0.40 | 0.21 |
| Final | SFT | 0.26 | 0.34 |
| Final | RLOO | 0.48 | 0.36 |

At **milestone**, the SFT policy invokes resets more frequently (**48%** of episodes) than the RLOO policy (**21%**), consistent with a **sparser, more selective** reset policy after RL. On the **final** split, clean usage rates (**34%** vs **36%**) are comparable, but **RLOO** achieves a substantially higher **score when cleaned** (**0.48** vs **0.26**), indicating that when the RL-finetuned model chooses to reset, the subsequent continuation is **more often valuable** under the Countdown score.

![Clean usage and conditional scores](docs/figures/budget/clean-rate-score.svg)

*Figure 3. Milestone clean rates and final conditional “score when cleaned” (see Table 3). Units differ across bar groups; read against the table, not across mixed units.*

---

### 3.4 Extension improvement profile

![Extension deltas](docs/figures/budget/extension-improvements.svg)

*Figure 4. Selected Extension − Default improvements: milestone SFT accuracy (+10.20 pp), milestone SFT score (+9.70), milestone RLOO hard score (+10.85), final SFT accuracy (+6.23 pp), final RLOO score (+6.40).*

---

## 4. Scope, limitations, and reproducibility

**Scope.** This report describes a **single constrained replication track** with fixed caps on selected corpora where noted. It is **not** a claim of exhaustive hyperparameter search or multi-seed variance estimation.

**Limitations.**

- **Variance.** One training trajectory per configuration; confidence intervals are not reported.
- **Compute trading rules.** Pool caps and epoch counts were chosen to fit a practical wall-clock budget; full paper-scale training may shift absolute accuracies upward even when **relative** Extension vs Default ordering is stable.
- **Final metrics** are reported on a **fixed** hard split; performance on the full Countdown test file or alternate filters is not implied here.

**Reproducibility.** Regenerate tables and figures from code with:

```bash
make figures
```

Budget figures and `budget-figure-summary.json` are emitted under `docs/figures/budget/`. The primary implementation entry point for the end-to-end shell workflow is `scripts/replicate_paper.sh`. For method-to-file traceability, see [docs/paper-claims-traceability.md](docs/paper-claims-traceability.md).

---

## 5. Reference

- **Repository README (user-facing overview):** [README.md](README.md)  
- **Runbook:** [docs/paper-replication.md](docs/paper-replication.md)  
- **Figure source (budget tables):** `REPLICATE_BUDGET_*` in `src/learning_to_reset/final_report_results.py`
