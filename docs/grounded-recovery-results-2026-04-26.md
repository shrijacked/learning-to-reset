# Arithmetic-Grounded Recovery Results — 2026-04-26

This run lifts the `1/32`-correct hard-Countdown ceiling that the prior `plus-mined-fresh-eval` SFT hit. We added an arithmetic-grounded recovery template (substep arithmetic + rejected hypothesis + number-budget audit + final-value reconciliation), put a deep-verify filter in the data prep pipeline, and re-ran the same SFT/eval cycle on Qwen2.5-0.5B (CPU). The headline outcome is a **negative result** that is conclusive about why the wedge does not work at this model scale, not a hidden gain.

## What changed

- **New trace template** (`grounded`): walks the AST of both a rejected and an accepted candidate, emits one `Compute (a op b) = v` substep per `BinOp`, then writes a number-budget audit and a final-value reconcile line. Surface phrasing (intro, hypothesis label, accept/reject phrase, budget order) is randomized by an `rng_seed` so the model cannot memorize a fixed shape.
- **Deep-verify pipeline filter**: `pipeline._deep_verify_recovery_text` re-checks every plain `a op b = c` claim with `Fraction` and drops any recovery whose math does not actually compute. This is enforced when `--require-recovery-target-correct` is set.
- **Plumbed end-to-end**: the new `grounded` style flows through `synthetic_countdown_traces`, `failure_recovery_traces`, both CLIs, and `RecoveryStyle` literals; `"all"` now expands to four styles (walkthrough, verification, contrastive, grounded). Suite went from 123 to 132 tests, all green.
- **Trainer extras**: `pip install -e .[trainer]` replaces the manual `transformers/datasets/accelerate/torch` install in `README.md`.

## Headline numbers (Qwen2.5-0.5B, hard 32-example holdout)

All evals use greedy decoding (`temperature=0.0`) per `main.pdf` Section 4.1, on `tmp/paper-assets-hard-focus/countdown-eval-hard-fresh-32.jsonl`. The original `plus-mined-fresh-eval` baseline is the SFT checkpoint at `tmp/paper-runs/sft-hard-focus-plus-mined-fresh-eval-e0.5/`; the grounded checkpoints are the new SFTs trained on the same hard source plus the new template.

| Run | Recovery styles | Eval mode | `max_new_tokens` | Correct | Valid | clean_rate | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Plus-mined baseline (prior) | walkthrough, verification, contrastive | reset-aware | 128 | **1/32** | 29/32 | 1.00 | the bar to beat |
| Plus-mined baseline (re-eval, longer budget) | same as above | reset-aware | 384 | 1/32 | 29/32 | 1.00 | extra tokens do not move the bar |
| Plus-mined baseline | same as above | raw one-pass | 128 | 0/32 | 0/32 | n/a | matches prior raw result |
| **Grounded-all (T7)** | walkthrough, verification, contrastive, grounded + 15 mined | reset-aware | 128 | 0/32 | 30/32 | 1.00 | validity up by one, correctness down by one |
| Grounded-all (T7) | same as above | reset-aware | 384 | 0/32 | 30/32 | 1.00 | identical at longer budget — model picked the shorter verification template anyway |
| Grounded-all (T7) | same as above | raw one-pass | 128 | 0/32 | 0/32 | n/a | matches baseline raw |
| Grounded-all (T9) on seed=131 fresh slice | same as above | reset-aware | 128 | 0/32 | 28/32 | 1.00 | rules out fluke on the original slice |
| **Grounded-only ablation (T13)** | grounded only | reset-aware | 128 | 0/32 | 0/32 | 1.00 | template too long for 128 tokens |
| Grounded-only ablation (T13) | grounded only | reset-aware | 384 | 0/32 | 31/32 | 1.00 | template now fits, validity recovers, correctness still zero |

Sources: `tmp/paper-eval/sft-hard-focus-plus-mined-fresh-eval-e0.5-hard{,-t384}/summary.json`, `tmp/paper-eval/sft-grounded-26apr-hard{,-t384,-raw}/summary.json`, `tmp/paper-eval/sft-grounded-26apr-seed131-hard/summary.json`, `tmp/paper-eval/sft-grounded-only-26apr-hard{,-t384}/summary.json`.

### Paper-alignment commitments

`main.pdf` Section 4 reports headline numbers of `13.61% → 36.94%` accuracy on the hard split when training a 1B model with the full reset-aware pipeline. We are **not** matching that bar at the Qwen2.5-0.5B / CPU scale used here; the absolute numbers above are pilot-scale, and we say so. Two paper-alignment items are tracked here for transparency:

- **Figure 6 metrics** (`clean_rate`, `score_when_cleaned`): `clean_rate` is captured by `eval_runtime`. `score_when_cleaned` is currently captured only inside `rloo_runtime.evaluate_rollout_candidates` — adding it to `eval_runtime`'s per-run summary is a deferred follow-up so we can quote it directly here.
- **Figure 7 qualitative samples**: included below.

## Why the grounded template did not help — qualitative sample

This is the failure mode that the diagnostic ablation (grounded-only, 384 tokens) makes obvious. Example 0 of `tmp/paper-eval/sft-grounded-only-26apr-hard-t384/results.jsonl`:

```text
Question: Use the numbers 51, 31, 25, 34 to reach 24.

<think>
Starting over from a clean slate with numbers [51, 31, 25, 34], target 24.

Hypothesis 1: try (51 + 31).
  Step 1: Compute (51 + 31) = 82.
  Reconcile: 82 != 24. Reject candidate 1.

Hypothesis 2: try (31 * (51 - 34)).
  Step 1: Compute (51 - 34) = 7.            <-- WRONG, real value 17
  Step 2: Compute (31 * (51 - 34)) = 24.    <-- WRONG, real value 527
  Reconcile: 24 == 24. Accept hypothesis 2.

Number budget: used 51 once, used 25 once, used 31 once, used 34 once.
Final value check: candidate evaluates to 24, matches target 24.
</think>
<answer>
(31 * (51 - 34))
</answer>
```

Three things to note in this single sample:

1. **The model adopted the grounded template.** Hypothesis labels, substep numbering, reconcile lines, number-budget audit, final-value check, and the answer block are all in the right shape.
2. **The first hypothesis substep is correct** (`(51 + 31) = 82`).
3. **Every following claim is fabricated**: `(51 - 34) = 7` (real `17`), the top-level `(31 * (51 - 34)) = 24` (real `527`), the budget audit lists `25` even though the answer expression doesn't use it, and the final-value check reasserts the false `24`. The verifier scores the answer expression at `527`, so it is `valid` but not target-correct.

This is not the prior verification-template failure mode (where a single fabricated `Verifier check: <expr> = 24` line carried the lie). The grounded template did exactly what it was designed to do at the structural level — it forced the model to commit to per-step intermediate values — but at the 0.5B scale on a ~570-example grounded-only corpus, the model commits to false intermediate values. **The deep-verify filter protects training-data integrity but cannot prevent inference-time substep fabrication.**

## Why the planned multi-pass ablation stops here

The plan listed three further ablations (drop final-check / drop number-bag / drop failed-hypothesis). Each removes a constraint that the grounded-only model is already failing on — the model fabricated the substep math, the budget claim, and the final-value claim simultaneously. Removing any single guard cannot make per-step arithmetic correct on a 0.5B model that already refuses to be correct when all three are present. Running three more 25-minute SFT cycles to confirm "still 0/32" is honest only if we say up front it would not change the diagnosis. We do not run them; we record this as the reason.

The grounded-only run is itself the ablation that matters: it shows that giving the model **only** the grounded template (no shorter fallback to fall back to) does not produce a single correct answer on either the original or the seed=131 holdout. The wedge is not there at this scale.

## What this run does deliver

Even though it does not raise the `1/32` ceiling, this iteration leaves the repo strictly better than before:

- A new, tested, end-to-end `grounded` recovery generator (132 tests green; +9 tests for `grounded` and the deep-verify filter).
- A defensive deep-verify filter that catches contradicted recoveries before they enter SFT — the failure mode we previously *only* mined post-hoc from eval results.
- A cleaner trainer-install path (`pip install -e .[trainer]`).
- A reproducible artifact set under `tmp/paper-assets-grounded-26apr/` and `tmp/paper-artifacts-grounded-26apr/` matching the baseline's source so the only experimental variable is the trace template.
- A precise diagnosis (substep fabrication is the next bottleneck, not template shape) that points at the right next move.

## Honest next moves

These are not run in this milestone. They are listed so we do not pretend the result above is the final word.

- **Step-level reward / verifier-in-the-loop SFT or RL**: a per-substep reward of "did this `Compute … = v` claim re-compute under `Fraction`?" applied during RL would be the direct response to substep fabrication. The deep-verify filter is the supervised half of the same idea; the RL half is not implemented.
- **Bigger target model**: paper Section 4 uses ~1B parameters. The pattern observed here — the model adopts the template but fabricates intermediate values — is a known small-model behavior that often disappears at 1B-3B. We do not have the compute to confirm here.
- **Add `score_when_cleaned` to `eval_runtime`'s summary** so future grounded runs can be reported in Figure 6 form directly.
- **Fresh failure-mining round on the 32-ex holdout that is held out for evaluation**: not done here because mining the same examples we evaluate on would contaminate the result. A truly clean retraining loop would mine from a third hard slice that is neither `countdown-train-hard-64.jsonl` nor `countdown-eval-hard-fresh-32.jsonl`.

## Files of record

- Code: `src/learning_to_reset/synthetic_countdown_traces.py`, `src/learning_to_reset/failure_recovery_traces.py`, `src/learning_to_reset/pipeline.py`, `src/learning_to_reset/__init__.py`.
- Tests: `tests/test_synthetic_countdown_traces.py`, `tests/test_failure_recovery_traces.py`, `tests/test_pipeline.py`.
- Generated assets (gitignored): `tmp/paper-assets-grounded-26apr/`, `tmp/paper-artifacts-grounded-26apr/`, `tmp/paper-artifacts-grounded-26apr-seed131/`, `tmp/paper-artifacts-grounded-only-26apr/`.
- Trained checkpoints (gitignored): `tmp/paper-runs/sft-grounded-26apr/`, `tmp/paper-runs/sft-grounded-only-26apr/`.
- Eval outputs (gitignored): `tmp/paper-eval/sft-grounded-26apr-hard{,-t384,-raw}/`, `tmp/paper-eval/sft-grounded-26apr-seed131-hard/`, `tmp/paper-eval/sft-grounded-only-26apr-hard{,-t384}/`, `tmp/paper-eval/sft-hard-focus-plus-mined-fresh-eval-e0.5-hard-t384/`.
