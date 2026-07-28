# Pilot findings — knowledge vs reasoning via same-model debate

**Date:** 2026-07-28 · **Data:** `New/baseline_v2_{mmlu-pro,gpqa}_s{7,17,42}.xlsx` (same-model Qwen 3-agent debate, 5 rounds, pooled 3 seeds). Script: `pilot_analysis.py`. No new generation, no interventions — proxy labels only.

**Verdict: GO.** The mechanism predicted in `RESEARCH_IDEA.md` is present and strong in both datasets.

## Headline result — debate recovers reasoning failures, not knowledge failures

Among **initially-wrong** questions, recovery rate (wrong → correct by final round), refined by whether the correct answer was latent in any of the 3 independent R1 answers:

| Group (initially wrong) | MMLU-Pro | GPQA | Interpretation |
|---|---|---|---|
| Unanimous-wrong (correct **absent**) | 13.2% (n=205) | 1.5% (n=135) | knowledge gap — shared blindspot |
| Disagree, correct **absent** from pool | 16.9% (n=71) | 3.0% (n=33) | knowledge gap — disagreement alone doesn't help |
| Disagree, correct **present** (minority) | **62.7%** (n=51) | **52.9%** (n=17) | reasoning/aggregation — debate promotes it |

**The binding variable is not disagreement — it is whether some clone already reached the correct answer.**
- Correct **absent** → recovery near floor (13–17% MMLU, 1.5–3% GPQA) regardless of agreement. Debate cannot manufacture absent knowledge.
- Correct **present as minority** → recovery >50%. The knowledge exists in the collective; debate's job is to let the correct minority win. Debate genuinely repairs these.

Coarser unanimous-vs-disagreement cut is also highly significant (Fisher exact): MMLU-Pro OR=3.72, p=1.9e-6; GPQA OR=16.6, p=4.2e-5.

## Why this beats single-shot confidence

Confidence **cannot** tell the two failure modes apart, but debate dynamics **can**:

| feature (initially-wrong) | unanim-wrong (knowledge) | disagree-wrong (reasoning) | MWU p |
|---|---|---|---|
| mean initial confidence | 0.799 / 0.827 | 0.765 / 0.809 | 0.30 / 0.06 (n.s.) |
| answer switches | 0.78 / 0.33 | 4.72 / 5.04 | 1e-39 / 3e-32 |
| oscillation (A→B→A) | 0.18 / 0.13 | 1.96 / 2.48 | 9e-22 / 5e-21 |
| rounds to consensus τ | 1.00 / 1.00 | 3.40 / 3.60 | 2e-68 / 1e-40 |

*(MMLU-Pro / GPQA)*. Confidence is statistically indistinguishable between the two failure modes; the process metrics (switches, oscillation, consensus time) separate them at p≈1e-20 to 1e-68. **This is the core value-add of debate over single-shot estimation.**

## Debate accuracy gain concentrates where it should

| | single-shot (1 agent) | self-consistency (R1 vote) | debate-final | gain vs single-shot |
|---|---|---|---|---|
| MMLU-Pro | 0.441 | 0.455 | 0.547 | **+0.106** |
| GPQA | 0.382 | 0.383 | 0.407 | +0.024 |

The gain is larger on MMLU-Pro, which has more "correct-present-minority" (reasoning-recoverable) mass. GPQA is dominated by correct-absent (knowledge-gap) failures, so debate has little to work with — consistent with the thesis.

## Caveat / double-edge

Disagreement is not free: among **initially-correct** questions, the disagreement group is corrupted (correct → wrong) 20–22% of the time vs ~1% for unanimous-correct. Debate moves things; sometimes the wrong way (bandwagon). Net effect is still positive.

## What this sets up

1. **Operational proxy labels validated:** correct-absent-from-pool ≈ knowledge-limited; correct-present-minority ≈ reasoning-limited. These give a cheap ground-truth to build a classifier before running interventions.
2. **Next (needs generation):** the double-intervention 2×2 (+gold-facts vs +reasoning-scaffold) to confirm that correct-absent cases recover under open-book and correct-present cases don't need it.
3. **Classifier:** predict failure type from debate dynamics (switches, oscillation, τ, initial disagreement) vs single-shot confidence / self-report baselines; report AUROC + risk–coverage.

## Honest limitations

- Proxy labels, not intervention ground truth (yet). "Correct-absent" conflates true knowledge gaps with *systematic* reasoning errors (same slip in every clone) — both are un-recoverable by same-model debate and look identical here. The +reasoning-scaffold intervention is what will separate them.
- MCQ answer space is small, so "correct in pool" is coarser than for open-ended QA.
- Single model, single family (Qwen). Generalization across models/scales untested.
