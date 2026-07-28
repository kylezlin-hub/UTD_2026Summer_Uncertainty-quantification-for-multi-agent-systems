# Knowledge or Reasoning? Diagnosing the Binding Constraint of an LLM via Same-Model Debate Dynamics

**Status:** research proposal (draft) · **Date:** 2026-07-28
**Constraint:** same-model multi-agent debate only (3 instances of one model, hardware-limited — no multi-model heterogeneity).

---

## 1. Core question

A model can fail a question for two distinct reasons:

1. **Knowledge limitation** — the required fact is not in the weights.
2. **Reasoning capability limitation** — the pieces are present but the model mis-combines them.

**Can same-model multi-agent debate quantify and *separate* these two limitations, better than single-model single-shot estimation?**

Single-shot confidence collapses both failure modes into one scalar. The hypothesis is that debate *dynamics* expose which limitation is binding, because the two failure modes have different mechanistic signatures.

## 2. The mechanism (why same-model debate can separate them)

- **Knowledge limitation is systematic and shared** across all clones (same weights → same missing fact). The 3 agents tend to agree confidently on the same wrong answer, and debate cannot manufacture knowledge none of them hold. → **debate cannot repair it.**
- **Reasoning limitation is largely stochastic and idiosyncratic per sample** (temperature → different chains → different slips). Different clones make different mistakes, so they can catch each other and converge to the correct answer. → **debate can partially repair it.**

### Central testable thesis
> The accuracy gain of same-model debate over single-shot comes almost entirely from *reasoning-limited* questions; on *knowledge-limited* questions debate gain ≈ 0. Therefore the debate dynamics themselves reveal which limitation is binding.

**Honest caveat:** debate only repairs *stochastic* reasoning errors. A *systematic* reasoning slip (same error every sample) won't be caught by clones and will masquerade as a knowledge limitation. This is a real limitation — and also a potential sub-result: debate distinguishes *recoverable (stochastic)* from *unrecoverable (systematic)* errors.

## 3. Predicted debate signatures

| | **Knowledge-limited** | **Reasoning-limited** |
|---|---|---|
| Initial disagreement | low → often **unanimous wrong** | high (idiosyncratic chains) |
| Confidence | high, stable | moderate, variable |
| Debate movement | little; converges to wrong | lots, and **productive** |
| Accuracy across rounds | flat | rises |
| **Supply gold facts (open-book)** | **recovers** | no change |
| **Supply reasoning scaffold** | no change | **recovers** |
| Challenge sensitivity | brittle collapse, no recovery | wobbles, then recovers to correct |

The last two rows are the **ground-truth anchors** (Section 5).

## 4. Is it better than single-shot?

- **Separating the two failure modes: yes, structurally.** Single-shot gives one confidence number that cannot distinguish "I don't know this" from "I keep miscomputing."
- **Detecting reasoning-limited failures: probably yes** — debate self-repair is a strong tell.
- **Pure knowledge limitation: debate ≈ self-consistency ≈ single-shot confidence** — all clones share the blindspot, so debate adds little. This is expected and is what makes the *contrast* diagnostic.

**Baseline to beat:** the model's own single-shot self-report ("am I failing on knowledge or reasoning?"). Debate should beat introspection.

## 5. Ground truth — double-intervention 2×2 (methodological core)

To label which limitation binds per question:

- **+Knowledge** (open-book: inject gold facts/context). Fixes it → knowledge was the bottleneck.
- **+Reasoning** (supply a correct decomposition / CoT scaffold, or a tool for the mechanical step). Fixes it → reasoning was the bottleneck.
- Neither → out of scope; both needed → mixed.

This yields clean labels to validate the debate-based classifier.

## 6. Connection to earlier work (supporting evidence already in hand)

The earlier same-model results are evidence *for* this mechanism, not noise:

- Engagement/responsiveness were **higher for wrong answers** → reasoning-struggle (agents working, revising). Low engagement + confident wrong consensus → knowledge blindspot.
- `P(correct | round-1 consensus)` ≫ non-consensus (e.g. 0.805 vs 0.522) → immediate unanimous *wrong* agreement is the classic "unanimous wrong" knowledge signature; productive non-consensus that resolves is the reasoning signature.

The earlier "inverted" process→accuracy correlation is reinterpreted here as the metrics tracking the knowledge–reasoning axis.

## 7. Proposed headline & contributions

**Headline:** *Knowledge or Reasoning? Diagnosing the Binding Constraint of an LLM via Same-Model Debate Dynamics.*

1. A mechanistic account + evidence that same-model debate repairs stochastic reasoning failures but not shared knowledge gaps.
2. A debate-dynamics classifier that labels a failure as knowledge- vs reasoning-limited, validated against the double-intervention ground truth — beating single-shot self-report and confidence.
3. Reinterpretation of process metrics (engagement/consensus) as indicators of the knowledge–reasoning axis, resolving the earlier inverted correlation.

## 8. Experiments / next steps

1. **Pilot on existing same-model data (cheap, first).** Split wrong-answer questions via the double intervention (gold facts vs reasoning scaffold). Test whether debate signatures (initial disagreement, engagement, round-1 consensus, oscillation) already separate the two groups. Go/no-go before generating anything new.
   - Data in hand: `ColMAD/`, `New/baseline_v2_mmlu-pro_*`, `New/baseline_v2_gpqa_*`.
   - A faster proxy split: **unanimous-wrong** vs **productive-disagreement** groups from existing debate traces.
2. **Clean two-axis dataset.** Knowledge-heavy/reasoning-light set (closed-book factual QA) vs reasoning-heavy/knowledge-light set (GSM8K or a logic set with all facts given). Run same-model debate on both; compare signatures + debate-vs-single-shot accuracy gain. Predicted: gain concentrated on the reasoning set.
3. **Control for difficulty.** Match groups on base single-shot accuracy so the signal isn't just re-detecting difficulty.
4. **Classifier + baselines.** Logistic model on debate features vs single-shot confidence vs self-report; AUROC / Brier / risk–coverage on the 2×2 labels.

## 9. Risks

- Knowledge/reasoning distinction is not always clean; start with datasets that isolate the axes before tackling mixed questions.
- Difficulty confound (reasoning-limited ≈ "harder") — must match on base accuracy.
- Open-book ground truth is imperfect (retrieval quality; model may ignore supplied facts) — prefer gold-fact injection.
- Systematic reasoning errors masquerade as knowledge limits (see Section 2 caveat).

## 10. Debate protocol (working default)

3 agents, same model, identical instructions, independent sampling. Functional roles (Candidate / Critic / Verifier) or role-free ablation. Round 0 independent answers (unseen); rounds 1–3 cross-examination + revision; final round: argue against the emerging consensus (guards against imitation-driven convergence); optional challenge-sensitivity probe (valid vs plausible-but-false counterargument). Keep to 3 rounds for the main run (earlier data shows extra rounds do not help and can hurt).
