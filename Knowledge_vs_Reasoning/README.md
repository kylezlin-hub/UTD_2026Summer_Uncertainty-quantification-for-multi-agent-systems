# Knowledge vs Reasoning via Same-Model Debate

Can same-model multi-agent debate (3 instances of one model) quantify and **separate** a model's
two failure modes — **knowledge limitation** (fact not in the weights) vs **reasoning limitation**
(pieces present, mis-combined) — better than single-shot confidence?

This folder holds the proposal, a completed pilot (GO), and a ready-to-run intervention harness.

## Files

| File | What it is |
|---|---|
| `RESEARCH_IDEA.md` | Full proposal: mechanism, signatures, headline, contributions, risks |
| `pilot_analysis.py` | Pilot on existing `New/baseline_v2_*` data (no generation) |
| `PILOT_FINDINGS.md` | Pilot results + interpretation (**GO**) |
| `generate_interventions.py` | Double-intervention 2×2 harness (turns proxy labels into ground truth) |
| `INTERVENTIONS_README.md` | Design + caveats + run reference for the harness |
| `train_failure_classifier.py` | Predict knowledge- vs reasoning-limited from debate dynamics; compares vs single-shot confidence |
| `selfconsistency_baseline.py` | **The decisive "is debate needed?" test**: debate dynamics vs *compute-matched* independent sampling |
| `interventions/correct_absent_subset.csv` | 132 correct-absent questions selected for intervention |
| `interventions/classifier_*.{csv,json,png}` | Classifier features, AUROC report, risk–coverage (proxy-label run so far) |
| `interventions/debate_vs_sampling_report.json` | Head-to-head AUROC + paired-bootstrap verdict |

## Current status (2026-07-28)

- **Pilot: done, GO.** On same-model Qwen debates (`baseline_v2` GPQA + MMLU-Pro, 3 seeds each), the
  mechanism holds. Among initially-wrong questions, recovery (wrong→correct by final round) depends on
  whether the correct answer was **latent in the R1 pool**, not on disagreement per se:
  - correct **absent** → recover 13% (MMLU) / 1.5% (GPQA) — knowledge-gap signature, debate can't fix
  - correct **present as minority** → recover 63% / 53% — reasoning/aggregation, debate promotes it
  - **initial confidence does NOT separate the two modes** (p=0.30/0.06) but debate dynamics
    (answer switches, oscillation, consensus-time τ) do (p≈1e-20 to 1e-68). ← the value-add over single-shot.
- **Intervention harness: built, smoke-tested offline, NOT yet run on the real model.**
- **Failure-type classifier: built and validated on real data (proxy labels).** Preliminary AUROC
  (knowledge vs reasoning): single-shot **confidence = 0.43** (below chance) vs debate **dynamics = 0.87**
  (+0.44). This is the expected shape — confidence can't tell the two failure modes apart, dynamics can —
  but proxy labels have structural dependence with initial-disagreement, so it only becomes the headline
  once run with `--labels causal` on the intervention output.
- **Decisive "is debate needed?" test: built.** Compute-matched sampling vs debate dynamics. Preliminary
  (proxy labels, 9 existing samples/q): sampling **0.844** vs dynamics **0.865**, difference **not
  significant** (p=0.25) and dynamics adds nothing over sampling (nested p=0.48). ⚠️ On the proxy target
  this says **debate may not be needed** — but the proxy structurally favors sampling, so this is
  provisional. The real verdict requires `--labels causal` and ideally `--source generate --k 15`.
- **Generalization checks: built (`--eval lodo` / `--eval cross-category`).** Preliminary (proxy labels):
  transfer barely degrades — LODO dynamics AUROC 0.88–0.89, cross-category (4 domains) 0.88, vs in-domain
  0.87; confidence stays at chance everywhere. So the signal is **not dataset-specific** — but again this
  is the proxy target (sampling-driven), so confirm on `--labels causal`.

## Run instructions

### 1. Reproduce the pilot (no model needed)
```bash
cd Knowledge_vs_Reasoning
python pilot_analysis.py
```

### 2. Preview the intervention subset (no model needed)
```bash
python generate_interventions.py --select-only --datasets gpqa mmlu-pro
# -> interventions/correct_absent_subset.csv  (132 qs: 51 GPQA + 81 MMLU-Pro)
```

### 3. Offline plumbing test (no model)
```bash
python generate_interventions.py --backend mock --limit 5
```

### 4. Real intervention run
```bash
# GPU (matches baseline_v2 model)
python generate_interventions.py --backend local --model-id Qwen/Qwen2.5-14B-Instruct \
    --datasets gpqa mmlu-pro --repeats 5 --require-gpu

# or Ollama (lighter)
python generate_interventions.py --backend ollama --model-id qwen2.5:7b-instruct \
    --datasets gpqa --repeats 5

# start small first
python generate_interventions.py --backend local --limit 10 --repeats 5 --require-gpu
```
Cost ≈ 132 × 4 conditions × R repeats solve calls + 132 brief calls. Runs are incremental and
safe to interrupt/resume (keyed on question × condition × repeat).

### 5. Re-label from existing results (no generation)
```bash
python generate_interventions.py --label-only --margin 0.34 --stoch 0.5 --hard 0.2
```

### 6. Train the failure-type classifier
```bash
# preliminary, runnable now (proxy labels from baseline data)
python train_failure_classifier.py

# headline result, after interventions have run
python train_failure_classifier.py --labels causal
python train_failure_classifier.py --labels causal --multiclass   # knowledge/reasoning/hard
```
Compares OOF AUROC across feature groups (confidence / self-consistency / dynamics / all),
writes coefficients + a selective risk–coverage curve to `interventions/classifier_*`.

**Generalization / dataset-specificity checks:**
```bash
python train_failure_classifier.py --eval lodo            # leave-one-dataset-out transfer
python train_failure_classifier.py --eval cross-category  # GroupKFold by domain (train on some, test on held-out)
```
`lodo` trains on one dataset and tests on the other (scaler fit on train only, handling scale shift);
`cross-category` holds out whole domains. Count features are agent/round-normalized (dimensionless) and
`n_options` is recorded but **excluded** from predictors so the model can't use it as a dataset shortcut.
Transfer AUROC ≈ in-domain AUROC ⇒ a mechanism, not a dataset artifact.

### 7. Decisive test — is the debate needed? (vs compute-matched sampling)
```bash
# runnable now: uses the ~9 R1 samples already in baseline data
python selfconsistency_baseline.py                       # proxy labels

# compute-matched: draw 15 fresh independent samples/question (no debate)
python selfconsistency_baseline.py --source generate --backend local --k 15 --require-gpu

# the real verdict, after interventions have produced causal labels
python selfconsistency_baseline.py --labels causal --source generate --k 15 --backend local --require-gpu
```
Prints OOF AUROC for sampling vs dynamics vs both, a paired-bootstrap p-value on the difference,
a nested test (does dynamics add over sampling?), and a one-line VERDICT. **This is the experiment
that decides whether "same-model debate" earns its place over just sampling the model K times.**

## Next steps (prioritized)

1. **Run the intervention harness** (step 4), starting with `--limit 10` to sanity-check, then full.
2. **Audit `interventions/knowledge_briefs.jsonl`.** The +knowledge brief is same-model, oracle-informed
   and answer-blind with a leak filter, but leakage would inflate the knowledge-limited count. This audit
   is the gate between "ran it" and "have ground truth." Consider a retrieval-based brief as a robustness check.
3. **Validate proxy vs causal labels.** Cross-tab the pilot's proxy label (correct-absent) against the
   intervention label (knowledge/reasoning/hard). Confirms correct-absent ≈ knowledge-limited and quantifies
   how often it's actually a *systematic reasoning* error masquerading as a knowledge gap.
4. **Failure-type classifier — DONE (`train_failure_classifier.py`).** Re-run with `--labels causal`
   once `intervention_labels.csv` exists to get the headline AUROC. Still TODO: add a model **self-report**
   baseline (ask the model "knowledge gap or reasoning error?") — needs a small generation pass.
4b. **Decisive test — DONE (`selfconsistency_baseline.py`).** The gate for the whole project: run with
   `--labels causal --source generate --k 15`. If debate dynamics don't beat compute-matched sampling,
   pivot the headline to **challenge-sensitivity** (the one feature sampling cannot produce).
5. **Add a clean reasoning-heavy dataset** (e.g. GSM8K / a logic set where all facts are given) to
   contrast with knowledge-heavy closed-book QA, and control for difficulty (match on base single-shot accuracy).
6. **Generalization:** repeat on a second model / scale to show the signal isn't Qwen-specific.

## Key caveats to carry into the paper

- Pilot uses **proxy** labels; interventions give causal labels but the +knowledge brief can leak (audit).
- "Correct-absent" conflates true knowledge gaps with **systematic** reasoning errors (same slip every
  clone); only same-model debate can't tell them apart — the +reasoning intervention is what separates them.
- MCQ answer space is small, so "correct in pool" is coarse; open-ended QA would sharpen it.
- Single model, single family so far (Qwen).
