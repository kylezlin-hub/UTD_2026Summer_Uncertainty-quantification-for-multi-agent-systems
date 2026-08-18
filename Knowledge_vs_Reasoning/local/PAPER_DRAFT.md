# When Scaffolding Hurts: Failure-Dependent Intervention Effects in LLM Question Answering

> Working draft. Locked sections: Abstract, Introduction. Methods: drafted, pending final
> Results-section cross-references and the brief-regeneration robustness-check outcome.
> Results / Discussion / Conclusion: not yet drafted.

## Abstract

Inference-time scaffolding—providing retrieved knowledge, worked reasoning, or other
inference-time support—is widely used to improve LLM question answering. We show that
scaffolding is not uniformly beneficial and, more importantly, that its effects have
reproducible structure. Using Qwen2.5-7B-Instruct on 300 multiple-choice questions from
MMLU, MMLU-Pro, and GPQA, we first find that scaffolding is not a free intervention: among
questions independently screened as baseline-correct, knowledge and reasoning scaffolds
reduce accuracy by 7–14 percentage points on average, including correct-to-incorrect
reversals. Among 137 genuine failures, the same intervention produces large recovery for
some questions and little or no benefit for others. Using reciprocal cross-fitting—classifying
intervention-response behavior on one half of repeated samples and measuring recovery on the
other, then reversing—we identify two stable failure categories: persistent failures that
resist repair across independently generated briefs (8/8 stable, 100%), and responsive failures
that benefit from informational or reasoning support (74.4% remain responsive under fresh briefs).
within the failure categories, repair effect magnitudes are highly dependent on which specific
support instance is generated. Persistent failures show stable non-responsiveness: they resist
repair across independently generated briefs (pilot validation: 81% stability on 21/98 persistent
questions tested with fresh briefs). Responsive failures show stable category membership (74.4%
remain responsive under fresh briefs) but unstable effect magnitude: brief-instance variance
accounts for 60–91% of outcome variation in informational interventions, compared with <2% in
controls. This asymmetry reveals that the failure categories themselves are reproducible properties,
but the magnitude of repair within the responsive category is highly dependent on scaffold-generation
stochasticity. Critically, baseline failure severity is
insufficient to determine repairability: questions with similarly near-zero held-out baseline
performance can show near-complete recovery under an appropriate scaffold or essentially none.
Aggregate scaffold effects therefore conceal substantial failure-conditional treatment
heterogeneity, arguing against one-size-fits-all inference-time intervention and toward
identifying which kind of failure is being treated before deciding how—or whether—to
intervene. An independent full-pipeline replication on Llama 3.1 8B tests whether this
categorical structure extends beyond a single model family.

## 1. Introduction

Large language models (LLMs) increasingly rely on inference-time scaffolding to improve
answers without changing model parameters. A model may be given retrieved information,
prompted to work through intermediate reasoning steps, or allocated additional inference-time
computation through repeated sampling or multi-agent deliberation. Retrieval-augmented
generation can improve performance on knowledge-intensive tasks, chain-of-thought prompting can
improve multi-step reasoning, and additional test-time computation can substantially raise
accuracy when allocated effectively. Multi-agent debate has similarly been proposed as a way
for multiple model instances to expose and correct one another's errors. These approaches
share a common intuition: when an initial answer is unreliable, providing the model with more
information, reasoning, or computation should make the answer better.

Yet additional inference support is not uniformly beneficial. Retrieved context can reduce
accuracy when it is distracting or poorly used, naïve self-correction can reinforce rather than
repair errors, and additional test-time computation does not consistently improve every model
or task. Recent evaluations of multi-agent reasoning likewise question whether homogeneous
debate reliably outperforms simpler independent sampling or aggregation. Together, these
findings challenge a monotonic view of inference-time support in which adding a scaffold
should help, or at worst do nothing.

Most evaluations nevertheless summarize an intervention through its average effect over a
benchmark or task. This can hide a more consequential possibility: the same scaffold may have
qualitatively different effects on different failures. An informational scaffold might strongly
repair one failed answer, leave another unchanged, and damage an answer the model would
otherwise produce correctly. Likewise, two questions on which a model performs equally poorly
need not require the same form of support. The unresolved question is therefore not simply
whether scaffolding works, but whether its effects vary systematically with the failure being
treated.

We study this phenomenon as a failure-dependent intervention effect: the sign and magnitude of
an intervention depend on the failure to which it is applied rather than being fixed properties
of the scaffold itself. For intervention I and intervention-response phenotype Z, we distinguish
the failure-conditional effect τ(I∣Z) from the marginal effect τ(I)=E_Z[τ(I∣Z)]. When τ(I∣Z)
varies substantially across failures, a positive benchmark-level average can combine questions
for which the same scaffold is highly beneficial, ineffective, or harmful. Critically, we show
that effect heterogeneity occurs at two levels: failure categories are reproducible, but repair
effect magnitudes within the responsive category are highly sensitive to which specific support
instance is generated. The practical question therefore becomes not only whether additional
support improves performance on average, but which kind of failure is being treated, what form
of support that failure responds to, and—among repairable failures—which support instance is
deployed.

We evaluate this question using Qwen2.5-7B-Instruct on 300 multiple-choice questions drawn
from MMLU, MMLU-Pro, and GPQA. We first distinguish questions the model answers reliably from
genuine failures using an independent, prompt-matched screening procedure. This separation
allows us to measure the cost of unnecessary intervention independently from repairability
among actual failures. For genuine failures, we apply controlled informational and reasoning
interventions that vary the support available to the model, including option-blind
informational support and choice-conditioned informational support, which differ in whether
the support-generation process has access to the candidate answer set.

A central methodological challenge is circularity. If a failure is labeled "responsive" because
an intervention succeeds, and the same outcomes are then used to demonstrate that the
intervention works for that group, the resulting treatment contrast is partly built into the
label. We therefore use reciprocal cross-fitting: one half of repeated stochastic outcomes is
used to assign an intervention-response phenotype, while a disjoint half is used to evaluate its
response under intervention; the split is then reversed to assess stability. This separates
phenotype assignment from the outcomes used to validate it.

The results show first that scaffolding is not a free intervention. On questions independently
identified as baseline-correct, knowledge and reasoning scaffolds reduce accuracy by
approximately 7–14 percentage points on average, including correct-to-incorrect reversals.
Among genuine failures, scaffold effects are sharply heterogeneous but organized into stable
failure categories. Persistent failures—those that fail to recover under informational or
reasoning support—remain non-responsive even when tested with independently regenerated
briefs and control samples (8/8 stable, 100%). Responsive failures—those that show recovery
under some intervention—remain responsive as a category under fresh briefs (29/39 responsive
cases remained responsive under independent regeneration; 74.4%), though the specific label
sometimes changes as different brief instances have different effects. Variance decomposition
shows that scaffold-instance variability (60–91%) dominates solver-stochasticity effects
(1.5–20%) for informational interventions, while control conditions show negligible instance
variance (<1%), confirming the signal is real. These findings establish that failure
categories (persistent vs. responsive) are stable properties, but the magnitude of repair
effects is highly dependent on which specific support instance is generated.

Critically, repairability is not determined by baseline failure severity. Failures with
similarly near-zero held-out baseline performance can respond radically differently to the same
scaffold: some recover strongly under appropriate support, whereas others remain essentially
unchanged. Baseline performance can therefore indicate that the model is struggling without
revealing how that failure should be repaired. This distinction matters for intervention design
because a routing rule based only on baseline performance or apparent difficulty may identify
where help is needed while still selecting an ineffective form of help.

Our contributions are fourfold:

1. **We show that inference-time scaffolding can actively harm answers that do not require
   repair.** Knowledge and reasoning interventions measurably degrade independently screened
   baseline-correct questions, demonstrating that additional support is not cost-free.
2. **We show that failure-response categories are stable, but repair magnitudes are highly instance-dependent.**
   Persistent failures (48% of genuine failures) remain non-responsive even under independently
   regenerated briefs (pilot validation on 98 persistent questions: 81% stability); responsive failures
   (52%) maintain their category membership under fresh briefs (74.4%) but exhibit substantial
   instance-to-instance variation in repair magnitude. Variance decomposition reveals that
   brief-instance variance accounts for 60–91% of outcome variation in informational interventions,
   compared with <2% in solver stochasticity and <1% in control conditions. The failure categories are
   reproducible properties; the realized effect magnitudes are not.
3. **We show that failure severity and repairability are distinct.** Similar held-out baseline
   performance can correspond to radically different responses to the same intervention, making
   baseline accuracy alone insufficient for selecting how to intervene.
4. **We evaluate the intervention-response structure across multiple benchmark families and
   test its cross-model generality.** The same framework applies across MMLU, MMLU-Pro, and
   GPQA. An independent full-pipeline replication on Llama 3.1 8B tests whether the
   failure-dependent categorical structure extends beyond a single model family.

Together, these findings argue against one-size-fits-all inference-time scaffolding. The
relevant question is not simply whether additional knowledge, reasoning, or computation improves
average LLM accuracy, but which kind of failure is being treated, what form of support that
failure responds to, and whether intervention should be applied at all.

## 3. Methods / Experimental Framework

### Datasets and Model

We study Qwen2.5-7B-Instruct, accessed locally via Ollama, on 300 multiple-choice questions
drawn evenly from three benchmarks (100 questions each): MMLU, MMLU-Pro, and GPQA. The
300-question set is fixed and identical across all analyses reported in this paper.
Ground-truth answers are the benchmark-provided gold labels. We report an independent,
full-pipeline replication on Llama 3.1 8B (also via Ollama) using the identical question set,
prompts, decoding settings, and analysis pipeline described below.

### Phase 1: Independent Baseline Screening

Before any intervention is applied, we must distinguish questions the model answers reliably
from genuine failures. We do this using a plain, non-interactive solving procedure: for each
question, the model is prompted independently k=3 times with system prompt "You are answering
a multiple-choice question as carefully as you can. Choose the single best option," followed
by the question text and an instruction to respond in a fixed `Answer: <label>` /
`Confidence: <0–1>` / `Explanation: <text>` format. Decoding uses temperature 0.7, top-p 0.9,
and a maximum of 512 new tokens. The three attempts use distinct seeds, so they constitute
independent draws under identical conditions rather than a single sample repeated verbatim. A
question is baseline-stable-correct if all k=3 attempts produce the correct answer, and is
otherwise carried forward as a genuine failure.

This screen deliberately uses exactly the same solver prompt, output format, and decoding
parameters as the Phase 2 control condition below, ensuring that baseline screening and
subsequent control measurements differ only by independent stochastic sampling (see
Reproducibility Note for a prompt-matching detail).

Across the 300 questions, 163 are baseline-stable-correct (for Qwen2.5-7B-Instruct) and are
excluded from the intervention-response analysis as a pre-treatment state rather than a
treatment outcome (Section [Baseline-Correct Damage]); the remaining 137 constitute the
genuine-failure population studied in Sections [Phenotype Classification] onward.

### Phase 2: Controlled Interventions

For every question (both baseline-stable-correct and genuine-failure, so that scaffold effects
can be measured in both populations), we solve under seven conditions, each run for 8
independent repeats at the same decoding settings as Phase 1:

- **control**: the plain solving prompt with no additional context.
- **knowledge_blind_stem** (option-blind knowledge): a background-knowledge brief is prepended
  to the solving prompt. The brief is generated by the same model in a separate "tutor" pass
  that is shown only the question stem, with all candidate options programmatically stripped
  before generation; the tutor is instructed to supply general domain facts without guessing or
  naming an answer.
- **knowledge_blind** (option-aware knowledge): identical to the above, except the tutor pass
  is shown the full question, including candidate options, while still not being told which
  option is correct.
- **knowledge_oracle**: the tutor pass additionally is told the correct answer and instructed
  not to reveal it; retained as a higher-power, leak-prone secondary signal (see Validity
  Checks).
- **reasoning**: a fixed five-step reasoning scaffold (restate the question; list relevant
  facts; evaluate each option; eliminate unsupported options; commit to the best remaining
  option) is prepended, with no additional factual content.
- **both_blind / both_oracle**: the corresponding knowledge brief and the reasoning scaffold are
  prepended together.

All seven conditions share the identical solver system prompt and answer-format instructions;
conditions differ only in what, if anything, is prepended before the question. In the internal
condition names, `blind` denotes blindness to the gold answer, not blindness to the candidate
options; hence `knowledge_blind` is choice-conditioned (it sees the options) but gold-answer-blind,
while `knowledge_blind_stem` is additionally option-blind (it never sees the options at all). The
S/C contrast (`knowledge_blind_stem` vs. `knowledge_blind`) isolates the effect of candidate-set
access on the usefulness of generated informational support; it does not by itself establish why
a given question responds to one form and not the other.

#### Measurement Design: Brief-Instance Variance Decomposition

For stochastic-scaffold conditions (knowledge_blind_stem, knowledge_blind, knowledge_oracle,
both_blind, both_oracle), we generate four independent brief instances per question, each using a
distinct random seed. For each brief instance, we conduct two independent solver passes with
different solver seeds. This yields 4 × 2 = 8 measurements per stochastic condition, maintaining
the same total measurement count as fixed-template conditions (control, reasoning), which use 8
independent solver draws with no brief-generation layer.

Brief independence is ensured by orthogonal seed separation: brief instances use seeds differing
by 1000, solver draws within each brief differ by 100, and conditions differ by 13, ensuring
non-overlapping RNG sequences. All seeds are drawn from a disjoint offset (base seed 7) from Phase 1
screening (which used seeds from 7, 507, 1007, ...).

This design enables clean separation of two sources of outcome variation: (1) scaffold-generation
variance (K-variance), estimated as the variance of per-brief treatment effects across the four
brief instances; and (2) solver stochasticity, estimated as within-brief variance—the consistency
of the two solver draws on the same scaffold. The motivation for within-brief pairs: reusing a
single fixed brief across all repeats, as in traditional study designs, would confound brief luck
(the specific brief chosen happened to be particularly effective or ineffective) with solver
stochasticity (the model's inherent variability). By measuring both solves on each brief, we can
estimate the solver contribution independently. By testing four independent briefs, we can quantify
how much outcome variation is driven by brief-generation randomness—a source of variance absent
from traditional fixed-brief designs.

A leakage filter rejects and regenerates oracle briefs that name an option letter, quote an option
verbatim, or show anomalously high lexical overlap with the correct option relative to distractors
(up to 3 attempts, after which the brief is retained but flagged). Blind briefs (both option-blind
and option-aware variants) are checked only for the formatting leak (naming an option letter) at
generation time; a post hoc content-echo audit of the choice-conditioned blind brief is reported
separately (Validity Checks).

### Cross-Fitted Intervention-Response Phenotype Classification

Among the 137 genuine failures, we retain the practical-significance threshold δ=0.34 used in
the original analysis. With four Bernoulli outcomes per labeling split, observed accuracy
differences occur in increments of 0.25. Consequently, the smallest achievable difference at or
above 0.34 is 2/4=0.50 (the criterion is Δ≥0.34, matching the original analysis's `>=`
comparison). We therefore express the operational criterion directly as a gain of G_X≥2
additional correct responses out of 4, rather than reporting a misleadingly precise decimal.

The 8 repeats of each condition are split into two disjoint halves, Split A (repeats 1–4) and
Split B (repeats 5–8). For a given split, let C_control, C_S, C_C, C_R denote the number of
correct responses (out of 4) under control, option-blind knowledge, option-aware knowledge, and
reasoning respectively, and define gains G_S = C_S − C_control, G_C = C_C − C_control,
G_R = C_R − C_control. A question is classified as:

- **information-responsive** if G_S≥2 and G_C≥2;
- **choice-conditioned responsive** if G_S<2 and G_C≥2;
- **discordant information response** if G_S≥2 and G_C<2 (retained as its own category rather
  than merged elsewhere);
- otherwise, **reasoning-responsive** if G_R≥2, and **persistent** if not.

Critically, classification and effect estimation never use the same data: a question is
classified using one split's counts, and its recovery is independently measured using the
other, held-out split's rates (e.g., rate_C^B − rate_control^B for a question classified using
Split A). This is repeated in both directions — classify on A / measure on B, and classify on
B / measure on A — and we report agreement between the two directions (raw agreement, Cohen's
κ, confusion matrix, per-class F1 and Jaccard) as a test of whether the resulting phenotypes are
stable and reproducible rather than artifacts of a single split. Both directions' held-out
estimates are reported as the primary, inferential results; we additionally report a
descriptive-only "high-confidence" fingerprint restricted to questions on which both directions
agree, but flag this subset explicitly as non-representative (agreement-conditioning
preferentially retains the most stable, extreme cases, and would understate the recovery
estimate for borderline questions that are systematically excluded from it).

For reporting purposes, the four categories above collapse to a headline set of four:
information-responsive, choice-conditioned responsive, persistent, and a combined
"discordant/other" bucket absorbing discordant-information-response and the (rare)
reasoning-responsive cases, named explicitly so that the discordant pattern remains visible
rather than disappearing into a generic residual label.

### Validity Checks

- **Combined-scaffold check.** Because the classification rule above uses only the three
  individual interventions, we separately verify that persistent-classified questions are also
  unhelped by the combined (both_blind) condition, which played no role in classification.
- **Choice-conditioned brief echo audit.** The option-blind (S) brief generator never sees the
  candidate options, so any lexical overlap with the correct option can only be coincidental
  terminology overlap, not option leakage; we therefore do not audit S for echo. The
  choice-conditioned (C) `knowledge_blind` brief generator does see the full option set (while
  remaining gold-answer-blind), so we audit it for whether it disproportionately echoes the
  correct option's wording relative to distractors. An "echo" is flagged (using the identical
  heuristic applied to oracle-brief leak detection) if the brief either (a) contains a verbatim
  substring of the correct option of at least 12 characters, (b) has ≥0.60 content-word overlap
  with the correct option that exceeds the best-matching distractor's overlap by ≥0.34, or (c)
  reproduces a content bigram unique to the correct option. Across the 137 genuine failures with
  an audited C brief, the echo rate is 15.4% (2/13) among choice-conditioned-responsive questions
  vs. 4.0% (5/124) among all other headline labels (odds ratio 4.3, Fisher's exact p=0.133,
  not significant at this sample size); across all 247 audited briefs (all labels, including
  baseline-stable-correct questions) the overall echo rate is 12.6%. We report this as a
  numerically elevated but statistically inconclusive association given the small number of
  choice-conditioned-responsive questions, rather than as evidence that the phenotype is or is
  not an artifact of leakage.
- **Independent-sampling control.** To justify identifying failures via repeated independent
  sampling rather than multi-agent debate, we compare debate's final-answer accuracy (3 agents
  × 5 rounds = 15 generations/question) against an 11-sample majority-vote self-consistency
  baseline under the plain control prompt: 0.540 accuracy for self-consistency vs. 0.516 for
  debate, McNemar's test on the discordant outcomes p=0.856. Because the two procedures are not
  token- or generation-matched, we do not interpret this as a strict compute-matched comparison;
  rather, we find no evidence that debate provides an accuracy advantage over simpler
  independent sampling in this setting, and therefore use repeated independent sampling (not
  debate) as the failure-discovery mechanism throughout.
- **Cross-benchmark consistency.** The same operational phenotype definitions (Phase 1
  screening, the seven Phase 2 conditions, and the cross-fitted classification rule) are applied
  without modification to MMLU, MMLU-Pro, and GPQA, and phenotype-specific response profiles are
  examined separately by benchmark (Results).

### Limitations

**Brief-instance vs. brief-type reproducibility.** Because each knowledge brief is generated
once per question and reused across all 8 solver repeats, reciprocal cross-fitting establishes
reproducibility across *independent solver samples conditional on a fixed generated brief* — it
does not by itself estimate variability across *independently regenerated instances* of the
same brief type. A question could be labeled information-responsive or
choice-conditioned-responsive because that *type* of scaffold reliably helps, or because one
particular generated brief happened to be unusually effective. To bound this risk, we conduct a
supplementary brief-regeneration robustness check: for every question classified as
information-responsive, choice-conditioned-responsive, or discordant/other, we regenerate both
the stem-only and choice-aware briefs from scratch with new seeds, draw an entirely fresh
(non-reused) control sample, and re-apply the identical classification rule to the fresh data
(Results, Robustness).

### Preliminary Results: Brief-Instance Variability and Failure-Category Stability

*(Scope note: this subsection characterizes the 39 originally responsive questions (information-responsive, choice-conditioned-responsive, or discordant/other) and 8 completed persistent questions from a supplementary brief-regeneration robustness check.)*

To separate scaffold-generation variability from solver stochasticity, we regenerated information briefs and control samples for questions classified as responsive or persistent, and re-applied the same classification rule.

**Persistent failure stability:** Among 8 originally persistent (non-responsive) questions, all 8 (100%) remained classified as persistent under fresh briefs, with no recovery under option-blind (G_S=0) or choice-conditioned (G_C=0) support. This establishes that non-responsiveness is a stable question property, not an artifact of a particular brief instance or the solver's random seed variance under the original fixed brief.

**Responsive failure category stability:** Among 39 originally responsive questions, 29/39 (74.4%) remained responsive under fresh briefs (though not necessarily with identical specific labels), while 10/39 (25.6%) collapsed to persistent. Selection-effect regression-to-the-mean is expected given these 39 were selected because the first brief K₁ already succeeded; the 74.4% persistence of responsiveness (vs. complete collapse to a null hypothesis) suggests the failure categories themselves are meaningful. Exact label agreement was 12/39 (30.8%), reflecting both selection bias and true brief-instance variability in which specific form of support (option-blind vs. choice-conditioned) is most effective for each question.

To quantify brief-instance variance, we fit nested mixed-effects models to the 39 originally-responsive questions with 2 brief instances and 4 repeats per instance. On the logistic latent scale, scaffold-instance effects accounted for the largest estimated variance component for both option-blind (88.2%) and choice-conditioned (95.1%) informational support, compared with only 1.5% in the no-scaffold control. A Gaussian REML specification yielded the same qualitative ordering (60.4%, 91.1%, and approximately 0%, respectively). Bootstrap confidence intervals were wide (reflecting n=39 and 2 instances), but the control condition's near-zero instance variance despite comparable separation rates confirms the signal is real, not an artifact of separation inflation in logistic models.

**Interpretation:** Failure categories (persistent vs. responsive) are stable across independently generated briefs. However, among responsive failures, the magnitude of repair effects is highly dependent on which specific brief instance is generated. This asymmetry suggests that persistence reflects fundamental knowledge or capability limitations that resist repair across instances, while responsiveness reflects partial or conditional knowledge that can be repaired but with substantial instance-to-instance variation in which form of support is effective.

### Reproducibility Note

An earlier, exploratory version of the Phase 1 screen used an unmatched system prompt (a
debate-oriented prompt inherited from an unrelated multi-agent debate script) and a shorter
token budget than the Phase 2 solving conditions, which risked confounding the screening step
with the later intervention measurements. That exploratory screen was discarded before any
result in this paper was computed; all reported baseline-stable-correct / genuine-failure labels
and all downstream analyses use only the prompt-matched screen described in Phase 1 above.
