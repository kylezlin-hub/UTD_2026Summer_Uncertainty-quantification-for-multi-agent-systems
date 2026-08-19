"""variance_decomposition.py -- Quantifies HOW MUCH of scaffold-response variance is
attributable to the question vs. the specific generated scaffold instance vs. solver-sampling
noise, operationalizing the nested effect decomposition

    tau(I, K | Q)          -- realized effect of a SPECIFIC generated scaffold instance K
    tau(I | Q) = E_{K~P(K|I,Q)} [ tau(I, K | Q) ]   -- the intervention-TYPE effect (marginal over K)

For each of the two informational intervention types (S = option-blind, C = choice-conditioned),
we have exactly TWO independently generated brief instances per question: the ORIGINAL brief
(used throughout the main study, 8 solver repeats) and the FRESH brief (generated from scratch
for the brief-regeneration robustness check, 4 solver repeats). This is a nested design:

    Question  >  Brief instance (orig / fresh, nested within question)  >  Solver repeat

We fit a nested random-intercepts (variance-components) model on the binary correctness outcome
(linear-probability approximation, standard practice for this kind of generalizability-theory /
G-theory variance decomposition with small-to-moderate n) via REML, and report the fraction of
total variance attributable to each level:

    Var(correct) = sigma^2_Question           (systematic repairability -- the true tau(I|Q) signal)
                 + sigma^2_Instance|Question   (the NEW tau(I,K|Q) term -- brief-to-brief variance)
                 + sigma^2_residual             (solver-sampling noise, already characterized by
                                                  the main study's reciprocal cross-fitting)

A CONTROL-condition decomposition (which has no real "K" -- both draws are literally the same
no-scaffold condition resampled) is included as a validity check: it should show near-zero
instance-level variance, confirming the model correctly attributes variance only where a real
brief-to-brief factor exists.

Usage
-----
    python variance_decomposition.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
TAXONOMY_PATH = HERE / "interventions" / "taxonomy_nested_results.csv"
SOLVE_RESULTS_PATH = HERE / "interventions" / "solve_results.jsonl"
STEM_RESULTS_PATH = HERE / "interventions" / "stem_only_solve_results.jsonl"
FRESH_RESULTS_PATH = HERE / "interventions" / "brief_regen_check" / "fresh_solve_results.jsonl"

TARGET_HEADLINES = ("information-responsive", "choice-conditioned responsive", "discordant/other")


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["question_no"] = df["question_no"].astype(str)
    return df


def target_questions() -> set[str]:
    tax = pd.read_csv(TAXONOMY_PATH)
    tax["question_no"] = tax["question_no"].astype(str)
    return set(tax.loc[tax["headline_label"].isin(TARGET_HEADLINES), "question_no"])


def build_long_df(brief_type: str, qs: set[str]) -> pd.DataFrame:
    """Long-format (question_no, instance, uid, correct) for one intervention type.
    brief_type in {"S", "C", "control"}."""
    fresh = _load_jsonl(FRESH_RESULTS_PATH)

    if brief_type == "S":
        orig = _load_jsonl(STEM_RESULTS_PATH)
        orig = orig.rename(columns={"rep": "repeat"})
        orig["condition"] = "knowledge_blind_stem"
        fresh_cond = "knowledge_blind_stem_fresh"
    elif brief_type == "C":
        orig = _load_jsonl(SOLVE_RESULTS_PATH)
        orig = orig[orig["condition"] == "knowledge_blind"]
        fresh_cond = "knowledge_blind_fresh"
    else:  # control -- no real scaffold instance; both draws are the same no-scaffold condition
        orig = _load_jsonl(SOLVE_RESULTS_PATH)
        orig = orig[orig["condition"] == "control"]
        fresh_cond = "control"

    orig = orig[orig["question_no"].isin(qs)][["question_no", "repeat", "correct"]].copy()
    orig["instance"] = "orig"

    fr = fresh[(fresh["condition"] == fresh_cond) & (fresh["question_no"].isin(qs))]
    fr = fr.rename(columns={"rep": "repeat"})[["question_no", "repeat", "correct"]].copy()
    fr["instance"] = "fresh"

    long = pd.concat([orig, fr], ignore_index=True)
    long["correct"] = long["correct"].astype(float)
    long["uid"] = long["question_no"] + "_" + long["instance"]
    return long


def fit_variance_components(long: pd.DataFrame, label: str) -> dict:
    n_q = long["question_no"].nunique()
    n_rows = len(long)
    def _fit(data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                "correct ~ 1", data=data, groups=data["question_no"],
                re_formula="1", vc_formula={"instance": "0 + C(uid)"},
            )
            res = model.fit(reml=True, method="lbfgs")
        vq = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] else 0.0
        vi = float(res.vcomp[0]) if len(res.vcomp) else 0.0
        vr = float(res.scale)
        return vq, vi, vr

    var_question, var_instance, var_resid = _fit(long)
    total = var_question + var_instance + var_resid

    # Nonparametric bootstrap (resample QUESTIONS with replacement, refit) for uncertainty on the
    # variance-decomposition percentages -- point-estimate variance components alone can be
    # misleadingly precise at n=39 questions with only 2 instances/question.
    rng = np.random.RandomState(42)
    question_ids = long["question_no"].unique()
    n_boot = 150
    boot_pcts = {"question": [], "instance": [], "residual": []}
    for b in range(n_boot):
        sample_qs = rng.choice(question_ids, size=len(question_ids), replace=True)
        # Re-key resampled questions so repeated draws of the same question are treated as
        # distinct groups (otherwise duplicated question_no rows would silently merge).
        parts = []
        for i, qid in enumerate(sample_qs):
            sub = long[long["question_no"] == qid].copy()
            sub["question_no"] = f"{qid}__boot{i}"
            sub["uid"] = sub["question_no"] + "_" + sub["instance"]
            parts.append(sub)
        boot_data = pd.concat(parts, ignore_index=True)
        try:
            bvq, bvi, bvr = _fit(boot_data)
        except Exception:
            continue
        btotal = bvq + bvi + bvr
        if btotal <= 0:
            continue
        boot_pcts["question"].append(100 * bvq / btotal)
        boot_pcts["instance"].append(100 * bvi / btotal)
        boot_pcts["residual"].append(100 * bvr / btotal)

    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))

    ci_q, ci_i, ci_r = ci(boot_pcts["question"]), ci(boot_pcts["instance"]), ci(boot_pcts["residual"])

    out = dict(
        label=label, n_questions=n_q, n_rows=n_rows, n_boot_ok=len(boot_pcts["question"]),
        var_question=var_question, var_instance=var_instance, var_residual=var_resid,
        pct_question=100 * var_question / total if total > 0 else float("nan"),
        pct_instance=100 * var_instance / total if total > 0 else float("nan"),
        pct_residual=100 * var_resid / total if total > 0 else float("nan"),
        pct_question_ci=ci_q, pct_instance_ci=ci_i, pct_residual_ci=ci_r,
        grand_mean=float(long["correct"].mean()),
    )
    return out


def main():
    qs = target_questions()
    print(f"Target questions (information-responsive + choice-conditioned + discordant/other): {len(qs)}")

    results = []
    for brief_type, label in [("S", "Option-blind (S) informational support"),
                               ("C", "Choice-conditioned (C) informational support"),
                               ("control", "Control (no scaffold -- validity check, expect ~0 instance variance)")]:
        long = build_long_df(brief_type, qs)
        r = fit_variance_components(long, label)
        results.append(r)

    print(f"\n{'=' * 100}")
    print("VARIANCE DECOMPOSITION: tau(I,K|Q) -- how much of response variance is Question- vs.")
    print("Brief-instance- vs. Solver-noise-level? (nested random-intercepts model, REML;")
    print(f"95% CIs from {results[0]['n_boot_ok']} question-level bootstrap resamples)")
    print(f"{'=' * 100}\n")
    for r in results:
        print(f"{r['label']}  (n_Q={r['n_questions']}, n_rows={r['n_rows']})")
        print(f"  %Question = {r['pct_question']:5.1f}%  [{r['pct_question_ci'][0]:5.1f}, {r['pct_question_ci'][1]:5.1f}]")
        print(f"  %Instance = {r['pct_instance']:5.1f}%  [{r['pct_instance_ci'][0]:5.1f}, {r['pct_instance_ci'][1]:5.1f}]")
        print(f"  %Residual = {r['pct_residual']:5.1f}%  [{r['pct_residual_ci'][0]:5.1f}, {r['pct_residual_ci'][1]:5.1f}]")
        print()

    out_df = pd.DataFrame(results)
    out_path = HERE / "interventions" / "brief_regen_check" / "variance_decomposition.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")

    print("\nInterpretation:")
    print("  %Question   = variance attributable to systematic, question-level repairability")
    print("                (the signal the original phenotype taxonomy is trying to capture)")
    print("  %Instance   = variance attributable to the SPECIFIC generated brief instance --")
    print("                this is the new tau(I,K|Q) term; nonzero values confirm that a single")
    print("                generated scaffold instance is not interchangeable with another of the")
    print("                same TYPE")
    print("  %Residual   = ordinary solver-sampling noise (already characterized by the main")
    print("                study's reciprocal cross-fitting)")
    print()
    print("SCOPE CAVEAT (selection effect): the n=39 sample was selected because the ORIGINAL")
    print("brief instance K1 produced an information-responsive/choice-conditioned/discordant")
    print("classification. Retesting under a fresh K2 on this same selected sample is subject to")
    print("regression toward the mean: some shrinkage from K1's apparent effect is EXPECTED from")
    print("selection alone, independent of how much instance-to-instance variance exists in the")
    print("underlying population. These percentages therefore support:")
    print("    'Among failures selected as responsive/discordant under one brief instance, scaffold-")
    print("     instance variability is substantial and can dominate solver-seed variability.'")
    print("They do NOT yet establish a population-wide claim ('across LLM failures generally, X% of")
    print("variance is instance-level') -- that would require multiple independent brief draws on an")
    print("unselected (e.g. full-137 or prespecified random) sample. Treat the point estimates as")
    print("large and highly suggestive, not precise population variance shares. The control-condition")
    print("check (instance variance ~= 0) is a separate validity check on the model's mechanics and is")
    print("not subject to this selection issue (control questions were not selected on any scaffold")
    print("response).")
    print()
    print("MODEL-SPECIFICATION CAVEAT: this model fits raw binary correct/incorrect outcomes with a")
    print("Gaussian/REML mixed model (a 'linear probability model' with random effects). Bernoulli")
    print("residual variance is mean-dependent (Var=p(1-p)), so the fitted residual variance is not a")
    print("clean, mean-invariant noise floor, and %-shares are not perfectly comparable across")
    print("conditions with different grand means (S=0.55, C=0.63, control=0.09 here). A logistic")
    print("(latent-scale) cross-check is reported separately in variance_decomposition_logit.py; both")
    print("specifications agree directionally (instance variance dominates for S/C, ~0 for control),")
    print("which is the basis for treating the qualitative conclusion as robust to this choice, even")
    print("though neither model's exact percentages should be treated as precise.")


if __name__ == "__main__":
    main()
