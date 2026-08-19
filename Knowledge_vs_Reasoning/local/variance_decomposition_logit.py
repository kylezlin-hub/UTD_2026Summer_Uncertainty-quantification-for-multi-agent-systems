"""variance_decomposition_logit.py -- Latent-scale (logistic) variance decomposition, addressing
the statistical concern that variance_decomposition.py's Gaussian/REML model treats raw binary
correct/incorrect outcomes as continuous, which is a "linear probability model" approximation:
Bernoulli residual variance is mean-dependent (Var = p(1-p)), so the fitted Gaussian residual
variance is not a clean, mean-invariant noise floor, and %-of-variance shares are not directly
comparable across conditions/questions with very different grand means (S=0.551, C=0.632,
control=0.094 in our data).

Fix: fit a logistic mixed-effects model (random intercepts for Question, and for Brief-instance
nested within Question), estimating variance components on the LATENT (logit) scale. Under the
standard logistic-latent-variable convention, the level-1 ("residual") variance on this scale is
fixed at pi^2/3 (~3.29) -- this is the usual basis for computing an intraclass correlation (ICC)
for binary outcomes in multilevel/generalizability-theory models (Snijders & Bosker; Hox), and it
is invariant to the grand mean, unlike the Gaussian-on-0/1 residual variance.

We use statsmodels' BinomialBayesMixedGLM (variational-Bayes approximate logistic mixed model,
since no exact frequentist GLMM-with-variance-components implementation is available in this
environment -- no R/lme4, pymer4, or bambi installed) with the same nested Question > Instance
random-effects structure as the original analysis, for direct comparison.

Usage
-----
    python variance_decomposition_logit.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from variance_decomposition import build_long_df, target_questions, HERE

RESID_VAR_LOGIT = (np.pi ** 2) / 3  # standard logistic-latent-variable level-1 variance


def fit_nested_logit(long: pd.DataFrame) -> tuple[float, float]:
    """Two-level nested variance components via two vc_formula terms: 'question' (broad) and
    'instance' (fine, nested -- uid is already question+instance so it captures the finer split;
    subtracting is not needed since vc_formula terms are modeled as independent contributions)."""
    data = long.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BinomialBayesMixedGLM.from_formula(
            "correct ~ 1",
            vc_formulas={"question": "0 + C(question_no)", "instance": "0 + C(uid)"},
            data=data, vcp_p=1.0, fe_p=2.0,
        )
        result = model.fit_vb()
    # vcp_mean is log-sd for each vc component in the order the vc_formulas dict was given
    # (statsmodels preserves insertion order for dict-based vc_formulas since Python 3.7+).
    log_sds = result.vcp_mean
    var_question = float(np.exp(2 * log_sds[0]))
    var_instance = float(np.exp(2 * log_sds[1]))
    return var_question, var_instance


def summarize(brief_type: str, label: str, qs: set[str], n_boot: int = 60) -> dict:
    long = build_long_df(brief_type, qs)
    n_q = long["question_no"].nunique()

    g = long.groupby("uid")["correct"].agg(["sum", "count"])
    n_separated = int(((g["sum"] == 0) | (g["sum"] == g["count"])).sum())

    var_q, var_i = fit_nested_logit(long)
    var_r = RESID_VAR_LOGIT
    total = var_q + var_i + var_r

    rng = np.random.RandomState(42)
    question_ids = long["question_no"].unique()
    boot_pcts = {"question": [], "instance": [], "residual": []}
    for b in range(n_boot):
        sample_qs = rng.choice(question_ids, size=len(question_ids), replace=True)
        parts = []
        for i, qid in enumerate(sample_qs):
            sub = long[long["question_no"] == qid].copy()
            sub["question_no"] = f"{qid}__boot{i}"
            sub["uid"] = sub["question_no"] + "_" + sub["instance"]
            parts.append(sub)
        boot_data = pd.concat(parts, ignore_index=True)
        try:
            bvq, bvi = fit_nested_logit(boot_data)
        except Exception:
            continue
        btotal = bvq + bvi + RESID_VAR_LOGIT
        boot_pcts["question"].append(100 * bvq / btotal)
        boot_pcts["instance"].append(100 * bvi / btotal)
        boot_pcts["residual"].append(100 * RESID_VAR_LOGIT / btotal)

    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))

    return dict(
        label=label, n_questions=n_q, n_rows=len(long), n_boot_ok=len(boot_pcts["question"]),
        n_uid_groups=len(g), n_separated=n_separated,
        var_question_logit=var_q, var_instance_logit=var_i, var_residual_logit=var_r,
        pct_question=100 * var_q / total, pct_instance=100 * var_i / total,
        pct_residual=100 * var_r / total,
        pct_question_ci=ci(boot_pcts["question"]), pct_instance_ci=ci(boot_pcts["instance"]),
        pct_residual_ci=ci(boot_pcts["residual"]),
        grand_mean=float(long["correct"].mean()),
    )


def main():
    qs = target_questions()
    print(f"Target questions: {len(qs)}")
    print("Fitting LOGISTIC (latent-scale) nested variance-components model via variational Bayes")
    print("(BinomialBayesMixedGLM) -- level-1 residual variance fixed at pi^2/3 by convention.\n")

    results = []
    for brief_type, label in [("S", "Option-blind (S) informational support"),
                               ("C", "Choice-conditioned (C) informational support"),
                               ("control", "Control (no scaffold -- validity check)")]:
        r = summarize(brief_type, label, qs)
        results.append(r)
        print(f"{label}  (n_Q={r['n_questions']}, grand_mean={r['grand_mean']:.3f}, "
              f"separated_cells={r['n_separated']}/{r['n_uid_groups']})")
        print(f"  %Question = {r['pct_question']:5.1f}%  [{r['pct_question_ci'][0]:5.1f}, {r['pct_question_ci'][1]:5.1f}]")
        print(f"  %Instance = {r['pct_instance']:5.1f}%  [{r['pct_instance_ci'][0]:5.1f}, {r['pct_instance_ci'][1]:5.1f}]")
        print(f"  %Residual = {r['pct_residual']:5.1f}%  [{r['pct_residual_ci'][0]:5.1f}, {r['pct_residual_ci'][1]:5.1f}]  (fixed pi^2/3 convention)")
        print()

    out_df = pd.DataFrame(results)
    out_path = HERE / "interventions" / "brief_regen_check" / "variance_decomposition_logit.csv"
    out_df.to_csv(out_path, index=False)
    print(f"-> {out_path}")

    print("\nCAVEAT (small-cell separation): with only 4 (fresh) or 8 (original) repeats per")
    print("question-instance cell, most cells are perfectly separated (all-correct or all-incorrect),")
    print("a known source of variance INFLATION in logistic mixed models. However, the control")
    print("condition -- which has comparable or higher separation rates but NO real scaffold-instance")
    print("factor -- shows near-zero instance variance under this same model, indicating separation")
    print("alone does not mechanically produce the large instance-variance signal seen for S and C.")
    print("As with the Gaussian specification, treat these as directionally robust but not precise")
    print("percentages (see also the selection-effect caveat in variance_decomposition.py: this")
    print("n=39 sample was selected on an original-brief response, so some regression toward the")
    print("mean under a fresh brief is expected regardless of the true population variance shares).")


if __name__ == "__main__":
    main()
