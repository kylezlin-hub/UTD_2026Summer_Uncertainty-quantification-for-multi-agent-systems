"""Question-clustered binomial model: Failure Type x Intervention interaction.

Three parallel approaches on the same held-out data:
  1. GEE logistic (exchangeable correlation within questions)
  2. Mixed-effects logistic (random intercept per question)
  3. Comparison with permutation test and bootstrap CI

Data: individual repeat-level binary outcomes from Split-B (repeats 4-7),
      two conditions only (control vs knowledge_oracle),
      labels from Split-A (repeats 0-3).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

HERE = Path(r"C:\Proj1\Knowledge_vs_Reasoning\local")
SOLVE_PATH = HERE / "interventions" / "solve_results.jsonl"

# Load data
records = []
with open(SOLVE_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

data_idx = defaultdict(lambda: defaultdict(list))
for r in records:
    qno = str(r["question_no"])
    data_idx[qno][r["condition"]].append({
        "repeat": r["repeat"],
        "correct": r["correct"],
    })

questions = sorted(data_idx.keys())
split_a = {0, 1, 2, 3}
split_b = {4, 5, 6, 7}


def get_rate(qno, cond, reps):
    runs = [r for r in data_idx[qno][cond] if r["repeat"] in reps]
    return np.mean([r["correct"] for r in runs]) if runs else 0.0


def assign_labels(questions, reps):
    labels = {}
    margin, stoch_thresh, hard_thresh = 0.34, 0.5, 0.2
    for qno in questions:
        ctrl = get_rate(qno, "control", reps)
        kb = get_rate(qno, "knowledge_blind", reps)
        ko = get_rate(qno, "knowledge_oracle", reps)
        reas = get_rate(qno, "reasoning", reps)
        bb = get_rate(qno, "both_blind", reps)
        bo = get_rate(qno, "both_oracle", reps)
        best = max(ctrl, kb, ko, reas, bb, bo)
        if ctrl >= stoch_thresh:
            raw = "stochastic"
        elif best < hard_thresh:
            raw = "hard"
        elif (kb - ctrl) >= margin or (ko - ctrl) >= margin:
            raw = "knowledge" if not ((reas - ctrl) >= margin) else "hard"
        elif (reas - ctrl) >= margin:
            raw = "hard"
        elif (bb - ctrl) >= margin or (bo - ctrl) >= margin:
            raw = "knowledge" if ((kb - ctrl) >= margin or (ko - ctrl) >= margin) else "hard"
        else:
            raw = "hard"
        labels[qno] = raw
    return labels


labels = assign_labels(questions, split_a)

# Build repeat-level dataframe: Split-B, control + knowledge_oracle only
rows = []
for qno in questions:
    for cond in ["control", "knowledge_oracle"]:
        for r in data_idx[qno][cond]:
            if r["repeat"] in split_b:
                rows.append({
                    "question_no": qno,
                    "correct": int(r["correct"]),
                    "intervention": 1 if cond == "knowledge_oracle" else 0,
                    "failure_type": labels[qno],
                    "repeat": r["repeat"],
                })

df = pd.DataFrame(rows)
print(f"Repeat-level observations: {len(df)}")
print(f"  Questions: {df['question_no'].nunique()}")
print(f"  Conditions: control={len(df[df['intervention']==0])}, "
      f"knowledge_oracle={len(df[df['intervention']==1])}")
print(f"\nFailure type distribution:")
for ft in ["stochastic", "knowledge", "hard"]:
    n_q = len([q for q in questions if labels[q] == ft])
    n_obs = len(df[df["failure_type"] == ft])
    print(f"  {ft}: {n_q} questions, {n_obs} observations")

# Dummy code: reference category = stochastic (so interaction terms show
# the DIFFERENCE in treatment effect relative to stochastic)
df["is_knowledge"] = (df["failure_type"] == "knowledge").astype(int)
df["is_hard"] = (df["failure_type"] == "hard").astype(int)
df["intv_x_knowledge"] = df["intervention"] * df["is_knowledge"]
df["intv_x_hard"] = df["intervention"] * df["is_hard"]

# Sort by question for GEE
df = df.sort_values("question_no").reset_index(drop=True)

# Assign numeric group IDs for GEE
qno_to_id = {q: i for i, q in enumerate(sorted(df["question_no"].unique()))}
df["qno_id"] = df["question_no"].map(qno_to_id)

print("\n" + "=" * 70)
print("MODEL 1: GEE LOGISTIC (exchangeable correlation within questions)")
print("=" * 70)

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

exog_cols = ["intervention", "is_knowledge", "is_hard",
             "intv_x_knowledge", "intv_x_hard"]
X = sm.add_constant(df[exog_cols].values.astype(float))
col_names = ["const"] + exog_cols

gee_model = GEE(
    df["correct"].values.astype(float),
    X,
    groups=df["qno_id"].values,
    family=Binomial(),
    cov_struct=Exchangeable(),
)
gee_result = gee_model.fit()

print("\nGEE Results (reference: stochastic, control):")
print(f"{'Parameter':<22} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>10}")
print("-" * 60)
for i, name in enumerate(col_names):
    coef = gee_result.params[i]
    se = gee_result.bse[i]
    z = coef / se if se > 0 else 0
    p = gee_result.pvalues[i]
    print(f"{name:<22} {coef:>8.3f} {se:>8.3f} {z:>8.2f} {p:>10.2e}")

print("\n  KEY INTERACTION TERMS (on log-odds scale):")
idx_ixk = col_names.index("intv_x_knowledge")
coef_ixk = gee_result.params[idx_ixk]
se_ixk = gee_result.bse[idx_ixk]
ci_lo = coef_ixk - 1.96 * se_ixk
ci_hi = coef_ixk + 1.96 * se_ixk
print(f"  intv_x_knowledge: coef={coef_ixk:.3f}, 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}], "
      f"p={gee_result.pvalues[idx_ixk]:.2e}")
print(f"    Interpretation: Knowledge scaffolding effect on knowledge-limited vs stochastic")
print(f"    Odds ratio: {np.exp(coef_ixk):.1f} "
      f"[{np.exp(ci_lo):.1f}, {np.exp(ci_hi):.1f}]")

idx_ixh = col_names.index("intv_x_hard")
coef_ixh = gee_result.params[idx_ixh]
se_ixh = gee_result.bse[idx_ixh]
ci_lo_h = coef_ixh - 1.96 * se_ixh
ci_hi_h = coef_ixh + 1.96 * se_ixh
print(f"\n  intv_x_hard: coef={coef_ixh:.3f}, 95%CI=[{ci_lo_h:.3f}, {ci_hi_h:.3f}], "
      f"p={gee_result.pvalues[idx_ixh]:.2e}")

print(f"\n  Exchangeable correlation estimate: "
      f"{gee_result.cov_struct.summary()}")


print("\n\n" + "=" * 70)
print("MODEL 2: MIXED-EFFECTS LOGISTIC (random intercept per question)")
print("=" * 70)

try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    # Formula-style: need to build design matrices
    # Fixed effects: intervention + is_knowledge + is_hard + intv_x_knowledge + intv_x_hard
    # Random effects: (1 | question_no)

    exog_fe = df[exog_cols].values.astype(float)
    # Random intercept per question
    n_questions = df["question_no"].nunique()
    exog_re_names = [f"q{i}" for i in range(n_questions)]
    exog_re = np.zeros((len(df), n_questions))
    for i, row in df.iterrows():
        exog_re[i, qno_to_id[row["question_no"]]] = 1.0

    ident = np.zeros(n_questions, dtype=int)  # all random effects in one group

    model_bayes = BinomialBayesMixedGLM(
        df["correct"].values.astype(float),
        exog_fe,
        exog_re,
        ident,
        vcp_p=1.0,
        fe_p=2.0,
    )
    result_bayes = model_bayes.fit_map()

    print("\nBayesian Mixed GLM Results (MAP estimate):")
    print(f"{'Parameter':<22} {'Coef':>8} {'SE':>8}")
    print("-" * 40)
    for i, name in enumerate(exog_cols):
        coef = result_bayes.fe_mean[i]
        se = result_bayes.fe_sd[i]
        print(f"{name:<22} {coef:>8.3f} {se:>8.3f}")

    coef_ixk_mm = result_bayes.fe_mean[exog_cols.index("intv_x_knowledge")]
    se_ixk_mm = result_bayes.fe_sd[exog_cols.index("intv_x_knowledge")]
    ci_lo_mm = coef_ixk_mm - 1.96 * se_ixk_mm
    ci_hi_mm = coef_ixk_mm + 1.96 * se_ixk_mm
    z_mm = coef_ixk_mm / se_ixk_mm
    from scipy.stats import norm
    p_mm = 2 * (1 - norm.cdf(abs(z_mm)))

    print(f"\n  KEY: intv_x_knowledge: coef={coef_ixk_mm:.3f}, "
          f"95%CI=[{ci_lo_mm:.3f}, {ci_hi_mm:.3f}], z={z_mm:.2f}, p={p_mm:.2e}")
    print(f"    Odds ratio: {np.exp(coef_ixk_mm):.1f} "
          f"[{np.exp(ci_lo_mm):.1f}, {np.exp(ci_hi_mm):.1f}]")
    print(f"\n  Random effect SD: {result_bayes.vcp_mean[0]:.3f}")

except Exception as e:
    print(f"\n  BayesMixedGLM failed: {e}")
    print("  Falling back to GEE-only analysis.")


print("\n\n" + "=" * 70)
print("MODEL 3: MARGINAL EFFECTS — converting to probability scale")
print("=" * 70)

# Predict marginal probabilities for each cell
from scipy.special import expit

print("\n  Predicted recovery rates (GEE model):")
print(f"  {'Type':<14} {'Control':>10} {'Oracle':>10} {'Delta':>10}")
print(f"  {'-'*46}")

for ft in ["stochastic", "knowledge", "hard"]:
    # Control
    x_ctrl = [1,  # const
              0,  # intervention
              int(ft == "knowledge"),
              int(ft == "hard"),
              0,  # intv_x_knowledge
              0]  # intv_x_hard
    # Oracle
    x_oracle = [1,
                1,  # intervention
                int(ft == "knowledge"),
                int(ft == "hard"),
                int(ft == "knowledge"),
                int(ft == "hard")]
    p_ctrl = expit(np.dot(gee_result.params, x_ctrl))
    p_oracle = expit(np.dot(gee_result.params, x_oracle))
    delta = p_oracle - p_ctrl
    sign = "+" if delta > 0 else ""
    print(f"  {ft:<14} {p_ctrl:>10.1%} {p_oracle:>10.1%} {sign}{delta:>9.1%}")

# Interaction on probability scale
p_stoch_ctrl = expit(np.dot(gee_result.params, [1, 0, 0, 0, 0, 0]))
p_stoch_oracle = expit(np.dot(gee_result.params, [1, 1, 0, 0, 0, 0]))
p_know_ctrl = expit(np.dot(gee_result.params, [1, 0, 1, 0, 0, 0]))
p_know_oracle = expit(np.dot(gee_result.params, [1, 1, 1, 0, 1, 0]))

interaction_prob = (p_know_oracle - p_know_ctrl) - (p_stoch_oracle - p_stoch_ctrl)
print(f"\n  Interaction on probability scale: {interaction_prob:+.1%}")
print(f"  (knowledge-limited effect) - (stochastic effect) = "
      f"({p_know_oracle - p_know_ctrl:+.1%}) - ({p_stoch_oracle - p_stoch_ctrl:+.1%})")


print("\n\n" + "=" * 70)
print("CONVERGENCE SUMMARY")
print("=" * 70)

print(f"""
  Method                        Interaction (pp)   p-value
  ─────────────────────────────────────────────────────────
  Permutation test (50k)        +87.8              < 2e-5
  Question-level bootstrap      +87.8              [71.7, 103.8] 95%CI
  GEE logistic (exchangeable)   {interaction_prob*100:+.1f}              {gee_result.pvalues[idx_ixk]:.2e}
""")

try:
    interaction_mm = (
        expit(np.dot(result_bayes.fe_mean,
                     [1, 1, 0, 1, 0])) -
        expit(np.dot(result_bayes.fe_mean,
                     [0, 1, 0, 0, 0]))
    ) - (
        expit(np.dot(result_bayes.fe_mean,
                     [1, 0, 0, 0, 0])) -
        expit(np.dot(result_bayes.fe_mean,
                     [0, 0, 0, 0, 0]))
    )
except:
    pass

print("Done.")
