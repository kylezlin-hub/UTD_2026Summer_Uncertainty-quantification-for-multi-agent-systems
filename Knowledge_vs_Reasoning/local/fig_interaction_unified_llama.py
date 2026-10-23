"""fig_interaction_unified_llama.py -- structural interaction in the unified 5-way framework
(Llama-3.1-8B): held-out oracle recovery for informational-responsive vs. persistent failures,
both reciprocal cross-fit directions. Llama sibling of fig_interaction_unified_qwen4brief.py --
same design, same contrast, so the two model figures are directly comparable.

Loading matches interaction_unified_llama.py (patched): the stem-only solves live in a SEPARATE
file (interventions_llama_fixed/stem_only_solve_results.jsonl, field "rep", condition
"knowledge_blind_stem") and MUST be joined, else gS is always negative and information-responsive/
discordant can never be assigned. Categories built from S/C/R only; oracle used solely as the
downstream outcome (non-circular). GEE recomputed in-script so the footnote is self-consistent.

Data: interventions_llama_fixed/solve_results.jsonl, interventions_llama_fixed/stem_only_solve_results.jsonl,
      rescreen_llama8b/phase1_matched_labels_k3.csv.
Output: interventions/results/fig_interaction_unified_llama.{png,pdf}
Usage: python fig_interaction_unified_llama.py
"""
import json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

HERE = Path(__file__).resolve().parent
SOLVE = HERE / "interventions_llama_fixed" / "solve_results.jsonl"
STEM = HERE / "interventions_llama_fixed" / "stem_only_solve_results.jsonl"
SCR = HERE / "rescreen_llama8b" / "phase1_matched_labels_k3.csv"
OUT = HERE / "interventions" / "results"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.RandomState(0)

# ---- load exactly as interaction_unified_llama.py (patched) ----
data = defaultdict(lambda: defaultdict(list))
for l in SOLVE.read_text(encoding="utf-8").splitlines():
    if not l.strip(): continue
    r = json.loads(l); data[str(r["question_no"])][r["condition"]].append((int(r["repeat"]), int(r["correct"])))
if STEM.exists():
    for l in STEM.read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        r = json.loads(l); data[str(r["question_no"])][r["condition"]].append((int(r["rep"]), int(r["correct"])))
scr = pd.read_csv(SCR); scr["question_no"] = scr["question_no"].astype(str)
STABLE = set(scr.loc[scr.n_correct == scr.k, "question_no"])
FAIL = [q for q in data if q not in STABLE]

def rate(q, c, reps):
    xs = [v for rep, v in data[q][c] if rep in reps]
    return np.mean(xs) if xs else 0.0

def five_rule(q, reps, thr=0.5):
    ctrl = rate(q, "control", reps)
    s = rate(q, "knowledge_blind_stem", reps) - ctrl >= thr
    c = rate(q, "knowledge_blind", reps) - ctrl >= thr
    r = rate(q, "reasoning", reps) - ctrl >= thr
    if s and c: return "information-responsive"
    if (not s) and c: return "choice-conditioned"
    if s and (not c): return "discordant"
    return "reasoning-responsive" if r else "persistent"

INFO = {"information-responsive", "choice-conditioned", "discordant"}

def boot_ci(vals):
    vals = np.array(vals, float)
    if len(vals) < 2: return (np.nan, np.nan)
    m = np.mean(RNG.choice(vals, size=(5000, len(vals)), replace=True), axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def gee_interaction(labs, out):
    use = [q for q in FAIL if labs[q] in INFO or labs[q] == "persistent"]
    rows = []
    for q in use:
        infoflag = 1 if labs[q] in INFO else 0
        for cond in ["control", "knowledge_oracle"]:
            for rep, v in data[q][cond]:
                if rep in out:
                    rows.append({"q": q, "correct": v, "intv": 1 if cond == "knowledge_oracle" else 0,
                                 "informational": infoflag})
    df = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)
    df["gid"] = pd.factorize(df["q"])[0]; df["intv_x_info"] = df.intv * df.informational
    X = sm.add_constant(df[["intv", "informational", "intv_x_info"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GEE(df["correct"], X, groups=df["gid"], family=Binomial(), cov_struct=Exchangeable()).fit()
    return res.pvalues["intv_x_info"], np.exp(res.params["intv_x_info"])

# ---- build figure ----
A, B = {0, 1, 2, 3}, {4, 5, 6, 7}
DIRS = [("A→B", A, B), ("B→A", B, A)]
CATS = [("informational-responsive", INFO, "#0072B2"), ("persistent", {"persistent"}, "#8a8a8a")]

dir_labs, dir_stats, dir_n = {}, {}, {}
for dname, lab, out in DIRS:
    labs = {q: five_rule(q, lab) for q in FAIL}
    dir_labs[dname] = (labs, out)
    dir_stats[dname] = gee_interaction(labs, out)
    dir_n[dname] = (sum(1 for q in FAIL if labs[q] in INFO),
                    sum(1 for q in FAIL if labs[q] == "persistent"))

fig, ax = plt.subplots(figsize=(7.6, 5.0))
xpos = np.arange(len(DIRS)); w = 0.36
for i, (cat_lbl, catset, color) in enumerate(CATS):
    means, los, his = [], [], []
    for dname, lab, out in DIRS:
        labs, _ = dir_labs[dname]
        qs = [q for q in FAIL if labs[q] in catset]
        deltas = [rate(q, "knowledge_oracle", out) - rate(q, "control", out) for q in qs]
        m = np.mean(deltas) * 100; lo, hi = [x * 100 for x in boot_ci(deltas)]
        means.append(m); los.append(m - lo); his.append(hi - m)
    ax.bar(xpos + (i - 0.5) * w, means, w, yerr=[los, his], capsize=4, color=color,
           edgecolor="white", label=cat_lbl)
    for xx, mm in zip(xpos + (i - 0.5) * w, means):
        ax.annotate(f"{mm:+.0f}", (xx, mm), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9, fontweight="bold", color=color)
ax.axhline(0, color="#222", lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels([f"{d[0]}\n(labels→outcomes)" for d in DIRS], fontsize=9)
ax.set_ylabel("Held-out oracle recovery Δ vs. control (pp)", fontsize=10)
ax.set_title("Failure-dependent intervention effect (unified 5-way framework, held-out & reciprocal)",
             fontsize=10.5, fontweight="bold", loc="left")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", color="#ddd", lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
(pab, oab), (pba, oba) = dir_stats["A→B"], dir_stats["B→A"]
nab, nba = dir_n["A→B"], dir_n["B→A"]
fig.text(0.5, -0.02,
         "Llama-3.1-8B, baseline-failure cohort (baseline-stable excluded). Category from the 5-way rule on the "
         "label split; oracle recovery on the held-out split (non-circular).\n"
         f"GEE intv×informational: A→B p={pab:.1e} (OR {oab:.1f}×, n_info={nab[0]}/pers={nab[1]}), "
         f"B→A p={pba:.1e} (OR {oba:.1f}×, n_info={nba[0]}/pers={nba[1]}). Error bars 95% question bootstrap.",
         ha="center", va="top", fontsize=7.4, color="#555")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_interaction_unified_llama.{ext}", dpi=200, bbox_inches="tight")
print(f"-> {OUT / 'fig_interaction_unified_llama.png'}")
print(f"   A->B: p={pab:.3g} OR={oab:.2f} n_info={nab[0]} n_pers={nab[1]}")
print(f"   B->A: p={pba:.3g} OR={oba:.2f} n_info={nba[0]} n_pers={nba[1]}")
