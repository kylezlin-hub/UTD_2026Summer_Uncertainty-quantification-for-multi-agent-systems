"""generate_fig4_regeneration.py -- Paper Figure 4 (two panels), regenerated-scaffold (C) result:
  A. Fixed-C vs regenerated-C five-way classification reproducibility (slope plot): Qwen drops
     sharply, Llama nearly flat (model-dependent). Annotated with paired McNemar p per model.
  B. What remains stable under regenerated C: exact-route < broad-informational stability, and
     persistent non-response is comparatively stable.

Fixed-scaffold agreement from the taxonomy stats JSONs (single-brief); regenerated agreement,
group stability and the paired McNemar recomputed here from the 4-brief x 2-solver data
(C-briefs {1,2} vs {3,4}; all axes split; only C is regenerated, S single-brief, R fixed).

Output: interventions/results/fig4_regeneration.{png,pdf}
Usage: python generate_fig4_regeneration.py
"""
import json, collections
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.stats.contingency_tables import mcnemar

HERE = Path(__file__).resolve().parent
OUT = HERE / "interventions" / "results"; OUT.mkdir(parents=True, exist_ok=True)
CATS = ["information-responsive", "choice-conditioned", "discordant", "reasoning-responsive", "persistent"]
INFO = {"information-responsive", "choice-conditioned", "discordant"}
COLM = {"Qwen2.5-7B": "#0072B2", "Llama-3.1-8B": "#D55E00"}
CFG = {
 "Qwen2.5-7B": dict(main="interventions/solve_results.jsonl", stem="interventions/stem_only_solve_results.jsonl",
                    scr="rescreen/phase1_matched_labels_k3.csv", fixstats="interventions/taxonomy_nested_agreement_stats.json",
                    fixtab="interventions/taxonomy_nested_results.csv"),
 "Llama-3.1-8B": dict(main="interventions_llama/solve_results.jsonl", stem="interventions_llama_fixed/stem_only_solve_results.jsonl",
                      scr="rescreen_llama8b/phase1_matched_labels_k3.csv", fixstats="interventions_llama_fixed/taxonomy_nested_agreement_stats.json",
                      fixtab="interventions_llama_fixed/taxonomy_nested_results.csv"),
}

def five(gS, gC, gR, thr=0.5):
    s, c, r = gS >= thr, gC >= thr, gR >= thr
    if s and c: return "information-responsive"
    if (not s) and c: return "choice-conditioned"
    if s and (not c): return "discordant"
    return "reasoning-responsive" if r else "persistent"

def half(lst, which):
    v = [c for _, c in sorted(lst)]; return np.mean(v[:4]) if which == 0 else np.mean(v[4:])

R = {}
for name, cfg in CFG.items():
    fx = collections.defaultdict(list); cbr = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in Path(cfg["main"]).read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        d = json.loads(l); q = str(d["question_no"]); c = d["condition"]; sd = int(d.get("solver_draw", 1))
        if c in ("control", "reasoning"): fx[(q, c)].append((sd, int(d["correct"])))
        if c == "knowledge_blind": cbr[q][int(d["brief_instance"])].append(int(d["correct"]))
    stem = collections.defaultdict(list)
    for l in Path(cfg["stem"]).read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        d = json.loads(l); stem[str(d["question_no"])].append((int(d.get("rep", 0)), int(d["correct"])))
    scr = pd.read_csv(cfg["scr"]); scr["question_no"] = scr["question_no"].astype(str)
    stable = set(scr.loc[scr.n_correct == scr.k, "question_no"]); fail = [q for q in cbr if q not in stable]
    la, lb = {}, {}
    for q in fail:
        cA, cB = half(fx[(q, "control")], 0), half(fx[(q, "control")], 1)
        la[q] = five(half(stem[q], 0) - cA, np.mean(cbr[q][1] + cbr[q][2]) - cA, half(fx[(q, "reasoning")], 0) - cA)
        lb[q] = five(half(stem[q], 1) - cB, np.mean(cbr[q][3] + cbr[q][4]) - cB, half(fx[(q, "reasoning")], 1) - cB)
    reg_agree = {q: int(la[q] == lb[q]) for q in fail}
    reg = np.mean(list(reg_agree.values())) * 100
    # group stability (half1 -> half2)
    info_q = [q for q in fail if la[q] in INFO]
    exact = np.mean([lb[q] == la[q] for q in info_q]) * 100
    remain = np.mean([lb[q] in INFO for q in info_q]) * 100
    pers_q = [q for q in fail if la[q] == "persistent"]
    perstab = np.mean([lb[q] == "persistent" for q in pers_q]) * 100
    # fixed agreement + paired McNemar (fixed labels_agree vs regenerated)
    st = json.loads(Path(cfg["fixstats"]).read_text(encoding="utf-8")); fixed = st["raw_agreement_real"] * 100
    tab = pd.read_csv(cfg["fixtab"]); tab["question_no"] = tab["question_no"].astype(str)
    fa = dict(zip(tab["question_no"], tab["labels_agree"].astype(int)))
    common = [q for q in fail if q in fa]
    F = np.array([fa[q] for q in common]); G = np.array([reg_agree[q] for q in common])
    a = int(((F == 1) & (G == 1)).sum()); b = int(((F == 1) & (G == 0)).sum())
    c = int(((F == 0) & (G == 1)).sum()); d = int(((F == 0) & (G == 0)).sum())
    pmc = mcnemar([[a, b], [c, d]], exact=True).pvalue
    R[name] = dict(fixed=fixed, reg=reg, pmc=pmc, exact=exact, remain=remain, perstab=perstab,
                   n_info=len(info_q), n_pers=len(pers_q))

# ---------------- figure ----------------
fig = plt.figure(figsize=(11, 4.8), layout="constrained")
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.25])
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

# Panel A: slope plot
for name in CFG:
    y = [R[name]["fixed"], R[name]["reg"]]
    axA.plot([0, 1], y, "-o", color=COLM[name], lw=2.4, ms=9, label=name)
    axA.annotate(f"{y[0]:.1f}%", (0, y[0]), textcoords="offset points", xytext=(-8, 6), ha="right", fontsize=9, color=COLM[name], fontweight="bold")
    axA.annotate(f"{y[1]:.1f}%", (1, y[1]), textcoords="offset points", xytext=(8, 6), ha="left", fontsize=9, color=COLM[name], fontweight="bold")
    p = R[name]["pmc"]; ptxt = f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.2f}"
    axA.annotate(f"McNemar {ptxt}", (1, y[1]), textcoords="offset points", xytext=(8, -12), ha="left", fontsize=7.6, color=COLM[name])
axA.set_xticks([0, 1]); axA.set_xticklabels(["Fixed C\n(single brief)", "Regenerated C\n(4 briefs)"], fontsize=9)
axA.set_xlim(-0.45, 1.65); axA.set_ylim(55, 100)
axA.set_ylabel("Exact five-way agreement (%)", fontsize=9.5)
axA.set_title("A. Fixed- vs regenerated-C reproducibility", fontsize=9.5, fontweight="bold", loc="left")
axA.legend(fontsize=8.5, loc="lower left")
axA.grid(axis="y", color="#ddd", lw=0.7); axA.set_axisbelow(True)
for sp in ("top", "right"): axA.spines[sp].set_visible(False)

# Panel B: grouped bars, 3 measures x 2 models
MEAS = [("Same informational\nlabel (route)", "exact"), ("Remains\ninformational", "remain"),
        ("Persistent remains\npersistent", "perstab")]
x = np.arange(len(MEAS)); w = 0.38
for i, name in enumerate(CFG):
    vals = [R[name][k] for _, k in MEAS]
    axB.bar(x + (i - 0.5) * w, vals, w, color=COLM[name], edgecolor="white", label=name)
    for xx, vv in zip(x + (i - 0.5) * w, vals):
        axB.annotate(f"{vv:.1f}", (xx, vv), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, fontweight="bold", color=COLM[name])
axB.set_xticks(x); axB.set_xticklabels([m for m, _ in MEAS], fontsize=8.3)
axB.set_ylim(0, 100)
axB.set_ylabel("Stability under regenerated C (%)", fontsize=9.5)
axB.set_title("B. What remains stable under regenerated C", fontsize=9.5, fontweight="bold", loc="left")
axB.legend(fontsize=8.5, loc="upper left")
axB.grid(axis="y", color="#ddd", lw=0.7); axB.set_axisbelow(True)
for sp in ("top", "right"): axB.spines[sp].set_visible(False)
axB.text(0.5, -0.30, "first two: conditional on informational-response cohort (nested); third: conditional on persistent",
         transform=axB.transAxes, ha="center", va="top", fontsize=7.3, color="#555")

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig4_regeneration.{ext}", dpi=200, bbox_inches="tight")
print(f"-> {OUT / 'fig4_regeneration.png'}")
for name in CFG:
    r = R[name]
    print(f"  {name}: fixed {r['fixed']:.1f}% -> reg {r['reg']:.1f}%  McNemar p={r['pmc']:.3g}  "
          f"| exact {r['exact']:.1f}% remain {r['remain']:.1f}% (n_info={r['n_info']}) persist {r['perstab']:.1f}% (n_pers={r['n_pers']})")
