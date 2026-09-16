"""generate_fig1_composite.py -- Paper Figure 1 (3 panels):
  A. Qwen held-out S/C/R effects by operational (5-way) category.
  B. Llama held-out S/C/R effects by operational (5-way) category.
  C. Out-of-classification oracle test: informational-response set vs persistent, Qwen & Llama.

Panels show the A->B held-out direction (labels on reps 0-3, effects on the disjoint reps 4-7);
the reciprocal B->A, category n's, GEE OR/CI/p and bootstrap CIs are printed to stdout for the
caption. Categories are built from S/C/R only (oracle excluded) -> Panel C is out-of-classification.
Stem-only solves are joined from the SEPARATE stem file (see five-way-stem-join-bug).

Output: interventions/results/fig1_composite.{png,pdf}
Usage: python generate_fig1_composite.py
"""
import json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

HERE = Path(__file__).resolve().parent
OUT = HERE / "interventions" / "results"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.RandomState(0)

MODELS = {
    "Qwen2.5-7B": dict(
        solve=HERE / "interventions" / "solve_results.jsonl",
        stem=HERE / "interventions" / "stem_only_solve_results.jsonl",
        scr=HERE / "rescreen" / "phase1_matched_labels_k3.csv",
        main_rep="_derive"),           # main file has brief_instance/solver_draw
    "Llama-3.1-8B": dict(
        solve=HERE / "interventions_llama_fixed" / "solve_results.jsonl",
        stem=HERE / "interventions_llama_fixed" / "stem_only_solve_results.jsonl",
        scr=HERE / "rescreen_llama8b" / "phase1_matched_labels_k3.csv",
        main_rep="repeat"),            # main file has repeat
}

def _derive(r):
    if "repeat" in r and r["repeat"] is not None: return int(r["repeat"])
    bi = int(r.get("brief_instance", 0) or 0); sd = int(r.get("solver_draw", 1))
    return (bi - 1) * 2 + (sd - 1) if bi >= 1 else (sd - 1)

def load(cfg):
    data = defaultdict(lambda: defaultdict(list))
    for l in Path(cfg["solve"]).read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        rep = _derive(r) if cfg["main_rep"] == "_derive" else int(r["repeat"])
        data[str(r["question_no"])][r["condition"]].append((rep, int(r["correct"])))
    for l in Path(cfg["stem"]).read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        data[str(r["question_no"])][r["condition"]].append((int(r["rep"]), int(r["correct"])))
    scr = pd.read_csv(cfg["scr"]); scr["question_no"] = scr["question_no"].astype(str)
    stable = set(scr.loc[scr.n_correct == scr.k, "question_no"])
    fail = [q for q in data if q not in stable]
    return data, fail

def rate(data, q, c, reps):
    xs = [v for rep, v in data[q][c] if rep in reps]
    return np.mean(xs) if xs else 0.0

def five_rule(data, q, reps, thr=0.5):
    ctrl = rate(data, q, "control", reps)
    s = rate(data, q, "knowledge_blind_stem", reps) - ctrl >= thr
    c = rate(data, q, "knowledge_blind", reps) - ctrl >= thr
    r = rate(data, q, "reasoning", reps) - ctrl >= thr
    if s and c: return "information-responsive"
    if (not s) and c: return "choice-conditioned"
    if s and (not c): return "discordant"
    return "reasoning-responsive" if r else "persistent"

INFO = {"information-responsive", "choice-conditioned", "discordant"}
A, B = {0, 1, 2, 3}, {4, 5, 6, 7}

def boot_ci(vals):
    vals = np.array(vals, float)
    if len(vals) < 2: return (np.nan, np.nan)
    m = np.mean(RNG.choice(vals, size=(5000, len(vals)), replace=True), axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def gee(data, fail, labs, out):
    use = [q for q in fail if labs[q] in INFO or labs[q] == "persistent"]
    rows = []
    for q in use:
        f = 1 if labs[q] in INFO else 0
        for cond in ["control", "knowledge_oracle"]:
            for rep, v in data[q][cond]:
                if rep in out:
                    rows.append({"q": q, "y": v, "intv": 1 if cond == "knowledge_oracle" else 0, "info": f})
    df = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)
    df["gid"] = pd.factorize(df["q"])[0]; df["ix"] = df["intv"] * df["info"]
    X = sm.add_constant(df[["intv", "info", "ix"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GEE(df["y"], X, groups=df["gid"], family=Binomial(), cov_struct=Exchangeable()).fit()
    b, se, p = res.params["ix"], res.bse["ix"], res.pvalues["ix"]
    return dict(coef=b, se=se, p=p, orr=np.exp(b),
                or_lo=np.exp(b - 1.96 * se), or_hi=np.exp(b + 1.96 * se))

# ---- compute everything ----
CATS = ["information-responsive", "choice-conditioned", "discordant", "reasoning-responsive", "persistent"]
CAT_SHORT = ["Information", "Choice-cond.", "Discordant", "Reasoning", "Persistent"]
SCR_CONDS = [("knowledge_blind_stem", "S", "#0072B2"), ("knowledge_blind", "C", "#E69F00"),
             ("reasoning", "R", "#009E73")]

def qwen_singlebrief_store():
    """Qwen panels on the single-brief fixed-scaffold design (coherent with Fig 2):
    Panel A held-out delta_{S,C,R}_heldout_B by label_from_A (from taxonomy CSV);
    Panel C oracle held-out on Split-B (reps 4-7) from the regenerated single-brief oracle
    minus the brief-free control, grouped by label_from_A INFO vs persistent, with a fresh GEE."""
    tax = pd.read_csv(HERE / "interventions" / "taxonomy_nested_results.csv")
    tax["question_no"] = tax["question_no"].astype(str)
    LABMAP = {"information-responsive": "information-responsive",
              "choice-conditioned responsive": "choice-conditioned",
              "discordant information response": "discordant",
              "reasoning-responsive": "reasoning-responsive", "persistent": "persistent"}
    tax["cat"] = tax["label_from_A"].map(LABMAP)
    panel = {}
    for cat in CATS:
        s = tax[tax["cat"] == cat]
        panel[cat] = dict(n=len(s), deltas={
            "S": s["delta_S_heldout_B"].mean() if len(s) else np.nan,
            "C": s["delta_C_heldout_B"].mean() if len(s) else np.nan,
            "R": s["delta_R_heldout_B"].mean() if len(s) else np.nan})
    # oracle (single-brief regen) and brief-free control, indexed by rep 0-7
    orc = defaultdict(dict)
    for l in (HERE / "interventions" / "oracle_singlebrief_solve_results.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        d = json.loads(l); orc[str(d["question_no"])][int(d["rep"])] = int(d["correct"])
    # within-run brief-free control (same single-brief solver setup as the oracle regen)
    ctl = defaultdict(dict)
    for l in (HERE / "interventions" / "control_singlebrief_solve_results.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        d = json.loads(l)
        ctl[str(d["question_no"])][int(d["rep"])] = int(d["correct"])
    def r_(dct, q): vs = [dct[q][r] for r in B if r in dct[q]]; return np.mean(vs) if vs else np.nan
    cpanel = {}
    for grp, sset in [("informational", INFO), ("persistent", {"persistent"})]:
        qs = tax[tax["cat"].isin(sset)]["question_no"].tolist()
        dd = [r_(orc, q) - r_(ctl, q) for q in qs]
        lo, hi = boot_ci(dd); cpanel[grp] = dict(n=len(qs), mean=np.mean(dd) * 100, lo=lo * 100, hi=hi * 100)
    def gee_sb(labcol, reps):
        rows = []
        for _, rr in tax.iterrows():
            q = rr["question_no"]; cat = LABMAP.get(rr[labcol])
            f = 1 if cat in INFO else (0 if cat == "persistent" else None)
            if f is None: continue
            for rp in reps:
                if rp in ctl[q]: rows.append({"q": q, "y": ctl[q][rp], "intv": 0, "info": f})
                if rp in orc[q]: rows.append({"q": q, "y": orc[q][rp], "intv": 1, "info": f})
        df = pd.DataFrame(rows).sort_values("q"); df["gid"] = pd.factorize(df["q"])[0]
        df["ix"] = df["intv"] * df["info"]
        X = sm.add_constant(df[["intv", "info", "ix"]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = GEE(df["y"], X, groups=df["gid"], family=Binomial(), cov_struct=Exchangeable()).fit()
        bx, sx = res.params["ix"], res.bse["ix"]
        return dict(coef=bx, se=sx, p=res.pvalues["ix"], orr=np.exp(bx),
                    or_lo=np.exp(bx - 1.96 * sx), or_hi=np.exp(bx + 1.96 * sx))
    ni = int(tax["cat"].isin(INFO).sum()); npv = int((tax["cat"] == "persistent").sum())
    nib = int(tax["label_from_B"].map(LABMAP).isin(INFO).sum())
    npb = int((tax["label_from_B"].map(LABMAP) == "persistent").sum())
    return dict(panel=panel, cpanel=cpanel,
                gee_ab=gee_sb("label_from_A", {4, 5, 6, 7}), gee_ba=gee_sb("label_from_B", {0, 1, 2, 3}),
                n_info_ab=ni, n_pers_ab=npv, n_info_ba=nib, n_pers_ba=npb)


store = {}
for name, cfg in MODELS.items():
    if name == "Qwen2.5-7B":
        store[name] = qwen_singlebrief_store(); continue
    data, fail = load(cfg)
    labs_ab = {q: five_rule(data, q, A) for q in fail}   # panels use A->B
    # panel A/B: mean held-out (split B) delta over control, per category, per S/C/R
    panel = {}
    for cat in CATS:
        qs = [q for q in fail if labs_ab[q] == cat]
        panel[cat] = dict(n=len(qs), deltas={
            key: (np.mean([rate(data, q, cond, B) - rate(data, q, "control", B) for q in qs]) if qs else np.nan)
            for cond, key, _ in SCR_CONDS})
    # panel C: oracle delta info vs persistent (A->B, held-out B) + bootstrap CI + GEE both dirs
    labs_ba = {q: five_rule(data, q, B) for q in fail}
    cpanel = {}
    for grp, sset in [("informational", INFO), ("persistent", {"persistent"})]:
        qs = [q for q in fail if labs_ab[q] in sset]
        d = [rate(data, q, "knowledge_oracle", B) - rate(data, q, "control", B) for q in qs]
        lo, hi = boot_ci(d)
        cpanel[grp] = dict(n=len(qs), mean=np.mean(d) * 100, lo=lo * 100, hi=hi * 100)
    store[name] = dict(panel=panel, cpanel=cpanel,
                       gee_ab=gee(data, fail, labs_ab, B), gee_ba=gee(data, fail, labs_ba, A),
                       n_info_ab=sum(1 for q in fail if labs_ab[q] in INFO),
                       n_pers_ab=sum(1 for q in fail if labs_ab[q] == "persistent"),
                       n_info_ba=sum(1 for q in fail if labs_ba[q] in INFO),
                       n_pers_ba=sum(1 for q in fail if labs_ba[q] == "persistent"))

# ---- figure ----
# 2x2-style: A | B on top row, C centered (spanning middle 4 of 6 cols) on bottom row.
# C is shorter than A/B (height_ratios) and separated by extra row spacing (hspace).
fig = plt.figure(figsize=(11, 7.8), layout="constrained")
gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 0.66], hspace=0.28)
axA = fig.add_subplot(gs[0, 0:3])
axB = fig.add_subplot(gs[0, 3:6])
axC = fig.add_subplot(gs[1, 1:5])

def draw_scr(ax, name, letter):
    panel = store[name]["panel"]; x = np.arange(len(CATS)); w = 0.26
    for j, (cond, key, color) in enumerate(SCR_CONDS):
        vals = [panel[c]["deltas"][key] * 100 for c in CATS]
        ax.bar(x + (j - 1) * w, vals, w, color=color, edgecolor="white", label=key)
    ax.axhline(0, color="#222", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(n={panel[c]['n']})" for s, c in zip(CAT_SHORT, CATS)], fontsize=8.5)
    ax.set_ylabel("Held-out Δ vs. control (pp)", fontsize=9.5)
    ax.set_title(f"{letter}. {name} — held-out scaffold effects",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(title="scaffold", fontsize=8, title_fontsize=8, loc="upper right", ncol=3)
    ax.grid(axis="y", color="#ddd", lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_ylim(-25, 100)

draw_scr(axA, "Qwen2.5-7B", "A")
draw_scr(axB, "Llama-3.1-8B", "B")

# Panel C
xpos = np.arange(len(MODELS)); w = 0.36
GRPS = [("informational", "#0072B2"), ("persistent", "#8a8a8a")]
for i, (grp, color) in enumerate(GRPS):
    means = [store[m]["cpanel"][grp]["mean"] for m in MODELS]
    los = [store[m]["cpanel"][grp]["mean"] - store[m]["cpanel"][grp]["lo"] for m in MODELS]
    his = [store[m]["cpanel"][grp]["hi"] - store[m]["cpanel"][grp]["mean"] for m in MODELS]
    ns = [store[m]["cpanel"][grp]["n"] for m in MODELS]
    axC.bar(xpos + (i - 0.5) * w, means, w, yerr=[los, his], capsize=4, color=color,
            edgecolor="white", label=grp)
    for xx, mm, nn in zip(xpos + (i - 0.5) * w, means, ns):
        axC.annotate(f"{mm:+.0f}", (xx, mm), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8.5, fontweight="bold", color=color)
        axC.annotate(f"n={nn}", (xx, 0), textcoords="offset points", xytext=(0, -10),
                     ha="center", va="top", fontsize=7, color=color)
axC.axhline(0, color="#222", lw=1)
axC.set_ylim(bottom=-9)
axC.set_xticks(xpos); axC.set_xticklabels(list(MODELS.keys()), fontsize=8.5)
axC.set_ylabel("Held-out oracle Δ vs. control (pp)", fontsize=9)
axC.set_title("C. Held-out oracle gain by non-oracle response group",
              fontsize=10, fontweight="bold", loc="left")
axC.legend(fontsize=8, loc="upper right")
axC.grid(axis="y", color="#ddd", lw=0.7); axC.set_axisbelow(True)
for sp in ("top", "right"): axC.spines[sp].set_visible(False)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig1_composite.{ext}", dpi=200, bbox_inches="tight")
print(f"-> {OUT / 'fig1_composite.png'}\n")

# ---- caption material ----
print("=== CAPTION NUMBERS (A->B panels shown; B->A reciprocal below) ===")
for name in MODELS:
    s = store[name]
    print(f"\n{name}:")
    print(f"  Panel category n (A->B): " + ", ".join(f"{c}={s['panel'][c]['n']}" for c in CATS))
    for tag, g, ni, npv in [("A->B", s["gee_ab"], s["n_info_ab"], s["n_pers_ab"]),
                            ("B->A", s["gee_ba"], s["n_info_ba"], s["n_pers_ba"])]:
        print(f"  GEE {tag}: OR={g['orr']:.2f} [{g['or_lo']:.2f}, {g['or_hi']:.2f}], "
              f"coef={g['coef']:+.3f} (SE {g['se']:.3f}), p={g['p']:.3g}  (n_info={ni}, n_pers={npv})")
    c = s["cpanel"]
    print(f"  Panel C oracle Δ (A->B): informational {c['informational']['mean']:+.1f}pp "
          f"[{c['informational']['lo']:.1f}, {c['informational']['hi']:.1f}] (n={c['informational']['n']}); "
          f"persistent {c['persistent']['mean']:+.1f}pp [{c['persistent']['lo']:.1f}, {c['persistent']['hi']:.1f}] "
          f"(n={c['persistent']['n']})")
