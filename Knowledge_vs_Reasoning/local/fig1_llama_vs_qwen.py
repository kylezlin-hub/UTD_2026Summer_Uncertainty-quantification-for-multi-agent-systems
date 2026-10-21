"""fig1_llama_vs_qwen.py -- Fig 1 reproduction check (paper-definition baseline set):
does Qwen-7B's "scaffolding harms baseline-correct answers" reproduce on Llama-3.1-8B?

baseline-stable-correct := 3/3 on the K=3 matched baseline screen (paper definition):
    Qwen  -> rescreen/phase1_matched_labels.csv           (n_correct==k)  => 163
    Llama -> rescreen_llama8b/phase1_matched_labels.csv    (n_correct==k)  => 147
harm(condition) := mean over that set of (rate_condition - rate_control), 95% bootstrap CI,
plus correct->incorrect reversal counts. Same 300-question fixture; identical method per model.

Rates: Qwen from surviving full_*_rate (old single-brief); Llama from interventions_llama_fixed/
(fixed-brief: solve_results.jsonl + stem_only_solve_results.jsonl), mean over 8 draws.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
IV = HERE / "interventions"
LF = HERE / "interventions_llama_fixed"
OUT = IV / "results"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.RandomState(0)

CONDS = [  # display, llama-condition, qwen full_*_rate col
    ("S (option-blind info)", "knowledge_blind_stem", "full_knowledge_blind_stem_rate"),
    ("C (choice-cond info)",  "knowledge_blind",      "full_knowledge_blind_rate"),
    ("R (reasoning)",         "reasoning",            "full_reasoning_rate"),
    ("both_blind",            "both_blind",           "full_both_blind_rate"),
]

def jl(p):
    return pd.DataFrame([json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()])

def boot(d, n=5000):
    d = d[~np.isnan(d)]
    if len(d) < 2: return (np.nan, np.nan)
    m = np.mean(RNG.choice(d, size=(n, len(d)), replace=True), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def screen_stable(path):
    s = pd.read_csv(path); s["question_no"] = s["question_no"].astype(str)
    return set(s.loc[s["n_correct"] == s["k"], "question_no"])

# ---- baseline-stable sets from the K=3 screens ----
q_stable = screen_stable(HERE / "rescreen" / "phase1_matched_labels.csv")
l_stable = screen_stable(HERE / "rescreen_llama8b" / "phase1_matched_labels.csv")

# ---- Llama fixed-brief rates ----
L = jl(LF / "solve_results.jsonl"); L["question_no"] = L["question_no"].astype(str)
lam = L.groupby(["question_no", "condition"])["correct"].mean().unstack("condition")
Sd = jl(LF / "stem_only_solve_results.jsonl"); Sd["question_no"] = Sd["question_no"].astype(str)
lam["knowledge_blind_stem"] = Sd.groupby("question_no")["correct"].mean()

# ---- Qwen surviving rates ----
qwen = pd.concat([pd.read_csv(IV / "taxonomy_nested_results.csv"),
                  pd.read_csv(IV / "taxonomy_nested_baseline_stable.csv")], ignore_index=True)
qwen["question_no"] = qwen["question_no"].astype(str); qwen = qwen.set_index("question_no")

def harm(model, stable, ctrl_series, rate_of):
    idx = sorted(stable & set(ctrl_series.dropna().index))
    ctrl = ctrl_series.loc[idx]
    print(f"\n===== {model}: harm on baseline-stable-correct (3/3 screen, n={len(idx)}) =====")
    print(f"  {'condition':<24}{'Δacc':>8}{'95% CI':>18}{'C→I rev':>10}")
    rows = []
    for disp, lc, qc in CONDS:
        rate = rate_of(disp).loc[idx]
        d = (rate - ctrl).to_numpy(float)
        lo, hi = boot(d); rev = int((rate < 0.5).sum())
        print(f"  {disp:<24}{np.nanmean(d):>+8.3f}   [{lo:+.3f},{hi:+.3f}]{rev:>7}/{len(idx)}")
        rows.append(dict(model=model, condition=disp, n=len(idx), mean_delta=round(float(np.nanmean(d)),4),
                         ci_lo=round(lo,4), ci_hi=round(hi,4), reversals=rev))
    return pd.DataFrame(rows)

lmap = {d: lc for d, lc, qc in CONDS}
qmap = {d: qc for d, lc, qc in CONDS}
rL = harm("Llama-3.1-8B", l_stable, lam["control"], lambda d: lam[lmap[d]])
rQ = harm("Qwen-2.5-7B (existing)", q_stable, qwen["full_control_rate"], lambda d: qwen[qmap[d]])
res = pd.concat([rL, rQ], ignore_index=True)
res.to_csv(OUT / "fig1_llama_vs_qwen_harm.csv", index=False)
print(f"\n-> {OUT/'fig1_llama_vs_qwen_harm.csv'}")

# ---- comparison figure ----
labels = [c[0] for c in CONDS]; x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(8.4, 4.9))
for i, (mdl, df, color, n) in enumerate([("Qwen-7B", rQ, "#D55E00", rQ['n'].iloc[0]),
                                          ("Llama-8B", rL, "#0072B2", rL['n'].iloc[0])]):
    m = df.set_index("condition").loc[labels]
    err = np.vstack([m["mean_delta"]-m["ci_lo"], m["ci_hi"]-m["mean_delta"]])
    ax.bar(x + (i-0.5)*w, m["mean_delta"], w, yerr=err, capsize=3, color=color,
           edgecolor="white", label=f"{mdl} (n={n})")
ax.axhline(0, color="#222", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Accuracy change vs. control", fontsize=10)
ax.set_title("Fig 1 reproduction: scaffolding harms baseline-correct answers in BOTH models",
             fontsize=11, fontweight="bold", loc="left")
ax.legend(); ax.grid(axis="y", color="#ddd", lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.5, -0.02, "baseline-stable-correct := 3/3 on the K=3 matched screen (paper definition); "
         "error bars = 95% bootstrap CIs. Same 300 questions.", ha="center", va="top",
         fontsize=7.5, color="#555")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig1_llama_vs_qwen.{ext}", dpi=200, bbox_inches="tight")
print(f"-> {OUT/'fig1_llama_vs_qwen.png'}")
