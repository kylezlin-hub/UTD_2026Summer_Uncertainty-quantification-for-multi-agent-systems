"""compare_llama_qwen.py -- cross-model comparison of Llama-3.1-8B vs Qwen2.5-7B on the SAME
300-question fixture, for the intervention conditions both models share.

Qwen: surviving per-question aggregate rates (full_*_rate) from the taxonomy CSVs (pre-loss
      reference; old single-brief format).
Llama: per-question rate = mean correctness over the 8 measurements/condition from
      interventions_llama/solve_results.jsonl (new 4-brief format).

Shared conditions: control, knowledge_blind (C), reasoning (R), both_blind, knowledge_oracle.
(Llama lacks knowledge_blind_stem (S); S + full S-based phenotype taxonomy are Qwen-only.)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IV = HERE / "interventions"

COND = [  # display, llama-condition, qwen-column
    ("control",              "control",          "full_control_rate"),
    ("C (choice-cond info)", "knowledge_blind",  "full_knowledge_blind_rate"),
    ("R (reasoning)",        "reasoning",        "full_reasoning_rate"),
    ("both_blind",           "both_blind",       "full_both_blind_rate"),
    ("oracle (knowledge)",   "knowledge_oracle", "full_knowledge_oracle_rate"),
]

# Llama per-question per-condition rate
L = pd.DataFrame([json.loads(l) for l in (HERE / "interventions_llama" / "solve_results.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip()])
L["question_no"] = L["question_no"].astype(str)
lam = L.groupby(["question_no", "condition"])["correct"].mean().unstack("condition")

# Qwen surviving per-question rates (137 failures + 163 stable = 300)
qwen = pd.concat([pd.read_csv(IV / "taxonomy_nested_results.csv"),
                  pd.read_csv(IV / "taxonomy_nested_baseline_stable.csv")], ignore_index=True)
qwen["question_no"] = qwen["question_no"].astype(str)
qwen = qwen.set_index("question_no")

# build aligned per-model rate tables (columns = display names)
ll = pd.DataFrame({d: lam[lc] for d, lc, qc in COND})
qq = pd.DataFrame({d: qwen[qc] for d, lc, qc in COND})
common = sorted(set(ll.index) & set(qq.index))
ll, qq = ll.loc[common], qq.loc[common]
print(f"Common questions: {len(common)}\n")

def report(name, df):
    c = df["control"]
    print(f"===== {name} (n={len(df)}) =====")
    print(f"  baseline control accuracy: mean={c.mean():.3f} | failure rate control==0: "
          f"{(c==0).mean()*100:.1f}% | majority-fail control<0.5: {(c<0.5).mean()*100:.1f}%")
    print(f"  {'condition':<22}{'mean acc':>9}{'Δ vs ctrl':>11}")
    for d, *_ in COND:
        print(f"  {d:<22}{df[d].mean():>9.3f}{(df[d]-c).mean():>+11.3f}")
    bc = c == 1.0
    print(f"  harm on baseline-correct (control==1.0, n={int(bc.sum())}):")
    for d, *_ in COND[1:]:
        print(f"     {d:<20} Δacc={(df.loc[bc, d]-1.0).mean():+.3f}")
    print()
    return c, bc

cl, bcl = report("LLAMA-3.1-8B", ll)
cq, bcq = report("QWEN-2.5-7B (existing/pre-loss)", qq)

print("===== HEAD-TO-HEAD (same 300 questions) =====")
print(f"  {'metric':<26}{'Llama':>9}{'Qwen':>9}{'Δ(L-Q)':>9}")
print(f"  {'baseline control acc':<26}{cl.mean():>9.3f}{cq.mean():>9.3f}{cl.mean()-cq.mean():>+9.3f}")
for d, *_ in COND[1:]:
    dl, dq = (ll[d]-cl).mean(), (qq[d]-cq).mean()
    print(f"  {'Δ '+d:<26}{dl:>+9.3f}{dq:>+9.3f}{dl-dq:>+9.3f}")
print(f"\n  harm on baseline-correct (mean Δacc over intervention conds, control==1.0):")
hl = np.mean([(ll.loc[bcl, d]-1.0).mean() for d, *_ in COND[1:]])
hq = np.mean([(qq.loc[bcq, d]-1.0).mean() for d, *_ in COND[1:]])
print(f"     Llama {hl:+.3f} (n={int(bcl.sum())})   vs   Qwen {hq:+.3f} (n={int(bcq.sum())})")
