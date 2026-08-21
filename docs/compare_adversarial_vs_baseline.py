"""Compare the adversarial ("tit for tat") debate set against the baseline set.

Answers one question: does forcing Liang-style disagreement (a) actually reduce
premature convergence, and (b) restore a process-metric -> accuracy correlation?

For each protocol it reports, per seed and pooled:
  - accuracy and round-1 consensus rate (is the "debate" actually a debate?)
  - Spearman(metric, correctness) per-debate (binary) AND per-question (aggregated
    across seeds, graded accuracy) -- the paper's estimand.
Then prints the baseline->adversarial delta on the avg-process composite.

Metrics are recomputed from the raw transcript with the SAME logic used in the
session analysis, weighting by the LLM judge's `explanation_good` when the
Reasoning_Quality sheet is present (falling back to confidence otherwise).

USAGE (from C:/Proj1 or C:/Proj1/docs):
    python compare_adversarial_vs_baseline.py \
        --baseline New/baseline_v2_mmlu-pro_s7.xlsx New/baseline_v2_mmlu-pro_s17.xlsx New/baseline_v2_mmlu-pro_s42.xlsx \
        --adversarial data/adversarial/seed_7/*.xlsx data/adversarial/seed_17/*.xlsx data/adversarial/seed_42/*.xlsx

If --adversarial paths are omitted, only the baseline block is printed (useful
before the adversarial run has finished).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from math import log

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EPS = 1e-9
METRICS = ["engagement", "responsiveness", "influence_asymmetry", "balance"]
AGENTS = ["Agent1", "Agent2", "Agent3"]


def entropy(vals: list) -> float:
    """Normalized Shannon entropy over categorical answers in one round."""
    vals = [v for v in vals if v and str(v) != "nan"]
    if len(vals) <= 1:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return float(-sum(pi * log(pi) for pi in p if pi > 0) / log(len(vals)))


def corrected_balance(disp: list[float]) -> float:
    """Corrected balance: penalize one-round collapse and oscillation."""
    n = len(disp)
    if n <= 1:
        return float("nan")
    steps = [disp[t - 1] - disp[t] for t in range(1, n)]
    pos = [max(0.0, s) for s in steps]
    total, mx = sum(pos), (max(pos) if pos else 0.0)
    if total <= EPS:
        collapse = 1.0 if max(abs(s) for s in steps) <= EPS else 0.0
    else:
        collapse = 1.0 - mx / (total + EPS)
    vol = 1.0 if n <= 2 else 1.0 - sum(steps[i - 1] * steps[i] < 0 for i in range(1, len(steps))) / (n - 2)
    return float(np.clip(collapse * vol, 0.0, 1.0))


def diagnostics(ans: list[list], qual: list[list]) -> dict:
    """Categorical process metrics from answer trajectories + quality weights."""
    n = len(ans[0])
    eng, resp = [], []
    influence = np.zeros(n)
    for t in range(1, len(ans)):
        for a in range(n):
            old, new = ans[t - 1][a], ans[t][a]
            if not old or not new:
                continue
            q = float(qual[t][a]) if qual[t][a] is not None else 0.5
            changed = old != new
            eng.append(q * float(changed))
            others = [ans[t - 1][j] for j in range(n) if j != a and ans[t - 1][j]]
            if others:
                old_sup = sum(o == old for o in others)
                new_sup = sum(o == new for o in others)
                resp.append(q * float(changed and new_sup > old_sup))
            for s in range(n):
                sa = ans[t - 1][s]
                if s == a or not sa:
                    continue
                if changed and new == sa and old != sa:
                    influence[s] += q
    total_inf = float(influence.sum())
    if total_inf > EPS and n > 1:
        p = influence / total_inf
        asym = float(1.0 - (-sum(pi * log(pi) for pi in p if pi > 0)) / log(n))
    else:
        asym = 0.0
    disp = [entropy(r) for r in ans]
    return {
        "engagement": float(np.mean(eng)) if eng else np.nan,
        "responsiveness": float(np.mean(resp)) if resp else np.nan,
        "influence_asymmetry": asym,
        "balance": corrected_balance(disp),
    }


def load_judge_quality(path: str) -> dict:
    """Map (question_no, round, agent) -> explanation_good, if the sheet exists."""
    try:
        rq = pd.read_excel(path, sheet_name="Reasoning_Quality")
    except (ValueError, KeyError, FileNotFoundError):
        return {}
    jq = {}
    for _, r in rq.iterrows():
        jq[(r["question_no"], int(r["round"]), str(r["agent"]))] = r.get("explanation_good")
    return jq


def parse_workbook(path: str) -> pd.DataFrame:
    """One row per debate with recomputed metrics, correctness, and round-1 consensus."""
    df = pd.read_excel(path, sheet_name="Debate_Traces" if _has_sheet(path, "Debate_Traces") else 0)
    rounds = sorted({int(m.group(1)) for c in df.columns if (m := re.match(r"R(\d+) Agent1 Answer", c))})
    jq = load_judge_quality(path)
    rows = []
    for _, r in df.iterrows():
        qno = r.get("Question #")
        ans, qual = [], []
        for rd in rounds:
            arow, qrow = [], []
            for ag in AGENTS:
                a = r.get(f"R{rd} {ag} Answer")
                c = r.get(f"R{rd} {ag} Conf")
                arow.append(None if pd.isna(a) else str(a).strip())
                jv = jq.get((qno, rd, ag))
                if jv is not None and not pd.isna(jv):
                    qrow.append(float(jv))
                else:
                    qrow.append(None if pd.isna(c) else float(c))
            ans.append(arow)
            qual.append(qrow)
        for a in range(3):
            for t in range(1, len(ans)):
                if not ans[t][a]:
                    ans[t][a] = ans[t - 1][a]
                if qual[t][a] is None:
                    qual[t][a] = qual[t - 1][a]
        d = diagnostics(ans, qual)
        corr, fin = r.get("Correct Answer"), r.get("Final Answer")
        d["is_correct"] = (
            int(str(corr).strip() == str(fin).strip())
            if pd.notna(corr) and pd.notna(fin) else np.nan
        )
        d["qno"] = qno
        d["r1_consensus"] = int(len({ans[0][a] for a in range(3) if ans[0][a]}) == 1)
        rows.append(d)
    return pd.DataFrame(rows)


def _has_sheet(path: str, name: str) -> bool:
    try:
        return name in pd.ExcelFile(path).sheet_names
    except (ValueError, FileNotFoundError, OSError):
        return False


def composite(frame: pd.DataFrame) -> pd.Series:
    """Paper's avg-process metric (influence asymmetry inverted)."""
    return (frame["engagement"] + frame["responsiveness"] + frame["balance"]
            + (1 - frame["influence_asymmetry"])) / 4


def summarize(label: str, paths: list[str]) -> dict:
    """Print per-debate and per-question correlations for one protocol; return key numbers."""
    frames = []
    for p in paths:
        f = parse_workbook(p)
        frames.append(f)
    D = pd.concat(frames, ignore_index=True)
    print("=" * 74)
    print(f"{label}   files={len(paths)}   debates={len(D)}   "
          f"accuracy={D.is_correct.mean():.3f}   round1_consensus={D.r1_consensus.mean():.2%}")

    print("  -- per-debate (binary correctness) --")
    for m in METRICS:
        v = D[[m, "is_correct"]].dropna()
        rho, pv = spearmanr(v[m], v["is_correct"])
        print(f"     {m:20s} rho={rho:+.4f} p={pv:.3f}")
    v = pd.DataFrame({"c": composite(D), "y": D["is_correct"]}).dropna()
    rho_pd, _ = spearmanr(v["c"], v["y"])
    print(f"     {'avg_process':20s} rho={rho_pd:+.4f}")

    agg = D.groupby("qno").agg(
        **{m: (m, "mean") for m in METRICS},
        acc=("is_correct", "mean"), n=("is_correct", "count")).reset_index()
    agg = agg[agg["n"] >= 2]
    rho_agg = np.nan
    if len(agg) >= 3:
        print(f"  -- per-question aggregated across seeds (n={len(agg)} questions) --")
        for m in METRICS:
            rho, pv = spearmanr(agg[m], agg["acc"])
            print(f"     {m:20s} rho={rho:+.4f} p={pv:.3f}")
        rho_agg, _ = spearmanr(composite(agg), agg["acc"])
        print(f"     {'avg_process':20s} rho={rho_agg:+.4f}")
    return {
        "accuracy": float(D.is_correct.mean()),
        "r1_consensus": float(D.r1_consensus.mean()),
        "composite_per_debate": float(rho_pd),
        "composite_aggregated": float(rho_agg),
    }


def main() -> None:
    """Parse args, summarize each protocol, print the baseline->adversarial delta."""
    ap = argparse.ArgumentParser(description="Compare adversarial vs baseline debate protocols.")
    ap.add_argument("--baseline", nargs="+", required=True, help="Baseline workbook paths.")
    ap.add_argument("--adversarial", nargs="*", default=[], help="Adversarial workbook paths.")
    args = ap.parse_args()

    base = summarize("BASELINE (cooperative, 'update only if warranted')", args.baseline)
    if not args.adversarial:
        print("\n(no --adversarial workbooks supplied; run the generator, then re-run this)")
        return
    adv = summarize("ADVERSARIAL (Liang tit-for-tat)", args.adversarial)

    print("=" * 74)
    print("DELTA  (adversarial - baseline)")
    print(f"  round-1 consensus : {base['r1_consensus']:.2%} -> {adv['r1_consensus']:.2%}  "
          f"({adv['r1_consensus'] - base['r1_consensus']:+.2%})")
    print(f"  accuracy          : {base['accuracy']:.3f} -> {adv['accuracy']:.3f}  "
          f"({adv['accuracy'] - base['accuracy']:+.3f})")
    print(f"  avg_process rho (per-debate)  : {base['composite_per_debate']:+.4f} -> "
          f"{adv['composite_per_debate']:+.4f}  ({adv['composite_per_debate'] - base['composite_per_debate']:+.4f})")
    print(f"  avg_process rho (aggregated)  : {base['composite_aggregated']:+.4f} -> "
          f"{adv['composite_aggregated']:+.4f}  ({adv['composite_aggregated'] - base['composite_aggregated']:+.4f})")
    print()
    print("READ: consensus should DROP and avg_process rho should RISE toward the")
    print("paper's ~0.7 if the cooperative protocol was suppressing the signal.")
    print("If rho stays ~0 despite lower consensus, the cause is model scale (14B vs 72B).")


if __name__ == "__main__":
    main()
