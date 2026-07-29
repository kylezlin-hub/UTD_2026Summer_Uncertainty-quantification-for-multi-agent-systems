"""selfconsistency_baseline.py -- The decisive "is the debate needed?" test.

Debate uses ~15 generations (3 agents x 5 rounds). The fair competitor is NOT single-shot confidence,
it is COMPUTE-MATCHED self-consistency: draw K independent samples (no interaction) and read off the
answer distribution. If a compute-matched sampler predicts the knowledge-vs-reasoning failure type as
well as the debate dynamics, then the debate machinery adds nothing and is not needed.

This script builds that sampler as a first-class estimator and runs the head-to-head against the
debate-dynamics features from train_failure_classifier.py (shared CV folds), with a paired bootstrap
test on the AUROC difference and a nested test (does dynamics add signal on top of sampling?).

Sample sources
    --source existing : harvest the R1 independent answers already in baseline_v2 (3 agents x 3 seeds
                        ~= 9 samples/question). Zero cost, runnable now.
    --source generate : draw K fresh independent solve passes/question via the model (true compute
                        match, default K=15). Backends local/ollama/mock, cached + resumable.

Labels
    --labels proxy  (default) : pilot proxy from baseline (knowledge=correct-absent, reasoning=present-minority)
    --labels causal           : interventions/intervention_labels.csv (real ground truth)

Usage
    python selfconsistency_baseline.py                                  # existing samples, proxy labels
    python selfconsistency_baseline.py --source generate --backend local --k 15 --require-gpu
    python selfconsistency_baseline.py --labels causal                  # after interventions run
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "docs"))

# Reuse the classifier's label logic + debate-dynamics features (single source of truth).
from train_failure_classifier import (  # noqa: E402
    build_feature_table, proxy_labels, causal_labels, FEATURE_GROUPS,
    DATASET_FILES, AGENTS, _norm,
)

NEW_DIR = HERE.parent / "New"
OUT_DIR = HERE / "interventions"
SC_FEATURES = ["vote_share", "answer_entropy", "n_distinct", "mean_conf", "conf_std", "top2_margin"]


# --------------------------------------------------------------------------- #
# Self-consistency features from a set of independent samples per question
# --------------------------------------------------------------------------- #
def sc_features_from_samples(answers: list[str], confs: list[float], correct: str) -> dict:
    ans = [a for a in answers if a]
    k = len(ans)
    if k == 0:
        return None
    counts = Counter(ans)
    ordered = counts.most_common()
    k_max = ordered[0][1]
    second = ordered[1][1] if len(ordered) > 1 else 0
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    ent_norm = ent / math.log(k) if k > 1 else 0.0
    cvals = [c for c in confs if c is not None and not pd.isna(c)]
    return dict(
        k=k,
        vote_share=k_max / k,
        answer_entropy=ent_norm,
        n_distinct=len(counts),
        top2_margin=(k_max - second) / k,
        mean_conf=float(np.mean(cvals)) if cvals else np.nan,
        conf_std=float(np.std(cvals)) if len(cvals) > 1 else 0.0,
        correct_present=int(correct in counts) if correct else np.nan,
        sc_pred=ordered[0][0],
        sc_correct=int(ordered[0][0] == correct) if correct else np.nan,
    )


def harvest_existing(datasets: list[str]) -> pd.DataFrame:
    """Pool the R1 independent answers across seeds+agents (~9 samples/question)."""
    by_q: dict[str, dict] = {}
    for ds in datasets:
        for f in DATASET_FILES[ds]:
            p = NEW_DIR / f
            if not p.exists():
                continue
            df = pd.read_excel(p, sheet_name="Debate_Traces")
            for _, r in df.iterrows():
                qno = str(r.get("Question #"))
                correct = _norm(r.get("Correct Answer"))
                rec = by_q.setdefault(qno, dict(question_no=qno, dataset=ds, correct=correct,
                                                answers=[], confs=[]))
                for ag in AGENTS:
                    rec["answers"].append(_norm(r.get(f"R1 {ag} Answer")))
                    c = r.get(f"R1 {ag} Conf")
                    rec["confs"].append(None if pd.isna(c) else float(c))
    rows = []
    for qno, rec in by_q.items():
        feats = sc_features_from_samples(rec["answers"], rec["confs"], rec["correct"])
        if feats:
            rows.append({"question_no": qno, "dataset": rec["dataset"], **feats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Fresh compute-matched sampling (K independent solve passes, no interaction)
# --------------------------------------------------------------------------- #
def generate_samples(datasets, k, args) -> pd.DataFrame:
    from qwen_methodology_code import (  # noqa: E402
        DebateQuestion, LocalQwenPipeline, OllamaQwenPipeline, OBJECTIVE_LABELS,
        normalize_answer, parse_qwen_turn,
    )
    from generate_interventions import (  # noqa: E402
        MockPipeline, parse_options, infer_labels, SOLVER_SYSTEM, answer_format_block,
    )

    def build_llm():
        if args.backend == "mock":
            return MockPipeline()
        if args.backend == "ollama":
            return OllamaQwenPipeline(model_id=args.model_id, host=args.ollama_host,
                                      temperature=args.temperature, top_p=0.9,
                                      max_new_tokens=args.max_new_tokens)
        return LocalQwenPipeline(model_id=args.model_id, temperature=args.temperature, top_p=0.9,
                                 max_new_tokens=args.max_new_tokens, device_map="auto",
                                 torch_dtype="auto", require_gpu=args.require_gpu)

    # question universe = same subset the interventions target (correct-absent) OR all if none
    subset_csv = OUT_DIR / "correct_absent_subset.csv"
    if subset_csv.exists():
        subset = pd.read_csv(subset_csv)
        subset = subset[subset["dataset"].isin(datasets)]
    else:
        # fall back to all questions in the requested datasets
        agg, _ = build_feature_table(datasets)
        base = harvest_existing(datasets)[["question_no", "dataset"]]
        subset = base
        subset["question"] = np.nan  # will be filled below from workbooks
    # question text lookup
    qtext = {}
    for ds in datasets:
        for f in DATASET_FILES[ds]:
            p = NEW_DIR / f
            if p.exists():
                d = pd.read_excel(p, sheet_name="Debate_Traces")
                for _, r in d.iterrows():
                    qtext[str(r.get("Question #"))] = (str(r.get("Question")),
                                                       _norm(r.get("Correct Answer")),
                                                       r.get("Dataset Category", ""))
    if args.limit:
        subset = subset.head(args.limit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "selfconsistency_samples.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done.setdefault(d["question_no"], set()).add(d["rep"])

    llm = build_llm()
    print(f"Generating {k} independent samples for {len(subset)} questions (backend={args.backend})")
    for i, row in subset.reset_index(drop=True).iterrows():
        qno = str(row["question_no"])
        if qno not in qtext:
            continue
        question, correct, cat = qtext[qno]
        labels = infer_labels(question)
        q = DebateQuestion("objective", qno, question, correct, labels, cat)
        fmt = answer_format_block(labels)
        msgs = [{"role": "system", "content": SOLVER_SYSTEM},
                {"role": "user", "content": f"{question}\n\n{fmt}"}]
        for rep in range(k):
            if rep in done.get(qno, set()):
                continue
            raw = llm.complete(msgs, seed=args.seed + rep, max_new_tokens=args.max_new_tokens)
            parsed = parse_qwen_turn(raw, "objective", labels, strict=False)
            pred = normalize_answer(str(parsed.get("answer", "")))
            conf = parsed.get("confidence")
            with cache.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(question_no=qno, dataset=row["dataset"], rep=rep,
                                         pred=pred, conf=conf, correct=correct)) + "\n")
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(subset)}", flush=True)

    # aggregate cache -> features
    recs = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    dfc = pd.DataFrame(recs)
    rows = []
    for (qno, ds), g in dfc.groupby(["question_no", "dataset"]):
        correct = _norm(g["correct"].iloc[0])
        feats = sc_features_from_samples(list(g["pred"]),
                                         [float(c) if c is not None and not pd.isna(c) else None
                                          for c in g["conf"]], correct)
        if feats:
            rows.append({"question_no": str(qno), "dataset": ds, **feats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Head-to-head evaluation
# --------------------------------------------------------------------------- #
def oof_proba(X, y, folds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    return cross_val_predict(pipe, X, y, cv=folds, method="predict_proba")[:, 1]


def paired_auc_test(y, p_a, p_b, n_boot=3000, seed=7):
    """Bootstrap p-value for AUROC(p_b) - AUROC(p_a) > 0 (paired over samples)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, p_b) - roc_auc_score(y, p_a)
    n, diffs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    diffs = np.array(diffs)
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(obs), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), float(p_two)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["gpqa", "mmlu-pro"], choices=list(DATASET_FILES))
    ap.add_argument("--labels", default="proxy", choices=["proxy", "causal"])
    ap.add_argument("--source", default="existing", choices=["existing", "generate"])
    ap.add_argument("--k", type=int, default=15, help="samples/question for --source generate")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    # generation backend (only for --source generate)
    ap.add_argument("--backend", default="mock", choices=["local", "ollama", "mock"])
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. debate-dynamics features + labels (from the classifier's builders)
    agg, _ = build_feature_table(args.datasets)
    agg["target_label"] = proxy_labels(agg) if args.labels == "proxy" else causal_labels(agg)
    if args.labels == "proxy":
        print("[note] PROXY labels: structurally favors sampling features. Preliminary only -- "
              "the decisive run is --labels causal.")

    # 2. self-consistency features (compute-matched)
    sc = harvest_existing(args.datasets) if args.source == "existing" else generate_samples(
        args.datasets, args.k, args)
    k_note = f"{int(sc['k'].median())} (median)" if len(sc) else "0"
    print(f"self-consistency samples/question: {k_note}  (source={args.source})")
    sc.to_csv(OUT_DIR / "selfconsistency_features.csv", index=False)

    # 3. join on question, binary target knowledge=1 vs reasoning=0
    merged = agg.merge(sc[["question_no"] + SC_FEATURES], on="question_no", how="inner")
    merged["y"] = merged["target_label"].map({"knowledge-limited": 1, "reasoning-limited": 0})
    d = merged.dropna(subset=SC_FEATURES + FEATURE_GROUPS["dynamics"] + ["y"]).copy()
    d["y"] = d["y"].astype(int)
    n_pos, n_neg = int(d["y"].sum()), int((d["y"] == 0).sum())
    print(f"\nusable questions: {len(d)}  (knowledge={n_pos}, reasoning={n_neg})")
    if min(n_pos, n_neg) < 2:
        print("[abort] need >=2 per class."); return

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    folds = StratifiedKFold(min(args.folds, min(n_pos, n_neg)), shuffle=True, random_state=args.seed)
    y = d["y"].values

    feat_sets = {
        "sampling (compute-matched)": SC_FEATURES,
        "debate dynamics":            FEATURE_GROUPS["dynamics"],
        "sampling + dynamics":        SC_FEATURES + FEATURE_GROUPS["dynamics"],
    }
    proba, auroc = {}, {}
    for name, feats in feat_sets.items():
        proba[name] = oof_proba(d[feats].values, y, folds)
        auroc[name] = round(float(roc_auc_score(y, proba[name])), 3)

    print("\nOOF AUROC (knowledge vs reasoning):")
    for name in feat_sets:
        print(f"  {name:>28}: {auroc[name]}")

    # 4. decisive tests
    obs1, ci1, p1 = paired_auc_test(y, proba["sampling (compute-matched)"], proba["debate dynamics"])
    obs2, ci2, p2 = paired_auc_test(y, proba["sampling (compute-matched)"], proba["sampling + dynamics"])
    print("\npaired bootstrap AUROC differences:")
    print(f"  dynamics - sampling          = {obs1:+.3f}  95%CI[{ci1[0]:+.3f},{ci1[1]:+.3f}]  p={p1:.3f}")
    print(f"  (sampling+dynamics) - sampling= {obs2:+.3f}  95%CI[{ci2[0]:+.3f},{ci2[1]:+.3f}]  p={p2:.3f}")

    added = (obs2 > 0 and p2 < 0.05)
    verdict = ("DEBATE JUSTIFIED: dynamics add signal beyond compute-matched sampling."
               if added else
               "DEBATE NOT NEEDED (on this target): compute-matched sampling matches debate dynamics.")
    print(f"\n>>> VERDICT: {verdict}")
    if args.labels == "proxy":
        print("    (proxy labels favor sampling; re-run --labels causal for the real verdict.)")

    report = dict(labels=args.labels, source=args.source, datasets=args.datasets,
                  n=len(d), n_knowledge=n_pos, n_reasoning=n_neg,
                  median_k=int(sc["k"].median()) if len(sc) else 0,
                  auroc=auroc,
                  dynamics_minus_sampling=dict(diff=obs1, ci=ci1, p=p1),
                  nested_add=dict(diff=obs2, ci=ci2, p=p2),
                  verdict=verdict)
    (OUT_DIR / "debate_vs_sampling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {OUT_DIR/'debate_vs_sampling_report.json'}")


if __name__ == "__main__":
    main()
