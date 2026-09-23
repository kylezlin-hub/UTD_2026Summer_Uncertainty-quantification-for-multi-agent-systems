"""train_failure_classifier.py -- Predict knowledge- vs reasoning-limited failure from debate dynamics.

Core scientific question: do same-model debate DYNAMICS predict the failure type better than
single-shot CONFIDENCE (and than initial self-consistency)?

Targets
-------
--labels causal  : read interventions/intervention_labels.csv (from generate_interventions.py).
                   Primary target = knowledge-limited (1) vs reasoning-limited (0).
--labels proxy   : (default until interventions are run) derive the pilot proxy from baseline data:
                   among initially-wrong questions, knowledge = correct-ABSENT from R1 pool,
                   reasoning = correct-PRESENT-as-minority. NOTE: this proxy has a structural
                   dependence on initial disagreement, so treat proxy-mode AUROCs as a plumbing /
                   preliminary check, not the headline result. Swap to --labels causal when ready.

Feature groups (each trained as a separate model with shared CV folds, then compared)
    confidence : mean_init_conf, conf_change          (what a single-shot estimate gives you)
    selfconsist: init_disagreement, final_disagreement (multi-sample, no debate process)
    dynamics   : switches, oscillation, tau, movement, final_unanimous
    all        : union of the above

Outputs (interventions/)
    classifier_features.csv     per-question feature table + target
    classifier_report.json      AUROC (OOF, per feature group) + LR coefficients
    classifier_risk_coverage.csv selective-classification curve for the best model
    classifier_report.png        AUROC bars + risk-coverage plot

Usage
-----
    python train_failure_classifier.py                 # proxy labels, gpqa+mmlu-pro
    python train_failure_classifier.py --labels causal # after interventions run
    python train_failure_classifier.py --multiclass    # knowledge/reasoning/hard one-vs-rest
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEW_DIR = HERE.parent / "New"
OUT_DIR = HERE / "interventions"
AGENTS = ["Agent1", "Agent2", "Agent3"]
DATASET_FILES = {
    "mmlu-pro": [f"baseline_v2_mmlu-pro_s{s}.xlsx" for s in (7, 17, 42)],
    "gpqa":     [f"baseline_v2_gpqa_s{s}.xlsx" for s in (7, 17, 42)],
}
FEATURE_GROUPS = {
    "confidence":  ["mean_init_conf", "conf_change"],
    "selfconsist": ["init_disagreement", "final_disagreement"],
    "dynamics":    ["switches", "oscillation", "tau", "movement", "final_unanimous"],
}
FEATURE_GROUPS["all"] = sorted({f for g in FEATURE_GROUPS.values() for f in g})
KNOWLEDGE_LABELS = {"knowledge-limited"}
REASONING_LABELS = {"reasoning-limited"}


def _norm(x):
    return None if pd.isna(x) else str(x).strip().upper()


def _plur(a):
    p = [x for x in a if x]
    return Counter(p).most_common(1)[0][0] if p else None


# --------------------------------------------------------------------------- #
# Per-(question, seed) debate-dynamics features from baseline_v2 workbooks
# --------------------------------------------------------------------------- #
def features_for_file(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Debate_Traces")
    rounds = sorted({int(m.group(1)) for c in df.columns if (m := re.match(r"R(\d+) Agent1 Answer", c))})
    n_ag = len(AGENTS)
    rows = []
    for _, r in df.iterrows():
        correct = _norm(r.get("Correct Answer"))
        qtext = str(r.get("Question"))
        n_options = len(re.findall(r"(?m)^\s*([A-J])[\.\)]\s", qtext)) or len(tuple("ABCD"))
        ans, conf = [], []
        la = {a: None for a in AGENTS}
        lc = {a: None for a in AGENTS}
        for rd in rounds:
            arow, crow = [], []
            for ag in AGENTS:
                a = r.get(f"R{rd} {ag} Answer")
                a = la[ag] if pd.isna(a) else _norm(a)
                la[ag] = a
                c = r.get(f"R{rd} {ag} Conf")
                c = lc[ag] if pd.isna(c) else float(c)
                lc[ag] = c
                arow.append(a)
                crow.append(c)
            ans.append(arow)
            conf.append(crow)

        init, final = ans[0], ans[-1]
        n_init = len({a for a in init if a})
        n_final = len({a for a in final if a})
        init_pred, final_pred = _plur(init), _plur(final)
        init_disagreement = (n_init - 1) / (n_ag - 1)
        final_disagreement = (n_final - 1) / (n_ag - 1)

        switches = osc = 0
        for k in range(n_ag):
            traj = [ans[t][k] for t in range(len(ans))]
            switches += sum(traj[t] != traj[t-1] for t in range(1, len(traj)))
            osc += sum(traj[t] == traj[t-2] and traj[t] != traj[t-1] for t in range(2, len(traj)))
        tau = len(rounds) + 1
        for t in range(len(ans)):
            present = [a for a in ans[t] if a]
            if len(present) == n_ag and len(set(present)) == 1:
                tau = t + 1
                break
        mic = np.nanmean([c for c in conf[0] if c is not None])
        mfc = np.nanmean([c for c in conf[-1] if c is not None])
        rows.append(dict(
            question_no=str(r.get("Question #")), dataset=dataset,
            category=str(r.get("Dataset Category", "")), n_options=int(n_options),
            correct_answer=correct,
            init_correct=(init_pred == correct) if correct else np.nan,
            final_correct=(final_pred == correct) if correct else np.nan,
            correct_in_init=(correct in {a for a in init if a}) if correct else np.nan,
            init_disagreement=init_disagreement, final_disagreement=final_disagreement,
            movement=init_disagreement - final_disagreement,
            switches=float(switches), oscillation=float(osc), tau=float(tau),
            final_unanimous=float(n_final == 1),
            mean_init_conf=mic, mean_final_conf=mfc, conf_change=mfc - mic,
        ))
    return pd.DataFrame(rows)


def build_feature_table(datasets: list[str]) -> pd.DataFrame:
    parts = []
    for ds in datasets:
        for f in DATASET_FILES[ds]:
            p = NEW_DIR / f
            if p.exists():
                parts.append(features_for_file(p, ds))
    per_seed = pd.concat(parts, ignore_index=True)
    # aggregate to one row per question (mean over seeds for numerics; first() for metadata)
    meta_cols = ("question_no", "dataset", "correct_answer", "category")
    num = [c for c in per_seed.columns if c not in meta_cols]
    agg = per_seed.groupby(["question_no", "dataset"])[num].mean().reset_index()
    meta = (per_seed.groupby(["question_no", "dataset"])["category"].first().reset_index())
    agg = agg.merge(meta, on=["question_no", "dataset"], how="left")
    return agg, per_seed


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def proxy_labels(agg: pd.DataFrame) -> pd.Series:
    """knowledge=1 (correct-absent), reasoning=0 (correct-present-minority), among initially-wrong."""
    wrong = agg["init_correct"] < 0.5
    lab = pd.Series(index=agg.index, dtype=object)
    lab[wrong & (agg["correct_in_init"] < 0.5)] = "knowledge-limited"
    lab[wrong & (agg["correct_in_init"] >= 0.5)] = "reasoning-limited"
    return lab


def causal_labels(agg: pd.DataFrame) -> pd.Series:
    path = OUT_DIR / "intervention_labels.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run generate_interventions.py first, or use --labels proxy.")
    lab = pd.read_csv(path)[["question_no", "label"]]
    lab["question_no"] = lab["question_no"].astype(str)
    merged = agg.merge(lab, on="question_no", how="left")
    return merged["label"]


# --------------------------------------------------------------------------- #
# Train + evaluate
# --------------------------------------------------------------------------- #
def oof_auroc(X, y, groups_feats, folds):
    """Return {group: (auroc, coef_dict)} using out-of-fold probabilities on shared folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    out = {}
    for gname, feats in groups_feats.items():
        Xg = X[feats].values
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        try:
            proba = cross_val_predict(pipe, Xg, y, cv=folds, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, proba)
        except ValueError:
            proba, auc = np.full(len(y), np.nan), np.nan
        pipe.fit(Xg, y)
        coefs = dict(zip(feats, pipe.named_steps["logisticregression"].coef_[0].round(3)))
        out[gname] = dict(auroc=round(float(auc), 3), coef=coefs, proba=proba)
    return out


def risk_coverage(y_true, proba):
    """Selective classification: sort by confidence=|p-0.5|; at each coverage report accuracy."""
    conf = np.abs(proba - 0.5)
    order = np.argsort(-conf)
    yt = np.asarray(y_true)[order]
    pred = (proba[order] >= 0.5).astype(int)
    rows = []
    for k in range(1, len(yt) + 1):
        cov = k / len(yt)
        acc = float((pred[:k] == yt[:k]).mean())
        rows.append((round(cov, 3), round(acc, 3), round(1 - acc, 3)))
    return pd.DataFrame(rows, columns=["coverage", "accuracy", "risk"])


def _fit_predict(train_x, train_y, test_x):
    """Fit scaler+LR on train only, return P(class=1) on test. Scaler-on-train handles scale shift."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    pipe.fit(train_x, train_y)
    return pipe.predict_proba(test_x)[:, 1]


def eval_lodo(bd, feature_groups):
    """Leave-one-dataset-out transfer: train on the other dataset(s), test on held-out one."""
    from sklearn.metrics import roc_auc_score
    datasets = sorted(bd["dataset"].unique())
    rows = []
    for test_ds in datasets:
        tr, te = bd[bd.dataset != test_ds], bd[bd.dataset == test_ds]
        row = {"held_out": test_ds, "n_train": len(tr), "n_test": len(te),
               "test_pos": int(te.y.sum()), "test_neg": int((te.y == 0).sum())}
        for g, feats in feature_groups.items():
            if len(set(tr.y)) < 2 or len(set(te.y)) < 2:
                row[g] = np.nan
                continue
            proba = _fit_predict(tr[feats].values, tr.y.values, te[feats].values)
            row[g] = round(float(roc_auc_score(te.y.values, proba)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def eval_cross_category(bd, feature_groups, folds):
    """GroupKFold by Dataset Category: train on some domains, test on held-out domains (OOF AUROC)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    groups = bd["category"].fillna("NA").astype(str).values
    n_groups = len(set(groups))
    n_splits = min(folds, n_groups)
    if n_splits < 2:
        return None, n_groups
    gkf = GroupKFold(n_splits=n_splits)
    out = {}
    for g, feats in feature_groups.items():
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))
        try:
            proba = cross_val_predict(pipe, bd[feats].values, bd.y.values, cv=gkf,
                                      groups=groups, method="predict_proba")[:, 1]
            out[g] = round(float(roc_auc_score(bd.y.values, proba)), 3)
        except ValueError:
            out[g] = np.nan
    return out, n_groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["gpqa", "mmlu-pro"], choices=list(DATASET_FILES))
    ap.add_argument("--labels", default="proxy", choices=["proxy", "causal"])
    ap.add_argument("--eval", default="cv", choices=["cv", "lodo", "cross-category"],
                    help="cv=in-domain k-fold; lodo=leave-one-dataset-out transfer; "
                         "cross-category=GroupKFold by domain")
    ap.add_argument("--multiclass", action="store_true", help="knowledge/reasoning/hard one-vs-rest macro-AUROC")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    agg, _ = build_feature_table(args.datasets)
    agg["target_label"] = proxy_labels(agg) if args.labels == "proxy" else causal_labels(agg)
    if args.labels == "proxy":
        print("[note] PROXY labels: structural dependence with initial-disagreement -- preliminary only.\n"
              "       Re-run with --labels causal once intervention_labels.csv exists.")

    feat_cols = FEATURE_GROUPS["all"]
    data = agg.dropna(subset=feat_cols + ["target_label"]).copy()
    data.to_csv(OUT_DIR / "classifier_features.csv", index=False)

    from sklearn.model_selection import StratifiedKFold
    report = {"labels": args.labels, "datasets": args.datasets, "n_total": int(len(data))}

    if args.multiclass:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        keep = data[data["target_label"].isin(
            ["knowledge-limited", "reasoning-limited", "hard/unrecoverable"])].copy()
        y = keep["target_label"].values
        classes = sorted(set(y))
        print(f"\nMulticlass target counts: {dict(Counter(y))}")
        if len(classes) < 3:
            print(f"[warn] only {len(classes)} classes present ({classes}); multiclass needs 3 "
                  "(knowledge/reasoning/hard). This is expected with --labels proxy. "
                  "Use the default binary mode, or run --multiclass with --labels causal.")
            return
        folds = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        results = {}
        for gname, feats in FEATURE_GROUPS.items():
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=1000, class_weight="balanced"))
            proba = cross_val_predict(pipe, keep[feats].values, y, cv=folds, method="predict_proba")
            auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
            results[gname] = round(float(auc), 3)
        report["macro_auroc_ovr"] = results
        print("\nmacro-AUROC (one-vs-rest) by feature group:")
        for g, a in results.items():
            print(f"  {g:>12}: {a}")
    else:
        bin_map = {}
        for lab in KNOWLEDGE_LABELS:
            bin_map[lab] = 1
        for lab in REASONING_LABELS:
            bin_map[lab] = 0
        data["y"] = data["target_label"].map(bin_map)
        bd = data.dropna(subset=["y"]).copy()
        bd["y"] = bd["y"].astype(int)
        n_pos, n_neg = int(bd["y"].sum()), int((bd["y"] == 0).sum())
        print(f"\nBinary target: knowledge-limited={n_pos}  reasoning-limited={n_neg}  (n={len(bd)})")
        report.update(n_knowledge=n_pos, n_reasoning=n_neg)

        # ---- transfer / generalization evaluations ----------------------------------- #
        if args.eval != "cv":
            print("[note] features are dimensionless (agent/round-normalized); n_options is recorded "
                  "but EXCLUDED from predictors so the model can't use it as a dataset shortcut.")
        if args.eval == "lodo":
            if bd["dataset"].nunique() < 2:
                print("[abort] LODO needs >=2 datasets; pass --datasets gpqa mmlu-pro.")
                return
            tbl = eval_lodo(bd, FEATURE_GROUPS)
            print("\nLeave-one-dataset-out transfer AUROC (train on other dataset -> test on held-out):")
            print(tbl.to_string(index=False))
            tbl.to_csv(OUT_DIR / "classifier_lodo.csv", index=False)
            report["lodo"] = tbl.to_dict("records")
            (OUT_DIR / "classifier_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nlodo -> {OUT_DIR/'classifier_lodo.csv'}   report -> {OUT_DIR/'classifier_report.json'}")
            return
        if args.eval == "cross-category":
            res_cc, n_groups = eval_cross_category(bd, FEATURE_GROUPS, args.folds)
            print(f"\nCross-category transfer (GroupKFold by domain, {n_groups} categories) OOF AUROC:")
            if res_cc is None:
                print("  [skip] need >=2 categories.")
            else:
                for g in ["confidence", "selfconsist", "dynamics", "all"]:
                    print(f"  {g:>12}: {res_cc[g]}")
                report["cross_category_auroc"] = res_cc
                report["n_categories"] = int(n_groups)
            (OUT_DIR / "classifier_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nreport -> {OUT_DIR/'classifier_report.json'}")
            return

        if min(n_pos, n_neg) < args.folds:
            print(f"[warn] too few in a class for {args.folds}-fold CV; results unstable.")
        folds = StratifiedKFold(min(args.folds, max(2, min(n_pos, n_neg))),
                                shuffle=True, random_state=args.seed)
        res = oof_auroc(bd, bd["y"].values, FEATURE_GROUPS, folds)
        report["auroc"] = {g: r["auroc"] for g, r in res.items()}
        report["coef"] = {g: r["coef"] for g, r in res.items()}
        print("\nOOF AUROC by feature group (knowledge vs reasoning):")
        for g in ["confidence", "selfconsist", "dynamics", "all"]:
            print(f"  {g:>12}: {res[g]['auroc']}")
        delta = res["dynamics"]["auroc"] - res["confidence"]["auroc"]
        print(f"\n>>> dynamics - confidence AUROC = {delta:+.3f}  "
              f"({'dynamics beats confidence' if delta > 0 else 'no gain'})")
        # risk-coverage for the 'all' model
        rc = risk_coverage(bd["y"].values, res["all"]["proba"])
        rc.to_csv(OUT_DIR / "classifier_risk_coverage.csv", index=False)

        # plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
            groups = ["confidence", "selfconsist", "dynamics", "all"]
            vals = [res[g]["auroc"] for g in groups]
            colors = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]
            a1.bar(groups, vals, color=colors)
            for i, v in enumerate(vals):
                a1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
            a1.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
            a1.set_ylim(0, 1); a1.set_ylabel("OOF AUROC")
            a1.set_title(f"Knowledge vs reasoning ({args.labels} labels, n={len(bd)})")
            a1.legend()
            a2.plot(rc["coverage"], rc["accuracy"], marker=".", color="#4C72B0")
            a2.set_xlabel("coverage"); a2.set_ylabel("accuracy (selective)")
            a2.set_title("Risk-coverage (all-features model)"); a2.grid(alpha=0.3); a2.set_ylim(0, 1)
            fig.tight_layout(); fig.savefig(OUT_DIR / "classifier_report.png", dpi=120)
            print(f"\nplot -> {OUT_DIR/'classifier_report.png'}")
        except Exception as e:  # noqa: BLE001
            print("[warn] plot skipped:", e)

    (OUT_DIR / "classifier_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {OUT_DIR/'classifier_report.json'}   features -> {OUT_DIR/'classifier_features.csv'}")


if __name__ == "__main__":
    main()
