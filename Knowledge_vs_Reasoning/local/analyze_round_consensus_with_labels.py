"""analyze_round_consensus_with_labels.py -- Round-level CONSENSUS rate for the same-model
Qwen2.5-7B multi-agent debate (data/debate_local_7b/runs/), broken out by seed (the only
sampling-diversity axis in this dataset -- no temperature sweep; every run uses temperature=0.7,
top_p=0.9), joined with each question's Phase 2 causal label from
interventions/intervention_labels.csv.

Mirrors analyze_round_accuracy_with_labels.py but for CONSENSUS instead of ACCURACY:
    C_r = consensus rate at round r = fraction of questions (not agent-turns) where all 3 agents
          hold the same answer at round r (Round_State's `consensus_reached` flag, which is a
          question-level property duplicated across its 3 agent rows -- deduplicated here to
          one row per (question_no, round) before averaging).
    D_r = C_r - C_{r-1}  (delta vs previous round; D_1 is left blank -- no round 0)

Also attaches, per question_no, its Phase 2 label (knowledge-limited / reasoning-limited /
hard-unrecoverable / stochastic-recoverable / ambiguous / both-sufficient / interaction (both
needed)); questions with no Phase 2 label on file are flagged "MISSING". Uses ALL 300 questions
(mmlu, mmlu-pro, gpqa; 100 each) across all 3 seeds (7, 17, 42).

Outputs (written next to this script, under analysis/):
    round_consensus_by_seed.csv         -- one row per (seed, round): C_r, D_r (pooled over all
                                            300 questions)
    round_consensus_by_seed_dataset.csv -- same, broken out additionally by dataset
    round_consensus_by_label.csv        -- one row per (label incl. MISSING, seed, round): C_r, D_r

(questions_with_labels_300.csv is shared with the accuracy analysis; not re-written here.)

Usage
-----
    python analyze_round_consensus_with_labels.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "data" / "debate_local_7b" / "runs"
LABELS_PATH = HERE / "interventions" / "intervention_labels.csv"
OUT_DIR = HERE / "analysis"

DATASETS = ["mmlu", "mmlu-pro", "gpqa"]
SEEDS = [7, 17, 42]


def _workbook_path(ds: str, seed: int) -> Path:
    return RUNS_DIR / f"{ds}_seed_{seed}" / f"debate_local_7b_{ds}_s{seed}.xlsx"


def load_all_round_state_dedup() -> pd.DataFrame:
    """One row per (dataset, seed, question_no, round) -- consensus_reached is a question-level
    flag duplicated across the 3 agent rows, so we take the first occurrence per group."""
    frames = []
    for ds in DATASETS:
        for seed in SEEDS:
            path = _workbook_path(ds, seed)
            if not path.exists():
                print(f"  [skip] missing {path}")
                continue
            df = pd.read_excel(path, sheet_name="Round_State")
            df["dataset"] = ds
            df["seed"] = seed
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No debate_local_7b workbooks found under {RUNS_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out["question_no"] = out["question_no"].astype(str)
    dedup = (out.sort_values(["dataset", "seed", "question_no", "round"])
             .drop_duplicates(["dataset", "seed", "question_no", "round"], keep="first"))
    return dedup


def load_all_question_meta() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        path = _workbook_path(ds, 7)
        if not path.exists():
            continue
        df = pd.read_excel(path, sheet_name="Debate_Traces")
        for _, r in df.iterrows():
            rows.append(dict(
                question_no=str(r.get("Question #")),
                dataset=ds,
                category=r.get("Dataset Category", ""),
                correct_answer=r.get("Correct Answer", ""),
            ))
    return pd.DataFrame(rows).drop_duplicates("question_no").reset_index(drop=True)


def compute_round_consensus(round_state_dedup: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """C_r = mean(consensus_reached) per group (e.g. seed, round), one row per QUESTION already
    (deduplicated), so n_questions is the true question count, not agent-turn count.
    D_r = C_r - C_{r-1} within each non-round group, ordered by round."""
    agg = (round_state_dedup.groupby(group_cols + ["round"])["consensus_reached"]
           .agg(C_r="mean", n_questions="count").reset_index())
    agg = agg.sort_values(group_cols + ["round"])
    agg["D_r"] = agg.groupby(group_cols)["C_r"].diff()
    return agg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Round_State (deduplicated to one row per question/round) "
          "from all debate_local_7b workbooks (3 datasets x 3 seeds)...")
    rs = load_all_round_state_dedup()
    print(f"  {len(rs)} (question, round) rows, {rs['question_no'].nunique()} unique questions, "
          f"seeds={sorted(rs['seed'].unique())}, rounds={sorted(rs['round'].unique())}")

    # --- 1. Round consensus by seed (pooled over all datasets/questions) ---
    by_seed = compute_round_consensus(rs, ["seed"])
    by_seed.to_csv(OUT_DIR / "round_consensus_by_seed.csv", index=False)
    print("\n=== C_r and D_r by seed and round (pooled over 300 questions) ===")
    print(by_seed.round(4).to_string(index=False))

    # --- 2. Round consensus by seed x dataset ---
    by_seed_ds = compute_round_consensus(rs, ["seed", "dataset"])
    by_seed_ds.to_csv(OUT_DIR / "round_consensus_by_seed_dataset.csv", index=False)

    # --- 3. Attach Phase 2 label (reuse the same join logic; flag MISSING) ---
    qmeta = load_all_question_meta()
    if LABELS_PATH.exists():
        labels = pd.read_csv(LABELS_PATH)
        labels["question_no"] = labels["question_no"].astype(str)
        qmeta = qmeta.merge(labels[["question_no", "label", "confidence"]],
                             on="question_no", how="left")
    else:
        print(f"  [warn] {LABELS_PATH} not found; all questions will be flagged MISSING")
        qmeta["label"] = np.nan
        qmeta["confidence"] = np.nan
    qmeta["phase2_label"] = qmeta["label"].fillna("MISSING")
    n_missing = int((qmeta["phase2_label"] == "MISSING").sum())
    print(f"\nQuestions with a Phase 2 label: {len(qmeta) - n_missing} / {len(qmeta)} "
          f"(MISSING: {n_missing})")

    # --- 4. Round consensus by Phase 2 label ---
    rs_labeled = rs.merge(qmeta[["question_no", "phase2_label"]], on="question_no", how="left")
    by_label = compute_round_consensus(rs_labeled, ["phase2_label", "seed"])
    by_label.to_csv(OUT_DIR / "round_consensus_by_label.csv", index=False)
    print("\n=== C_r by Phase 2 label, seed, and round (see round_consensus_by_label.csv) ===")
    print(by_label.round(4).to_string(index=False))

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
