"""rescreen_phase1_matched.py -- Independent Phase 1 failure screen, matched to Phase 2 control.

This script performs plain, independent LLM question answering: for each question, the model is
asked to solve it K separate times (independent single-shot attempts, no interaction between
attempts, no other model instances, no rounds). It has NOTHING to do with multi-agent debate --
there is only one model, answering one question at a time, K times with different seeds. A
question is "screened in" (flagged as an initial failure) iff none of the K independent attempts
produced the correct answer.

Purpose
-------
The debate-based Phase 1 screen (generate_debate_local_7b.py) defines "correct-absent" from the
R1 turn of a 3-agent NEUTRAL debate -- a different system prompt (debate framing) than the plain
solver prompt used by Phase 2's control condition (local/generate_interventions.py, SOLVER_SYSTEM).
This script re-runs the Phase 1 screening step using EXACTLY the Phase 2 control prompt and
decoding settings (imported directly from local/generate_interventions.py, zero drift), so the
"correct absent from all K unscaffolded attempts" criterion is measured under the same regime as
the recovery numbers it will be compared against.

What it does
------------
1. Loads the same 300-question fixture set used throughout the project:
     ../data/question/mmlu_5x20.csv, mmlu-pro_5x20.csv, gpqa_random_100.csv
   (100 questions each, deterministic, already on disk -- no new sampling of questions).
2. For each question, draws K (default 3) independent solve passes using:
     - the exact SOLVER_SYSTEM system prompt and answer_format_block user-prompt template
       from local/generate_interventions.py (Phase 2 control condition),
     - the exact decoding settings used for Phase 2 control (temperature=0.7, top_p=0.9,
       max_new_tokens=512),
     - varied seeds: seed = base_seed + 97 * attempt_idx (K distinct seeds per question).
3. A question enters the study (is "screened in" as an initial failure) iff the correct answer
   is absent from all K attempts.
4. Writes:
     rescreen/phase1_matched_samples.jsonl   -- one row per (question, attempt) raw solve
     rescreen/phase1_matched_labels.csv      -- one row per question: k, n_correct, correct_absent,
                                                 screened_in_new, already_in_phase2

Backends: --backend {local,ollama,mock} (same conventions as generate_interventions.py).

Usage
-----
    # offline plumbing test (no model)
    python rescreen_phase1_matched.py --backend mock --limit 5

    # preview question universe only
    python rescreen_phase1_matched.py --select-only

    # real run (Ollama, matches local Phase 2's default backend/model)
    python rescreen_phase1_matched.py --backend ollama --model-id qwen2.5:7b-instruct --k 3

    # re-label from existing samples only (no new generation)
    python rescreen_phase1_matched.py --label-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../Knowledge_vs_Reasoning/local
KVR = HERE.parent                                # .../Knowledge_vs_Reasoning
PROJ = KVR.parent                                # .../Proj1
sys.path.insert(0, str(PROJ / "docs"))

from qwen_methodology_code import (  # noqa: E402
    DebateQuestion as Question, LocalQwenPipeline, OllamaQwenPipeline,
    normalize_answer, parse_qwen_turn,
)
# Reuse the *exact* Phase 2 control prompt/format so there is zero drift between screen and control.
from generate_interventions import (  # noqa: E402
    MockPipeline, SOLVER_SYSTEM, answer_format_block, infer_labels, parse_options,
)

DATA_DIR = KVR / "data" / "question"
QUESTION_FILES = {
    "mmlu": "mmlu_5x20.csv",
    "mmlu-pro": "mmlu-pro_5x20.csv",
    "gpqa": "gpqa_random_100.csv",
}
OUT_DIR = HERE / "rescreen"
INTERVENTION_LABELS_PATH = HERE / "interventions" / "intervention_labels.csv"

# Phase 2 control decoding defaults (must match local/generate_interventions.py defaults exactly).
PHASE2_TEMPERATURE = 0.7
PHASE2_TOP_P = 0.9
PHASE2_MAX_NEW_TOKENS = 512
DEFAULT_K = 3  # 3 independent prompt-matched screening attempts (matches Phase 2 control)


# --------------------------------------------------------------------------- #
# Question loading
# --------------------------------------------------------------------------- #
def load_questions(datasets: list[str]) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        path = DATA_DIR / QUESTION_FILES[ds]
        if not path.exists():
            print(f"  [skip] missing {path}")
            continue
        df = pd.read_csv(path)
        df["dataset"] = ds
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No question files found under {DATA_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out["question_no"] = out["question_no"].astype(str)
    out["correct_answer"] = out["correct_answer"].astype(str).str.strip().str.upper()
    return out


# --------------------------------------------------------------------------- #
# Backend (identical conventions to generate_interventions.py)
# --------------------------------------------------------------------------- #
def build_pipeline(args):
    if args.backend == "mock":
        return MockPipeline()
    if args.backend == "ollama":
        return OllamaQwenPipeline(
            model_id=args.model_id, host=args.ollama_host,
            temperature=PHASE2_TEMPERATURE, top_p=PHASE2_TOP_P,
            max_new_tokens=PHASE2_MAX_NEW_TOKENS,
        )
    return LocalQwenPipeline(
        model_id=args.model_id, temperature=PHASE2_TEMPERATURE, top_p=PHASE2_TOP_P,
        max_new_tokens=PHASE2_MAX_NEW_TOKENS, device_map="auto",
        torch_dtype="auto", require_gpu=args.require_gpu,
    )


# --------------------------------------------------------------------------- #
# Solve pass -- EXACT Phase 2 control prompt (no knowledge brief, no reasoning scaffold)
# --------------------------------------------------------------------------- #
def build_control_messages(q: Question) -> list[dict]:
    fmt = answer_format_block(q.answer_labels)
    user = f"{q.question}\n\n{fmt}"
    return [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": user},
    ]


def solve_once(llm, q: Question, seed: int):
    msgs = build_control_messages(q)
    raw = llm.complete(msgs, seed=seed, max_new_tokens=PHASE2_MAX_NEW_TOKENS,
                       temperature=PHASE2_TEMPERATURE)
    parsed = parse_qwen_turn(raw, "objective", q.answer_labels, strict=False)
    pred = normalize_answer(str(parsed.get("answer", "")))
    return pred, (pred == q.correct_answer), raw


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                done.add((d["question_no"], d["rep"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_generation(args, questions: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_path = OUT_DIR / "phase1_matched_samples.jsonl"
    llm = build_pipeline(args)
    done = load_done(samples_path)

    subset = questions.head(args.limit) if args.limit else questions
    print(f"Re-screening {len(subset)} questions x {args.k} independent prompt-matched attempts "
          f"(backend={args.backend}, model={args.model_id}, "
          f"T={PHASE2_TEMPERATURE}, top_p={PHASE2_TOP_P}, max_new_tokens={PHASE2_MAX_NEW_TOKENS})",
          flush=True)

    for i, row in subset.reset_index(drop=True).iterrows():
        labels = infer_labels(row["question"])
        q = Question("objective", row["question_no"], row["question"],
                     row["correct_answer"], labels, row.get("category", ""))
        for rep in range(args.k):
            if (q.question_no, rep) in done:
                continue
            # Varied seeds: no shared structure across attempts (unlike the old debate screen).
            seed = args.seed + 97 * rep
            pred, correct, raw = solve_once(llm, q, seed)
            append_jsonl(samples_path, dict(
                question_no=q.question_no, dataset=row["dataset"], rep=rep, seed=seed,
                pred=pred, correct=bool(correct), raw=raw[:800],
            ))
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(subset)} questions done", flush=True)
    print(f"Done. Samples -> {samples_path}", flush=True)


def run_labeling(args) -> pd.DataFrame:
    samples_path = OUT_DIR / "phase1_matched_samples.jsonl"
    if not samples_path.exists():
        print(f"No samples at {samples_path}. Run generation first.")
        return pd.DataFrame()
    rows = [json.loads(l) for l in samples_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)

    agg = (df.groupby(["question_no", "dataset"])
             .agg(k=("correct", "count"), n_correct=("correct", "sum"))
             .reset_index())
    agg["correct_absent"] = agg["n_correct"] == 0
    agg["screened_in_new"] = agg["correct_absent"]

    covered = set()
    if INTERVENTION_LABELS_PATH.exists():
        covered = set(pd.read_csv(INTERVENTION_LABELS_PATH)["question_no"].astype(str))
    agg["already_in_phase2"] = agg["question_no"].astype(str).isin(covered)

    out_csv = OUT_DIR / "phase1_matched_labels.csv"
    agg.to_csv(out_csv, index=False)

    n_total = len(agg)
    n_absent = int(agg["correct_absent"].sum())
    n_new = int((agg["correct_absent"] & ~agg["already_in_phase2"]).sum())
    print(f"\nLabeled {n_total} questions -> {out_csv}")
    print(f"correct-absent-from-{agg['k'].median():.0f}-attempts (screened in, matched prompt): "
          f"{n_absent}/{n_total} ({100 * n_absent / max(n_total, 1):.1f}%)")
    print(f"already covered by Phase 2 (in intervention_labels.csv): {int(agg['already_in_phase2'].sum())}")
    print(f"NEW correct-absent questions needing Phase 2: {n_new}")
    print("\nby dataset (correct_absent / already_in_phase2 / new-needing-phase2):")
    agg["new_needing_phase2"] = agg["correct_absent"] & ~agg["already_in_phase2"]
    summary = agg.groupby("dataset")[["correct_absent", "already_in_phase2", "new_needing_phase2"]].sum()
    print(summary.to_string())
    return agg


def write_new_subset_csv(agg: pd.DataFrame, questions: pd.DataFrame) -> Path:
    """Build the Phase-2-ready CSV (question_no, dataset, category, question, correct_answer)
    for NEW correct-absent questions not already covered, for use with
    generate_interventions.py --questions-csv."""
    new_mask = agg["correct_absent"] & ~agg["already_in_phase2"]
    new_qnos = set(agg.loc[new_mask, "question_no"].astype(str))
    sub = questions[questions["question_no"].astype(str).isin(new_qnos)].copy()
    cols = ["question_no", "dataset", "category", "question", "correct_answer"]
    for c in cols:
        if c not in sub.columns:
            sub[c] = ""
    out_path = OUT_DIR / "new_correct_absent_for_phase2.csv"
    sub[cols].to_csv(out_path, index=False)
    print(f"\nNew Phase-2-ready subset ({len(sub)} questions) -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "mmlu-pro", "gpqa"],
                    choices=list(QUESTION_FILES))
    ap.add_argument("--backend", default="ollama", choices=["local", "ollama", "mock"])
    ap.add_argument("--model-id", default="qwen2.5:7b-instruct")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="independent prompt-matched screening attempts per question")
    ap.add_argument("--seed", type=int, default=7, help="base seed; reps use seed + 97*rep")
    ap.add_argument("--limit", type=int, default=0, help="cap number of questions (0 = all)")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--select-only", action="store_true", help="print question universe and exit")
    ap.add_argument("--label-only", action="store_true", help="label from existing samples and exit")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override output directory (default: rescreen/) -- IMPORTANT: use a "
                         "distinct directory per model (e.g. rescreen_llama8b/) to avoid mixing "
                         "or overwriting another model's samples/labels in the same files")
    ap.add_argument("--interventions-dir", type=Path, default=None,
                    help="override where to look for an existing intervention_labels.csv when "
                         "annotating 'already_in_phase2' (default: interventions/). Point this at "
                         "the matching model-specific Phase 2 output directory, or a nonexistent "
                         "path if none exists yet (reports 0 covered, which is correct pre-Phase-2)")
    args = ap.parse_args()

    global OUT_DIR, INTERVENTION_LABELS_PATH
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    if args.interventions_dir is not None:
        INTERVENTION_LABELS_PATH = args.interventions_dir / "intervention_labels.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.datasets)

    if args.select_only:
        print(f"Question universe: {len(questions)} questions")
        print(questions.groupby("dataset").size().to_string())
        return
    if args.label_only:
        agg = run_labeling(args)
        if not agg.empty:
            write_new_subset_csv(agg, questions)
        return

    run_generation(args, questions)
    agg = run_labeling(args)
    if not agg.empty:
        write_new_subset_csv(agg, questions)


if __name__ == "__main__":
    main()
