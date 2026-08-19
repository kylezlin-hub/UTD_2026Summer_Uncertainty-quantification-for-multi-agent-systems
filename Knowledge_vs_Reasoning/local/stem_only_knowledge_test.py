"""stem_only_knowledge_test.py -- Decisive robustness check: does the knowledge_blind recovery
effect survive when the brief generator sees ONLY the question stem, never the answer choices?

Motivation
----------
The existing "blind" knowledge brief is generated from q.question, which EMBEDS the full
multiple-choice options (verified directly from the data). So even though the brief-writer is
never told the gold answer, it can see all candidate options and could, in principle, silently
solve the question itself and write a brief that content-echoes the correct option -- inflating
the knowledge_blind condition's recovery for reasons unrelated to genuine background-knowledge
injection. A prior audit (audit_blind_brief_echo.py) found this content-echo confound is present
but small and not concentrated in knowledge-limited questions (see blind_brief_echo_audit_with_
labels.csv), and recovery was similar for echoing vs non-echoing briefs. This script is the
decisive follow-up: regenerate the blind brief from the STEM ONLY (options stripped out, so
option-conditioned reconstruction is structurally impossible), then re-solve under a new
"knowledge_blind_stem" condition using the FULL question (stem + options, as normal) plus this
stem-only brief, and compare its recovery against the existing `control` and `knowledge_blind`
(stem+choices brief) rates already on file for the SAME questions.

Per the agreed design: no new control-question group is generated. The existing Phase 2 `control`
condition (plain re-ask, same solver prompt/format/temperature/max_new_tokens/8 repeats) already
IS the correct control for this comparison, since the new knowledge_blind_stem condition uses the
identical solver prompt, format, decoding settings, and repeat count -- only the brief-generation
input changes (stem-only vs stem+choices).

Population: ALL questions currently labeled 'knowledge-limited' in intervention_labels.csv (the
complete, final set now that Phase 2 covers all 300 questions).

New condition: knowledge_blind_stem
    brief generation : TUTOR_SYSTEM_BLIND (unchanged) + ONLY the question stem (options stripped
                        via a regex that removes all "^<LETTER>. text$" lines used by parse_options,
                        so the brief-writer cannot see or reconstruct from the candidate answers).
    solve pass        : SOLVER_SYSTEM (unchanged) + "Relevant background information:\n{stem-only
                        brief}\n" + the FULL question (stem + options, as in every other condition)
                        + answer_format_block -- identical to how knowledge_blind is solved, just
                        with a differently-sourced brief.
    repeats           : 8 (matches the existing 8 repeats/condition used throughout Phase 2)
    decoding          : temperature=0.7, top_p=0.9, max_new_tokens=512 (identical to all other
                        conditions -- see PHASE2_* constants)

Outputs (under interventions/):
    stem_only_knowledge_briefs.jsonl     -- one row per question: stem_only_brief, leaked flag
    stem_only_solve_results.jsonl        -- one row per (question, repeat) solve call
    stem_only_knowledge_test_results.csv -- one row per question: control, knowledge_blind
                                            (original, stem+choices), knowledge_blind_stem (new),
                                            plus deltas, joined against the existing label

Usage
-----
    python stem_only_knowledge_test.py --backend mock --limit 3      # offline smoke test
    python stem_only_knowledge_test.py --backend ollama --model-id qwen2.5:7b-instruct
    python stem_only_knowledge_test.py --label-only                  # summarize existing results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "docs"))

from qwen_methodology_code import (  # noqa: E402
    DebateQuestion as Question, LocalQwenPipeline, OllamaQwenPipeline,
    normalize_answer, parse_qwen_turn,
)
from generate_interventions import (  # noqa: E402
    MockPipeline, SOLVER_SYSTEM, TUTOR_SYSTEM_BLIND, _BRIEF_RULES,
    answer_format_block, infer_labels, brief_mentions_options,
    RUNS_DIR, DATASET_FILES,
)

OUT_DIR = HERE / "interventions"
LABELS_PATH = OUT_DIR / "intervention_labels.csv"
BRIEFS_OUT = OUT_DIR / "stem_only_knowledge_briefs.jsonl"
RESULTS_OUT = OUT_DIR / "stem_only_solve_results.jsonl"

PHASE2_TEMPERATURE = 0.7
PHASE2_TOP_P = 0.9
PHASE2_MAX_NEW_TOKENS = 512
DEFAULT_REPEATS = 8

_OPTION_LINE_RE = re.compile(r"(?m)^\s*[A-J][\.\)]\s+.*\S\s*$")


def strip_options_to_stem(question_text: str) -> str:
    """Remove every '<LETTER>. option text' line, keeping only the stem. Structurally prevents
    the brief generator from ever seeing the candidate answers."""
    stem = _OPTION_LINE_RE.sub("", question_text)
    return re.sub(r"\n{2,}", "\n", stem).strip()


def load_knowledge_limited_questions(only_knowledge_limited: bool = True) -> pd.DataFrame:
    """Load questions to run the stem-only knowledge condition on.

    only_knowledge_limited=True  -> just the 36 knowledge-limited questions (original scope).
    only_knowledge_limited=False -> ALL 300 labeled questions, so option-aware vs option-blind
                                     knowledge elicitation can be compared across every label,
                                     not just the knowledge-limited subset.
    """
    labels = pd.read_csv(LABELS_PATH)
    labels["question_no"] = labels["question_no"].astype(str)
    if only_knowledge_limited:
        pool = labels[labels["label"] == "knowledge-limited"].copy()
    else:
        pool = labels.copy()

    rows = []
    for ds, files in DATASET_FILES.items():
        seed7 = next((f for f in files if "_s7" in f or "seed_7" in f), files[0])
        path = RUNS_DIR / seed7
        if not path.exists():
            continue
        df = pd.read_excel(path, sheet_name="Debate_Traces")
        for _, r in df.iterrows():
            qno = str(r.get("Question #"))
            rows.append(dict(question_no=qno, dataset=ds,
                              category=r.get("Dataset Category", ""),
                              question=str(r.get("Question")),
                              correct_answer=str(r.get("Correct Answer")).strip().upper()))
    qmeta = pd.DataFrame(rows).drop_duplicates("question_no")

    merged = pool[["question_no", "label", "control", "knowledge_blind"]].merge(
        qmeta, on="question_no", how="left")
    missing = merged["question"].isna().sum()
    if missing:
        print(f"  [warn] {missing} questions had no question text found")
    return merged.dropna(subset=["question"]).reset_index(drop=True)


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


def generate_stem_only_brief(llm, stem: str, correct_letter: str, seed: int, max_tries: int = 3):
    """Blind brief generated from the STEM ONLY -- options are never shown, so content-echo via
    option-conditioned reconstruction is structurally impossible (only a formatting slip -- the
    model spontaneously naming a letter it was never given -- is still checked, as a safety net)."""
    user = f"Question:\n{stem}\n\n{_BRIEF_RULES}"
    msgs = [{"role": "system", "content": TUTOR_SYSTEM_BLIND}, {"role": "user", "content": user}]
    brief = ""
    for t in range(max_tries):
        brief = llm.complete(msgs, seed=seed + t, max_new_tokens=350, temperature=0.5).strip()
        if not brief_mentions_options(brief, correct_letter):
            return brief, False
    return brief, brief_mentions_options(brief, correct_letter)


def build_solve_messages_stem_brief(q: Question, brief: str) -> list[dict]:
    fmt = answer_format_block(q.answer_labels)
    parts = [f"Relevant background information:\n{brief}\n", f"{q.question}\n\n{fmt}"]
    return [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def solve_once(llm, q: Question, brief: str, seed: int):
    msgs = build_solve_messages_stem_brief(q, brief)
    raw = llm.complete(msgs, seed=seed, max_new_tokens=PHASE2_MAX_NEW_TOKENS,
                       temperature=PHASE2_TEMPERATURE)
    parsed = parse_qwen_turn(raw, "objective", q.answer_labels, strict=False)
    pred = normalize_answer(str(parsed.get("answer", "")))
    return pred, (pred == q.correct_answer), raw


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
    llm = build_pipeline(args)

    briefs = {}
    if BRIEFS_OUT.exists():
        for line in BRIEFS_OUT.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            briefs[d["question_no"]] = d
    done = load_done(RESULTS_OUT)

    subset = questions.head(args.limit) if args.limit else questions
    print(f"Stem-only knowledge test: {len(subset)} questions x {args.repeats} "
          f"repeats (backend={args.backend}, model={args.model_id})", flush=True)

    for i, row in subset.reset_index(drop=True).iterrows():
        labels = infer_labels(row["question"])
        q = Question("objective", row["question_no"], row["question"],
                     row["correct_answer"], labels, row.get("category", ""))
        stem = strip_options_to_stem(row["question"])

        if q.question_no not in briefs:
            brief, leaked = generate_stem_only_brief(llm, stem, q.correct_answer, seed=args.seed)
            rec = dict(question_no=q.question_no, dataset=row["dataset"], stem=stem,
                       stem_only_brief=brief, leaked=leaked, correct_answer=q.correct_answer)
            append_jsonl(BRIEFS_OUT, rec)
            briefs[q.question_no] = rec
        brief_text = briefs[q.question_no]["stem_only_brief"]

        for rep in range(args.repeats):
            if (q.question_no, rep) in done:
                continue
            seed = args.seed + 1000 * rep + 777  # distinct offset; no collision with other conds
            pred, correct, raw = solve_once(llm, q, brief_text, seed)
            append_jsonl(RESULTS_OUT, dict(
                question_no=q.question_no, dataset=row["dataset"], condition="knowledge_blind_stem",
                rep=rep, seed=seed, pred=pred, correct=bool(correct), raw=raw[:800],
            ))
        if (i + 1) % 5 == 0:
            print(f"  ...{i + 1}/{len(subset)} questions done", flush=True)
    print(f"Done. Results -> {RESULTS_OUT}", flush=True)


def run_labeling(questions: pd.DataFrame):
    if not RESULTS_OUT.exists():
        print(f"No results at {RESULTS_OUT}. Run generation first.")
        return
    rows = [json.loads(l) for l in RESULTS_OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["question_no"] = df["question_no"].astype(str)
    agg = (df.groupby("question_no")["correct"]
           .agg(knowledge_blind_stem="mean", n="count").reset_index())

    merged = questions[["question_no", "dataset", "label", "control", "knowledge_blind"]].merge(
        agg, on="question_no", how="left")
    merged["delta_stem_vs_control"] = merged["knowledge_blind_stem"] - merged["control"]
    merged["delta_stem_vs_choices_blind"] = merged["knowledge_blind_stem"] - merged["knowledge_blind"]
    merged.to_csv(OUT_DIR / "stem_only_knowledge_test_results.csv", index=False)

    n_done = merged["knowledge_blind_stem"].notna().sum()
    print(f"\n{n_done}/{len(merged)} questions have stem-only results")
    print("\n=== mean rates across ALL questions with results ===")
    have = merged[merged["knowledge_blind_stem"].notna()]
    cols = ["control", "knowledge_blind", "knowledge_blind_stem"]
    print(have[cols].mean().round(3).to_string())
    print(f"\nmean recovery, knowledge_blind (stem+choices) - control: "
          f"{(have['knowledge_blind']-have['control']).mean():+.3f}")
    print(f"mean recovery, knowledge_blind_stem (stem-only) - control: "
          f"{have['delta_stem_vs_control'].mean():+.3f}")
    print(f"mean difference, stem-only vs stem+choices brief: "
          f"{have['delta_stem_vs_choices_blind'].mean():+.3f}")
    print("\n=== by Phase 2 label ===")
    print(have.groupby("label")[cols].agg(["mean", "count"]).round(3).to_string())
    print(f"\n-> {OUT_DIR / 'stem_only_knowledge_test_results.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="ollama", choices=["local", "ollama", "mock"])
    ap.add_argument("--model-id", default="qwen2.5:7b-instruct")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--label-only", action="store_true")
    ap.add_argument("--knowledge-limited-only", action="store_true",
                    help="restrict to the 36 knowledge-limited questions only "
                         "(default: run on ALL 300 labeled questions)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override output directory (default: interventions/) -- IMPORTANT: use "
                         "a distinct directory per model (e.g. interventions_llama8b/) to avoid "
                         "mixing or overwriting another model's briefs/results/labels")
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="override debate-workbook directory used to read question text/gold "
                         "(default: the Qwen RUNS_DIR). Point this at a model's own runs/ dir "
                         "(e.g. data/debate_llama_8b/runs) for a clean per-model replication. "
                         "Question text is model-independent, so this is for provenance/cleanliness.")
    args = ap.parse_args()

    global OUT_DIR, LABELS_PATH, BRIEFS_OUT, RESULTS_OUT, RUNS_DIR
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
        LABELS_PATH = OUT_DIR / "intervention_labels.csv"
        BRIEFS_OUT = OUT_DIR / "stem_only_knowledge_briefs.jsonl"
        RESULTS_OUT = OUT_DIR / "stem_only_solve_results.jsonl"
    if args.runs_dir is not None:
        RUNS_DIR = args.runs_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_knowledge_limited_questions(only_knowledge_limited=args.knowledge_limited_only)
    print(f"Loaded {len(questions)} questions "
          f"({'knowledge-limited only' if args.knowledge_limited_only else 'ALL labels'})")

    if args.label_only:
        run_labeling(questions)
        return

    run_generation(args, questions)
    run_labeling(questions)


if __name__ == "__main__":
    main()
