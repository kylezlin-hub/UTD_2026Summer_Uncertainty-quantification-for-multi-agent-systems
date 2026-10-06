"""control_singlebrief_test.py -- regenerate the brief-free CONTROL condition (8 solver draws
per question) in the same single-brief solver setup as oracle_singlebrief_test.py, so Figure 1C
pairs oracle vs control WITHIN one run (removes the cross-run baseline confound).

No brief generation (control is brief-free) -> no leak-retries, faster than the oracle run.
Output: interventions/control_singlebrief_solve_results.jsonl (condition "control_sb",
        fields question_no, dataset, rep 0-7, seed, pred, correct, raw)
Usage:  python control_singlebrief_test.py --backend ollama --model-id qwen2.5:7b-instruct
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "docs"))

from qwen_methodology_code import DebateQuestion  # noqa: E402
from generate_interventions import build_pipeline, solve_once, parse_options, infer_labels  # noqa: E402
from stem_only_knowledge_test import load_knowledge_limited_questions  # noqa: E402

OUT_DIR = HERE / "interventions"
RESULTS_OUT = OUT_DIR / "control_singlebrief_solve_results.jsonl"
DEFAULT_REPEATS = 8


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line); done.add((d["question_no"], d["rep"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["local", "ollama", "mock"])
    ap.add_argument("--model-id", default="qwen2.5:7b-instruct")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    llm = build_pipeline(args)
    questions = load_knowledge_limited_questions(only_knowledge_limited=False)
    if args.limit:
        questions = questions.head(args.limit)
    done = load_done(RESULTS_OUT)
    print(f"Single-brief CONTROL regen: {len(questions)} questions x {args.repeats} draws "
          f"(backend={args.backend}, model={args.model_id})", flush=True)

    for i, row in questions.reset_index(drop=True).iterrows():
        qno = str(row["question_no"])
        options = parse_options(row["question"])
        labels = infer_labels(row["question"])
        q = DebateQuestion("objective", qno, row["question"],
                           str(row["correct_answer"]).strip().upper(), labels, row.get("category", ""))
        for rep in range(args.repeats):
            if (qno, rep) in done:
                continue
            seed = args.seed + 1000 * rep + 555  # distinct offset from oracle (+777) and other conds
            pred, correct, raw = solve_once(llm, q, "control", {}, seed)
            append_jsonl(RESULTS_OUT, dict(
                question_no=qno, dataset=row["dataset"], condition="control_sb",
                rep=rep, seed=seed, pred=pred, correct=bool(correct), raw=raw[:800],
            ))
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(questions)} questions done", flush=True)
    print(f"Done. Results -> {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
