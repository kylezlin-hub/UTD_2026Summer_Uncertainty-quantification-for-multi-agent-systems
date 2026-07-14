"""generate_baseline_v2.py — Multi-seed, multi-dataset baseline for AIB research.

Generates the expanded baseline needed to train the learned minority predictor:

    MMLU-Pro: 200 questions x 3 seeds  (seeds 7, 17, 42)
    GPQA:     100 questions x 3 seeds  (seeds 7, 17, 42)

Each (dataset, seed) pair produces one per-seed workbook under:
    data/baseline_v2/runs/<dataset>_seed_<seed>/baseline_v2_<dataset>_s<seed>.xlsx

A combined workbook merging all runs is written to:
    data/baseline_v2/baseline_v2_combined.xlsx

Design choices (from Codex + Kylezlin review):
- Qwen2.5-14B-Instruct via HuggingFace transformers for debate + judging
- 3 agents, 5 rounds — consistent with Exp1-3b for comparability
- 3 seeds per dataset — multiplies minority situations without changing questions
- Checkpointing per question — safe to interrupt and resume
- Metadata records dataset, seed, question_id for grouped CV later

Usage:
    # Full run (all 6 seed×dataset combos, GPU required):
    python Pred_Minority/generate_baseline_v2.py --require-gpu

    # MMLU-Pro only, 1 seed (quick test):
    python Pred_Minority/generate_baseline_v2.py --dataset mmlu-pro --seeds 7
        --objective-limit 5 --require-gpu

    # Resume after interruption:
    python Pred_Minority/generate_baseline_v2.py --require-gpu --resume

    # Skip judging (faster, for debugging debates):
    python Pred_Minority/generate_baseline_v2.py --skip-judging --require-gpu
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root or from Pred_Minority/
sys.path.insert(0, str(Path(__file__).parent.parent / "docs"))

from qwen_methodology_code import (
    DebateQuestion,
    LocalQwenPipeline,
    OBJECTIVE_LABELS,
    QWEN_AGENTS,
    EPS,
    answer_is_valid,
    clean_text,
    coerce_options,
    empty_judgments,
    final_answer_correctness,
    first_consensus_round_for_answer,
    judge_debates_with_qwen,
    majority_answer,
    mmlu_pro_correct_label,
    parse_confidence,
    parse_qwen_turn,
    qwen_initial_messages,
    qwen_moderator_messages,
    qwen_reprompt_messages,
    qwen_update_messages,
    score_mixed_debates,
    write_qwen_excel_report,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID   = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUT_DIR    = Path("data/baseline_v2")
DEFAULT_SEEDS      = [7, 17, 42]
MMLU_PRO_LIMIT     = 200
GPQA_LIMIT         = 100
DEFAULT_ROUNDS     = 5
DEFAULT_TEMP       = 0.7
DEFAULT_TOP_P      = 0.9
DEFAULT_MAX_TOKENS = 220
DEFAULT_JUDGE_TOKENS = 220
DEFAULT_JUDGE_BATCH  = 15

REQUIRED_SHEETS = {
    "Debate_Traces",
    "Reasoning_Quality",
    "Diagnostic_Scores",
    "Run_Metadata",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate multi-seed, multi-dataset baseline for AIB predictor."
    )
    p.add_argument("--model-id",         default=DEFAULT_MODEL_ID)
    p.add_argument("--out-dir",          type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seeds",            type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seeds to run (default: 7 17 42)")
    p.add_argument("--dataset",          choices=["mmlu-pro", "gpqa", "both"],
                   default="both")
    p.add_argument("--objective-limit",  type=int, default=None,
                   help="Override question limit (default: 200 MMLU-Pro, 100 GPQA)")
    p.add_argument("--rounds",           type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--temperature",      type=float, default=DEFAULT_TEMP)
    p.add_argument("--top-p",            type=float, default=DEFAULT_TOP_P)
    p.add_argument("--max-new-tokens",   type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--judge-max-new-tokens", type=int, default=DEFAULT_JUDGE_TOKENS)
    p.add_argument("--judge-batch-size", type=int, default=DEFAULT_JUDGE_BATCH)
    p.add_argument("--device-map",       default="auto")
    p.add_argument("--torch-dtype",      default="auto")
    p.add_argument("--require-gpu",      action="store_true")
    p.add_argument("--skip-judging",     action="store_true",
                   help="Skip LLM judging (produces debates only; use for debug)")
    p.add_argument("--q-source",         choices=["llm", "confidence"], default="llm")
    p.add_argument("--metric-version",   choices=["paper", "corrected"], default="paper")
    p.add_argument("--sleep",            type=float, default=0.0)
    p.add_argument("--resume",           action="store_true",
                   help="Resume from per-question checkpoint (skips completed questions)")
    p.add_argument("--overwrite",        action="store_true",
                   help="Overwrite existing completed workbooks")
    p.add_argument("--mmlu-pro-dataset", default="TIGER-Lab/MMLU-Pro")
    p.add_argument("--mmlu-pro-split",   default="test")
    p.add_argument("--gpqa-dataset",     default="Idavidrein/gpqa",
                   help="HuggingFace GPQA dataset name")
    p.add_argument("--gpqa-subset",      default="gpqa_main",
                   help="GPQA config: gpqa_main (198q) or gpqa_extended (546q)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def require_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        print("Warning: nvidia-smi not on PATH; relying on PyTorch check.", flush=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch required.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required but PyTorch cannot see CUDA. "
            "Run with --no-require-gpu to skip this check (CPU will be very slow)."
        )
    visible = ", ".join(
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    )
    print(f"CUDA GPU available: {visible}", flush=True)


# ---------------------------------------------------------------------------
# GPQA loader
# ---------------------------------------------------------------------------

def load_gpqa_questions(limit: int, dataset_name: str, subset: str) -> list[DebateQuestion]:
    """Load GPQA questions from HuggingFace.

    GPQA (Graduate-Level Google-Proof Q&A) contains 4-option multiple-choice
    science questions at graduate level.  Dataset: Idavidrein/gpqa
    Subsets: gpqa_main (198 questions), gpqa_extended (546 questions).
    """
    if limit <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "GPQA loading requires the `datasets` package. "
            "Install: pip install datasets"
        ) from exc

    print(f"Loading GPQA ({subset}, limit={limit})...", flush=True)
    rows = list(load_dataset(dataset_name, subset, split="train"))

    questions: list[DebateQuestion] = []
    for idx, row in enumerate(rows):
        if len(questions) >= limit:
            break

        # GPQA columns: Question, Correct Answer, Incorrect Answer 1/2/3
        question_text = clean_text(row.get("Question", ""))
        correct_text  = clean_text(row.get("Correct Answer", ""))
        wrong1 = clean_text(row.get("Incorrect Answer 1", ""))
        wrong2 = clean_text(row.get("Incorrect Answer 2", ""))
        wrong3 = clean_text(row.get("Incorrect Answer 3", ""))

        if not question_text or not correct_text:
            continue

        # Shuffle options so correct answer is not always A
        # Use a deterministic shuffle based on question index so seeds are stable
        options_raw = [correct_text, wrong1, wrong2, wrong3]
        rng = random.Random(idx)
        rng.shuffle(options_raw)
        options = [o for o in options_raw if o]
        if len(options) < 2:
            continue

        labels = OBJECTIVE_LABELS[: len(options)]
        correct_label = labels[options.index(correct_text)]

        formatted_options = "\n".join(
            f"{label}. {opt}" for label, opt in zip(labels, options)
        )
        question_no = clean_text(row.get("Record ID", "")) or f"gpqa-{idx + 1:04d}"

        questions.append(
            DebateQuestion(
                dataset_type="objective",
                question_no=str(question_no),
                question=f"{question_text}\n{formatted_options}",
                correct_answer=correct_label,
                answer_labels=labels,
                category=clean_text(row.get("High-level domain", "science")),
            )
        )

    print(f"Loaded {len(questions)} GPQA questions.", flush=True)
    return questions


# ---------------------------------------------------------------------------
# Per-question checkpoint (survive interruptions)
# ---------------------------------------------------------------------------

def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.jsonl"


def load_checkpoint(run_dir: Path) -> tuple[list[dict], set]:
    cp = checkpoint_path(run_dir)
    rows, done = [], set()
    if not cp.exists():
        return rows, done
    with cp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.append(entry["data"])
            done.add(entry["question_no"])
    print(f"Checkpoint: {len(done)} questions already completed.", flush=True)
    return rows, done


def save_checkpoint(run_dir: Path, question_no: str, debate_row: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with checkpoint_path(run_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"question_no": question_no, "data": debate_row}) + "\n")


# ---------------------------------------------------------------------------
# Single debate
# ---------------------------------------------------------------------------

def run_one_debate(
    llm: LocalQwenPipeline,
    question: DebateQuestion,
    seed: int,
    rounds: int,
    sleep: float,
    rng: random.Random,
) -> tuple[dict | None, str]:
    """Run one question through all rounds; return (debate_row, discard_reason)."""
    agent_order = QWEN_AGENTS[:]
    rng.shuffle(agent_order)
    all_rounds: list[dict] = []
    discard_reason = ""

    for round_no in range(1, rounds + 1):
        messages_by_agent = {}
        for agent in agent_order:
            if round_no == 1:
                messages_by_agent[agent] = qwen_initial_messages(question, agent)
            else:
                messages_by_agent[agent] = qwen_update_messages(
                    question, agent, all_rounds[-1]
                )

        round_turns: dict[str, dict] = {}
        for agent in agent_order:
            base_seed = seed + round_no * 100 + QWEN_AGENTS.index(agent) * 10
            raw = llm.complete(messages_by_agent[agent], seed=base_seed,
                               max_new_tokens=llm.max_new_tokens)
            parsed = parse_qwen_turn(
                raw, question.dataset_type, question.answer_labels, strict=True
            )
            # Retry once on parse failure
            if parsed["parse_failed"]:
                retry_msgs = qwen_reprompt_messages(messages_by_agent[agent], question)
                raw2 = llm.complete(retry_msgs, seed=base_seed + 1,
                                    max_new_tokens=llm.max_new_tokens)
                parsed2 = parse_qwen_turn(
                    raw2, question.dataset_type, question.answer_labels, strict=True
                )
                if not parsed2["parse_failed"]:
                    parsed = parsed2
                else:
                    discard_reason = (
                        f"{question.question_no} round {round_no} {agent}: "
                        f"{parsed['parse_error']}"
                    )
                    break
            round_turns[agent] = parsed

        if sleep:
            time.sleep(sleep)
        if discard_reason:
            print(f"  Discarding: {discard_reason}", flush=True)
            return None, discard_reason

        all_rounds.append(round_turns)

    # Determine final answer
    final_answers = [all_rounds[-1][agent]["answer"] for agent in QWEN_AGENTS]
    valid = [a for a in final_answers if a]
    if valid and len(set(valid)) == 1:
        final_answer, final_source = valid[0], "agent_consensus"
    else:
        raw = llm.complete(
            qwen_moderator_messages(question, all_rounds),
            seed=seed + rounds * 100 + 999
        )
        parsed = parse_qwen_turn(
            raw, question.dataset_type, question.answer_labels, strict=False
        )
        candidate = parsed["answer"]
        if answer_is_valid(candidate, question):
            final_answer, final_source = candidate, "moderator"
        else:
            final_answer, final_source = majority_answer(final_answers), "majority_vote_no_moderator"

    row: dict = {
        "Question #":          question.question_no,
        "Dataset Type":        question.dataset_type,
        "Dataset Category":    question.category,
        "Question":            question.question,
        "Correct Answer":      question.correct_answer,
        "Final Answer":        final_answer,
        "Final Answer Source": final_source,
        "Fixture Pattern":     "baseline",
        "Rounds to Consensus": first_consensus_round_for_answer(all_rounds, final_answer),
    }
    for round_no, turns in enumerate(all_rounds, start=1):
        for agent in QWEN_AGENTS:
            t = turns[agent]
            row[f"R{round_no} {agent} Answer"]   = t["answer"]
            row[f"R{round_no} {agent} Conf"]     = t["confidence"]
            row[f"R{round_no} {agent} Response"] = t["response"]

    return row, ""


# ---------------------------------------------------------------------------
# Run all debates for one (dataset, seed) pair
# ---------------------------------------------------------------------------

def run_debates_for_seed(
    llm: LocalQwenPipeline,
    questions: list[DebateQuestion],
    seed: int,
    run_dir: Path,
    args: argparse.Namespace,
    dataset_label: str,
) -> pd.DataFrame:
    """Run debates for one seed, using checkpoint to skip completed questions."""
    rows, done_nos = load_checkpoint(run_dir) if args.resume else ([], set())
    rng = random.Random(seed)

    for q_idx, question in enumerate(questions):
        # Always advance rng so ordering is deterministic even when skipping
        rng_state_before = rng.getstate()

        if question.question_no in done_nos:
            # Still need to advance rng for this question's shuffle
            tmp_rng = random.Random(seed)
            tmp_rng.setstate(rng_state_before)
            tmp_rng.shuffle(QWEN_AGENTS[:])  # consume same rng calls
            rng.setstate(tmp_rng.getstate())
            print(
                f"{dataset_label} seed={seed} q {q_idx+1}/{len(questions)}: "
                f"{question.question_no} [skip — checkpoint]",
                flush=True,
            )
            continue

        print(
            f"{dataset_label} seed={seed} q {q_idx+1}/{len(questions)}: "
            f"{question.question_no}",
            flush=True,
        )

        debate_row, discard_reason = run_one_debate(
            llm, question,
            seed=seed + q_idx * 1000,
            rounds=args.rounds,
            sleep=args.sleep,
            rng=rng,
        )

        if debate_row is None:
            continue  # discarded due to parse failure

        rows.append(debate_row)
        save_checkpoint(run_dir, question.question_no, debate_row)

    debates = pd.DataFrame(rows)
    if not debates.empty:
        debates["Correct?"] = debates.apply(final_answer_correctness, axis=1)
    return debates


# ---------------------------------------------------------------------------
# Per-seed workbook writer
# ---------------------------------------------------------------------------

def workbook_is_complete(path: Path) -> bool:
    try:
        sheets = set(pd.ExcelFile(path).sheet_names)
    except Exception:
        return False
    return REQUIRED_SHEETS.issubset(sheets)


def run_one_seed(
    llm: LocalQwenPipeline,
    questions: list[DebateQuestion],
    seed: int,
    dataset_label: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Generate, judge, score, and save one (dataset, seed) workbook."""
    output_path = run_dir / f"baseline_v2_{dataset_label}_s{seed}.xlsx"
    source_file = output_path.name

    # Resume: reuse completed workbook
    if args.resume and not args.overwrite:
        if output_path.exists() and workbook_is_complete(output_path):
            print(f"Reusing completed workbook: {output_path}", flush=True)
            return output_path

    run_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: debates
    print(f"\n--- Debates: {dataset_label} seed={seed} ---", flush=True)
    debates = run_debates_for_seed(llm, questions, seed, run_dir, args, dataset_label)

    if debates.empty:
        raise RuntimeError(
            f"No debates completed for {dataset_label} seed={seed}. "
            "Check GPU memory and parse-failure rate."
        )
    print(f"Completed {len(debates)} debates.", flush=True)

    # Phase 2: judge
    if args.skip_judging or args.q_source == "confidence":
        print("Skipping judging (--skip-judging or --q-source=confidence).", flush=True)
        judgments = empty_judgments()
    else:
        print(f"\n--- Judging: {dataset_label} seed={seed} ---", flush=True)
        judgments = judge_debates_with_qwen(
            llm,
            debates,
            source_file=source_file,
            seed=seed + 100_000,
            judge_max_new_tokens=args.judge_max_new_tokens,
            judge_batch_size=args.judge_batch_size,
            sleep=args.sleep,
        )
        print(f"Completed {len(judgments)} judgments.", flush=True)

    # Phase 3: score
    if judgments.empty and args.q_source == "llm":
        scores = pd.DataFrame()
    else:
        judgments["source_file"] = source_file
        scores = score_mixed_debates(
            debates, judgments, source_file,
            q_source=args.q_source,
            metric_version=args.metric_version,
        )

    # Phase 4: write workbook
    run_args = argparse.Namespace(**vars(args))
    run_args.backend     = "transformers"
    run_args.seed        = seed
    run_args.dataset_label = dataset_label
    write_qwen_excel_report(
        output_path,
        f"baseline_v2_{dataset_label}",
        run_args,
        debates,
        judgments,
        scores,
    )
    print(f"Wrote {output_path}", flush=True)
    return output_path


# ---------------------------------------------------------------------------
# Combine all per-seed workbooks
# ---------------------------------------------------------------------------

def combine_all(
    output_path: Path,
    run_specs: list[tuple[Path, str, int]],   # (path, dataset_label, seed)
    args: argparse.Namespace,
) -> None:
    """Merge all per-seed workbooks into one combined workbook."""
    debate_frames, judgment_frames, score_frames = [], [], []

    metadata_rows = [
        {"field": "run_mode",          "value": "baseline_v2_multi_seed_multi_dataset"},
        {"field": "model_id",          "value": args.model_id},
        {"field": "mmlu_pro_limit",    "value": MMLU_PRO_LIMIT},
        {"field": "gpqa_limit",        "value": GPQA_LIMIT},
        {"field": "seeds",             "value": ",".join(str(s) for s in args.seeds)},
        {"field": "num_seeds",         "value": len(args.seeds)},
        {"field": "rounds",            "value": args.rounds},
        {"field": "temperature",       "value": args.temperature},
        {"field": "top_p",             "value": args.top_p},
        {"field": "q_source",          "value": args.q_source},
        {"field": "design_note",       "value": (
            "Multi-seed baseline for AIB learned predictor. "
            "MMLU-Pro: 200q x 3 seeds. GPQA: 100q x 3 seeds. "
            "Group CV by (question_no, dataset_label). "
            "Use MMLU-Pro for training, GPQA for cross-domain evaluation."
        )},
    ]

    row_offset = 0
    for run_id, (path, dataset_label, seed) in enumerate(run_specs, start=1):
        debate   = pd.read_excel(path, sheet_name="Debate_Traces")
        judg     = pd.read_excel(path, sheet_name="Reasoning_Quality")
        score    = pd.read_excel(path, sheet_name="Diagnostic_Scores")

        # Tag each row with run provenance for grouped CV
        for df in (debate, judg, score):
            df.insert(0, "Run ID",       run_id)
            df.insert(1, "Seed",         seed)
            df.insert(2, "Dataset",      dataset_label)

        score["row_index"] = np.arange(row_offset, row_offset + len(score))
        debate_frames.append(debate)
        judgment_frames.append(judg)
        score_frames.append(score)
        row_offset += len(debate)

    debates   = pd.concat(debate_frames,   ignore_index=True)
    judgments = pd.concat(judgment_frames, ignore_index=True)
    scores    = pd.concat(score_frames,    ignore_index=True)

    if "Final Answer Source" in debates.columns:
        for src, cnt in debates["Final Answer Source"].value_counts(dropna=False).items():
            metadata_rows.append(
                {"field": f"final_answer_source_count:{src}", "value": int(cnt)}
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        debates  .to_excel(writer, sheet_name="Debate_Traces",    index=False)
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Run_Metadata", index=False)
        judgments.to_excel(writer, sheet_name="Reasoning_Quality", index=False)
        scores   .to_excel(writer, sheet_name="Diagnostic_Scores", index=False)

    print(f"\nWrote combined workbook: {output_path}", flush=True)
    print(f"  Debates:   {len(debates)}", flush=True)
    print(f"  Judgments: {len(judgments)}", flush=True)
    print(f"  Scores:    {len(scores)}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.require_gpu:
        require_gpu()

    # Determine which datasets to run
    run_datasets = []
    if args.dataset in ("mmlu-pro", "both"):
        run_datasets.append("mmlu-pro")
    if args.dataset in ("gpqa", "both"):
        run_datasets.append("gpqa")

    print("=" * 70, flush=True)
    print("BASELINE V2 — Multi-seed, multi-dataset", flush=True)
    print(f"  Datasets:  {run_datasets}", flush=True)
    print(f"  Seeds:     {args.seeds}", flush=True)
    print(f"  Model:     {args.model_id}", flush=True)
    print(f"  Rounds:    {args.rounds}", flush=True)
    print(f"  Judging:   {'skip' if args.skip_judging else args.q_source}", flush=True)
    print("=" * 70, flush=True)

    # Load model once (shared across all seeds/datasets)
    print("\nLoading Qwen model...", flush=True)
    llm = LocalQwenPipeline(
        model_id=args.model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        require_gpu=args.require_gpu,
    )
    print("Model loaded.\n", flush=True)

    run_specs: list[tuple[Path, str, int]] = []

    for dataset_label in run_datasets:
        # Load questions (same set for all seeds of this dataset)
        limit = args.objective_limit
        if dataset_label == "mmlu-pro":
            limit = limit or MMLU_PRO_LIMIT
            print(f"Loading MMLU-Pro questions (limit={limit})...", flush=True)
            questions = load_mmlu_pro_questions(
                limit,
                args.mmlu_pro_dataset,
                args.mmlu_pro_split,
                config=None,
                categories=None,
            )
        else:  # gpqa
            limit = limit or GPQA_LIMIT
            questions = load_gpqa_questions(
                limit,
                args.gpqa_dataset,
                args.gpqa_subset,
            )

        if not questions:
            print(f"WARNING: No questions loaded for {dataset_label}, skipping.", flush=True)
            continue

        print(f"Loaded {len(questions)} {dataset_label} questions.\n", flush=True)

        for seed in args.seeds:
            run_dir = args.out_dir / "runs" / f"{dataset_label}_seed_{seed}"
            path = run_one_seed(
                llm=llm,
                questions=questions,
                seed=seed,
                dataset_label=dataset_label,
                run_dir=run_dir,
                args=args,
            )
            run_specs.append((path, dataset_label, seed))

    if len(run_specs) > 1:
        combined_path = args.out_dir / "baseline_v2_combined.xlsx"
        combine_all(combined_path, run_specs, args)
    elif len(run_specs) == 1:
        print(f"\nOnly one run completed; combined workbook not needed.", flush=True)
        print(f"Output: {run_specs[0][0]}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("BASELINE V2 COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print("\nNext steps:", flush=True)
    print("  1. Extract prefix features:", flush=True)
    print("     python Pred_Minority/extract_features_prefix.py \\", flush=True)
    print("         --input data/baseline_v2/baseline_v2_combined.xlsx \\", flush=True)
    print("         --output Pred_Minority/features_v2.csv", flush=True)
    print("  2. Run exploratory analysis:", flush=True)
    print("     python Pred_Minority/train_exploratory.py \\", flush=True)
    print("         --features Pred_Minority/features_v2.csv \\", flush=True)
    print("         --output-dir Pred_Minority/exploratory_v2", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
