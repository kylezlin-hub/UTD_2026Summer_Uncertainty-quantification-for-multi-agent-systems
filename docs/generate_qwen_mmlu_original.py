"""Original-MMLU debate generator -- reproduces the PAPER'S actual benchmark.

WHY THIS EXISTS
---------------
Your baseline/exp1 runs use MMLU-Pro (TIGER-Lab): up to 10 options, much harder,
and in the workbooks we inspected, filtered to a single "business" domain. The
paper uses ORIGINAL MMLU (Hendrycks et al. 2020): 4 options (A-D), and randomly
samples 100 questions ACROSS ALL SUBJECTS ("Question sets ... held fixed across
seeds", paper A.2).

This script runs the SAME pipeline / model / seeds / rounds / judge / metrics as
your baseline, changing ONLY the question set: original 4-choice MMLU, broad
subject sampling. It isolates "benchmark & difficulty" from "model scale".

    MMLU-Pro business (yours)   ->  hard, 10-choice, single domain, chance=10%
    MMLU broad (paper / this)   ->  easier, 4-choice, all subjects,  chance=25%

If the process->accuracy correlation reappears here at 14B, the null was driven
by the BENCHMARK, not model scale. If it stays null, model scale is implicated.

The question set is sampled ONCE with a fixed --sample-seed and held IDENTICAL
across debate seeds (matching the paper), so per-question aggregation across
seeds is valid -- feed the resulting workbooks straight into
compare_adversarial_vs_baseline.py.

USAGE (from C:/Proj1/docs; match --model-id to your baseline run):
    python generate_qwen_mmlu_original.py --seed 7 --seed 17 --seed 42 \
        --model-id Qwen/Qwen2.5-14B-Instruct --objective-limit 100

Writes one workbook per seed under <out-dir>/seed_<seed>/.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import qwen_methodology_code as qmc

# Sample the question set once, identically across debate seeds (paper protocol).
# Overridable via --sample-seed; kept separate from the debate RNG seed.
_SAMPLE_SEED = 20240101
_SUBJECT_FILTER: list[str] | None = None
_MMLU_DATASET = "cais/mmlu"
_MMLU_CONFIG = "all"


def load_original_mmlu_questions(
    limit: int,
    dataset_name: str,   # noqa: ARG001 - kept for signature parity with the patched fn
    split: str,
    config: str | None,  # noqa: ARG001
    categories: list[str] | None,  # noqa: ARG001 - we use module-level _SUBJECT_FILTER
) -> list["qmc.DebateQuestion"]:
    """Load original 4-choice MMLU, sampled across subjects. Same signature as the
    MMLU-Pro loader it replaces, so main() calls it unchanged."""
    if limit <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Original-MMLU loading requires the `datasets` package "
            "(`pip install datasets`)."
        ) from exc

    rows = list(load_dataset(_MMLU_DATASET, _MMLU_CONFIG, split=split))

    if _SUBJECT_FILTER:
        allowed = {s.casefold() for s in _SUBJECT_FILTER}
        rows = [r for r in rows if qmc.clean_text(r.get("subject")).casefold() in allowed]

    # Random sample across subjects, fixed & reproducible (paper: sets held fixed across seeds).
    rng = random.Random(_SAMPLE_SEED)
    rng.shuffle(rows)

    labels = qmc.OBJECTIVE_LABELS[:4]  # MMLU is always 4-way
    questions: list[qmc.DebateQuestion] = []
    for index, row in enumerate(rows):
        if len(questions) >= limit:
            break
        question_text = qmc.clean_text(row.get("question"))
        choices = row.get("choices") or []
        choices = [qmc.clean_text(c) for c in choices if qmc.clean_text(c)]
        if not question_text or len(choices) != 4:
            continue
        answer_idx = row.get("answer")
        try:
            correct_label = labels[int(answer_idx)]
        except (TypeError, ValueError, IndexError):
            continue  # skip rows with an unusable gold label
        formatted = "\n".join(f"{lab}. {opt}" for lab, opt in zip(labels, choices))
        subject = qmc.clean_text(row.get("subject"))
        question_no = f"mmlu-{subject or 'all'}-{index + 1:05d}"
        questions.append(
            qmc.DebateQuestion(
                "objective",
                question_no,
                f"{question_text}\n{formatted}",
                correct_label,
                labels,
                subject,
            )
        )
    if len(questions) < limit:
        print(f"Loaded {len(questions)} original-MMLU questions; requested {limit}.", flush=True)
    return questions


def install_original_mmlu(sample_seed: int, subjects: list[str] | None) -> None:
    """Point the pipeline's objective loader at original MMLU."""
    global _SAMPLE_SEED, _SUBJECT_FILTER
    _SAMPLE_SEED = sample_seed
    _SUBJECT_FILTER = subjects
    qmc.load_mmlu_pro_questions = load_original_mmlu_questions
    scope = ",".join(subjects) if subjects else "ALL subjects"
    print(f"[original-mmlu] Loader installed: cais/mmlu (4-choice), {scope}, "
          f"sample_seed={sample_seed}.", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the original-MMLU generator."""
    p = argparse.ArgumentParser(
        description="Original 4-choice MMLU debate generator (paper benchmark).")
    p.add_argument("--out-dir", type=Path, default=Path("data/mmlu_original"))
    p.add_argument("--objective-limit", type=int, default=100, help="Paper uses 100.")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, action="append", default=None,
                   help="Debate seed. Repeat for multiple seeds. Default: 7 17 42.")
    p.add_argument("--sample-seed", type=int, default=20240101,
                   help="Seed for the (fixed-across-debate-seeds) question sample.")
    p.add_argument("--subject", action="append", default=None,
                   help="Restrict to MMLU subject(s). Omit for broad cross-subject sampling.")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct",
                   help="MUST match your baseline_v2 model for a fair comparison.")
    p.add_argument("--split", default="test")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--judge-max-new-tokens", type=int, default=220)
    p.add_argument("--judge-batch-size", type=int, default=15)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def run_one_seed(args: argparse.Namespace, seed: int) -> None:
    """Invoke the unmodified pipeline entrypoint for one debate seed."""
    run_dir = args.out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "qwen_methodology_code.py",
        "--llm-provider", "qwen",
        "--backend", "transformers",
        "--model-id", args.model_id,
        "--dataset-source", "mmlu-pro",   # loader is monkeypatched; flag just satisfies the parser
        "--mmlu-pro-dataset", _MMLU_DATASET,
        "--mmlu-pro-split", args.split,
        "--objective-limit", str(args.objective_limit),
        "--subjective-limit", "0",
        "--rounds", str(args.rounds),
        "--seed", str(seed),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--max-new-tokens", str(args.max_new_tokens),
        "--judge-max-new-tokens", str(args.judge_max_new_tokens),
        "--judge-batch-size", str(args.judge_batch_size),
        "--require-gpu",
        "--device-map", "auto",
        "--torch-dtype", "auto",
        "--sleep", str(args.sleep),
        "--out-dir", str(run_dir),
    ]
    if args.resume:
        sys.argv.append("--resume")
    print(f"[original-mmlu] Running seed {seed} -> {run_dir}", flush=True)
    qmc.qwen_methodology_main()


def main() -> None:
    """Install the original-MMLU loader and run the pipeline once per seed."""
    args = parse_args()
    seeds = args.seed if args.seed is not None else [7, 17, 42]
    install_original_mmlu(args.sample_seed, args.subject)
    for seed in seeds:
        run_one_seed(args, seed)
    print("[original-mmlu] Done. Compare against baseline with "
          "compare_adversarial_vs_baseline.py (baseline vs these workbooks).", flush=True)


if __name__ == "__main__":
    main()
