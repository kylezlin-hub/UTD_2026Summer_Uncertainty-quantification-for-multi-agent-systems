"""Paper-faithful debate generator: original MMLU + Liang-style adversarial protocol.

This is the closest we can get to the paper's Table 1 setup on local hardware.
It composes the two single-variable patches we already validated:

  1. original-MMLU question loader        (from generate_qwen_mmlu_original.py)
       -> 4-choice MMLU, sampled across ALL subjects, 100 questions
  2. Liang et al. 2024 adversarial prompts (from generate_qwen_mmlu_exp1_adversarial.py)
       -> opposing/committed personas + "it is NOT necessary to agree" tit-for-tat

The two patches touch disjoint parts of the pipeline (the objective loader vs.
the debate-prompt builders), so composing them is safe and changes nothing else:
same model, seeds, rounds, judge, metrics, scoring, and output format.

After this, the ONLY remaining divergence from the paper is model scale
(14B here vs. GPT-4o / Gemini-1.5-Pro / Qwen2.5-72B in the paper). So:

  - correlation reappears here  -> the null was benchmark + protocol, not scale
  - still null here             -> model scale is the prime suspect; run 72B next

USAGE (from C:/Proj1/docs; match --model-id to your other runs):
    python generate_qwen_mmlu_faithful.py --seed 7 --seed 17 --seed 42 \
        --model-id Qwen/Qwen2.5-14B-Instruct --objective-limit 100

Writes one workbook per seed under <out-dir>/seed_<seed>/. Feed the results to
compare_adversarial_vs_baseline.py against your baseline_v2 workbooks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import qwen_methodology_code as qmc

# Reuse the already-validated single-variable patches -- do NOT re-implement them.
from generate_qwen_mmlu_original import install_original_mmlu, _MMLU_DATASET
from generate_qwen_mmlu_exp1_adversarial import install_adversarial_protocol


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the paper-faithful generator."""
    p = argparse.ArgumentParser(
        description="Paper-faithful MMLU debate generator (original MMLU + Liang protocol).")
    p.add_argument("--out-dir", type=Path, default=Path("data/mmlu_faithful"))
    p.add_argument("--objective-limit", type=int, default=100, help="Paper uses 100.")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, action="append", default=None,
                   help="Debate seed. Repeat for multiple seeds. Default: 7 17 42.")
    p.add_argument("--sample-seed", type=int, default=20240101,
                   help="Seed for the (fixed-across-debate-seeds) question sample.")
    p.add_argument("--subject", action="append", default=None,
                   help="Restrict to MMLU subject(s). Omit for broad cross-subject sampling.")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct",
                   help="MUST match your other runs for a fair comparison.")
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
    print(f"[faithful] Running seed {seed} -> {run_dir}", flush=True)
    qmc.qwen_methodology_main()


def main() -> None:
    """Install BOTH patches (original MMLU + adversarial protocol), then run per seed."""
    args = parse_args()
    seeds = args.seed if args.seed is not None else [7, 17, 42]

    # Order does not matter: the two patches target disjoint pipeline functions.
    install_original_mmlu(args.sample_seed, args.subject)
    install_adversarial_protocol()
    print("[faithful] Paper-faithful condition active: original MMLU + Liang tit-for-tat. "
          "Only model scale still differs from the paper.", flush=True)

    for seed in seeds:
        run_one_seed(args, seed)
    print("[faithful] Done. Compare against baseline_v2 with compare_adversarial_vs_baseline.py.",
          flush=True)


if __name__ == "__main__":
    main()
