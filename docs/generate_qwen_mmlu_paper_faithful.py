"""Single paper-faithful reproduction runner.

Composes the validated pieces so the ONLY intended deviations from the paper are
model scale (14B vs 72B/GPT-4o/Gemini) and the still-open stance representation.

FAITHFUL ON:
  - Benchmark:  original MMLU (cais/mmlu), 4-choice, all subjects, 100 Q
                (install_original_mmlu, from generate_qwen_mmlu_original.py).
  - Protocol:   the paper's COOPERATIVE protocol (§A.3.4: "consider opposing
                arguments, update their stance if warranted") = the pipeline
                DEFAULT prompts. We deliberately do NOT install the Liang
                adversarial variant -- that is a separate hypothesis test,
                not the paper's protocol.
  - Judge:      context-rich judge matching the authors' released code
                (context_rich_judge.py): current + own-prev + peers'-prev,
                LLM-judge schema. (This matches authors_code.py, which differs
                from the paper's §A.3.6 ReCEval/context-blind prose -- see that
                module's header for why we follow the code, not the prose.)

NOT YET FAITHFUL (documented, blocked on author confirmation):
  - Stance representation. The paper (§A.3.5) uses a numeric 5-point Likert
    agreement stance; this runner still emits ANSWER LETTERS -> categorical
    metrics (binary engagement). Swapping to the Likert-agreement framing
    (generate_qwen_mmlu_likert.py) requires author confirmation of the MMLU->
    stance mapping. Until then `--stance likert` is intentionally blocked.

CAVEAT: the context-rich judge is a faithfulness fix; we already showed the
quality weight q does not move the correlations. The stance representation is
the lever that can actually change results.

USAGE (from C:/Proj1/docs; match --model-id across all runs):
    python generate_qwen_mmlu_paper_faithful.py --seed 7 --seed 17 --seed 42 \
        --model-id Qwen/Qwen2.5-14B-Instruct --objective-limit 100

Writes one workbook per seed under <out-dir>/seed_<seed>/. Feed results to
compare_adversarial_vs_baseline.py against baseline_v2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Reuse the already-validated single-variable patches (no re-implementation).
from generate_qwen_mmlu_original import install_original_mmlu, run_one_seed
import context_rich_judge


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the paper-faithful runner."""
    p = argparse.ArgumentParser(
        description="Paper-faithful MMLU reproduction (original MMLU + cooperative "
                    "protocol + authors' context-rich judge).")
    p.add_argument("--out-dir", type=Path, default=Path("data/mmlu_paper_faithful"))
    p.add_argument("--objective-limit", type=int, default=100, help="Paper uses 100.")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, action="append", default=None,
                   help="Debate seed. Repeat for multiple seeds. Default: 7 17 42.")
    p.add_argument("--sample-seed", type=int, default=20240101,
                   help="Seed for the (fixed-across-debate-seeds) question sample.")
    p.add_argument("--subject", action="append", default=None,
                   help="Restrict to MMLU subject(s). Omit for broad cross-subject sampling.")
    p.add_argument("--stance", choices=["categorical", "likert"], default="categorical",
                   help="'categorical' (letters, runnable now). 'likert' is blocked pending "
                        "author confirmation of the MMLU->Likert stance mapping.")
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


def main() -> None:
    """Install the faithful configuration, then run the pipeline once per seed."""
    args = parse_args()
    seeds = args.seed if args.seed is not None else [7, 17, 42]

    if args.stance == "likert":
        raise SystemExit(
            "--stance likert is not yet available: the paper's MMLU->Likert stance "
            "mapping is unconfirmed. Finish generate_qwen_mmlu_likert.py after the "
            "authors clarify §A.3.5 for objective tasks, then wire it here."
        )

    # 1) Benchmark: original 4-choice MMLU, broad subjects.
    install_original_mmlu(args.sample_seed, args.subject)
    # 2) Judge: authors' context-rich protocol.
    context_rich_judge.install_context_rich_judge()
    # 3) Protocol: paper's cooperative default -- do NOT install adversarial prompts.

    print("[paper-faithful] Config: original MMLU + cooperative protocol (§A.3.4) + "
          "context-rich judge (authors_code.py).", flush=True)
    print("[paper-faithful] Faithful on benchmark/protocol/judge. NOT faithful on stance "
          "(letters+categorical vs paper's Likert). Model scale also differs (14B).", flush=True)

    for seed in seeds:
        run_one_seed(args, seed)

    print("[paper-faithful] Done. Compare against baseline_v2 with "
          "compare_adversarial_vs_baseline.py.", flush=True)


if __name__ == "__main__":
    main()
