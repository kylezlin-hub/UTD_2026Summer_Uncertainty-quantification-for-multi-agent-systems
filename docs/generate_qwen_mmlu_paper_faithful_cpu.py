"""Paper-faithful reproduction runner -- LOCAL CPU / Ollama / Qwen2.5-7B.

Same faithful configuration as generate_qwen_mmlu_paper_faithful.py:
  - Benchmark: original MMLU (cais/mmlu), 4-choice, all subjects.
  - Protocol:  cooperative default (§A.3.4).
  - Judge:     context-rich, matching authors_code.py.
  - Stance:    letters + categorical (Likert pending author confirmation).

DIFFERENCE FROM THE GPU RUNNER
------------------------------
This copy runs generation + judging on the LOCAL CPU via **Ollama** with the
**qwen2.5:7b-instruct** model, instead of transformers + CUDA. It passes
`--backend ollama` and an Ollama model TAG, and omits all GPU flags
(--require-gpu / --device-map / --torch-dtype).

Model scale note: 7B is even smaller than the 14B used elsewhere and far below
the paper's 72B/GPT-4o/Gemini, so the model-scale gap is LARGER here. Use this
for a no-GPU smoke test / pipeline check, not as evidence about scale.

PREREQUISITES
-------------
  1. Install Ollama (OllamaSetup.exe is in the repo root) and start it.
  2. Pull the model:   ollama pull qwen2.5:7b-instruct
  3. Ollama serves on http://127.0.0.1:11434 by default.

PERFORMANCE
-----------
CPU inference of a 7B model is slow (~seconds per call; ~3000 calls for 100
questions). Start small:  --objective-limit 20  for a first pass.

USAGE (from C:/Proj1/docs):
    python generate_qwen_mmlu_paper_faithful_cpu.py --seed 7 --objective-limit 20
    # full run:
    python generate_qwen_mmlu_paper_faithful_cpu.py --seed 7 --seed 17 --seed 42 \
        --objective-limit 100

Writes one workbook per seed under <out-dir>/seed_<seed>/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure localhost (the Ollama server) bypasses any corporate HTTP proxy set in
# the environment. Without this, urllib routes 127.0.0.1 through the proxy and
# hangs until the 600s timeout (curl bypasses it, so Ollama looks "up"). Internet
# traffic (HuggingFace dataset download) still uses the proxy.
_LOCALHOSTS = "127.0.0.1,localhost,::1"
for _proxy_var in ("no_proxy", "NO_PROXY"):
    _current = os.environ.get(_proxy_var, "")
    os.environ[_proxy_var] = (
        f"{_current},{_LOCALHOSTS}".strip(",") if _current else _LOCALHOSTS
    )

import qwen_methodology_code as qmc

# Reuse the validated single-variable patches (no re-implementation).
from generate_qwen_mmlu_original import install_original_mmlu
import context_rich_judge

_MMLU_DATASET = "cais/mmlu"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the CPU/Ollama paper-faithful runner."""
    p = argparse.ArgumentParser(
        description="Paper-faithful MMLU reproduction on LOCAL CPU via Ollama (Qwen2.5-7B).")
    p.add_argument("--out-dir", type=Path, default=Path("data/mmlu_paper_faithful_cpu"))
    p.add_argument("--objective-limit", type=int, default=100,
                   help="Questions per seed. Start small (e.g. 20) on CPU.")
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
    p.add_argument("--model-id", default="qwen2.5:7b-instruct",
                   help="Ollama model TAG (not an HF path). Pull it first: "
                        "`ollama pull qwen2.5:7b-instruct`.")
    p.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    p.add_argument("--split", default="test")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--judge-max-new-tokens", type=int, default=220)
    p.add_argument("--judge-batch-size", type=int, default=1,
                   help="Ollama runs sequentially on CPU; batching gives no speedup.")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def run_one_seed_cpu(args: argparse.Namespace, seed: int) -> None:
    """Invoke the pipeline entrypoint for one seed on the Ollama/CPU backend.

    No GPU flags: uses --backend ollama with an Ollama model tag.
    """
    run_dir = args.out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "qwen_methodology_code.py",
        "--llm-provider", "qwen",
        "--backend", "ollama",
        "--model-id", args.model_id,
        "--ollama-host", args.ollama_host,
        "--dataset-source", "mmlu-pro",   # loader is monkeypatched to original MMLU
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
        "--sleep", str(args.sleep),
        "--out-dir", str(run_dir),
    ]
    if args.resume:
        sys.argv.append("--resume")
    print(f"[paper-faithful-cpu] Running seed {seed} on Ollama "
          f"({args.model_id}) -> {run_dir}", flush=True)
    qmc.qwen_methodology_main()


def main() -> None:
    """Install the faithful configuration, then run per seed on CPU/Ollama."""
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

    print("[paper-faithful-cpu] Config: original MMLU + cooperative protocol (§A.3.4) + "
          "context-rich judge (authors_code.py).", flush=True)
    print("[paper-faithful-cpu] Backend: Ollama on local CPU, model="
          f"{args.model_id}. Model-scale gap is LARGER at 7B; use for smoke tests.",
          flush=True)

    for seed in seeds:
        run_one_seed_cpu(args, seed)

    print("[paper-faithful-cpu] Done. Compare against baseline_v2 with "
          "compare_adversarial_vs_baseline.py.", flush=True)


if __name__ == "__main__":
    main()
