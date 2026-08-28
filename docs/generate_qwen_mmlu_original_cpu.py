"""LOCAL CPU / Ollama version of generate_qwen_mmlu_original.py.

Same configuration as generate_qwen_mmlu_original.py:
  - Original 4-choice MMLU (cais/mmlu), broad subject sampling.
  - COOPERATIVE distinct role personas (Agent1=expert, Agent2=lead/decision-maker,
    Agent3=practitioner), domain-adapted from the MMLU subject. No adversarial
    "must disagree" dynamic.
  - Per-agent sampling temperature (default 0.4 / 0.7 / 1.0).
  - Pipeline's default judge (same as the GPU version).

DIFFERENCE: runs generation + judging on the LOCAL CPU via **Ollama** with a
small **qwen2.5:7b-instruct** model, instead of transformers + CUDA. No GPU flags.

Model-scale note: 7B is smaller than the 14B used elsewhere and far below the
paper's 72B/GPT-4o/Gemini -- use this for a no-GPU smoke test of the cooperative
role-persona + per-agent-temperature setup, not as evidence about scale.

PREREQUISITES
-------------
  1. Install Ollama (OllamaSetup.exe is in the repo root) and start it.
  2. Pull the model:   ollama pull qwen2.5:7b-instruct
  3. `pip install datasets truststore` (datasets for MMLU; truststore for the
     corporate-SSL fix, handled in generate_qwen_mmlu_original).

PERFORMANCE: CPU 7B is slow and per-agent temperature runs agents sequentially
(~3000 calls for 100 questions). Start small:  --objective-limit 20.

USAGE (from C:/Proj1/docs):
    python generate_qwen_mmlu_original_cpu.py --seed 7 --objective-limit 20
    # full:
    python generate_qwen_mmlu_original_cpu.py --seed 7 --seed 17 --seed 42 \
        --objective-limit 100 --agent-temps 0.4,0.7,1.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Localhost (Ollama) must bypass any corporate HTTP proxy, or urllib routes
# 127.0.0.1 through the proxy and hangs until the 600s timeout. Internet traffic
# (HuggingFace dataset download) still uses the proxy.
_LOCALHOSTS = "127.0.0.1,localhost,::1"
for _proxy_var in ("no_proxy", "NO_PROXY"):
    _current = os.environ.get(_proxy_var, "")
    os.environ[_proxy_var] = (
        f"{_current},{_LOCALHOSTS}".strip(",") if _current else _LOCALHOSTS
    )

import qwen_methodology_code as qmc

# Reuse the validated installers from the GPU version (loader + cooperative
# role personas + per-agent temperature). Importing also runs its truststore
# SSL bootstrap.
from generate_qwen_mmlu_original import (
    install_original_mmlu,
    install_cooperative_personas,
    install_per_agent_temperature,
    _MMLU_DATASET,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the CPU/Ollama original-MMLU runner."""
    p = argparse.ArgumentParser(
        description="Original MMLU + cooperative role personas + per-agent temperature, "
                    "on LOCAL CPU via Ollama (Qwen2.5-7B).")
    p.add_argument("--out-dir", type=Path, default=Path("data/mmlu_original_cpu"))
    p.add_argument("--objective-limit", type=int, default=100,
                   help="Questions per seed. Start small (e.g. 20) on CPU.")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, action="append", default=None,
                   help="Debate seed. Repeat for multiple seeds. Default: 7 17 42.")
    p.add_argument("--sample-seed", type=int, default=20240101,
                   help="Seed for the (fixed-across-debate-seeds) question sample.")
    p.add_argument("--subject", action="append", default=None,
                   help="Restrict to MMLU subject(s). Omit for broad cross-subject sampling.")
    p.add_argument("--agent-temps", default="0.4,0.7,1.0",
                   help="Comma-separated sampling temperatures for Agent1,Agent2,Agent3.")
    p.add_argument("--model-id", default="qwen2.5:7b-instruct",
                   help="Ollama model TAG (not an HF path). Pull it first: "
                        "`ollama pull qwen2.5:7b-instruct`.")
    p.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    p.add_argument("--split", default="test")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Fallback temperature (judge + any unmapped agent).")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--judge-max-new-tokens", type=int, default=220)
    p.add_argument("--judge-batch-size", type=int, default=1,
                   help="Ollama runs sequentially on CPU; batching gives no speedup.")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def run_one_seed_cpu(args: argparse.Namespace, seed: int) -> None:
    """Invoke the pipeline entrypoint for one seed on the Ollama/CPU backend."""
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
    print(f"[original-cpu] Running seed {seed} on Ollama ({args.model_id}) -> {run_dir}",
          flush=True)
    qmc.qwen_methodology_main()


def main() -> None:
    """Install loader + cooperative personas + per-agent temps, run per seed on CPU."""
    args = parse_args()
    seeds = args.seed if args.seed is not None else [7, 17, 42]

    install_original_mmlu(args.sample_seed, args.subject)
    temps_list = [float(x) for x in str(args.agent_temps).split(",") if x.strip()]
    temps = {f"Agent{i + 1}": t for i, t in enumerate(temps_list)}
    install_cooperative_personas()
    install_per_agent_temperature(temps)

    print("[original-cpu] Config: original MMLU + cooperative role personas + per-agent "
          "temps, Ollama/CPU, model=" + args.model_id + ".", flush=True)

    for seed in seeds:
        run_one_seed_cpu(args, seed)

    print("[original-cpu] Done. Compare against baseline_v2 with "
          "compare_adversarial_vs_baseline.py.", flush=True)


if __name__ == "__main__":
    main()
