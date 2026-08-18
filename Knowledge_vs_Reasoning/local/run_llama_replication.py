"""run_llama_replication.py — Minimal Llama 3.1:8B replication experiment.

Two phases:
  Phase 1: Run 3-agent debates on the same 300 questions (3 datasets x 100, seed 7)
            to produce Llama debate workbooks and extract dynamics features.
  Phase 2: Run the intervention experiment on the SAME 200 labeled questions
            using Llama, to verify type-dependent recovery generalizes.

Usage:
    python run_llama_replication.py --phase 1   # debates only
    python run_llama_replication.py --phase 2   # interventions only
    python run_llama_replication.py --phase all  # both sequentially
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
MODEL_ID = "llama3.1:8b"
LLAMA_OUT_DIR = HERE / "data" / "debate_llama_8b"


def run_phase1():
    """Phase 1: Run debates with Llama (3 datasets, seed 7, 100 questions each)."""
    print("=" * 70)
    print(f"PHASE 1: Llama 3.1:8B Debates (seed 7, all datasets)")
    print(f"Output: {LLAMA_OUT_DIR}")
    print("=" * 70)

    cmd = [
        PYTHON, str(HERE / "generate_debate_local_7b.py"),
        "--backend", "ollama",
        "--model-id", MODEL_ID,
        "--out-dir", str(LLAMA_OUT_DIR),
        "--seeds", "7",
        "--dataset", "all",
        "--rounds", "5",
        "--temperature", "0.7",
    ]
    print(f"\nCommand: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        print(f"[ERROR] Phase 1 exited with code {result.returncode}")
        sys.exit(1)
    print("\n[Phase 1 COMPLETE]")


def run_phase2():
    """Phase 2: Run interventions on 300 questions with Llama.

    IMPORTANT: Generates FRESH briefs for Llama (not Qwen briefs).
    This tests whether the categorical structure (persistent vs responsive)
    is robust across brief instances and models simultaneously.
    """
    print("=" * 70)
    print(f"PHASE 2: Llama 3.1:8B Interventions (with FRESH briefs)")
    print(f"Model: {MODEL_ID}")
    print("=" * 70)

    llama_runs_dir = LLAMA_OUT_DIR / "runs"
    llama_interv_dir = HERE / "interventions_llama"

    cmd = [
        PYTHON, str(HERE / "generate_interventions.py"),
        "--backend", "ollama",
        "--model-id", MODEL_ID,
        "--datasets", "mmlu", "mmlu-pro", "gpqa",
        "--selection", "stratified",
        "--repeats", "8",
        "--seed", "7",
        "--runs-dir", str(llama_runs_dir),
        "--out-dir", str(llama_interv_dir),
        "--regenerate-briefs",  # CRITICAL: Generate independent briefs for Llama
    ]
    print(f"\nCommand: {' '.join(cmd)}\n")
    print("NOTE: This will take ~8-12 hours for full 6-condition x 300 question x 8 repeats.")
    print("      Fresh brief generation adds ~1-2 hours overhead.")
    print("      Consider running with --limit 50 for a quick validation first.\n")
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        print(f"[ERROR] Phase 2 exited with code {result.returncode}")
        sys.exit(1)
    print("\n[Phase 2 COMPLETE]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        run_phase1()
    if args.phase in ("2", "all"):
        run_phase2()
