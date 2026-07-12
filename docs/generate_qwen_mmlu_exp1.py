"""Batch-friendly MMLU-Pro debate generation for the Qwen Table 1 analogue.

The original wrapper produced one 200-question, one-seed workbook. This wrapper
can run either a single seed for quick experiments or repeated seeds for the
paper-style objective protocol.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q_CNT=200
DEFAULT_OUT_DIR = Path("data")
DEFAULT_OUTPUT = DEFAULT_OUT_DIR / "qwen_mmlu_exp1.xlsx"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_LEGACY_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
# Use repeated --seed flags, for example --seed 7 --seed 17, for multi-seed runs.
DEFAULT_SEEDS = [7]
REQUIRED_RUN_SHEETS = {"Debate_Traces", "Reasoning_Quality", "Diagnostic_Scores"}
PROTECTED_OUTPUTS = {
    Path("data/qwen_mmlu_pro_50/qwen_mmlu_pro_debate_traces.xlsx").resolve(),
}
METRIC_COLUMNS = [
    "rounds_to_consensus",
    "engagement",
    "responsiveness",
    "influence_asymmetry_inv",
    "balance",
    "stability",
    "group_welfare",
    "avg_process_metrics",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Qwen MMLU-Pro batch wrapper."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Qwen MMLU-Pro debate workbook. Defaults to one seed; "
            "repeat --seed for multi-seed runs."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--objective-limit", type=int, default=Q_CNT)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Seed to run. Repeat for multiple runs. Defaults to one fixed seed.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "Qwen model id. The paper text names Qwen2.5-72B-Instruct; use "
            f"{DEFAULT_LEGACY_MODEL_ID} explicitly for the smaller local run."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--judge-max-new-tokens", type=int, default=220)
    parser.add_argument("--judge-batch-size", type=int, default=15)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing 200-question output workbook.",
    )
    parser.add_argument(
        "--legacy-single-run",
        action="store_true",
        help=(
            "Reproduce the old one-seed 200-question run. This is not the "
            "paper-style 100x5 objective protocol."
        ),
    )
    parser.add_argument(
        "--allow-missing-source",
        action="store_true",
        help=(
            "Allow running when qwen_methodology_code.py is missing. This is "
            "discouraged because Python will not reliably import stale pyc-only "
            "modules and the run is not reproducible."
        ),
    )
    return parser.parse_args()


def load_qwen_methodology(allow_missing_source: bool):
    """Import the main Qwen methodology runner from the sibling source file."""
    source = Path("qwen_methodology_code.py")
    if not source.exists() and not allow_missing_source:
        raise RuntimeError(
            "qwen_methodology_code.py is missing. Restore the source file before "
            "rerunning so the generated workbook is reproducible and not tied to "
            "stale __pycache__ bytecode."
        )
    try:
        from qwen_methodology_code import qwen_methodology_main
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import qwen_methodology_code. Restore qwen_methodology_code.py "
            "or run from the directory that contains it."
        ) from exc
    return qwen_methodology_main


def require_gpu() -> None:
    """Fail early if PyTorch cannot see a CUDA GPU for the transformers backend."""
    if shutil.which("nvidia-smi") is None:
        print("Warning: nvidia-smi is not on PATH; relying on PyTorch CUDA check.", flush=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen transformers backend.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU was required, but PyTorch cannot see CUDA. "
            "Submit the batch job with a GPU allocation and rerun."
        )
    visible = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    print(f"PyTorch CUDA visible: {visible}", flush=True)


def spearman(series: pd.Series, accuracy: pd.Series) -> tuple[float, int]:
    """Compute Spearman correlation between a metric column and accuracy."""
    paired = pd.DataFrame(
        {
            "metric": pd.to_numeric(series, errors="coerce"),
            "accuracy": pd.to_numeric(accuracy, errors="coerce"),
        }
    ).dropna()
    if len(paired) < 3 or paired["metric"].nunique() <= 1 or paired["accuracy"].nunique() <= 1:
        return np.nan, len(paired)
    rho = float(np.corrcoef(paired["metric"].rank(), paired["accuracy"].rank())[0, 1])
    return rho, len(paired)


def aggregate_report(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnostic score columns by dataset type and stance mode."""
    if scores.empty or not {"dataset_type", "stance_mode"}.issubset(scores.columns):
        return pd.DataFrame()
    numeric = [
        "accuracy",
        "engagement",
        "responsiveness",
        "influence_asymmetry",
        "influence_asymmetry_inv",
        "balance",
        "stability",
        "group_welfare",
        "avg_process_metrics",
    ]
    rows = []
    for keys, group in scores.groupby(["dataset_type", "stance_mode"], dropna=False):
        row = {"dataset_type": keys[0], "stance_mode": keys[1]}
        for column in numeric:
            values = pd.to_numeric(
                group[column] if column in group.columns else pd.Series(dtype=float),
                errors="coerce",
            )
            row[f"{column}_count"] = int(values.count())
            row[f"{column}_mean"] = float(values.mean()) if values.count() else np.nan
            row[f"{column}_std"] = float(values.std()) if values.count() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def objective_correlation_report(scores: pd.DataFrame) -> pd.DataFrame:
    """Report objective-question metric correlations against final-answer accuracy."""
    required = {"dataset_type", "accuracy", *METRIC_COLUMNS}
    if scores.empty or not required.issubset(scores.columns):
        return pd.DataFrame()
    objective = scores[scores["dataset_type"] == "objective"].copy()
    rows = []
    for metric in METRIC_COLUMNS:
        rho, n = spearman(objective[metric], objective["accuracy"])
        rows.append({"signal_metric": metric, "spearman_rho_with_accuracy": rho, "n": n})
    return pd.DataFrame(rows)


def find_run_workbook(run_dir: Path) -> Path:
    """Locate the generated workbook for one seed's run directory."""
    candidates = [
        run_dir / DEFAULT_OUTPUT.name,
        run_dir / "qwen_mmlu_pro_debate_traces.xlsx",
        run_dir / "qwen_debate_traces.xlsx",
    ]
    candidates.extend(sorted(run_dir.glob("qwen_mmlu_exp*.xlsx")))
    candidates.extend(sorted(run_dir.glob("*mmlu_pro_debate_traces.xlsx")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No generated workbook found in {run_dir}")


def workbook_has_required_sheets(path: Path) -> bool:
    """Check whether a workbook has the sheets needed to be considered complete."""
    try:
        sheet_names = set(pd.ExcelFile(path).sheet_names)
    except Exception:
        return False
    return REQUIRED_RUN_SHEETS.issubset(sheet_names)


def append_run_columns(df: pd.DataFrame, run_id: int, seed: int) -> pd.DataFrame:
    """Add run metadata columns before combining per-seed workbook sheets."""
    out = df.copy()
    out.insert(0, "Run ID", run_id)
    out.insert(1, "Seed", seed)
    return out


def combine_workbooks(output_path: Path, run_paths: list[Path], seeds: list[int], args: argparse.Namespace) -> None:
    """Merge per-seed Qwen workbooks into one multi-seed Excel report."""
    debate_frames = []
    judgment_frames = []
    score_frames = []
    metadata_rows = [
        {"field": "run_mode", "value": "qwen_transformers_multi_seed"},
        {"field": "model_id", "value": args.model_id},
        {"field": "dataset_source", "value": "mmlu-pro"},
        {"field": "objective_questions_per_seed", "value": args.objective_limit},
        {"field": "num_seeds", "value": len(seeds)},
        {"field": "seeds", "value": ",".join(str(seed) for seed in seeds)},
        {"field": "total_expected_debates", "value": args.objective_limit * len(seeds)},
        {"field": "paper_protocol_alignment", "value": "100 questions x 5 seeds by default"},
        {"field": "objective_metric_definition", "value": "Objective MCQ rows use an explicit categorical adaptation, not the paper's Likert stance equations."},
        {"field": "reasoning_quality_context", "value": "Evaluator receives only the current agent justification text."},
        {"field": "parse_failure_policy", "value": "Agent turns are re-prompted up to two times; debates with unresolved answer/explanation parse failures are discarded."},
        {"field": "final_answer_fallback_required", "value": "moderator fallback; majority_vote_no_moderator should be investigated"},
        {"field": "temperature", "value": args.temperature},
        {"field": "top_p", "value": args.top_p},
    ]

    for run_id, (path, seed) in enumerate(zip(run_paths, seeds), start=1):
        debate = pd.read_excel(path, sheet_name="Debate_Traces")
        judgments = pd.read_excel(path, sheet_name="Reasoning_Quality")
        scores = pd.read_excel(path, sheet_name="Diagnostic_Scores")

        debate_frames.append(append_run_columns(debate, run_id, seed))
        judgment_frames.append(append_run_columns(judgments, run_id, seed))
        score = append_run_columns(scores, run_id, seed)
        score["row_index"] = np.arange(len(score_frames) * args.objective_limit, len(score_frames) * args.objective_limit + len(score))
        score_frames.append(score)

    debates = pd.concat(debate_frames, ignore_index=True)
    judgments = pd.concat(judgment_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)

    if "Final Answer Source" in debates.columns:
        source_counts = debates["Final Answer Source"].value_counts(dropna=False)
        for source, count in source_counts.items():
            metadata_rows.append({"field": f"final_answer_source_count:{source}", "value": int(count)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        debates.to_excel(writer, sheet_name="Debate_Traces", index=False)
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Run_Metadata", index=False)
        judgments.to_excel(writer, sheet_name="Reasoning_Quality", index=False)
        scores.to_excel(writer, sheet_name="Diagnostic_Scores", index=False)
        aggregate_report(scores).to_excel(writer, sheet_name="Aggregate_Summary", index=False)
        objective_correlation_report(scores).to_excel(writer, sheet_name="Objective_Correlations", index=False)


def run_one(qwen_methodology_main, run_dir: Path, seed: int, args: argparse.Namespace) -> Path:
    """Run or reuse one seed's Qwen methodology workbook."""
    if args.resume:
        try:
            existing = find_run_workbook(run_dir)
        except FileNotFoundError:
            existing = None
        if existing is not None and workbook_has_required_sheets(existing):
            print(f"Reusing completed seed {seed} workbook at {existing}", flush=True)
            return existing

    sys.argv = [
        "qwen_methodology_code.py",
        "--llm-provider",
        "qwen",
        "--backend",
        "transformers",
        "--model-id",
        args.model_id,
        "--dataset-source",
        "mmlu-pro",
        "--mmlu-pro-dataset",
        "TIGER-Lab/MMLU-Pro",
        "--mmlu-pro-split",
        "test",
        "--objective-limit",
        str(args.objective_limit),
        "--subjective-limit",
        "0",
        "--rounds",
        str(args.rounds),
        "--seed",
        str(seed),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--judge-max-new-tokens",
        str(args.judge_max_new_tokens),
        "--judge-batch-size",
        str(args.judge_batch_size),
        "--require-gpu",
        "--device-map",
        "auto",
        "--torch-dtype",
        "auto",
        "--sleep",
        str(args.sleep),
        "--out-dir",
        str(run_dir),
    ]
    if args.resume:
        sys.argv.append("--resume")

    print(f"Running seed {seed}; writing intermediate workbook under {run_dir}", flush=True)
    qwen_methodology_main()
    return find_run_workbook(run_dir)


def main() -> None:
    """Coordinate argument parsing, per-seed runs, and final workbook assembly."""
    args = parse_args()
    qwen_methodology_main = load_qwen_methodology(args.allow_missing_source)
    out_dir = args.out_dir
    output_path = out_dir / DEFAULT_OUTPUT.name
    resolved_output = output_path.resolve()
    seeds = args.seed if args.seed is not None else list(DEFAULT_SEEDS)

    if resolved_output in PROTECTED_OUTPUTS:
        raise RuntimeError(f"Refusing to write to protected existing output: {output_path}")
    if output_path.exists() and not args.overwrite:
        if args.resume and workbook_has_required_sheets(output_path):
            print(f"Final workbook already exists at {output_path}; nothing to resume.", flush=True)
            return
        raise RuntimeError(
            f"{output_path} already exists. Move it, delete it, or rerun with --overwrite."
        )
    if args.legacy_single_run:
        if args.seed is None:
            seeds = [7]
        #args.objective_limit = 200
        args.objective_limit = Q_CNT
        if args.model_id == DEFAULT_MODEL_ID:
            args.model_id = DEFAULT_LEGACY_MODEL_ID

    require_gpu()

    if args.legacy_single_run:
        run_path = run_one(qwen_methodology_main, out_dir, seeds[0], args)
        if run_path != output_path:
            shutil.copy2(run_path, output_path)
        print(f"Wrote legacy single-run workbook to {output_path}", flush=True)
        return

    run_paths = []
    runs_dir = out_dir / "runs"
    for run_id, seed in enumerate(seeds, start=1):
        run_dir = runs_dir / f"seed_{seed}"
        run_paths.append(run_one(qwen_methodology_main, run_dir, seed, args))

    combine_workbooks(output_path, run_paths, seeds, args)
    print(f"Wrote combined {args.objective_limit}x{len(seeds)} workbook to {output_path}", flush=True)


if __name__ == "__main__":
    main()
