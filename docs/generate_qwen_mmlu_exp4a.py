"""Experiment 4a: Rule-Triggered Devil's Advocate for Qwen MMLU-Pro.

This file started as a copy of generate_qwen_mmlu_exp1.py, but Experiment 4a
needs per-round prompt control. It therefore follows the direct generation
style of the later experiment scripts while keeping the same batch/report shape.

The 4a trigger intentionally does not use the diagnostic metrics from the
paper. It fires when the same majority answer persists for two completed rounds.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from qwen_methodology_code import (
    DebateQuestion,
    LocalQwenPipeline,
    OBJECTIVE_LABELS,
    QWEN_AGENTS,
    answer_is_valid,
    complete_qwen_turns_with_retry,
    empty_judgments,
    final_answer_correctness,
    first_consensus_round_for_answer,
    judge_debates_with_qwen,
    majority_answer,
    parse_qwen_turn,
    qwen_initial_messages,
    qwen_moderator_messages,
    qwen_update_messages,
    score_mixed_debates,
    write_qwen_excel_report,
)


DEFAULT_BASE_WORKBOOK = Path("data/qwen_mmlu_pro_200/qwen_mmlu_pro_debate_traces_base.xlsx")
DEFAULT_OUT_DIR = Path("data")
DEFAULT_OUTPUT = DEFAULT_OUT_DIR / "qwen_mmlu_exp4a.xlsx"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
Q_CNT = 200
DEFAULT_SEEDS = [7]
REQUIRED_RUN_SHEETS = {
    "Debate_Traces",
    "Reasoning_Quality",
    "Diagnostic_Scores",
    "Experiment_4a_State",
}
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
    """Parse command-line options for Experiment 4a."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 4a, Rule-Triggered Devil's Advocate, on the exact "
            "questions stored in the baseline Qwen MMLU-Pro workbook."
        )
    )
    parser.add_argument("--base-workbook", type=Path, default=DEFAULT_BASE_WORKBOOK)
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
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--judge-max-new-tokens", type=int, default=220)
    parser.add_argument("--judge-batch-size", type=int, default=15)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--skip-judging", action="store_true")
    parser.add_argument("--q-source", choices=["llm", "confidence"], default="llm")
    parser.add_argument("--metric-version", choices=["paper", "corrected"], default="paper")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-require-gpu", action="store_true")
    parser.add_argument(
        "--majority-window",
        type=int,
        default=2,
        help="Number of consecutive completed rounds with the same majority answer required to trigger.",
    )
    parser.add_argument(
        "--majority-min-support",
        type=int,
        default=2,
        help="Minimum number of agents supporting the majority answer in each trigger round.",
    )
    return parser.parse_args()


def require_gpu() -> None:
    """Fail early if PyTorch cannot see CUDA for the transformers backend."""
    if shutil.which("nvidia-smi") is None:
        print("Warning: nvidia-smi is not on PATH; relying on PyTorch CUDA check.", flush=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen transformers backend.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU was required, but PyTorch cannot see CUDA. Submit the batch job "
            "with a GPU allocation or rerun with --no-require-gpu for a local dry run."
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


def infer_answer_labels(question_text: str, correct_answer: str) -> tuple[str, ...]:
    """Infer allowed answer labels from formatted MMLU-Pro question text."""
    found = re.findall(r"(?m)^([A-J])\.\s+", question_text)
    labels = tuple(label for label in OBJECTIVE_LABELS if label in set(found))
    if correct_answer and correct_answer not in labels:
        labels = OBJECTIVE_LABELS[: max(4, OBJECTIVE_LABELS.index(correct_answer) + 1)]
    return labels or OBJECTIVE_LABELS


def load_questions_from_base_workbook(path: Path, limit: int) -> list[DebateQuestion]:
    """Load the exact baseline workbook questions as DebateQuestion objects."""
    if not path.exists():
        raise FileNotFoundError(f"Baseline workbook not found: {path}")
    debates = pd.read_excel(path, sheet_name="Debate_Traces")
    if limit is not None:
        debates = debates.head(limit)

    questions: list[DebateQuestion] = []
    required = {"Question #", "Dataset Type", "Question", "Correct Answer"}
    missing = required - set(debates.columns)
    if missing:
        raise ValueError(f"Baseline workbook is missing required columns: {sorted(missing)}")

    for _, row in debates.iterrows():
        dataset_type = str(row["Dataset Type"]).strip()
        if dataset_type != "objective":
            continue
        question_text = str(row["Question"]).strip()
        correct_answer = str(row["Correct Answer"]).strip().upper()
        questions.append(
            DebateQuestion(
                dataset_type="objective",
                question_no=row["Question #"],
                question=question_text,
                correct_answer=correct_answer,
                answer_labels=infer_answer_labels(question_text, correct_answer),
                category=str(row.get("Dataset Category", "")).strip(),
            )
        )
    if not questions:
        raise RuntimeError(f"No objective questions found in {path}")
    return questions


def answer_counts(turns: dict[str, dict[str, object]]) -> Counter:
    """Count valid-looking answer labels in one completed round."""
    return Counter(str(turns[agent].get("answer", "")).strip() for agent in QWEN_AGENTS if turns[agent].get("answer"))


def round_majority(turns: dict[str, dict[str, object]], min_support: int) -> tuple[str, int]:
    """Return the majority answer and support if it clears the support threshold."""
    counts = answer_counts(turns)
    if not counts:
        return "", 0
    answer, support = counts.most_common(1)[0]
    if support < min_support:
        return "", support
    return answer, support


def majority_persistence_state(
    rounds: list[dict[str, dict[str, str]]],
    args: argparse.Namespace,
) -> tuple[bool, str, int, int, dict[str, bool], str, str]:
    """Trigger when the same majority answer persists for the configured window."""
    flags = {agent: False for agent in QWEN_AGENTS}
    window = max(2, args.majority_window)
    if len(rounds) < window:
        return False, "", 0, 0, flags, "", ""

    recent_majorities = [round_majority(round_turns, args.majority_min_support) for round_turns in rounds[-window:]]
    recent_answers = [answer for answer, _support in recent_majorities]
    recent_supports = [support for _answer, support in recent_majorities]
    if not recent_answers or not recent_answers[0] or len(set(recent_answers)) != 1:
        return False, "", max(recent_supports or [0]), 0, flags, "", ""

    majority_now = recent_answers[0]
    previous_round = rounds[-1]
    for agent in QWEN_AGENTS:
        flags[agent] = str(previous_round[agent].get("answer", "")).strip() == majority_now

    resurrected = select_resurrected_answer(rounds, majority_now)
    reason = f"majority answer {majority_now} persisted for {window} rounds"
    return True, majority_now, min(recent_supports), window, flags, resurrected, reason


def select_resurrected_answer(rounds: list[dict[str, dict[str, str]]], majority_answer_now: str) -> str:
    """Choose a previously seen non-majority answer to re-evaluate, if one exists."""
    seen_order: list[str] = []
    for turns in rounds:
        for agent in QWEN_AGENTS:
            answer = str(turns[agent].get("answer", "")).strip()
            if answer and answer not in seen_order:
                seen_order.append(answer)
    for answer in seen_order:
        if answer != majority_answer_now:
            return answer
    return ""


def devils_advocate_update_messages(
    question: DebateQuestion,
    agent: str,
    previous_round: dict[str, dict[str, str]],
    intervention_flag: bool,
    majority_answer_now: str,
    resurrected_answer: str,
    trigger_reason: str,
) -> list[dict[str, str]]:
    """Build a normal update prompt with Exp4a Devil's Advocate context."""
    messages = qwen_update_messages(question, agent, previous_round)
    context = "\n\nExperiment 4a rule-triggered Devil's Advocate status: "
    if intervention_flag:
        alternative = (
            f" Re-evaluate previously rejected answer {resurrected_answer} as the strongest alternative."
            if resurrected_answer
            else " If no earlier alternative is available, construct the strongest plausible alternative answer."
        )
        context += (
            f"the non-diagnostic trigger fired because {trigger_reason}. Your current answer is aligned "
            f"with the persistent majority answer {majority_answer_now}. Before updating, argue against "
            "your own current answer, identify one hidden assumption behind your reasoning, and state the "
            "strongest counterargument to the majority position."
            + alternative
            + " Keep the majority answer only if it survives this self-challenge."
        )
    else:
        context += "no Devil's Advocate intervention is active for your next answer."
    patched = list(messages)
    patched[-1] = {**patched[-1], "content": patched[-1]["content"] + context}
    return patched


def select_final_answer_with_existing_logic(
    llm: LocalQwenPipeline,
    question: DebateQuestion,
    rounds: list[dict[str, dict[str, str]]],
    seed: int,
) -> tuple[str, str]:
    """Use the baseline final-answer path so only the intervention changes."""
    final_answers = [rounds[-1][agent]["answer"] for agent in QWEN_AGENTS]
    valid = [answer for answer in final_answers if answer]
    if valid and len(set(valid)) == 1:
        return valid[0], "agent_consensus"
    raw = llm.complete(qwen_moderator_messages(question, rounds), seed=seed)
    parsed = parse_qwen_turn(raw, question.dataset_type, question.answer_labels, strict=False)["answer"]
    if answer_is_valid(parsed, question):
        return parsed, "moderator"
    return majority_answer(final_answers), "majority_vote_no_moderator"


def make_experiment_debate_row(
    question: DebateQuestion,
    rounds: list[dict[str, dict[str, str]]],
    model_name: str,
    final_answer: str,
    final_answer_source: str,
) -> dict[str, object]:
    """Flatten one Experiment 4a debate into the baseline-compatible schema."""
    row: dict[str, object] = {
        "Question #": question.question_no,
        "Dataset Type": question.dataset_type,
        "Dataset Category": question.category,
        "Question": question.question,
        "Correct Answer": question.correct_answer,
        "Final Answer": final_answer,
        "Final Answer Source": final_answer_source,
        "Model": model_name,
        "Fixture Pattern": "rule_triggered_devils_advocate",
        "Rounds to Consensus": first_consensus_round_for_answer(rounds, final_answer),
    }
    for round_no, turns in enumerate(rounds, start=1):
        for agent in QWEN_AGENTS:
            turn = turns[agent]
            row[f"R{round_no} {agent} Answer"] = turn["answer"]
            row[f"R{round_no} {agent} Conf"] = turn["confidence"]
            row[f"R{round_no} {agent} Response"] = turn["response"]
    return row


def run_experiment_4a_debates(
    llm: LocalQwenPipeline,
    questions: list[DebateQuestion],
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Qwen debates with a two-round majority-persistence Devil's Advocate trigger."""
    rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    rng = random.Random(seed)

    for q_index, question in enumerate(questions):
        print(
            f"Experiment 4a generating question {q_index + 1}/{len(questions)}: {question.question_no}",
            flush=True,
        )
        agent_order = QWEN_AGENTS[:]
        rng.shuffle(agent_order)
        all_rounds: list[dict[str, dict[str, str]]] = []
        question_state_rows: list[dict[str, object]] = []
        discard_reason = ""

        for round_no in range(1, args.rounds + 1):
            trigger_active, majority_now, support, persisted_rounds, flags, resurrected, reason = (
                majority_persistence_state(all_rounds, args)
            )

            print(
                f"Experiment 4a question {q_index + 1} round {round_no}/{args.rounds}",
                flush=True,
            )
            messages_by_agent = {}
            for agent in agent_order:
                if round_no == 1:
                    messages_by_agent[agent] = qwen_initial_messages(question, agent)
                else:
                    messages_by_agent[agent] = devils_advocate_update_messages(
                        question,
                        agent,
                        all_rounds[-1],
                        flags[agent],
                        majority_now,
                        resurrected,
                        reason,
                    )

            seeds_by_agent = {
                agent: seed + q_index * 1000 + round_no * 100 + QWEN_AGENTS.index(agent) * 10
                for agent in agent_order
            }
            round_turns = complete_qwen_turns_with_retry(
                llm,
                messages_by_agent,
                question,
                seeds_by_agent,
            )
            for agent in agent_order:
                turn = round_turns[agent]
                if turn["parse_failed"]:
                    discard_reason = (
                        f"{question.question_no} round {round_no} {agent}: {turn['parse_error']}"
                    )
                    break
            if args.sleep:
                time.sleep(args.sleep)
            if discard_reason:
                print(f"Discarding debate after unresolved parse failure: {discard_reason}", flush=True)
                break

            previous_round = all_rounds[-1] if all_rounds else None
            current_majority, current_support = round_majority(round_turns, args.majority_min_support)
            for agent in QWEN_AGENTS:
                previous_answer = (
                    previous_round[agent]["answer"] if previous_round is not None else ""
                )
                current_answer = round_turns[agent]["answer"]
                question_state_rows.append(
                    {
                        "row_index": len(rows),
                        "question_no": question.question_no,
                        "round": round_no,
                        "agent": agent,
                        "previous_answer": previous_answer,
                        "answer": current_answer,
                        "current_round_majority_answer": current_majority,
                        "current_round_majority_support": current_support,
                        "majority_answer_before_turn": majority_now,
                        "majority_support_before_turn": support,
                        "majority_persisted_rounds_before_turn": persisted_rounds,
                        "resurrected_answer": resurrected,
                        "trigger_reason": reason,
                        "agent_in_persistent_majority": bool(flags[agent]),
                        "intervention_applied": bool(flags[agent]),
                        "answer_changed": bool(previous_answer and current_answer != previous_answer),
                        "left_persistent_majority": bool(
                            flags[agent] and majority_now and current_answer != majority_now
                        ),
                    }
                )
            all_rounds.append(round_turns)

        if discard_reason:
            continue

        final_answer, final_source = select_final_answer_with_existing_logic(
            llm,
            question,
            all_rounds,
            seed + q_index * 1000 + 999,
        )
        rows.append(
            make_experiment_debate_row(
                question,
                all_rounds,
                args.model_id,
                final_answer=final_answer,
                final_answer_source=final_source,
            )
        )
        state_rows.extend(question_state_rows)

    debates = pd.DataFrame(rows)
    if not debates.empty:
        debates["Correct?"] = debates.apply(final_answer_correctness, axis=1)
    return debates, pd.DataFrame(state_rows)


def append_experiment_state_sheet(output_path: Path, state: pd.DataFrame) -> None:
    """Append the Experiment_4a_State sheet after the shared report writer runs."""
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        state.to_excel(writer, sheet_name="Experiment_4a_State", index=False)


def combine_workbooks(output_path: Path, run_paths: list[Path], seeds: list[int], args: argparse.Namespace) -> None:
    """Merge per-seed Experiment 4a workbooks into one multi-seed Excel report."""
    debate_frames = []
    judgment_frames = []
    score_frames = []
    state_frames = []
    metadata_rows = [
        {"field": "run_mode", "value": "qwen_transformers_experiment_4a_rule_triggered_devils_advocate"},
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
        {"field": "experiment", "value": "rule_triggered_devils_advocate"},
        {"field": "trigger_uses_paper_diagnostics", "value": False},
        {"field": "trigger", "value": "same majority answer persists for consecutive rounds"},
        {"field": "majority_window", "value": args.majority_window},
        {"field": "majority_min_support", "value": args.majority_min_support},
        {"field": "temperature", "value": args.temperature},
        {"field": "top_p", "value": args.top_p},
    ]

    row_offset = 0
    for run_id, (path, seed) in enumerate(zip(run_paths, seeds), start=1):
        debate = pd.read_excel(path, sheet_name="Debate_Traces")
        judgments = pd.read_excel(path, sheet_name="Reasoning_Quality")
        scores = pd.read_excel(path, sheet_name="Diagnostic_Scores")
        state = pd.read_excel(path, sheet_name="Experiment_4a_State")
        if "Fixture Pattern" not in debate.columns and "Experiment" in debate.columns:
            debate = debate.rename(columns={"Experiment": "Fixture Pattern"})
        if "fixture_pattern" not in scores.columns and "dataset_type" in scores.columns:
            scores.insert(
                scores.columns.get_loc("dataset_type") + 1,
                "fixture_pattern",
                "rule_triggered_devils_advocate",
            )

        debate_frames.append(append_run_columns(debate, run_id, seed))
        judgment_frames.append(append_run_columns(judgments, run_id, seed))
        score = append_run_columns(scores, run_id, seed)
        score["row_index"] = np.arange(
            len(score_frames) * args.objective_limit,
            len(score_frames) * args.objective_limit + len(score),
        )
        score_frames.append(score)
        if "row_index" in state.columns:
            state["row_index"] = pd.to_numeric(state["row_index"], errors="coerce") + row_offset
        state_frames.append(append_run_columns(state, run_id, seed))
        row_offset += len(debate)

    debates = pd.concat(debate_frames, ignore_index=True)
    judgments = pd.concat(judgment_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    states = pd.concat(state_frames, ignore_index=True)

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
        states.to_excel(writer, sheet_name="Experiment_4a_State", index=False)


def run_one(
    run_dir: Path,
    seed: int,
    questions: list[DebateQuestion],
    llm: LocalQwenPipeline,
    args: argparse.Namespace,
) -> Path:
    """Run or reuse one seed's Experiment 4a workbook."""
    if args.resume:
        try:
            existing = find_run_workbook(run_dir)
        except FileNotFoundError:
            existing = None
        if existing is not None and workbook_has_required_sheets(existing):
            print(f"Reusing completed seed {seed} workbook at {existing}", flush=True)
            return existing

    output_path = run_dir / DEFAULT_OUTPUT.name
    source_file = output_path.name
    print(f"Running Experiment 4a seed {seed}; writing intermediate workbook under {run_dir}", flush=True)
    debates, state = run_experiment_4a_debates(llm, questions, seed, args)
    if debates.empty:
        raise RuntimeError("Experiment 4a generated no completed debate rows.")

    if args.skip_judging or args.q_source == "confidence":
        judgments = empty_judgments()
    else:
        judgments = judge_debates_with_qwen(
            llm,
            debates,
            source_file=source_file,
            seed=seed + 100_000,
            judge_max_new_tokens=args.judge_max_new_tokens,
            judge_batch_size=args.judge_batch_size,
            sleep=args.sleep,
        )

    if judgments.empty and args.q_source == "llm":
        scores = pd.DataFrame()
    else:
        judgments["source_file"] = source_file
        scores = score_mixed_debates(
            debates,
            judgments,
            source_file,
            q_source=args.q_source,
            metric_version=args.metric_version,
        )
        if "fixture_pattern" not in scores.columns:
            scores.insert(
                scores.columns.get_loc("dataset_type") + 1,
                "fixture_pattern",
                "rule_triggered_devils_advocate",
            )

    run_args = argparse.Namespace(**vars(args))
    run_args.backend = "transformers"
    run_args.seed = seed
    write_qwen_excel_report(
        output_path,
        "qwen_transformers_experiment_4a_rule_triggered_devils_advocate",
        run_args,
        debates,
        judgments,
        scores,
    )
    append_experiment_state_sheet(output_path, state)
    return find_run_workbook(run_dir)


def main() -> None:
    """Coordinate argument parsing, per-seed runs, and final workbook assembly."""
    args = parse_args()
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

    if not args.no_require_gpu:
        require_gpu()

    questions = load_questions_from_base_workbook(args.base_workbook, args.objective_limit)
    llm = LocalQwenPipeline(
        model_id=args.model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        require_gpu=not args.no_require_gpu,
    )

    run_paths = []
    runs_dir = out_dir / "runs"
    for _run_id, seed in enumerate(seeds, start=1):
        run_dir = runs_dir / f"seed_{seed}"
        run_paths.append(run_one(run_dir, seed, questions, llm, args))

    combine_workbooks(output_path, run_paths, seeds, args)
    print(f"Wrote combined Experiment 4a {args.objective_limit}x{len(seeds)} workbook to {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
