"""Experiment 2: Metric-Triggered Resistance for the Qwen MMLU-Pro debate run.

This script intentionally does not replace the baseline generator. It reuses the
same 200 questions from the saved baseline workbook and writes a separate
experiment-2b workbook at data/qwen_mmlu_exp2b.xlsx by default.
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
    parse_confidence,
    parse_qwen_turn,
    qwen_initial_messages,
    qwen_moderator_messages,
    qwen_update_messages,
    score_mixed_debates,
    write_qwen_excel_report,
)


DEFAULT_BASE_WORKBOOK = Path("data/qwen_mmlu_pro_200/qwen_mmlu_pro_debate_traces_base.xlsx")
DEFAULT_OUT_DIR = Path("data")
DEFAULT_OUTPUT = DEFAULT_OUT_DIR / "qwen_mmlu_exp2b.xlsx"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
Q_CNT = 200
# Use repeated --seed flags, for example --seed 7 --seed 17, for multi-seed runs.
DEFAULT_SEEDS = [7]
REQUIRED_RUN_SHEETS = {
    "Debate_Traces",
    "Reasoning_Quality",
    "Diagnostic_Scores",
    "Experiment_2_State",
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

INFLUENCE_ASYMMETRY_TRIGGER = 0.65
BALANCE_TRIGGER = 0.30
RESISTANCE_ALPHA = 0.08
RESISTANCE_CAP = 0.65
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 2, Metric-Triggered Resistance, on the exact questions "
            "stored in the baseline Qwen MMLU-Pro workbook."
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
    parser.add_argument("--influence-asymmetry-trigger", type=float, default=INFLUENCE_ASYMMETRY_TRIGGER)
    parser.add_argument("--balance-trigger", type=float, default=BALANCE_TRIGGER)
    parser.add_argument("--resistance-alpha", type=float, default=RESISTANCE_ALPHA)
    parser.add_argument("--resistance-cap", type=float, default=RESISTANCE_CAP)
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


def support_counts(turns: dict[str, dict[str, object]]) -> dict[str, int]:
    """Return each agent's same-answer support count, including itself."""
    answers = {agent: str(turns[agent].get("answer", "")).strip() for agent in QWEN_AGENTS}
    counts = Counter(answer for answer in answers.values() if answer)
    return {agent: counts.get(answer, 0) if answer else 0 for agent, answer in answers.items()}


def normalized_entropy(values: list[str | None]) -> float:
    """Return answer dispersion on a 0-to-1 entropy scale."""
    vals = [value for value in values if value]
    if len(vals) <= 1:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return -float(sum(pi * np.log(pi) for pi in p if pi > 0)) / np.log(len(vals))


def corrected_balance_from_dispersion(dispersion: list[float]) -> float:
    """Penalize one-step collapse and oscillation in answer dispersion."""
    t_count = len(dispersion)
    if t_count <= 1:
        return np.nan
    steps = [dispersion[t - 1] - dispersion[t] for t in range(1, t_count)]
    positive_convergence = [max(0.0, step) for step in steps]
    total_positive_convergence = sum(positive_convergence)
    max_single_convergence = max(positive_convergence) if positive_convergence else 0.0
    if total_positive_convergence <= EPS:
        collapse_score = 1.0 if max(abs(step) for step in steps) <= EPS else 0.0
    else:
        collapse_score = 1.0 - max_single_convergence / (total_positive_convergence + EPS)
    if t_count <= 2:
        volatility_score = 1.0
    else:
        reversals = sum(steps[t - 1] * steps[t] < 0 for t in range(1, len(steps)))
        volatility_score = 1.0 - reversals / (t_count - 2)
    return float(np.clip(collapse_score * volatility_score, 0.0, 1.0))


def round_answers(rounds: list[dict[str, dict[str, str]]]) -> list[list[str | None]]:
    """Extract answer labels in fixed agent order."""
    return [
        [str(turns[agent].get("answer", "")).strip() or None for agent in QWEN_AGENTS]
        for turns in rounds
    ]


def round_quality(rounds: list[dict[str, dict[str, str]]]) -> np.ndarray:
    """Use confidence as the live reasoning-quality proxy during generation."""
    q = np.zeros((len(rounds), len(QWEN_AGENTS)), dtype=float)
    for ti, turns in enumerate(rounds):
        for ai, agent in enumerate(QWEN_AGENTS):
            quality = parse_confidence(turns[agent].get("confidence", ""))
            if np.isnan(quality):
                quality = 0.5
            q[ti, ai] = float(np.clip(quality, 0.0, 1.0))
    return q


def partial_categorical_metric_state(
    rounds: list[dict[str, dict[str, str]]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, bool], dict[str, str]]:
    """Compute live categorical diagnostics and choose per-agent resistance targets."""
    flags = {agent: False for agent in QWEN_AGENTS}
    reasons = {agent: "" for agent in QWEN_AGENTS}
    metrics = {
        "engagement": np.nan,
        "responsiveness": np.nan,
        "influence_asymmetry": np.nan,
        "balance": np.nan,
    }
    if len(rounds) < 2:
        return metrics, flags, reasons

    answers = round_answers(rounds)
    q = round_quality(rounds)
    a_count = len(QWEN_AGENTS)
    engagement_terms = []
    responsiveness_terms = []
    influence = np.zeros(a_count, dtype=float)

    for t in range(1, len(answers)):
        for a, agent in enumerate(QWEN_AGENTS):
            old = answers[t - 1][a]
            new = answers[t][a]
            if old is None or new is None:
                continue
            changed = old != new
            engagement_terms.append(q[t, a] * float(changed))
            others_prev = [answers[t - 1][j] for j in range(a_count) if j != a and answers[t - 1][j]]
            if others_prev:
                old_support = sum(other == old for other in others_prev)
                new_support = sum(other == new for other in others_prev)
                responsiveness_terms.append(q[t, a] * float(changed and new_support > old_support))
            for source in range(a_count):
                source_answer = answers[t - 1][source]
                if source == a or source_answer is None:
                    continue
                if changed and new == source_answer and old != source_answer:
                    influence[source] += q[t, a]

    total_influence = float(influence.sum())
    if total_influence <= EPS or a_count <= 1:
        influence_asymmetry = 0.0
    else:
        p = influence / total_influence
        h = -sum(pi * np.log(pi) for pi in p if pi > 0)
        influence_asymmetry = float(1.0 - h / np.log(a_count))

    dispersion = [normalized_entropy(round_answer) for round_answer in answers]
    balance = corrected_balance_from_dispersion(dispersion)
    metrics.update(
        {
            "engagement": float(np.mean(engagement_terms)) if engagement_terms else np.nan,
            "responsiveness": float(np.mean(responsiveness_terms)) if responsiveness_terms else np.nan,
            "influence_asymmetry": influence_asymmetry,
            "balance": balance,
        }
    )

    previous_answers = answers[-2]
    latest_answers = answers[-1]
    latest_counts = Counter(answer for answer in latest_answers if answer)
    majority_answer_now = latest_counts.most_common(1)[0][0] if latest_counts else None
    dominant_sources = []
    if total_influence > EPS and influence_asymmetry >= args.influence_asymmetry_trigger:
        max_influence = float(influence.max())
        dominant_sources = [
            idx for idx, value in enumerate(influence) if max_influence > EPS and value >= max_influence - EPS
        ]
    dominant_prior_answers = {
        previous_answers[idx] for idx in dominant_sources if previous_answers[idx] is not None
    }

    for ai, agent in enumerate(QWEN_AGENTS):
        old = previous_answers[ai]
        new = latest_answers[ai]
        changed = old is not None and new is not None and old != new
        agent_reasons = []
        if changed and new in dominant_prior_answers:
            agent_reasons.append("high influence asymmetry")
        if (
            changed
            and majority_answer_now is not None
            and new == majority_answer_now
            and not np.isnan(balance)
            and balance <= args.balance_trigger
        ):
            agent_reasons.append("low balance / rapid convergence")
        if agent_reasons:
            flags[agent] = True
            reasons[agent] = "; ".join(agent_reasons)

    return metrics, flags, reasons


def resistance_update_messages(
    question: DebateQuestion,
    agent: str,
    previous_round: dict[str, dict[str, str]],
    resistance: float,
    trigger_reason: str,
) -> list[dict[str, str]]:
    """Build a normal update prompt with the current resistance context."""
    messages = qwen_update_messages(question, agent, previous_round)
    context = (
        f"\n\nExperiment 2 resistance score: your current resistance level is "
        f"{resistance:.2f} on a 0 to 1 scale."
    )
    if resistance > 0.0:
        status = trigger_reason or "the recent debate pattern may be over-converging"
        context += (
            f" Because the live diagnostics indicate {status}, do not switch "
            "merely because other agents agree with each other. Re-check your "
            "own answer from first principles, identify the strongest evidence "
            "for and against it, and change only if a specific peer argument "
            "defeats your reasoning. If your prior answer remains defensible, "
            "preserve it and explain why."
        )
    patched = list(messages)
    patched[-1] = {**patched[-1], "content": patched[-1]["content"] + context}
    return patched


def resistance_initial_messages(
    question: DebateQuestion,
    agent: str,
    resistance: float,
) -> list[dict[str, str]]:
    """Build an initial prompt with the starting resistance score included."""
    messages = qwen_initial_messages(question, agent)
    context = (
        f"\n\nExperiment 2 resistance score: your current resistance level is "
        f"{resistance:.2f} on a 0 to 1 scale."
    )
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
    """Flatten one Experiment 2 debate into the baseline-compatible schema."""
    row: dict[str, object] = {
        "Question #": question.question_no,
        "Dataset Type": question.dataset_type,
        "Dataset Category": question.category,
        "Question": question.question,
        "Correct Answer": question.correct_answer,
        "Final Answer": final_answer,
        "Final Answer Source": final_answer_source,
        "Model": model_name,
        "Fixture Pattern": "metric_triggered_resistance",
        "Rounds to Consensus": first_consensus_round_for_answer(rounds, final_answer),
    }
    for round_no, turns in enumerate(rounds, start=1):
        for agent in QWEN_AGENTS:
            turn = turns[agent]
            row[f"R{round_no} {agent} Answer"] = turn["answer"]
            row[f"R{round_no} {agent} Conf"] = turn["confidence"]
            row[f"R{round_no} {agent} Response"] = turn["response"]
    return row


def run_experiment_2_debates(
    llm: LocalQwenPipeline,
    questions: list[DebateQuestion],
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Qwen debates with metric-triggered resistance."""
    rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    rng = random.Random(seed)

    for q_index, question in enumerate(questions):
        print(
            f"Experiment 2 generating question {q_index + 1}/{len(questions)}: {question.question_no}",
            flush=True,
        )
        agent_order = QWEN_AGENTS[:]
        rng.shuffle(agent_order)
        resistance = {agent: 0.0 for agent in QWEN_AGENTS}
        support_history: list[dict[str, int]] = []
        all_rounds: list[dict[str, dict[str, str]]] = []
        question_state_rows: list[dict[str, object]] = []
        discard_reason = ""

        for round_no in range(1, args.rounds + 1):
            metric_values, resistance_flags, trigger_reasons = partial_categorical_metric_state(
                all_rounds,
                args,
            )
            resistance_before = resistance.copy()
            for agent, is_triggered in resistance_flags.items():
                if is_triggered:
                    resistance[agent] = min(
                        args.resistance_cap,
                        resistance[agent] + args.resistance_alpha,
                    )

            print(
                f"Experiment 2 question {q_index + 1} round {round_no}/{args.rounds}",
                flush=True,
            )
            messages_by_agent = {}
            for agent in agent_order:
                if round_no == 1:
                    messages_by_agent[agent] = resistance_initial_messages(
                        question,
                        agent,
                        resistance[agent],
                    )
                else:
                    messages_by_agent[agent] = resistance_update_messages(
                        question,
                        agent,
                        all_rounds[-1],
                        resistance[agent],
                        trigger_reasons[agent],
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

            round_support = support_counts(round_turns)
            support_history.append(round_support)
            previous_support = support_history[-2] if len(support_history) >= 2 else {}
            for agent in QWEN_AGENTS:
                question_state_rows.append(
                    {
                        "row_index": len(rows),
                        "question_no": question.question_no,
                        "round": round_no,
                        "agent": agent,
                        "answer": round_turns[agent]["answer"],
                        "support": round_support.get(agent, 0),
                        "support_trend": round_support.get(agent, 0) - previous_support.get(agent, round_support.get(agent, 0)),
                        "metric_trigger_before_turn": bool(resistance_flags[agent]),
                        "metric_trigger_reason": trigger_reasons[agent],
                        "live_engagement": metric_values["engagement"],
                        "live_responsiveness": metric_values["responsiveness"],
                        "live_influence_asymmetry": metric_values["influence_asymmetry"],
                        "live_balance": metric_values["balance"],
                        "resistance_before": resistance_before[agent],
                        "resistance_after": resistance[agent],
                        "intervention_applied": bool(resistance[agent] > 0.0 and round_no > 1),
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
    """Append the Experiment_2_State sheet after the shared report writer runs."""
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        state.to_excel(writer, sheet_name="Experiment_2_State", index=False)


def combine_workbooks(output_path: Path, run_paths: list[Path], seeds: list[int], args: argparse.Namespace) -> None:
    """Merge per-seed Experiment 2 workbooks into one multi-seed Excel report."""
    debate_frames = []
    judgment_frames = []
    score_frames = []
    state_frames = []
    metadata_rows = [
        {"field": "run_mode", "value": "qwen_transformers_experiment_2_metric_triggered_resistance"},
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
        {"field": "experiment", "value": "metric_triggered_resistance"},
        {"field": "live_metric_q_source", "value": "confidence"},
        {"field": "influence_asymmetry_trigger", "value": args.influence_asymmetry_trigger},
        {"field": "balance_trigger", "value": args.balance_trigger},
        {"field": "resistance_alpha", "value": args.resistance_alpha},
        {"field": "resistance_cap", "value": args.resistance_cap},
        {"field": "temperature", "value": args.temperature},
        {"field": "top_p", "value": args.top_p},
    ]

    row_offset = 0
    for run_id, (path, seed) in enumerate(zip(run_paths, seeds), start=1):
        debate = pd.read_excel(path, sheet_name="Debate_Traces")
        judgments = pd.read_excel(path, sheet_name="Reasoning_Quality")
        scores = pd.read_excel(path, sheet_name="Diagnostic_Scores")
        state = pd.read_excel(path, sheet_name="Experiment_2_State")
        if "Fixture Pattern" not in debate.columns and "Experiment" in debate.columns:
            debate = debate.rename(columns={"Experiment": "Fixture Pattern"})
        if "fixture_pattern" not in scores.columns and "dataset_type" in scores.columns:
            scores.insert(
                scores.columns.get_loc("dataset_type") + 1,
                "fixture_pattern",
                "metric_triggered_resistance",
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
        states.to_excel(writer, sheet_name="Experiment_2_State", index=False)


def run_one(
    run_dir: Path,
    seed: int,
    questions: list[DebateQuestion],
    llm: LocalQwenPipeline,
    args: argparse.Namespace,
) -> Path:
    """Run or reuse one seed's Experiment 2 workbook."""
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
    print(f"Running Experiment 2 seed {seed}; writing intermediate workbook under {run_dir}", flush=True)
    debates, state = run_experiment_2_debates(llm, questions, seed, args)
    if debates.empty:
        raise RuntimeError("Experiment 2 generated no completed debate rows.")

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
                "metric_triggered_resistance",
            )

    run_args = argparse.Namespace(**vars(args))
    run_args.backend = "transformers"
    run_args.seed = seed
    write_qwen_excel_report(
        output_path,
        "qwen_transformers_experiment_2_metric_triggered_resistance",
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
    for run_id, seed in enumerate(seeds, start=1):
        run_dir = runs_dir / f"seed_{seed}"
        run_paths.append(run_one(run_dir, seed, questions, llm, args))

    combine_workbooks(output_path, run_paths, seeds, args)
    print(f"Wrote combined Experiment 2 {args.objective_limit}x{len(seeds)} workbook to {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
