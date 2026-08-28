"""Experiment 4: Learned Evidence-Aware Minority Protection.

Mirrors generate_qwen_mmlu_exp3b.py exactly, with one change:
instead of hard threshold triggers (influence_asymmetry >= 0.65,
balance <= 0.3), each minority agent's protection decision and token
budget are set by the trained MinorityPredictor.

The predictor outputs p(minority correct | features) and the token
budget scales linearly: extra_tokens = int(MAX_EXTRA_TOKENS * p).
An agent is protected only when p >= PRECISION_THRESHOLD and the
reasoning quality score >= MIN_QUALITY_THRESHOLD.

All debate generation, judging, scoring, and workbook writing logic
is identical to Exp3b so results are directly comparable.
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

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "docs"))

from predictor import MinorityPredictor, create_feature_dict

from qwen_methodology_code import (
    DebateQuestion,
    LocalQwenPipeline,
    OllamaQwenPipeline,
    OBJECTIVE_LABELS,
    QWEN_AGENTS,
    EPS,
    answer_is_valid,
    empty_judgments,
    final_answer_correctness,
    first_consensus_round_for_answer,
    judge_debates_with_qwen,
    majority_answer,
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

DEFAULT_BASE_WORKBOOK = Path("data/qwen_mmlu_pro_200/qwen_mmlu_pro_debate_traces_base.xlsx")
DEFAULT_PREDICTOR = Path("Pred_Minority/models/ensemble_model.pkl")
DEFAULT_OUT_DIR = Path("data")
DEFAULT_OUTPUT = DEFAULT_OUT_DIR / "qwen_mmlu_exp4.xlsx"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_SEEDS = [7]
Q_CNT = 200

PRECISION_THRESHOLD = 0.35   # minimum p(correct) to trigger protection
MIN_QUALITY_THRESHOLD = 0.30  # minimum reasoning quality to trigger protection
MAX_EXTRA_TOKENS = 150        # token budget at p=1.0; scales linearly with p

REQUIRED_RUN_SHEETS = {
    "Debate_Traces",
    "Reasoning_Quality",
    "Diagnostic_Scores",
    "Experiment_4_State",
}
PROTECTED_OUTPUTS: set[Path] = set()

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4: learned evidence-aware minority protection."
    )
    parser.add_argument("--base-workbook", type=Path, default=DEFAULT_BASE_WORKBOOK)
    parser.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--objective-limit", type=int, default=Q_CNT)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, action="append", default=None)
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
        "--backend", choices=["transformers", "ollama"], default="transformers",
        help="Use 'ollama' for local 7B via Ollama, 'transformers' for 14B on GPU",
    )
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    # Predictor thresholds (can be overridden from CLI for ablation)
    parser.add_argument("--precision-threshold", type=float, default=PRECISION_THRESHOLD)
    parser.add_argument("--min-quality-threshold", type=float, default=MIN_QUALITY_THRESHOLD)
    parser.add_argument("--max-extra-tokens", type=int, default=MAX_EXTRA_TOKENS)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def require_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        print("Warning: nvidia-smi not on PATH; relying on PyTorch check.", flush=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch required for the transformers backend.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required but PyTorch cannot see CUDA.")
    visible = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    print(f"CUDA GPU available: {visible}", flush=True)


# ---------------------------------------------------------------------------
# Helpers shared with exp3b (copied verbatim so this file is self-contained)
# ---------------------------------------------------------------------------

def spearman(series: pd.Series, accuracy: pd.Series) -> tuple[float, int]:
    paired = pd.DataFrame(
        {"metric": pd.to_numeric(series, errors="coerce"),
         "accuracy": pd.to_numeric(accuracy, errors="coerce")}
    ).dropna()
    if len(paired) < 3 or paired["metric"].nunique() <= 1 or paired["accuracy"].nunique() <= 1:
        return np.nan, len(paired)
    rho = float(np.corrcoef(paired["metric"].rank(), paired["accuracy"].rank())[0, 1])
    return rho, len(paired)


def aggregate_report(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or not {"dataset_type", "stance_mode"}.issubset(scores.columns):
        return pd.DataFrame()
    numeric = [
        "accuracy", "engagement", "responsiveness",
        "influence_asymmetry", "influence_asymmetry_inv",
        "balance", "stability", "group_welfare", "avg_process_metrics",
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
    candidates = [run_dir / DEFAULT_OUTPUT.name]
    candidates.extend(sorted(run_dir.glob("qwen_mmlu_exp4*.xlsx")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No generated workbook found in {run_dir}")


def workbook_has_required_sheets(path: Path) -> bool:
    try:
        sheet_names = set(pd.ExcelFile(path).sheet_names)
    except Exception:
        return False
    return REQUIRED_RUN_SHEETS.issubset(sheet_names)


def append_run_columns(df: pd.DataFrame, run_id: int, seed: int) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "Run ID", run_id)
    out.insert(1, "Seed", seed)
    return out


def infer_answer_labels(question_text: str, correct_answer: str) -> tuple[str, ...]:
    found = re.findall(r"(?m)^([A-J])\.\s+", question_text)
    labels = tuple(label for label in OBJECTIVE_LABELS if label in set(found))
    if correct_answer and correct_answer not in labels:
        labels = OBJECTIVE_LABELS[: max(4, OBJECTIVE_LABELS.index(correct_answer) + 1)]
    return labels or OBJECTIVE_LABELS


def load_questions_from_base_workbook(path: Path, limit: int) -> list[DebateQuestion]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline workbook not found: {path}")
    debates = pd.read_excel(path, sheet_name="Debate_Traces")
    if limit is not None:
        debates = debates.head(limit)
    questions: list[DebateQuestion] = []
    required = {"Question #", "Dataset Type", "Question", "Correct Answer"}
    missing = required - set(debates.columns)
    if missing:
        raise ValueError(f"Baseline workbook missing required columns: {sorted(missing)}")
    for _, row in debates.iterrows():
        if str(row["Dataset Type"]).strip() != "objective":
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


def support_counts(turns: dict[str, dict]) -> dict[str, int]:
    answers = {agent: str(turns[agent].get("answer", "")).strip() for agent in QWEN_AGENTS}
    counts = Counter(answer for answer in answers.values() if answer)
    return {agent: counts.get(answer, 0) if answer else 0 for agent, answer in answers.items()}


def normalized_entropy(values: list) -> float:
    vals = [v for v in values if v]
    if len(vals) <= 1:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return -float(sum(pi * np.log(pi) for pi in p if pi > 0)) / np.log(len(vals))


def corrected_balance_from_dispersion(dispersion: list[float]) -> float:
    t_count = len(dispersion)
    if t_count <= 1:
        return np.nan
    steps = [dispersion[t - 1] - dispersion[t] for t in range(1, t_count)]
    positive_convergence = [max(0.0, step) for step in steps]
    total_positive_convergence = sum(positive_convergence)
    max_single_convergence = max(positive_convergence) if positive_convergence else 0.0
    if total_positive_convergence <= EPS:
        collapse_score = 1.0 if max(abs(s) for s in steps) <= EPS else 0.0
    else:
        collapse_score = 1.0 - max_single_convergence / (total_positive_convergence + EPS)
    if t_count <= 2:
        volatility_score = 1.0
    else:
        reversals = sum(steps[t - 1] * steps[t] < 0 for t in range(1, len(steps)))
        volatility_score = 1.0 - reversals / (t_count - 2)
    return float(np.clip(collapse_score * volatility_score, 0.0, 1.0))


def round_answers(rounds: list[dict]) -> list[list]:
    return [
        [str(turns[agent].get("answer", "")).strip() or None for agent in QWEN_AGENTS]
        for turns in rounds
    ]


def round_quality(rounds: list[dict]) -> np.ndarray:
    q = np.zeros((len(rounds), len(QWEN_AGENTS)), dtype=float)
    for ti, turns in enumerate(rounds):
        for ai, agent in enumerate(QWEN_AGENTS):
            quality = parse_confidence(turns[agent].get("confidence", ""))
            q[ti, ai] = float(np.clip(0.5 if np.isnan(quality) else quality, 0.0, 1.0))
    return q


# ---------------------------------------------------------------------------
# KEY CHANGE: learned predictor replaces hard thresholds
# ---------------------------------------------------------------------------

def predictor_minority_state(
    rounds: list[dict],
    predictor: MinorityPredictor,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict, dict, dict, str]:
    """Compute live diagnostics and decide protection via the learned predictor.

    Returns the same signature as exp3b's diagnostic_minority_state plus
    one extra value: p_correct_by_agent (float per agent, nan if not minority).

    The process-level diagnostics (influence_asymmetry, balance, engagement,
    responsiveness) are computed identically to exp3b.  Those values feed the
    predictor as features along with per-agent support and quality signals.
    """
    flags = {agent: False for agent in QWEN_AGENTS}
    reasons = {agent: "" for agent in QWEN_AGENTS}
    influence_share = {agent: np.nan for agent in QWEN_AGENTS}
    p_correct_by_agent = {agent: np.nan for agent in QWEN_AGENTS}
    metrics = {
        "engagement": np.nan,
        "responsiveness": np.nan,
        "influence_asymmetry": np.nan,
        "balance": np.nan,
    }
    if len(rounds) < 2:
        return metrics, flags, reasons, influence_share, p_correct_by_agent, ""

    answers = round_answers(rounds)
    q = round_quality(rounds)
    a_count = len(QWEN_AGENTS)

    # --- compute process metrics (identical to exp3b) ---
    engagement_terms, responsiveness_terms = [], []
    influence = np.zeros(a_count, dtype=float)

    for t in range(1, len(answers)):
        for a, agent in enumerate(QWEN_AGENTS):
            old = answers[t - 1][a]
            new = answers[t][a]
            if old is None or new is None:
                continue
            changed = old != new
            engagement_terms.append(q[t, a] * float(changed))
            others_prev = [answers[t - 1][j] for j in range(a_count)
                          if j != a and answers[t - 1][j]]
            if others_prev:
                old_support = sum(o == old for o in others_prev)
                new_support = sum(o == new for o in others_prev)
                responsiveness_terms.append(q[t, a] * float(changed and new_support > old_support))
            for source in range(a_count):
                src_ans = answers[t - 1][source]
                if source == a or src_ans is None:
                    continue
                if changed and new == src_ans and old != src_ans:
                    influence[source] += q[t, a]

    total_influence = float(influence.sum())
    if total_influence <= EPS or a_count <= 1:
        influence_asymmetry = 0.0
        p_inf = np.zeros(a_count, dtype=float)
    else:
        p_inf = influence / total_influence
        h = -sum(pi * np.log(pi) for pi in p_inf if pi > 0)
        influence_asymmetry = float(1.0 - h / np.log(a_count))

    for idx, agent in enumerate(QWEN_AGENTS):
        influence_share[agent] = float(p_inf[idx]) if total_influence > EPS else np.nan

    dispersion = [normalized_entropy(ra) for ra in answers]
    balance = corrected_balance_from_dispersion(dispersion)
    engagement = float(np.mean(engagement_terms)) if engagement_terms else np.nan
    responsiveness = float(np.mean(responsiveness_terms)) if responsiveness_terms else np.nan

    metrics.update({
        "engagement": engagement,
        "responsiveness": responsiveness,
        "influence_asymmetry": influence_asymmetry,
        "balance": balance,
    })

    # --- identify majority answer for context ---
    latest_answers = answers[-1]
    prev_answers = answers[-2]
    latest_counts = Counter(a for a in latest_answers if a)
    majority_answer_now = latest_counts.most_common(1)[0][0] if latest_counts else ""
    max_support = max(latest_counts.values()) if latest_counts else 0

    # --- per-agent predictor decision ---
    for a_idx, agent in enumerate(QWEN_AGENTS):
        curr_ans = latest_answers[a_idx]
        if curr_ans is None:
            continue

        curr_support = latest_counts.get(curr_ans, 0)
        # Only evaluate agents currently in the minority
        if curr_support >= max_support:
            continue

        prev_ans = prev_answers[a_idx]
        prev_counts = Counter(x for x in prev_answers if x)
        prev_max = max(prev_counts.values()) if prev_counts else 0
        was_majority = prev_counts.get(curr_ans, 0) >= prev_max if prev_counts else False

        num_defections = sum(
            1 for j in range(a_count)
            if prev_answers[j] == curr_ans and latest_answers[j] != curr_ans
        )

        # LLM quality for this agent at the latest round
        agent_quality = float(q[-1, a_idx])
        support_delta = curr_support - prev_counts.get(curr_ans, 0)

        features = create_feature_dict(
            influence_asymmetry=influence_asymmetry,
            balance=balance if not np.isnan(balance) else 0.5,
            engagement=engagement if not np.isnan(engagement) else 0.1,
            responsiveness=responsiveness if not np.isnan(responsiveness) else 0.1,
            stability=0.0,   # not computed live
            group_welfare=0.0,
            support_share=curr_support / a_count,
            support_delta=support_delta,
            minority_conf_mean=float(q[-1, a_idx]),
            minority_conf_std=0.0,
            minority_quality_mean=agent_quality,
            minority_quality_std=0.0,
            was_majority_before=was_majority,
            num_defections=num_defections,
            num_supporters=curr_support,
            rounds_remaining=args.rounds - len(rounds),
        )

        should_protect, extra_tokens, p_correct = predictor.should_protect(
            features,
            precision_threshold=args.precision_threshold,
            min_quality_threshold=args.min_quality_threshold,
        )

        p_correct_by_agent[agent] = float(p_correct)

        if should_protect:
            flags[agent] = True
            reasons[agent] = (
                f"learned predictor: p(correct)={p_correct:.2f}, "
                f"influence_asymmetry={influence_asymmetry:.2f}, "
                f"balance={balance:.2f}"
            )

    return metrics, flags, reasons, influence_share, p_correct_by_agent, majority_answer_now


# ---------------------------------------------------------------------------
# Prompt builder (same as exp3b, updated experiment label)
# ---------------------------------------------------------------------------

def learned_minority_update_messages(
    question: DebateQuestion,
    agent: str,
    previous_round: dict,
    minority_flag: bool,
    trigger_reason: str,
    p_correct: float,
) -> list[dict]:
    """Build update prompt; minority agents get an evidence-review instruction."""
    messages = qwen_update_messages(question, agent, previous_round)
    context = "\n\nExperiment 4 learned minority-protection status: "
    if minority_flag:
        context += (
            f"a trained predictor estimates your current position has "
            f"{p_correct:.0%} probability of being correct (based on reasoning "
            f"quality and debate dynamics: {trigger_reason}). "
            "Before updating, give your hypothesis a rigorous test: state the "
            "strongest specific evidence for it, state the strongest specific "
            "evidence against it, and change only if a peer argument concretely "
            "defeats your reasoning. If your answer remains defensible, preserve "
            "it and explain why."
        )
    else:
        context += "no minority-protection intervention is active for your next answer."
    patched = list(messages)
    patched[-1] = {**patched[-1], "content": patched[-1]["content"] + context}
    return patched


# ---------------------------------------------------------------------------
# Turn generation with dynamic token budget
# ---------------------------------------------------------------------------

def complete_qwen_turn_with_budget(
    llm: LocalQwenPipeline,
    messages: list[dict],
    question: DebateQuestion,
    seed: int,
    max_new_tokens: int,
    max_attempts: int = 3,
) -> dict:
    first_error = ""
    for attempt in range(max(1, max_attempts)):
        attempt_messages = messages if attempt == 0 else qwen_reprompt_messages(messages, question)
        raw = llm.complete(attempt_messages, seed=seed + attempt, max_new_tokens=max_new_tokens)
        parsed = parse_qwen_turn(raw, question.dataset_type, question.answer_labels, strict=True)
        if attempt == 0 and parsed["parse_failed"]:
            first_error = str(parsed["parse_error"])
        parsed["re_prompted"] = attempt > 0
        if first_error:
            parsed["first_parse_error"] = first_error
        if not parsed["parse_failed"] or attempt == max(1, max_attempts) - 1:
            return parsed
    raise RuntimeError("unreachable Qwen turn retry state")


def complete_turns_with_learned_budget(
    llm: LocalQwenPipeline,
    messages_by_agent: dict,
    question: DebateQuestion,
    seeds_by_agent: dict,
    minority_flags: dict,
    p_correct_by_agent: dict,
    args: argparse.Namespace,
) -> dict:
    """Generate all agent turns, scaling token budget by predicted p(correct)."""
    turns = {}
    for agent, messages in messages_by_agent.items():
        if minority_flags.get(agent, False):
            p = float(p_correct_by_agent.get(agent, args.precision_threshold))
            extra = int(args.max_extra_tokens * min(p, 0.9))
        else:
            extra = 0
        turns[agent] = complete_qwen_turn_with_budget(
            llm, messages, question, seeds_by_agent[agent],
            args.max_new_tokens + extra,
        )
    return turns


# ---------------------------------------------------------------------------
# Final answer selection (identical to exp3b)
# ---------------------------------------------------------------------------

def select_final_answer(
    llm: LocalQwenPipeline,
    question: DebateQuestion,
    rounds: list[dict],
    seed: int,
) -> tuple[str, str]:
    final_answers = [rounds[-1][agent]["answer"] for agent in QWEN_AGENTS]
    valid = [a for a in final_answers if a]
    if valid and len(set(valid)) == 1:
        return valid[0], "agent_consensus"
    raw = llm.complete(qwen_moderator_messages(question, rounds), seed=seed)
    parsed = parse_qwen_turn(raw, question.dataset_type, question.answer_labels, strict=False)["answer"]
    if answer_is_valid(parsed, question):
        return parsed, "moderator"
    return majority_answer(final_answers), "majority_vote_no_moderator"


# ---------------------------------------------------------------------------
# Debate row builder
# ---------------------------------------------------------------------------

def make_debate_row(
    question: DebateQuestion,
    rounds: list[dict],
    model_name: str,
    final_answer: str,
    final_answer_source: str,
) -> dict:
    row = {
        "Question #": question.question_no,
        "Dataset Type": question.dataset_type,
        "Dataset Category": question.category,
        "Question": question.question,
        "Correct Answer": question.correct_answer,
        "Final Answer": final_answer,
        "Final Answer Source": final_answer_source,
        "Model": model_name,
        "Fixture Pattern": "learned_evidence_aware_protection",
        "Rounds to Consensus": first_consensus_round_for_answer(rounds, final_answer),
    }
    for round_no, turns in enumerate(rounds, start=1):
        for agent in QWEN_AGENTS:
            turn = turns[agent]
            row[f"R{round_no} {agent} Answer"] = turn["answer"]
            row[f"R{round_no} {agent} Conf"] = turn["confidence"]
            row[f"R{round_no} {agent} Response"] = turn["response"]
    return row


# ---------------------------------------------------------------------------
# Question-level checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(run_dir: Path) -> Path:
    """Return path to the per-question JSONL checkpoint file."""
    return run_dir / "exp4_checkpoint.jsonl"


def load_checkpoint(run_dir: Path) -> tuple[list[dict], list[dict], set]:
    """Load completed debate rows, state rows, and done question numbers."""
    cp = checkpoint_path(run_dir)
    rows, state_rows, done = [], [], set()
    if not cp.exists():
        return rows, state_rows, done
    with cp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("_type") == "debate":
                rows.append(entry["data"])
                done.add(entry["question_no"])
            elif entry.get("_type") == "state":
                state_rows.append(entry["data"])
    print(f"Checkpoint: resuming after {len(done)} completed questions.", flush=True)
    return rows, state_rows, done


def save_checkpoint_debate(run_dir: Path, question_no, debate_row: dict,
                           state_batch: list[dict]) -> None:
    """Append one completed debate + its state rows to the checkpoint file."""
    cp = checkpoint_path(run_dir)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"_type": "debate", "question_no": question_no,
                            "data": debate_row}) + "\n")
        for sr in state_batch:
            f.write(json.dumps({"_type": "state", "question_no": question_no,
                                "data": sr}) + "\n")


# ---------------------------------------------------------------------------
# Main debate loop
# ---------------------------------------------------------------------------

def run_experiment_4_debates(
    llm: LocalQwenPipeline,
    predictor: MinorityPredictor,
    questions: list[DebateQuestion],
    seed: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run debates with learned minority protection; return traces and state."""
    # Load any previously completed questions from checkpoint
    rows, state_rows, done_question_nos = load_checkpoint(run_dir)
    rng = random.Random(seed)

    for q_index, question in enumerate(questions):
        # Advance RNG regardless so agent ordering is deterministic
        agent_order = QWEN_AGENTS[:]
        rng.shuffle(agent_order)

        if question.question_no in done_question_nos:
            print(
                f"Exp4 question {q_index + 1}/{len(questions)}: {question.question_no} "
                f"[skipped - already in checkpoint]",
                flush=True,
            )
            continue

        print(
            f"Exp4 question {q_index + 1}/{len(questions)}: {question.question_no}",
            flush=True,
        )
        all_rounds: list[dict] = []
        question_state_rows: list[dict] = []
        discard_reason = ""

        for round_no in range(1, args.rounds + 1):
            previous_round = all_rounds[-1] if all_rounds else None

            # Compute diagnostics + predictor decisions
            (metric_values, minority_flags, trigger_reasons,
             influence_share, p_correct_by_agent, majority_answer_now) = (
                predictor_minority_state(all_rounds, predictor, args)
            )

            print(
                f"  round {round_no}/{args.rounds}  "
                f"protected={[a for a, f in minority_flags.items() if f]}",
                flush=True,
            )

            messages_by_agent = {}
            for agent in agent_order:
                if round_no == 1:
                    messages_by_agent[agent] = qwen_initial_messages(question, agent)
                else:
                    messages_by_agent[agent] = learned_minority_update_messages(
                        question, agent, all_rounds[-1],
                        minority_flags[agent], trigger_reasons[agent],
                        p_correct_by_agent.get(agent, 0.0),
                    )

            seeds_by_agent = {
                agent: seed + q_index * 1000 + round_no * 100 + QWEN_AGENTS.index(agent) * 10
                for agent in agent_order
            }

            round_turns = complete_turns_with_learned_budget(
                llm, messages_by_agent, question, seeds_by_agent,
                minority_flags, p_correct_by_agent, args,
            )

            for agent in agent_order:
                if round_turns[agent]["parse_failed"]:
                    discard_reason = (
                        f"{question.question_no} round {round_no} {agent}: "
                        f"{round_turns[agent]['parse_error']}"
                    )
                    break

            if args.sleep:
                time.sleep(args.sleep)
            if discard_reason:
                print(f"Discarding debate: {discard_reason}", flush=True)
                break

            current_support = support_counts(round_turns)
            previous_support = support_counts(previous_round) if previous_round else {}

            for agent in QWEN_AGENTS:
                prev_answer = previous_round[agent]["answer"] if previous_round else ""
                curr_answer = round_turns[agent]["answer"]
                p_cor = p_correct_by_agent.get(agent, np.nan)
                extra_tokens = (
                    int(args.max_extra_tokens * min(float(p_cor), 0.9))
                    if minority_flags.get(agent) and not np.isnan(p_cor) else 0
                )
                question_state_rows.append({
                    "row_index": len(rows),
                    "question_no": question.question_no,
                    "round": round_no,
                    "agent": agent,
                    "previous_answer": prev_answer,
                    "answer": curr_answer,
                    "previous_support": previous_support.get(agent, 0),
                    "support": current_support.get(agent, 0),
                    "majority_answer_before_turn": majority_answer_now,
                    "minority_flag_before_turn": bool(minority_flags[agent]),
                    "p_correct_estimate": p_cor,          # NEW vs exp3b
                    "metric_trigger_reason": trigger_reasons[agent],
                    "live_engagement": metric_values["engagement"],
                    "live_responsiveness": metric_values["responsiveness"],
                    "live_influence_asymmetry": metric_values["influence_asymmetry"],
                    "live_balance": metric_values["balance"],
                    "live_influence_share": influence_share[agent],
                    "extra_reasoning_tokens": extra_tokens,
                    "intervention_applied": bool(minority_flags[agent]),
                    "answer_changed": bool(prev_answer and curr_answer != prev_answer),
                    "joined_previous_majority": bool(
                        majority_answer_now and curr_answer == majority_answer_now
                    ),
                })

            all_rounds.append(round_turns)

        if discard_reason:
            continue

        final_answer, final_source = select_final_answer(
            llm, question, all_rounds, seed + q_index * 1000 + 999
        )
        debate_row = make_debate_row(
            question, all_rounds, args.model_id, final_answer, final_source
        )
        rows.append(debate_row)
        state_rows.extend(question_state_rows)
        # Save immediately so a crash/kill can resume from here
        save_checkpoint_debate(run_dir, question.question_no, debate_row,
                               question_state_rows)

    debates = pd.DataFrame(rows)
    if not debates.empty:
        debates["Correct?"] = debates.apply(final_answer_correctness, axis=1)
    return debates, pd.DataFrame(state_rows)


# ---------------------------------------------------------------------------
# Workbook I/O
# ---------------------------------------------------------------------------

def append_state_sheet(output_path: Path, state: pd.DataFrame) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        state.to_excel(writer, sheet_name="Experiment_4_State", index=False)


def combine_workbooks(
    output_path: Path,
    run_paths: list[Path],
    seeds: list[int],
    args: argparse.Namespace,
) -> None:
    debate_frames, judgment_frames, score_frames, state_frames = [], [], [], []
    metadata_rows = [
        {"field": "run_mode", "value": "qwen_transformers_experiment_4_learned_protection"},
        {"field": "model_id", "value": args.model_id},
        {"field": "dataset_source", "value": "mmlu-pro"},
        {"field": "objective_questions_per_seed", "value": args.objective_limit},
        {"field": "num_seeds", "value": len(seeds)},
        {"field": "seeds", "value": ",".join(str(s) for s in seeds)},
        {"field": "total_expected_debates", "value": args.objective_limit * len(seeds)},
        {"field": "experiment", "value": "learned_evidence_aware_minority_protection"},
        {"field": "predictor_path", "value": str(args.predictor)},
        {"field": "precision_threshold", "value": args.precision_threshold},
        {"field": "min_quality_threshold", "value": args.min_quality_threshold},
        {"field": "max_extra_tokens", "value": args.max_extra_tokens},
        {"field": "temperature", "value": args.temperature},
        {"field": "top_p", "value": args.top_p},
    ]

    row_offset = 0
    for run_id, (path, seed) in enumerate(zip(run_paths, seeds), start=1):
        debate = pd.read_excel(path, sheet_name="Debate_Traces")
        judgments = pd.read_excel(path, sheet_name="Reasoning_Quality")
        scores = pd.read_excel(path, sheet_name="Diagnostic_Scores")
        state = pd.read_excel(path, sheet_name="Experiment_4_State")

        if "fixture_pattern" not in scores.columns and "dataset_type" in scores.columns:
            scores.insert(
                scores.columns.get_loc("dataset_type") + 1,
                "fixture_pattern",
                "learned_evidence_aware_protection",
            )

        debate_frames.append(append_run_columns(debate, run_id, seed))
        judgment_frames.append(append_run_columns(judgments, run_id, seed))
        score = append_run_columns(scores, run_id, seed)
        score["row_index"] = np.arange(
            row_offset, row_offset + len(score)
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
        for source, count in debates["Final Answer Source"].value_counts(dropna=False).items():
            metadata_rows.append({"field": f"final_answer_source_count:{source}", "value": int(count)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        debates.to_excel(writer, sheet_name="Debate_Traces", index=False)
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Run_Metadata", index=False)
        judgments.to_excel(writer, sheet_name="Reasoning_Quality", index=False)
        scores.to_excel(writer, sheet_name="Diagnostic_Scores", index=False)
        aggregate_report(scores).to_excel(writer, sheet_name="Aggregate_Summary", index=False)
        objective_correlation_report(scores).to_excel(writer, sheet_name="Objective_Correlations", index=False)
        states.to_excel(writer, sheet_name="Experiment_4_State", index=False)


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_one(
    run_dir: Path,
    seed: int,
    questions: list[DebateQuestion],
    llm: LocalQwenPipeline,
    predictor: MinorityPredictor,
    args: argparse.Namespace,
) -> Path:
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
    print(f"Running Exp4 seed {seed} under {run_dir}", flush=True)

    debates, state = run_experiment_4_debates(llm, predictor, questions, seed,
                                              run_dir, args)
    if debates.empty:
        raise RuntimeError("Experiment 4 generated no completed debate rows.")

    if args.skip_judging or args.q_source == "confidence":
        judgments = empty_judgments()
    else:
        judgments = judge_debates_with_qwen(
            llm, debates,
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
            debates, judgments, source_file,
            q_source=args.q_source, metric_version=args.metric_version,
        )
        if "fixture_pattern" not in scores.columns:
            scores.insert(
                scores.columns.get_loc("dataset_type") + 1,
                "fixture_pattern",
                "learned_evidence_aware_protection",
            )

    run_args = argparse.Namespace(**vars(args))
    run_args.backend = "transformers"
    run_args.seed = seed
    write_qwen_excel_report(
        output_path,
        "qwen_transformers_experiment_4_learned_protection",
        run_args, debates, judgments, scores,
    )
    append_state_sheet(output_path, state)
    return find_run_workbook(run_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = args.out_dir / DEFAULT_OUTPUT.name

    if output_path.exists() and not args.overwrite:
        if args.resume and workbook_has_required_sheets(output_path):
            print(f"Final workbook already exists at {output_path}; nothing to do.", flush=True)
            return
        raise RuntimeError(
            f"{output_path} already exists. Use --overwrite or --resume."
        )

    if not args.no_require_gpu:
        require_gpu()

    print(f"Loading predictor from: {args.predictor}", flush=True)
    predictor = MinorityPredictor(args.predictor, model_type="ensemble")
    print(
        f"Predictor loaded. Thresholds: precision>={args.precision_threshold}, "
        f"quality>={args.min_quality_threshold}, max_extra_tokens={args.max_extra_tokens}",
        flush=True,
    )

    questions = load_questions_from_base_workbook(args.base_workbook, args.objective_limit)

    if args.backend == "ollama":
        print(f"Using Ollama backend at {args.ollama_host} with model {args.model_id}", flush=True)
        llm = OllamaQwenPipeline(
            model_id=args.model_id,
            host=args.ollama_host,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        llm = LocalQwenPipeline(
            model_id=args.model_id,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            require_gpu=not args.no_require_gpu,
        )

    seeds = args.seed if args.seed is not None else list(DEFAULT_SEEDS)
    runs_dir = args.out_dir / "runs"
    run_paths = []
    for seed in seeds:
        run_dir = runs_dir / f"seed_{seed}"
        run_paths.append(run_one(run_dir, seed, questions, llm, predictor, args))

    combine_workbooks(output_path, run_paths, seeds, args)
    print(f"Wrote Exp4 {args.objective_limit}x{len(seeds)} workbook to {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
