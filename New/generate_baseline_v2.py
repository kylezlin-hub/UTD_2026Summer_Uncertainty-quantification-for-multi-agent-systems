"""generate_baseline_v2.py — Multi-seed, multi-dataset baseline for AIB research.

Generates the expanded baseline needed to train the learned minority predictor:

    MMLU-Pro: 200 questions × 3 seeds  (seeds 7, 17, 42)
    GPQA:     100 questions × 3 seeds  (seeds 7, 17, 42)

Output workbooks:
    data/baseline_v2/runs/<dataset>_seed_<N>/baseline_v2_<dataset>_s<N>.xlsx
    data/baseline_v2/baseline_v2_combined.xlsx   (all runs merged)

Each workbook contains five sheets:
    Debate_Traces      — one row per question, answer columns R1-R5 per agent
    Reasoning_Quality  — one row per (question, round, agent) LLM judge score
    Diagnostic_Scores  — one row per question, aggregate metrics
    Round_State        — one row per (question, round, agent), full round-level
                         diagnostics and labels (see ROUND_STATE_COLUMNS below)
    Run_Metadata       — key-value experiment configuration

Round_State is the key addition over Exp1.  It stores every computable
per-round signal including prefix-safe process diagnostics, answer dynamics,
influence attribution, and trajectory labels (answer_at_next_round,
drops_next_round, is_correct_minority, etc.).  This sheet is the direct
input to extract_features_prefix.py and the AIB learned predictor.

Usage:
    # Full run (GPU required, ~14 hours):
    python Pred_Minority/generate_baseline_v2.py --require-gpu

    # Quick 5-question test before committing:
    python Pred_Minority/generate_baseline_v2.py --dataset mmlu-pro --seeds 7
        --objective-limit 5 --require-gpu

    # Resume after interruption — just re-run the same command (automatic):
    python Pred_Minority/generate_baseline_v2.py --require-gpu
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

sys.path.insert(0, str(Path(__file__).parent.parent / "docs"))

from qwen_methodology_code import (
    DebateQuestion,
    LocalQwenPipeline,
    OBJECTIVE_LABELS,
    QWEN_AGENTS,
    EPS,
    SYSTEM_PROMPT,
    answer_is_valid,
    build_turn_prompt,
    clean_text,
    coerce_options,
    empty_judgments,
    final_answer_correctness,
    first_consensus_round_for_answer,
    majority_answer,
    mmlu_pro_correct_label,
    normalize_answer,
    ordered_agents_and_rounds,
    parse_confidence,
    parse_qwen_judgment,
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

DEFAULT_MODEL_ID     = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUT_DIR      = Path("data/baseline_v2")
DEFAULT_SEEDS        = [7, 17, 42]
MMLU_PRO_LIMIT       = 200
GPQA_LIMIT           = 100
DEFAULT_ROUNDS       = 5
DEFAULT_TEMP         = 0.7
DEFAULT_TOP_P        = 0.9
DEFAULT_MAX_TOKENS   = 220
DEFAULT_JUDGE_TOKENS = 220
DEFAULT_JUDGE_BATCH  = 15

REQUIRED_SHEETS = {
    "Debate_Traces",
    "Reasoning_Quality",
    "Diagnostic_Scores",
    "Run_Metadata",
    "Round_State",
}

# ---------------------------------------------------------------------------
# Round_State column reference
# (printed with --list-columns, saved as header in every Round_State sheet)
# ---------------------------------------------------------------------------

ROUND_STATE_COLUMNS = [
    # --- Identity ---
    "run_id",               # run counter (dataset × seed)
    "dataset",              # mmlu-pro / gpqa
    "seed",                 # random seed
    "question_no",          # question identifier
    "correct_answer",       # ground truth label
    "final_answer",         # debate final answer (filled post-debate)
    "final_answer_correct", # whether final_answer == correct_answer
    "round",                # 1–5
    "agent",                # Agent1/Agent2/Agent3

    # --- Raw agent output ---
    "answer",               # answer label chosen this round
    "confidence",           # normalised confidence (0–1)
    "explanation_text",     # full explanation text
    "response_tokens",      # approximate token length of explanation
    "re_prompted",          # True if agent was re-prompted for parse failure

    # --- Answer dynamics (agent-level) ---
    "prev_answer",          # answer in previous round (NaN for round 1)
    "answer_changed",       # True if answer != prev_answer
    "joined_majority",      # moved from non-majority to majority answer
    "left_majority",        # moved from majority to non-majority answer
    "is_minority",          # answer has strictly less than max support
    "support",              # count of agents holding this answer (incl. self)
    "support_delta",        # support change vs previous round
    "minority_size",        # how many agents share this minority answer (0 if majority)
    "was_majority_before",  # this answer was majority in previous round
    "num_defectors",        # agents who left this answer this round
    "num_joiners",          # agents who joined this answer this round

    # --- Debate-level answer distribution (same for all agents in round) ---
    "majority_answer",      # most common answer at this round
    "majority_support",     # count of agents on majority answer
    "n_distinct_answers",   # number of distinct answers at this round
    "consensus_reached",    # all agents agree

    # --- Prefix-safe process diagnostics (rounds 1..t, NO leakage) ---
    "prefix_engagement",          # quality-weighted answer changes
    "prefix_responsiveness",      # quality-weighted peer-following updates
    "prefix_influence_asymmetry", # entropy-based influence concentration (0=equal, 1=dominated)
    "prefix_balance",             # collapse+oscillation penalty
    "prefix_dispersion",          # normalised entropy of answer distribution at round t

    # --- Agent-level influence (from prefix computation) ---
    "prefix_influence_share",       # this agent's share of total attributed influence
    "prefix_attributed_influence",  # raw influence score for this agent

    # --- Trajectory labels (use rounds > t — ONLY for labels, not features) ---
    "answer_at_next_round",     # answer in round t+1 (NaN for round 5)
    "drops_next_round",         # True if answer disappears in round t+1
    "answer_at_final",          # answer at round 5
    "answer_survives_to_final", # True if answer still held at round 5

    # --- Predictor labels ---
    "is_correct",               # answer == correct_answer
    "is_correct_minority",      # is_minority AND is_correct
    "correct_drops_next",       # is_correct AND drops_next_round
    "correct_survives_final",   # is_correct AND answer_survives_to_final
]


# ---------------------------------------------------------------------------
# Prefix-safe diagnostic computation (no future leakage)
# ---------------------------------------------------------------------------

def _normalized_entropy(values: list) -> float:
    vals = [v for v in values if v and str(v) != "nan"]
    if len(vals) <= 1:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return float(-sum(pi * np.log(pi) for pi in p if pi > 0) / np.log(len(vals)))


def _corrected_balance(dispersion: list[float]) -> float:
    n = len(dispersion)
    if n <= 1:
        return np.nan
    steps = [dispersion[t - 1] - dispersion[t] for t in range(1, n)]
    pos = [max(0.0, s) for s in steps]
    total_pos = sum(pos)
    max_pos = max(pos) if pos else 0.0
    if total_pos <= EPS:
        collapse = 1.0 if max(abs(s) for s in steps) <= EPS else 0.0
    else:
        collapse = 1.0 - max_pos / (total_pos + EPS)
    if n <= 2:
        volatility = 1.0
    else:
        reversals = sum(steps[i - 1] * steps[i] < 0 for i in range(1, len(steps)))
        volatility = 1.0 - reversals / (n - 2)
    return float(np.clip(collapse * volatility, 0.0, 1.0))


def compute_prefix_diagnostics(
    answers_by_round: list[list],   # [round_idx][agent_idx] -> answer str or None
    quality_by_round: list[list],   # [round_idx][agent_idx] -> float quality
    up_to_round: int,               # 1-indexed inclusive
) -> dict:
    """Compute all prefix-safe diagnostics from rounds 1..up_to_round.

    Returns a dict with keys:
        engagement, responsiveness, influence_asymmetry, balance, dispersion,
        influence_per_agent (list of floats), influence_share_per_agent (list)
    """
    answers = answers_by_round[:up_to_round]
    quality = quality_by_round[:up_to_round]
    n_agents = len(answers[0]) if answers else len(QWEN_AGENTS)

    result = {
        "engagement":           np.nan,
        "responsiveness":       np.nan,
        "influence_asymmetry":  0.0,
        "balance":              np.nan,
        "dispersion":           _normalized_entropy(answers[-1]) if answers else 0.0,
        "influence_per_agent":  [0.0] * n_agents,
        "influence_share_per_agent": [np.nan] * n_agents,
    }

    if len(answers) < 2:
        return result

    eng_terms, resp_terms = [], []
    influence = np.zeros(n_agents, dtype=float)

    for t in range(1, len(answers)):
        for a in range(n_agents):
            old = answers[t - 1][a]
            new = answers[t][a]
            if not old or not new or str(old) == "nan" or str(new) == "nan":
                continue
            q = float(quality[t][a]) if quality[t][a] is not None else 0.5
            changed = (old != new)
            eng_terms.append(q * float(changed))

            others_prev = [
                answers[t - 1][j] for j in range(n_agents)
                if j != a and answers[t - 1][j] and str(answers[t - 1][j]) != "nan"
            ]
            if others_prev:
                old_sup = sum(o == old for o in others_prev)
                new_sup = sum(o == new for o in others_prev)
                resp_terms.append(q * float(changed and new_sup > old_sup))

            for src in range(n_agents):
                src_ans = answers[t - 1][src]
                if src == a or not src_ans or str(src_ans) == "nan":
                    continue
                if changed and new == src_ans and old != src_ans:
                    influence[src] += q

    total_inf = float(influence.sum())
    if total_inf > EPS and n_agents > 1:
        p_inf = influence / total_inf
        h = -sum(pi * np.log(pi) for pi in p_inf if pi > 0)
        result["influence_asymmetry"]       = float(1.0 - h / np.log(n_agents))
        result["influence_share_per_agent"] = p_inf.tolist()
    result["influence_per_agent"] = influence.tolist()

    dispersion = [_normalized_entropy(rnd) for rnd in answers]
    result["balance"]         = _corrected_balance(dispersion)
    result["dispersion"]      = dispersion[-1]
    result["engagement"]      = float(np.mean(eng_terms))  if eng_terms  else np.nan
    result["responsiveness"]  = float(np.mean(resp_terms)) if resp_terms else np.nan
    return result


# ---------------------------------------------------------------------------
# Round-state builder
# ---------------------------------------------------------------------------

def build_round_state(
    all_rounds: list[dict],   # list of {agent: parsed_turn} dicts
    question: DebateQuestion,
    final_answer: str,
    dataset_label: str,
    seed: int,
    run_id: int,
) -> list[dict]:
    """Build one row per (round, agent) with all computable state.

    Trajectory labels (drops_next_round, answer_at_final, etc.) require the
    complete debate to be finished — they are computed here after all rounds.
    These columns are clearly labeled for use as LABELS ONLY, not features.
    """
    n_rounds = len(all_rounds)
    n_agents = len(QWEN_AGENTS)

    # Build answer / quality matrices
    answers_by_round: list[list] = []
    quality_by_round: list[list] = []
    for turns in all_rounds:
        answers_by_round.append([
            str(turns[a].get("answer", "") or "").strip() or None
            for a in QWEN_AGENTS
        ])
        quality_by_round.append([
            float(turns[a].get("confidence", 0.5) or 0.5)
            for a in QWEN_AGENTS
        ])

    final_correct = (final_answer == question.correct_answer)
    rows = []

    for round_idx, turns in enumerate(all_rounds):
        round_num = round_idx + 1

        # Prefix diagnostics computed up to this round (leak-free)
        diag = compute_prefix_diagnostics(
            answers_by_round, quality_by_round, round_num
        )

        curr_answers = answers_by_round[round_idx]
        prev_answers = answers_by_round[round_idx - 1] if round_idx > 0 else [None] * n_agents
        next_answers = answers_by_round[round_idx + 1] if round_idx + 1 < n_rounds else [None] * n_agents
        final_answers = answers_by_round[-1]

        curr_counts = Counter(a for a in curr_answers if a)
        prev_counts = Counter(a for a in prev_answers if a)
        maj_answer = curr_counts.most_common(1)[0][0] if curr_counts else ""
        maj_support = curr_counts.most_common(1)[0][1] if curr_counts else 0
        n_distinct  = len(curr_counts)
        consensus   = (n_distinct == 1 and len(curr_counts) > 0)

        for a_idx, agent in enumerate(QWEN_AGENTS):
            turn = turns[agent]
            ans  = curr_answers[a_idx]
            prev = prev_answers[a_idx]
            nxt  = next_answers[a_idx]
            fin  = final_answers[a_idx]

            conf_raw = turn.get("confidence", 0.5)
            conf = float(conf_raw) if conf_raw is not None else 0.5
            if conf > 1.0:
                conf /= 100.0

            expl = str(turn.get("response", "") or "")
            resp_tokens = len(expl.split())

            # Answer dynamics
            support      = curr_counts.get(ans, 0) if ans else 0
            prev_support = prev_counts.get(ans, 0) if ans else 0
            support_delta = support - prev_support
            is_minority   = bool(ans and support < maj_support)
            minority_size = support if is_minority else 0

            prev_max = max(prev_counts.values()) if prev_counts else 0
            was_maj_before = bool(ans and prev_counts.get(ans, 0) >= prev_max and prev_max > 0)

            # Who joined / left this answer this round
            num_defectors = sum(
                1 for j in range(n_agents)
                if prev_answers[j] == ans and curr_answers[j] != ans
            ) if ans else 0
            num_joiners = sum(
                1 for j in range(n_agents)
                if prev_answers[j] != ans and curr_answers[j] == ans
            ) if ans else 0

            answer_changed   = bool(prev and ans and ans != prev)
            joined_majority  = bool(ans == maj_answer and prev and prev != maj_answer)
            left_majority    = bool(prev and prev == (prev_counts.most_common(1)[0][0] if prev_counts else "") and ans != prev)

            # Trajectory labels
            drops_next      = bool(ans and nxt is not None and nxt != ans)
            survives_final  = bool(ans and fin == ans)
            is_correct      = bool(ans and ans == question.correct_answer)

            rows.append({
                # Identity
                "run_id":                  run_id,
                "dataset":                 dataset_label,
                "seed":                    seed,
                "question_no":             question.question_no,
                "correct_answer":          question.correct_answer,
                "final_answer":            final_answer,
                "final_answer_correct":    final_correct,
                "round":                   round_num,
                "agent":                   agent,

                # Raw output
                "answer":                  ans or "",
                "confidence":              round(conf, 4),
                "explanation_text":        expl,
                "response_tokens":         resp_tokens,
                "re_prompted":             bool(turn.get("re_prompted", False)),

                # Answer dynamics
                "prev_answer":             prev or "",
                "answer_changed":          answer_changed,
                "joined_majority":         joined_majority,
                "left_majority":           left_majority,
                "is_minority":             is_minority,
                "support":                 support,
                "support_delta":           support_delta,
                "minority_size":           minority_size,
                "was_majority_before":     was_maj_before,
                "num_defectors":           num_defectors,
                "num_joiners":             num_joiners,

                # Debate-level distribution
                "majority_answer":         maj_answer,
                "majority_support":        maj_support,
                "n_distinct_answers":      n_distinct,
                "consensus_reached":       consensus,

                # Prefix-safe diagnostics
                "prefix_engagement":              _safe(diag["engagement"]),
                "prefix_responsiveness":          _safe(diag["responsiveness"]),
                "prefix_influence_asymmetry":     round(diag["influence_asymmetry"], 6),
                "prefix_balance":                 _safe(diag["balance"]),
                "prefix_dispersion":              round(diag["dispersion"], 6),
                "prefix_influence_share":         round(float(diag["influence_share_per_agent"][a_idx]), 6)
                                                  if not np.isnan(diag["influence_share_per_agent"][a_idx]) else np.nan,
                "prefix_attributed_influence":    round(float(diag["influence_per_agent"][a_idx]), 6),

                # Trajectory labels (post-debate, for learning targets only)
                "answer_at_next_round":    nxt or "",
                "drops_next_round":        drops_next,
                "answer_at_final":         fin or "",
                "answer_survives_to_final": survives_final,

                # Predictor labels
                "is_correct":              is_correct,
                "is_correct_minority":     bool(is_minority and is_correct),
                "correct_drops_next":      bool(is_correct and drops_next),
                "correct_survives_final":  bool(is_correct and survives_final),
            })

    return rows


def _safe(v) -> float:
    """Round a float, preserving NaN."""
    return round(float(v), 6) if v is not None and not np.isnan(float(v)) else np.nan


# ---------------------------------------------------------------------------
# GPQA loader
# ---------------------------------------------------------------------------

def load_gpqa_questions(
    limit: int, dataset_name: str, subset: str
) -> list[DebateQuestion]:
    """Load GPQA from HuggingFace (Idavidrein/gpqa).

    Columns used: Question, Correct Answer, Incorrect Answer 1/2/3
    Options are shuffled deterministically by question index so the
    correct answer is not always option A.
    """
    if limit <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "GPQA loading requires `datasets`. Install: pip install datasets"
        ) from exc

    print(f"Loading GPQA ({subset}, limit={limit})...", flush=True)
    rows = list(load_dataset(dataset_name, subset, split="train"))

    questions: list[DebateQuestion] = []
    for idx, row in enumerate(rows):
        if len(questions) >= limit:
            break
        q_text   = clean_text(row.get("Question", ""))
        correct  = clean_text(row.get("Correct Answer", ""))
        wrong1   = clean_text(row.get("Incorrect Answer 1", ""))
        wrong2   = clean_text(row.get("Incorrect Answer 2", ""))
        wrong3   = clean_text(row.get("Incorrect Answer 3", ""))
        if not q_text or not correct:
            continue

        options_raw = [correct, wrong1, wrong2, wrong3]
        rng = random.Random(idx)          # deterministic per question
        rng.shuffle(options_raw)
        options = [o for o in options_raw if o]
        if len(options) < 2:
            continue

        labels        = OBJECTIVE_LABELS[: len(options)]
        correct_label = labels[options.index(correct)]
        formatted     = "\n".join(f"{l}. {o}" for l, o in zip(labels, options))
        q_no          = clean_text(row.get("Record ID", "")) or f"gpqa-{idx + 1:04d}"

        questions.append(DebateQuestion(
            dataset_type  = "objective",
            question_no   = str(q_no),
            question      = f"{q_text}\n{formatted}",
            correct_answer = correct_label,
            answer_labels  = labels,
            category       = clean_text(row.get("High-level domain", "science")),
        ))

    print(f"Loaded {len(questions)} GPQA questions.", flush=True)
    return questions


# ---------------------------------------------------------------------------
# Per-question checkpoint
# ---------------------------------------------------------------------------

def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.jsonl"


def load_checkpoint(run_dir: Path) -> tuple[list[dict], list[dict], set]:
    """Return (debate_rows, round_state_rows, done_question_nos).

    Deduplicates by question_no — safe even if the file was accidentally
    appended to from a prior non-resume run.
    """
    cp = checkpoint_path(run_dir)
    debate_by_qno: dict[str, dict] = {}   # keep last entry per question_no
    state_by_qno:  dict[str, list] = {}
    if not cp.exists():
        return [], [], set()
    with cp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            qno = entry.get("question_no", "")
            if entry["_type"] == "debate":
                debate_by_qno[qno] = entry["data"]
            elif entry["_type"] == "state":
                state_by_qno[qno] = entry["data"]

    # Bugs 3+4 fix: only trust questions where BOTH debate AND state are present.
    # A crash mid-write can leave a debate entry without a state entry; treating
    # such a question as "done" would cause it to be skipped forever while its
    # Round_State rows are silently absent.  Re-running it is safe (idempotent).
    complete_qnos  = {k for k in debate_by_qno if k in state_by_qno}
    orphan_debates = set(debate_by_qno.keys()) - complete_qnos
    if orphan_debates:
        print(
            f"WARNING: {len(orphan_debates)} question(s) have a debate entry but "
            f"no state entry (likely from a mid-write crash). They will be re-run: "
            f"{sorted(orphan_debates)}",
            flush=True,
        )

    done        = complete_qnos
    debate_rows = [debate_by_qno[k] for k in debate_by_qno if k in complete_qnos]
    state_rows  = [row for k in debate_by_qno if k in complete_qnos
                   for row in state_by_qno[k]]
    print(
        f"Checkpoint loaded: {len(done)} completed questions, "
        f"{len(state_rows)} round-state rows.",
        flush=True,
    )
    return debate_rows, state_rows, done


def clear_checkpoint(run_dir: Path) -> None:
    """Delete debate and judge checkpoints so a fresh run starts clean."""
    for path in (checkpoint_path(run_dir), judge_cache_path(run_dir)):
        if path.exists():
            path.unlink()
            print(f"Cleared checkpoint: {path}", flush=True)


def save_checkpoint(
    run_dir: Path,
    question_no: str,
    debate_row: dict,
    state_rows: list[dict],
) -> None:
    """Append one completed question to the debate checkpoint atomically.

    Both the debate row and the state rows are concatenated into a single
    string and written with one f.write() call.  This ensures that a crash
    or power failure cannot leave the debate entry written but the state
    entry absent (which would cause the question to be skipped on resume
    while its Round_State rows are silently lost).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    line1 = json.dumps({"_type": "debate", "question_no": question_no,
                        "data": debate_row})
    line2 = json.dumps({"_type": "state",  "question_no": question_no,
                        "data": state_rows})
    with checkpoint_path(run_dir).open("a", encoding="utf-8") as f:
        f.write(line1 + "\n" + line2 + "\n")  # single write = atomic on most OSes


# ---------------------------------------------------------------------------
# Per-call judge checkpoint  (covers the ~2.5 h judging phase)
# ---------------------------------------------------------------------------

def judge_cache_path(run_dir: Path) -> Path:
    return run_dir / "judge_cache.jsonl"


def load_judge_cache(run_dir: Path) -> tuple[list[dict], set]:
    """Return (cached_rows, done_keys).

    done_keys is a set of (question_no, round, agent) tuples already judged.
    Deduplicates automatically — safe even if the file was double-written.
    """
    cp = judge_cache_path(run_dir)
    rows_by_key: dict[tuple, dict] = {}
    if not cp.exists():
        return [], set()
    with cp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (entry["question_no"], entry["round"], entry["agent"])
            rows_by_key[key] = entry        # keep last write per (q, r, a)

    done = set(rows_by_key.keys())
    print(f"Judge cache loaded: {len(done)} calls already completed.", flush=True)
    return list(rows_by_key.values()), done


def save_judge_result(run_dir: Path, row: dict) -> None:
    """Append one judge result immediately to the judge cache."""
    with judge_cache_path(run_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def judge_with_cache(
    llm: LocalQwenPipeline,
    debates: pd.DataFrame,
    source_file: str,
    seed: int,
    judge_max_new_tokens: int,
    judge_batch_size: int,
    sleep: float,
    run_dir: Path,
) -> pd.DataFrame:
    """Judge all debates with per-call caching for safe resumption.

    Replaces judge_debates_with_qwen().  Every judge result is written to
    run_dir/judge_cache.jsonl immediately after the LLM returns, so
    interrupting mid-judging only loses the current batch (≤ judge_batch_size
    calls, typically ≤ 15 seconds of work).

    On resume=True, already-cached results are loaded and skipped.
    """
    agents, rounds = ordered_agents_and_rounds(debates)

    # Always load from judge cache if one exists.
    # The cache is only wiped by --overwrite (handled upstream).
    cached_rows, done_keys = load_judge_cache(run_dir)

    new_rows: list[dict] = []

    for row_index, row in debates.iterrows():
        q_no = row.get("Question #")
        question_num = int(row_index) + 1
        total_q = len(debates)

        # Build tasks for this question, skipping cached ones
        tasks = []
        for round_no in rounds:
            for agent in agents:
                key = (str(q_no), int(round_no), agent)
                if key in done_keys:
                    continue
                task_seed = (
                    seed
                    + int(row_index) * 1000
                    + round_no * 100
                    + agents.index(agent)
                )
                tasks.append((
                    round_no,
                    agent,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_turn_prompt(
                            row, source_file, round_no, agent, agents
                        )},
                    ],
                    task_seed,
                ))

        if not tasks:
            print(
                f"  Judging q {question_num}/{total_q}: {q_no} [all cached]",
                flush=True,
            )
            continue

        n_cached = len(rounds) * len(agents) - len(tasks)
        print(
            f"  Judging q {question_num}/{total_q}: {q_no}  "
            f"({len(tasks)} calls, {n_cached} cached)",
            flush=True,
        )

        # Process in batches; save each result immediately
        batch_size = max(1, judge_batch_size)
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start: batch_start + batch_size]
            messages_batch = [t[2] for t in batch]
            batch_seeds    = [t[3] for t in batch]

            raws = llm.complete_many(
                messages_batch,
                seeds=batch_seeds,
                max_new_tokens=judge_max_new_tokens,
            )

            for (round_no, agent, _, _), raw in zip(batch, raws):
                judged = parse_qwen_judgment(raw)
                result_row = {
                    "source_file":     source_file,
                    "sheet_name":      "Debate_Traces",
                    "row_index":       int(row_index),
                    "question_no":     str(q_no),
                    "round":           int(round_no),
                    "agent":           agent,
                    "answer":          normalize_answer(
                                           row.get(f"R{round_no} {agent} Answer")
                                       ),
                    "raw_answer":      row.get(f"R{round_no} {agent} Answer"),
                    "raw_confidence":  parse_confidence(
                                           row.get(f"R{round_no} {agent} Conf")
                                       ),
                    **judged,
                }
                new_rows.append(result_row)
                save_judge_result(run_dir, result_row)   # immediate write

            if sleep:
                time.sleep(sleep)

    all_rows = cached_rows + new_rows
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Single debate runner
# ---------------------------------------------------------------------------

def run_one_debate(
    llm: LocalQwenPipeline,
    question: DebateQuestion,
    seed: int,
    rounds: int,
    sleep: float,
    rng: random.Random,
    run_id: int,
    dataset_label: str,
) -> tuple[dict | None, list[dict], str]:
    """Run one debate; return (debate_row, round_state_rows, discard_reason)."""
    agent_order = QWEN_AGENTS[:]
    rng.shuffle(agent_order)
    all_rounds: list[dict] = []

    for round_no in range(1, rounds + 1):
        messages = {}
        for agent in agent_order:
            if round_no == 1:
                messages[agent] = qwen_initial_messages(question, agent)
            else:
                messages[agent] = qwen_update_messages(question, agent, all_rounds[-1])

        round_turns: dict[str, dict] = {}
        discard_reason = ""
        for agent in agent_order:
            base_seed = seed + round_no * 100 + QWEN_AGENTS.index(agent) * 10
            raw = llm.complete(messages[agent], seed=base_seed,
                               max_new_tokens=llm.max_new_tokens)
            parsed = parse_qwen_turn(
                raw, question.dataset_type, question.answer_labels, strict=True
            )
            if parsed["parse_failed"]:
                retry = qwen_reprompt_messages(messages[agent], question)
                raw2  = llm.complete(retry, seed=base_seed + 1,
                                     max_new_tokens=llm.max_new_tokens)
                p2    = parse_qwen_turn(
                    raw2, question.dataset_type, question.answer_labels, strict=True
                )
                if not p2["parse_failed"]:
                    parsed = {**p2, "re_prompted": True}
                else:
                    discard_reason = (
                        f"{question.question_no} r{round_no} {agent}: "
                        f"{parsed['parse_error']}"
                    )
                    break
            round_turns[agent] = parsed

        if sleep:
            time.sleep(sleep)
        if discard_reason:
            print(f"  Discarding: {discard_reason}", flush=True)
            return None, [], discard_reason
        all_rounds.append(round_turns)

    # Final answer
    final_ans_list = [all_rounds[-1][a]["answer"] for a in QWEN_AGENTS]
    valid = [a for a in final_ans_list if a]
    if valid and len(set(valid)) == 1:
        final_answer, final_source = valid[0], "agent_consensus"
    else:
        raw = llm.complete(
            qwen_moderator_messages(question, all_rounds),
            seed=seed + rounds * 100 + 999
        )
        parsed = parse_qwen_turn(
            raw, question.dataset_type, question.answer_labels, strict=False
        )
        candidate = parsed["answer"]
        if answer_is_valid(candidate, question):
            final_answer, final_source = candidate, "moderator"
        else:
            final_answer = majority_answer(final_ans_list)
            final_source = "majority_vote_no_moderator"

    # Debate trace row
    debate_row: dict = {
        "Question #":          question.question_no,
        "Dataset Type":        question.dataset_type,
        "Dataset Category":    question.category,
        "Question":            question.question,
        "Correct Answer":      question.correct_answer,
        "Final Answer":        final_answer,
        "Final Answer Source": final_source,
        "Fixture Pattern":     "baseline_v2",
        "Rounds to Consensus": first_consensus_round_for_answer(all_rounds, final_answer),
    }
    for rn, turns in enumerate(all_rounds, start=1):
        for agent in QWEN_AGENTS:
            t = turns[agent]
            debate_row[f"R{rn} {agent} Answer"]   = t["answer"]
            debate_row[f"R{rn} {agent} Conf"]     = t.get("confidence", "")
            debate_row[f"R{rn} {agent} Response"] = t.get("response", "")

    # Round-state rows (all round-level diagnostics + labels)
    state_rows = build_round_state(
        all_rounds, question, final_answer,
        dataset_label=dataset_label, seed=seed, run_id=run_id,
    )

    return debate_row, state_rows, ""


# ---------------------------------------------------------------------------
# Seed-level runner (uses checkpoint)
# ---------------------------------------------------------------------------

def run_debates_for_seed(
    llm: LocalQwenPipeline,
    questions: list[DebateQuestion],
    seed: int,
    run_dir: Path,
    args: argparse.Namespace,
    dataset_label: str,
    run_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all debates for one (dataset, seed) pair; return (debates, round_state)."""
    # Always load from checkpoint if one exists.
    # Checkpoints are cleared only by --overwrite (handled in run_one_seed
    # before this function is called).  load_checkpoint returns empty lists
    # when no checkpoint file is present, so this is safe on a fresh run.
    debate_rows, state_rows, done_nos = load_checkpoint(run_dir)

    for q_idx, question in enumerate(questions):
        # Per-question RNG: derived deterministically from (seed, q_idx).
        # This avoids fragile global RNG state tracking for skipped questions —
        # each question's agent shuffle is always the same regardless of which
        # questions were skipped on resume.
        q_rng = random.Random(seed * 100_000 + q_idx)

        if question.question_no in done_nos:
            print(
                f"  {dataset_label} s={seed} q={q_idx+1}/{len(questions)}: "
                f"{question.question_no} [skip — debate checkpoint]",
                flush=True,
            )
            continue

        print(
            f"  {dataset_label} s={seed} q={q_idx+1}/{len(questions)}: "
            f"{question.question_no}",
            flush=True,
        )

        debate_row, q_state_rows, _ = run_one_debate(
            llm, question,
            seed=seed + q_idx * 1000,
            rounds=args.rounds,
            sleep=args.sleep,
            rng=q_rng,
            run_id=run_id,
            dataset_label=dataset_label,
        )

        if debate_row is None:
            continue

        debate_rows.append(debate_row)
        state_rows.extend(q_state_rows)
        save_checkpoint(run_dir, question.question_no, debate_row, q_state_rows)

    debates = pd.DataFrame(debate_rows)
    if not debates.empty:
        debates["Correct?"] = debates.apply(final_answer_correctness, axis=1)

    return debates, pd.DataFrame(state_rows)


# ---------------------------------------------------------------------------
# Workbook writer for one seed
# ---------------------------------------------------------------------------

def workbook_is_complete(path: Path) -> bool:
    try:
        return REQUIRED_SHEETS.issubset(set(pd.ExcelFile(path).sheet_names))
    except Exception:
        return False


def run_one_seed(
    llm: LocalQwenPipeline,
    questions: list[DebateQuestion],
    seed: int,
    dataset_label: str,
    run_dir: Path,
    run_id: int,
    args: argparse.Namespace,
) -> Path:
    output_path = run_dir / f"baseline_v2_{dataset_label}_s{seed}.xlsx"
    source_file = output_path.name

    # Skip entirely if workbook is already complete, unless --overwrite is set.
    # This protects against accidental re-runs (with OR without --resume).
    if not args.overwrite:
        if output_path.exists() and workbook_is_complete(output_path):
            print(f"[{dataset_label} s={seed}] Workbook complete — skipping.",
                  flush=True)
            return output_path

    # If --overwrite, wipe checkpoints AND any stale workbook so a subsequent
    # crash cannot leave the old workbook as a false "complete" result (Bug 8).
    if args.overwrite:
        print(f"[{dataset_label} s={seed}] --overwrite: clearing checkpoints.",
              flush=True)
        clear_checkpoint(run_dir)
        if output_path.exists():
            output_path.unlink()
            print(f"  Removed stale workbook: {output_path}", flush=True)
        tmp = output_path.with_suffix(".tmp.xlsx")
        if tmp.exists():
            tmp.unlink()

    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Debates: {dataset_label} seed={seed} ===", flush=True)
    debates, round_state = run_debates_for_seed(
        llm, questions, seed, run_dir, args, dataset_label, run_id
    )

    if debates.empty:
        raise RuntimeError(f"No debates for {dataset_label} seed={seed}.")
    print(f"Completed {len(debates)} debates, {len(round_state)} round-state rows.",
          flush=True)

    print(f"\n=== Judging: {dataset_label} seed={seed} ===", flush=True)
    if args.skip_judging or args.q_source == "confidence":
        judgments = empty_judgments()
    else:
        judgments = judge_with_cache(
            llm, debates,
            source_file=source_file,
            seed=seed + 100_000,
            judge_max_new_tokens=args.judge_max_new_tokens,
            judge_batch_size=args.judge_batch_size,
            sleep=args.sleep,
            run_dir=run_dir,
        )
        print(f"Completed {len(judgments)} judgments.", flush=True)

    if judgments.empty and args.q_source == "llm":
        scores = pd.DataFrame()
    else:
        judgments["source_file"] = source_file
        scores = score_mixed_debates(
            debates, judgments, source_file,
            q_source=args.q_source,
            metric_version=args.metric_version,
        )

    # Tag round_state with LLM quality scores.
    # Use question_no directly from the judgments DataFrame — it was stored
    # there during judging and is stable regardless of row ordering.
    # Do NOT re-derive question_no from row_index, which would silently
    # mis-assign scores if debate row order ever changes.
    if not judgments.empty and not round_state.empty:
        judge_cols = ["question_no", "round", "agent",
                      "explanation_good", "uses_past_round_reasoning",
                      "justifies_current_stance", "independent_reasoning",
                      "judge_parse_failed"]
        jq = judgments[[c for c in judge_cols if c in judgments.columns]].copy()
        jq = jq.rename(columns={
            "explanation_good":          "judge_explanation_good",
            "uses_past_round_reasoning": "judge_uses_past_reasoning",
            "justifies_current_stance":  "judge_justifies_stance",
            "independent_reasoning":     "judge_independent_reasoning",
        })
        jq["question_no"] = jq["question_no"].astype(str)
        round_state["question_no"] = round_state["question_no"].astype(str)
        round_state = round_state.merge(
            jq[["question_no", "round", "agent",
                "judge_explanation_good", "judge_uses_past_reasoning",
                "judge_justifies_stance", "judge_independent_reasoning",
                "judge_parse_failed"]],
            on=["question_no", "round", "agent"],
            how="left",
        )

    # Write workbook atomically via a temp file.
    # Using two separate pd.ExcelWriter contexts (one for the base sheets,
    # one to append Round_State) creates a crash window: a kill between the two
    # writes leaves an incomplete workbook that looks like it exists but fails
    # workbook_is_complete(), causing a permanent no-recovery loop (Bug 5).
    # Writing everything to a .tmp file then os.replace() is atomic on Windows
    # NTFS and on Linux/macOS, so workbook_is_complete() can never see a partial file.
    import os
    tmp_path = output_path.with_suffix(".tmp.xlsx")
    run_args = argparse.Namespace(**vars(args))
    run_args.backend = "transformers"
    run_args.seed    = seed

    # Step 1: write base sheets to the temp file
    write_qwen_excel_report(
        tmp_path,
        f"baseline_v2_{dataset_label}",
        run_args, debates, judgments, scores,
    )
    # Step 2: append Round_State to the temp file
    with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        round_state.to_excel(writer, sheet_name="Round_State", index=False)

    # Step 3: atomic rename — output_path is either absent or fully complete
    os.replace(tmp_path, output_path)
    print(f"Wrote {output_path}  (+Round_State: {len(round_state)} rows)", flush=True)
    return output_path


# ---------------------------------------------------------------------------
# Combine workbooks
# ---------------------------------------------------------------------------

def combine_all(
    output_path: Path,
    run_specs: list[tuple[Path, str, int]],
    args: argparse.Namespace,
) -> None:
    debate_frames, judg_frames, score_frames, state_frames = [], [], [], []
    metadata_rows = [
        {"field": "run_mode",        "value": "baseline_v2_multi_seed_multi_dataset"},
        {"field": "model_id",        "value": args.model_id},
        {"field": "mmlu_pro_limit",  "value": MMLU_PRO_LIMIT},
        {"field": "gpqa_limit",      "value": GPQA_LIMIT},
        {"field": "seeds",           "value": ",".join(str(s) for s in args.seeds)},
        {"field": "num_seeds",       "value": len(args.seeds)},
        {"field": "rounds",          "value": args.rounds},
        {"field": "temperature",     "value": args.temperature},
        {"field": "top_p",           "value": args.top_p},
        {"field": "q_source",        "value": args.q_source},
        {"field": "round_state_note","value": (
            "Round_State has one row per (question, round, agent). "
            "Prefix diagnostics are leak-free (computed from rounds 1..t). "
            "Trajectory labels (drops_next_round etc.) use future rounds — "
            "use only as labels, not features. "
            "CV grouping: group by cv_group = dataset__question_no."
        )},
    ]
    row_offset = 0
    for run_id, (path, dataset_label, seed) in enumerate(run_specs, start=1):
        debate = pd.read_excel(path, sheet_name="Debate_Traces")
        judg   = pd.read_excel(path, sheet_name="Reasoning_Quality")
        score  = pd.read_excel(path, sheet_name="Diagnostic_Scores")
        state  = pd.read_excel(path, sheet_name="Round_State")

        for df in (debate, judg, score, state):
            if "Run ID" not in df.columns:
                df.insert(0, "Run ID", run_id)
            if "Seed" not in df.columns:
                df.insert(1, "Seed", seed)
            if "Dataset" not in df.columns:
                df.insert(2, "Dataset", dataset_label)

        # Bug 7 fix: advance row_offset by len(score), not len(debate).
        # If score_mixed_debates ever filters out rows, len(score) < len(debate)
        # and using len(debate) would leave gaps / overlaps in row_index for
        # subsequent seeds in the combined workbook.
        score["row_index"] = np.arange(row_offset, row_offset + len(score))

        # cv_group: all seeds of same (dataset, question) stay in one fold
        state["cv_group"] = dataset_label + "__" + state["question_no"].astype(str)

        debate_frames.append(debate)
        judg_frames.append(judg)
        score_frames.append(score)
        state_frames.append(state)
        row_offset += len(score)   # Bug 7 fix: was len(debate)

    debates   = pd.concat(debate_frames, ignore_index=True)
    judgments = pd.concat(judg_frames,   ignore_index=True)
    scores    = pd.concat(score_frames,  ignore_index=True)
    states    = pd.concat(state_frames,  ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        debates  .to_excel(writer, sheet_name="Debate_Traces",    index=False)
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Run_Metadata", index=False)
        judgments.to_excel(writer, sheet_name="Reasoning_Quality", index=False)
        scores   .to_excel(writer, sheet_name="Diagnostic_Scores", index=False)
        states   .to_excel(writer, sheet_name="Round_State",       index=False)

    print(f"\nCombined workbook: {output_path}", flush=True)
    print(f"  Debates:     {len(debates)}", flush=True)
    print(f"  Round_State: {len(states)}", flush=True)

    # Summary stats
    if "is_correct_minority" in states.columns:
        r24 = states[states["round"].between(2, 4)]
        print(f"\n  Minority obs (rounds 2-4):  {len(r24)}", flush=True)
        print(f"  Correct minority (R2-4):    {r24['is_correct_minority'].sum()}", flush=True)
        print(f"  Correct+drops (R2-4):       {r24['correct_drops_next'].sum()}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed, multi-dataset baseline for AIB learned predictor."
    )
    p.add_argument("--model-id",          default=DEFAULT_MODEL_ID)
    p.add_argument("--out-dir",           type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seeds",             type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--dataset",           choices=["mmlu-pro", "gpqa", "both"],
                   default="both")
    p.add_argument("--objective-limit",   type=int, default=None,
                   help="Override question limit (default 200 MMLU, 100 GPQA)")
    p.add_argument("--rounds",            type=int,   default=DEFAULT_ROUNDS)
    p.add_argument("--temperature",       type=float, default=DEFAULT_TEMP)
    p.add_argument("--top-p",             type=float, default=DEFAULT_TOP_P)
    p.add_argument("--max-new-tokens",    type=int,   default=DEFAULT_MAX_TOKENS)
    p.add_argument("--judge-max-new-tokens", type=int, default=DEFAULT_JUDGE_TOKENS)
    p.add_argument("--judge-batch-size",  type=int,   default=DEFAULT_JUDGE_BATCH)
    p.add_argument("--device-map",        default="auto")
    p.add_argument("--torch-dtype",       default="auto")
    p.add_argument("--require-gpu",       action="store_true")
    p.add_argument("--skip-judging",      action="store_true")
    p.add_argument("--q-source",          choices=["llm", "confidence"], default="llm")
    p.add_argument("--metric-version",    choices=["paper", "corrected"], default="paper")
    p.add_argument("--sleep",             type=float, default=0.0)
    p.add_argument("--overwrite",         action="store_true",
                   help=(
                       "Wipe checkpoints and re-run from scratch. "
                       "Without this flag, the program always resumes from "
                       "any existing checkpoint automatically."
                   ))
    p.add_argument("--mmlu-pro-dataset",  default="TIGER-Lab/MMLU-Pro")
    p.add_argument("--mmlu-pro-split",    default="test")
    p.add_argument("--gpqa-dataset",      default="Idavidrein/gpqa")
    p.add_argument("--gpqa-subset",       default="gpqa_main")
    p.add_argument("--list-columns",      action="store_true",
                   help="Print all Round_State columns and exit")
    p.add_argument("--status",            action="store_true",
                   help="Show checkpoint progress for each seed and exit")
    return p.parse_args()


def _print_status(args: argparse.Namespace) -> None:
    """Print resume status for every (dataset, seed) run directory."""
    run_datasets = []
    if args.dataset in ("mmlu-pro", "both"):
        run_datasets.append(("mmlu-pro", args.objective_limit or MMLU_PRO_LIMIT))
    if args.dataset in ("gpqa", "both"):
        run_datasets.append(("gpqa", args.objective_limit or GPQA_LIMIT))

    print("=" * 70)
    print("BASELINE V2 — RESUME STATUS")
    print("=" * 70)

    for dataset_label, q_limit in run_datasets:
        for seed in args.seeds:
            run_dir = args.out_dir / "runs" / f"{dataset_label}_seed_{seed}"
            out_wb  = run_dir / f"baseline_v2_{dataset_label}_s{seed}.xlsx"

            print(f"\n  [{dataset_label}  seed={seed}]  {run_dir}")

            # Workbook complete?
            if out_wb.exists() and workbook_is_complete(out_wb):
                print(f"    Workbook:     COMPLETE  ({out_wb.name})")
                continue

            # Debate checkpoint
            cp = checkpoint_path(run_dir)
            if cp.exists():
                _, _, done = load_checkpoint(run_dir)
                pct = len(done) / q_limit * 100
                print(f"    Debates:      {len(done)}/{q_limit} saved  ({pct:.0f}%)")
            else:
                print(f"    Debates:      0/{q_limit} saved  (not started)")

            # Judge cache
            jcp = judge_cache_path(run_dir)
            if jcp.exists():
                _, done_j = load_judge_cache(run_dir)
                total_calls = q_limit * 5 * 3         # questions × rounds × agents
                pct_j = len(done_j) / total_calls * 100
                print(f"    Judging:      {len(done_j)}/{total_calls} saved  ({pct_j:.0f}%)")
            else:
                print(f"    Judging:      0 saved  (not started)")

    print()
    print("To continue:  just re-run the same command (resume is automatic).")
    print("To restart:   add --overwrite flag.")
    print("=" * 70)


def require_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        print("Warning: nvidia-smi not on PATH.", flush=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch required.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required but CUDA not visible to PyTorch.")
    visible = ", ".join(
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    )
    print(f"CUDA GPU: {visible}", flush=True)


def main() -> None:
    args = parse_args()

    if args.status:
        _print_status(args)
        return

    if args.list_columns:
        print("\nRound_State columns (one row per question × round × agent):\n")
        groups = [
            ("Identity",           ROUND_STATE_COLUMNS[:9]),
            ("Raw agent output",   ROUND_STATE_COLUMNS[9:16]),
            ("Answer dynamics",    ROUND_STATE_COLUMNS[16:27]),
            ("Debate distribution",ROUND_STATE_COLUMNS[27:31]),
            ("Prefix diagnostics", ROUND_STATE_COLUMNS[31:39]),
            ("Trajectory labels",  ROUND_STATE_COLUMNS[39:43]),
            ("Predictor labels",   ROUND_STATE_COLUMNS[43:]),
        ]
        for group_name, cols in groups:
            print(f"  [{group_name}]")
            for col in cols:
                print(f"    {col}")
        return

    if args.require_gpu:
        require_gpu()

    run_datasets = []
    if args.dataset in ("mmlu-pro", "both"):
        run_datasets.append("mmlu-pro")
    if args.dataset in ("gpqa", "both"):
        run_datasets.append("gpqa")

    print("=" * 70, flush=True)
    print("BASELINE V2 — Multi-seed, multi-dataset", flush=True)
    print(f"  Datasets: {run_datasets}  Seeds: {args.seeds}", flush=True)
    print(f"  Model:    {args.model_id}", flush=True)
    print(f"  Rounds:   {args.rounds}   Judging: {args.q_source}", flush=True)
    print("=" * 70, flush=True)

    # Pre-check: two separate signals per seed.
    #   needs_llm  — True when LLM calls are still required (load model)
    #   needs_work — True when anything at all still needs to be done
    #                (includes the case where all LLM is cached but workbook
    #                 is missing/incomplete — Bug 1 fix)
    def _seed_status(dataset_label: str, seed: int) -> tuple[bool, bool]:
        """Return (needs_llm, needs_work) for one seed.

        needs_work is True  whenever the final workbook is absent or incomplete,
                            regardless of cache state.
        needs_llm  is True  only when LLM calls are still required; False when
                            all debate + judge caches are full (only write needed).
        """
        run_dir = args.out_dir / "runs" / f"{dataset_label}_seed_{seed}"
        wb = run_dir / f"baseline_v2_{dataset_label}_s{seed}.xlsx"

        if args.overwrite:
            return True, True                       # --overwrite: redo everything

        if wb.exists() and workbook_is_complete(wb):
            return False, False                     # completely done

        # Workbook missing or incomplete → needs_work is True
        # Now determine whether LLM is needed or only the write step
        cp  = checkpoint_path(run_dir)
        jcp = judge_cache_path(run_dir)
        limit = args.objective_limit or (
            MMLU_PRO_LIMIT if dataset_label == "mmlu-pro" else GPQA_LIMIT
        )

        if not cp.exists():
            return True, True                       # no debates at all

        _, _, done_nos = load_checkpoint(run_dir)
        if len(done_nos) < limit:
            return True, True                       # debates still incomplete

        # All debates cached.  Now check judging.
        judging_skipped = args.skip_judging or args.q_source == "confidence"
        if judging_skipped:
            return False, True                      # no LLM needed; just write

        if not jcp.exists():
            return True, True                       # no judge cache at all

        _, done_j = load_judge_cache(run_dir)
        expected_calls = limit * args.rounds * len(QWEN_AGENTS)
        if len(done_j) < expected_calls:
            return True, True                       # judging still incomplete

        # All LLM work cached; only the workbook write is pending
        return False, True

    combined_path = args.out_dir / "baseline_v2_combined.xlsx"
    # Build status dict once — avoids repeated file I/O.
    seed_status: dict[tuple, tuple[bool, bool]] = {
        (ds, s): _seed_status(ds, s)
        for ds in run_datasets
        for s in args.seeds
    }
    # needs_llm  → must load model
    # needs_work → must call run_one_seed (may or may not need model)
    seeds_need_llm  = [(ds, s) for (ds, s), (nl, nw) in seed_status.items() if nl]
    seeds_need_work = [(ds, s) for (ds, s), (nl, nw) in seed_status.items() if nw]

    combined_needs_work = (
        args.overwrite
        or not combined_path.exists()
        or not workbook_is_complete(combined_path)
    )

    if not seeds_need_work and not combined_needs_work:
        print("\nAll workbooks are already complete. Nothing to do.", flush=True)
        print(f"Combined: {combined_path}", flush=True)
        print("Use --overwrite to force a full re-run.", flush=True)
        return

    if seeds_need_work and not seeds_need_llm:
        print("\nAll LLM work is cached; only workbook writes are pending.",
              flush=True)
    elif not seeds_need_work and combined_needs_work:
        print("\nAll per-seed workbooks complete; writing combined workbook only.",
              flush=True)
    else:
        print(f"\n{len(seeds_need_llm)} seed(s) need LLM work, "
              f"{len(seeds_need_work)} need any work.", flush=True)

    # Load model only when at least one seed still needs LLM calls.
    llm = None
    if seeds_need_llm:
        print("\nLoading Qwen model...", flush=True)
        llm = LocalQwenPipeline(
            model_id=args.model_id,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            require_gpu=args.require_gpu,
        )
        print("Model loaded.\n", flush=True)

    run_specs: list[tuple[Path, str, int]] = []
    run_id = 0

    for dataset_label in run_datasets:
        limit = args.objective_limit or (MMLU_PRO_LIMIT if dataset_label == "mmlu-pro" else GPQA_LIMIT)

        # Load questions only for this dataset (questions are same across seeds).
        # Load lazily — skip if all seeds for this dataset are already complete.
        dataset_seeds_need_work = [s for ds, s in seeds_need_work if ds == dataset_label]
        questions = None

        if dataset_seeds_need_work:
            if dataset_label == "mmlu-pro":
                from qwen_methodology_code import load_mmlu_pro_questions
                print(f"Loading MMLU-Pro (limit={limit})...", flush=True)
                questions = load_mmlu_pro_questions(
                    limit, args.mmlu_pro_dataset, args.mmlu_pro_split,
                    config=None, categories=None,
                )
            else:
                questions = load_gpqa_questions(limit, args.gpqa_dataset, args.gpqa_subset)

            if not questions:
                print(f"WARNING: No questions for {dataset_label}, skipping.",
                      flush=True)
                continue

        for seed in args.seeds:
            run_id += 1
            run_dir = args.out_dir / "runs" / f"{dataset_label}_seed_{seed}"

            _needs_llm, _needs_work = seed_status.get((dataset_label, seed), (False, False))
            if not _needs_work:
                # Workbook complete — add to run_specs for combine step only.
                wb = run_dir / f"baseline_v2_{dataset_label}_s{seed}.xlsx"
                print(f"[{dataset_label} s={seed}] Complete — skipping.",
                      flush=True)
                run_specs.append((wb, dataset_label, seed))
                continue

            path = run_one_seed(
                llm=llm, questions=questions, seed=seed,
                dataset_label=dataset_label, run_dir=run_dir,
                run_id=run_id, args=args,
            )
            run_specs.append((path, dataset_label, seed))

    # Write combined workbook only when needed.
    if len(run_specs) > 1 and combined_needs_work:
        combine_all(combined_path, run_specs, args)
    elif len(run_specs) > 1 and not combined_needs_work:
        print(f"\nCombined workbook already exists: {combined_path}", flush=True)
    elif len(run_specs) == 1:
        print(f"\nSingle run output: {run_specs[0][0]}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("BASELINE V2 COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
