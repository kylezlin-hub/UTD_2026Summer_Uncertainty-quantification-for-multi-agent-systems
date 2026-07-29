from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


EPS = 1e-9
#DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MODEL="Qwen/Qwen2.5-14B-Instruct"
DEFAULT_QWEN_TEMPERATURE = 0.7
OUT_DIR = Path("diagnostic_metric_results")
EXCLUDED_AGENT_NAMES = {"Supervisor", "Moderator"}
REQUIRED_QWEN_REPORT_SHEETS = {
    "Debate_Traces",
    "Run_Metadata",
    "Reasoning_Quality",
    "Diagnostic_Scores",
}

QWEN_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
#OLLAMA_QWEN_MODEL_ID = "qwen2.5:7b"
QWEN_OUT_DIR = Path("qwen_methodology_results")
QWEN_AGENTS = ["Agent1", "Agent2", "Agent3"]
QWEN_ROUNDS = [1, 2, 3, 4, 5]
DEFAULT_QWEN_ROUNDS = 5
OBJECTIVE_LABELS = tuple("ABCDEFGHIJ")

@dataclass(frozen=True)
class AgentColumn:
    round_no: int
    agent: str
    field: str
    column: str


JUDGE_SCHEMA = {
    "name": "round_context_explanation_quality",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "explanation_good": {"type": "number", "minimum": 0, "maximum": 1},
            "uses_past_round_reasoning": {"type": "number", "minimum": 0, "maximum": 1},
            "justifies_current_stance": {"type": "number", "minimum": 0, "maximum": 1},
            "independent_reasoning": {"type": "number", "minimum": 0, "maximum": 1},
            "brief_rationale": {"type": "string"},
        },
        "required": [
            "explanation_good",
            "uses_past_round_reasoning",
            "justifies_current_stance",
            "independent_reasoning",
            "brief_rationale",
        ],
    },
}


SYSTEM_PROMPT = """You are judging one agent explanation inside a multi-agent debate.

You are NOT judging whether the final answer is correct.
You are NOT asked to solve the question.

Judge whether the agent's current explanation is good as deliberative reasoning.
The paper protocol gives you only the current justification text. Do not infer
correctness from the answer, other agents, previous rounds, or final outcomes.

Score from 0 to 1:
- explanation_good: overall quality of the current explanation as a justification.
- uses_past_round_reasoning: whether the text itself explicitly references prior reasoning.
- justifies_current_stance: whether it explains why the current stance/answer is warranted.
- independent_reasoning: whether it reasons rather than merely copying or deferring.

Do not reward consensus alone.
Do not reward confidence alone.
Do not reward fluent but generic text.
High scores require specific, checkable reasons in the current justification.

Return JSON only."""


def discover_agent_columns(columns: list[str]) -> list[AgentColumn]:
    """Find debate answer, confidence, and response columns for real agents."""
    found: list[AgentColumn] = []
    pattern = re.compile(r"^R(\d+)\s+(.+)\s+(Answer|Conf|Response)$")
    for column in columns:
        match = pattern.match(column)
        if not match:
            continue
        agent = match.group(2)
        if agent in EXCLUDED_AGENT_NAMES:
            continue
        found.append(
            AgentColumn(
                round_no=int(match.group(1)),
                agent=agent,
                field=match.group(3),
                column=column,
            )
        )
    return found


def ordered_agents_and_rounds(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    """Return agent names and debate rounds in workbook column order."""
    cols = discover_agent_columns(list(df.columns))
    agents = sorted(
        {col.agent for col in cols},
        key=lambda name: list(df.columns).index(f"R1 {name} Answer")
        if f"R1 {name} Answer" in df.columns
        else name,
    )
    rounds = sorted({col.round_no for col in cols})
    return agents, rounds


def clean_text(value: object) -> str:
    """Convert a spreadsheet value into a stripped string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_answer(value: object) -> str | None:
    """Normalize an answer cell to a compact label or cleaned text."""
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"\b([A-J])\b", text.upper())
    if match:
        return match.group(1)
    if len(text) == 1 and text.upper().isalpha():
        return text.upper()
    return re.sub(r"\s+", " ", text)


def parse_confidence(value: object) -> float:
    """Parse confidence values as floats on the 0-to-1 scale."""
    text = clean_text(value)
    if not text:
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    conf = float(match.group(1))
    if conf > 1.0:
        conf /= 100.0
    return float(np.clip(conf, 0.0, 1.0))


def correct_to_int(value: object) -> int | None:
    """Convert correctness labels into 1, 0, or None."""
    text = clean_text(value).upper()
    if text in {"YES", "Y", "TRUE", "1", "CORRECT"}:
        return 1
    if text in {"NO", "N", "FALSE", "0", "INCORRECT"}:
        return 0
    return None


def first_existing(row: pd.Series, names: list[str]) -> str:
    """Return the first non-empty value from a list of candidate row fields."""
    for name in names:
        if name in row.index and not pd.isna(row.get(name)):
            return str(row.get(name))
    return ""


def extract_likert_stance(value: object) -> float:
    """Map stances to {-2,-1,0,1,2}; also accepts workbook 1-5 Likert."""
    text = clean_text(value).lower()
    if not text:
        return np.nan
    numeric = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if numeric:
        val = float(numeric.group(0))
        if val in {-2.0, -1.0, 0.0, 1.0, 2.0}:
            return val
        if val in {1.0, 2.0, 3.0, 4.0, 5.0}:
            return val - 3.0
    labels = {
        "strongly disagree": -2.0,
        "strongly agree": 2.0,
        "disagree": -1.0,
        "neutral": 0.0,
        "agree": 1.0,
    }
    for label, stance in labels.items():
        if label in text:
            return stance
    return np.nan


def call_openai_chat(prompt: str, model: str, temperature: float = 0.0) -> dict:
    """Call the OpenAI chat API to score one reasoning explanation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_schema", "json_schema": JUDGE_SCHEMA},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
    return json.loads(raw["choices"][0]["message"]["content"])


def build_turn_prompt(
    row: pd.Series,
    source_file: str,
    round_no: int,
    agent: str,
    agents: list[str],
) -> str:
    """Build the JSON payload used to judge one agent turn."""
    payload = {
        "source_file": source_file,
        "question_no": row.get("Question #"),
        "round": round_no,
        "agent": agent,
        "current_justification_only": clean_text(row.get(f"R{round_no} {agent} Response")),
    }
    return json.dumps(payload, ensure_ascii=False)


def judge_workbook(
    path: Path,
    out_dir: Path,
    model: str,
    refresh: bool,
    sleep: float,
) -> pd.DataFrame:
    """Judge every agent explanation in an existing debate workbook."""
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    agents, rounds = ordered_agents_and_rounds(df)
    rows: list[dict[str, object]] = []
    for row_index, row in df.iterrows():
        for round_no in rounds:
            for agent in agents:
                answer = clean_text(row.get(f"R{round_no} {agent} Answer"))
                response = clean_text(row.get(f"R{round_no} {agent} Response"))
                if not answer and not response:
                    continue
                prompt = build_turn_prompt(row, path.name, round_no, agent, agents)
                cache_key = f"{path.stem}_row{row_index}_r{round_no}_{agent}".replace("/", "_")
                cache_path = out_dir / f"{cache_key}.judge.json"
                if cache_path.exists() and not refresh:
                    judged = json.loads(cache_path.read_text(encoding="utf-8"))
                else:
                    print(f"Judging {path.name} row {row_index} round {round_no} {agent}")
                    judged = call_openai_chat(prompt, model)
                    cache_path.write_text(
                        json.dumps(judged, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    time.sleep(sleep)
                rows.append(
                    {
                        "source_file": path.name,
                        "sheet_name": sheet,
                        "row_index": int(row_index),
                        "question_no": row.get("Question #"),
                        "round": int(round_no),
                        "agent": agent,
                        "answer": normalize_answer(answer),
                        "raw_answer": answer,
                        "raw_confidence": parse_confidence(row.get(f"R{round_no} {agent} Conf")),
                        **judged,
                    }
                )
    return pd.DataFrame(rows)


def empty_judgments() -> pd.DataFrame:
    """Create an empty judgments frame with the expected core columns."""
    return pd.DataFrame(
        columns=[
            "source_file",
            "row_index",
            "round",
            "agent",
            "explanation_good",
        ]
    )


def build_matrices(
    row: pd.Series,
    judgments: pd.DataFrame,
    agents: list[str],
    rounds: list[int],
    stance_mode: str,
    q_source: str,
) -> tuple[np.ndarray, np.ndarray, list[list[str | None]]]:
    """Convert one debate row into stance, quality, and answer matrices."""
    x = np.full((len(rounds), len(agents)), np.nan)
    q = np.zeros((len(rounds), len(agents)), dtype=float)
    categorical_answers: list[list[str | None]] = []
    last_x = {agent: np.nan for agent in agents}
    last_q = {agent: 0.0 for agent in agents}
    last_answer = {agent: None for agent in agents}
    answer_vocab: dict[str, float] = {}

    for ti, round_no in enumerate(rounds):
        round_answers: list[str | None] = []
        for ai, agent in enumerate(agents):
            answer_value = row.get(f"R{round_no} {agent} Answer")
            normalized = normalize_answer(answer_value)
            if normalized is None:
                normalized = last_answer[agent]
            last_answer[agent] = normalized
            round_answers.append(normalized)

            if stance_mode == "likert":
                stance = extract_likert_stance(answer_value)
            else:
                if normalized is None:
                    stance = np.nan
                else:
                    if normalized not in answer_vocab:
                        answer_vocab[normalized] = float(len(answer_vocab))
                    stance = answer_vocab[normalized]
            if np.isnan(stance):
                stance = last_x[agent]
            x[ti, ai] = stance
            last_x[agent] = stance

            if q_source == "llm":
                judged = judgments[
                    (judgments["round"] == round_no) & (judgments["agent"] == agent)
                ]
                quality = (
                    float(judged.iloc[0]["explanation_good"])
                    if len(judged) and not pd.isna(judged.iloc[0]["explanation_good"])
                    else np.nan
                )
            else:
                quality = parse_confidence(row.get(f"R{round_no} {agent} Conf"))
            if np.isnan(quality):
                quality = last_q[agent]
            q[ti, ai] = float(np.clip(quality, 0.0, 1.0))
            last_q[agent] = q[ti, ai]
        categorical_answers.append(round_answers)
    return x, q, categorical_answers


def mean_except(values: np.ndarray, idx: int) -> float:
    """Compute the mean of all values except one index, ignoring NaNs."""
    others = np.delete(values, idx)
    valid = others[~np.isnan(others)]
    return float(valid.mean()) if len(valid) else np.nan


def paper_likert_metrics(x: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Equations (3)-(13), assuming x contains ordered stances in {-2,-1,0,1,2}."""
    t_count, a_count = x.shape
    if t_count < 1 or a_count < 1:
        return empty_scores()

    # Eq. (3): mean stance per round; used directly and via leave-one-agent-out means.
    mu = np.nanmean(x, axis=1)

    # Eq. (4): average reasoning-quality-weighted absolute stance change.
    engagement_terms = []
    for t in range(1, t_count):
        for a in range(a_count):
            if not np.isnan(x[t, a]) and not np.isnan(x[t - 1, a]):
                engagement_terms.append(q[t, a] * abs(x[t, a] - x[t - 1, a]))
    engagement = float(np.sum(engagement_terms) / (a_count * max(t_count - 1, 1)))

    # Responsiveness: we aggregate weighted Delta over updates.
    responsiveness_terms = []
    for t in range(1, t_count):
        for a in range(a_count):
            prev_other_mean = mean_except(x[t - 1], a)
            if np.isnan(prev_other_mean) or np.isnan(x[t, a]) or np.isnan(x[t - 1, a]):
                continue
            moved_closer = abs(x[t, a] - prev_other_mean) < abs(x[t - 1, a] - prev_other_mean)
            responsiveness_terms.append(q[t, a] * float(moved_closer))
    responsiveness = float(np.mean(responsiveness_terms)) if responsiveness_terms else np.nan

    # Eq. (5): how often each source agent attracts other agents toward its prior stance.
    influence = np.zeros(a_count, dtype=float)
    for a in range(a_count):
        total = 0.0
        for b in range(a_count):
            if b == a:
                continue
            for t in range(1, t_count):
                if any(np.isnan(v) for v in [x[t, b], x[t - 1, b], x[t - 1, a]]):
                    continue
                closer = abs(x[t, b] - x[t - 1, a]) < abs(x[t - 1, b] - x[t - 1, a])
                total += q[t, b] * float(closer)
        influence[a] = total

    # Eqs. (6)-(7): normalized entropy concentration of influence.
    total_influence = float(influence.sum())
    if total_influence <= EPS or a_count <= 1:
        influence_asymmetry = 0.0
    else:
        p = influence / total_influence
        h = -sum(pi * log(pi) for pi in p if pi > 0)
        influence_asymmetry = float(1.0 - h / log(a_count))

    # Eqs. (8)-(11): convergence collapse and volatility from stance dispersion.
    sigma = np.nanstd(x, axis=1)
    c_total = float(sigma[0] - sigma[-1])
    step_convergence = [float(sigma[t - 1] - sigma[t]) for t in range(1, t_count)]
    c_max = max(step_convergence) if step_convergence else 0.0
    reversals = 0
    for t in range(2, t_count):
        if (sigma[t - 2] - sigma[t - 1]) * (sigma[t - 1] - sigma[t]) < 0:
            reversals += 1
    volatility = reversals
    volatility_component = 1.0 if t_count <= 2 else 1.0 - volatility / (t_count - 2)
    balance = (1.0 - c_max / (c_total + EPS)) * volatility_component
    balance = float(np.clip(balance, 0.0, 1.0))

    # Eq. (12): Nash-style stability under a simple peer-agreement utility.
    stability = stability_score(x[-1])

    # Eq. (13): average utility change from initial to final stances.
    welfare = float(np.nanmean([utility(x[-1], a) - utility(x[0], a) for a in range(a_count)]))

    return {
        "mean_final_stance_mu_T": float(mu[-1]),
        "engagement": engagement,
        "responsiveness": responsiveness,
        "influence_asymmetry": influence_asymmetry,
        "balance": balance,
        "stability": stability,
        "group_welfare": welfare,
    }


def corrected_balance_from_dispersion(dispersion: list[float]) -> float:
    """A semantic correction for Eq. (11): penalize collapse and volatility separately.

    The printed equation can give high scores to divergent or volatile debates after
    clipping. This version treats one-round collapse and oscillation as separate
    penalties, matching the prose around pluralism and conditional convergence.
    """
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


def corrected_likert_metrics(x: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Compute Likert metrics with the corrected balance definition."""
    scores = paper_likert_metrics(x, q)
    scores["balance"] = corrected_balance_from_dispersion(list(np.nanstd(x, axis=1)))
    return scores


def entropy(values: list[str | None]) -> float:
    """Compute normalized entropy for categorical answer labels."""
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return -float(sum(pi * log(pi) for pi in p if pi > 0)) / log(len(vals))


def categorical_metrics(answers: list[list[str | None]], q: np.ndarray) -> dict[str, float]:
    """Practical adaptation for answer letters where there is no ordinal stance geometry."""
    t_count = len(answers)
    a_count = len(answers[0]) if answers else 0
    if t_count < 1 or a_count < 1:
        return empty_scores()

    engagement_terms = []
    responsiveness_terms = []
    influence = np.zeros(a_count, dtype=float)
    for t in range(1, t_count):
        for a in range(a_count):
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
        h = -sum(pi * log(pi) for pi in p if pi > 0)
        influence_asymmetry = float(1.0 - h / log(a_count))

    dispersion = [entropy(round_answers) for round_answers in answers]
    c_total = dispersion[0] - dispersion[-1]
    step_convergence = [dispersion[t - 1] - dispersion[t] for t in range(1, t_count)]
    c_max = max(step_convergence) if step_convergence else 0.0
    reversals = sum(
        (dispersion[t - 2] - dispersion[t - 1]) * (dispersion[t - 1] - dispersion[t]) < 0
        for t in range(2, t_count)
    )
    volatility_component = 1.0 if t_count <= 2 else 1.0 - reversals / (t_count - 2)
    balance = (1.0 - c_max / (c_total + EPS)) * volatility_component
    balance = float(np.clip(balance, 0.0, 1.0))

    initial_utility = categorical_peer_utility(answers[0])
    final_utility = categorical_peer_utility(answers[-1])
    welfare = float(np.nanmean(final_utility - initial_utility))

    return {
        "mean_final_stance_mu_T": np.nan,
        "engagement": float(np.mean(engagement_terms)) if engagement_terms else np.nan,
        "responsiveness": float(np.mean(responsiveness_terms)) if responsiveness_terms else np.nan,
        "influence_asymmetry": influence_asymmetry,
        "balance": balance,
        "stability": categorical_stability(answers[-1]),
        "group_welfare": welfare,
    }


def corrected_categorical_metrics(answers: list[list[str | None]], q: np.ndarray) -> dict[str, float]:
    """Compute categorical metrics with the corrected balance definition."""
    scores = categorical_metrics(answers, q)
    scores["balance"] = corrected_balance_from_dispersion([entropy(round_answers) for round_answers in answers])
    return scores


def utility(profile: np.ndarray, idx: int) -> float:
    """Simple utility u_a(x_a, x_-a): negative average distance from peers."""
    own = profile[idx]
    if np.isnan(own):
        return np.nan
    others = np.delete(profile, idx)
    others = others[~np.isnan(others)]
    if len(others) == 0:
        return 0.0
    return -float(np.mean(np.abs(own - others)))


def stability_score(final_profile: np.ndarray) -> float:
    """Estimate whether final Likert stances are stable against unilateral changes."""
    candidate_stances = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    stable = 0
    comparable = 0
    for a in range(len(final_profile)):
        if np.isnan(final_profile[a]):
            continue
        current = utility(final_profile, a)
        best = current
        for candidate in candidate_stances:
            deviated = final_profile.copy()
            deviated[a] = candidate
            best = max(best, utility(deviated, a))
        comparable += 1
        stable += int(current >= best - EPS)
    return stable / comparable if comparable else np.nan


def categorical_peer_utility(round_answers: list[str | None]) -> np.ndarray:
    """Measure each agent's peer agreement utility for categorical answers."""
    utilities = np.full(len(round_answers), np.nan)
    for a, answer in enumerate(round_answers):
        if answer is None:
            continue
        others = [ans for j, ans in enumerate(round_answers) if j != a and ans is not None]
        utilities[a] = sum(other == answer for other in others) / len(others) if others else 0.0
    return utilities


def categorical_stability(final_answers: list[str | None]) -> float:
    """Estimate whether final categorical answers are stable against switching."""
    stable = 0
    comparable = 0
    for a, answer in enumerate(final_answers):
        if answer is None:
            continue
        others = [ans for j, ans in enumerate(final_answers) if j != a and ans is not None]
        if not others:
            continue
        support = Counter(others)
        current_support = support.get(answer, 0)
        best_support = max(support.values())
        comparable += 1
        stable += int(current_support >= best_support)
    return stable / comparable if comparable else np.nan


def empty_scores() -> dict[str, float]:
    """Return a metrics dictionary populated with NaN placeholders."""
    return {
        "mean_final_stance_mu_T": np.nan,
        "engagement": np.nan,
        "responsiveness": np.nan,
        "influence_asymmetry": np.nan,
        "balance": np.nan,
        "stability": np.nan,
        "group_welfare": np.nan,
    }


def score_workbook(
    path: Path,
    judgments: pd.DataFrame,
    stance_mode: str,
    q_source: str,
    metric_version: str,
    limit: int | None,
) -> pd.DataFrame:
    """Score every debate row in an existing workbook using selected diagnostics."""
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    if limit is not None:
        df = df.head(limit)
    agents, rounds = ordered_agents_and_rounds(df)
    rows: list[dict[str, object]] = []
    for row_index, row in df.iterrows():
        row_judgments = judgments[
            (judgments["source_file"] == path.name) & (judgments["row_index"] == int(row_index))
        ]
        x, q, answers = build_matrices(row, row_judgments, agents, rounds, stance_mode, q_source)
        if stance_mode == "likert":
            scores = (
                paper_likert_metrics(x, q)
                if metric_version == "paper"
                else corrected_likert_metrics(x, q)
            )
        else:
            scores = (
                categorical_metrics(answers, q)
                if metric_version == "paper"
                else corrected_categorical_metrics(answers, q)
            )
        rows.append(
            {
                "source_file": path.name,
                "sheet_name": sheet,
                "row_index": int(row_index),
                "question_no": row.get("Question #"),
                "accuracy": correct_to_int(row.get("Correct?")),
                "rounds_to_consensus": row.get("Rounds to Consensus"),
                "stance_mode": stance_mode,
                "q_source": q_source,
                "metric_version": metric_version,
                **scores,
            }
        )
    return pd.DataFrame(rows)


def run_workbook_diagnostics(args: argparse.Namespace) -> tuple[Path | None, Path]:
    """Run the original workbook-scoring workflow from this unified script."""
    out_dir = args.out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(args.input)
    df = pd.read_excel(args.input, sheet_name=xls.sheet_names[0])
    if args.limit is not None:
        limited_path = out_dir / f"{args.input.stem}.limited_{args.limit}.xlsx"
        df.head(args.limit).to_excel(limited_path, index=False)
        judge_input = limited_path
    else:
        judge_input = args.input

    if args.q_source == "llm":
        judgments = judge_workbook(
            judge_input,
            out_dir,
            args.model,
            refresh=args.refresh,
            sleep=args.sleep,
        )
        judgments["source_file"] = args.input.name
        judgments_out = out_dir / f"{args.input.stem}.llm_judgments.csv"
        judgments.to_csv(judgments_out, index=False)
    else:
        judgments = empty_judgments()
        judgments_out = None

    scores = score_workbook(
        args.input,
        judgments,
        stance_mode=args.stance_mode,
        q_source=args.q_source,
        metric_version=args.metric_version,
        limit=args.limit,
    )
    scores_out = (
        out_dir
        / f"{args.input.stem}.{args.stance_mode}.{args.q_source}.{args.metric_version}.scores.csv"
    )
    scores.to_csv(scores_out, index=False)
    return judgments_out, scores_out


def main() -> None:
    """Run the legacy workbook diagnostic CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Workbook path.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stance-mode", choices=["likert", "categorical"], default="likert")
    parser.add_argument("--q-source", choices=["llm", "confidence"], default="llm")
    parser.add_argument("--metric-version", choices=["corrected", "paper"], default="paper")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    judgments_out, scores_out = run_workbook_diagnostics(args)
    if judgments_out is not None:
        print(f"Wrote {judgments_out}")
    print(f"Wrote {scores_out}")






@dataclass(frozen=True)
class DebateQuestion:
    dataset_type: str
    question_no: str
    question: str
    correct_answer: str = ""
    answer_labels: tuple[str, ...] = ("A", "B", "C", "D")
    category: str = ""


def coerce_options(value: object) -> list[str]:
    """Coerce MMLU-Pro option values into a clean list of strings."""
    if isinstance(value, (list, tuple)):
        return [clean_text(option) for option in value if clean_text(option)]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            import ast

            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
    if isinstance(parsed, (list, tuple)):
        return [clean_text(option) for option in parsed if clean_text(option)]
    return [part.strip() for part in re.split(r"\s*\|\s*|\n", text) if part.strip()]


def mmlu_pro_correct_label(row: dict, options: list[str], labels: tuple[str, ...]) -> str:
    """Resolve the MMLU-Pro correct answer into an option label."""
    for key in ("answer_index", "answer_idx", "correct_index"):
        raw = row.get(key)
        if raw is None or pd.isna(raw):
            continue
        try:
            return labels[int(raw)]
        except (TypeError, ValueError, IndexError):
            pass
    raw_answer = clean_text(row.get("answer"))
    if raw_answer:
        upper = raw_answer.upper()
        if upper in labels:
            return upper
        for label, option in zip(labels, options):
            if raw_answer.casefold() == option.casefold():
                return label
    raise ValueError("Could not determine MMLU-Pro correct label for row.")


def load_mmlu_pro_questions(
    limit: int,
    dataset_name: str,
    split: str,
    config: str | None,
    categories: list[str] | None,
) -> list[DebateQuestion]:
    """Load and normalize objective questions from the MMLU-Pro dataset."""
    if limit <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "MMLU-Pro loading requires the `datasets` package. Install it with "
            "`python3.9 -m pip install datasets` and rerun."
        ) from exc

    load_args = [dataset_name]
    if config:
        load_args.append(config)
    rows = list(load_dataset(*load_args, split=split))
    if categories:
        allowed = {category.casefold() for category in categories}
        rows = [row for row in rows if clean_text(row.get("category")).casefold() in allowed]

    questions: list[DebateQuestion] = []
    for index, row in enumerate(rows):
        if len(questions) >= limit:
            break
        question_text = clean_text(row.get("question"))
        options = coerce_options(row.get("options"))
        if not question_text or len(options) < 2:
            continue
        if len(options) > len(OBJECTIVE_LABELS):
            raise ValueError(
                f"MMLU-Pro row {index} has {len(options)} options; "
                f"supported maximum is {len(OBJECTIVE_LABELS)}."
            )
        labels = OBJECTIVE_LABELS[: len(options)]
        correct_label = mmlu_pro_correct_label(row, options, labels)
        formatted_options = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
        question_no = clean_text(row.get("question_id")) or clean_text(row.get("id")) or f"mmlu-pro-{index + 1:05d}"
        questions.append(
            DebateQuestion(
                "objective",
                str(question_no),
                f"{question_text}\n{formatted_options}",
                correct_label,
                labels,
                clean_text(row.get("category")),
            )
        )
    if len(questions) < limit:
        print(f"Loaded {len(questions)} MMLU-Pro questions after filtering; requested {limit}.")
    return questions


OBJECTIVE_PERSONAS = {
    "Agent1": "You are a careful analyst who starts from first principles and checks each option.",
    "Agent2": "You are a skeptical reviewer who challenges weak assumptions and looks for counterexamples.",
    "Agent3": "You are a concise domain generalist who weighs the other agents' arguments before updating.",
}


SUBJECTIVE_PERSONAS = {
    "Agent1": "You represent a public agency focused on feasibility, legality, and long-term infrastructure planning.",
    "Agent2": "You represent a community NGO focused on affordability, displacement risk, and resident voice.",
    "Agent3": "You represent private industry focused on financeability, permitting risk, and construction capacity.",
}


class LocalQwenPipeline:
    """Thin wrapper around transformers.pipeline for Qwen2.5 chat generation."""

    def __init__(
        self,
        model_id: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        device_map: str,
        torch_dtype: str,
        require_gpu: bool = False,
    ) -> None:
        """Load a transformers text-generation pipeline for local Qwen inference."""
        try:
            import torch
            from transformers import pipeline, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "Qwen mode requires `transformers` plus its runtime dependencies. "
                "Install them and make sure the Qwen2.5-14B-Instruct weights are available."
            ) from exc
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU was required, but PyTorch cannot see CUDA. "
                "Submit the batch job with a GPU allocation and rerun."
            )

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.set_seed = set_seed

        kwargs: dict[str, object] = {
            "model": model_id,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if device_map == "none":
            kwargs["device"] = 0 if torch.cuda.is_available() else -1
        else:
            kwargs["device_map"] = device_map

        self.generator = pipeline("text-generation", **kwargs)
        self.tokenizer = self.generator.tokenizer

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format chat messages into the prompt string expected by the model."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt = "\n\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        )
        return prompt + "\n\nASSISTANT:"

    def _generation_kwargs(
        self,
        max_new_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, object]:
        """Build shared generation arguments for one or more prompts."""
        requested_temperature = self.temperature if temperature is None else temperature
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "return_full_text": False,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if requested_temperature <= 0.0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": requested_temperature,
                    "top_p": self.top_p,
                }
            )
        return generation_kwargs

    @staticmethod
    def _extract_generated_text(output: object) -> str:
        """Normalize transformers pipeline output into plain generated text."""
        if isinstance(output, list):
            if not output:
                return ""
            output = output[0]
        if isinstance(output, dict):
            generated = output.get("generated_text", "")
        else:
            generated = output
        if isinstance(generated, list):
            return str(generated[-1].get("content", "")).strip()
        return str(generated).strip()

    def complete(
        self,
        messages: list[dict[str, str]],
        seed: int,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate one chat completion from the local transformers pipeline."""
        return self.complete_many(
            [messages],
            seeds=[seed],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0]

    def complete_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        seeds: list[int],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Generate chat completions for a small batch of prompts."""
        if not messages_batch:
            return []
        self.set_seed(seeds[0] if seeds else 0)
        prompts = [self._messages_to_prompt(messages) for messages in messages_batch]
        outputs = self.generator(
            prompts,
            **self._generation_kwargs(max_new_tokens, temperature),
        )
        return [self._extract_generated_text(output) for output in outputs]


class OllamaQwenPipeline:
    """Local Qwen2.5 backend for Ollama-installed models."""

    def __init__(
        self,
        model_id: str,
        host: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> None:
        """Store Ollama connection and generation settings for Qwen calls."""
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def complete(
        self,
        messages: list[dict[str, str]],
        seed: int,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate one chat completion through the Ollama chat API."""
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "top_p": self.top_p,
                "num_predict": max_new_tokens or self.max_new_tokens,
                "seed": seed,
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not call Ollama at {self.host}. Make sure Ollama is running "
                f"and the model `{self.model_id}` is installed."
            ) from exc
        return str(data.get("message", {}).get("content", "")).strip()

    def complete_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        seeds: list[int],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Generate multiple Ollama completions sequentially for API compatibility."""
        return [
            self.complete(messages, seed, max_new_tokens, temperature)
            for messages, seed in zip(messages_batch, seeds)
        ]


def qwen_initial_messages(question: DebateQuestion, agent: str) -> list[dict[str, str]]:
    """Build the first-round prompt messages for one Qwen debate agent."""
    if question.dataset_type == "objective":
        answer_slot = "/".join(question.answer_labels)
        user = (
            f"{question.question}\n\n"
            "Give your initial answer independently. Use exactly this format:\n"
            f"Answer: <{answer_slot}>\n"
            "Confidence: <number from 0 to 1>\n"
            "Explanation: <brief justification>"
        )
        return [{"role": "system", "content": OBJECTIVE_PERSONAS[agent]}, {"role": "user", "content": user}]

    user = (
        f"Policy statement: {question.question}\n\n"
        "Give your initial stance independently on this Likert scale: "
        "1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree.\n"
        "Use exactly this format:\n"
        "Answer: <1/2/3/4/5>\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <brief justification>"
    )
    return [{"role": "system", "content": SUBJECTIVE_PERSONAS[agent]}, {"role": "user", "content": user}]


def qwen_update_messages(
    question: DebateQuestion,
    agent: str,
    previous_round: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Build later-round prompt messages using previous peer responses."""
    personas = OBJECTIVE_PERSONAS if question.dataset_type == "objective" else SUBJECTIVE_PERSONAS
    answer_slot = (
        "<" + "/".join(question.answer_labels) + ">"
        if question.dataset_type == "objective"
        else "<1/2/3/4/5>"
    )
    other_lines = []
    for other_agent, turn in previous_round.items():
        if other_agent == agent:
            continue
        other_lines.append(
            f"{other_agent}: Answer={turn['answer']}; Explanation={turn['response']}"
        )
    own = previous_round[agent]
    user = (
        f"Question or policy statement:\n{question.question}\n\n"
        f"Your previous answer was {own['answer']} with explanation: {own['response']}\n\n"
        "Other agents' previous-round messages:\n"
        + "\n".join(other_lines)
        + "\n\nConsider the other agents' reasoning. Update only if warranted by the arguments. "
        "Use exactly this format:\n"
        f"Answer: {answer_slot}\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <brief justification grounded in the debate>"
    )
    return [{"role": "system", "content": personas[agent]}, {"role": "user", "content": user}]


def parse_qwen_turn(
    raw: str,
    dataset_type: str,
    answer_labels: tuple[str, ...] = ("A", "B", "C", "D"),
    strict: bool = True,
) -> dict[str, object]:
    """Parse a Qwen agent response into answer, confidence, and explanation fields."""
    if dataset_type == "objective":
        escaped = "".join(re.escape(label) for label in answer_labels)
        answer_regex = rf"\b([{escaped}])\b"
    else:
        answer_regex = r"\b([1-5])\b"
    answer = ""
    answer_patterns = [
        r"^\s*(?:[*_`]+)?[\(<\[\{`\"'*_]*" + answer_regex + r"[\)>\]\}`\"'*_]*\s*$",
        r"(?:^|\n)\s*(?:[*_`]+)?(?:final\s+answer|answer|choice)(?:[*_`]+)?\s*[:=\-]\s*(?:[*_`]+)?\s*(?:option|choice)?\s*[\(<\[\{`\"'*_]*"
        + answer_regex,
        r"\b(?:final\s+answer|answer|choice)(?:[*_`\"']+)?\s*[:=\-]\s*(?:[*_`\"']+)?\s*(?:option|choice)?\s*[\(<\[\{`\"'*_]*"
        + answer_regex,
        r"\b(?:final\s+answer|answer|choice)\s+(?:is|would\s+be|should\s+be)\s*(?:option|choice)?\s*[\(<\[\{`\"'*_]*"
        + answer_regex,
        r"\b(?:the\s+)?(?:correct|best)\s+(?:answer|choice|option)\s*(?:is|would\s+be|should\s+be|:)\s*(?:option|choice)?\s*[\(<\[\{`\"'*_]*"
        + answer_regex,
        r"\b(?:my\s+)?(?:selection|pick)\s*(?:is|would\s+be|should\s+be|:)\s*(?:option|choice)?\s*[\(<\[\{`\"'*_]*"
        + answer_regex,
        r"\b(?:i\s+)?(?:choose|select|pick)\s*(?:option|choice)?\s*[\(<\[\{`\"'*_]*"
        + answer_regex,
        r"\b(?:option|choice)\s*[\(<\[\{`\"'*_]*" + answer_regex,
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            if raw[match.end(1):].lstrip().startswith("/"):
                continue
            answer = match.group(1).upper()
            break
    if not answer and not strict:
        fallback = re.search(answer_regex, raw, flags=re.IGNORECASE)
        if fallback:
            answer = fallback.group(1).upper()

    confidence = ""
    conf_match = re.search(r"confidence\s*[:=\-]\s*([01](?:\.\d+)?)", raw, flags=re.IGNORECASE)
    if conf_match:
        confidence = str(float(conf_match.group(1)))

    explanation = raw.strip()
    exp_match = re.search(r"explanation\s*[:=\-]\s*(.+)", raw, flags=re.IGNORECASE | re.DOTALL)
    if exp_match:
        explanation = exp_match.group(1).strip()
    parse_errors = []
    if not answer:
        parse_errors.append("missing_or_invalid_answer")
    if not explanation:
        parse_errors.append("missing_explanation")
    return {
        "answer": answer,
        "confidence": confidence or "0.5",
        "response": re.sub(r"\s+", " ", explanation),
        "parse_failed": bool(parse_errors),
        "parse_error": ",".join(parse_errors),
    }


def qwen_reprompt_messages(
    messages: list[dict[str, str]],
    question: DebateQuestion,
) -> list[dict[str, str]]:
    """Add a repair prompt when an agent response cannot be parsed."""
    allowed = (
        ", ".join(question.answer_labels)
        if question.dataset_type == "objective"
        else "1, 2, 3, 4, 5"
    )
    repair = (
        "Your previous response could not be parsed. Reply again using exactly "
        "three lines and no extra text. Replace X with exactly one allowed answer.\n"
        f"Allowed answers for X: {allowed}\n"
        "Answer: X\n"
        "Confidence: 0.00\n"
        "Explanation: brief justification"
    )
    return [*messages, {"role": "user", "content": repair}]


def complete_qwen_turn_with_retry(
    llm: LocalQwenPipeline | OllamaQwenPipeline,
    messages: list[dict[str, str]],
    question: DebateQuestion,
    seed: int,
    max_attempts: int = 3,
) -> dict[str, object]:
    """Generate and parse one agent turn, retrying after parse failure."""
    first_error = ""
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        attempt_messages = messages if attempt == 0 else qwen_reprompt_messages(messages, question)
        raw = llm.complete(attempt_messages, seed=seed + attempt)
        parsed = parse_qwen_turn(raw, question.dataset_type, question.answer_labels, strict=True)
        if attempt == 0 and parsed["parse_failed"]:
            first_error = str(parsed["parse_error"])
        parsed["re_prompted"] = attempt > 0
        if first_error:
            parsed["first_parse_error"] = first_error
        if not parsed["parse_failed"] or attempt == attempts - 1:
            return parsed
    raise RuntimeError("unreachable Qwen turn retry state")


def complete_qwen_turns_with_retry(
    llm: LocalQwenPipeline | OllamaQwenPipeline,
    messages_by_agent: dict[str, list[dict[str, str]]],
    question: DebateQuestion,
    seeds_by_agent: dict[str, int],
    max_attempts: int = 3,
) -> dict[str, dict[str, object]]:
    """Generate and parse a batch of agent turns, retrying failed parses."""
    agents = list(messages_by_agent)
    turns: dict[str, dict[str, object]] = {}
    first_errors: dict[str, str] = {}

    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        pending_agents = [agent for agent in agents if agent not in turns]
        if not pending_agents:
            break
        messages_batch = [
            messages_by_agent[agent]
            if attempt == 0
            else qwen_reprompt_messages(messages_by_agent[agent], question)
            for agent in pending_agents
        ]
        seeds = [seeds_by_agent[agent] + attempt for agent in pending_agents]
        raws = llm.complete_many(messages_batch, seeds=seeds)

        for agent, raw in zip(pending_agents, raws):
            parsed = parse_qwen_turn(raw, question.dataset_type, question.answer_labels, strict=True)
            if attempt == 0 and parsed["parse_failed"]:
                first_errors[agent] = str(parsed["parse_error"])
            parsed["re_prompted"] = attempt > 0
            if agent in first_errors:
                parsed["first_parse_error"] = first_errors[agent]
            if not parsed["parse_failed"] or attempt == attempts - 1:
                turns[agent] = parsed

    return turns


def run_qwen_debates(
    llm: LocalQwenPipeline,
    questions: list[DebateQuestion],
    round_count: int,
    seed: int,
    sleep: float,
    model_id: str = QWEN_MODEL_ID,
) -> pd.DataFrame:
    """Run multi-agent Qwen debates and return workbook-ready debate rows."""
    import random

    rows: list[dict[str, object]] = []
    rng = random.Random(seed)
    for q_index, question in enumerate(questions):
        print(
            f"Generating question {q_index + 1}/{len(questions)}: {question.question_no}",
            flush=True,
        )
        agent_order = QWEN_AGENTS[:]
        rng.shuffle(agent_order)
        all_rounds: list[dict[str, dict[str, str]]] = []
        discard_reason = ""

        for round_no in range(1, round_count + 1):
            print(
                f"Generating question {q_index+1} round {round_no}/{round_count}",
                flush=True,
            )
            messages_by_agent = {
                agent: (
                    qwen_initial_messages(question, agent)
                    if round_no == 1
                    else qwen_update_messages(question, agent, all_rounds[-1])
                )
                for agent in agent_order
            }
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
            if sleep:
                time.sleep(sleep)
            if discard_reason:
                print(f"Discarding debate after unresolved parse failure: {discard_reason}", flush=True)
                break
            all_rounds.append(round_turns)
        if discard_reason:
            continue
        final_answer, final_source = select_final_answer(llm, question, all_rounds, seed + q_index * 1000 + 999)
        rows.append(
            make_debate_row(
                question,
                all_rounds,
                model_id,
                final_answer=final_answer,
                final_answer_source=final_source,
            )
        )

    df = pd.DataFrame(rows)
    df["Correct?"] = df.apply(final_answer_correctness, axis=1)
    return df


def make_debate_row(
    question: DebateQuestion,
    rounds: list[dict[str, dict[str, str]]],
    model_name: str,
    final_answer: str | None = None,
    final_answer_source: str | None = None,
) -> dict[str, object]:
    """Flatten one completed debate into the report's row schema."""
    final_answers = [rounds[-1][agent]["answer"] for agent in QWEN_AGENTS]
    if final_answer is None:
        final_answer = majority_answer(final_answers)
        final_answer_source = "agent_consensus" if len(set(final_answers)) == 1 else "majority_vote_no_moderator"
    row: dict[str, object] = {
        "Question #": question.question_no,
        "Dataset Type": question.dataset_type,
        "Dataset Category": question.category,
        "Question": question.question,
        "Correct Answer": question.correct_answer,
        "Final Answer": final_answer,
        "Final Answer Source": final_answer_source or "",
        "Model": model_name,
        "Rounds to Consensus": first_consensus_round_for_answer(rounds, final_answer),
    }
    for round_no, turns in enumerate(rounds, start=1):
        for agent in QWEN_AGENTS:
            turn = turns[agent]
            row[f"R{round_no} {agent} Answer"] = turn["answer"]
            row[f"R{round_no} {agent} Conf"] = turn["confidence"]
            row[f"R{round_no} {agent} Response"] = turn["response"]
    return row


def majority_answer(answers: list[str]) -> str:
    """Return the most common non-empty answer label."""
    valid = [answer for answer in answers if answer]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]


def select_final_answer(
    llm: LocalQwenPipeline | OllamaQwenPipeline,
    question: DebateQuestion,
    rounds: list[dict[str, dict[str, str]]],
    seed: int,
) -> tuple[str, str]:
    """Choose the final answer from consensus, moderator, or majority fallback."""
    final_answers = [rounds[-1][agent]["answer"] for agent in QWEN_AGENTS]
    valid = [answer for answer in final_answers if answer]
    if valid and len(set(valid)) == 1:
        return valid[0], "agent_consensus"
    raw = llm.complete(qwen_moderator_messages(question, rounds), seed=seed)
    parsed = parse_qwen_turn(raw, question.dataset_type, question.answer_labels, strict=False)["answer"]
    if answer_is_valid(parsed, question):
        return parsed, "moderator"
    return majority_answer(final_answers), "majority_vote_no_moderator"


def qwen_moderator_messages(
    question: DebateQuestion,
    rounds: list[dict[str, dict[str, str]]],
) -> list[dict[str, str]]:
    """Build the moderator prompt used when agents do not fully agree."""
    answer_slot = (
        "<" + "/".join(question.answer_labels) + ">"
        if question.dataset_type == "objective"
        else "<1/2/3/4/5>"
    )
    lines = []
    for round_no, turns in enumerate(rounds, start=1):
        for agent in QWEN_AGENTS:
            turn = turns[agent]
            lines.append(
                f"Round {round_no} {agent}: Answer={turn['answer']}; Explanation={turn['response']}"
            )
    user = (
        f"Question or policy statement:\n{question.question}\n\n"
        "Debate transcript:\n"
        + "\n".join(lines)
        + "\n\nChoose the best final answer from the allowed labels. "
        "Use exactly this format:\n"
        f"Answer: {answer_slot}\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <brief justification>"
    )
    return [
        {"role": "system", "content": "You are a neutral moderator selecting the final debate answer."},
        {"role": "user", "content": user},
    ]


def answer_is_valid(answer: str, question: DebateQuestion) -> bool:
    """Check whether a parsed answer is allowed for the question type."""
    text = clean_text(answer).upper()
    if question.dataset_type == "objective":
        return text in question.answer_labels
    return text in {"1", "2", "3", "4", "5"}


def first_consensus_round(rounds: list[dict[str, dict[str, str]]]) -> int | str:
    """Find the first round where all agents gave the same answer."""
    for round_no, turns in enumerate(rounds, start=1):
        answers = [turns[agent]["answer"] for agent in QWEN_AGENTS]
        if all(answers) and len(set(answers)) == 1:
            return round_no
    return ""


def first_consensus_round_for_answer(
    rounds: list[dict[str, dict[str, str]]],
    final_answer: str | None,
) -> int | str:
    """Find the first consensus round that matches the selected final answer."""
    final_answer = clean_text(final_answer)
    if not final_answer:
        return ""
    for round_no, turns in enumerate(rounds, start=1):
        answers = [turns[agent]["answer"] for agent in QWEN_AGENTS]
        if all(answers) and len(set(answers)) == 1 and answers[0] == final_answer:
            return round_no
    return ""


def final_answer_correctness(row: pd.Series) -> str:
    """Mark objective final answers as correct or incorrect."""
    if row.get("Dataset Type") != "objective":
        return ""
    return "Yes" if str(row.get("Final Answer")) == str(row.get("Correct Answer")) else "No"


def judge_debates_with_qwen(
    llm: LocalQwenPipeline,
    debates: pd.DataFrame,
    source_file: str,
    seed: int,
    judge_max_new_tokens: int,
    judge_batch_size: int,
    sleep: float,
) -> pd.DataFrame:
    """Use Qwen as a judge for every generated agent explanation."""
    agents, rounds = ordered_agents_and_rounds(debates)
    rows: list[dict[str, object]] = []
    for row_index, row in debates.iterrows():
        question_index = int(row_index) + 1
        tasks: list[tuple[int, str, list[dict[str, str]], int]] = []
        for round_no in rounds:
            print(
                f"Judging question {question_index}/{len(debates)} round {round_no}",
                flush=True,
            )
            for agent in agents:
                tasks.append(
                    (
                        round_no,
                        agent,
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": build_turn_prompt(row, source_file, round_no, agent, agents),
                            },
                        ],
                        seed + int(row_index) * 1000 + round_no * 100 + agents.index(agent),
                    )
                )
        batch_size = max(1, judge_batch_size)
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            messages_batch = [task[2] for task in batch]
            seeds = [task[3] for task in batch]
            raws = llm.complete_many(
                messages_batch,
                seeds=seeds,
                max_new_tokens=judge_max_new_tokens,
            )
            for (round_no, agent, _, _), raw in zip(batch, raws):
                judged = parse_qwen_judgment(raw)
                rows.append(
                    {
                        "source_file": source_file,
                        "sheet_name": "Debate_Traces",
                        "row_index": int(row_index),
                        "question_no": row.get("Question #"),
                        "round": int(round_no),
                        "agent": agent,
                        "answer": normalize_answer(row.get(f"R{round_no} {agent} Answer")),
                        "raw_answer": row.get(f"R{round_no} {agent} Answer"),
                        "raw_confidence": parse_confidence(row.get(f"R{round_no} {agent} Conf")),
                        **judged,
                    }
                )
            if sleep:
                time.sleep(sleep)
    return pd.DataFrame(rows)


def parse_qwen_judgment(raw: str) -> dict[str, object]:
    """Parse Qwen judge JSON into bounded reasoning-quality scores."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    parse_failed = False
    try:
        data = json.loads(match.group(0) if match else raw)
    except json.JSONDecodeError:
        data = {}
        parse_failed = True
    defaults: dict[str, object] = {
        "explanation_good": 0.5,
        "uses_past_round_reasoning": 0.5,
        "justifies_current_stance": 0.5,
        "independent_reasoning": 0.5,
        "brief_rationale": "Could not parse judge JSON; assigned a neutral score.",
    }
    defaults.update(data)
    defaults["judge_parse_failed"] = parse_failed
    if not parse_failed and defaults["brief_rationale"] == "Could not parse judge JSON; assigned a neutral score.":
        defaults["brief_rationale"] = "Parsed Qwen judge JSON."
    for key in [
        "explanation_good",
        "uses_past_round_reasoning",
        "justifies_current_stance",
        "independent_reasoning",
    ]:
        try:
            defaults[key] = float(np.clip(float(defaults.get(key, 0.5)), 0.0, 1.0))
        except (TypeError, ValueError):
            # Judge sometimes emits non-numeric scores (e.g. "N/A"); treat as neutral.
            defaults[key] = 0.5
    return defaults


def score_mixed_debates(
    debates: pd.DataFrame,
    judgments: pd.DataFrame,
    source_file: str,
    q_source: str,
    metric_version: str,
) -> pd.DataFrame:
    """Score generated objective and subjective debates with matching metrics."""
    agents, rounds = ordered_agents_and_rounds(debates)
    rows: list[dict[str, object]] = []
    for row_index, row in debates.iterrows():
        stance_mode = "categorical" if row["Dataset Type"] == "objective" else "likert"
        if stance_mode == "categorical":
            effective_metric_version = (
                "categorical_adaptation"
                if metric_version == "paper"
                else "corrected_categorical_adaptation"
            )
            metric_definition = (
                "MCQ answer-letter adaptation: change/support/entropy over categorical answers; "
                "not the paper's Likert stance equations."
            )
        else:
            effective_metric_version = metric_version
            metric_definition = "Paper Likert stance equations over {-2,-1,0,1,2}."
        row_judgments = judgments[
            (judgments["source_file"] == source_file)
            & (judgments["row_index"] == int(row_index))
        ]
        x, q, answers = build_matrices(row, row_judgments, agents, rounds, stance_mode, q_source)
        if stance_mode == "categorical":
            scores = (
                categorical_metrics(answers, q)
                if metric_version == "paper"
                else corrected_categorical_metrics(answers, q)
            )
        else:
            scores = (
                paper_likert_metrics(x, q)
                if metric_version == "paper"
                else corrected_likert_metrics(x, q)
            )
        rows.append(
            {
                "source_file": source_file,
                "row_index": int(row_index),
                "question_no": row.get("Question #"),
                "dataset_type": row.get("Dataset Type"),
                "stance_mode": stance_mode,
                "q_source": q_source,
                "requested_metric_version": metric_version,
                "metric_version": effective_metric_version,
                "metric_definition": metric_definition,
                "accuracy": correct_to_int(row.get("Correct?")),
                "rounds_to_consensus": row.get("Rounds to Consensus"),
                "final_answer": row.get("Final Answer"),
                "final_answer_source": row.get("Final Answer Source"),
                "correct_answer": row.get("Correct Answer"),
                **scores,
            }
        )
    scored = pd.DataFrame(rows)
    scored["influence_asymmetry_inv"] = 1.0 - scored["influence_asymmetry"]
    scored["avg_process_metrics"] = scored[
        ["engagement", "responsiveness", "influence_asymmetry_inv", "balance"]
    ].mean(axis=1, skipna=True)
    return scored


def aggregate_report(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize generated diagnostic scores by dataset type and stance mode."""
    if scores.empty:
        return pd.DataFrame()
    metrics = [
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
    grouped = scores.groupby(["dataset_type", "stance_mode"], dropna=False)[metrics].agg(["count", "mean", "std"])
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    return grouped.reset_index()


def objective_correlation_report(scores: pd.DataFrame) -> pd.DataFrame:
    """Report objective-question metric correlations against accuracy."""
    if scores.empty:
        return pd.DataFrame()
    objective = scores[scores["dataset_type"] == "objective"].copy()
    signals = [
        "rounds_to_consensus",
        "engagement",
        "responsiveness",
        "influence_asymmetry_inv",
        "balance",
        "stability",
        "group_welfare",
        "avg_process_metrics",
    ]
    rows = []
    for signal in signals:
        paired = objective[[signal, "accuracy"]].copy()
        paired[signal] = pd.to_numeric(paired[signal], errors="coerce")
        paired["accuracy"] = pd.to_numeric(paired["accuracy"], errors="coerce")
        paired = paired.dropna()
        rho = np.nan
        if len(paired) >= 3 and paired[signal].nunique() > 1 and paired["accuracy"].nunique() > 1:
            rho = float(np.corrcoef(paired[signal].rank(), paired["accuracy"].rank())[0, 1])
        rows.append({"signal_metric": signal, "spearman_rho_with_accuracy": rho, "n": len(paired)})
    return pd.DataFrame(rows)


def write_qwen_excel_report(
    path: Path,
    mode: str,
    args: argparse.Namespace,
    debates: pd.DataFrame,
    judgments: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    """Write debate traces, metadata, judgments, and scores to an Excel report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(
        [
            {"field": "run_mode", "value": mode},
            {"field": "backend", "value": getattr(args, "backend", "")},
            {"field": "model_id", "value": args.model_id},
            {"field": "dataset_source", "value": getattr(args, "dataset_source", "builtin")},
            {"field": "mmlu_pro_dataset", "value": getattr(args, "mmlu_pro_dataset", "")},
            {"field": "mmlu_pro_split", "value": getattr(args, "mmlu_pro_split", "")},
            {"field": "paper_protocol", "value": "3 agents, 5 rounds, stance trajectories, q_ta reasoning scores, paper diagnostics"},
            {"field": "objective_metric_definition", "value": "Objective MCQ rows use an explicit categorical adaptation, not the paper's Likert stance equations."},
            {"field": "reasoning_quality_context", "value": "Evaluator receives only the current agent justification text."},
            {"field": "parse_failure_policy", "value": "Agent turns are re-prompted up to two times; debates with unresolved answer/explanation parse failures are discarded."},
            {"field": "final_answer_fallback", "value": "Uses moderator fallback when agents do not reach final-round consensus."},
            {"field": "judge_parse_failures", "value": int(judgments.get("judge_parse_failed", pd.Series(dtype=bool)).sum()) if not judgments.empty else 0},
            {"field": "temperature", "value": args.temperature},
            {"field": "top_p", "value": args.top_p},
            {"field": "seed", "value": args.seed},
            {"field": "note", "value": "Generated from MMLU-Pro and judged with the local Qwen pipeline."},
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        debates.to_excel(writer, sheet_name="Debate_Traces", index=False)
        metadata.to_excel(writer, sheet_name="Run_Metadata", index=False)
        judgments.to_excel(writer, sheet_name="Reasoning_Quality", index=False)
        scores.to_excel(writer, sheet_name="Diagnostic_Scores", index=False)
        aggregate_report(scores).to_excel(writer, sheet_name="Aggregate_Summary", index=False)
        objective_correlation_report(scores).to_excel(writer, sheet_name="Objective_Correlations", index=False)


def qwen_report_is_complete(path: Path) -> bool:
    """Check whether a Qwen report workbook has all required sheets."""
    try:
        sheet_names = set(pd.ExcelFile(path).sheet_names)
    except Exception:
        return False
    return REQUIRED_QWEN_REPORT_SHEETS.issubset(sheet_names)


def qwen_methodology_main() -> None:
    """Run the unified Qwen methodology CLI workflow."""
    parser = argparse.ArgumentParser(
        description="Run the paper-style multi-agent debate diagnostics with Qwen2.5-14B-Instruct."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Existing debate workbook to score. When set, runs the original diagnostic workbook workflow.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model for --input scoring when --q-source llm.",
    )
    parser.add_argument("--model-id", default=QWEN_MODEL_ID)
    parser.add_argument("--llm-provider", choices=["qwen"], default="qwen")
    parser.add_argument(
        "--backend",
        choices=["auto", "transformers", "ollama"],
        default="auto",
        help="Generation backend. Auto uses transformers if installed, otherwise Ollama.",
    )
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to diagnostic_metric_results for --input, otherwise qwen_methodology_results.",
    )
    parser.add_argument("--dataset-source", choices=["mmlu-pro"], default="mmlu-pro")
    parser.add_argument("--mmlu-pro-dataset", default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--mmlu-pro-config", default=None)
    parser.add_argument("--mmlu-pro-split", default="test")
    parser.add_argument("--mmlu-pro-category", action="append", default=None)
    parser.add_argument("--objective-limit", type=int, default=0)
    parser.add_argument("--subjective-limit", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=DEFAULT_QWEN_ROUNDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=DEFAULT_QWEN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--judge-max-new-tokens", type=int, default=220)
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=15,
        help="Number of judge prompts to send to the transformers pipeline at once.",
    )
    parser.add_argument("--stance-mode", choices=["likert", "categorical"], default="likert")
    parser.add_argument("--q-source", choices=["llm", "confidence"], default="llm")
    parser.add_argument("--metric-version", choices=["corrected", "paper"], default="paper")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--skip-judging", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--legacy-score-cli",
        action="store_true",
        help="Use the original copied scoring CLI instead of the Qwen reproduction workflow.",
    )
    args = parser.parse_args()

    if args.input is not None or args.legacy_score_cli:
        if args.input is None:
            raise SystemExit("--legacy-score-cli requires --input when used from qwen_methodology_code.py")
        if args.sleep is None:
            args.sleep = 0.2
        judgments_out, scores_out = run_workbook_diagnostics(args)
        if judgments_out is not None:
            print(f"Wrote {judgments_out}")
        print(f"Wrote {scores_out}")
        return

    if args.sleep is None:
        args.sleep = 0.0
    out_dir = args.out_dir or QWEN_OUT_DIR

    output_name = (
        "qwen_mmlu_pro_debate_traces.xlsx"
        if args.dataset_source == "mmlu-pro"
        else "qwen2_5_methodology_comparable_results.xlsx"
    )
    output_path = out_dir / output_name
    source_file = output_path.name
    if args.resume and output_path.exists() and qwen_report_is_complete(output_path):
        print(f"Reusing completed workbook at {output_path}")
        return

    backend = args.backend
    if backend == "auto":
        try:
            import transformers  # noqa: F401

            backend = "transformers"
        except ImportError:
            backend = "ollama"

    model_id = args.model_id

    if backend == "ollama":
        llm = OllamaQwenPipeline(
            model_id=model_id,
            host=args.ollama_host,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        llm = LocalQwenPipeline(
            model_id=model_id,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            require_gpu=args.require_gpu,
        )
    questions = load_mmlu_pro_questions(
        args.objective_limit,
        args.mmlu_pro_dataset,
        args.mmlu_pro_split,
        args.mmlu_pro_config,
        args.mmlu_pro_category,
    )
    if not questions:
        raise RuntimeError("No MMLU-Pro questions were selected. Check split, category, and objective limit.")
    debates = run_qwen_debates(
        llm,
        questions,
        args.rounds,
        args.seed,
        args.sleep,
        model_id=model_id,
    )
    if args.skip_judging or args.q_source == "confidence":
        judgments = pd.DataFrame()
    else:
        judgments = judge_debates_with_qwen(
            llm,
            debates,
            source_file=source_file,
            seed=args.seed + 100_000,
            judge_max_new_tokens=args.judge_max_new_tokens,
            judge_batch_size=args.judge_batch_size,
            sleep=args.sleep,
        )
    mode = f"qwen_{backend}"
    args.backend = backend
    args.model_id = model_id

    if judgments.empty and args.q_source == "llm":
        scores = pd.DataFrame()
    else:
        if judgments.empty:
            judgments = empty_judgments()
        judgments["source_file"] = source_file
        scores = score_mixed_debates(
            debates,
            judgments,
            source_file,
            q_source=args.q_source,
            metric_version=args.metric_version,
        )
    write_qwen_excel_report(output_path, mode, args, debates, judgments, scores)
    print(f"Wrote {output_path}")
    if scores.empty:
        print("Skipped diagnostic scoring; wrote debate traces only.")
    else:
        print(aggregate_report(scores).to_string(index=False))


if __name__ == "__main__":
    qwen_methodology_main()
