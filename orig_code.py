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
DEFAULT_MODEL = "gpt-4.1-mini"
OUT_DIR = Path("diagnostic_metric_results")
EXCLUDED_AGENT_NAMES = {"Supervisor", "Moderator"}


@dataclass(frozen=True)
class AgentColumn:
    round_no: int
    agent: str
    field: str
    column: str


0, "maximum": 1}, "justifies_current_stance": {"type": "number", "minimum":


SYSTEM_PROMPT = """You are judging one agent explanation inside a
multi-agent debate.

You are NOT judging whether the final answer is correct.
You are NOT asked to solve the question.

- other agents' previous-round answers/explanations when available.

Score from 0 to 1:
- explanation_good: overall quality of the current explanation as a
debate update.
- uses_past_round_reasoning: whether it meaningfully engages the previous round.
- justifies_current_stance: whether it explains why the current
stance/answer is warranted.
- independent_reasoning: whether it reasons rather than merely copying
or deferring.

Do not reward consensus alone.
Do not reward confidence alone.
Do not reward fluent but generic text.
High scores require specific, checkable reasons tied to the current
and past round.

Return JSON only."""


def discover_agent_columns(columns: list[str]) -> list[AgentColumn]:
    found: list[AgentColumn] = []
    pattern = re.compile(r"^R(\d+)\s+(.+)\s+(Answer|Conf|Response)$")
    for column in columns:
        match = pattern.match(column)
        if not match:
            continue
        agent = match.group(2)
        if agent in EXCLUDED_AGENT_NAMES:
            continue
            )
        )
    return found


def ordered_agents_and_rounds(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    cols = discover_agent_columns(list(df.columns))
        key=lambda name: list(df.columns).index(f"R1 {name} Answer")
        if f"R1 {name} Answer" in df.columns
    rounds = sorted({col.round_no for col in cols})
    return agents, rounds


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_answer(value: object) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"\b([A-H])\b", text.upper())
    if match:
        return match.group(1)
    if len(text) == 1 and text.upper().isalpha():
        return text.upper()
    return re.sub(r"\s+", " ", text)


def parse_confidence(value: object) -> float:
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
    text = clean_text(value).upper()
    if text in {"YES", "Y", "TRUE", "1", "CORRECT"}:
        return 1
    if text in {"NO", "N", "FALSE", "0", "INCORRECT"}:
        return 0
    return None


def first_existing(row: pd.Series, names: list[str]) -> str:
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
    }
    for label, stance in labels.items():
        if label in text:
            return stance
    return np.nan


def call_openai_chat(prompt: str, model: str, temperature: float = 0.0) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    }
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
    return json.loads(raw["choices"][0]["message"]["content"])


) -> str:
    previous_round = round_no - 1
        }
        if previous_round >= 1
        else {}
    )
    other_previous = []
    if previous_round >= 1:
        for other in agents:
            if other == agent:
                continue
            )
    return json.dumps(payload, ensure_ascii=False)


) -> pd.DataFrame:
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
                    time.sleep(sleep)
                        "raw_confidence":
                    }
                )
    return pd.DataFrame(rows)


def empty_judgments() -> pd.DataFrame:
    )


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
(judgments["agent"] == agent)
                ]
                    if len(judged) and not
pd.isna(judged.iloc[0]["explanation_good"])
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
    others = np.delete(values, idx)
    valid = others[~np.isnan(others)]
    return float(valid.mean()) if len(valid) else np.nan


def paper_likert_metrics(x: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Equations (3)-(13), assuming x contains ordered stances in
{-2,-1,0,1,2}."""
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

    # Responsiveness: we aggregate weighted Delta over updates.
    responsiveness_terms = []
    for t in range(1, t_count):
        for a in range(a_count):
            prev_other_mean = mean_except(x[t - 1], a)
            if np.isnan(prev_other_mean) or np.isnan(x[t, a]) or np.isnan(x[t - 1, a]):
                continue
            moved_closer = abs(x[t, a] - prev_other_mean) < abs(x[t - 1, a] - prev_other_mean)
            responsiveness_terms.append(q[t, a] * float(moved_closer))

    # Eq. (5): how often each source agent attracts other agents
toward its prior stance.
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
    step_convergence = [float(sigma[t - 1] - sigma[t]) for t in
range(1, t_count)]
    c_max = max(step_convergence) if step_convergence else 0.0
    reversals = 0
    for t in range(2, t_count):
        if (sigma[t - 2] - sigma[t - 1]) * (sigma[t - 1] - sigma[t]) < 0:
            reversals += 1
    volatility = reversals
    balance = (1.0 - c_max / (c_total + EPS)) * volatility_component
    balance = float(np.clip(balance, 0.0, 1.0))

    # Eq. (12): Nash-style stability under a simple peer-agreement utility.
    stability = stability_score(x[-1])

    # Eq. (13): average utility change from initial to final stances.
    welfare = float(np.nanmean([utility(x[-1], a) - utility(x[0], a)
for a in range(a_count)]))

    }


def corrected_balance_from_dispersion(dispersion: list[float]) -> float:

    The printed equation can give high scores to divergent or volatile
debates after
    clipping. This version treats one-round collapse and oscillation as separate
    penalties, matching the prose around pluralism and conditional convergence.
    """
    t_count = len(dispersion)
    if t_count <= 1:
        return np.nan
    steps = [dispersion[t - 1] - dispersion[t] for t in range(1, t_count)]
    positive_convergence = [max(0.0, step) for step in steps]
    total_positive_convergence = sum(positive_convergence)
    if total_positive_convergence <= EPS:
        collapse_score = 1.0 if max(abs(step) for step in steps) <= EPS else 0.0
    else:
    if t_count <= 2:
        volatility_score = 1.0
    else:
        volatility_score = 1.0 - reversals / (t_count - 2)
    return float(np.clip(collapse_score * volatility_score, 0.0, 1.0))


def corrected_likert_metrics(x: np.ndarray, q: np.ndarray) -> dict[str, float]:
    scores = paper_likert_metrics(x, q)
    return scores


def entropy(values: list[str | None]) -> float:
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return -float(sum(pi * log(pi) for pi in p if pi > 0)) / log(len(vals))


def categorical_metrics(answers: list[list[str | None]], q:
np.ndarray) -> dict[str, float]:
    """Practical adaptation for answer letters where there is no
ordinal stance geometry."""
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
            others_prev = [answers[t - 1][j] for j in range(a_count)
if j != a and answers[t - 1][j]]
            if others_prev:
                old_support = sum(other == old for other in others_prev)
                new_support = sum(other == new for other in others_prev)
                responsiveness_terms.append(q[t, a] * float(changed
and new_support > old_support))
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
    step_convergence = [dispersion[t - 1] - dispersion[t] for t in
range(1, t_count)]
    c_max = max(step_convergence) if step_convergence else 0.0
dispersion[t]) < 0
        for t in range(2, t_count)
    )
    balance = (1.0 - c_max / (c_total + EPS)) * volatility_component
    balance = float(np.clip(balance, 0.0, 1.0))

    initial_utility = categorical_peer_utility(answers[0])
    final_utility = categorical_peer_utility(answers[-1])
    welfare = float(np.nanmean(final_utility - initial_utility))

    }


def corrected_categorical_metrics(answers: list[list[str | None]], q:
np.ndarray) -> dict[str, float]:
    scores = categorical_metrics(answers, q)
round_answers in answers])
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
    utilities = np.full(len(round_answers), np.nan)
    for a, answer in enumerate(round_answers):
        if answer is None:
            continue
        others = [ans for j, ans in enumerate(round_answers) if j != a
and ans is not None]
    return utilities


def categorical_stability(final_answers: list[str | None]) -> float:
    stable = 0
    comparable = 0
    for a, answer in enumerate(final_answers):
        if answer is None:
            continue
        others = [ans for j, ans in enumerate(final_answers) if j != a
and ans is not None]
        if not others:
            continue
        support = Counter(others)
        current_support = support.get(answer, 0)
        best_support = max(support.values())
        comparable += 1
        stable += int(current_support >= best_support)
    return stable / comparable if comparable else np.nan


def empty_scores() -> dict[str, float]:
    }


    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    if limit is not None:
        df = df.head(limit)
    agents, rounds = ordered_agents_and_rounds(df)
    rows: list[dict[str, object]] = []
    for row_index, row in df.iterrows():
(judgments["row_index"] == int(row_index))
        ]
        if stance_mode == "likert":
                if metric_version == "paper"
                else corrected_likert_metrics(x, q)
            )
        else:
                if metric_version == "paper"
                else corrected_categorical_metrics(answers, q)
            )
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(args.input)
    df = pd.read_excel(args.input, sheet_name=xls.sheet_names[0])
    if args.limit is not None:
        df.head(args.limit).to_excel(limited_path, index=False)
        judge_input = limited_path
    else:
        judge_input = args.input

    if args.q_source == "llm":
        )
        judgments["source_file"] = args.input.name
        judgments_out = args.out_dir / f"{args.input.stem}.llm_judgments.csv"
        judgments.to_csv(judgments_out, index=False)
    else:
        judgments = empty_judgments()
        judgments_out = None

        / f"{args.input.stem}.{args.stance_mode}.{args.q_source}.{args.metric_version}.scores.csv"
    )
    scores.to_csv(scores_out, index=False)

    if judgments_out is not None:
        print(f"Wrote {judgments_out}")
    print(f"Wrote {scores_out}")



if __name__ == "__main__":
    main()