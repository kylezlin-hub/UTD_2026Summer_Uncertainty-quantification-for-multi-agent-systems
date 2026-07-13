"""Utility functions for minority prediction."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def load_workbook_sheets(workbook_path: Path) -> Dict[str, pd.DataFrame]:
    """Load all relevant sheets from a debate workbook.

    Args:
        workbook_path: Path to Excel workbook

    Returns:
        Dictionary with sheet names as keys and DataFrames as values
    """
    required_sheets = ['Debate_Traces', 'Diagnostic_Scores', 'Reasoning_Quality']
    sheets = {}

    try:
        for sheet_name in required_sheets:
            sheets[sheet_name] = pd.read_excel(workbook_path, sheet_name=sheet_name)
    except ValueError as e:
        raise ValueError(f"Missing required sheet in {workbook_path}: {e}")

    return sheets


def get_agent_names(debates: pd.DataFrame) -> List[str]:
    """Extract agent names from debate columns.

    Args:
        debates: Debate traces DataFrame

    Returns:
        List of agent names (e.g., ['Agent1', 'Agent2', 'Agent3'])
    """
    import re
    pattern = re.compile(r'R\d+ (Agent\d+) Answer')

    agent_names = set()
    for col in debates.columns:
        match = pattern.match(col)
        if match:
            agent_names.add(match.group(1))

    return sorted(list(agent_names))


def get_answer_at_round(debate_row: pd.Series, agent: str, round_num: int) -> str:
    """Get agent's answer at specific round.

    Args:
        debate_row: Single row from Debate_Traces
        agent: Agent name (e.g., 'Agent1')
        round_num: Round number (1-5)

    Returns:
        Answer string, or None if not available
    """
    col_name = f'R{round_num} {agent} Answer'
    return debate_row.get(col_name, None)


def get_confidence_at_round(debate_row: pd.Series, agent: str, round_num: int) -> float:
    """Get agent's confidence at specific round.

    Args:
        debate_row: Single row from Debate_Traces
        agent: Agent name (e.g., 'Agent1')
        round_num: Round number (1-5)

    Returns:
        Confidence value (0-1), or None if not available
    """
    col_name = f'R{round_num} {agent} Conf'
    conf = debate_row.get(col_name, None)
    if conf is None or pd.isna(conf):
        return None
    return float(conf)


def get_quality_at_round(judgments: pd.DataFrame, row_idx: int, agent: str,
                        round_num: int) -> float:
    """Get LLM-judged explanation quality for agent at round.

    Args:
        judgments: Reasoning_Quality DataFrame
        row_idx: Question index
        agent: Agent name
        round_num: Round number

    Returns:
        Explanation quality score (0-1), or None if not available
    """
    mask = (
        (judgments['row_index'] == row_idx) &
        (judgments['round'] == round_num) &
        (judgments['agent'] == agent)
    )

    matches = judgments[mask]
    if len(matches) == 0:
        return None

    return matches.iloc[0].get('explanation_good', None)


def compute_support_distribution(answers: List[str]) -> Dict[str, int]:
    """Compute support counts for each answer.

    Args:
        answers: List of agent answers

    Returns:
        Dictionary mapping answer -> count
    """
    from collections import Counter
    return dict(Counter([a for a in answers if a is not None and not pd.isna(a)]))


def identify_minorities(support_dist: Dict[str, int]) -> List[str]:
    """Identify minority answers (not most supported).

    Args:
        support_dist: Dictionary mapping answer -> count

    Returns:
        List of minority answers
    """
    if not support_dist:
        return []

    max_support = max(support_dist.values())
    return [answer for answer, count in support_dist.items() if 0 < count < max_support]


def was_majority_in_previous_round(debate_row: pd.Series, agents: List[str],
                                   answer: str, round_num: int) -> bool:
    """Check if answer was majority in previous round.

    Args:
        debate_row: Single row from Debate_Traces
        agents: List of agent names
        answer: Answer to check
        round_num: Current round number

    Returns:
        True if answer was majority in round (round_num - 1)
    """
    if round_num <= 1:
        return False

    prev_answers = [get_answer_at_round(debate_row, agent, round_num - 1)
                    for agent in agents]
    prev_dist = compute_support_distribution(prev_answers)

    if not prev_dist:
        return False

    max_support = max(prev_dist.values())
    return prev_dist.get(answer, 0) == max_support


def count_defections(debate_row: pd.Series, agents: List[str],
                    answer: str, round_num: int) -> int:
    """Count agents who switched away from this answer.

    Args:
        debate_row: Single row from Debate_Traces
        agents: List of agent names
        answer: Answer to track
        round_num: Current round number

    Returns:
        Number of agents who held this answer before but not now
    """
    if round_num <= 1:
        return 0

    defections = 0
    for agent in agents:
        prev_answer = get_answer_at_round(debate_row, agent, round_num - 1)
        curr_answer = get_answer_at_round(debate_row, agent, round_num)

        if prev_answer == answer and curr_answer != answer:
            defections += 1

    return defections


def normalize_confidence(conf_value: float) -> float:
    """Normalize confidence to 0-1 range.

    Handles both 0-1 and 0-100 scales.

    Args:
        conf_value: Raw confidence value

    Returns:
        Normalized confidence in [0, 1]
    """
    if conf_value is None or pd.isna(conf_value):
        return None

    if conf_value > 1.0:
        return conf_value / 100.0
    return conf_value


def safe_mean(values: List[float]) -> float:
    """Compute mean, handling None/NaN values.

    Args:
        values: List of numeric values

    Returns:
        Mean of non-null values, or 0 if all null
    """
    valid = [v for v in values if v is not None and not pd.isna(v)]
    return np.mean(valid) if valid else 0.0


def safe_std(values: List[float]) -> float:
    """Compute standard deviation, handling None/NaN values.

    Args:
        values: List of numeric values

    Returns:
        Std of non-null values, or 0 if insufficient data
    """
    valid = [v for v in values if v is not None and not pd.isna(v)]
    return np.std(valid) if len(valid) > 1 else 0.0
