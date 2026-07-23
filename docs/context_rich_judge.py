"""Context-rich judge that matches authors_code.py (NOT the paper's §A.3.6 prose).

WHY
---
The paper text (§A.3.6) says judging is context-BLIND and uses ReCEval. But the
authors' actual scoring code (authors_code.py) does neither:
  - SYSTEM_PROMPT (authors_code.py:57-79) tells the judge it is given the current
    turn, the SAME agent's previous round, AND all OTHER agents' previous rounds.
  - build_turn_prompt (authors_code.py:219-261) actually feeds that context.
  - It uses the LLM-judge schema (explanation_good/...), not ReCEval.

Your qwen pipeline is context-BLIND (build_turn_prompt:240-255 sends only
`current_justification_only`, and SYSTEM_PROMPT:72 says "only the current
justification"). So it diverges from the authors' implementation.

This module makes the qwen judge match authors_code.py exactly:
  1. Swap in the authors' context-rich SYSTEM_PROMPT.
  2. Replace build_turn_prompt so the judge receives current + own-prev + peers'-prev.
The JUDGE_SCHEMA and parse_qwen_judgment already match authors_code.py, so no
other changes are needed. Both globals are resolved at call time by
judge_debates_with_qwen / judge_workbook, so monkeypatching takes effect.

CAVEAT: we empirically found the quality weight q does NOT move the
process-metric/accuracy correlations (explanation_good vs confidence were
identical). This aligns the JUDGING PROTOCOL with the authors' code; it is a
faithfulness fix, not expected to change the correlation results (that gap is
the stance representation, letters vs Likert).

USAGE (before running the pipeline entrypoint):
    import context_rich_judge
    context_rich_judge.install_context_rich_judge()
    qmc.qwen_methodology_main()
"""

from __future__ import annotations

import json

import qwen_methodology_code as qmc


# Verbatim from authors_code.py:57-79 (context-rich judge instructions).
CONTEXT_RICH_SYSTEM_PROMPT = """You are judging one agent explanation inside a multi-agent debate.

You are NOT judging whether the final answer is correct.
You are NOT asked to solve the question.

Judge whether the agent's current explanation is good as deliberative reasoning,
given:
- the current round answer/explanation,
- the same agent's previous-round answer/explanation when available,
- other agents' previous-round answers/explanations when available.

Score from 0 to 1:
- explanation_good: overall quality of the current explanation as a debate update.
- uses_past_round_reasoning: whether it meaningfully engages the previous round.
- justifies_current_stance: whether it explains why the current stance/answer is warranted.
- independent_reasoning: whether it reasons rather than merely copying or deferring.

Do not reward consensus alone.
Do not reward confidence alone.
Do not reward fluent but generic text.
High scores require specific, checkable reasons tied to the current and past round.

Return JSON only."""


def _question_text(row) -> str:
    """First non-empty of Question/Scenario/Prompt (mirrors authors' first_existing)."""
    for col in ("Question", "Scenario", "Prompt"):
        val = qmc.clean_text(row.get(col))
        if val:
            return val
    return ""


def context_rich_build_turn_prompt(row, source_file, round_no, agent, agents) -> str:
    """Context-rich judge input, mirroring authors_code.py:219-261.

    Feeds the current turn, the same agent's previous round, and all other
    agents' previous-round answers+explanations.
    """
    previous_round = round_no - 1
    current = {
        "answer": qmc.clean_text(row.get(f"R{round_no} {agent} Answer")),
        "explanation": qmc.clean_text(row.get(f"R{round_no} {agent} Response")),
    }
    agent_previous = (
        {
            "answer": qmc.clean_text(row.get(f"R{previous_round} {agent} Answer")),
            "explanation": qmc.clean_text(row.get(f"R{previous_round} {agent} Response")),
        }
        if previous_round >= 1
        else {}
    )
    other_previous = []
    if previous_round >= 1:
        for other in agents:
            if other == agent:
                continue
            other_previous.append(
                {
                    "agent": other,
                    "answer": qmc.clean_text(row.get(f"R{previous_round} {other} Answer")),
                    "explanation": qmc.clean_text(row.get(f"R{previous_round} {other} Response")),
                }
            )
    payload = {
        "source_file": source_file,
        "question_no": row.get("Question #"),
        "question_or_scenario": _question_text(row),
        "round": round_no,
        "agent": agent,
        "agent_previous_round": agent_previous,
        "other_agents_previous_round": other_previous,
        "agent_current_round": current,
    }
    return json.dumps(payload, ensure_ascii=False)


def install_context_rich_judge() -> None:
    """Point the qwen judge at the authors' context-rich protocol.

    Patches SYSTEM_PROMPT and build_turn_prompt (both resolved as module globals
    at call time by judge_debates_with_qwen and judge_workbook).
    """
    qmc.SYSTEM_PROMPT = CONTEXT_RICH_SYSTEM_PROMPT
    qmc.build_turn_prompt = context_rich_build_turn_prompt
    print("[context-rich-judge] Installed authors_code.py judging protocol "
          "(current + own-prev + peers'-prev context).", flush=True)


if __name__ == "__main__":
    print(__doc__)
