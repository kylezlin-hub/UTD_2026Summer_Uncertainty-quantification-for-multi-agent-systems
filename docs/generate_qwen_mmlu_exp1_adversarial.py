"""Adversarial ("tit for tat") debate variant for the Qwen MMLU-Pro Table 1 analogue.

WHY THIS EXISTS
---------------
The baseline generator (generate_qwen_mmlu_exp1.py -> qwen_methodology_code.py)
uses cooperative personas and a convergence-oriented update instruction
("Update only if warranted by the arguments"). In our runs that yields ~63%
round-1 consensus: most "debates" are immediate agreement, so the process
diagnostics have little deliberation to measure and do not track accuracy.

Liang et al. 2024 (Encouraging Divergent Thinking in LLMs through Multi-Agent
Debate, arXiv:2305.19118) assign OPPOSING stances and explicitly instruct
"It's not necessary to fully agree with each other's perspectives, as our
objective is to find the correct answer." Their central claim is that debate
helps ONLY when this adversarial "tit for tat" is maintained; without it MAD
collapses into "degeneration of thought".

This script reproduces the SAME pipeline, model, data, seeds, rounds, judge,
metrics and OUTPUT FORMAT as the baseline, changing ONLY the debate prompts:
  1. Adversarial personas (committed advocate / mandatory challenger /
     independent reasoner).
  2. A Liang-style update instruction that forbids agreeing for its own sake
     and requires each agent to defend or attack with new, specific reasons.

Everything downstream (parsing, LLM judge, diagnostic metrics, scoring,
correlation report) is byte-for-byte the baseline code, so any change in the
process-metric / accuracy relationship is attributable to the debate protocol
alone -- isolating "protocol" from "model scale (14B vs 72B)".

USAGE (mirror your baseline invocation; match --model-id to your baseline run)
------------------------------------------------------------------------------
    cd C:/Proj1/docs
    python generate_qwen_mmlu_exp1_adversarial.py --seed 7 --seed 17 --seed 42 \
        --model-id Qwen/Qwen2.5-14B-Instruct

Writes one workbook per seed to <out-dir>/adversarial_mmlu-pro_s<seed>.xlsx,
directly comparable to New/baseline_v2_mmlu-pro_s<seed>.xlsx.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import qwen_methodology_code as qmc


# ---------------------------------------------------------------------------
# Adversarial personas (replaces qmc.OBJECTIVE_PERSONAS for objective/MCQ).
# Distinct, committed roles that resist premature convergence.
# ---------------------------------------------------------------------------
ADVERSARIAL_OBJECTIVE_PERSONAS = {
    "Agent1": (
        "You are a committed advocate. You defend your current answer with "
        "specific, checkable reasoning. You concede ONLY when another agent "
        "gives an argument that decisively refutes yours -- never merely to "
        "agree or to end the debate."
    ),
    "Agent2": (
        "You are the challenger / devil's advocate. Your job is to attack the "
        "answer the group is converging on: find the strongest case for a "
        "DIFFERENT option and press it. Do not fall in line with the majority "
        "unless its argument is airtight. Manufacture the best possible "
        "counterargument every round."
    ),
    "Agent3": (
        "You are an independent reasoner. You work the problem from first "
        "principles and must NOT defer to the majority. If the others agree "
        "too quickly, treat that as a warning sign and re-examine whether the "
        "agreed answer is actually correct."
    ),
}

# Subjective (Likert) analogue: defend your stance, do not drift to the mean.
ADVERSARIAL_SUBJECTIVE_PERSONAS = {
    "Agent1": (
        "You represent a public agency (feasibility, legality, long-term "
        "planning). Argue your position forcefully; do not soften your stance "
        "just to reach agreement."
    ),
    "Agent2": (
        "You represent a community NGO (affordability, displacement, resident "
        "voice). Your role is to contest the emerging consensus and defend the "
        "under-represented view. Do not converge unless genuinely persuaded."
    ),
    "Agent3": (
        "You represent private industry (financeability, permitting, "
        "construction capacity). Reason independently and resist bandwagoning "
        "toward the group's stance."
    ),
}


def _majority_answer(previous_round: dict[str, dict[str, str]], self_agent: str) -> str | None:
    """The answer the OTHER agents are currently converging on (for the challenger)."""
    others = [
        str(turn.get("answer", "")).strip()
        for other, turn in previous_round.items()
        if other != self_agent and str(turn.get("answer", "")).strip()
    ]
    if not others:
        return None
    answer, _ = Counter(others).most_common(1)[0]
    return answer or None


def adversarial_initial_messages(
    question: "qmc.DebateQuestion", agent: str
) -> list[dict[str, str]]:
    """Round-1 independent answer, but under an adversarial persona.

    Output format is IDENTICAL to qmc.qwen_initial_messages so parsing/metrics
    are unaffected.
    """
    if question.dataset_type == "objective":
        answer_slot = "/".join(question.answer_labels)
        user = (
            f"{question.question}\n\n"
            "Give your initial answer independently. Use exactly this format:\n"
            f"Answer: <{answer_slot}>\n"
            "Confidence: <number from 0 to 1>\n"
            "Explanation: <brief justification>"
        )
        return [
            {"role": "system", "content": ADVERSARIAL_OBJECTIVE_PERSONAS[agent]},
            {"role": "user", "content": user},
        ]

    user = (
        f"Policy statement: {question.question}\n\n"
        "Give your initial stance independently on this Likert scale: "
        "1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree.\n"
        "Use exactly this format:\n"
        "Answer: <1/2/3/4/5>\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <brief justification>"
    )
    return [
        {"role": "system", "content": ADVERSARIAL_SUBJECTIVE_PERSONAS[agent]},
        {"role": "user", "content": user},
    ]


def adversarial_update_messages(
    question: "qmc.DebateQuestion",
    agent: str,
    previous_round: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Later-round update with Liang-style 'tit for tat' disagreement pressure.

    Output format is IDENTICAL to qmc.qwen_update_messages.
    """
    if question.dataset_type == "objective":
        personas = ADVERSARIAL_OBJECTIVE_PERSONAS
        answer_slot = "<" + "/".join(question.answer_labels) + ">"
    else:
        personas = ADVERSARIAL_SUBJECTIVE_PERSONAS
        answer_slot = "<1/2/3/4/5>"

    other_lines = []
    for other_agent, turn in previous_round.items():
        if other_agent == agent:
            continue
        other_lines.append(
            f"{other_agent}: Answer={turn['answer']}; Explanation={turn['response']}"
        )
    own = previous_round[agent]

    # Liang et al. 2024 core instruction: consensus is NOT the objective.
    tit_for_tat = (
        "It is NOT necessary to agree with the other agents. Our shared "
        "objective is to find the correct answer, not to reach consensus. "
        "Keep your answer and defend it with NEW, specific reasons unless "
        "another agent's argument decisively refutes yours. "
    )
    # Give the challenger an explicit target to attack.
    if question.dataset_type == "objective" and agent == "Agent2":
        target = _majority_answer(previous_round, agent)
        if target:
            tit_for_tat += (
                f"The others are leaning toward '{target}'. Make the strongest "
                f"possible case for a DIFFERENT option before considering it. "
            )
    tit_for_tat += (
        "If all agents have already agreed, do not simply confirm it -- raise "
        "the single strongest objection you can and test whether the agreed "
        "answer survives it."
    )

    user = (
        f"Question or policy statement:\n{question.question}\n\n"
        f"Your previous answer was {own['answer']} with explanation: {own['response']}\n\n"
        "Other agents' previous-round messages:\n"
        + "\n".join(other_lines)
        + "\n\n" + tit_for_tat + " "
        "Use exactly this format:\n"
        f"Answer: {answer_slot}\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <brief justification grounded in the debate>"
    )
    return [
        {"role": "system", "content": personas[agent]},
        {"role": "user", "content": user},
    ]


def install_adversarial_protocol() -> None:
    """Monkeypatch the pipeline's debate prompts. Everything else is unchanged.

    run_qwen_debates() resolves these names from module globals at call time,
    so reassigning them here takes effect for the whole run.
    """
    qmc.OBJECTIVE_PERSONAS = ADVERSARIAL_OBJECTIVE_PERSONAS
    qmc.SUBJECTIVE_PERSONAS = ADVERSARIAL_SUBJECTIVE_PERSONAS
    qmc.qwen_initial_messages = adversarial_initial_messages
    qmc.qwen_update_messages = adversarial_update_messages
    print("[adversarial] Installed Liang-style tit-for-tat debate prompts.", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the adversarial debate generator."""
    p = argparse.ArgumentParser(description="Adversarial-protocol MMLU-Pro debate generator.")
    p.add_argument("--out-dir", type=Path, default=Path("data/adversarial"))
    p.add_argument("--objective-limit", type=int, default=200)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, action="append", default=None,
                   help="Seed to run. Repeat for multiple seeds. Default: 7 17 42.")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct",
                   help="MUST match the model used for your baseline_v2 run for a fair comparison.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--judge-max-new-tokens", type=int, default=220)
    p.add_argument("--judge-batch-size", type=int, default=15)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def run_one_seed(args: argparse.Namespace, seed: int) -> None:
    """Invoke the unmodified pipeline entrypoint for one seed, adversarial prompts installed."""
    run_dir = args.out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "qwen_methodology_code.py",
        "--llm-provider", "qwen",
        "--backend", "transformers",
        "--model-id", args.model_id,
        "--dataset-source", "mmlu-pro",
        "--mmlu-pro-dataset", "TIGER-Lab/MMLU-Pro",
        "--mmlu-pro-split", "test",
        "--objective-limit", str(args.objective_limit),
        "--subjective-limit", "0",
        "--rounds", str(args.rounds),
        "--seed", str(seed),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--max-new-tokens", str(args.max_new_tokens),
        "--judge-max-new-tokens", str(args.judge_max_new_tokens),
        "--judge-batch-size", str(args.judge_batch_size),
        "--require-gpu",
        "--device-map", "auto",
        "--torch-dtype", "auto",
        "--sleep", str(args.sleep),
        "--out-dir", str(run_dir),
    ]
    if args.resume:
        sys.argv.append("--resume")
    print(f"[adversarial] Running seed {seed} -> {run_dir}", flush=True)
    qmc.qwen_methodology_main()


def main() -> None:
    """Install adversarial prompts and run the pipeline once per seed."""
    args = parse_args()
    seeds = args.seed if args.seed is not None else [7, 17, 42]
    install_adversarial_protocol()
    for seed in seeds:
        run_one_seed(args, seed)
    print(
        "[adversarial] Done. Compare each adversarial workbook's "
        "Objective_Correlations sheet against the matching baseline_v2 seed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
