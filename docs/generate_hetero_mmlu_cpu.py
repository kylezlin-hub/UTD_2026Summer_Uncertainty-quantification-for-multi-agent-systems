"""Heterogeneous multi-model debate (ReConcile-style) on LOCAL CPU via Ollama.

Tests the ReConcile (Chen et al. 2024, arXiv:2309.13007) hypothesis on our
setup: that DEBATE DIVERSITY comes from using DIFFERENT model families -- not
copies of one model -- which should reduce premature convergence
(degeneration-of-thought) and the null process/accuracy correlation we saw with
3x the same Qwen.

WHAT'S DIFFERENT FROM generate_qwen_mmlu_original_cpu.py
-------------------------------------------------------
  1. Each agent is a DISTINCT Ollama model (--agent-models), e.g.
     qwen2.5:7b-instruct + llama3.1:8b + mistral:7b.
  2. Discussion prompt GROUPS peers by answer with their confidence (ReConcile
     "grouped responses"), instead of a flat per-agent list.
  3. Final answer is a CONFIDENCE-WEIGHTED VOTE over the last round
     (argmax_a sum_i conf_i * [answer_i == a]), replacing consensus->moderator->
     majority. (Uncalibrated weights, per ReConcile "uncalibrated also works".)
Same original-MMLU benchmark, cooperative update rule, per-agent temperature.
The judge (for reasoning-quality q) stays on a single model (--judge-model).

Diversity is the variable under test, so the agents share an IDENTICAL neutral
system prompt -- any change vs the same-model baseline is attributable to model
diversity, not persona wording.

PREREQUISITES
-------------
  ollama pull qwen2.5:7b-instruct
  ollama pull llama3.1:8b
  ollama pull mistral:7b
  (any 3 distinct instruct models work; pass them via --agent-models)
  pip install datasets truststore

PERFORMANCE: with 3 models (~15GB total) Ollama may swap models in/out each call
if they don't all fit in memory -> slow. Start small: --objective-limit 20.

USAGE (from C:/Proj1/docs):
    python generate_hetero_mmlu_cpu.py --seed 7 --objective-limit 20 \
        --agent-models qwen2.5:7b-instruct,llama3.1:8b,mistral:7b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# Localhost (Ollama) must bypass any corporate HTTP proxy (see CPU runner).
_LOCALHOSTS = "127.0.0.1,localhost,::1"
for _pv in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_pv, "")
    os.environ[_pv] = f"{_cur},{_LOCALHOSTS}".strip(",") if _cur else _LOCALHOSTS

import qwen_methodology_code as qmc

# install_original_mmlu also runs the truststore SSL bootstrap on import.
from generate_qwen_mmlu_original import install_original_mmlu, _MMLU_DATASET

# Identical neutral system prompt for all agents (diversity comes from the model).
_SYS = ("You are an expert reasoner collaborating with other agents to find the correct "
        "answer. Reason carefully step by step. Report your answer and how confident you are.")

AGENT_LLMS: dict[str, object] = {}          # agent -> its own OllamaQwenPipeline
AGENT_TEMPS: dict[str, float] = {}          # agent -> sampling temperature


def hetero_initial(question, agent):
    """Round-1 independent answer (identical neutral prompt for every agent)."""
    if question.dataset_type != "objective":
        return [{"role": "system", "content": _SYS},
                {"role": "user", "content": question.question}]
    answer_slot = "/".join(question.answer_labels)
    user = (f"{question.question}\n\n"
            "Reason step by step, then give your answer. Use exactly this format:\n"
            f"Answer: <{answer_slot}>\n"
            "Confidence: <number from 0 to 1>\n"
            "Explanation: <brief justification>")
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]


def group_by_answer_update(question, agent, previous_round):
    """ReConcile-style discussion prompt: peers GROUPED by answer, with confidence."""
    answer_slot = "<" + "/".join(question.answer_labels) + ">"
    groups: dict[str, list] = defaultdict(list)
    for other, turn in previous_round.items():
        if other == agent:
            continue
        groups[turn.get("answer") or "?"].append(turn)
    lines = []
    for ans, members in groups.items():
        confs = ", ".join(f"conf {t.get('confidence', '0.5')}" for t in members)
        reasons = " | ".join(t.get("response", "") for t in members)
        lines.append(f"- Answer {ans} ({len(members)} agent(s), {confs}): {reasons}")
    own = previous_round[agent]
    own_conf = own.get("confidence", "0.5")
    user = (f"Question:\n{question.question}\n\n"
            f"Your previous answer was {own.get('answer')} (confidence {own_conf}): "
            f"{own.get('response', '')}\n\n"
            "Other agents' answers last round, grouped by answer:\n" + "\n".join(lines) +
            "\n\nWeigh these arguments and their confidence. Update your answer only if another "
            "agent's reasoning is genuinely stronger; otherwise keep and defend yours. "
            "Use exactly this format:\n"
            f"Answer: {answer_slot}\n"
            "Confidence: <number from 0 to 1>\n"
            "Explanation: <brief justification grounded in the discussion>")
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]


def hetero_turns(llm, messages_by_agent, question, seeds_by_agent, max_attempts=3):
    """Per-agent generation: each agent uses ITS OWN model (AGENT_LLMS) and
    temperature (AGENT_TEMPS). Sequential (models/temperatures differ per agent)."""
    turns: dict[str, dict[str, object]] = {}
    attempts = max(1, max_attempts)
    for agent, msgs in messages_by_agent.items():
        agent_llm = AGENT_LLMS.get(agent, llm)
        temp = AGENT_TEMPS.get(agent)
        seed = seeds_by_agent[agent]
        first_error = ""
        for attempt in range(attempts):
            am = msgs if attempt == 0 else qmc.qwen_reprompt_messages(msgs, question)
            raw = agent_llm.complete(am, seed=seed + attempt, temperature=temp)
            parsed = qmc.parse_qwen_turn(
                raw, question.dataset_type, question.answer_labels, strict=True)
            if attempt == 0 and parsed["parse_failed"]:
                first_error = str(parsed["parse_error"])
            parsed["re_prompted"] = attempt > 0
            if first_error:
                parsed["first_parse_error"] = first_error
            if not parsed["parse_failed"] or attempt == attempts - 1:
                turns[agent] = parsed
                break
    return turns


def confidence_weighted_final(llm, question, rounds, seed):  # noqa: ARG001
    """ReConcile confidence-weighted vote over the final round:
    argmax_a  sum_i conf_i * [answer_i == a]."""
    last = rounds[-1]
    scores: dict[str, float] = defaultdict(float)
    answers = []
    for agent in qmc.QWEN_AGENTS:
        turn = last.get(agent, {})
        ans = turn.get("answer")
        if not ans:
            continue
        answers.append(ans)
        try:
            conf = float(turn.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        scores[ans] += conf
    if not scores:
        return "", "no_answers"
    best = max(scores, key=scores.get)
    source = "agent_consensus" if len(set(answers)) == 1 else "confidence_weighted_vote"
    return best, source


def installed_models(host: str) -> set | None:
    """Query Ollama for installed model tags; None if unreachable."""
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=10) as resp:
            return {m["name"] for m in json.load(resp).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def preflight(host: str, needed: list[str]) -> None:
    """Fail fast with a clear pull command if any required model is missing."""
    have = installed_models(host)
    if have is None:
        raise SystemExit(f"Cannot reach Ollama at {host}. Is it running?")
    missing = [m for m in needed if m not in have and f"{m}:latest" not in have]
    if missing:
        cmds = "\n".join(f"  ollama pull {m}" for m in missing)
        raise SystemExit(
            f"Missing Ollama model(s): {', '.join(missing)}\nInstalled: {sorted(have)}\n"
            f"Pull them first:\n{cmds}")


def install_hetero(models: list[str], temps: dict[str, float], host: str,
                   top_p: float, max_new_tokens: int) -> None:
    """Bind one distinct model (+ temperature) per agent and install the
    grouped-prompt + confidence-weighted-vote overrides."""
    global AGENT_LLMS, AGENT_TEMPS
    AGENT_TEMPS = temps
    AGENT_LLMS = {}
    for i, agent in enumerate(qmc.QWEN_AGENTS):
        model = models[i % len(models)]
        AGENT_LLMS[agent] = qmc.OllamaQwenPipeline(
            model_id=model, host=host,
            temperature=temps.get(agent, 0.7), top_p=top_p, max_new_tokens=max_new_tokens)
    qmc.qwen_initial_messages = hetero_initial
    qmc.qwen_update_messages = group_by_answer_update
    qmc.complete_qwen_turns_with_retry = hetero_turns
    qmc.select_final_answer = confidence_weighted_final
    mapping = ", ".join(f"{a}={AGENT_LLMS[a].model_id}" for a in qmc.QWEN_AGENTS)
    print(f"[hetero] Agents -> models: {mapping}", flush=True)
    print("[hetero] Grouped discussion prompt + confidence-weighted final vote installed.",
          flush=True)


def parse_args() -> argparse.Namespace:
    """Parse options for the heterogeneous multi-model CPU runner."""
    p = argparse.ArgumentParser(
        description="Heterogeneous multi-model MMLU debate (ReConcile-style) on Ollama/CPU.")
    p.add_argument("--out-dir", type=Path, default=Path("data/mmlu_hetero_cpu"))
    p.add_argument("--objective-limit", type=int, default=100,
                   help="Questions per seed. Start small (e.g. 20) on CPU.")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, action="append", default=None,
                   help="Debate seed. Repeat for multiple seeds. Default: 7 17 42.")
    p.add_argument("--sample-seed", type=int, default=20240101)
    p.add_argument("--subject", action="append", default=None)
    p.add_argument("--agent-models", default="qwen2.5:7b-instruct,llama3.1:8b,mistral:7b",
                   help="Comma-separated Ollama model tags, one per agent (3 agents).")
    p.add_argument("--agent-temps", default="0.4,0.7,1.0",
                   help="Comma-separated per-agent sampling temperatures.")
    p.add_argument("--judge-model", default="qwen2.5:7b-instruct",
                   help="Single model used for reasoning-quality judging.")
    p.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    p.add_argument("--split", default="test")
    p.add_argument("--temperature", type=float, default=0.7, help="Judge/fallback temperature.")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--judge-max-new-tokens", type=int, default=220)
    p.add_argument("--judge-batch-size", type=int, default=1)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def run_one_seed(args: argparse.Namespace, seed: int) -> None:
    """Invoke the pipeline for one seed on Ollama; --model-id is the JUDGE model."""
    run_dir = args.out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "qwen_methodology_code.py",
        "--llm-provider", "qwen",
        "--backend", "ollama",
        "--model-id", args.judge_model,      # pipeline llm = judge; agents use AGENT_LLMS
        "--ollama-host", args.ollama_host,
        "--dataset-source", "mmlu-pro",       # loader monkeypatched to original MMLU
        "--mmlu-pro-dataset", _MMLU_DATASET,
        "--mmlu-pro-split", args.split,
        "--objective-limit", str(args.objective_limit),
        "--subjective-limit", "0",
        "--rounds", str(args.rounds),
        "--seed", str(seed),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--max-new-tokens", str(args.max_new_tokens),
        "--judge-max-new-tokens", str(args.judge_max_new_tokens),
        "--judge-batch-size", str(args.judge_batch_size),
        "--sleep", str(args.sleep),
        "--out-dir", str(run_dir),
    ]
    if args.resume:
        sys.argv.append("--resume")
    print(f"[hetero] Running seed {seed} -> {run_dir}", flush=True)
    qmc.qwen_methodology_main()


def main() -> None:
    """Preflight models, install hetero config, run per seed."""
    args = parse_args()
    seeds = args.seed if args.seed is not None else [7, 17, 42]
    models = [m.strip() for m in args.agent_models.split(",") if m.strip()]
    temps_list = [float(x) for x in str(args.agent_temps).split(",") if x.strip()]
    temps = {a: temps_list[i % len(temps_list)] for i, a in enumerate(qmc.QWEN_AGENTS)}

    preflight(args.ollama_host, list(dict.fromkeys(models + [args.judge_model])))
    install_original_mmlu(args.sample_seed, args.subject)
    install_hetero(models, temps, args.ollama_host, args.top_p, args.max_new_tokens)

    print("[hetero] Config: original MMLU + heterogeneous models + grouped prompt + "
          "confidence-weighted vote. Judge=" + args.judge_model + ".", flush=True)
    for seed in seeds:
        run_one_seed(args, seed)
    print("[hetero] Done. Compare against the same-model baseline with the notebook.", flush=True)


if __name__ == "__main__":
    main()
