"""
generate_exp_a.py  —  Adaptive Influence Balancing Controller (Experiment A)

Design:
  - Gate:  quality_proxy >= 0.80 (confidence used as proxy during debate)
           AND support_delta <= 0 (only protect minorities losing ground)
  - p_truth predictor: logistic regression trained on MMLU-Pro baseline
  - Intervention: prompt-only, ZERO extra tokens
  - 5 pathology-matched prompts (domination, capitulation, dogmatism,
    parallel_reasoning, over_engagement)

Resume:
  Safe to interrupt at any time — per-question JSONL checkpoint.
  Re-run the same command to continue from the last completed question.
  Use --overwrite to start fresh.

Usage:
    python New/generate_exp_a.py \\
        --base-workbook New/baseline_v2_gpqa_s7.xlsx \\
        --predictor New/models_p_truth \\
        --out-dir New/exp_a_results \\
        --dataset-label gpqa --seed 7 \\
        --backend ollama --model-id qwen2.5:7b-instruct \\
        --no-require-gpu
"""

from __future__ import annotations

import argparse, json, re, random, sys, time, traceback
from collections import Counter
from math import log
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "docs"))

from qwen_methodology_code import (
    DebateQuestion, LocalQwenPipeline, OllamaQwenPipeline,
    OBJECTIVE_LABELS, QWEN_AGENTS, EPS,
    answer_is_valid, empty_judgments, final_answer_correctness,
    first_consensus_round_for_answer, judge_debates_with_qwen,
    majority_answer, parse_confidence, parse_qwen_turn,
    qwen_initial_messages, qwen_moderator_messages,
    qwen_reprompt_messages, qwen_update_messages,
    score_mixed_debates, write_qwen_excel_report,
)

# ── Constants ──────────────────────────────────────────────────────────────
# NOTE: QUALITY_GATE is applied to confidence scores during the debate
# (real LLM judge scores are not available until after all debates finish).
# The 0.80 threshold was derived from actual judge scores in the baseline;
# confidence is used as an online proxy with similar distribution.
QUALITY_GATE    = 0.80   # confidence >= 0.80 (proxy for judge quality)
P_TRUTH_TRIGGER = 0.50   # Youden-J threshold from OOF analysis
P_TRUTH_STRONG  = 0.60   # above this → also apply to majority agents
BASE_TOKENS     = 220    # same for ALL agents — no extra budget


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Checkpoint helpers (resume support)
# ═══════════════════════════════════════════════════════════════════════════

def checkpoint_path(out_dir: Path, dataset_label: str, seed: int) -> Path:
    return out_dir / f"exp_a_{dataset_label}_s{seed}.checkpoint.jsonl"


def load_checkpoint(out_dir: Path, dataset_label: str, seed: int
                    ) -> tuple[list[dict], list[dict], list[dict], set]:
    """Load completed debates, states, interventions and done question nos."""
    cp = checkpoint_path(out_dir, dataset_label, seed)
    debate_by_qno: dict[str, dict] = {}
    state_by_qno:  dict[str, list] = {}
    intv_by_qno:   dict[str, list] = {}
    if not cp.exists():
        return [], [], [], set()
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
            t   = entry.get("_type", "")
            if t == "debate":
                debate_by_qno[qno] = entry["data"]
            elif t == "state":
                state_by_qno[qno]  = entry["data"]
            elif t == "intervention":
                intv_by_qno[qno]   = entry["data"]
    done  = set(debate_by_qno.keys())
    debates = [debate_by_qno[k] for k in debate_by_qno]
    states  = [r for k in debate_by_qno for r in state_by_qno.get(k, [])]
    intvs   = [r for k in debate_by_qno for r in intv_by_qno.get(k, [])]
    if done:
        print(f"Checkpoint: {len(done)} questions already completed.", flush=True)
    return debates, states, intvs, done


def save_checkpoint(out_dir: Path, dataset_label: str, seed: int,
                    question_no: str, debate_row: dict,
                    state_rows: list[dict], intv_rows: list[dict]) -> None:
    """Append one completed question atomically (single write call)."""
    cp = checkpoint_path(out_dir, dataset_label, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    line1 = json.dumps({"_type": "debate",       "question_no": question_no,
                        "data": debate_row})
    line2 = json.dumps({"_type": "state",        "question_no": question_no,
                        "data": state_rows})
    line3 = json.dumps({"_type": "intervention", "question_no": question_no,
                        "data": intv_rows})
    with cp.open("a", encoding="utf-8") as f:
        f.write(line1 + "\n" + line2 + "\n" + line3 + "\n")


def clear_checkpoint(out_dir: Path, dataset_label: str, seed: int) -> None:
    cp = checkpoint_path(out_dir, dataset_label, seed)
    if cp.exists():
        cp.unlink()
        print(f"Cleared checkpoint: {cp.name}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Predictor
# ═══════════════════════════════════════════════════════════════════════════

class PtruthPredictor:
    """Logistic regression predictor of minority hypothesis correctness."""

    def __init__(self, model_dir: Path):
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.model  = joblib.load(model_dir / "p_truth_logistic.pkl")
        with open(model_dir / "feature_meta.json") as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        print(f"Loaded p_truth predictor ({len(self.feature_names)} features, "
              f"CV AUC={meta['best_auc_cv']:.3f})", flush=True)

    def predict(self, diag: dict, agent_data: dict, round_num: int) -> float:
        """Compute p_truth from agent-level and debate-level features."""
        q   = float(agent_data.get("quality_proxy", 0.5) or 0.5)
        sd  = float(agent_data.get("support_delta", 0.0) or 0.0)
        nd  = float(agent_data.get("num_defectors", 0) or 0)
        ms  = float(agent_data.get("minority_size", 1) or 1)
        wmb = float(agent_data.get("was_majority_before", 0))
        lm  = float(agent_data.get("left_majority", 0))
        conf= float(agent_data.get("confidence", 0.5) or 0.5)
        rr  = max(0, 5 - round_num)
        pai = float(agent_data.get("prefix_attributed_influence", 0.0) or 0.0)

        sd_c = max(-3.0, min(3.0, sd))
        feat = {
            "quality_x_support_delta":        q * sd_c,
            "support_loss_rate":               sd / (rr + 1),
            "support_delta":                   sd,
            "judge_explanation_good":          q,
            "defector_ratio":                  nd / (ms + 1),
            "num_defectors":                   nd,
            "all_judge_scores":                q,          # proxy: all same
            "was_majority_x_delta":            wmb * sd_c,
            "confidence_x_quality":            conf * q,
            "independent_reasoning_x_quality": q * q,     # proxy
            "quality_x_defectors":             q * (-nd),
            "left_majority":                   lm,
            "isolating":                       float(sd < 0 and nd > 0),
            "quality_x_rounds":                q * rr,
            "prefix_attributed_influence":     pai,
        }
        x = np.array([[feat.get(f, 0.0) for f in self.feature_names]])
        try:
            x_scaled = self.scaler.transform(x)
            return float(self.model.predict_proba(x_scaled)[0, 1])
        except Exception:
            return 0.0   # safe fallback


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Prefix diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def _entropy(vals: list) -> float:
    v = [x for x in vals if x and str(x) != "nan"]
    if len(v) <= 1:
        return 0.0
    counts = np.array(list(Counter(v).values()), dtype=float)
    p = counts / counts.sum()
    return float(-sum(pi * log(pi) for pi in p if pi > 0) / log(len(v)))


def _safe(v, default=float("nan")):
    """Return float(v) or default if v is None / NaN."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if f != f else f   # NaN check
    except (TypeError, ValueError):
        return default


def compute_prefix_diagnostics(
    answers_by_round: list[list],
    quality_by_round: list[list],
    up_to_round: int,
) -> dict:
    """Prefix-safe diagnostics from rounds 1..up_to_round (no future leakage)."""
    answers = answers_by_round[:up_to_round]
    quality = quality_by_round[:up_to_round]
    n = len(answers[0]) if answers else len(QWEN_AGENTS)

    base = {
        "prefix_engagement":           float("nan"),
        "prefix_responsiveness":       float("nan"),
        "prefix_influence_asymmetry":  0.0,
        "prefix_balance":              float("nan"),
        "prefix_dispersion":           _entropy(answers[-1]) if answers else 0.0,
        "prefix_influence_per_agent":  [0.0] * n,
    }
    if len(answers) < 2:
        return base

    eng, resp, influence = [], [], np.zeros(n)
    for t in range(1, len(answers)):
        for a in range(n):
            old, new = answers[t-1][a], answers[t][a]
            if not old or not new:
                continue
            q = _safe(quality[t][a] if t < len(quality) and a < len(quality[t])
                      else 0.5, 0.5)
            changed = (old != new)
            eng.append(q * float(changed))
            others = [answers[t-1][j] for j in range(n) if j != a and answers[t-1][j]]
            if others:
                resp.append(q * float(changed and
                    sum(o == new for o in others) > sum(o == old for o in others)))
            for src in range(n):
                src_ans = answers[t-1][src]
                if src == a or not src_ans:
                    continue
                if changed and new == src_ans and old != src_ans:
                    influence[src] += q

    total_inf = float(influence.sum())
    if total_inf > EPS and n > 1:
        p_inf = influence / total_inf
        h = -sum(pi * log(pi) for pi in p_inf if pi > 0)
        base["prefix_influence_asymmetry"] = float(1.0 - h / log(n))

    base["prefix_influence_per_agent"] = influence.tolist()
    dispersion = [_entropy(r) for r in answers]

    steps = [dispersion[t-1] - dispersion[t] for t in range(1, len(dispersion))]
    pos   = [max(0.0, s) for s in steps]
    total_pos = sum(pos); max_pos = max(pos) if pos else 0.0
    if total_pos > EPS:
        collapse = 1.0 - max_pos / (total_pos + EPS)
    else:
        collapse = 1.0 if max((abs(s) for s in steps), default=0) <= EPS else 0.0
    revs = sum(steps[i-1]*steps[i] < 0 for i in range(1, len(steps)))
    vol  = 1.0 if len(dispersion) <= 2 else 1.0 - revs / (len(dispersion) - 2)
    base["prefix_balance"]       = float(np.clip(collapse * vol, 0.0, 1.0))
    base["prefix_dispersion"]    = dispersion[-1]
    base["prefix_engagement"]    = float(np.mean(eng))  if eng  else float("nan")
    base["prefix_responsiveness"]= float(np.mean(resp)) if resp else float("nan")
    return base


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Gate
# ═══════════════════════════════════════════════════════════════════════════

def passes_gate(quality_proxy: float, support_delta: int) -> bool:
    """
    Gate 1 (Quality proxy): confidence >= 0.80
      The LLM judge uses discrete levels; 0.80 is the natural modal boundary.
      During the debate we use the agent's self-reported confidence as a proxy
      (real LLM judge scores are computed after all debates complete).

    Gate 2 (Necessity): support_delta <= 0
      Minorities gaining support are already winning naturally — skip them.
      From analysis: gaining minorities are 41% correct vs 21% for losing.
    """
    return quality_proxy >= QUALITY_GATE and support_delta <= 0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Pathology detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_pathology(diag: dict) -> tuple[str, float]:
    asym = _safe(diag.get("prefix_influence_asymmetry"), 0.0)
    bal  = _safe(diag.get("prefix_balance"),             1.0)
    eng  = _safe(diag.get("prefix_engagement"),          0.1)
    resp = _safe(diag.get("prefix_responsiveness"),      0.05)

    scores = {
        "domination":         max(0.0, asym - 0.15),
        "capitulation":       max(0.0, 0.60 - bal),
        "dogmatism":          max(0.0, 0.05 - eng)  if eng  < 0.05  else 0.0,
        "parallel_reasoning": max(0.0, 0.02 - resp) if resp < 0.02  else 0.0,
        "over_engagement":    max(0.0, eng  - 0.25),
    }
    worst    = max(scores, key=scores.get)
    severity = scores[worst]
    return worst, severity


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Intervention prompts
# ═══════════════════════════════════════════════════════════════════════════

def minority_prompt(pathology: str, p_truth: float) -> str:
    base = (
        f"\n\n[Adaptive Controller — Minority Protection "
        f"(estimated correctness: {p_truth:.0%})]\n"
    )
    additions = {
        "domination":
            "High influence concentration detected. Re-examine your reasoning "
            "independently. Address the strongest counterargument, then restate "
            "your position with explicit evidence.",
        "capitulation":
            "Rapid convergence detected. Before accepting the majority view, "
            "identify one specific piece of evidence that would need to be wrong "
            "for your answer to be incorrect. If that evidence is sound, defend your position.",
        "dogmatism":
            "Revisit this question from first principles. What core concept is "
            "being tested? State your reasoning from scratch, not from your prior response.",
        "parallel_reasoning":
            "Directly address what peers argued. For each peer, state explicitly "
            "whether their argument changes your view and why.",
        "over_engagement":
            "You have updated multiple times. Before updating again, identify "
            "the specific new evidence that justifies each change. "
            "Return to your most defensible position if unsure.",
    }
    return base + additions.get(pathology, "Carefully re-examine your position.")


def majority_prompt(pathology: str, majority_answer: str,
                    minority_explanation: str) -> str:
    if pathology == "domination":
        return (
            f"\n\n[Adaptive Controller — Structured Rebuttal]\n"
            f"A peer holds a different position with this argument:\n"
            f'"{minority_explanation[:250]}"\n'
            f"Before confirming your answer:\n"
            f"1. Restate their argument in your own words\n"
            f"2. Identify the specific evidence that defeats it\n"
            f"3. Only then give your updated answer"
        )
    elif pathology == "capitulation":
        return (
            f"\n\n[Adaptive Controller — Devil's Advocate]\n"
            f"The group is converging on: {majority_answer}\n"
            f"Before committing, identify the strongest argument AGAINST "
            f"{majority_answer}. What evidence would disprove it? "
            f"Give your honest best answer after this check."
        )
    elif pathology == "parallel_reasoning":
        return (
            f"\n\n[Adaptive Controller — Direct Response]\n"
            f"You MUST address what peers actually argued. "
            f"Write one sentence per peer: "
            f"'[Agent X] argued [point]. This [supports/contradicts] my view because...'"
        )
    return (
        f"\n\n[Adaptive Controller — Peer Engagement]\n"
        f"Address at least one peer argument specifically in your response."
    )


def stability_anchor_prompt() -> str:
    return (
        "\n\n[Adaptive Controller — Stability Check]\n"
        "Before deciding: What specific new evidence justified your "
        "most recent position change? If you cannot identify concrete evidence, "
        "return to your most defensible position."
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Question loader
# ═══════════════════════════════════════════════════════════════════════════

def load_questions(path: Path) -> list[DebateQuestion]:
    df = pd.read_excel(path, sheet_name="Debate_Traces")
    questions = []
    for _, row in df.iterrows():
        q_text  = str(row.get("Question", "")).strip()
        correct = str(row.get("Correct Answer", "")).strip().upper()
        if not q_text or not correct:
            continue
        found  = re.findall(r"(?m)^([A-J])\.\s+", q_text)
        labels = tuple(l for l in OBJECTIVE_LABELS if l in set(found))
        if correct not in labels:
            labels = OBJECTIVE_LABELS[:max(4, OBJECTIVE_LABELS.index(correct) + 1)]
        labels = labels or OBJECTIVE_LABELS[:4]
        questions.append(DebateQuestion(
            dataset_type   = "objective",
            question_no    = str(row.get("Question #", "")),
            question       = q_text,
            correct_answer = correct,
            answer_labels  = labels,
            category       = str(row.get("Dataset Category", "")),
        ))
    print(f"Loaded {len(questions)} questions from {path.name}", flush=True)
    return questions


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Core debate loop
# ═══════════════════════════════════════════════════════════════════════════

def run_one_debate(
    llm,
    question: DebateQuestion,
    predictor: PtruthPredictor,
    seed: int,
    rounds: int,
    sleep: float,
    rng: random.Random,
    dataset_label: str = "gpqa",
    exp_seed: int = 7,
    run_id: int = 1,
) -> tuple[dict | None, list[dict], list[dict]]:
    """
    Run one debate with the adaptive controller.
    Returns (debate_row, round_state_rows, intervention_log).
    debate_row is None if the debate was discarded (parse failures).
    """
    agent_order = QWEN_AGENTS[:]
    rng.shuffle(agent_order)

    all_rounds:    list[dict] = []
    quality_cache: dict[int, dict] = {}   # round_idx -> {agent: confidence}
    diag_by_round: dict[int, dict] = {}   # round_num  -> prefix diagnostics
    intervention_log: list[dict] = []

    for round_no in range(1, rounds + 1):

        # ── 1. Build base prompts ───────────────────────────────────
        messages: dict[str, list] = {}
        for agent in agent_order:
            if round_no == 1:
                messages[agent] = qwen_initial_messages(question, agent)
            else:
                messages[agent] = qwen_update_messages(
                    question, agent, all_rounds[-1])

        # ── 2. Adaptive controller (rounds 2–5 only) ────────────────
        if round_no >= 2 and all_rounds:
            # Compute and cache diagnostics for this prefix (used in _build_round_state)
            answers_hist = [[str(r[a].get("answer","") or "") or None
                             for a in QWEN_AGENTS] for r in all_rounds]
            quality_hist = [[quality_cache.get(i, {}).get(a, 0.5)
                             for a in QWEN_AGENTS] for i in range(len(all_rounds))]
            diag_now = compute_prefix_diagnostics(answers_hist, quality_hist,
                                                  len(all_rounds))
            diag_by_round[round_no] = diag_now   # store for Round_State

            controller_result = _apply_controller(
                messages, all_rounds, quality_cache,
                predictor, round_no, question.question_no,
                intervention_log
            )
            messages = controller_result

        # ── 3. Generate responses ───────────────────────────────────
        round_turns: dict[str, dict] = {}
        discard_reason = ""
        for agent in agent_order:
            base_seed = seed + round_no * 100 + QWEN_AGENTS.index(agent) * 10
            try:
                raw = llm.complete(messages[agent], seed=base_seed,
                                   max_new_tokens=BASE_TOKENS)
            except Exception as e:
                print(f"  LLM error at r{round_no} {agent}: {e}", flush=True)
                discard_reason = f"LLM error r{round_no} {agent}"
                break

            parsed = parse_qwen_turn(
                raw, question.dataset_type, question.answer_labels, strict=True)
            if parsed["parse_failed"]:
                try:
                    retry = qwen_reprompt_messages(messages[agent], question)
                    raw2  = llm.complete(retry, seed=base_seed + 1,
                                         max_new_tokens=BASE_TOKENS)
                except Exception as e:
                    print(f"  LLM retry error r{round_no} {agent}: {e}", flush=True)
                    discard_reason = f"LLM retry error r{round_no} {agent}"
                    break
                p2 = parse_qwen_turn(
                    raw2, question.dataset_type, question.answer_labels, strict=True)
                if not p2["parse_failed"]:
                    parsed = {**p2, "re_prompted": True}
                else:
                    discard_reason = (f"{question.question_no} r{round_no} "
                                      f"{agent}: {parsed['parse_error']}")
                    break
            round_turns[agent] = parsed

        if sleep:
            time.sleep(sleep)
        if discard_reason:
            print(f"  Discarding: {discard_reason}", flush=True)
            return None, [], []

        # Cache confidence as quality proxy for next round's controller
        quality_cache[round_no - 1] = {
            a: _safe(parse_confidence(round_turns[a].get("confidence", "")), 0.5)
            for a in QWEN_AGENTS
        }
        all_rounds.append(round_turns)

        # Store POST-round diagnostics (rounds 1..round_no inclusive) for Round_State.
        # These match the baseline schema: round-t diagnostics include round-t answers.
        post_answers = [[str(r[a].get("answer","") or "") or None
                         for a in QWEN_AGENTS] for r in all_rounds]
        post_quality = [[quality_cache.get(i, {}).get(a, 0.5)
                         for a in QWEN_AGENTS] for i in range(len(all_rounds))]
        diag_post = compute_prefix_diagnostics(post_answers, post_quality,
                                               len(all_rounds))
        # compute influence_share (not set by compute_prefix_diagnostics)
        inf_list = diag_post.get("prefix_influence_per_agent", [0.0]*len(QWEN_AGENTS))
        total_inf = sum(inf_list)
        if total_inf > EPS:
            diag_post["prefix_influence_share"] = [v / total_inf for v in inf_list]
        else:
            diag_post["prefix_influence_share"] = [float("nan")] * len(QWEN_AGENTS)
        diag_by_round[round_no] = diag_post   # overwrite pre-round entry

    # ── 4. Final answer ─────────────────────────────────────────────
    final_ans_list = [all_rounds[-1][a]["answer"] for a in QWEN_AGENTS]
    valid = [a for a in final_ans_list if a]
    if valid and len(set(valid)) == 1:
        final_answer, final_source = valid[0], "agent_consensus"
    else:
        try:
            raw = llm.complete(
                qwen_moderator_messages(question, all_rounds),
                seed=seed + rounds * 100 + 999)
            parsed = parse_qwen_turn(
                raw, question.dataset_type, question.answer_labels, strict=False)
            candidate = parsed["answer"]
            if answer_is_valid(candidate, question):
                final_answer, final_source = candidate, "moderator"
            else:
                final_answer = majority_answer(final_ans_list)
                final_source = "majority_vote_no_moderator"
        except Exception:
            final_answer = majority_answer(final_ans_list)
            final_source = "majority_vote_no_moderator"

    # ── 5. Build debate row ─────────────────────────────────────────
    debate_row: dict = {
        "Question #": question.question_no, "Dataset Type": question.dataset_type,
        "Dataset Category": question.category, "Question": question.question,
        "Correct Answer": question.correct_answer, "Final Answer": final_answer,
        "Final Answer Source": final_source, "Fixture Pattern": "exp_a_adaptive",
        "Rounds to Consensus": first_consensus_round_for_answer(
            all_rounds, final_answer),
    }
    for rn, turns in enumerate(all_rounds, start=1):
        for agent in QWEN_AGENTS:
            t = turns[agent]
            debate_row[f"R{rn} {agent} Answer"]   = t["answer"]
            debate_row[f"R{rn} {agent} Conf"]     = t.get("confidence", "")
            debate_row[f"R{rn} {agent} Response"] = t.get("response", "")

    state_rows = _build_round_state(
        all_rounds, question, final_answer, intervention_log,
        diag_by_round=diag_by_round,
        dataset_label=dataset_label,
        seed=exp_seed,
        run_id=run_id)

    return debate_row, state_rows, intervention_log


def _apply_controller(
    messages: dict, all_rounds: list, quality_cache: dict,
    predictor: PtruthPredictor, round_no: int,
    question_no: str, intervention_log: list,
) -> dict:
    """
    Compute diagnostics, detect minorities, apply gate + predictor,
    modify prompts. Returns updated messages dict.
    """
    # Build answer/quality history up to previous round
    answers_hist = [
        [str(r[a].get("answer", "") or "") or None for a in QWEN_AGENTS]
        for r in all_rounds
    ]
    quality_hist = [
        [quality_cache.get(i, {}).get(a, 0.5) for a in QWEN_AGENTS]
        for i in range(len(all_rounds))
    ]
    diag      = compute_prefix_diagnostics(answers_hist, quality_hist, len(all_rounds))
    pathology, severity = detect_pathology(diag)

    # Current answer distribution
    curr_answers  = {a: str(all_rounds[-1][a].get("answer", "")) for a in QWEN_AGENTS}
    answer_counts = Counter(v for v in curr_answers.values() if v)
    if not answer_counts:
        return messages

    max_sup      = max(answer_counts.values())
    majority_ans = answer_counts.most_common(1)[0][0]
    prev_counts  = Counter(
        str(all_rounds[-2][a].get("answer", "")) for a in QWEN_AGENTS
    ) if len(all_rounds) > 1 else Counter()

    minority_agents = [
        a for a in QWEN_AGENTS
        if curr_answers[a] and answer_counts[curr_answers[a]] < max_sup
    ]
    print(f"  [R{round_no}] answers={curr_answers}  "
          f"pathology={pathology}(sev={severity:.3f})  "
          f"minorities={minority_agents}", flush=True)

    influenced = diag.get("prefix_influence_per_agent", [0.0] * len(QWEN_AGENTS))
    if not isinstance(influenced, list):
        influenced = [0.0] * len(QWEN_AGENTS)

    intervention_applied_this_round = False

    for a_idx, agent in enumerate(QWEN_AGENTS):
        curr_ans = curr_answers[agent]
        if not curr_ans or answer_counts[curr_ans] >= max_sup:
            continue   # majority or unanimous — skip minority gate

        curr_sup  = answer_counts[curr_ans]
        prev_sup  = prev_counts.get(curr_ans, 0)
        prev_max  = max(prev_counts.values()) if prev_counts else 0
        wmb       = int(prev_sup >= prev_max and prev_max > 0)
        nd        = sum(
            1 for aa in QWEN_AGENTS
            if len(all_rounds) > 1
            and str(all_rounds[-2][aa].get("answer", "")) == curr_ans
            and curr_answers[aa] != curr_ans
        )
        prev_maj  = prev_counts.most_common(1)[0][0] if prev_counts else ""
        lm        = int(
            len(all_rounds) > 1
            and str(all_rounds[-2][agent].get("answer", "")) == prev_maj
            and curr_ans != prev_maj
        )
        conf_raw  = all_rounds[-1][agent].get("confidence", "0.5")
        conf      = _safe(parse_confidence(conf_raw), 0.5)
        # Use confidence from PREVIOUS round as quality proxy (most recent judge)
        q_proxy   = quality_cache.get(len(all_rounds) - 1, {}).get(agent, conf)
        q_proxy   = _safe(q_proxy, conf)
        # If previous round quality unavailable, fall back to current confidence
        if q_proxy == 0.5 and conf > 0.5:
            q_proxy = conf

        delta     = curr_sup - prev_sup
        pai       = influenced[a_idx] if a_idx < len(influenced) else 0.0

        # ── GATE ────────────────────────────────────────────────────
        gate_ok = passes_gate(q_proxy, delta)
        print(f"    [GATE] r{round_no} {agent}: "
              f"quality={q_proxy:.2f}({'OK' if q_proxy>=QUALITY_GATE else 'FAIL'}) "
              f"delta={delta:+d}({'OK' if delta<=0 else 'FAIL'}) "
              f"-> {'PASS' if gate_ok else 'SKIP'}", flush=True)
        if not gate_ok:
            continue

        # ── PREDICTOR ───────────────────────────────────────────────
        agent_data = {
            "quality_proxy":              q_proxy,
            "support_delta":              delta,
            "num_defectors":              nd,
            "minority_size":              curr_sup,
            "support":                    curr_sup,
            "was_majority_before":        wmb,
            "left_majority":              lm,
            "confidence":                 conf,
            "prefix_attributed_influence":pai,
        }
        p_truth = predictor.predict(diag, agent_data, round_no)

        print(f"    [PRED] r{round_no} {agent}: p_truth={p_truth:.3f} "
              f"({'TRIGGER' if p_truth>=P_TRUTH_TRIGGER else 'below threshold'})",
              flush=True)

        if p_truth < P_TRUTH_TRIGGER:
            continue

        # ── INTERVENTION ────────────────────────────────────────────
        min_expl = str(all_rounds[-1][agent].get("response", ""))[:250]

        # Minority always gets protection prompt
        messages[agent][-1] = {
            **messages[agent][-1],
            "content": messages[agent][-1]["content"] +
                       minority_prompt(pathology, p_truth)
        }

        # High-confidence → also apply to majority agents
        if p_truth >= P_TRUTH_STRONG:
            for maj_agent in QWEN_AGENTS:
                if curr_answers[maj_agent] == majority_ans:
                    messages[maj_agent][-1] = {
                        **messages[maj_agent][-1],
                        "content": messages[maj_agent][-1]["content"] +
                                   majority_prompt(pathology, majority_ans, min_expl)
                    }

        intervention_log.append({
            "question_no":        question_no,
            "round":              round_no,
            "agent":              agent,
            "pathology":          pathology,
            "severity":           round(severity, 4),
            "p_truth":            round(p_truth, 4),
            "quality_proxy":      round(q_proxy, 4),
            "support_delta":      delta,
            "applied_to_majority":p_truth >= P_TRUTH_STRONG,
        })
        intervention_applied_this_round = True
        print(f"    [CTRL] r{round_no} {agent}: "
              f"pathology={pathology} p_truth={p_truth:.2f}  "
              f"-> INTERVENTION APPLIED", flush=True)

    # Over-engagement: stability anchor on all agents (only if minorities exist
    # and no other intervention already fired this round)
    if (pathology == "over_engagement" and severity > 0.05
            and minority_agents and not intervention_applied_this_round):
        for agent in QWEN_AGENTS:
            messages[agent][-1] = {
                **messages[agent][-1],
                "content": messages[agent][-1]["content"] +
                           stability_anchor_prompt()
            }
        print(f"    [CTRL] r{round_no} stability_anchor applied to all agents",
              flush=True)

    return messages


def _build_round_state(all_rounds, question, final_answer, intervention_log,
                       dataset_label="gpqa", seed=7, run_id=1,
                       diag_by_round=None, judgments_df=None) -> list[dict]:
    """Build Round_State with the same 49-column schema as the baseline workbook.
    Adds 4 extra columns for ExpA: intervention_applied, pathology_detected,
    p_truth_score, applied_to_majority."""
    rows = []
    final_correct = (final_answer == question.correct_answer)
    intv_by_ra    = {(e["round"], e["agent"]): e for e in intervention_log}
    diag_by_round = diag_by_round or {}

    for round_idx, turns in enumerate(all_rounds):
        round_num  = round_idx + 1
        next_r     = all_rounds[round_idx + 1] if round_idx + 1 < len(all_rounds) else None
        final_r    = all_rounds[-1]
        curr  = {a: str(turns[a].get("answer", ""))            for a in QWEN_AGENTS}
        prev  = ({a: str(all_rounds[round_idx-1][a].get("answer", "")) for a in QWEN_AGENTS}
                 if round_idx > 0 else {a: "" for a in QWEN_AGENTS})
        cnts  = Counter(v for v in curr.values() if v)
        max_s = max(cnts.values()) if cnts else 0
        maj   = cnts.most_common(1)[0][0] if cnts else ""
        prev_cnts = (Counter(v for v in prev.values() if v) if round_idx > 0 else Counter())
        prev_max  = max(prev_cnts.values()) if prev_cnts else 0
        prev_maj  = prev_cnts.most_common(1)[0][0] if prev_cnts else ""
        diag      = diag_by_round.get(round_num, {})
        pinfs     = diag.get("prefix_influence_per_agent", [0.0] * len(QWEN_AGENTS))
        if not isinstance(pinfs, list): pinfs = [0.0] * len(QWEN_AGENTS)
        pshares   = diag.get("prefix_influence_share", [float("nan")] * len(QWEN_AGENTS))
        if not isinstance(pshares, list): pshares = [float("nan")] * len(QWEN_AGENTS)

        for a_idx, agent in enumerate(QWEN_AGENTS):
            ans   = curr[agent]; prev_a = prev[agent]
            sup   = cnts.get(ans, 0); prev_sup = prev_cnts.get(ans, 0)
            intv  = intv_by_ra.get((round_num, agent), {})
            turn  = turns[agent]
            next_ans  = str(next_r[agent].get("answer", "")) if next_r else ""
            final_ans = str(final_r[agent].get("answer", ""))
            drops     = bool(ans and next_r and next_ans and next_ans != ans)
            survives  = bool(ans and final_ans == ans)
            is_cor    = bool(ans and ans == question.correct_answer)
            nd = sum(1 for aa in QWEN_AGENTS
                     if round_idx > 0
                     and str(all_rounds[round_idx-1][aa].get("answer","")) == ans
                     and curr[aa] != ans)
            nj = sum(1 for aa in QWEN_AGENTS
                     if round_idx > 0
                     and str(all_rounds[round_idx-1][aa].get("answer","")) != ans
                     and curr[aa] == ans) if round_idx > 0 else sup
            wmb = bool(round_idx > 0 and prev_cnts.get(ans,0) >= prev_max and prev_max > 0)
            lft = bool(round_idx > 0 and prev_a == prev_maj and ans != prev_maj and prev_maj)
            jnd = bool(round_idx > 0 and prev_a != maj and ans == maj)

            # Judge scores — filled in later; NaN until judging
            je = jr = jj = ji = float("nan"); jfail = False
            if judgments_df is not None and len(judgments_df) > 0:
                mask = ((judgments_df.get("question_no", pd.Series()) == question.question_no) &
                        (judgments_df.get("round", pd.Series()) == round_num) &
                        (judgments_df.get("agent", pd.Series()) == agent))
                m = judgments_df[mask]
                if len(m) > 0:
                    je    = _safe(m.iloc[0].get("explanation_good"))
                    jr    = _safe(m.iloc[0].get("uses_past_round_reasoning"))
                    jj    = _safe(m.iloc[0].get("justifies_current_stance"))
                    ji    = _safe(m.iloc[0].get("independent_reasoning"))
                    jfail = bool(m.iloc[0].get("judge_parse_failed", False))

            rows.append({
                # ── 49 baseline columns ──────────────────────────────
                "run_id":                     run_id,
                "dataset":                    dataset_label,
                "seed":                       seed,
                "question_no":                question.question_no,
                "correct_answer":             question.correct_answer,
                "final_answer":               final_answer,
                "final_answer_correct":       final_correct,
                "round":                      round_num,
                "agent":                      agent,
                "answer":                     ans,
                "confidence":                 _safe(parse_confidence(turn.get("confidence","")), 0.5),
                "explanation_text":           str(turn.get("response", "")),
                "response_tokens":            len(str(turn.get("response","")).split()),
                "re_prompted":                bool(turn.get("re_prompted", False)),
                "prev_answer":                prev_a,
                "answer_changed":             bool(prev_a and ans != prev_a),
                "joined_majority":            jnd,
                "left_majority":              lft,
                "is_minority":                bool(0 < sup < max_s),
                "support":                    sup,
                "support_delta":              sup - prev_sup,
                "minority_size":              sup if 0 < sup < max_s else 0,
                "was_majority_before":        wmb,
                "num_defectors":              nd,
                "num_joiners":                nj,
                "majority_answer":            maj,
                "majority_support":           max_s,
                "n_distinct_answers":         len(cnts),
                "consensus_reached":          len(cnts) == 1,
                "prefix_engagement":          _safe(diag.get("prefix_engagement")),
                "prefix_responsiveness":      _safe(diag.get("prefix_responsiveness")),
                "prefix_influence_asymmetry": _safe(diag.get("prefix_influence_asymmetry"), 0.0),
                "prefix_balance":             _safe(diag.get("prefix_balance")),
                "prefix_dispersion":          _safe(diag.get("prefix_dispersion"), 0.0),
                "prefix_influence_share":     _safe(pshares[a_idx] if a_idx < len(pshares) else float("nan")),
                "prefix_attributed_influence":_safe(pinfs[a_idx] if a_idx < len(pinfs) else 0.0, 0.0),
                "answer_at_next_round":       next_ans,
                "drops_next_round":           drops,
                "answer_at_final":            final_ans,
                "answer_survives_to_final":   survives,
                "is_correct":                 is_cor,
                "is_correct_minority":        bool(is_cor and 0 < sup < max_s),
                "correct_drops_next":         bool(is_cor and drops),
                "correct_survives_final":     bool(is_cor and survives),
                "judge_explanation_good":     je,
                "judge_uses_past_reasoning":  jr,
                "judge_justifies_stance":     jj,
                "judge_independent_reasoning":ji,
                "judge_parse_failed":         jfail,
                # ── 4 ExpA-specific columns (appended) ───────────────
                "intervention_applied":       bool(intv),
                "pathology_detected":         intv.get("pathology", ""),
                "p_truth_score":              intv.get("p_truth", float("nan")),
                "applied_to_majority":        intv.get("applied_to_majority", False),
            })
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Experiment A — Adaptive Controller")
    p.add_argument("--base-workbook",  type=Path,
                   default=Path("New/baseline_v2_gpqa_s7.xlsx"))
    p.add_argument("--predictor",      type=Path,
                   default=Path("New/models_p_truth"))
    p.add_argument("--out-dir",        type=Path,
                   default=Path("New/exp_a_results"))
    p.add_argument("--dataset-label",  default="gpqa")
    p.add_argument("--seed",           type=int, default=7)
    p.add_argument("--rounds",         type=int, default=5)
    p.add_argument("--model-id",       default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--backend",        choices=["ollama","transformers"],
                   default="transformers")
    p.add_argument("--ollama-host",    default="http://127.0.0.1:11434")
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top-p",          type=float, default=0.9)
    p.add_argument("--sleep",          type=float, default=0.0)
    p.add_argument("--limit",          type=int,   default=None)
    p.add_argument("--skip-judging",   action="store_true")
    p.add_argument("--q-source",       choices=["llm","confidence"],
                   default="llm")
    p.add_argument("--no-require-gpu", action="store_true")
    p.add_argument("--overwrite",      action="store_true",
                   help="Wipe checkpoint and start fresh")
    return p.parse_args()


def _detect_gpu() -> str:
    """Return a string describing GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i)
                     for i in range(torch.cuda.device_count())]
            return f"GPU AVAILABLE: {', '.join(names)}"
        return "NO CUDA GPU detected — will run on CPU (very slow for 14B)"
    except ImportError:
        return "PyTorch not installed — cannot detect GPU"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / f"exp_a_{args.dataset_label}_s{args.seed}.xlsx"

    # Handle overwrite
    if args.overwrite:
        clear_checkpoint(args.out_dir, args.dataset_label, args.seed)
        if output_path.exists():
            output_path.unlink()

    # Check if already fully complete
    if output_path.exists():
        try:
            sheets = set(pd.ExcelFile(output_path).sheet_names)
            if {"Debate_Traces", "Round_State"}.issubset(sheets):
                print(f"Already complete: {output_path}. "
                      f"Use --overwrite to redo.", flush=True)
                return
        except Exception:
            pass

    print("=" * 65, flush=True)
    print("EXPERIMENT A — Adaptive Influence Balancing Controller", flush=True)
    print(f"  Dataset:   {args.dataset_label}  seed={args.seed}", flush=True)
    print(f"  Model:     {args.model_id}  backend={args.backend}", flush=True)
    print(f"  {_detect_gpu()}", flush=True)
    print(f"  Gate:      confidence >= {QUALITY_GATE}  AND  support_delta <= 0",
          flush=True)
    print(f"  Trigger:   p_truth >= {P_TRUTH_TRIGGER}  "
          f"(strong >= {P_TRUTH_STRONG} also applies to majority)",
          flush=True)
    print(f"  Tokens:    {BASE_TOKENS} for ALL agents — no extra budget",
          flush=True)
    print(f"  Resume:    automatic (checkpoint per question)", flush=True)
    print("=" * 65, flush=True)

    # Load checkpoint (empty if fresh start)
    debate_rows, state_rows, all_interventions, done_nos = load_checkpoint(
        args.out_dir, args.dataset_label, args.seed)

    predictor = PtruthPredictor(args.predictor)
    questions = load_questions(args.base_workbook)
    if args.limit:
        questions = questions[:args.limit]

    if args.backend == "ollama":
        llm = OllamaQwenPipeline(
            model_id=args.model_id, host=args.ollama_host,
            temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=BASE_TOKENS)
    else:
        llm = LocalQwenPipeline(
            model_id=args.model_id, temperature=args.temperature,
            top_p=args.top_p, max_new_tokens=BASE_TOKENS,
            device_map="auto", torch_dtype="auto",
            require_gpu=not args.no_require_gpu)

    for q_idx, question in enumerate(questions):
        q_rng = random.Random(args.seed * 100_000 + q_idx)

        if question.question_no in done_nos:
            print(f"Q {q_idx+1}/{len(questions)}: {question.question_no} "
                  f"[skip — checkpoint]", flush=True)
            continue

        print(f"\nQ {q_idx+1}/{len(questions)}: {question.question_no} "
              f"[{question.category}]  correct={question.correct_answer}",
              flush=True)

        try:
            debate_row, q_state, intv_log = run_one_debate(
                llm=llm, question=question, predictor=predictor,
                seed=args.seed + q_idx * 1000,
                rounds=args.rounds, sleep=args.sleep, rng=q_rng,
                dataset_label=args.dataset_label,
                exp_seed=args.seed,
                run_id=q_idx + 1,
            )
        except Exception as e:
            print(f"  ERROR on {question.question_no}: {e}", flush=True)
            traceback.print_exc()
            continue   # skip this question, continue with next

        if debate_row is None:
            continue

        correct = (debate_row["Final Answer"] == question.correct_answer)
        mark    = "OK" if correct else "WRONG"
        print(f"  -> Final={debate_row['Final Answer']}  "
              f"Correct={question.correct_answer}  "
              f"[{mark}]  interventions={len(intv_log)}", flush=True)

        debate_rows.append(debate_row)
        state_rows.extend(q_state)
        all_interventions.extend(intv_log)

        # Save immediately — resume-safe
        save_checkpoint(args.out_dir, args.dataset_label, args.seed,
                        question.question_no, debate_row, q_state, intv_log)

    if not debate_rows:
        print("No debates completed.", flush=True)
        return

    # ── Summary ─────────────────────────────────────────────────────
    debates = pd.DataFrame(debate_rows)
    debates["Correct?"] = debates.apply(final_answer_correctness, axis=1)
    states  = pd.DataFrame(state_rows)
    df_intv = pd.DataFrame(all_interventions) if all_interventions else pd.DataFrame()

    print("\n" + "=" * 65, flush=True)
    print("EXPERIMENT A — RESULTS SUMMARY", flush=True)
    print("=" * 65, flush=True)
    acc  = (debates["Correct?"] == "Yes").mean()
    cons = debates[debates["Final Answer Source"] == "agent_consensus"]
    cr   = (cons["Correct?"] == "Yes").mean() if len(cons) > 0 else 0.0
    print(f"  Debates:               {len(debates)}", flush=True)
    print(f"  Accuracy:              {acc:.1%}", flush=True)
    print(f"  Consensus rate:        {len(cons)/len(debates):.1%}", flush=True)
    print(f"  Consensus reliability: {cr:.1%}", flush=True)
    if not df_intv.empty:
        print(f"  Total interventions:   {len(df_intv)}", flush=True)
        print(f"  Questions intervened:  "
              f"{df_intv['question_no'].nunique()}", flush=True)
        print("  Pathology breakdown:", flush=True)
        for pat, cnt in df_intv["pathology"].value_counts().items():
            print(f"    {pat:<25}: {cnt}", flush=True)
    else:
        print("  Total interventions:   0", flush=True)

    # ── Judge ────────────────────────────────────────────────────────
    source_file = output_path.name
    if not args.skip_judging and args.q_source == "llm":
        print("\nRunning LLM judging...", flush=True)
        try:
            judgments = judge_debates_with_qwen(
                llm, debates, source_file=source_file,
                seed=args.seed + 100_000,
                judge_max_new_tokens=220, judge_batch_size=15,
                sleep=args.sleep)
        except Exception as e:
            print(f"Judging failed: {e}. Saving without judgments.", flush=True)
            judgments = empty_judgments()
    else:
        judgments = empty_judgments()

    if not judgments.empty:
        judgments["source_file"] = source_file
        scores = score_mixed_debates(
            debates, judgments, source_file,
            q_source=args.q_source, metric_version="paper")
    else:
        scores = pd.DataFrame()

    # ── Write workbook ───────────────────────────────────────────────
    import argparse as _ap
    run_args = _ap.Namespace(
        backend="ollama", seed=args.seed, model_id=args.model_id,
        temperature=args.temperature, top_p=args.top_p,
        objective_limit=len(debates), rounds=args.rounds,
        q_source=args.q_source, metric_version="paper",
        skip_judging=args.skip_judging,
    )
    write_qwen_excel_report(
        output_path, "exp_a_adaptive_controller",
        run_args, debates, judgments, scores)

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        states.to_excel(writer, sheet_name="Round_State", index=False)
        if not df_intv.empty:
            df_intv.to_excel(writer, sheet_name="Intervention_Log", index=False)

    print(f"\nSaved: {output_path}", flush=True)
    print("Checkpoint file retained (delete manually if no longer needed).",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
