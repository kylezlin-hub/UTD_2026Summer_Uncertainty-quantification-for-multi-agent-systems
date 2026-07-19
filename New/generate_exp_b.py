"""
generate_exp_b.py  —  Proactive Consensus Prevention Controller (Experiment B)

Motivation from Exp A results on GPQA:
  Exp A (reactive minority protection) fired on only 10-13 questions and showed
  no improvement over baseline. The primary GPQA failure mode is COLLECTIVE
  CAPITULATION — all 3 agents converge on the wrong answer from round 1-2,
  leaving no minority to protect.

Exp B design — TWO MODES run at every round:

  Mode A (inherited from Exp A):
    Triggered when a minority exists with quality >= 0.8 and support_delta <= 0.
    Protects the minority via p_truth predictor.  Same as Exp A.

  Mode B (new — proactive diversity injection):
    Triggered when convergence is detected as PREMATURE, even with no minority.
    Three conditions (any one fires):
      1. Early consensus   : all 3 agents agree AND round <= 3
      2. Low balance       : prefix_balance < 0.40
      3. Rapid convergence : prefix_balance < 0.60 AND prefix_engagement < 0.05
    Action: apply devil's advocate or diversity-injection prompt to ALL agents.
    No minority required. Challenges the forming majority before it locks in.

Key difference:
  Exp A: "Protect the dissenter"   (reactive, minority-gated)
  Exp B: "Challenge the conformist" (proactive, fires even on unanimous debates)

Output: exp_b_{dataset_label}_s{seed}.xlsx  — same 53-column schema as Exp A.
        Adds 2 extra columns: mode_b_applied, mode_b_reason.

Usage:
    python New/generate_exp_b.py \\
        --base-workbook New/baseline_v2_gpqa_s7.xlsx \\
        --predictor New/models_p_truth \\
        --out-dir New/exp_b_results \\
        --dataset-label gpqa --seed 7 \\
        --backend transformers --model-id Qwen/Qwen2.5-14B-Instruct
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
# Mode A (inherited from Exp A)
QUALITY_GATE      = 0.80
P_TRUTH_TRIGGER   = 0.50
P_TRUTH_STRONG    = 0.60

# Mode B thresholds (proactive diversity)
MODE_B_EARLY_CONSENSUS_MAX_ROUND = 3     # unanimous by this round → fire
MODE_B_LOW_BALANCE                = 0.40  # balance below this → fire
MODE_B_RAPID_BALANCE              = 0.60  # balance below this AND eng < threshold
MODE_B_RAPID_ENGAGEMENT           = 0.05  # engagement below this for rapid check

BASE_TOKENS = 220    # same for ALL agents — no extra budget


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Checkpoint helpers (identical to Exp A)
# ═══════════════════════════════════════════════════════════════════════════

def _cp_path(out_dir, label, seed):
    return out_dir / f"exp_b_{label}_s{seed}.checkpoint.jsonl"


def load_checkpoint(out_dir, label, seed):
    cp = _cp_path(out_dir, label, seed)
    debate_by_qno, state_by_qno, intv_by_qno = {}, {}, {}
    if not cp.exists():
        return [], [], [], set()
    with cp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            qno, t = e.get("question_no",""), e.get("_type","")
            if   t == "debate":      debate_by_qno[qno] = e["data"]
            elif t == "state":       state_by_qno[qno]  = e["data"]
            elif t == "intervention":intv_by_qno[qno]   = e["data"]
    done    = set(debate_by_qno.keys())
    debates = [debate_by_qno[k] for k in debate_by_qno]
    states  = [r for k in debate_by_qno for r in state_by_qno.get(k, [])]
    intvs   = [r for k in debate_by_qno for r in intv_by_qno.get(k, [])]
    if done:
        print(f"Checkpoint: {len(done)} questions already completed.", flush=True)
    return debates, states, intvs, done


def save_checkpoint(out_dir, label, seed, qno, debate_row, state_rows, intv_rows):
    cp = _cp_path(out_dir, label, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    l1 = json.dumps({"_type":"debate",       "question_no":qno, "data":debate_row})
    l2 = json.dumps({"_type":"state",        "question_no":qno, "data":state_rows})
    l3 = json.dumps({"_type":"intervention", "question_no":qno, "data":intv_rows})
    with cp.open("a", encoding="utf-8") as f:
        f.write(l1+"\n"+l2+"\n"+l3+"\n")


def clear_checkpoint(out_dir, label, seed):
    cp = _cp_path(out_dir, label, seed)
    if cp.exists():
        cp.unlink()
        print(f"Cleared checkpoint: {cp.name}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Predictor (identical to Exp A)
# ═══════════════════════════════════════════════════════════════════════════

class PtruthPredictor:
    def __init__(self, model_dir: Path):
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.model  = joblib.load(model_dir / "p_truth_logistic.pkl")
        with open(model_dir / "feature_meta.json") as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        print(f"Loaded p_truth predictor ({len(self.feature_names)} features, "
              f"CV AUC={meta['best_auc_cv']:.3f})", flush=True)

    def predict(self, diag, agent_data, round_num) -> float:
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
            "quality_x_support_delta":         q * sd_c,
            "support_loss_rate":                sd / (rr + 1),
            "support_delta":                    sd,
            "judge_explanation_good":           q,
            "defector_ratio":                   nd / (ms + 1),
            "num_defectors":                    nd,
            "all_judge_scores":                 q,
            "was_majority_x_delta":             wmb * sd_c,
            "confidence_x_quality":             conf * q,
            "independent_reasoning_x_quality":  q * q,
            "quality_x_defectors":              q * (-nd),
            "left_majority":                    lm,
            "isolating":                        float(sd < 0 and nd > 0),
            "quality_x_rounds":                 q * rr,
            "prefix_attributed_influence":      pai,
        }
        x = np.array([[feat.get(f, 0.0) for f in self.feature_names]])
        try:
            return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])
        except Exception:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Prefix diagnostics (identical to Exp A)
# ═══════════════════════════════════════════════════════════════════════════

def _entropy(vals):
    v = [x for x in vals if x and str(x) != "nan"]
    if len(v) <= 1:
        return 0.0
    counts = np.array(list(Counter(v).values()), dtype=float)
    p = counts / counts.sum()
    return float(-sum(pi * log(pi) for pi in p if pi > 0) / log(len(v)))


def _safe(v, default=float("nan")):
    if v is None:
        return default
    try:
        f = float(v)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def compute_prefix_diagnostics(answers_by_round, quality_by_round, up_to_round):
    answers = answers_by_round[:up_to_round]
    quality = quality_by_round[:up_to_round]
    n = len(answers[0]) if answers else len(QWEN_AGENTS)
    base = {
        "prefix_engagement": float("nan"), "prefix_responsiveness": float("nan"),
        "prefix_influence_asymmetry": 0.0,  "prefix_balance": float("nan"),
        "prefix_dispersion": _entropy(answers[-1]) if answers else 0.0,
        "prefix_influence_per_agent": [0.0] * n,
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
    total = sum(pos); mx = max(pos) if pos else 0.0
    collapse  = (1.0 - mx / (total + EPS)) if total > EPS else (
        1.0 if max((abs(s) for s in steps), default=0) <= EPS else 0.0)
    revs = sum(steps[i-1]*steps[i] < 0 for i in range(1, len(steps)))
    vol  = 1.0 if len(dispersion) <= 2 else 1.0 - revs / (len(dispersion) - 2)
    base["prefix_balance"]        = float(np.clip(collapse * vol, 0.0, 1.0))
    base["prefix_dispersion"]     = dispersion[-1]
    base["prefix_engagement"]     = float(np.mean(eng))  if eng  else float("nan")
    base["prefix_responsiveness"] = float(np.mean(resp)) if resp else float("nan")
    return base


def detect_pathology(diag):
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
    worst = max(scores, key=scores.get)
    return worst, scores[worst]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Gate (identical to Exp A)
# ═══════════════════════════════════════════════════════════════════════════

def passes_gate(quality_proxy, support_delta):
    return quality_proxy >= QUALITY_GATE and support_delta <= 0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Mode A prompts (inherited from Exp A — unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def minority_prompt(pathology, p_truth):
    base = (
        f"\n\n[Adaptive Controller — Minority Protection "
        f"(estimated correctness: {p_truth:.0%})]\n"
    )
    additions = {
        "domination":
            "High influence concentration detected. Re-examine your reasoning "
            "independently. Address the strongest counterargument with explicit evidence.",
        "capitulation":
            "Rapid convergence detected. Identify one piece of evidence that would "
            "need to be wrong for your answer to be incorrect. If sound, defend it.",
        "dogmatism":
            "Revisit from first principles. State reasoning from scratch, not from "
            "your prior response.",
        "parallel_reasoning":
            "Directly address what peers argued. State explicitly whether each "
            "argument changes your view and why.",
        "over_engagement":
            "Before updating again, identify the specific new evidence justifying "
            "each change. Return to your most defensible position if unsure.",
    }
    return base + additions.get(pathology, "Carefully re-examine your position.")


def majority_prompt(pathology, majority_ans, minority_explanation):
    if pathology == "domination":
        return (
            f"\n\n[Adaptive Controller — Structured Rebuttal]\n"
            f"A peer holds a different position:\n"
            f'"{minority_explanation[:250]}"\n'
            f"Before confirming your answer:\n"
            f"1. Restate their argument in your own words\n"
            f"2. Identify the specific evidence that defeats it\n"
            f"3. Only then give your updated answer"
        )
    elif pathology == "capitulation":
        return (
            f"\n\n[Adaptive Controller — Devil's Advocate]\n"
            f"The group is converging on: {majority_ans}\n"
            f"Before committing, identify the strongest argument AGAINST "
            f"{majority_ans}. What evidence would disprove it? "
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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Mode B prompts (NEW — proactive diversity injection)
# ═══════════════════════════════════════════════════════════════════════════

def devil_advocate_prompt(majority_ans: str) -> str:
    """Applied to ALL agents when unanimous early consensus is detected."""
    return (
        f"\n\n[Adaptive Controller — Devil's Advocate (Proactive)]\n"
        f"All agents currently agree on: {majority_ans}\n\n"
        f"Before confirming, you MUST argue the other side:\n"
        f"1. What is the strongest argument AGAINST {majority_ans}?\n"
        f"2. Which alternative answer deserves the most serious consideration?\n"
        f"3. Would a domain expert challenge this consensus? Why?\n\n"
        f"After this reflection, give your honest best answer."
    )


def diversity_injection_prompt(bal: float, eng: float) -> str:
    """Applied to ALL agents when balance is low but not unanimous."""
    return (
        f"\n\n[Adaptive Controller — Diversity Injection]\n"
        f"Debate diagnostics indicate rapid convergence "
        f"(balance={bal:.2f}, engagement={eng:.2f}).\n\n"
        f"Before updating your answer:\n"
        f"1. Is there an alternative explanation you have not fully considered?\n"
        f"2. Are you changing position based on argument quality, "
        f"or social alignment with peers?\n"
        f"3. What specific evidence in YOUR reasoning supports your current position?\n\n"
        f"Give your honest best answer after this reflection."
    )


def mode_b_trigger(diag: dict, n_distinct: int, round_no: int
                   ) -> tuple[bool, str, str]:
    """
    Determine if Mode B (proactive diversity) should fire.

    Returns (should_fire, reason_string, prompt_type)
    where prompt_type is 'devil_advocate' or 'diversity_injection'.

    Three conditions (any one fires):
      1. Early consensus:   all 3 agents agree AND round <= 3
      2. Low balance:       prefix_balance < 0.40
      3. Rapid convergence: prefix_balance < 0.60 AND prefix_engagement < 0.05
    """
    bal = _safe(diag.get("prefix_balance"), 1.0)
    eng = _safe(diag.get("prefix_engagement"), 0.1)

    # Condition 1: unanimous before round 3
    if n_distinct == 1 and round_no <= MODE_B_EARLY_CONSENSUS_MAX_ROUND:
        return True, f"early_consensus(r{round_no})", "devil_advocate"

    # Condition 2: balance collapsed
    if bal < MODE_B_LOW_BALANCE:
        return True, f"low_balance={bal:.2f}", "devil_advocate"

    # Condition 3: converging fast with no real deliberation
    if bal < MODE_B_RAPID_BALANCE and eng < MODE_B_RAPID_ENGAGEMENT:
        return True, f"rapid_collapse(bal={bal:.2f},eng={eng:.2f})", "diversity_injection"

    return False, "", ""


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Question loader
# ═══════════════════════════════════════════════════════════════════════════

def load_questions(path: Path) -> list:
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
            dataset_type="objective",
            question_no=str(row.get("Question #", "")),
            question=q_text,
            correct_answer=correct,
            answer_labels=labels,
            category=str(row.get("Dataset Category", "")),
        ))
    print(f"Loaded {len(questions)} questions from {path.name}", flush=True)
    return questions


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Core debate loop — Mode A + Mode B combined
# ═══════════════════════════════════════════════════════════════════════════

def run_one_debate(
    llm, question, predictor, seed, rounds, sleep, rng,
    dataset_label="gpqa", exp_seed=7, run_id=1,
):
    agent_order = QWEN_AGENTS[:]
    rng.shuffle(agent_order)

    all_rounds:     list[dict] = []
    quality_cache:  dict       = {}
    diag_by_round:  dict       = {}
    intervention_log: list     = []

    for round_no in range(1, rounds + 1):

        # ── 1. Build base prompts ────────────────────────────────────
        messages = {}
        for agent in agent_order:
            if round_no == 1:
                messages[agent] = qwen_initial_messages(question, agent)
            else:
                messages[agent] = qwen_update_messages(question, agent, all_rounds[-1])

        # ── 2. Adaptive controller (rounds 2+) ───────────────────────
        if round_no >= 2 and all_rounds:
            # Compute pre-round diagnostics for controller decisions
            a_hist = [[str(r[a].get("answer","") or "") or None for a in QWEN_AGENTS]
                      for r in all_rounds]
            q_hist = [[quality_cache.get(i, {}).get(a, 0.5) for a in QWEN_AGENTS]
                      for i in range(len(all_rounds))]
            diag = compute_prefix_diagnostics(a_hist, q_hist, len(all_rounds))
            pathology, severity = detect_pathology(diag)

            curr_answers  = {a: str(all_rounds[-1][a].get("answer","")) for a in QWEN_AGENTS}
            answer_counts = Counter(v for v in curr_answers.values() if v)
            if answer_counts:
                max_sup      = max(answer_counts.values())
                majority_ans = answer_counts.most_common(1)[0][0]
                n_distinct   = len(answer_counts)
                prev_counts  = Counter(str(all_rounds[-2][a].get("answer",""))
                                       for a in QWEN_AGENTS) if len(all_rounds) > 1 else Counter()

                # ── MODE B: proactive diversity injection ────────────
                mode_b_fire, mode_b_reason, mode_b_type = mode_b_trigger(
                    diag, n_distinct, round_no)

                if mode_b_fire:
                    if mode_b_type == "devil_advocate":
                        prompt_text = devil_advocate_prompt(majority_ans)
                    else:
                        bal = _safe(diag.get("prefix_balance"), 1.0)
                        eng = _safe(diag.get("prefix_engagement"), 0.1)
                        prompt_text = diversity_injection_prompt(bal, eng)

                    for agent in QWEN_AGENTS:
                        messages[agent][-1] = {
                            **messages[agent][-1],
                            "content": messages[agent][-1]["content"] + prompt_text
                        }

                    intervention_log.append({
                        "question_no":   question.question_no,
                        "round":         round_no,
                        "agent":         "ALL",
                        "mode":          "B",
                        "mode_b_type":   mode_b_type,
                        "mode_b_reason": mode_b_reason,
                        "n_distinct":    n_distinct,
                        "majority_ans":  majority_ans,
                        "pathology":     pathology,
                        "p_truth":       float("nan"),
                    })
                    print(f"  [MODE-B] r{round_no}: {mode_b_reason} "
                          f"-> {mode_b_type} applied to ALL agents", flush=True)

                # ── MODE A: minority protection (same as Exp A) ──────
                influenced = diag.get("prefix_influence_per_agent",
                                      [0.0] * len(QWEN_AGENTS))
                if not isinstance(influenced, list):
                    influenced = [0.0] * len(QWEN_AGENTS)
                minority_agents = [
                    a for a in QWEN_AGENTS
                    if curr_answers[a] and answer_counts[curr_answers[a]] < max_sup
                ]
                if minority_agents:
                    print(f"  [R{round_no}] pathology={pathology}(sev={severity:.3f}) "
                          f"minorities={minority_agents}", flush=True)

                for a_idx, agent in enumerate(QWEN_AGENTS):
                    curr_ans = curr_answers[agent]
                    if not curr_ans or answer_counts[curr_ans] >= max_sup:
                        continue
                    curr_sup = answer_counts[curr_ans]
                    prev_sup = prev_counts.get(curr_ans, 0)
                    prev_max = max(prev_counts.values()) if prev_counts else 0
                    wmb      = int(prev_sup >= prev_max and prev_max > 0)
                    nd       = sum(1 for aa in QWEN_AGENTS
                                   if len(all_rounds) > 1
                                   and str(all_rounds[-2][aa].get("answer","")) == curr_ans
                                   and curr_answers[aa] != curr_ans)
                    prev_maj = prev_counts.most_common(1)[0][0] if prev_counts else ""
                    lm       = int(len(all_rounds) > 1
                                   and str(all_rounds[-2][agent].get("answer","")) == prev_maj
                                   and curr_ans != prev_maj)
                    conf_raw = all_rounds[-1][agent].get("confidence", "0.5")
                    conf     = _safe(parse_confidence(conf_raw), 0.5)
                    q_proxy  = quality_cache.get(len(all_rounds)-1, {}).get(agent, conf)
                    q_proxy  = _safe(q_proxy, conf)
                    if q_proxy == 0.5 and conf > 0.5:
                        q_proxy = conf
                    delta    = curr_sup - prev_sup
                    pai      = influenced[a_idx] if a_idx < len(influenced) else 0.0

                    gate_ok = passes_gate(q_proxy, delta)
                    print(f"    [GATE-A] r{round_no} {agent}: "
                          f"quality={q_proxy:.2f}({'OK' if q_proxy>=QUALITY_GATE else 'FAIL'}) "
                          f"delta={delta:+d}({'OK' if delta<=0 else 'FAIL'}) "
                          f"-> {'PASS' if gate_ok else 'SKIP'}", flush=True)
                    if not gate_ok:
                        continue

                    agent_data = {
                        "quality_proxy": q_proxy, "support_delta": delta,
                        "num_defectors": nd, "minority_size": curr_sup,
                        "support": curr_sup, "was_majority_before": wmb,
                        "left_majority": lm, "confidence": conf,
                        "prefix_attributed_influence": pai,
                    }
                    p_truth = predictor.predict(diag, agent_data, round_no)
                    print(f"    [PRED-A] r{round_no} {agent}: p_truth={p_truth:.3f} "
                          f"({'TRIGGER' if p_truth>=P_TRUTH_TRIGGER else 'below'})",
                          flush=True)
                    if p_truth < P_TRUTH_TRIGGER:
                        continue

                    min_expl = str(all_rounds[-1][agent].get("response",""))[:250]
                    messages[agent][-1] = {
                        **messages[agent][-1],
                        "content": messages[agent][-1]["content"] +
                                   minority_prompt(pathology, p_truth)
                    }
                    if p_truth >= P_TRUTH_STRONG:
                        for maj_agent in QWEN_AGENTS:
                            if curr_answers[maj_agent] == majority_ans:
                                messages[maj_agent][-1] = {
                                    **messages[maj_agent][-1],
                                    "content": messages[maj_agent][-1]["content"] +
                                               majority_prompt(pathology, majority_ans, min_expl)
                                }
                    intervention_log.append({
                        "question_no":   question.question_no,
                        "round":         round_no,
                        "agent":         agent,
                        "mode":          "A",
                        "mode_b_type":   "",
                        "mode_b_reason": "",
                        "n_distinct":    n_distinct,
                        "majority_ans":  majority_ans,
                        "pathology":     pathology,
                        "p_truth":       round(p_truth, 4),
                    })
                    print(f"    [MODE-A] r{round_no} {agent}: p_truth={p_truth:.2f} "
                          f"-> INTERVENTION APPLIED", flush=True)

        # ── 3. Generate responses ────────────────────────────────────
        round_turns = {}
        discard_reason = ""
        for agent in agent_order:
            base_seed = seed + round_no * 100 + QWEN_AGENTS.index(agent) * 10
            try:
                raw = llm.complete(messages[agent], seed=base_seed,
                                   max_new_tokens=BASE_TOKENS)
            except Exception as e:
                discard_reason = f"LLM error r{round_no} {agent}: {e}"
                break
            parsed = parse_qwen_turn(raw, question.dataset_type,
                                     question.answer_labels, strict=True)
            if parsed["parse_failed"]:
                try:
                    retry = qwen_reprompt_messages(messages[agent], question)
                    raw2  = llm.complete(retry, seed=base_seed+1,
                                         max_new_tokens=BASE_TOKENS)
                except Exception as e:
                    discard_reason = f"LLM retry error r{round_no} {agent}: {e}"
                    break
                p2 = parse_qwen_turn(raw2, question.dataset_type,
                                     question.answer_labels, strict=True)
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

        # Cache confidence + post-round diagnostics for Round_State
        quality_cache[round_no - 1] = {
            a: _safe(parse_confidence(round_turns[a].get("confidence","")), 0.5)
            for a in QWEN_AGENTS
        }
        all_rounds.append(round_turns)

        # Post-round diagnostics (inclusive, for Round_State schema match)
        pa = [[str(r[a].get("answer","") or "") or None for a in QWEN_AGENTS]
              for r in all_rounds]
        pq = [[quality_cache.get(i, {}).get(a, 0.5) for a in QWEN_AGENTS]
              for i in range(len(all_rounds))]
        dp = compute_prefix_diagnostics(pa, pq, len(all_rounds))
        inf_l = dp.get("prefix_influence_per_agent", [0.0]*len(QWEN_AGENTS))
        ti    = sum(inf_l)
        dp["prefix_influence_share"] = ([v/ti for v in inf_l] if ti > EPS
                                        else [float("nan")]*len(QWEN_AGENTS))
        diag_by_round[round_no] = dp

    # ── 4. Final answer ──────────────────────────────────────────────
    final_ans_list = [all_rounds[-1][a]["answer"] for a in QWEN_AGENTS]
    valid = [a for a in final_ans_list if a]
    if valid and len(set(valid)) == 1:
        final_answer, final_source = valid[0], "agent_consensus"
    else:
        try:
            raw = llm.complete(qwen_moderator_messages(question, all_rounds),
                               seed=seed+rounds*100+999)
            parsed = parse_qwen_turn(raw, question.dataset_type,
                                     question.answer_labels, strict=False)
            candidate = parsed["answer"]
            if answer_is_valid(candidate, question):
                final_answer, final_source = candidate, "moderator"
            else:
                final_answer = majority_answer(final_ans_list)
                final_source = "majority_vote_no_moderator"
        except Exception:
            final_answer = majority_answer(final_ans_list)
            final_source = "majority_vote_no_moderator"

    # ── 5. Build debate row ──────────────────────────────────────────
    debate_row = {
        "Question #": question.question_no, "Dataset Type": question.dataset_type,
        "Dataset Category": question.category, "Question": question.question,
        "Correct Answer": question.correct_answer, "Final Answer": final_answer,
        "Final Answer Source": final_source, "Fixture Pattern": "exp_b_proactive",
        "Rounds to Consensus": first_consensus_round_for_answer(all_rounds, final_answer),
    }
    for rn, turns in enumerate(all_rounds, start=1):
        for agent in QWEN_AGENTS:
            t = turns[agent]
            debate_row[f"R{rn} {agent} Answer"]   = t["answer"]
            debate_row[f"R{rn} {agent} Conf"]     = t.get("confidence","")
            debate_row[f"R{rn} {agent} Response"] = t.get("response","")

    # mode_b map for Round_State
    mode_b_by_round = {e["round"]: e for e in intervention_log if e.get("mode") == "B"}

    state_rows = _build_round_state(
        all_rounds, question, final_answer, intervention_log,
        diag_by_round=diag_by_round,
        dataset_label=dataset_label, seed=exp_seed, run_id=run_id,
        mode_b_by_round=mode_b_by_round)

    return debate_row, state_rows, intervention_log


def _build_round_state(all_rounds, question, final_answer, intervention_log,
                       diag_by_round=None, dataset_label="gpqa", seed=7,
                       run_id=1, mode_b_by_round=None):
    """49 baseline + 4 ExpA + 2 ExpB = 55 columns total."""
    rows = []
    final_correct = (final_answer == question.correct_answer)
    intv_by_ra    = {(e["round"], e["agent"]): e for e in intervention_log
                     if e.get("mode") == "A"}
    mode_b_by_round = mode_b_by_round or {}
    diag_by_round   = diag_by_round or {}

    for round_idx, turns in enumerate(all_rounds):
        round_num  = round_idx + 1
        next_r     = all_rounds[round_idx+1] if round_idx+1 < len(all_rounds) else None
        final_r    = all_rounds[-1]
        curr  = {a: str(turns[a].get("answer","")) for a in QWEN_AGENTS}
        prev  = ({a: str(all_rounds[round_idx-1][a].get("answer","")) for a in QWEN_AGENTS}
                 if round_idx > 0 else {a:"" for a in QWEN_AGENTS})
        cnts  = Counter(v for v in curr.values() if v)
        max_s = max(cnts.values()) if cnts else 0
        maj   = cnts.most_common(1)[0][0] if cnts else ""
        prev_cnts = (Counter(v for v in prev.values() if v) if round_idx > 0 else Counter())
        prev_max  = max(prev_cnts.values()) if prev_cnts else 0
        prev_maj  = prev_cnts.most_common(1)[0][0] if prev_cnts else ""
        diag      = diag_by_round.get(round_num, {})
        pinfs     = diag.get("prefix_influence_per_agent", [0.0]*len(QWEN_AGENTS))
        if not isinstance(pinfs, list): pinfs = [0.0]*len(QWEN_AGENTS)
        pshares   = diag.get("prefix_influence_share", [float("nan")]*len(QWEN_AGENTS))
        if not isinstance(pshares, list): pshares = [float("nan")]*len(QWEN_AGENTS)
        mb_event  = mode_b_by_round.get(round_num, {})

        for a_idx, agent in enumerate(QWEN_AGENTS):
            ans   = curr[agent]; prev_a = prev[agent]
            sup   = cnts.get(ans, 0); prev_sup = prev_cnts.get(ans, 0)
            intv  = intv_by_ra.get((round_num, agent), {})
            turn  = turns[agent]
            next_ans  = str(next_r[agent].get("answer","")) if next_r else ""
            final_ans = str(final_r[agent].get("answer",""))
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

            rows.append({
                # ── 49 baseline columns ──────────────────────────────
                "run_id": run_id, "dataset": dataset_label, "seed": seed,
                "question_no": question.question_no, "correct_answer": question.correct_answer,
                "final_answer": final_answer, "final_answer_correct": final_correct,
                "round": round_num, "agent": agent, "answer": ans,
                "confidence": _safe(parse_confidence(turn.get("confidence","")), 0.5),
                "explanation_text": str(turn.get("response","")),
                "response_tokens": len(str(turn.get("response","")).split()),
                "re_prompted": bool(turn.get("re_prompted", False)),
                "prev_answer": prev_a, "answer_changed": bool(prev_a and ans != prev_a),
                "joined_majority": jnd, "left_majority": lft,
                "is_minority": bool(0 < sup < max_s), "support": sup,
                "support_delta": sup - prev_sup,
                "minority_size": sup if 0 < sup < max_s else 0,
                "was_majority_before": wmb, "num_defectors": nd, "num_joiners": nj,
                "majority_answer": maj, "majority_support": max_s,
                "n_distinct_answers": len(cnts), "consensus_reached": len(cnts) == 1,
                "prefix_engagement":          _safe(diag.get("prefix_engagement")),
                "prefix_responsiveness":      _safe(diag.get("prefix_responsiveness")),
                "prefix_influence_asymmetry": _safe(diag.get("prefix_influence_asymmetry"),0.0),
                "prefix_balance":             _safe(diag.get("prefix_balance")),
                "prefix_dispersion":          _safe(diag.get("prefix_dispersion"), 0.0),
                "prefix_influence_share":     _safe(pshares[a_idx] if a_idx<len(pshares) else float("nan")),
                "prefix_attributed_influence":_safe(pinfs[a_idx] if a_idx<len(pinfs) else 0.0, 0.0),
                "answer_at_next_round": next_ans, "drops_next_round": drops,
                "answer_at_final": final_ans, "answer_survives_to_final": survives,
                "is_correct": is_cor,
                "is_correct_minority": bool(is_cor and 0 < sup < max_s),
                "correct_drops_next": bool(is_cor and drops),
                "correct_survives_final": bool(is_cor and survives),
                "judge_explanation_good": float("nan"),
                "judge_uses_past_reasoning": float("nan"),
                "judge_justifies_stance": float("nan"),
                "judge_independent_reasoning": float("nan"),
                "judge_parse_failed": False,
                # ── 4 Exp A columns ──────────────────────────────────
                "intervention_applied":  bool(intv),
                "pathology_detected":    intv.get("pathology",""),
                "p_truth_score":         intv.get("p_truth", float("nan")),
                "applied_to_majority":   intv.get("applied_to_majority", False),
                # ── 2 new Exp B columns ──────────────────────────────
                "mode_b_applied": bool(mb_event),
                "mode_b_reason":  mb_event.get("mode_b_reason",""),
            })
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: GPU detection + main
# ═══════════════════════════════════════════════════════════════════════════

def _detect_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i)
                     for i in range(torch.cuda.device_count())]
            return f"GPU AVAILABLE: {', '.join(names)}"
        return "NO CUDA GPU detected — will run on CPU (very slow for 14B)"
    except ImportError:
        return "PyTorch not installed"


def parse_args():
    p = argparse.ArgumentParser(description="Experiment B — Proactive Consensus Prevention")
    p.add_argument("--base-workbook",  type=Path, default=Path("New/baseline_v2_gpqa_s7.xlsx"))
    p.add_argument("--predictor",      type=Path, default=Path("New/models_p_truth"))
    p.add_argument("--out-dir",        type=Path, default=Path("New/exp_b_results"))
    p.add_argument("--dataset-label",  default="gpqa")
    p.add_argument("--seed",           type=int, default=7)
    p.add_argument("--rounds",         type=int, default=5)
    p.add_argument("--model-id",       default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--backend",        choices=["ollama","transformers"], default="transformers")
    p.add_argument("--ollama-host",    default="http://127.0.0.1:11434")
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top-p",          type=float, default=0.9)
    p.add_argument("--sleep",          type=float, default=0.0)
    p.add_argument("--limit",          type=int,   default=None)
    p.add_argument("--skip-judging",   action="store_true")
    p.add_argument("--q-source",       choices=["llm","confidence"], default="llm")
    p.add_argument("--no-require-gpu", action="store_true")
    p.add_argument("--overwrite",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / f"exp_b_{args.dataset_label}_s{args.seed}.xlsx"

    if args.overwrite:
        clear_checkpoint(args.out_dir, args.dataset_label, args.seed)
        if output_path.exists():
            output_path.unlink()

    if output_path.exists():
        try:
            if {"Debate_Traces","Round_State"}.issubset(
                    set(pd.ExcelFile(output_path).sheet_names)):
                print(f"Already complete: {output_path}. Use --overwrite.", flush=True)
                return
        except Exception:
            pass

    print("=" * 65, flush=True)
    print("EXPERIMENT B — Proactive Consensus Prevention Controller", flush=True)
    print(f"  Dataset:   {args.dataset_label}  seed={args.seed}", flush=True)
    print(f"  Model:     {args.model_id}  backend={args.backend}", flush=True)
    print(f"  {_detect_gpu()}", flush=True)
    print(f"  Mode A:    minority protection (quality>={QUALITY_GATE}, p_truth>={P_TRUTH_TRIGGER})", flush=True)
    print(f"  Mode B:    proactive diversity  (balance<{MODE_B_LOW_BALANCE} OR "
          f"early consensus by r{MODE_B_EARLY_CONSENSUS_MAX_ROUND})", flush=True)
    print(f"  Tokens:    {BASE_TOKENS} for ALL agents — no extra budget", flush=True)
    print(f"  Resume:    automatic (checkpoint per question)", flush=True)
    print("=" * 65, flush=True)

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
            print(f"Q {q_idx+1}/{len(questions)}: {question.question_no} [skip]",
                  flush=True)
            continue

        print(f"\nQ {q_idx+1}/{len(questions)}: {question.question_no} "
              f"[{question.category}]  correct={question.correct_answer}", flush=True)

        try:
            debate_row, q_state, intv_log = run_one_debate(
                llm=llm, question=question, predictor=predictor,
                seed=args.seed + q_idx * 1000,
                rounds=args.rounds, sleep=args.sleep, rng=q_rng,
                dataset_label=args.dataset_label,
                exp_seed=args.seed, run_id=q_idx + 1,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
            continue

        if debate_row is None:
            continue

        correct = (debate_row["Final Answer"] == question.correct_answer)
        mode_a_n = sum(1 for e in intv_log if e.get("mode") == "A")
        mode_b_n = sum(1 for e in intv_log if e.get("mode") == "B")
        print(f"  -> Final={debate_row['Final Answer']}  "
              f"Correct={question.correct_answer}  "
              f"[{'OK' if correct else 'WRONG'}]  "
              f"Mode-A={mode_a_n}  Mode-B={mode_b_n}", flush=True)

        debate_rows.append(debate_row)
        state_rows.extend(q_state)
        all_interventions.extend(intv_log)
        save_checkpoint(args.out_dir, args.dataset_label, args.seed,
                        question.question_no, debate_row, q_state, intv_log)

    if not debate_rows:
        print("No debates completed.", flush=True)
        return

    debates = pd.DataFrame(debate_rows)
    debates["Correct?"] = debates.apply(final_answer_correctness, axis=1)
    states  = pd.DataFrame(state_rows)
    df_intv = pd.DataFrame(all_interventions) if all_interventions else pd.DataFrame()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65, flush=True)
    print("EXPERIMENT B — RESULTS SUMMARY", flush=True)
    print("=" * 65, flush=True)
    acc  = (debates["Correct?"] == "Yes").mean()
    cons = debates[debates["Final Answer Source"] == "agent_consensus"]
    cr   = (cons["Correct?"] == "Yes").mean() if len(cons) > 0 else 0.0
    print(f"  Debates:                {len(debates)}", flush=True)
    print(f"  Accuracy:               {acc:.1%}", flush=True)
    print(f"  Consensus rate:         {len(cons)/len(debates):.1%}", flush=True)
    print(f"  Consensus reliability:  {cr:.1%}", flush=True)
    if not df_intv.empty:
        mode_a_total = (df_intv["mode"] == "A").sum()
        mode_b_total = (df_intv["mode"] == "B").sum()
        print(f"  Mode A interventions:   {mode_a_total}", flush=True)
        print(f"  Mode B interventions:   {mode_b_total}", flush=True)
        if mode_b_total > 0:
            print("  Mode B type breakdown:", flush=True)
            for bt, cnt in df_intv[df_intv["mode"]=="B"]["mode_b_type"].value_counts().items():
                print(f"    {bt:<25}: {cnt}", flush=True)

    # ── Judge ─────────────────────────────────────────────────────────
    source_file = output_path.name
    if not args.skip_judging and args.q_source == "llm":
        print("\nRunning LLM judging...", flush=True)
        try:
            judgments = judge_debates_with_qwen(
                llm, debates, source_file=source_file,
                seed=args.seed+100_000, judge_max_new_tokens=220,
                judge_batch_size=15, sleep=args.sleep)
        except Exception as e:
            print(f"Judging failed: {e}", flush=True)
            judgments = empty_judgments()
    else:
        judgments = empty_judgments()

    if not judgments.empty:
        judgments["source_file"] = source_file
        scores = score_mixed_debates(debates, judgments, source_file,
                                     q_source=args.q_source, metric_version="paper")
    else:
        scores = pd.DataFrame()

    # ── Write workbook ───────────────────────────────────────────────
    run_args = argparse.Namespace(
        backend=args.backend, seed=args.seed, model_id=args.model_id,
        temperature=args.temperature, top_p=args.top_p,
        objective_limit=len(debates), rounds=args.rounds,
        q_source=args.q_source, metric_version="paper",
        skip_judging=args.skip_judging,
    )
    write_qwen_excel_report(output_path, "exp_b_proactive_controller",
                            run_args, debates, judgments, scores)
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        states.to_excel(writer, sheet_name="Round_State", index=False)
        if not df_intv.empty:
            df_intv.to_excel(writer, sheet_name="Intervention_Log", index=False)

    print(f"\nSaved: {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run to resume.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
