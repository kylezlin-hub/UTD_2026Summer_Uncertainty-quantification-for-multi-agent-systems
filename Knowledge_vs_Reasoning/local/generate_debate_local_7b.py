"""generate_debate_local_7b.py -- Same-model (Qwen2.5-7B) NEUTRAL SYMMETRIC 3-agent debate.

Companion to ../generate_debate_roles_7b.py, but with NO epistemic roles and NO judge round.
All three agents receive the *identical* instruction (neutral, symmetric). The point is to observe
how a single model, sampled three times, revises (or holds) its answers under exposure to peers --
and to *categorise why* each revision happens.

Protocol (identical for every agent)
------------------------------------
  * No social / deferential language ("I agree with Agent B", etc.). Refer only to facts + reasoning.
  * Round 1 (== "Round 0" in the spec): independently solve. State answer, confidence, concise
    justification. Do NOT consider what other agents might think.
  * Rounds 2..T: read the other agents' RESPONSES (their confidence is HIDDEN). Revise the answer
    only if another agent presents evidence/reasoning more convincing than one's own. Explain why the
    answer changed or was retained, and label the reason with one of a fixed taxonomy.

Per-round change accounting (rounds 2..T)
-----------------------------------------
For each debate round after the initial one we record, per agent:
  * answer_changed          -- objective: did the answer letter change vs the previous round?
  * self_reported_changed   -- the agent's own Changed: yes/no
  * reason_for_change        -- one of the taxonomy codes below (objective no-change => no_change)
  * reason_raw               -- the agent's verbatim Reason: line (nothing discarded)
and, per round, the group pattern from the objective changer count:
  * KKK = no one changed, CKK = one changed, CCK = two changed, CCC = all three changed.

Reason taxonomy (Reason: field is mapped to these codes)
  fact_recall        -- "another agent recalled a fact I missed"
  stronger_logic     -- "another agent's logical reasoning is stronger"
  self_inconsistency -- "I discovered an inconsistency in my own reasoning"
  misunderstood      -- "I misunderstood the question"
  more_plausible     -- "I simply find another answer more plausible"
  no_change          -- "no change"
  unspecified        -- changed, but reason not classifiable

Final answer: MAJORITY VOTE of the final-round answers (no moderator, no judge).

Question sets (deterministic, cached under local/data/question/; identical across seeds)
  * mmlu      -- cais/mmlu (all): --n-categories (5) random categories x --per-category (20).
  * mmlu-pro  -- TIGER-Lab/MMLU-Pro: --n-categories (5) random categories x --per-category (20).
  * gpqa      -- Idavidrein/gpqa (gpqa_main): a FLAT random sample of --gpqa-limit (100) questions.
--dataset all (default) runs all three (300 questions/seed).

Output (per seed, one workbook: debate_local_7b_<dataset>_s<seed>.xlsx)
  * Debate_Traces    -- per-(round,agent) answer/confidence/response + the R{n} {agent} Changed/Reason
                        columns and the R{n} Change Pattern / Num Changed columns; Correct + Final.
  * Round_State      -- the shared ~47-col process-metric sheet PLUS: reason_for_change,
                        self_reported_changed, reason_raw, change_pattern, n_changed_in_round,
                        n_correct_agents, correct_in_round_pool.
  * Change_Patterns  -- KKK/CKK/CCK/CCC counts per (dataset, round).
  * Reason_Summary   -- reason_for_change counts per (dataset, round).
  * Diagnostic_Scores-- process metrics with --q-source confidence (there is NO LLM judge here).
  * Reasoning_Quality-- intentionally EMPTY (no judge round).
  * Protocol         -- the exact shared prompt + reason taxonomy, for the record.

Usage
-----
    # smoke test (5 questions, mmlu, seed 7)
    python generate_debate_local_7b.py --dataset mmlu --seeds 7 --objective-limit 5

    # full run (300 questions, seed 7)
    python generate_debate_local_7b.py

    # progress / resume status
    python generate_debate_local_7b.py --status
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Use the OS certificate store for TLS behind an intercepting proxy (no-op if truststore absent).
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent          # .../Knowledge_vs_Reasoning/local
KVR = HERE.parent                                # .../Knowledge_vs_Reasoning
sys.path.insert(0, str(KVR))                     # for `import generate_debate_roles_7b`
sys.path.insert(0, str(KVR.parent / "New"))      # for `import generate_baseline_v2`
sys.path.insert(0, str(KVR.parent / "docs"))     # for `import qwen_methodology_code`

# --- reused heavy machinery from the baseline generator (round-state + checkpoints) ---
from generate_baseline_v2 import (  # noqa: E402
    build_round_state, load_checkpoint, save_checkpoint, clear_checkpoint,
    checkpoint_path, workbook_is_complete,
)
# --- shared Qwen helpers ---
from qwen_methodology_code import (  # noqa: E402
    DebateQuestion, LocalQwenPipeline, OllamaQwenPipeline,
    QWEN_AGENTS, parse_qwen_turn, majority_answer,
    first_consensus_round_for_answer, final_answer_correctness, empty_judgments,
    score_mixed_debates, write_qwen_excel_report,
    qwen_reprompt_messages,
)
# --- reuse the EXACT deterministic question sets from the roles study (identical questions,
#     shared cache under ../data/question): gpqa=flat 100, mmlu/mmlu-pro=5 categories x 20 ---
from generate_debate_roles_7b import (  # noqa: E402
    load_questions as roles_load_questions,
)

DEFAULT_N_CATEGORIES = 5
DEFAULT_PER_CATEGORY = 20

FIXTURE_PATTERN = "debate_local_neutral_7b"

# --------------------------------------------------------------------------- #
# Neutral, symmetric protocol -- identical for every agent
# --------------------------------------------------------------------------- #
NEUTRAL_SYSTEM = (
    "You are one of three independent experts answering a multiple-choice question. "
    "Reason carefully and decide strictly on the merits of the facts and logic. "
    "Do NOT use social or deferential language -- never say things like 'I agree with the other "
    "expert' or name/credit another agent. Refer only to the facts and reasoning themselves."
)

INITIAL_TASK = (
    "Independently solve the problem. State your answer, your confidence, and a concise "
    "justification. Do NOT consider what other experts might think."
)
UPDATE_TASK = (
    "Read the other experts' responses below (their confidence is hidden). Revise your answer ONLY "
    "if another expert presents evidence or reasoning you find more convincing than your own. "
    "If you change your answer, explain why; if you keep it, explain why. Do not use social language."
)

# Reason taxonomy: canonical phrasing shown to the model, mapped to a stable code for analysis.
REASON_CHOICES = [
    "another agent recalled a fact I missed",
    "another agent's logical reasoning is stronger",
    "I discovered an inconsistency in my own reasoning",
    "I misunderstood the question",
    "I simply find another answer more plausible",
    "no change",
]
REASON_CODES = [
    "fact_recall", "stronger_logic", "self_inconsistency",
    "misunderstood", "more_plausible", "no_change", "unspecified",
]


def _fmt_initial(labels: tuple[str, ...]) -> str:
    return (
        "Use exactly this format:\n"
        f"Answer: <{'/'.join(labels)}>\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <concise justification>"
    )


def _fmt_update(labels: tuple[str, ...]) -> str:
    choices = " | ".join(f'"{c}"' for c in REASON_CHOICES)
    return (
        "Use exactly this format:\n"
        f"Answer: <{'/'.join(labels)}>\n"
        "Confidence: <number from 0 to 1>\n"
        "Changed: <yes or no>\n"
        f"Reason: <exactly one of: {choices}>\n"
        "Explanation: <why you changed or kept your answer>"
    )


def neutral_initial_messages(question: DebateQuestion, agent: str) -> list[dict]:
    user = f"{question.question}\n\n{INITIAL_TASK}\n\n{_fmt_initial(question.answer_labels)}"
    return [{"role": "system", "content": NEUTRAL_SYSTEM}, {"role": "user", "content": user}]


def neutral_update_messages(question: DebateQuestion, agent: str,
                            previous_round: dict[str, dict]) -> list[dict]:
    # Present the two OTHER experts anonymously (no identity, no confidence).
    others = []
    n = 0
    for other, turn in previous_round.items():
        if other == agent:
            continue
        n += 1
        others.append(f"Expert #{n} answer: {turn['answer']}\n"
                      f"Expert #{n} reasoning: {turn.get('response', '')}")
    own = previous_round[agent]
    user = (
        f"Question:\n{question.question}\n\n"
        f"Your previous answer was {own['answer']} with reasoning: {own.get('response', '')}\n\n"
        "The other experts' latest responses (confidence hidden):\n" + "\n\n".join(others) + "\n\n"
        f"{UPDATE_TASK}\n\n{_fmt_update(question.answer_labels)}"
    )
    return [{"role": "system", "content": NEUTRAL_SYSTEM}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Change / reason parsing
# --------------------------------------------------------------------------- #
_CHANGED_RE = re.compile(r"(?im)^\s*Changed\s*:\s*(yes|no|true|false|y|n)\b")
_REASON_RE = re.compile(r"(?im)^\s*Reason\s*:\s*(.+?)\s*$")


def parse_self_change(raw: str) -> tuple[bool, str, str]:
    """Return (self_changed, reason_raw, reason_code) from a raw update-turn response."""
    m = _CHANGED_RE.search(raw or "")
    self_changed = bool(m) and m.group(1).lower() in ("yes", "true", "y")
    rm = _REASON_RE.search(raw or "")
    reason_raw = rm.group(1).strip() if rm else ""
    return self_changed, reason_raw, classify_reason(reason_raw)


def classify_reason(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return ""
    if "no change" in t or t in ("none", "n/a", "na"):
        return "no_change"
    if "recall" in t or ("fact" in t and "miss" in t):
        return "fact_recall"
    if "inconsist" in t:
        return "self_inconsistency"
    if "misunderst" in t or "misread" in t or "misinterpret" in t:
        return "misunderstood"
    if "logic" in t or "reasoning is stronger" in t or "stronger" in t or "argument" in t:
        return "stronger_logic"
    if "plausible" in t or "more likely" in t or "prefer" in t:
        return "more_plausible"
    return "unspecified"


_PATTERN_BY_COUNT = {0: "KKK", 1: "CKK", 2: "CCK", 3: "CCC"}


def pattern_for(n_changed: int) -> str:
    return _PATTERN_BY_COUNT.get(n_changed, f"{n_changed}C")


# --------------------------------------------------------------------------- #
# One neutral-symmetric debate
# --------------------------------------------------------------------------- #
def run_one_debate(llm, question, seed, rounds, sleep, run_id, dataset_label):
    """Return (debate_row, round_state_rows, discard_reason). No roles, no judge, majority-vote final."""
    agent_order = list(QWEN_AGENTS)
    all_rounds: list[dict] = []

    for round_no in range(1, rounds + 1):
        round_turns: dict[str, dict] = {}
        discard_reason = ""
        for agent in agent_order:
            initial = round_no == 1
            msgs = (neutral_initial_messages(question, agent) if initial
                    else neutral_update_messages(question, agent, all_rounds[-1]))
            base_seed = seed + round_no * 100 + QWEN_AGENTS.index(agent) * 10
            raw = llm.complete(msgs, seed=base_seed, max_new_tokens=llm.max_new_tokens,
                               temperature=llm.temperature)
            parsed = parse_qwen_turn(raw, question.dataset_type, question.answer_labels, strict=True)
            if parsed["parse_failed"]:
                retry = qwen_reprompt_messages(msgs, question)
                raw2 = llm.complete(retry, seed=base_seed + 1, max_new_tokens=llm.max_new_tokens,
                                    temperature=llm.temperature)
                p2 = parse_qwen_turn(raw2, question.dataset_type, question.answer_labels, strict=True)
                if not p2["parse_failed"]:
                    parsed, raw = {**p2, "re_prompted": True}, raw2
                else:
                    discard_reason = f"{question.question_no} r{round_no} {agent}: {parsed['parse_error']}"
                    break
            # Self-reported change accounting (rounds >= 2 only).
            if not initial:
                sc, rraw, rcode = parse_self_change(raw)
                parsed["self_changed"], parsed["reason_raw"], parsed["reason_code"] = sc, rraw, rcode
            round_turns[agent] = parsed
        if sleep:
            time.sleep(sleep)
        if discard_reason:
            print(f"  Discarding: {discard_reason}", flush=True)
            return None, [], discard_reason
        all_rounds.append(round_turns)

    # Final answer: majority vote of the final round (no moderator).
    final_list = [all_rounds[-1][a]["answer"] for a in QWEN_AGENTS]
    final_answer = majority_answer(final_list)
    final_source = "majority_vote"

    debate_row = {
        "Question #": question.question_no, "Dataset Type": question.dataset_type,
        "Dataset Category": question.category, "Question": question.question,
        "Correct Answer": question.correct_answer, "Final Answer": final_answer,
        "Final Answer Source": final_source, "Fixture Pattern": FIXTURE_PATTERN,
        "Model": getattr(llm, "model_id", ""),
        "Rounds to Consensus": first_consensus_round_for_answer(all_rounds, final_answer),
    }
    for rn, turns in enumerate(all_rounds, start=1):
        for agent in QWEN_AGENTS:
            t = turns[agent]
            debate_row[f"R{rn} {agent} Answer"] = t["answer"]
            debate_row[f"R{rn} {agent} Conf"] = t.get("confidence", "")
            debate_row[f"R{rn} {agent} Response"] = t.get("response", "")

    state_rows = build_round_state(all_rounds, question, final_answer,
                                   dataset_label=dataset_label, seed=seed, run_id=run_id)
    _augment_state(state_rows, all_rounds)

    # Per-agent change/reason + per-round pattern columns on the debate trace.
    state_by = {(r["round"], r["agent"]): r for r in state_rows}
    for rn in range(2, rounds + 1):
        for agent in QWEN_AGENTS:
            r = state_by[(rn, agent)]
            debate_row[f"R{rn} {agent} Changed"] = "Yes" if r["answer_changed"] else "No"
            debate_row[f"R{rn} {agent} Reason"] = r["reason_for_change"]
        anchor = state_by[(rn, QWEN_AGENTS[0])]
        debate_row[f"R{rn} Change Pattern"] = anchor["change_pattern"]
        debate_row[f"R{rn} Num Changed"] = anchor["n_changed_in_round"]

    return debate_row, state_rows, ""


def _augment_state(state_rows: list[dict], all_rounds: list[dict]) -> None:
    """Add reason / change-pattern / correctness-pool columns to the round-state rows in place."""
    parsed_by = {(rn, a): t for rn, turns in enumerate(all_rounds, start=1)
                 for a, t in turns.items()}
    by_round: dict[int, list[dict]] = {}
    for row in state_rows:
        by_round.setdefault(row["round"], []).append(row)

    for rn, rows_ in by_round.items():
        initial = rn == 1
        n_changed = sum(1 for x in rows_ if x.get("answer_changed"))
        n_corr = sum(1 for x in rows_ if x.get("is_correct"))
        pattern = "" if initial else pattern_for(n_changed)
        for x in rows_:
            p = parsed_by.get((rn, x["agent"]), {})
            if initial:
                x["self_reported_changed"] = ""
                x["reason_raw"] = ""
                x["reason_for_change"] = ""
            else:
                x["self_reported_changed"] = "Yes" if p.get("self_changed") else "No"
                x["reason_raw"] = p.get("reason_raw", "")
                if not x.get("answer_changed"):
                    x["reason_for_change"] = "no_change"
                else:
                    code = p.get("reason_code", "")
                    x["reason_for_change"] = code if code and code != "no_change" else "unspecified"
            x["change_pattern"] = pattern
            x["n_changed_in_round"] = "" if initial else n_changed
            # Direct knowledge-vs-reasoning signals (same as the roles run).
            x["n_correct_agents"] = n_corr
            x["correct_in_round_pool"] = int(n_corr > 0)


# --------------------------------------------------------------------------- #
# Seed runner (reuses checkpoint machinery)
# --------------------------------------------------------------------------- #
def run_debates_for_seed(llm, questions, seed, run_dir, args, dataset_label, run_id):
    debate_rows, state_rows, done_nos = load_checkpoint(run_dir)
    for q_idx, question in enumerate(questions):
        if question.question_no in done_nos:
            print(f"  {dataset_label} s={seed} q={q_idx+1}/{len(questions)}: "
                  f"{question.question_no} [skip - checkpoint]", flush=True)
            continue
        print(f"  {dataset_label} s={seed} q={q_idx+1}/{len(questions)}: {question.question_no}",
              flush=True)
        debate_row, q_state_rows, _ = run_one_debate(
            llm, question, seed=seed + q_idx * 1000, rounds=args.rounds, sleep=args.sleep,
            run_id=run_id, dataset_label=dataset_label)
        if debate_row is None:
            continue
        debate_rows.append(debate_row)
        state_rows.extend(q_state_rows)
        save_checkpoint(run_dir, question.question_no, debate_row, q_state_rows)

    debates = pd.DataFrame(debate_rows)
    if not debates.empty:
        debates["Correct?"] = debates.apply(final_answer_correctness, axis=1)
    return debates, pd.DataFrame(state_rows)


def change_pattern_summary(round_state: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """KKK/CKK/CCK/CCC counts per (dataset, round) -- one row of counts per debate round >= 2."""
    if round_state.empty:
        return pd.DataFrame()
    # One pattern label per (question_no, round) -- take the first agent row.
    per = (round_state[round_state["round"] >= 2]
           .groupby(["question_no", "round"])["change_pattern"].first().reset_index())
    rows = []
    for rn, grp in per.groupby("round"):
        counts = grp["change_pattern"].value_counts().to_dict()
        rows.append({"dataset": dataset_label, "round": int(rn),
                     "n_debates": int(len(grp)),
                     "KKK": int(counts.get("KKK", 0)), "CKK": int(counts.get("CKK", 0)),
                     "CCK": int(counts.get("CCK", 0)), "CCC": int(counts.get("CCC", 0))})
    return pd.DataFrame(rows)


def reason_summary(round_state: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """reason_for_change counts per (dataset, round) across all agent turns in rounds >= 2."""
    if round_state.empty:
        return pd.DataFrame()
    sub = round_state[(round_state["round"] >= 2) & (round_state["reason_for_change"] != "")]
    rows = []
    for rn, grp in sub.groupby("round"):
        counts = grp["reason_for_change"].value_counts().to_dict()
        row = {"dataset": dataset_label, "round": int(rn), "n_turns": int(len(grp))}
        for code in REASON_CODES:
            row[code] = int(counts.get(code, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def protocol_sheet() -> pd.DataFrame:
    rows = [
        {"field": "system_prompt", "value": NEUTRAL_SYSTEM},
        {"field": "initial_task (Round 1 / spec Round 0)", "value": INITIAL_TASK},
        {"field": "update_task (Rounds 2..T)", "value": UPDATE_TASK},
        {"field": "final_answer", "value": "majority vote of final-round answers (no moderator, no judge)"},
        {"field": "change_patterns", "value": "KKK=0 changed, CKK=1, CCK=2, CCC=3 (objective answer change)"},
    ]
    for choice, code in zip(REASON_CHOICES, REASON_CODES):
        rows.append({"field": f"reason::{code}", "value": choice})
    rows.append({"field": "reason::unspecified", "value": "changed answer but reason not classifiable"})
    return pd.DataFrame(rows)


# Control chars disallowed in Excel's XML (openpyxl raises IllegalCharacterError on these).
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_illegal_xml(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Excel-illegal control characters from all string cells (in place-safe copy)."""
    if df is None or df.empty:
        return df
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda v: _ILLEGAL_XML_RE.sub("", v) if isinstance(v, str) else v)
    return df


def run_one_seed(llm, questions, seed, dataset_label, run_dir, run_id, args):
    output_path = run_dir / f"debate_local_7b_{dataset_label}_s{seed}.xlsx"
    if not args.overwrite and output_path.exists() and workbook_is_complete(output_path):
        print(f"[{dataset_label} s={seed}] Workbook complete - skipping.", flush=True)
        return output_path
    if args.overwrite:
        print(f"[{dataset_label} s={seed}] --overwrite: clearing checkpoints.", flush=True)
        clear_checkpoint(run_dir)
        for p in (output_path, output_path.with_suffix(".tmp.xlsx")):
            if p.exists():
                p.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Neutral debate (local, 7B): {dataset_label} seed={seed} ===", flush=True)
    debates, round_state = run_debates_for_seed(
        llm, questions, seed, run_dir, args, dataset_label, run_id)
    if debates.empty:
        raise RuntimeError(f"No debates for {dataset_label} seed={seed}.")
    print(f"Completed {len(debates)} debates, {len(round_state)} round-state rows.", flush=True)

    # NO judge round. Metrics use agent-reported confidence as the quality source.
    judgments = empty_judgments()
    scores = score_mixed_debates(debates, judgments, output_path.name,
                                 q_source="confidence", metric_version=args.metric_version)

    # Qwen occasionally emits control characters that openpyxl/Excel XML forbid;
    # strip them from all string cells before writing (loses nothing meaningful).
    debates = _strip_illegal_xml(debates)
    round_state = _strip_illegal_xml(round_state)
    scores = _strip_illegal_xml(scores)

    tmp_path = output_path.with_suffix(".tmp.xlsx")
    run_args = argparse.Namespace(**vars(args))
    run_args.backend = args.backend
    run_args.seed = seed
    run_args.model_id = args.model_id
    write_qwen_excel_report(tmp_path, f"debate_local_7b_{dataset_label}",
                            run_args, debates, judgments, scores)
    with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        round_state.to_excel(writer, sheet_name="Round_State", index=False)
        change_pattern_summary(round_state, dataset_label).to_excel(
            writer, sheet_name="Change_Patterns", index=False)
        reason_summary(round_state, dataset_label).to_excel(
            writer, sheet_name="Reason_Summary", index=False)
        protocol_sheet().to_excel(writer, sheet_name="Protocol", index=False)
    os.replace(tmp_path, output_path)
    print(f"Wrote {output_path}  (+Round_State: {len(round_state)} rows, "
          f"+Change_Patterns, +Reason_Summary, +Protocol)", flush=True)
    return output_path


# --------------------------------------------------------------------------- #
def build_pipeline(args):
    if args.backend == "ollama":
        return OllamaQwenPipeline(model_id=args.model_id, host=args.ollama_host,
                                  temperature=args.temperature, top_p=args.top_p,
                                  max_new_tokens=args.max_new_tokens)
    return LocalQwenPipeline(model_id=args.model_id, temperature=args.temperature,
                             top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                             device_map=args.device_map, torch_dtype=args.torch_dtype,
                             require_gpu=args.require_gpu)


def resolve_datasets(name: str) -> list[str]:
    if name == "all":
        return ["mmlu", "mmlu-pro", "gpqa"]
    if name == "both":
        return ["gpqa", "mmlu-pro"]
    return [name]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    p.add_argument("--model-id", default="qwen2.5:7b-instruct")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--out-dir", type=Path, default=HERE / "data" / "debate_local_7b")
    p.add_argument("--seeds", type=int, nargs="+", default=[7])
    p.add_argument("--dataset", choices=["mmlu", "mmlu-pro", "gpqa", "both", "all"], default="all",
                   help="all = mmlu + mmlu-pro + gpqa (100 each); both = gpqa + mmlu-pro")
    p.add_argument("--objective-limit", type=int, default=None, help="optional extra cap per dataset")
    p.add_argument("--n-categories", type=int, default=DEFAULT_N_CATEGORIES,
                   help="mmlu-pro/gpqa: randomly select this many categories per dataset")
    p.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY,
                   help="mmlu-pro/gpqa: questions per selected category")
    p.add_argument("--gpqa-limit", type=int, default=100,
                   help="gpqa: flat random sample size (matches the roles-study cache)")
    p.add_argument("--refresh-questions", action="store_true",
                   help="rebuild the cached question selection from HuggingFace")
    p.add_argument("--rounds", type=int, default=5,
                   help="R1 = independent (spec Round 0); R2..T = debate rounds")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--metric-version", choices=["paper", "corrected"], default="paper")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--status", action="store_true")
    # transformers-backend extras / HF fallbacks (mirrors the roles script)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--mmlu-pro-dataset", default="TIGER-Lab/MMLU-Pro")
    p.add_argument("--mmlu-pro-split", default="test")
    p.add_argument("--gpqa-dataset", default="Idavidrein/gpqa")
    p.add_argument("--gpqa-subset", default="gpqa_main")
    p.add_argument("--gpqa-category-field", default="Subdomain",
                   help="GPQA field used as category (Subdomain=many, 'High-level domain'=3)")
    p.add_argument("--mmlu-dataset", default="cais/mmlu")
    p.add_argument("--mmlu-config", default="all")
    p.add_argument("--mmlu-split", default="test")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = resolve_datasets(args.dataset)

    if args.status:
        print("=" * 70 + "\nDEBATE-LOCAL-7B (neutral, no judge) - RESUME STATUS\n" + "=" * 70)
        for ds in datasets:
            for seed in args.seeds:
                run_dir = args.out_dir / "runs" / f"{ds}_seed_{seed}"
                wb = run_dir / f"debate_local_7b_{ds}_s{seed}.xlsx"
                if wb.exists() and workbook_is_complete(wb):
                    print(f"  [{ds} s={seed}] COMPLETE ({wb.name})")
                elif checkpoint_path(run_dir).exists():
                    _, _, done = load_checkpoint(run_dir)
                    print(f"  [{ds} s={seed}] {len(done)} debates checkpointed")
                else:
                    print(f"  [{ds} s={seed}] not started")
        return

    print("=" * 70, flush=True)
    print(f"DEBATE-LOCAL-7B (neutral, symmetric, NO judge)  model={args.model_id} "
          f"backend={args.backend}", flush=True)
    print(f"  datasets={datasets} seeds={args.seeds} rounds={args.rounds} "
          f"(R1=independent, R2..={args.rounds} debate)", flush=True)
    print("  final answer = majority vote; quality source = confidence", flush=True)
    print("=" * 70, flush=True)

    llm = build_pipeline(args)
    run_id = 0
    for ds in datasets:
        questions = roles_load_questions(ds, args)
        if not questions:
            print(f"WARNING: no questions for {ds}; skipping.", flush=True)
            continue
        for seed in args.seeds:
            run_id += 1
            run_dir = args.out_dir / "runs" / f"{ds}_seed_{seed}"
            run_one_seed(llm, questions, seed, ds, run_dir, run_id, args)

    print("\n" + "=" * 70 + "\nDEBATE-LOCAL-7B COMPLETE\n" + "=" * 70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
