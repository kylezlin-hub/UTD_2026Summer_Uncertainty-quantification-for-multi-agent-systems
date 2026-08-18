"""generate_interventions.py (local, Qwen2.5-7B) — Double-intervention 2x2 to label
knowledge- vs reasoning-limited failures.

Local variant of ../generate_interventions.py. Two differences:
  1. Model: Qwen2.5-7B via Ollama by default (was Qwen2.5-14B / transformers).
  2. Source data: the same-model NEUTRAL debate workbooks produced by
     generate_debate_local_7b.py, under data/debate_local_7b/runs/<ds>_seed_<s>/
     debate_local_7b_<ds>_s<s>.xlsx  (mmlu, mmlu-pro, gpqa; one file each for seed 7).

Turns the pilot's *proxy* labels (correct-absent-from-pool) into *causal* ground truth by
re-testing each hard question under a 2x2 (knowledge x reasoning) design and measuring which
scaffold unlocks it. The knowledge arm is run with TWO briefs, giving six conditions:

    condition          scaffold given
    -----------------  ---------------------------------------------------------
    control            plain re-ask
    knowledge_blind    + brief written WITHOUT the answer shown  (leak-proof)
    knowledge_oracle   + brief written WITH the answer shown      (higher power)
    reasoning          + step-by-step reasoning scaffold
    both_blind         + blind brief + reasoning scaffold
    both_oracle        + oracle brief + reasoning scaffold

Each condition is run R times (independent seeds); recovery rate = P(correct) over R repeats.

Labeling (a scaffold 'unlocks' only if it beats control BOTH statistically -- one-sided
Fisher/z superiority test, p < alpha -- AND practically, gain >= margin):
    control high             -> stochastic-recoverable (was never a real failure)
    knowledge unlocks only   -> KNOWLEDGE-limited      (facts were missing)
    reasoning unlocks only   -> REASONING-limited      (facts present, mis-combined)
    either alone unlocks     -> both-sufficient
    only both unlocks        -> interaction (both needed)
    nothing unlocks (best < hard) -> hard/unrecoverable (deep gap OR systematic reasoning error)

Confidence: knowledge is judged PRIMARILY from the leak-proof BLIND brief. A knowledge-limited
label driven only by the oracle brief is downgraded to medium/low (possible leak). Each label
carries per-condition recovery rate, Wilson CI, n, and the superiority p-values.

Briefs: the oracle brief is generated in an answer-shown "tutor" pass; a hardened leakage filter
(option-letter mentions AND lexical echo of the correct option's wording vs distractors)
rejects/regenerates leaking briefs. The blind brief is generated answer-blind (leakage impossible
by construction). Both briefs are cached to JSONL for manual audit.

Backends: --backend {ollama,local,mock}. `ollama` (default) = OllamaQwenPipeline (qwen2.5:7b-instruct);
`local` = transformers Qwen2.5; `mock` = offline plumbing test (no model).

All outputs are written incrementally and are safe to interrupt/resume.

Usage
-----
    # 1. Preview the subset only (no model calls):
    python generate_interventions.py --select-only --datasets mmlu mmlu-pro gpqa

    # 2. Smoke-test the whole pipeline offline:
    python generate_interventions.py --backend mock --limit 5

    # 3. Real run (Ollama, Qwen2.5-7B), 5 repeats/condition, all three datasets:
    python generate_interventions.py --backend ollama --model-id qwen2.5:7b-instruct \
        --datasets mmlu mmlu-pro gpqa --repeats 5

    # 4. Label from existing solve results (no new generation):
    python generate_interventions.py --label-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the project's Qwen infrastructure.
HERE = Path(__file__).resolve().parent          # .../Knowledge_vs_Reasoning/local
KVR = HERE.parent                                # .../Knowledge_vs_Reasoning

PROJ = KVR.parent                                # .../Proj1
sys.path.insert(0, str(PROJ / "docs"))
from qwen_methodology_code import (  # noqa: E402
    DebateQuestion,
    LocalQwenPipeline,
    OllamaQwenPipeline,
    OBJECTIVE_LABELS,
    normalize_answer,
    parse_qwen_turn,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Source debate workbooks live under data/debate_local_7b/runs/<ds>_seed_<s>/.
# Override with --runs-dir for replication with a different model.
RUNS_DIR = HERE / "data" / "debate_local_7b" / "runs"
DEBATE_SEEDS = [7, 17, 42]


def _workbook_rel(ds: str, seed: int) -> str:
    return f"{ds}_seed_{seed}/debate_local_7b_{ds}_s{seed}.xlsx"


DATASET_FILES = {
    ds: [_workbook_rel(ds, s) for s in DEBATE_SEEDS]
    for ds in ("mmlu", "mmlu-pro", "gpqa")
}
# Six conditions: the knowledge/both arms are run with TWO briefs each --
#   *_blind   : brief written WITHOUT showing the model the correct answer (leak-proof
#               by construction; the conservative knowledge signal).
#   *_oracle  : brief written WITH the correct answer shown (higher power, but leak-prone;
#               used only as a secondary confirmation).
CONDITIONS = ["control", "knowledge_blind", "knowledge_oracle",
              "reasoning", "both_blind", "both_oracle"]
KNOWLEDGE_CONDS = ("knowledge_blind", "knowledge_oracle")
REAL_CONDITIONS = [c for c in CONDITIONS if c != "control"]
OUT_DIR = HERE / "interventions"
AGENTS = ["Agent1", "Agent2", "Agent3"]

SOLVER_SYSTEM = (
    "You are answering a multiple-choice question as carefully as you can. "
    "Choose the single best option."
)
REASONING_SCAFFOLD = (
    "Work through the problem systematically before answering:\n"
    "1. Restate precisely what the question asks.\n"
    "2. List the key facts, definitions, or principles that bear on it.\n"
    "3. Evaluate each option in turn, stating why it could be right or wrong.\n"
    "4. Eliminate the options that fail.\n"
    "5. Commit to the single best remaining option.\n"
    "Think step by step, then give your final answer.\n\n"
)
TUTOR_SYSTEM = (
    "You are a subject-matter expert writing a neutral background briefing that will help a "
    "student reason about a hard exam question. You know the correct answer, but you must NOT "
    "reveal it or hint at which option is correct."
)
TUTOR_SYSTEM_BLIND = (
    "You are a subject-matter expert writing a neutral background briefing that will help a "
    "student reason about a hard exam question. You are NOT told which option is correct and "
    "must not guess; simply supply the general domain knowledge relevant to the topic."
)


def answer_format_block(labels: tuple[str, ...]) -> str:
    slot = "/".join(labels)
    return (
        "Use exactly this format:\n"
        f"Answer: <{slot}>\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <brief justification>"
    )


# --------------------------------------------------------------------------- #
# Subset selection: correct-absent-from-R1-pool questions
# --------------------------------------------------------------------------- #
def _norm(x):
    return None if pd.isna(x) else str(x).strip().upper()


def parse_options(question_text: str) -> dict[str, str]:
    """Extract {letter: option_text} from an embedded 'A. ...' style question."""
    opts = {}
    for m in re.finditer(r"(?m)^\s*([A-J])[\.\)]\s+(.*\S)\s*$", question_text):
        opts[m.group(1).upper()] = m.group(2).strip()
    return opts


def infer_labels(question_text: str) -> tuple[str, ...]:
    opts = parse_options(question_text)
    if opts:
        return tuple(sorted(opts, key=OBJECTIVE_LABELS.index))
    return tuple("ABCD")


def _rounds_in(columns) -> list[int]:
    rs = {int(m.group(1)) for c in columns if (m := re.match(r"R(\d+) Agent1 Answer$", c))}
    return sorted(rs)


def _last_answer(row, rounds, agent):
    """Forward-filled final answer for an agent (last non-null across rounds)."""
    val = None
    for rn in rounds:
        a = _norm(row.get(f"R{rn} {agent} Answer"))
        if a is not None:
            val = a
    return val


def _mean_conf(row, rounds, which: str) -> float:
    rn = rounds[0] if which == "init" else rounds[-1]
    vals = []
    for a in AGENTS:
        try:
            v = float(row.get(f"R{rn} {a} Conf"))
            vals.append(v / 100.0 if v > 1 else v)
        except (TypeError, ValueError):
            continue
    return float(np.mean(vals)) if vals else np.nan


def build_question_table(datasets: list[str]) -> pd.DataFrame:
    """One row per unique question with GOLD-FREE observables + gold (for labeling only).

    Observable columns (available at deployment, no gold answer needed): init_distinct,
    init_unanimous, final_distinct, final_consensus, any_switch, rounds_to_consensus,
    mean_init_conf, mean_final_conf, and an observable `stratum`. Also records correct_answer
    and per-seed absent counts -- these are GOLD and are used ONLY for the correct-absent
    selection mode and for causal labeling, never as classifier features.
    """
    per_q: dict[str, dict] = {}
    seed_absent: dict[str, list[bool]] = defaultdict(list)
    for ds in datasets:
        for fname in DATASET_FILES[ds]:
            path = RUNS_DIR / fname
            if not path.exists():
                print(f"  [skip] missing {path}")
                continue
            df = pd.read_excel(path, sheet_name="Debate_Traces")
            rounds = _rounds_in(df.columns)
            if not rounds:
                print(f"  [skip] no round columns in {path}")
                continue
            for _, r in df.iterrows():
                qno = str(r.get("Question #"))
                correct = _norm(r.get("Correct Answer"))
                r1 = [_norm(r.get(f"R{rounds[0]} {a} Answer")) for a in AGENTS]
                r1v = [x for x in r1 if x is not None]
                absent = correct not in set(r1v)
                seed_absent[qno].append(absent)
                if qno in per_q:
                    per_q[qno]["n_seeds"] += 1
                    per_q[qno]["n_absent"] += int(absent)
                    continue
                last = [_last_answer(r, rounds, a) for a in AGENTS]
                lastv = [x for x in last if x is not None]
                final_distinct = len(set(lastv))
                final_consensus = len(lastv) == len(AGENTS) and final_distinct == 1
                any_switch = any(r1[i] and last[i] and r1[i] != last[i] for i in range(len(AGENTS)))
                tau = len(rounds) + 1  # rounds to unanimous consensus (T+1 = never)
                for rn in rounds:
                    ans = [_norm(r.get(f"R{rn} {a} Answer")) for a in AGENTS]
                    ans = [x for x in ans if x is not None]
                    if len(ans) == len(AGENTS) and len(set(ans)) == 1:
                        tau = rn
                        break
                per_q[qno] = dict(
                    question_no=qno, dataset=ds,
                    category=r.get("Dataset Category", ""),
                    question=str(r.get("Question")),
                    correct_answer=correct,
                    init_distinct=len(set(r1v)),
                    init_unanimous=int(len(set(r1v)) == 1),
                    init_correct_present=int(correct is not None and correct in set(r1v)),
                    final_distinct=final_distinct,
                    final_consensus=int(final_consensus),
                    any_switch=int(any_switch),
                    rounds_to_consensus=tau,
                    mean_init_conf=_mean_conf(r, rounds, "init"),
                    mean_final_conf=_mean_conf(r, rounds, "final"),
                    n_seeds=1, n_absent=int(absent),
                )
    tbl = pd.DataFrame(list(per_q.values()))
    if tbl.empty:
        return tbl
    tbl["stratum"] = tbl.apply(
        lambda x: f"{int(x['init_distinct'])}way_{'cons' if x['final_consensus'] else 'nocons'}",
        axis=1)
    return tbl.sort_values(["dataset", "question_no"]).reset_index(drop=True)


def select_questions(datasets: list[str], args) -> pd.DataFrame:
    """Dispatch selection.

    correct-absent : gold filter (correct answer absent from the R1 pool) -- knowledge-limited
                     ENRICHMENT / proxy validation; uses gold to select.
    correct-present-minority : gold filter (correct answer PRESENT in the R1 pool but agents
                     DISAGREE) -- the reasoning-limited-rich region; ENRICHMENT for the scarce
                     reasoning-limited class.
    all            : every unique question -- the true deployment distribution.
    stratified     : representative sample across observable strata (init_distinct x
                     final_consensus), with sample_weight = population/sampled so downstream
                     training can reweight to the full distribution. This is the recommended
                     REPRESENTATIVE population; use enrichment modes only to add minority-class
                     training examples (they are NOT population-representative).

    `--exclude PATH...` drops any question_no already present in the given CSV(s) -- use it to
    avoid re-labeling questions already covered by a prior (e.g. stratified) run.
    """
    tbl = build_question_table(datasets)
    if tbl.empty:
        return tbl
    if args.selection == "correct-absent":
        def keep(row):
            n, a = row["n_seeds"], row["n_absent"]
            if args.absent_mode == "all":
                return a == n
            if args.absent_mode == "any":
                return a > 0
            return a > n / 2
        sub = tbl[tbl.apply(keep, axis=1)].copy()
        sub["sample_weight"] = np.nan  # enrichment: not population-representative
    elif args.selection == "correct-present-minority":
        # a clone holds the correct answer but the panel disagrees -> reasoning-rich
        sub = tbl[(tbl["init_correct_present"] == 1) & (tbl["init_unanimous"] == 0)].copy()
        sub["sample_weight"] = np.nan  # enrichment: not population-representative
    elif args.selection == "all":
        sub = tbl.copy()
        sub["sample_weight"] = 1.0
    else:  # stratified
        rng = np.random.default_rng(args.sample_seed)
        parts = []
        for _, grp in tbl.groupby("stratum"):
            n_take = min(args.per_stratum, len(grp))
            idx = sorted(rng.choice(len(grp), size=n_take, replace=False).tolist())
            pick = grp.iloc[idx].copy()
            pick["sample_weight"] = len(grp) / n_take
            parts.append(pick)
        sub = pd.concat(parts)
    sub = _apply_exclusions(sub, getattr(args, "exclude", None))
    return sub.sort_values(["dataset", "question_no"]).reset_index(drop=True)


def _apply_exclusions(sub: pd.DataFrame, exclude_paths) -> pd.DataFrame:
    """Drop question_nos already present in any --exclude CSV (e.g. a prior run's selection)."""
    if not exclude_paths:
        return sub
    excluded: set[str] = set()
    for p in exclude_paths:
        p = Path(p)
        if not p.exists():
            print(f"  [exclude] missing {p} (skipped)")
            continue
        col = pd.read_csv(p)["question_no"].astype(str)
        excluded |= set(col)
        print(f"  [exclude] {p.name}: {col.nunique()} question_nos")
    if excluded:
        before = len(sub)
        sub = sub[~sub["question_no"].astype(str).isin(excluded)].copy()
        print(f"  [exclude] removed {before - len(sub)} already-covered questions, {len(sub)} remain")
    return sub


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class MockPipeline:
    """Offline stand-in: deterministic pseudo-answers for plumbing tests only."""

    def complete(self, messages, seed, max_new_tokens=None, temperature=None):
        content = messages[-1]["content"]
        labels = re.findall(r"Answer: <([A-J/]+)>", content)
        choices = labels[0].split("/") if labels else list("ABCD")
        pick = choices[(seed + len(content)) % len(choices)]
        if "background briefing" in messages[0]["content"].lower():
            return "- Relevant principle one.\n- Relevant principle two.\n- Relevant definition three."
        return f"Answer: {pick}\nConfidence: 0.7\nExplanation: mock reasoning."


def build_pipeline(args):
    if args.backend == "mock":
        return MockPipeline()
    if args.backend == "ollama":
        return OllamaQwenPipeline(
            model_id=args.model_id, host=args.ollama_host,
            temperature=args.temperature, top_p=0.9, max_new_tokens=args.max_new_tokens,
        )
    return LocalQwenPipeline(
        model_id=args.model_id, temperature=args.temperature, top_p=0.9,
        max_new_tokens=args.max_new_tokens, device_map="auto",
        torch_dtype="auto", require_gpu=args.require_gpu,
    )


# --------------------------------------------------------------------------- #
# Knowledge-brief generation (oracle-informed, answer-blind, leak-filtered)
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    "the a an of to in on at for and or but is are was were be been being with as by from "
    "that this these those it its their his her they them we you your our i he she of into over "
    "under about between among such not no than then so if can may might will would should could "
    "which who whom whose what when where why how each any all both few more most other some".split()
)


def _content_tokens(text: str) -> set[str]:
    """Lowercase content words (drop stopwords and tokens < 3 chars) for lexical overlap."""
    toks = re.findall(r"[a-z0-9][a-z0-9\-']+", (text or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOPWORDS}


def brief_mentions_options(brief: str, correct_letter: str) -> bool:
    """Formatting-hygiene leak: the brief names an option letter or asserts an answer."""
    if re.search(r"\b(the\s+)?(correct\s+)?(answer|option|choice)\s+is\b", brief.lower()):
        return True
    if re.search(rf"\boption\s+{correct_letter}\b", brief, re.IGNORECASE):
        return True
    if re.search(rf"(?<![A-Za-z]){correct_letter}[\.\)]\s", brief):  # "B. " / "B) "
        return True
    return False


def brief_echoes_correct(brief: str, correct_letter: str, options: dict[str, str],
                         overlap_hi: float = 0.6, gap: float = 0.34) -> bool:
    """Content leak: the brief distinctively reproduces the CORRECT option's wording.

    Flags when (a) a verbatim substring of the correct option (>=12 chars) appears, OR
    (b) the fraction of the correct option's content words present in the brief is both
    high (>= overlap_hi) and exceeds the best distractor's overlap by >= gap, OR
    (c) a content bigram unique to the correct option appears verbatim in the brief.
    """
    low = brief.lower()
    correct_text = options.get(correct_letter, "")
    if len(correct_text) >= 12 and correct_text.lower() in low:
        return True
    btoks = _content_tokens(brief)
    if not btoks:
        return False

    def option_overlap(opt_text: str) -> float:
        otoks = _content_tokens(opt_text)
        return len(otoks & btoks) / len(otoks) if otoks else 0.0

    correct_ov = option_overlap(correct_text)
    distractor_ov = max((option_overlap(t) for l, t in options.items() if l != correct_letter),
                        default=0.0)
    if correct_ov >= overlap_hi and (correct_ov - distractor_ov) >= gap:
        return True
    # distinctive-bigram check: a two-word content phrase unique to the correct option
    ctoks = _content_tokens(correct_text)
    other = set().union(*[_content_tokens(t) for l, t in options.items()
                          if l != correct_letter]) if len(options) > 1 else set()
    distinctive = ctoks - other
    words = re.findall(r"[a-z0-9][a-z0-9\-']+", correct_text.lower())
    for w1, w2 in zip(words, words[1:]):
        if w1 in distinctive and w2 in distinctive and f"{w1} {w2}" in low:
            return True
    return False


def brief_leaks(brief: str, correct_letter: str, options: dict[str, str]) -> bool:
    """True if an oracle brief appears to reveal the answer (formatting OR content leak)."""
    return (brief_mentions_options(brief, correct_letter)
            or brief_echoes_correct(brief, correct_letter, options))


_BRIEF_RULES = (
    "Write 3-6 short factual statements giving the domain knowledge a student needs to work "
    "out the answer themselves. RULES:\n"
    "- Do NOT mention any option letter (A, B, C, ...).\n"
    "- Do NOT say or imply which option is correct.\n"
    "- Do NOT restate or quote any answer option verbatim.\n"
    "- Give only general facts, definitions, formulas, or principles.\n"
    "Output as a bulleted list."
)


def generate_brief(llm, q: DebateQuestion, options: dict[str, str], seed: int,
                   oracle: bool, max_tries: int = 3):
    """Return (brief, leaked_flag).

    oracle=True  -> the model is shown the correct answer (higher power, leak-prone). We
                    regenerate up to max_tries to avoid BOTH formatting and content leaks.
    oracle=False -> the model is NOT shown the answer (leak-proof knowledge signal). We only
                    regenerate on formatting slips (naming an option); a content 'echo' of the
                    correct option is legitimate here (it's just what a knowledgeable expert says).
    """
    if oracle:
        system = TUTOR_SYSTEM
        user = (f"Question:\n{q.question}\n\n"
                f"(For your reference only, the correct answer is {q.correct_answer}.)\n\n"
                f"{_BRIEF_RULES}")
    else:
        system = TUTOR_SYSTEM_BLIND
        user = f"Question:\n{q.question}\n\n{_BRIEF_RULES}"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    brief = ""
    for t in range(max_tries):
        brief = llm.complete(msgs, seed=seed + t, max_new_tokens=350, temperature=0.5).strip()
        if oracle:
            if not brief_leaks(brief, q.correct_answer, options):
                return brief, False
        else:
            # only formatting hygiene matters for the blind brief
            if not brief_mentions_options(brief, q.correct_answer):
                return brief, False
    # kept last attempt; flag = did it still leak (oracle) / name an option (blind)?
    leaked = (brief_leaks(brief, q.correct_answer, options) if oracle
              else brief_mentions_options(brief, q.correct_answer))
    return brief, leaked


# --------------------------------------------------------------------------- #
# Solve passes for the four conditions
# --------------------------------------------------------------------------- #
def condition_spec(condition: str) -> tuple[str | None, bool]:
    """Map a condition to (brief_key, use_reasoning_scaffold). brief_key in {blind, oracle, None}."""
    use_reasoning = condition == "reasoning" or condition.startswith("both")
    if condition in ("knowledge_blind", "both_blind"):
        brief_key = "blind"
    elif condition in ("knowledge_oracle", "both_oracle"):
        brief_key = "oracle"
    else:
        brief_key = None
    return brief_key, use_reasoning


def build_solve_messages(q: DebateQuestion, condition: str, briefs: dict[str, str]) -> list[dict]:
    fmt = answer_format_block(q.answer_labels)
    brief_key, use_reasoning = condition_spec(condition)
    brief = briefs.get(brief_key, "") if brief_key else ""
    parts = []
    if brief:
        parts.append(f"Relevant background information:\n{brief}\n")
    if use_reasoning:
        parts.append(REASONING_SCAFFOLD)
    parts.append(f"{q.question}\n\n{fmt}")
    return [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def solve_once(llm, q: DebateQuestion, condition: str, briefs: dict[str, str], seed: int):
    msgs = build_solve_messages(q, condition, briefs)
    raw = llm.complete(msgs, seed=seed, max_new_tokens=512)
    parsed = parse_qwen_turn(raw, "objective", q.answer_labels, strict=False)
    pred = normalize_answer(str(parsed.get("answer", "")))
    return pred, (pred == q.correct_answer), raw


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                done.add((d["question_no"], d["condition"], d["repeat"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_generation(args, subset: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    briefs_path = OUT_DIR / "knowledge_briefs.jsonl"
    results_path = OUT_DIR / "solve_results.jsonl"
    llm = build_pipeline(args)

    # cached briefs
    # If --regenerate-briefs is set, ignore cached briefs and force fresh generation.
    # This is useful for cross-model studies where brief-instance variability matters.
    briefs = {}
    if briefs_path.exists() and not args.regenerate_briefs:
        for line in briefs_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            briefs[d["question_no"]] = d
        print(f"[Briefs] Loaded {len(briefs)} cached briefs from {briefs_path}")
    elif args.regenerate_briefs:
        print(f"[Briefs] --regenerate-briefs is set; forcing fresh brief generation")
    
    done = load_done(results_path)

    subset = subset.head(args.limit) if args.limit else subset
    print(f"Generating interventions for {len(subset)} questions x {len(CONDITIONS)} conditions "
          f"x {args.repeats} repeats  (backend={args.backend}, model={args.model_id})")

    for i, row in subset.reset_index(drop=True).iterrows():
        labels = infer_labels(row["question"])
        options = parse_options(row["question"])
        q = DebateQuestion("objective", row["question_no"], row["question"],
                            row["correct_answer"], labels, row.get("category", ""))
        # two briefs (once per question): blind (leak-proof) + oracle (higher power)
        # If --regenerate-briefs is set, always generate fresh briefs even if cached
        if q.question_no not in briefs or args.regenerate_briefs:
            blind_brief, blind_leaked = generate_brief(llm, q, options, seed=args.seed, oracle=False)
            oracle_brief, oracle_leaked = generate_brief(
                llm, q, options, seed=args.seed + 500, oracle=True)
            rec = dict(question_no=q.question_no, dataset=row["dataset"],
                       brief_blind=blind_brief, brief_oracle=oracle_brief,
                       blind_leaked=blind_leaked, oracle_leaked=oracle_leaked,
                       correct_answer=q.correct_answer)
            append_jsonl(briefs_path, rec)
            briefs[q.question_no] = rec
        brief_texts = {"blind": briefs[q.question_no].get("brief_blind", ""),
                       "oracle": briefs[q.question_no].get("brief_oracle", "")}
        oracle_leaked = bool(briefs[q.question_no].get("oracle_leaked", False))

        for cond in CONDITIONS:
            for rep in range(args.repeats):
                if (q.question_no, cond, rep) in done:
                    continue
                seed = args.seed + 1000 * rep + 13 * CONDITIONS.index(cond)
                pred, correct, raw = solve_once(llm, q, cond, brief_texts, seed)
                append_jsonl(results_path, dict(
                    question_no=q.question_no, dataset=row["dataset"], condition=cond,
                    repeat=rep, seed=seed, pred=pred, correct=bool(correct),
                    oracle_leaked=oracle_leaked,
                    raw=raw[:800],
                ))
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(subset)} questions done", flush=True)
    print(f"Done. Results -> {results_path}")


# --------------------------------------------------------------------------- #
# Statistics: Wilson CI + one-sided superiority test (condition > control)
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def superiority_p(k1: int, n1: int, k0: int, n0: int) -> float:
    """One-sided p-value for H1: rate(cond) > rate(control).

    Uses Fisher's exact test (exact, good for small n) when scipy is available,
    else a normal-approximation two-proportion test.
    """
    if n1 == 0 or n0 == 0:
        return 1.0
    try:
        from scipy.stats import fisher_exact  # noqa: PLC0415
        _, p = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")
        return float(p)
    except Exception:  # noqa: BLE001
        p1, p0 = k1 / n1, k0 / n0
        pooled = (k1 + k0) / (n1 + n0)
        se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
        if se == 0:
            return 1.0 if p1 <= p0 else 0.0
        z = (p1 - p0) / se
        return 0.5 * (1 - erf(z / sqrt(2)))  # upper-tail P(Z > z)


# --------------------------------------------------------------------------- #
# Labeling (counts + significance + blind/oracle confidence)
# --------------------------------------------------------------------------- #
def label_question(counts: dict[str, tuple[int, int]], oracle_leaked: bool,
                   margin: float, stoch: float, hard: float, alpha: float) -> dict:
    """counts[cond] = (successes, n). Returns a dict with label, confidence, rates, p-values.

    A scaffold 'unlocks' a question only if it beats control BOTH statistically
    (one-sided p < alpha) AND practically (recovery gain >= margin). Knowledge is judged
    primarily from the leak-proof BLIND brief; the oracle brief only corroborates.
    """
    def rate(c):
        k, n = counts.get(c, (0, 0))
        return (k / n) if n else float("nan")

    kc, nc = counts.get("control", (0, 0))
    c = rate("control")

    def sig(cond):
        kk, nn = counts.get(cond, (0, 0))
        if nn == 0 or nc == 0 or np.isnan(rate(cond)):
            return False, 1.0
        p = superiority_p(kk, nn, kc, nc)
        return (p < alpha and (rate(cond) - c) >= margin), p

    kb_sig, kb_p = sig("knowledge_blind")
    ko_sig, ko_p = sig("knowledge_oracle")
    r_sig, r_p = sig("reasoning")
    bb_sig, bb_p = sig("both_blind")
    bo_sig, bo_p = sig("both_oracle")

    real_rates = [rate(x) for x in REAL_CONDITIONS if not np.isnan(rate(x))]
    best = max(real_rates) if real_rates else float("nan")

    k_sig = kb_sig or ko_sig
    both_sig = bb_sig or bo_sig
    confidence = "high"

    if not np.isnan(c) and c >= stoch:
        label = "stochastic-recoverable"
    elif not np.isnan(best) and best < hard:
        label, confidence = "hard/unrecoverable", "medium"
    elif k_sig and not r_sig:
        label = "knowledge-limited"
        # leak-proof if the BLIND brief drove it; else oracle-only => treat with caution
        if kb_sig:
            confidence = "high"
        else:
            confidence = "low(oracle-only,leak-suspect)" if oracle_leaked else "medium(oracle-only)"
    elif r_sig and not k_sig:
        label = "reasoning-limited"
    elif k_sig and r_sig:
        label = "both-sufficient"
    elif both_sig:
        label = "interaction (both needed)"
    else:
        label, confidence = "ambiguous", "low"

    out = {"label": label, "confidence": confidence, "oracle_leaked": bool(oracle_leaked),
           "p_knowledge_blind": round(kb_p, 4), "p_knowledge_oracle": round(ko_p, 4),
           "p_reasoning": round(r_p, 4), "p_both_blind": round(bb_p, 4),
           "p_both_oracle": round(bo_p, 4)}
    for cnd in CONDITIONS:
        k, n = counts.get(cnd, (0, 0))
        lo, hi = wilson_ci(k, n)
        out[cnd] = round(rate(cnd), 3) if n else np.nan
        out[f"{cnd}_ci"] = f"[{lo:.2f},{hi:.2f}]" if n else ""
        out[f"{cnd}_n"] = n
    return out


def run_labeling(args):
    results_path = OUT_DIR / "solve_results.jsonl"
    if not results_path.exists():
        print(f"No results at {results_path}. Run generation first.")
        return
    rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    # per (question, condition): successes k and trials n
    grp = (df.groupby(["question_no", "dataset", "condition"])["correct"]
             .agg(k="sum", n="count").reset_index())
    oracle_leaked_by_q = df.groupby("question_no")["oracle_leaked"].max()

    out_rows = []
    for (qno, ds), sub in grp.groupby(["question_no", "dataset"]):
        counts = {row["condition"]: (int(row["k"]), int(row["n"])) for _, row in sub.iterrows()}
        res = label_question(counts, bool(oracle_leaked_by_q.get(qno, False)),
                             args.margin, args.stoch, args.hard, args.alpha)
        out_rows.append({"question_no": qno, "dataset": ds, **res})
    agg = pd.DataFrame(out_rows)
    # carry selection metadata (stratum, sample_weight, selection_source, observables) so
    # downstream training can reweight / separate representative vs enrichment rows. Merge from
    # ALL selection files (selected_questions.csv = the stratified run, selected_<mode>.csv =
    # enrichment runs); provenance comes from selection_source (derived from filename if absent).
    sel_files = sorted(OUT_DIR.glob("selected_*.csv"))
    if sel_files:
        keep = ["question_no", "stratum", "sample_weight", "selection_source", "init_distinct",
                "final_consensus", "rounds_to_consensus", "any_switch",
                "mean_init_conf", "mean_final_conf"]
        parts = []
        for p in sel_files:
            s = pd.read_csv(p)
            if "selection_source" not in s.columns:
                stem = p.stem.replace("selected_", "")
                s["selection_source"] = "stratified" if stem in ("questions", "stratified") else stem
            parts.append(s[[c for c in keep if c in s.columns]])
        sel = pd.concat(parts, ignore_index=True)
        sel["question_no"] = sel["question_no"].astype(str)
        sel = sel.drop_duplicates("question_no", keep="first")
        agg["question_no"] = agg["question_no"].astype(str)
        agg = agg.merge(sel, on="question_no", how="left")
    out_csv = OUT_DIR / "intervention_labels.csv"
    agg.to_csv(out_csv, index=False)

    print(f"\nLabeled {len(agg)} questions -> {out_csv}")
    print(f"(repeats/condition; margin={args.margin}, stochastic>={args.stoch}, "
          f"hard<{args.hard}, alpha={args.alpha})")
    present = [c for c in CONDITIONS if c in agg.columns]
    print("\nmean recovery rate by condition (over questions):")
    print(agg[present].mean(numeric_only=True).round(3).to_string())
    print("\nlabel distribution by dataset:")
    print(agg.groupby("dataset")["label"].value_counts().to_string())
    print("\nlabel confidence distribution:")
    print(agg["confidence"].value_counts().to_string())
    n_leak = int(agg["oracle_leaked"].sum())
    if n_leak:
        print(f"\n[note] {n_leak} questions had answer-leaking ORACLE briefs; their knowledge-limited "
              f"labels are trusted only if the BLIND brief also unlocked them (see confidence). "
              f"Audit {OUT_DIR/'knowledge_briefs.jsonl'}.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "mmlu-pro", "gpqa"],
                    choices=list(DATASET_FILES))
    ap.add_argument("--selection", default="correct-absent",
                    choices=["correct-absent", "correct-present-minority", "all", "stratified"],
                    help="which questions to label: stratified (representative, recommended), "
                         "all (full distribution), correct-absent (knowledge-limited enrichment / "
                         "proxy validation), correct-present-minority (reasoning-limited "
                         "enrichment). Enrichment modes are NOT population-representative.")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="CSV path(s); drop any question_no already present (e.g. a prior run's "
                         "selected_*.csv) so enrichment does not re-label covered questions")
    ap.add_argument("--absent-mode", default="all", choices=["all", "any", "majority"],
                    help="correct-absent only: absent in all/any/majority of seeds (default all)")
    ap.add_argument("--per-stratum", type=int, default=30,
                    help="stratified only: questions sampled per observable stratum")
    ap.add_argument("--sample-seed", type=int, default=12345,
                    help="stratified only: RNG seed for deterministic sampling")
    ap.add_argument("--backend", default="ollama", choices=["local", "ollama", "mock"])
    ap.add_argument("--model-id", default="qwen2.5:7b-instruct")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--repeats", type=int, default=12,
                    help="solve passes per condition (more = tighter recovery-rate estimate)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="cap number of questions (0 = all)")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--select-only", action="store_true", help="write subset CSV and exit")
    ap.add_argument("--regenerate-briefs", action="store_true",
                    help="ignore cached briefs and generate fresh ones (use for cross-model studies "
                         "to test brief-instance variability independently of model)")
    ap.add_argument("--label-only", action="store_true", help="label from existing results and exit")
    # labeling thresholds
    ap.add_argument("--margin", type=float, default=0.34, help="recovery gain over control to call a limit")
    ap.add_argument("--stoch", type=float, default=0.5, help="control rate >= this => stochastic-recoverable")
    ap.add_argument("--hard", type=float, default=0.2, help="best-condition rate < this => hard/unrecoverable")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for the one-sided condition>control superiority test")
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="override source debate workbooks directory (default: data/debate_local_7b/runs)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override output directory (default: interventions/)")
    ap.add_argument("--questions-csv", type=Path, default=None,
                    help="use this CSV directly as the question subset (columns: question_no, "
                         "dataset, category, question, correct_answer), bypassing --selection's "
                         "debate-workbook-based sourcing entirely. Use this to label a custom "
                         "population (e.g. newly-screened correct-absent questions from a "
                         "prompt-matched, non-debate rescreen) with the same 6-condition causal "
                         "labeling methodology. Results append into the same interventions/ "
                         "solve_results.jsonl, so a subsequent --label-only re-labels the union.")
    args = ap.parse_args()

    # Allow overriding globals for multi-model replication
    global RUNS_DIR, OUT_DIR
    if args.runs_dir is not None:
        RUNS_DIR = args.runs_dir
    if args.out_dir is not None:
        OUT_DIR = args.out_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.label_only:
        run_labeling(args)
        return

    if args.questions_csv is not None:
        subset = pd.read_csv(args.questions_csv)
        subset["question_no"] = subset["question_no"].astype(str)
        if "category" not in subset.columns:
            subset["category"] = ""
        subset["selection_source"] = "matched_rescreen_new"
        subset["sample_weight"] = np.nan  # enrichment-style population, not representative
        subset = _apply_exclusions(subset, getattr(args, "exclude", None))
        subset_csv = OUT_DIR / "selected_matched_rescreen_new.csv"
        subset.to_csv(subset_csv, index=False)
        print(f"Using custom question subset ({len(subset)} questions) from {args.questions_csv} "
              f"-> {subset_csv}")
        print(subset.groupby("dataset").size().to_string())
        if args.select_only:
            return
        run_generation(args, subset)
        run_labeling(args)
        return

    subset = select_questions(args.datasets, args)
    subset["selection_source"] = args.selection
    # stratified writes the canonical selected_questions.csv; other modes write per-mode files
    # so enrichment runs never clobber the representative sample.
    fname = "selected_questions.csv" if args.selection == "stratified" else f"selected_{args.selection}.csv"
    subset_csv = OUT_DIR / fname
    subset.to_csv(subset_csv, index=False)
    detail = {"correct-absent": f", absent-mode={args.absent_mode}",
              "stratified": f", per-stratum={args.per_stratum}"}.get(args.selection, "")
    print(f"Selected {len(subset)} questions (selection={args.selection}{detail}; "
          f"{', '.join(args.datasets)}) -> {subset_csv}")
    print(subset.groupby("dataset").size().to_string())
    if args.selection == "stratified" and not subset.empty:
        print("\nby observable stratum (sampled / weight):")
        summ = (subset.groupby("stratum")
                .agg(sampled=("question_no", "size"), weight=("sample_weight", "first")))
        print(summ.round(2).to_string())
    if args.select_only:
        return

    run_generation(args, subset)
    run_labeling(args)


if __name__ == "__main__":
    main()
