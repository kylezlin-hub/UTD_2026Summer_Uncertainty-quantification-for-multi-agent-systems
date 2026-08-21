"""brief_regeneration_robustness_check.py -- Tests whether information-responsive /
choice-conditioned-responsive / discordant classifications survive a FRESH regeneration
of the knowledge brief, not just a fresh solver sample against the ORIGINAL (fixed) brief.

Motivation (methodological limitation raised by the user)
-----------------------------------------------------------------------------------------
In the main study, each knowledge brief (stem-only S and choice-aware C) is generated ONCE
per question and then reused across all 8 solver repeats. The main cross-fitted taxonomy
(taxonomy_nested_information.py) therefore only establishes:

    P(correct | Q, FIXED brief instance, solver seed)  reproducible across solver seeds

not

    P(correct | Q, brief ~ generator, solver seed)      reproducible across brief instances

A question could be labeled "information-responsive" because that TYPE of stem-only brief
reliably helps, or merely because this one particular generated brief happened to be
unusually good (or the choice-aware brief happened to leak/cue unusually well). This script
regenerates BOTH informational conditions (stem-only S and choice-aware C) from scratch --
brand-new seeds, brand-new brief text -- and also draws a brand-new (not reused) control
sample, then re-derives the classification-relevant gains under the SAME integer-count
decision rule used in the main analysis (G >= 2 out of 4).

Scope: the 39 questions with headline_label in {information-responsive,
choice-conditioned responsive, discordant/other} (15 + 13 + 11). For the 3 questions whose
label_from_A is specifically "reasoning-responsive" (a subset of the "discordant/other"
headline bucket), we ALSO refresh G_R using a brand-new control + brand-new reasoning-scaffold
solves (the scaffold text itself is static, so this is a solver-stochasticity check rather
than a brief-regeneration check, but it's cheap to include for completeness).

Design (per question): 4 fresh control reps + 1 fresh stem-only brief + 4 solves against it
+ 1 fresh choice-aware brief + 4 solves against it = 12 solve calls + 2 brief calls.
For the reasoning-responsive subset, +4 more solve calls against the (static) reasoning
scaffold using the SAME fresh control already drawn (no extra brief calls).
Total over 39 questions: ~468 solve calls + ~78 brief calls -- cheap relative to the main
7-condition x 8-repeat x 300-question study.

All seeds are drawn from a disjoint offset (FRESH_SEED_BASE) from every seed used anywhere
in the main study or the stem-only study, so every generation here is genuinely independent
of (not merely non-identical to) the original data.

Usage
-----
    # Smoke test (no model calls):
    python brief_regeneration_robustness_check.py --backend mock --limit 3

    # Real run (Ollama, Qwen2.5-7B):
    python brief_regeneration_robustness_check.py --backend ollama --model-id qwen2.5:7b-instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from generate_interventions import (  # noqa: E402
    DebateQuestion, MockPipeline, RUNS_DIR, DATASET_FILES,
    SOLVER_SYSTEM, TUTOR_SYSTEM_BLIND, _BRIEF_RULES,
    answer_format_block, infer_labels, parse_options,
    build_solve_messages, solve_once, generate_brief, build_pipeline,
    normalize_answer, parse_qwen_turn,
)
from stem_only_knowledge_test import (  # noqa: E402
    strip_options_to_stem, generate_stem_only_brief, build_solve_messages_stem_brief,
)

HERE = Path(__file__).resolve().parent
TAXONOMY_PATH = HERE / "interventions" / "taxonomy_nested_results.csv"
OUT_DIR = HERE / "interventions" / "brief_regen_check"
BRIEFS_OUT = OUT_DIR / "fresh_briefs.jsonl"
RESULTS_OUT = OUT_DIR / "fresh_solve_results.jsonl"
SUMMARY_OUT = OUT_DIR / "brief_regen_summary.csv"

THRESHOLD_COUNT = 2       # out of 4 -- identical rule to the main taxonomy
FRESH_REPEATS = 4         # matches the main study's per-split repeat count
FRESH_SEED_BASE = 900000  # disjoint from every seed range used in the main / stem-only studies

DEFAULT_TARGET_HEADLINES = ("information-responsive", "choice-conditioned responsive", "discordant/other")


def load_target_questions(headlines=DEFAULT_TARGET_HEADLINES) -> pd.DataFrame:
    """The questions (by default, the 39 originally responsive/discordant ones) whose
    headline classification could be brief-instance-dependent, joined with their original
    label_from_A / label_from_B / agreed_label for comparison. Pass headlines=("persistent",)
    to instead test whether PERSISTENT questions remain persistent under a freshly regenerated
    brief -- the both_blind combined-scaffold check reuses the same fixed cached brief
    (condition_spec() maps both_blind's brief_key to the identical "blind" brief used by
    knowledge_blind), so it never tested brief-instance variability for the persistent group."""
    tax = pd.read_csv(TAXONOMY_PATH)
    tax["question_no"] = tax["question_no"].astype(str)
    target = tax[tax["headline_label"].isin(headlines)].copy()

    rows = []
    for ds, files in DATASET_FILES.items():
        seed7 = next((f for f in files if "_s7" in f or "seed_7" in f), files[0])
        path = RUNS_DIR / seed7
        if not path.exists():
            continue
        df = pd.read_excel(path, sheet_name="Debate_Traces")
        for _, r in df.iterrows():
            qno = str(r.get("Question #"))
            rows.append(dict(question_no=qno, dataset=ds,
                              category=r.get("Dataset Category", ""),
                              question=str(r.get("Question")),
                              correct_answer=str(r.get("Correct Answer")).strip().upper()))
    qmeta = pd.DataFrame(rows).drop_duplicates("question_no")

    merged = target.merge(qmeta[["question_no", "question", "correct_answer", "category"]],
                           on="question_no", how="left")
    missing = merged["question"].isna().sum()
    if missing:
        print(f"  [warn] {missing} target questions had no question text found")
    return merged.dropna(subset=["question"]).reset_index(drop=True)


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                done.add((d["question_no"], d["condition"], d["rep"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def classify_info(g_s, g_c) -> str:
    """Same decision rule as taxonomy_nested_information.classify(), restricted to the
    information/choice axes (reasoning is handled separately for the small subset that needs it)."""
    s_hit, c_hit = g_s >= THRESHOLD_COUNT, g_c >= THRESHOLD_COUNT
    if s_hit and c_hit:
        return "information-responsive"
    if (not s_hit) and c_hit:
        return "choice-conditioned responsive"
    if s_hit and (not c_hit):
        return "discordant information response"
    return "neither (persistent under fresh brief)"


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_target_questions(headlines=args.headlines)
    if args.limit:
        questions = questions.head(args.limit)
    print(f"Brief-regeneration robustness check: {len(questions)} target questions "
          f"(headlines={args.headlines}, backend={args.backend}, model={args.model_id})", flush=True)

    llm = build_pipeline(args)
    briefs = {}
    if BRIEFS_OUT.exists():
        for line in BRIEFS_OUT.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            briefs[d["question_no"]] = d
    done = load_done(RESULTS_OUT)

    for i, row in questions.reset_index(drop=True).iterrows():
        qno = row["question_no"]
        labels = infer_labels(row["question"])
        options = parse_options(row["question"])
        q = DebateQuestion("objective", qno, row["question"], row["correct_answer"],
                           labels, row.get("category", ""))
        needs_reasoning = row["label_from_A"] == "reasoning-responsive"

        # --- fresh control (never reused from the original 8 solve calls) ---
        for rep in range(FRESH_REPEATS):
            if (qno, "control", rep) in done:
                continue
            seed = FRESH_SEED_BASE + 13 * rep
            pred, correct, raw = solve_once(llm, q, "control", {}, seed)
            append_jsonl(RESULTS_OUT, dict(question_no=qno, dataset=row["dataset"],
                                            condition="control", rep=rep, seed=seed,
                                            pred=pred, correct=bool(correct), raw=raw[:800]))

        # --- fresh stem-only (S) brief + 4 fresh solves ---
        if qno not in briefs or "stem_only_brief" not in briefs[qno]:
            stem = strip_options_to_stem(row["question"])
            brief_s, leaked_s = generate_stem_only_brief(
                llm, stem, q.correct_answer, seed=FRESH_SEED_BASE + 1)
            rec = briefs.get(qno, dict(question_no=qno, dataset=row["dataset"]))
            rec.update(stem_only_brief=brief_s, stem_only_leaked=leaked_s)
            briefs[qno] = rec
        brief_s_text = briefs[qno]["stem_only_brief"]
        for rep in range(FRESH_REPEATS):
            if (qno, "knowledge_blind_stem_fresh", rep) in done:
                continue
            seed = FRESH_SEED_BASE + 100 + 13 * rep
            msgs = build_solve_messages_stem_brief(q, brief_s_text)
            raw = llm.complete(msgs, seed=seed, max_new_tokens=512, temperature=0.7)
            parsed = parse_qwen_turn(raw, "objective", q.answer_labels, strict=False)
            pred = normalize_answer(str(parsed.get("answer", "")))
            correct = pred == q.correct_answer
            append_jsonl(RESULTS_OUT, dict(question_no=qno, dataset=row["dataset"],
                                            condition="knowledge_blind_stem_fresh", rep=rep,
                                            seed=seed, pred=pred, correct=bool(correct), raw=raw[:800]))

        # --- fresh choice-aware (C) brief + 4 fresh solves ---
        if "choice_aware_brief" not in briefs[qno]:
            brief_c, leaked_c = generate_brief(llm, q, options, seed=FRESH_SEED_BASE + 2, oracle=False)
            briefs[qno].update(choice_aware_brief=brief_c, choice_aware_leaked=leaked_c)
        brief_c_text = briefs[qno]["choice_aware_brief"]
        for rep in range(FRESH_REPEATS):
            if (qno, "knowledge_blind_fresh", rep) in done:
                continue
            seed = FRESH_SEED_BASE + 200 + 13 * rep
            pred, correct, raw = solve_once(llm, q, "knowledge_blind",
                                             {"blind": brief_c_text}, seed)
            append_jsonl(RESULTS_OUT, dict(question_no=qno, dataset=row["dataset"],
                                            condition="knowledge_blind_fresh", rep=rep,
                                            seed=seed, pred=pred, correct=bool(correct), raw=raw[:800]))

        # --- optional fresh reasoning check (static scaffold; solver-stochasticity only) ---
        if needs_reasoning:
            for rep in range(FRESH_REPEATS):
                if (qno, "reasoning_fresh", rep) in done:
                    continue
                seed = FRESH_SEED_BASE + 300 + 13 * rep
                pred, correct, raw = solve_once(llm, q, "reasoning", {}, seed)
                append_jsonl(RESULTS_OUT, dict(question_no=qno, dataset=row["dataset"],
                                                condition="reasoning_fresh", rep=rep,
                                                seed=seed, pred=pred, correct=bool(correct),
                                                raw=raw[:800]))

        # rewrite the briefs cache in full after every question (small file; simplest way to
        # keep it resumable without incremental de-dup bookkeeping).
        with BRIEFS_OUT.open("w", encoding="utf-8") as f:
            for rec_qno, rec in briefs.items():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i + 1) % 5 == 0:
            print(f"  ...{i + 1}/{len(questions)} questions done", flush=True)

    print(f"Done. Results -> {RESULTS_OUT}", flush=True)


def summarize(headlines=DEFAULT_TARGET_HEADLINES):
    if not RESULTS_OUT.exists():
        print(f"No results at {RESULTS_OUT}. Run generation first.")
        return
    rows = [json.loads(l) for l in RESULTS_OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    rs = pd.DataFrame(rows)
    rs["question_no"] = rs["question_no"].astype(str)

    tax = pd.read_csv(TAXONOMY_PATH)
    tax["question_no"] = tax["question_no"].astype(str)
    target = tax[tax["headline_label"].isin(headlines)].copy()

    counts = rs.groupby(["question_no", "condition"])["correct"].agg(["sum", "count"]).reset_index()
    piv_sum = counts.pivot(index="question_no", columns="condition", values="sum").fillna(0)
    piv_n = counts.pivot(index="question_no", columns="condition", values="count").fillna(0)

    out = []
    for _, row in target.iterrows():
        qno = row["question_no"]
        if qno not in piv_sum.index:
            continue

        def n_correct(cond):
            return piv_sum.loc[qno].get(cond, float("nan"))

        def n_reps(cond):
            return piv_n.loc[qno].get(cond, 0)

        if n_reps("control") != FRESH_REPEATS:
            continue
        c_ctrl = n_correct("control")
        g_s = n_correct("knowledge_blind_stem_fresh") - c_ctrl if n_reps("knowledge_blind_stem_fresh") == FRESH_REPEATS else float("nan")
        g_c = n_correct("knowledge_blind_fresh") - c_ctrl if n_reps("knowledge_blind_fresh") == FRESH_REPEATS else float("nan")
        g_r = (n_correct("reasoning_fresh") - c_ctrl) if n_reps("reasoning_fresh") == FRESH_REPEATS else float("nan")

        fresh_label = classify_info(g_s, g_c) if pd.notna(g_s) and pd.notna(g_c) else "insufficient_data"
        if row["label_from_A"] == "reasoning-responsive" and pd.notna(g_r):
            fresh_label_full = "reasoning-responsive" if g_r >= THRESHOLD_COUNT else fresh_label
        else:
            fresh_label_full = fresh_label

        # classify_info() emits "neither (persistent under fresh brief)" for a question that
        # stays persistent; canonicalize it to "persistent" so it matches an original
        # label_from_A of "persistent" when computing the reproduction rate.
        def _canonical(lbl):
            return "persistent" if str(lbl).startswith("neither") else lbl

        stayed_persistent = _canonical(fresh_label_full) == "persistent"
        out.append(dict(
            question_no=qno, dataset=row["dataset"],
            original_headline_label=row["headline_label"], original_label_from_A=row["label_from_A"],
            control_rate_fresh=c_ctrl / FRESH_REPEATS,
            G_S_fresh=g_s, G_C_fresh=g_c, G_R_fresh=g_r,
            fresh_label=fresh_label_full,
            stayed_persistent=stayed_persistent,
            reproduced=(_canonical(fresh_label_full) == row["label_from_A"]),
        ))

    df = pd.DataFrame(out)
    df.to_csv(SUMMARY_OUT, index=False)

    print(f"\n{'=' * 70}\nBRIEF-REGENERATION ROBUSTNESS CHECK -- {len(df)} questions scored\n{'=' * 70}")
    print("\nReproduction rate by ORIGINAL headline label (fresh brief label == original label_from_A):")
    for lab, grp in df.groupby("original_headline_label"):
        rate = grp["reproduced"].mean()
        print(f"  {lab:35s}  {grp['reproduced'].sum():2d}/{len(grp):2d}  ({rate:.1%})")
    print(f"\n  OVERALL                              {df['reproduced'].sum():2d}/{len(df):2d}  ({df['reproduced'].mean():.1%})")

    if "stayed_persistent" in df.columns and df["stayed_persistent"].any():
        n_stay = int(df["stayed_persistent"].sum())
        print(f"\nStayed persistent under fresh briefs (no recovery on S or C): "
              f"{n_stay}/{len(df)}  ({n_stay / len(df):.1%})")

    print("\nFresh-label distribution vs. original label_from_A (confusion):")
    print(pd.crosstab(df["original_label_from_A"], df["fresh_label"], margins=True))
    print(f"\n-> {SUMMARY_OUT}")


def main():
    global OUT_DIR, BRIEFS_OUT, RESULTS_OUT, SUMMARY_OUT

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="ollama", choices=["local", "ollama", "mock"])
    ap.add_argument("--model-id", default="qwen2.5:7b-instruct")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap number of target questions (0 = all)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--headlines", nargs="+", default=list(DEFAULT_TARGET_HEADLINES),
                     help="which headline_label(s) to test, e.g. --headlines persistent")
    ap.add_argument("--scope-name", default=None,
                     help="output subfolder name under interventions/brief_regen_check_<name>/ "
                          "(default: 'default' for the original 3-headline scope, else the first "
                          "headline name, e.g. 'persistent')")
    args = ap.parse_args()

    scope = args.scope_name or (
        "default" if list(args.headlines) == list(DEFAULT_TARGET_HEADLINES)
        else "_".join(h.replace("/", "-").replace(" ", "-") for h in args.headlines))
    if scope != "default":
        OUT_DIR = HERE / "interventions" / f"brief_regen_check_{scope}"
        BRIEFS_OUT = OUT_DIR / "fresh_briefs.jsonl"
        RESULTS_OUT = OUT_DIR / "fresh_solve_results.jsonl"
        SUMMARY_OUT = OUT_DIR / "brief_regen_summary.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        run(args)
    summarize(headlines=args.headlines)


if __name__ == "__main__":
    main()
