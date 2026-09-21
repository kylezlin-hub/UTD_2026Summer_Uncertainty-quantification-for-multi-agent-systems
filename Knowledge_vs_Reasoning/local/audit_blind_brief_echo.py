"""audit_blind_brief_echo.py -- Audit whether the BLIND knowledge brief (never shown the gold
answer) nonetheless lexically echoes/points to the correct option.

Motivation (methodological gap flagged during review)
-------------------------------------------------------
generate_brief()'s docstring explicitly states that for blind briefs "a content 'echo' of the
correct option is legitimate here" and only checks the FORMATTING leak (naming an option letter,
via brief_mentions_options) before accepting a blind brief -- it never runs the CONTENT-echo check
(brief_echoes_correct) that oracle briefs go through. But the blind brief generator is shown the
full question INCLUDING all multiple-choice options (q.question embeds them), so a capable model
can effectively solve the question itself while writing the "blind" brief and produce content that
substantively points to the correct option -- not gold data-leakage (it was never told the answer),
but a confound: the "knowledge_blind" condition's lift may partly come from the tutor-pass silently
solving the question, not from generic domain-knowledge injection.

This script applies the SAME leak-detection heuristics used for oracle briefs (brief_mentions_
options + brief_echoes_correct, imported directly from generate_interventions.py -- zero drift) to
every BLIND brief on file, purely as a diagnostic audit (no relabeling). It reports:
    - how often the blind brief's own overall best-matching option equals the correct answer
      (content overlap ranking, not just the yes/no echo flag)
    - the binary echo-flag rate (same threshold as oracle's leak filter)
    - whether echoing correlates with the Phase 2 label (e.g. is 'knowledge-limited' driven
      disproportionately by blind briefs that happen to echo the correct option?)

Usage
-----
    python audit_blind_brief_echo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "docs"))

from generate_interventions import (  # noqa: E402
    brief_mentions_options, brief_echoes_correct, _content_tokens, parse_options, RUNS_DIR,
    DATASET_FILES,
)

BRIEFS_PATH = HERE / "interventions" / "knowledge_briefs.jsonl"
LABELS_PATH = HERE / "interventions" / "intervention_labels.csv"
OUT_DIR = HERE / "interventions"


def load_question_options() -> dict[str, tuple[str, dict[str, str]]]:
    """question_no -> (correct_answer, {letter: option_text}), from the seed-7 Debate_Traces."""
    out = {}
    for ds, files in DATASET_FILES.items():
        seed7 = next((f for f in files if "_s7" in f or "seed_7" in f), files[0])
        path = RUNS_DIR / seed7
        if not path.exists():
            continue
        df = pd.read_excel(path, sheet_name="Debate_Traces")
        for _, r in df.iterrows():
            qno = str(r.get("Question #"))
            if qno in out:
                continue
            question = str(r.get("Question"))
            correct = str(r.get("Correct Answer")).strip().upper()
            opts = parse_options(question)
            out[qno] = (correct, opts)
    return out


def best_matching_option(brief: str, options: dict[str, str]) -> tuple[str, float]:
    """Return (best_letter, overlap_score) -- which option's content words the brief overlaps
    with most, by the same content-token overlap metric brief_echoes_correct uses internally."""
    btoks = _content_tokens(brief)
    if not btoks or not options:
        return "", 0.0
    best_letter, best_score = "", -1.0
    for letter, text in options.items():
        otoks = _content_tokens(text)
        score = len(otoks & btoks) / len(otoks) if otoks else 0.0
        if score > best_score:
            best_letter, best_score = letter, score
    return best_letter, best_score


def main():
    if not BRIEFS_PATH.exists():
        print(f"No briefs file at {BRIEFS_PATH}")
        return
    briefs = pd.read_json(BRIEFS_PATH, lines=True)
    briefs["question_no"] = briefs["question_no"].astype(str)
    print(f"Loaded {len(briefs)} question briefs")

    qopts = load_question_options()
    print(f"Loaded options for {len(qopts)} questions")

    rows = []
    for _, r in briefs.iterrows():
        qno = r["question_no"]
        if qno not in qopts:
            continue
        correct, options = qopts[qno]
        blind = str(r.get("brief_blind", "") or "")
        if not blind or not options:
            continue
        mentions = brief_mentions_options(blind, correct)
        echoes = brief_echoes_correct(blind, correct, options)
        best_letter, best_score = best_matching_option(blind, options)
        rows.append(dict(
            question_no=qno, dataset=r.get("dataset", ""),
            correct_answer=correct, n_options=len(options),
            blind_mentions_option_letter=mentions,   # formatting leak (already filtered at gen time)
            blind_content_echoes_correct=echoes,      # NEVER checked at generation time -- the audit
            best_matching_option=best_letter,
            best_matching_overlap=round(best_score, 3),
            best_matches_correct=(best_letter == correct),
        ))
    audit = pd.DataFrame(rows)
    out_csv = OUT_DIR / "blind_brief_echo_audit.csv"
    audit.to_csv(out_csv, index=False)
    print(f"\nAudited {len(audit)} blind briefs -> {out_csv}")

    print(f"\nblind briefs that MENTION an option letter (formatting; should be ~0, "
          f"pre-filtered at generation time): {int(audit['blind_mentions_option_letter'].sum())}")
    print(f"blind briefs that CONTENT-ECHO the correct option "
          f"(never filtered -- this is the audit finding): "
          f"{int(audit['blind_content_echoes_correct'].sum())} / {len(audit)} "
          f"({100*audit['blind_content_echoes_correct'].mean():.1f}%)")
    print(f"\nblind brief's single best content-overlap option == correct answer "
          f"(chance level ~= 1/n_options per question): "
          f"{int(audit['best_matches_correct'].sum())} / {len(audit)} "
          f"({100*audit['best_matches_correct'].mean():.1f}%)")
    print(f"mean n_options: {audit['n_options'].mean():.2f} "
          f"(implied chance rate if random: {100*(1/audit['n_options']).mean():.1f}%)")

    # Cross-tab against Phase 2 label, if available
    if LABELS_PATH.exists():
        labels = pd.read_csv(LABELS_PATH)
        labels["question_no"] = labels["question_no"].astype(str)
        merged = audit.merge(labels[["question_no", "label", "knowledge_blind", "control"]],
                              on="question_no", how="left")
        print("\n=== best_matches_correct rate by Phase 2 label ===")
        print(merged.groupby("label")["best_matches_correct"].agg(["mean", "count"]).round(3)
              .to_string())
        print("\n=== blind_content_echoes_correct rate by Phase 2 label ===")
        print(merged.groupby("label")["blind_content_echoes_correct"].agg(["mean", "count"]).round(3)
              .to_string())
        merged.to_csv(OUT_DIR / "blind_brief_echo_audit_with_labels.csv", index=False)
        print(f"\n-> {OUT_DIR / 'blind_brief_echo_audit_with_labels.csv'}")


if __name__ == "__main__":
    main()
