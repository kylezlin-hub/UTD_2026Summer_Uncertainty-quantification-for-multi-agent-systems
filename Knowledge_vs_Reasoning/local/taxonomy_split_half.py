"""taxonomy_split_half.py -- Redesigned causal taxonomy treating option-blind (stem-only) and
option-aware (stem+choices) self-elicited knowledge as two DISTINCT interventions, with a
held-out split-half evaluation to avoid tautological ("responsive questions respond") claims.

Design (per user specification)
--------------------------------
1. Baseline-stable correct is a PRE-TREATMENT STATE, not a recovery label: a question is
   baseline-stable-correct iff all 3 independent attempts in the matched Phase-1 rescreen
   (rescreen/phase1_matched_labels.csv, n_correct == 3) were correct. These questions are
   reported separately and EXCLUDED from the recovery taxonomy below (there is nothing to
   recover -- applying scaffold labels to them was the source of the "stochastic-recoverable"
   contamination diagnosed earlier).

2. For every other (initially/persistently incorrect) question, each of the 8 repeats per
   condition is split into two independent halves:
       Split A = repeats 0-3      Split B = repeats 4-7
   One half is used to CLASSIFY the question's response type (which scaffold(s) clear the
   delta=0.34 recovery-margin threshold); the OTHER half is used to independently ESTIMATE the
   recovery effect size for that classification (held-out, out-of-sample). This is done in BOTH
   directions (A classifies/B estimates, then B classifies/A estimates) and the two directions
   are compared for agreement -- a robustness check against classify-and-measure-on-the-same-data
   circularity.

3. Decision tree (applied per split, using ONLY that split's 4-repeat rates; delta = 0.34,
   equivalent to a difference of >= 2 correct out of 4 repeats given the 4-repeat quantization):
       stem_gain    = rate(knowledge_blind_stem) - rate(control)      [option-blind knowledge]
       options_gain = rate(knowledge_blind)      - rate(control)      [option-aware knowledge]
       reasoning_gain = rate(reasoning)          - rate(control)
       combo_gain   = rate(both_blind)           - rate(control)      [[combined scaffold;
                                                                          blind variant, the
                                                                          leak-proof one]]
       n_individual_hits = count([stem_gain>=delta, options_gain>=delta, reasoning_gain>=delta])

       if n_individual_hits >= 2:                          -> "multi-responsive"
       elif stem_gain >= delta:                             -> "option-blind knowledge-responsive"
       elif options_gain >= delta:                          -> "option-dependent knowledge-responsive"
       elif reasoning_gain >= delta:                        -> "reasoning-responsive"
       elif combo_gain >= delta:                            -> "combination-dependent"
       elif all deltas essentially null/negative:           -> "intervention-resistant"
       else:                                                -> "ambiguous" (residual catch-all;
                                                                  should rarely trigger given the
                                                                  above is exhaustive over the
                                                                  tested conditions)

Outputs (interventions/):
    taxonomy_split_half_results.csv -- one row per non-baseline-stable-correct question:
        full-8-repeat rates per condition, Split A/B rates, label_from_A (+ its held-out
        effect measured in B), label_from_B (+ its held-out effect measured in A), and an
        agreement flag between the two directions.
    taxonomy_baseline_stable.csv    -- the excluded baseline-stable-correct questions, reported
        separately with their full-8-repeat rates (for reference / appendix tables only).

Usage
-----
    python taxonomy_split_half.py                 # requires stem_only_solve_results.jsonl to be
                                                    # complete for all non-baseline-stable questions
    python taxonomy_split_half.py --allow-partial  # run on whatever stem-only data exists so far
                                                    # (dry-run / sanity check while the background
                                                    # stem-only job is still in progress)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "interventions"
SOLVE_RESULTS_PATH = OUT_DIR / "solve_results.jsonl"
STEM_RESULTS_PATH = OUT_DIR / "stem_only_solve_results.jsonl"
RESCREEN_LABELS_PATH = HERE / "rescreen" / "phase1_matched_labels.csv"

DELTA = 0.34
SPLIT_A_REPS = {0, 1, 2, 3}
SPLIT_B_REPS = {4, 5, 6, 7}

# Conditions used by the taxonomy (both_oracle/knowledge_oracle are loaded too but not used in
# the core decision tree -- kept in the output for reference/appendix).
CORE_CONDITIONS = ["control", "knowledge_blind", "knowledge_blind_stem", "reasoning", "both_blind"]


def load_repeat_level_data() -> pd.DataFrame:
    """One row per (question_no, dataset, condition, repeat, correct) across ALL conditions,
    from both the main Phase 2 file (control/knowledge_blind/knowledge_oracle/reasoning/
    both_blind/both_oracle) and the stem-only file (knowledge_blind_stem)."""
    rows = []
    for line in SOLVE_RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(dict(question_no=str(d["question_no"]), dataset=d.get("dataset", ""),
                          condition=d["condition"], repeat=int(d["repeat"]),
                          correct=bool(d["correct"])))
    if STEM_RESULTS_PATH.exists():
        for line in STEM_RESULTS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(dict(question_no=str(d["question_no"]), dataset=d.get("dataset", ""),
                              condition="knowledge_blind_stem", repeat=int(d["rep"]),
                              correct=bool(d["correct"])))
    return pd.DataFrame(rows)


def load_baseline_stable_flag() -> pd.DataFrame:
    df = pd.read_csv(RESCREEN_LABELS_PATH)
    df["question_no"] = df["question_no"].astype(str)
    df["baseline_stable_correct"] = df["n_correct"] == df["k"]
    return df[["question_no", "dataset", "baseline_stable_correct"]]


def rate_table(rs: pd.DataFrame, repset: set[int] | None) -> pd.DataFrame:
    """question_no x condition -> mean(correct) [optionally restricted to a repeat subset]."""
    sub = rs[rs["repeat"].isin(repset)] if repset is not None else rs
    return (sub.groupby(["question_no", "condition"])["correct"]
            .mean().unstack("condition"))


def classify_one(rates: pd.Series) -> tuple[str, str, float]:
    """Return (label, driving_condition, gain_at_classification_time) for one question's rates
    (a pandas Series indexed by condition, values = recovery rate over the 4 reps of one split)."""
    def g(cond):
        v = rates.get(cond, np.nan)
        c = rates.get("control", np.nan)
        if pd.isna(v) or pd.isna(c):
            return np.nan
        return v - c

    stem_gain = g("knowledge_blind_stem")
    options_gain = g("knowledge_blind")
    reasoning_gain = g("reasoning")
    combo_gain = g("both_blind")

    hits = {
        "knowledge_blind_stem": stem_gain,
        "knowledge_blind": options_gain,
        "reasoning": reasoning_gain,
    }
    n_hits = sum(1 for v in hits.values() if not pd.isna(v) and v >= DELTA)

    if any(pd.isna(v) for v in (stem_gain, options_gain, reasoning_gain)):
        return "ambiguous", "insufficient_data", np.nan
    if n_hits >= 2:
        winners = [k for k, v in hits.items() if v >= DELTA]
        return "multi-responsive", "+".join(winners), max(hits[w] for w in winners)
    if stem_gain >= DELTA:
        return "option-blind knowledge-responsive", "knowledge_blind_stem", stem_gain
    if options_gain >= DELTA:
        return "option-dependent knowledge-responsive", "knowledge_blind", options_gain
    if reasoning_gain >= DELTA:
        return "reasoning-responsive", "reasoning", reasoning_gain
    if not pd.isna(combo_gain) and combo_gain >= DELTA:
        return "combination-dependent", "both_blind", combo_gain
    # nothing cleared delta anywhere -> intervention-resistant, UNLESS all signals are NaN
    return "intervention-resistant", "none", max(
        v for v in (stem_gain, options_gain, reasoning_gain, combo_gain) if not pd.isna(v))


def held_out_effect(rates_other_split: pd.Series, driving_condition: str) -> float:
    """Recompute the SAME scaffold's gain, but using the OTHER (held-out) split's rates."""
    if driving_condition in ("none", "insufficient_data") or "+" in driving_condition:
        # multi-responsive / intervention-resistant: report the max individual gain out-of-sample
        conds = driving_condition.split("+") if "+" in driving_condition else \
            ["knowledge_blind_stem", "knowledge_blind", "reasoning"]
        gains = []
        for c in conds:
            v, ctrl = rates_other_split.get(c, np.nan), rates_other_split.get("control", np.nan)
            if not pd.isna(v) and not pd.isna(ctrl):
                gains.append(v - ctrl)
        return max(gains) if gains else np.nan
    v = rates_other_split.get(driving_condition, np.nan)
    ctrl = rates_other_split.get("control", np.nan)
    if pd.isna(v) or pd.isna(ctrl):
        return np.nan
    return v - ctrl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true",
                    help="run on whatever stem-only data currently exists, even if incomplete "
                         "(dry-run while the background stem-only job is still in progress)")
    args = ap.parse_args()

    rs = load_repeat_level_data()
    baseline = load_baseline_stable_flag()
    print(f"Loaded {len(rs)} repeat-level rows across "
          f"{rs.groupby('condition')['question_no'].nunique().to_dict()}")

    full_rates = rate_table(rs, None)
    a_rates = rate_table(rs, SPLIT_A_REPS)
    b_rates = rate_table(rs, SPLIT_B_REPS)

    all_qnos = sorted(set(rs["question_no"]))
    meta = rs[["question_no", "dataset"]].drop_duplicates("question_no")
    meta = meta.merge(baseline[["question_no", "baseline_stable_correct"]], on="question_no", how="left")

    if not args.allow_partial:
        n_conditions_present = full_rates.notna().sum(axis=1)
        incomplete = n_conditions_present[n_conditions_present < len(CORE_CONDITIONS)]
        non_baseline_incomplete = set(incomplete.index) & set(
            meta.loc[meta["baseline_stable_correct"] != True, "question_no"])  # noqa: E712
        if non_baseline_incomplete:
            print(f"[abort] {len(non_baseline_incomplete)} non-baseline-stable questions are "
                  f"missing >=1 core condition (likely stem-only job still running). "
                  f"Re-run with --allow-partial for a dry-run, or wait for completion.")
            return

    rows = []
    for qno in all_qnos:
        if qno not in full_rates.index:
            continue
        is_baseline_stable = bool(meta.loc[meta["question_no"] == qno, "baseline_stable_correct"].iloc[0]) \
            if qno in set(meta["question_no"]) else False
        dataset = meta.loc[meta["question_no"] == qno, "dataset"].iloc[0]

        rec = dict(question_no=qno, dataset=dataset, baseline_stable_correct=is_baseline_stable)
        for c in CORE_CONDITIONS + ["knowledge_oracle", "both_oracle"]:
            rec[f"full_{c}"] = full_rates.loc[qno, c] if (qno in full_rates.index and c in full_rates.columns) else np.nan

        if is_baseline_stable:
            rows.append(rec)
            continue

        a_row = a_rates.loc[qno] if qno in a_rates.index else pd.Series(dtype=float)
        b_row = b_rates.loc[qno] if qno in b_rates.index else pd.Series(dtype=float)

        label_A, driver_A, gain_A = classify_one(a_row)
        effect_A_heldout_in_B = held_out_effect(b_row, driver_A)
        label_B, driver_B, gain_B = classify_one(b_row)
        effect_B_heldout_in_A = held_out_effect(a_row, driver_B)

        rec.update(dict(
            label_from_A=label_A, driver_from_A=driver_A, gain_in_A=gain_A,
            effect_of_A_label_measured_in_B=effect_A_heldout_in_B,
            label_from_B=label_B, driver_from_B=driver_B, gain_in_B=gain_B,
            effect_of_B_label_measured_in_A=effect_B_heldout_in_A,
            labels_agree=(label_A == label_B),
        ))
        rows.append(rec)

    out = pd.DataFrame(rows)
    baseline_out = out[out["baseline_stable_correct"] == True].copy()  # noqa: E712
    tax_out = out[out["baseline_stable_correct"] != True].copy()  # noqa: E712

    baseline_out.to_csv(OUT_DIR / "taxonomy_baseline_stable.csv", index=False)
    tax_out.to_csv(OUT_DIR / "taxonomy_split_half_results.csv", index=False)

    print(f"\nBaseline-stable-correct (pre-treatment, excluded from taxonomy): {len(baseline_out)}")
    print(f"Non-baseline-stable questions entering the recovery taxonomy: {len(tax_out)}")

    print("\n=== label distribution, classified on Split A (n=0-3), held-out effect in Split B ===")
    print(tax_out["label_from_A"].value_counts().to_string())
    print("\nmean held-out effect (Split B) by label_from_A:")
    print(tax_out.groupby("label_from_A")["effect_of_A_label_measured_in_B"].agg(["mean", "count"]).round(3).to_string())

    print("\n=== label distribution, classified on Split B (n=4-7), held-out effect in Split A ===")
    print(tax_out["label_from_B"].value_counts().to_string())
    print("\nmean held-out effect (Split A) by label_from_B:")
    print(tax_out.groupby("label_from_B")["effect_of_B_label_measured_in_A"].agg(["mean", "count"]).round(3).to_string())

    agree = tax_out["labels_agree"].mean()
    print(f"\n=== A/B direction agreement rate: {agree:.3f} ({int(tax_out['labels_agree'].sum())}/{len(tax_out)}) ===")
    print("\nagreement by label_from_A (does the label replicate when classified on the other half?):")
    print(tax_out.groupby("label_from_A")["labels_agree"].agg(["mean", "count"]).round(3).to_string())

    print(f"\n-> {OUT_DIR / 'taxonomy_split_half_results.csv'}")
    print(f"-> {OUT_DIR / 'taxonomy_baseline_stable.csv'}")


if __name__ == "__main__":
    main()
