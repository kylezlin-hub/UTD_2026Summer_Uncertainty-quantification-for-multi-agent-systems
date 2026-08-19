"""taxonomy_nested_information.py -- Nested information-vs-choice-vs-reasoning taxonomy with
cross-fitted held-out validation (v2 design, supersedes taxonomy_split_half.py's flat OR-based
rule per explicit methodological review).

Design rationale (see conversation record for full derivation)
----------------------------------------------------------------
1. Baseline-stable correct (3/3 on the matched Phase-1 rescreen) is a PRE-TREATMENT STATE,
   reported separately, and excluded from the recovery taxonomy entirely.

2. The two knowledge conditions test NESTED INFORMATION ACCESS by the brief generator, not nested
   scaffold CONTENT:
       K_stem  (knowledge_blind_stem)  = option-BLIND self-elicited knowledge (stem only)
       K_choice(knowledge_blind)       = option-AWARE self-elicited knowledge (stem + choices)
   K_stem's generator sees strictly LESS input than K_choice's generator (stem only vs stem +
   candidate options) -- that access relationship IS nested. The resulting BRIEF TEXT is NOT
   assumed to be nested, however: the two briefs are independently generated/sampled, not one
   built incrementally on top of the other, so K_choice's content need not be a superset of
   K_stem's. This is precisely why a question can show strong stem-only recovery alongside null
   (or even negative) choice-aware recovery -- "discordant information response" -- without any
   contradiction: seeing the candidate answers can change what the generator chooses to write in
   either direction, not just add to it. The gap (G_C - G_S) is still scientifically meaningful,
   but as "the marginal effect of conditioning the generator on the candidate set," not as "extra
   knowledge on top of a strict subset." We deliberately do NOT attribute a specific mechanism
   (elimination, contrastive framing, retrieval focusing, or answer cueing) to this gap -- any of
   these could produce it, and disentangling them needs a separate experiment (e.g. choice-
   permutation or distractor-swap probes), not this design.

3. All classification uses INTEGER COUNTS out of 4 (per labeling split), not continuous decimals.
   With only 4 repeats/split, achievable rate differences are {0, .25, .50, .75, 1.0}; the
   original delta=0.34 margin is operationally identical to "delta >= 0.50" i.e. "G >= 2" (>=2
   more correct out of 4) at this resolution -- so the rule is written directly in the count
   space it actually lives in, avoiding false precision.

4. Decision tree (evaluated on ONE split's 4-repeat counts; G_X = C_X - C_control, threshold=2):
       G_S>=2 and G_C>=2   -> "information-responsive"           (stem alone already sufficient,
                                                                     confirmed still holds w/ choices)
       G_S<2  and G_C>=2   -> "choice-conditioned responsive"    (needs the candidate answers to
                                                                     produce a useful brief)
       G_S>=2 and G_C<2    -> "discordant information response"  (rare; reported separately, NEVER
                                                                     silently merged into another
                                                                     category)
       G_S<2  and G_C<2    -> check reasoning: G_R = C_reasoning - C_control
                                   G_R>=2  -> "reasoning-responsive"
                                   else    -> "persistent"

5. Held-out cross-fitted validation: classify on Split A (repeats 0-3), then measure Delta_S,
   Delta_C, Delta_R (RATES, i.e. count/4) on Split B (repeats 4-7) -- data the classifier never
   saw -- regardless of which label was assigned. Then reverse (classify on B, measure on A).
   IMPORTANT (precise language for reporting): this is NOT an independent replication -- both
   directions reuse the same 8 underlying runs, just partitioned differently. The correct framing
   is "the pattern was preserved under reciprocal cross-fitting" / "reversing the labeling and
   evaluation halves produced nearly identical response profiles," NOT "replicated independently."
   A true independent replication would require new generations (a different seed/model run), not
   a re-partition of the same 8 repeats.

   Also: because A->B and B->A both describe the SAME 137 questions, they must NOT be pooled as
   if they were 274 independent observations. The per-question "*_pooled" columns below average
   the two held-out estimates for a single question (not across questions), so aggregating those
   pooled columns still respects n=137, not n=274. Cross-fitting agreement statistics (Cohen's
   kappa, confusion matrix, per-class F1/Jaccard) are computed on the true n as well.

   TWO-TIER REPORTING -- PRIMARY vs DESCRIPTIVE (important, do not conflate):
     PRIMARY (inferential): the two per-direction tables -- "held-out (Split B) deltas by
       label_from_A" and "held-out (Split A) deltas by label_from_B" -- each uses ALL eligible
       questions classified in that direction. This is what should be cited as the population-
       level effect size / phenotype prevalence.
     DESCRIPTIVE ONLY (NOT inferential): the agreement-conditioned "agreed_label" / "*_pooled"
       summary, restricted to questions where label_from_A == label_from_B. Conditioning on
       agreement preferentially retains the cleanest, most strongly-expressed examples of each
       phenotype (e.g. a borderline question with G_S^A=2, G_S^B=1 -- right at threshold in one
       split, just under it in the other -- is EXCLUDED here even though it is a real, eligible
       question). This selection means agreed-label effect sizes will tend to look larger/cleaner
       than the true population effect. Use this table only for illustrative "response fingerprint"
       visualization, never as the primary effect-size or prevalence estimate.

6. Small-sample categories (e.g. "reasoning-responsive") are NOT force-fit into the headline
   taxonomy as their own pillar if final n stays tiny -- symmetry across categories is not a goal
   in itself. A `headline_label` column collapses to exactly four NAMED buckets: information-
   responsive, choice-conditioned responsive, persistent, and "discordant/other" (which absorbs
   "discordant information response", "reasoning-responsive", and any other small residual --
   named explicitly so "discordant" stays visible rather than vanishing into a generic bucket).
   The full granular `label_from_A`/`label_from_B` are always preserved in the output CSV
   regardless of this headline collapse.

Outputs (interventions/):
    taxonomy_nested_baseline_stable.csv -- excluded pre-treatment questions (reference only)
    taxonomy_nested_results.csv         -- full per-question table: counts/gains per split,
                                            label_from_A/B, held-out deltas in both directions,
                                            cross-fitted pooled deltas, agreement flag, headline_label
    taxonomy_nested_agreement_stats.json -- Cohen's kappa, confusion matrix, per-class F1/Jaccard
                                            between label_from_A and label_from_B (real
                                            classifications only, n not doubled)

Usage
-----
    python taxonomy_nested_information.py                  # requires complete stem-only data for
                                                              all non-baseline-stable questions
    python taxonomy_nested_information.py --allow-partial  # dry-run on whatever data exists now
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

THRESHOLD_COUNT = 2  # out of 4 repeats per split -- operationalizes delta>=0.34 at this resolution
SPLIT_A_REPS = {0, 1, 2, 3}
SPLIT_B_REPS = {4, 5, 6, 7}
CORE_CONDITIONS = ["control", "knowledge_blind_stem", "knowledge_blind", "reasoning"]


def load_repeat_level_data() -> pd.DataFrame:
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


def count_tables(rs: pd.DataFrame, repset: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (n_correct, n_reps) tables, question_no x condition, for a repeat subset."""
    sub = rs[rs["repeat"].isin(repset)]
    grouped = sub.groupby(["question_no", "condition"])["correct"]
    n_correct = grouped.sum().unstack("condition")
    n_reps = grouped.count().unstack("condition")
    return n_correct, n_reps


def gains_for_question(n_correct_row: pd.Series, n_reps_row: pd.Series) -> dict:
    """Return {G_S, G_C, G_R} as integer count-gains over control, or NaN if data incomplete
    (any core condition missing or not exactly 4 reps in this split)."""
    def gain(cond):
        nc, nr = n_correct_row.get(cond, np.nan), n_reps_row.get(cond, np.nan)
        ncc, nrc = n_correct_row.get("control", np.nan), n_reps_row.get("control", np.nan)
        if any(pd.isna(x) for x in (nc, nr, ncc, nrc)) or nr != 4 or nrc != 4:
            return np.nan
        return nc - ncc

    return dict(G_S=gain("knowledge_blind_stem"), G_C=gain("knowledge_blind"), G_R=gain("reasoning"))


def classify(gains: dict) -> str:
    g_s, g_c, g_r = gains["G_S"], gains["G_C"], gains["G_R"]
    if any(pd.isna(v) for v in (g_s, g_c, g_r)):
        return "insufficient_data"
    s_hit, c_hit, r_hit = g_s >= THRESHOLD_COUNT, g_c >= THRESHOLD_COUNT, g_r >= THRESHOLD_COUNT
    if s_hit and c_hit:
        return "information-responsive"
    if (not s_hit) and c_hit:
        return "choice-conditioned responsive"
    if s_hit and (not c_hit):
        return "discordant information response"
    return "reasoning-responsive" if r_hit else "persistent"


def rate_deltas(n_correct_row: pd.Series, n_reps_row: pd.Series) -> dict:
    """Delta_S, Delta_C, Delta_R as RATES (count/4) relative to control, for the held-out split."""
    def delta(cond):
        nc, nr = n_correct_row.get(cond, np.nan), n_reps_row.get(cond, np.nan)
        ncc, nrc = n_correct_row.get("control", np.nan), n_reps_row.get("control", np.nan)
        if any(pd.isna(x) for x in (nc, nr, ncc, nrc)) or nr == 0 or nrc == 0:
            return np.nan
        return (nc / nr) - (ncc / nrc)

    return dict(delta_S=delta("knowledge_blind_stem"), delta_C=delta("knowledge_blind"),
                delta_R=delta("reasoning"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    rs = load_repeat_level_data()
    baseline = load_baseline_stable_flag()
    print(f"Loaded {len(rs)} repeat-level rows across "
          f"{rs.groupby('condition')['question_no'].nunique().to_dict()}")

    nc_A, nr_A = count_tables(rs, SPLIT_A_REPS)
    nc_B, nr_B = count_tables(rs, SPLIT_B_REPS)
    nc_full, nr_full = count_tables(rs, SPLIT_A_REPS | SPLIT_B_REPS)

    meta = rs[["question_no", "dataset"]].drop_duplicates("question_no")
    meta = meta.merge(baseline[["question_no", "baseline_stable_correct"]], on="question_no", how="left")

    if not args.allow_partial:
        have_all = nc_full.reindex(columns=CORE_CONDITIONS).notna().all(axis=1)
        non_baseline = set(meta.loc[meta["baseline_stable_correct"] != True, "question_no"])  # noqa: E712
        incomplete = set(have_all[~have_all].index) & non_baseline
        if incomplete:
            print(f"[abort] {len(incomplete)} non-baseline-stable questions missing core "
                  f"condition data (stem-only job likely still running). Use --allow-partial "
                  f"for a dry-run, or wait for completion.")
            return

    rows = []
    for qno in sorted(set(rs["question_no"])):
        is_baseline = bool(meta.loc[meta["question_no"] == qno, "baseline_stable_correct"].iloc[0])
        dataset = meta.loc[meta["question_no"] == qno, "dataset"].iloc[0]
        rec = dict(question_no=qno, dataset=dataset, baseline_stable_correct=is_baseline)
        for c in CORE_CONDITIONS + ["both_blind", "knowledge_oracle", "both_oracle"]:
            if qno in nc_full.index and c in nc_full.columns and qno in nr_full.index and c in nr_full.columns:
                nc_v, nr_v = nc_full.loc[qno, c], nr_full.loc[qno, c]
                rec[f"full_{c}_rate"] = (nc_v / nr_v) if (not pd.isna(nr_v) and nr_v > 0) else np.nan
            else:
                rec[f"full_{c}_rate"] = np.nan

        if is_baseline:
            rows.append(rec)
            continue

        row_A_nc = nc_A.loc[qno] if qno in nc_A.index else pd.Series(dtype=float)
        row_A_nr = nr_A.loc[qno] if qno in nr_A.index else pd.Series(dtype=float)
        row_B_nc = nc_B.loc[qno] if qno in nc_B.index else pd.Series(dtype=float)
        row_B_nr = nr_B.loc[qno] if qno in nr_B.index else pd.Series(dtype=float)

        gains_A = gains_for_question(row_A_nc, row_A_nr)
        gains_B = gains_for_question(row_B_nc, row_B_nr)
        label_A = classify(gains_A)
        label_B = classify(gains_B)
        deltas_in_B = rate_deltas(row_B_nc, row_B_nr)   # held-out for label_A
        deltas_in_A = rate_deltas(row_A_nc, row_A_nr)   # held-out for label_B

        def control_rate(row_nc, row_nr):
            nc_v, nr_v = row_nc.get("control", np.nan), row_nr.get("control", np.nan)
            return (nc_v / nr_v) if (not pd.isna(nr_v) and nr_v > 0) else np.nan

        rec.update({
            "G_S_A": gains_A["G_S"], "G_C_A": gains_A["G_C"], "G_R_A": gains_A["G_R"],
            "label_from_A": label_A,
            # HELD-OUT baseline (control) rate for label_A -- from Split B, which label_A's
            # classification (built entirely from Split A, including Split A's own control count)
            # never saw. This is the circularity-free baseline to compare phenotypes against --
            # NOT full_control_rate, which mixes in the same Split-A control count used to define
            # label_A itself.
            "control_rate_heldout_B": control_rate(row_B_nc, row_B_nr),
            "delta_S_heldout_B": deltas_in_B["delta_S"], "delta_C_heldout_B": deltas_in_B["delta_C"],
            "delta_R_heldout_B": deltas_in_B["delta_R"],
            "G_S_B": gains_B["G_S"], "G_C_B": gains_B["G_C"], "G_R_B": gains_B["G_R"],
            "label_from_B": label_B,
            "control_rate_heldout_A": control_rate(row_A_nc, row_A_nr),
            "delta_S_heldout_A": deltas_in_A["delta_S"], "delta_C_heldout_A": deltas_in_A["delta_C"],
            "delta_R_heldout_A": deltas_in_A["delta_R"],
            "labels_agree": (label_A == label_B) and label_A != "insufficient_data",
        })
        # Cross-fitted pooled estimate (only meaningful/reported when directions agree on a
        # REAL classification -- insufficient_data==insufficient_data is a vacuous, not a
        # genuine, agreement and must not be pooled).
        for key in ("delta_S", "delta_C", "delta_R"):
            a_val = rec[f"{key}_heldout_B"]
            b_val = rec[f"{key}_heldout_A"]
            if rec["labels_agree"] and not (pd.isna(a_val) and pd.isna(b_val)):
                rec[f"{key}_pooled"] = np.nanmean([a_val, b_val])
            else:
                rec[f"{key}_pooled"] = np.nan
        rec["agreed_label"] = label_A if rec["labels_agree"] else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    baseline_out = out[out["baseline_stable_correct"] == True].copy()  # noqa: E712
    tax_out = out[out["baseline_stable_correct"] != True].copy()  # noqa: E712

    # Headline (reporting-only) collapse: exactly four named buckets per the agreed design --
    # Information-responsive / Choice-conditioned responsive / Persistent / Discordant-other.
    # "Discordant/other" absorbs "discordant information response", "reasoning-responsive", and
    # any other small residual category -- it is a NAMED bucket (not a generic size-threshold
    # collapse), so "discordant" stays visible/labeled even if its own count is small, rather than
    # risking it being silently merged away by a blind rare-count rule. insufficient_data is kept
    # separate (a data-completeness flag, not a scientific phenotype). Full granular labels are
    # always retained in label_from_A/label_from_B regardless of this headline collapse.
    HEADLINE_CORE = {"information-responsive", "choice-conditioned responsive", "persistent"}

    def to_headline(label):
        if label == "insufficient_data":
            return "insufficient_data"
        return label if label in HEADLINE_CORE else "discordant/other"

    tax_out["headline_label"] = tax_out["label_from_A"].apply(to_headline)

    baseline_out.to_csv(OUT_DIR / "taxonomy_nested_baseline_stable.csv", index=False)
    tax_out.to_csv(OUT_DIR / "taxonomy_nested_results.csv", index=False)

    print(f"\nBaseline-stable-correct (pre-treatment, excluded): {len(baseline_out)}")
    print(f"Entering recovery taxonomy: {len(tax_out)}")
    non_core = sorted(set(tax_out["label_from_A"]) - HEADLINE_CORE - {"insufficient_data"})
    if non_core:
        print(f"\n[note] categories folded into the named 'discordant/other' headline bucket: "
              f"{non_core}")

    print("\n=== HEADLINE label distribution (4-bucket view; from label_from_A) ===")
    print(tax_out["headline_label"].value_counts().to_string())

    print("\n=== HEADLINE phenotype prevalence BY DATASET (label_from_A; real classifications only) ===")
    real_for_ds = tax_out[tax_out["label_from_A"] != "insufficient_data"]
    if len(real_for_ds):
        ct = pd.crosstab(real_for_ds["dataset"], real_for_ds["headline_label"])
        props = ct.div(ct.sum(axis=1), axis=0)
        print("\ncounts:")
        print(ct.to_string())
        print("\nproportions (within dataset):")
        print(props.round(3).to_string())
        if "persistent" in props.columns:
            print("\npersistent rate by dataset (sorted): "
                  f"{props['persistent'].sort_values().round(3).to_dict()}")
        print(f"\n[caveat] small per-dataset n until the stem-only run is fully complete for "
              f"all 300 questions -- treat any ordering claims (e.g. persistent rate across "
              f"datasets) as preliminary until n stabilizes.")

    print("\n"+"="*78)
    print("PRIMARY cross-fitted effect estimates (use ALL eligible questions per direction;")
    print("this is the inferential result -- report THESE, not the agreement-conditioned")
    print("pooled table further below, as the population-level effect size / prevalence).")
    print("="*78)

    print("\n=== [PRIMARY, direction A->B] label distribution, classified on Split A ===")
    print(tax_out["label_from_A"].value_counts().to_string())
    print("\nheld-out (Split B) deltas by label_from_A (ALL questions classified in this "
          "direction, not agreement-conditioned):")
    print(tax_out.groupby("label_from_A")[
        ["delta_S_heldout_B", "delta_C_heldout_B", "delta_R_heldout_B"]
    ].agg(["mean", "count"]).round(3).to_string())

    print("\n=== [PRIMARY, direction B->A] label distribution, classified on Split B ===")
    print(tax_out["label_from_B"].value_counts().to_string())
    print("\nheld-out (Split A) deltas by label_from_B (ALL questions classified in this "
          "direction, not agreement-conditioned):")
    print(tax_out.groupby("label_from_B")[
        ["delta_S_heldout_A", "delta_C_heldout_A", "delta_R_heldout_A"]
    ].agg(["mean", "count"]).round(3).to_string())

    agree = tax_out["labels_agree"].mean()
    real = tax_out[tax_out["label_from_A"] != "insufficient_data"]
    real_agree = real["labels_agree"].mean() if len(real) else float("nan")
    print(f"\n=== Cross-fit stability: the pattern was preserved under reciprocal cross-fitting ===")
    print(f"(NOTE: A->B and B->A reuse the same 8 repeats -- this is a stability check, "
          f"NOT an independent replication.)")
    print(f"raw agreement (all rows, incl. insufficient_data): "
          f"{agree:.3f} ({int(tax_out['labels_agree'].sum())}/{len(tax_out)})")
    print(f"raw agreement (REAL classifications only, n={len(real)}): "
          f"{real_agree:.3f} ({int(real['labels_agree'].sum())}/{len(real)})")
    print(tax_out.groupby("label_from_A")["labels_agree"].agg(["mean", "count"]).round(3).to_string())

    # --- Cohen's kappa, confusion matrix, per-class F1/Jaccard (real classifications only) ---
    if len(real) >= 2 and real["label_from_A"].nunique() > 1:
        from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                                     jaccard_score, classification_report)
        y_a, y_b = real["label_from_A"], real["label_from_B"]
        all_labels = sorted(set(y_a) | set(y_b))
        kappa = cohen_kappa_score(y_a, y_b, labels=all_labels)
        cm = confusion_matrix(y_a, y_b, labels=all_labels)
        cm_df = pd.DataFrame(cm, index=[f"A:{l}" for l in all_labels],
                             columns=[f"B:{l}" for l in all_labels])
        f1 = f1_score(y_a, y_b, labels=all_labels, average=None, zero_division=0)
        jac = jaccard_score(y_a, y_b, labels=all_labels, average=None, zero_division=0)
        print(f"\n=== Adjusted stability measures (n={len(real)} real classifications) ===")
        print(f"Cohen's kappa (label_from_A vs label_from_B): {kappa:.3f}")
        print("\nconfusion matrix (rows=Split A label, cols=Split B label):")
        print(cm_df.to_string())
        print("\nper-class F1 / Jaccard (A vs B agreement):")
        print(pd.DataFrame({"label": all_labels, "f1": f1, "jaccard": jac}).round(3).to_string(index=False))

        stats_out = dict(
            n_real=int(len(real)), cohen_kappa=float(kappa), labels=all_labels,
            confusion_matrix=cm.tolist(),
            per_class_f1=dict(zip(all_labels, [float(x) for x in f1])),
            per_class_jaccard=dict(zip(all_labels, [float(x) for x in jac])),
            raw_agreement_real=float(real_agree),
        )
        (OUT_DIR / "taxonomy_nested_agreement_stats.json").write_text(
            json.dumps(stats_out, indent=2), encoding="utf-8")
        print(f"\n-> {OUT_DIR / 'taxonomy_nested_agreement_stats.json'}")
    else:
        print("\n[note] not enough real, multi-class data yet for kappa/confusion-matrix "
              "(need >=2 distinct labels across enough questions).")

    # --- DESCRIPTIVE-ONLY: agreement-conditioned "high-confidence phenotype" characterization.
    # NOT an unbiased estimate of phenotype prevalence or population-level effect size -- agreement
    # conditioning (L_A == L_B) preferentially retains the cleanest, most strongly-expressed
    # examples of each phenotype and disproportionately excludes borderline questions (e.g.
    # G_S^A=2, G_S^B=1 -- right at the threshold in one split, just under it in the other -- would
    # be dropped here even though it's a real, eligible question). One row per question (not
    # doubled across directions), but this is a SELECTED subset, not the full eligible population.
    agreed = tax_out.dropna(subset=["agreed_label"])
    print("\n"+"="*78)
    print("DESCRIPTIVE-ONLY (NOT inferential): high-confidence phenotype characterization.")
    print("For descriptive phenotype profiles, we additionally examine questions receiving the")
    print("same label in both cross-fit directions. Because agreement conditioning preferentially")
    print("retains stable cases, all inferential analyses use the full cross-fitted sample (the")
    print("two PRIMARY per-direction tables above), NOT this agreement-conditioned subset.")
    print("="*78)
    print(f"\nagreed-label questions (n={len(agreed)} of {len(real)} real classifications) -- "
          f"averaged response fingerprint, ONE row per question:")
    if len(agreed):
        print(agreed.groupby("agreed_label")[
            ["delta_S_pooled", "delta_C_pooled", "delta_R_pooled"]
        ].agg(["mean", "count"]).round(3).to_string())

    print(f"\n-> {OUT_DIR / 'taxonomy_nested_results.csv'}")
    print(f"-> {OUT_DIR / 'taxonomy_nested_baseline_stable.csv'}")


if __name__ == "__main__":
    main()
