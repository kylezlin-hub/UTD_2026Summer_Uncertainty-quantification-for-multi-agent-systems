from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import itertools

import numpy as np
import pandas as pd


METRICS: List[Tuple[str, str]] = [
    ("Num. Rounds", "rounds_to_consensus"),
    ("Nash-Stability", "stability"),
    ("Agent Utility", "group_welfare"),
    ("Engagement", "engagement"),
    ("Responsiveness", "responsiveness"),
    ("Balance", "balance"),
    ("Influence Asym. (inv.)", "influence_asymmetry_inv"),
    ("Avg. Process Metrics", "avg_process_metrics"),
]


FIXTURE_PREFERENCE_RANK: Dict[str, int] = {
    "premature_consensus": 0,
    "dogmatic": 1,
    "balanced_gradual": 2,
    "productive_pluralism": 3,
}


def workbook_label(path: Path) -> str:
    try:
        metadata = pd.read_excel(path, sheet_name="Run_Metadata")
    except ValueError:
        return "Workbook"
    fields = dict(zip(metadata["field"], metadata["value"]))
    model = fields.get("model_id", "Workbook")
    source = fields.get("dataset_source", "")
    return f"{model} {source}".strip()


def spearman_with_accuracy(scores: pd.DataFrame, label_column: str) -> pd.DataFrame:
    objective = scores[scores["dataset_type"] == "objective"].copy()
    rows = []
    for label, column in METRICS:
        paired = objective[[column, "accuracy"]].copy()
        paired[column] = pd.to_numeric(paired[column], errors="coerce")
        paired["accuracy"] = pd.to_numeric(paired["accuracy"], errors="coerce")
        paired = paired.dropna()
        rho = np.nan
        if len(paired) >= 3 and paired[column].nunique() > 1 and paired["accuracy"].nunique() > 1:
            rho = float(np.corrcoef(paired[column].rank(), paired["accuracy"].rank())[0, 1])
        rows.append({"Signal / Metric": label, label_column: rho, "n": len(paired)})
    return pd.DataFrame(rows)


def pairwise_fixture_preference(scores: pd.DataFrame, label_column: str) -> pd.DataFrame:
    subjective = scores[scores["dataset_type"] == "subjective"].copy()
    subjective["fixture_preference_rank"] = subjective["fixture_pattern"].map(FIXTURE_PREFERENCE_RANK)
    rows = []
    for label, column in METRICS[1:]:
        correct = []
        for (_, left), (_, right) in itertools.combinations(subjective.iterrows(), 2):
            left_rank = left["fixture_preference_rank"]
            right_rank = right["fixture_preference_rank"]
            if pd.isna(left_rank) or pd.isna(right_rank) or left_rank == right_rank:
                continue
            left_score = left[column]
            right_score = right[column]
            if pd.isna(left_score) or pd.isna(right_score) or left_score == right_score:
                continue
            correct.append(float((left_rank > right_rank) == (left_score > right_score)))
        rows.append(
            {
                "Signal / Metric": label,
                label_column: float(np.mean(correct)) if correct else np.nan,
                "n_pairs": len(correct),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print PDF Table 1/Table 2 analogues from a generated methodology workbook."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/qwen_mmlu_pro_200/qwen_mmlu_pro_debate_traces.xlsx"),
    )
    args = parser.parse_args()

    label = workbook_label(args.input)
    scores = pd.read_excel(args.input, sheet_name="Diagnostic_Scores")
    print("Table 1 analogue: Spearman rho with objective accuracy")
    print(spearman_with_accuracy(scores, label).to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print()
    subjective = scores[scores["dataset_type"] == "subjective"]
    if subjective.empty:
        print("Table 2 analogue: skipped because this workbook has no subjective rows.")
    else:
        print("Table 2 analogue: pairwise accuracy with fixture preference proxy, not human labels")
        print(pairwise_fixture_preference(scores, label).to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
