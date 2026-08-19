"""generate_fig3_transition.py -- Figure 3A: regenerating the scaffold substantially reduces
fine-grained response stability.

We compare each genuine failure's ORIGINAL Split-A phenotype (assigned on the fixed brief used
throughout the main study) to its state under an INDEPENDENTLY REGENERATED brief (fresh seeds,
fresh control sample), re-classified with the identical G>=2-of-4 rule. The transition matrix
(rows = initial Split-A category, cols = fresh-brief state) shows exact counts, so the churn in
the responsive categories -- versus the stability of the persistent category -- is readable
directly (preferred over a Sankey for exact-count legibility).

Data (both scopes of the brief-regeneration robustness check):
    interventions/brief_regen_check/brief_regen_summary.csv            (39 originally responsive)
    interventions/brief_regen_check_persistent/brief_regen_summary.csv (98 originally persistent)

Usage: python generate_fig3_transition.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RESP_PATH = HERE / "interventions" / "brief_regen_check" / "brief_regen_summary.csv"
PERS_PATH = HERE / "interventions" / "brief_regen_check_persistent" / "brief_regen_summary.csv"
OUT_DIR = HERE / "interventions" / "claim_figures"

CATS = ["information-responsive", "choice-conditioned responsive",
        "discordant information response", "reasoning-responsive", "persistent"]
CAT_SHORT = {
    "information-responsive": "information-\nresponsive",
    "choice-conditioned responsive": "choice-\nconditioned",
    "discordant information response": "discordant\ninfo.",
    "reasoning-responsive": "reasoning-\nresponsive",
    "persistent": "persistent",
}
BLUES = LinearSegmentedColormap.from_list("seqblue", ["#f7fbff", "#08306b"])


def norm_fresh(lbl: str) -> str:
    # classify_info emits "neither (persistent under fresh brief)" for a stayed-persistent case
    return "persistent" if str(lbl).startswith("neither") else str(lbl)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resp = pd.read_csv(RESP_PATH)
    pers = pd.read_csv(PERS_PATH)
    df = pd.concat([resp, pers], ignore_index=True)
    df["initial"] = df["original_label_from_A"].astype(str)
    df["fresh"] = df["fresh_label"].map(norm_fresh)

    ct = pd.crosstab(df["initial"], df["fresh"]).reindex(index=CATS, columns=CATS).fillna(0).astype(int)
    row_tot = ct.sum(axis=1)
    prop = ct.div(row_tot.replace(0, np.nan), axis=0).fillna(0)

    # fine-grained stability = fraction whose fresh state == initial category
    diag = int(np.trace(ct.to_numpy()))
    total = int(ct.to_numpy().sum())
    resp_mask = df["initial"] != "persistent"
    resp_stable = int((df.loc[resp_mask, "initial"] == df.loc[resp_mask, "fresh"]).sum())
    resp_n = int(resp_mask.sum())
    pers_stable = int(((df["initial"] == "persistent") & (df["fresh"] == "persistent")).sum())
    pers_n = int((df["initial"] == "persistent").sum())

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    P = prop.to_numpy()
    C = ct.to_numpy()
    im = ax.imshow(P, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            c = C[i, j]
            if c == 0:
                continue
            ax.text(j, i, f"{c}\n{P[i, j]*100:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if P[i, j] > 0.55 else "#222222", fontweight="bold" if i == j else "normal")
    # row totals in the right margin
    for i, cat in enumerate(CATS):
        ax.annotate(f"n={row_tot[cat]}", (1.02, i), xycoords=("axes fraction", "data"),
                    va="center", fontsize=8, color="#666666")
    ax.set_xticks(range(len(CATS)))
    ax.set_yticks(range(len(CATS)))
    ax.set_xticklabels([CAT_SHORT[c] for c in CATS], fontsize=8)
    ax.set_yticklabels([CAT_SHORT[c] for c in CATS], fontsize=8)
    ax.set_xlabel("State under an independently regenerated brief", fontsize=9)
    ax.set_ylabel("Initial Split-A category (fixed brief)", fontsize=9)
    ax.set_title("Regenerating the scaffold reduces fine-grained response stability\n"
                 f"cells: count and row-% ;  persistent stays persistent {pers_stable}/{pers_n} "
                 f"({pers_stable/pers_n*100:.0f}%),  responsive keeps exact label {resp_stable}/{resp_n} "
                 f"({resp_stable/resp_n*100:.0f}%)",
                 fontsize=9.5, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.10)
    cb.set_label("P(fresh-brief state | initial category)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    png = OUT_DIR / "fig3_brief_regen_transition.png"
    pdf = OUT_DIR / "fig3_brief_regen_transition.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    # ---- console summary ----
    print("Transition matrix (rows = initial Split-A category, cols = fresh-brief state):")
    print(ct.to_string())
    print(f"\nExact-label stability overall: {diag}/{total} ({diag/total*100:.1f}%)")
    print(f"  persistent -> persistent:     {pers_stable}/{pers_n} ({pers_stable/pers_n*100:.1f}%)")
    print(f"  responsive keeps exact label: {resp_stable}/{resp_n} ({resp_stable/resp_n*100:.1f}%)")
    resp_stays_resp = int((df.loc[resp_mask, "fresh"] != "persistent").sum())
    print(f"  responsive stays responsive (any responsive label): "
          f"{resp_stays_resp}/{resp_n} ({resp_stays_resp/resp_n*100:.1f}%)")
    print(f"\n-> {png}\n-> {pdf}")


if __name__ == "__main__":
    main()
