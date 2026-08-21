"""generate_fig3_transition.py -- Figure 3: the central contrast figure.

Figure 2 shows response patterns look highly reproducible when the scaffold (brief) is held
fixed. Figure 3 shows what changes when the scaffold itself is INDEPENDENTLY REGENERATED
(fresh seeds, fresh control sample, re-classified with the identical G>=2-of-4 rule):

  Panel A: informational-response cases (n=36) transition matrix -- initial Split-A category
           (rows) vs. state under a fresh scaffold (cols). Fine-grained repair-category
           replication is low (11/36 = 30.6%), but most cases stay responsive to >=1 fresh
           informational intervention (24/36 = 66.7%): the repair ROUTE changes more than
           whether repair happens.
  Panel B: initially persistent cases (n=98) -- a single stacked outcome bar. Persistent
           non-response largely reproduces (87/98 = 88.8%).

Deliberately: no 94.9% (that is Figure 2's fixed-scaffold result), no Sankey, no
original-vs-fresh magnitude scatter. Exact counts + row-% in every occupied cell.

Data:
    interventions/brief_regen_check/brief_regen_summary.csv            (36 informational + 3 reasoning)
    interventions/brief_regen_check_persistent/brief_regen_summary.csv (98 originally persistent)
Usage: python generate_fig3_transition.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RESP_PATH = HERE / "interventions" / "brief_regen_check" / "brief_regen_summary.csv"
PERS_PATH = HERE / "interventions" / "brief_regen_check_persistent" / "brief_regen_summary.csv"
OUT_DIR = HERE / "interventions" / "claim_figures"

# Panel A: the three informational-response origin categories (exclude the 3 reasoning-responsive,
# whose label derives from a static reasoning scaffold, not a regenerated informational brief).
ROW_CATS = ["information-responsive", "choice-conditioned responsive",
            "discordant information response"]
COL_CATS = ["information-responsive", "choice-conditioned responsive",
            "discordant information response", "neither"]
ROW_SHORT = {
    "information-responsive": "information-\nresponsive",
    "choice-conditioned responsive": "choice-\nconditioned",
    "discordant information response": "discordant\ninfo.",
}
COL_SHORT = {
    "information-responsive": "info-\nresponsive",
    "choice-conditioned responsive": "choice-\nconditioned",
    "discordant information response": "discordant\ninfo.",
    "neither": "neither info.-\nresponsive",
}
# Panel B outcome order + Okabe-Ito CVD-safe palette (persistent = dominant grey-blue).
PERS_ORDER = ["persistent", "choice-conditioned responsive",
              "information-responsive", "discordant information response"]
PERS_COLOR = {
    "persistent": "#4C6B8A",
    "choice-conditioned responsive": "#E69F00",
    "information-responsive": "#0072B2",
    "discordant information response": "#009E73",
}
PERS_LABEL = {
    "persistent": "persistent",
    "choice-conditioned responsive": "choice-conditioned",
    "information-responsive": "information-responsive",
    "discordant information response": "discordant-info.",
}
BLUES = LinearSegmentedColormap.from_list("seqblue", ["#f7fbff", "#08306b"])


def norm_fresh(lbl: str) -> str:
    """classify_info emits 'neither (persistent under fresh brief)' for a stayed-persistent case."""
    return "neither" if str(lbl).startswith("neither") else str(lbl)


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return centre - half, centre + half


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resp = pd.read_csv(RESP_PATH)
    pers = pd.read_csv(PERS_PATH)
    resp["initial"] = resp["original_label_from_A"].astype(str)
    resp["fresh"] = resp["fresh_label"].map(norm_fresh)
    pers["fresh"] = pers["fresh_label"].map(norm_fresh)  # 'neither' -> persistent bucket below

    # ---- Panel A: 36 informational-response cases (exclude reasoning-responsive) ----
    info = resp[resp["initial"].isin(ROW_CATS)].copy()
    ct = pd.crosstab(info["initial"], info["fresh"]).reindex(index=ROW_CATS, columns=COL_CATS).fillna(0).astype(int)
    row_tot = ct.sum(axis=1)
    prop = ct.div(row_tot.replace(0, np.nan), axis=0).fillna(0)
    n_info = int(row_tot.sum())
    exact = int(sum(ct.loc[c, c] for c in ROW_CATS))                      # exact same-label
    stay_resp = int(info["fresh"].isin(ROW_CATS).sum())                   # any informational label
    e_lo, e_hi = wilson_ci(exact, n_info)
    r_lo, r_hi = wilson_ci(stay_resp, n_info)

    # ---- Panel B: 98 persistent cases ----
    pcanon = pers["fresh"].map(lambda x: "persistent" if x == "neither" else x)
    pcounts = pcanon.value_counts()
    n_pers = int(len(pers))
    n_stay = int(pcounts.get("persistent", 0))
    p_lo, p_hi = wilson_ci(n_stay, n_pers)
    # zero-gain-under-both among stable-persistent (frozen Audit Q6 value)
    stab = pers[pcanon == "persistent"]
    zero_both = int(((stab["G_S_fresh"] == 0) & (stab["G_C_fresh"] == 0)).sum())

    # ============================ figure ============================
    fig = plt.figure(figsize=(13.2, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # -------- Panel A: transition heatmap --------
    P = prop.to_numpy()
    C = ct.to_numpy()
    im = axA.imshow(P, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(len(ROW_CATS)):
        for j in range(len(COL_CATS)):
            c = C[i, j]
            if c == 0:
                continue
            axA.text(j, i, f"{P[i, j]:.2f}\n(n={c})", ha="center", va="center", fontsize=8.5,
                     color="white" if P[i, j] > 0.55 else "#222222")
        # thin, neutral outline marking the same-label (diagonal) cell -- publication-style,
        # not a "target" highlight; the low diagonal mass is already clear from the layout.
        axA.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#555555",
                                lw=0.9, zorder=5))
        axA.annotate(f"n={row_tot.iloc[i]}", (1.02, i), xycoords=("axes fraction", "data"),
                     va="center", fontsize=8, color="#666666")
    axA.set_xticks(range(len(COL_CATS)))
    axA.set_yticks(range(len(ROW_CATS)))
    axA.set_xticklabels([COL_SHORT[c] for c in COL_CATS], fontsize=8)
    axA.set_yticklabels([ROW_SHORT[c] for c in ROW_CATS], fontsize=8)
    axA.set_xlabel("State under an independently regenerated brief", fontsize=9)
    axA.set_ylabel("Initial Split-A category (fixed brief)", fontsize=9)
    axA.set_title("A. Informational-response categories change\nacross regenerated scaffolds\n"
                  f"Exact category replication = {exact}/{n_info} = {exact/n_info*100:.1f}% "
                  f"(95% CI {e_lo*100:.1f}–{e_hi*100:.1f}%)",
                  fontsize=9.5, loc="left", fontweight="bold")
    ROUTE_MSG = (f"{stay_resp}/{n_info} = {stay_resp/n_info*100:.1f}% "
                 f"(95% CI {r_lo*100:.1f}–{r_hi*100:.1f}%) remain responsive "
                 f"to ≥1 regenerated informational intervention")
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.14)
    cb.set_label("P(fresh state | initial category)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # -------- Panel B: persistence stacked outcome bar --------
    left = 0.0
    recovered_bits = []
    for cat in PERS_ORDER:
        n = int(pcounts.get(cat, 0))
        if n == 0:
            continue
        w = n / n_pers
        axB.barh(0, w, left=left, height=0.5, color=PERS_COLOR[cat], edgecolor="white", lw=1.5)
        if cat == "persistent":
            axB.text(left + w / 2, 0, f"{PERS_LABEL[cat]}\n{n} ({w*100:.1f}%)", ha="center",
                     va="center", fontsize=11, color="white", fontweight="bold")
        else:
            recovered_bits.append((cat, n, w))  # summarize the small segments below the bar
        left += w
    axB.set_xlim(0, 1)
    axB.set_ylim(-1.15, 0.55)
    axB.set_yticks([])
    axB.set_xlabel("Fraction of the 98 initially persistent questions", fontsize=9)
    axB.set_title("B. Persistent non-response largely reproduces\n"
                  f"{n_stay}/{n_pers} = {n_stay/n_pers*100:.1f}%  (95% CI {p_lo*100:.1f}–{p_hi*100:.1f}%)",
                  fontsize=9.5, loc="left", fontweight="bold")
    for s in ("top", "right", "left"):
        axB.spines[s].set_visible(False)
    # colour-swatch summary of the 11 recovered cases, laid out horizontally below the bar
    n_rec = sum(n for _, n, _ in recovered_bits)
    axB.text(0.0, -0.42, f"Switch to an informational-response category: {n_rec}/{n_pers}",
             ha="left", va="top", fontsize=8.5, color="#333333", fontweight="bold",
             transform=axB.transData)
    xsw = 0.0
    for cat, n, _ in recovered_bits:
        axB.add_patch(Rectangle((xsw, -0.66), 0.03, 0.10, color=PERS_COLOR[cat], clip_on=False))
        txt = f"{PERS_LABEL[cat]} {n}"
        axB.text(xsw + 0.045, -0.61, txt, ha="left", va="center", fontsize=8, color="#333333")
        xsw += 0.045 + 0.017 * len(txt)
    axB.text(0.0, -0.85, f"{zero_both}/{n_stay} stable-persistent cases show zero gain under BOTH "
             f"regenerated informational\ninterventions; none reaches the responsiveness threshold "
             f"(max fresh gain < +2/4).", ha="left", va="top", fontsize=8, color="#333333")

    # ---- deliberate 30.6% vs 88.8% contrast strip, explicitly labelled (different denominators) ----
    fig.suptitle("Fine-grained response categories are less reproducible across regenerated "
                 "scaffolds, while persistent non-response remains stable",
                 fontsize=12, fontweight="bold", y=1.03)
    # key secondary message for Panel A, centered under it, clear of the footnote
    fig.text(0.30, 0.005, ROUTE_MSG, ha="center", va="top", fontsize=8.5,
             color="#c1435c", fontweight="bold")
    fig.text(0.5, -0.075,
             f"Exact fine-grained response-category reproduction: {exact}/{n_info} = "
             f"{exact/n_info*100:.1f}%   vs.   persistent non-response reproduction: "
             f"{n_stay}/{n_pers} = {n_stay/n_pers*100:.1f}%   "
             "(distinct outcome definitions and denominators; not the same population).\n"
             "Qwen2.5-7B-Instruct; independently regenerated briefs, fresh control sample, "
             "identical G≥2-of-4 classification rule; Wilson 95% CIs.",
             ha="center", va="top", fontsize=7.6, color="#555555")

    png = OUT_DIR / "fig3_brief_regen_transition.png"
    pdf = OUT_DIR / "fig3_brief_regen_transition.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    # ---- console summary ----
    print("Panel A transition matrix (rows = initial, cols = fresh state):")
    print(ct.to_string())
    print(f"\nExact repair-category replication: {exact}/{n_info} = {exact/n_info*100:.1f}% "
          f"(Wilson {e_lo*100:.1f}-{e_hi*100:.1f}%)")
    print(f"Remain responsive to >=1 fresh informational intervention: {stay_resp}/{n_info} = "
          f"{stay_resp/n_info*100:.1f}% (Wilson {r_lo*100:.1f}-{r_hi*100:.1f}%)")
    print(f"\nPanel B persistent outcomes (n={n_pers}): {dict(pcounts)}")
    print(f"Persistent reproduction: {n_stay}/{n_pers} = {n_stay/n_pers*100:.1f}% "
          f"(Wilson {p_lo*100:.1f}-{p_hi*100:.1f}%)")
    print(f"Zero gain under both fresh informational interventions: {zero_both}/{n_stay}")
    print(f"\n-> {png}\n-> {pdf}")


if __name__ == "__main__":
    main()
