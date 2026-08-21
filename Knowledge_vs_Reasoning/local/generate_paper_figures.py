"""Generate paper-ready figures for debate recovery by failure type analysis."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

HERE = Path(r"C:\Proj1\Knowledge_vs_Reasoning\local")
OUT_DIR = HERE / "interventions" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Palette (from dataviz skill reference) ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

TYPE_COLORS = {"stochastic": BLUE, "knowledge": ORANGE, "hard": AQUA}
TYPE_LABELS = {"stochastic": "Stochastic", "knowledge": "Knowledge-limited", "hard": "Hard/unrecoverable"}

# --- Data ---
labels = pd.read_csv(HERE / "interventions" / "intervention_labels.csv")
LABEL_MAP = {
    "stochastic-recoverable": "stochastic",
    "knowledge-limited": "knowledge",
    "hard/unrecoverable": "hard",
    "reasoning-limited": "hard",
    "ambiguous": "hard",
    "both-sufficient": "hard",
    "interaction (both needed)": "hard",
}
labels["label_3"] = labels["label"].map(LABEL_MAP)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "sans-serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig1_recovery_by_type():
    """Figure 1: Vanilla debate recovery rate by failure type (the main result)."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    types = ["stochastic", "knowledge", "hard"]
    means = [labels[labels["label_3"] == t]["control"].mean() for t in types]
    stds = [labels[labels["label_3"] == t]["control"].std() for t in types]
    ns = [len(labels[labels["label_3"] == t]) for t in types]
    ses = [s / np.sqrt(n) for s, n in zip(stds, ns)]

    x = np.arange(len(types))
    bars = ax.bar(x, means, width=0.55, color=[TYPE_COLORS[t] for t in types],
                  edgecolor="none", zorder=3)

    # 4px rounded data-end (approximate with bar rounding)
    for bar in bars:
        bar.set_linewidth(0)

    # Error bars (SE)
    ax.errorbar(x, means, yerr=ses, fmt="none", ecolor=INK_SECONDARY,
                capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)

    # Value labels at bar tips
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 0.03, f"{m:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="medium", color=INK_PRIMARY)
        ax.text(i, -0.06, f"n={n}", ha="center", va="top",
                fontsize=7, color=INK_MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[t] for t in types], color=INK_PRIMARY)
    ax.set_ylabel("Recovery rate (control condition)", color=INK_SECONDARY)
    ax.set_ylim(-0.08, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.axhline(0, color=INK_MUTED, linewidth=0.8, zorder=2)
    ax.grid(axis="x", visible=False)
    ax.set_title("Debate recovery rate by causal failure type", pad=12,
                 fontweight="medium", color=INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_recovery_by_type.png", dpi=300, bbox_inches="tight",
                facecolor=SURFACE)
    fig.savefig(OUT_DIR / "fig1_recovery_by_type.pdf", bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved fig1_recovery_by_type.png/pdf")


def fig2_scaffold_lift():
    """Figure 2: Scaffold lift over control, grouped by failure type."""
    fig, ax = plt.subplots(figsize=(7, 4))

    types = ["stochastic", "knowledge", "hard"]
    scaffolds = ["knowledge_blind", "knowledge_oracle", "reasoning", "both_blind", "both_oracle"]
    scaffold_labels = ["Knowledge\n(blind)", "Knowledge\n(oracle)", "Reasoning", "Both\n(blind)", "Both\n(oracle)"]

    n_types = len(types)
    n_scaffolds = len(scaffolds)
    bar_width = 0.22
    group_width = n_types * bar_width + 0.08

    for i, t in enumerate(types):
        subset = labels[labels["label_3"] == t]
        ctrl = subset["control"].mean()
        lifts = [subset[s].mean() - ctrl for s in scaffolds]
        x = np.arange(n_scaffolds) + i * bar_width - (n_types - 1) * bar_width / 2
        bars = ax.bar(x, lifts, width=bar_width - 0.02, color=TYPE_COLORS[t],
                      edgecolor="none", label=TYPE_LABELS[t], zorder=3)

    ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=2)
    ax.set_xticks(np.arange(n_scaffolds))
    ax.set_xticklabels(scaffold_labels, color=INK_PRIMARY)
    ax.set_ylabel("Lift over control (pp)", color=INK_SECONDARY)
    ax.set_ylim(-0.30, 0.85)
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["-20", "0", "+20", "+40", "+60", "+80"])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor=GRIDLINE)
    ax.set_title("Scaffold lift over vanilla debate by failure type", pad=12,
                 fontweight="medium", color=INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_scaffold_lift.png", dpi=300, bbox_inches="tight",
                facecolor=SURFACE)
    fig.savefig(OUT_DIR / "fig2_scaffold_lift.pdf", bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved fig2_scaffold_lift.png/pdf")


def fig3_recovery_heatmap():
    """Figure 3: Recovery rate heatmap (type x condition)."""
    fig, ax = plt.subplots(figsize=(6, 2.8))

    types = ["stochastic", "knowledge", "hard"]
    conditions = ["control", "knowledge_blind", "knowledge_oracle", "reasoning", "both_blind", "both_oracle"]
    cond_labels = ["Control", "Know.\n(blind)", "Know.\n(oracle)", "Reason.", "Both\n(blind)", "Both\n(oracle)"]

    data = np.zeros((len(types), len(conditions)))
    for i, t in enumerate(types):
        subset = labels[labels["label_3"] == t]
        for j, c in enumerate(conditions):
            data[i, j] = subset[c].mean()

    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(cond_labels, color=INK_PRIMARY)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([TYPE_LABELS[t] for t in types], color=INK_PRIMARY)

    for i in range(len(types)):
        for j in range(len(conditions)):
            val = data[i, j]
            color = "white" if val > 0.5 else INK_PRIMARY
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, fontweight="medium", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("Recovery rate", fontsize=8, color=INK_SECONDARY)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_title("Recovery rate: failure type x intervention condition", pad=12,
                 fontweight="medium", color=INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_recovery_heatmap.png", dpi=300, bbox_inches="tight",
                facecolor=SURFACE)
    fig.savefig(OUT_DIR / "fig3_recovery_heatmap.pdf", bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved fig3_recovery_heatmap.png/pdf")


def fig4_dynamics_indistinguishable():
    """Figure 4: Debate dynamics are indistinguishable across failure types."""
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    features = pd.read_csv(HERE / "interventions" / "features_multiseed.csv")
    features["label_3"] = features["label"].map(LABEL_MAP)

    types = ["stochastic", "knowledge", "hard"]

    panels = [
        ("consensus_stability", "Consensus stability\n(fraction of seeds reaching consensus)"),
        ("any_switch_mean", "Switch rate\n(fraction of seeds with any answer change)"),
        ("answer_stability", "Answer stability\n(fraction of seeds with same final answer)"),
    ]

    for ax, (col, title) in zip(axes, panels):
        positions = []
        data_groups = []
        for i, t in enumerate(types):
            vals = features[features["label_3"] == t][col].dropna().values
            data_groups.append(vals)
            positions.append(i)

        bp = ax.boxplot(data_groups, positions=positions, widths=0.5,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=3, alpha=0.4),
                        medianprops=dict(color=INK_PRIMARY, linewidth=1.5))

        for patch, t in zip(bp["boxes"], types):
            patch.set_facecolor(TYPE_COLORS[t])
            patch.set_alpha(0.7)
            patch.set_edgecolor(INK_SECONDARY)
            patch.set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels([TYPE_LABELS[t].split("/")[0] for t in types],
                          fontsize=7, color=INK_PRIMARY)
        ax.set_title(title, fontsize=8, color=INK_PRIMARY, pad=8)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(-0.05, 1.15)

    fig.suptitle("Debate dynamics are indistinguishable across failure types (all p > 0.16)",
                 fontsize=9, fontweight="medium", color=INK_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_dynamics_indistinguishable.png", dpi=300,
                bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(OUT_DIR / "fig4_dynamics_indistinguishable.pdf",
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved fig4_dynamics_indistinguishable.png/pdf")


if __name__ == "__main__":
    print("Generating paper-ready figures...")
    fig1_recovery_by_type()
    fig2_scaffold_lift()
    fig3_recovery_heatmap()
    fig4_dynamics_indistinguishable()
    print("\nDone. All figures in:", OUT_DIR)
