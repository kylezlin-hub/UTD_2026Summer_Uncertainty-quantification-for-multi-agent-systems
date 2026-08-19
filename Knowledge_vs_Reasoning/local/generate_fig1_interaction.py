"""Figure 1: The interaction — scaffolding reverses sign across failure types.

Slope graph: Control → Knowledge Oracle, one line per failure type.
Stochastic goes DOWN, Knowledge-limited goes UP, Hard stays flat.
Communicates the entire paper in five seconds.

Uses held-out data: Split-A labels, Split-B recovery rates.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

HERE = Path(r"C:\Proj1\Knowledge_vs_Reasoning\local")
SOLVE_PATH = HERE / "interventions" / "solve_results.jsonl"
OUT_DIR = HERE / "interventions" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#888888"

records = []
with open(SOLVE_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

data = defaultdict(lambda: defaultdict(list))
for r in records:
    qno = str(r["question_no"])
    data[qno][r["condition"]].append({
        "repeat": r["repeat"],
        "correct": r["correct"],
    })

questions = sorted(data.keys())
split_a = {0, 1, 2, 3}
split_b = {4, 5, 6, 7}


def get_rate(qno, cond, reps):
    runs = [r for r in data[qno][cond] if r["repeat"] in reps]
    return np.mean([r["correct"] for r in runs]) if runs else 0.0


def assign_labels(questions, reps):
    labels = {}
    margin, stoch_thresh, hard_thresh = 0.34, 0.5, 0.2
    for qno in questions:
        ctrl = get_rate(qno, "control", reps)
        kb = get_rate(qno, "knowledge_blind", reps)
        ko = get_rate(qno, "knowledge_oracle", reps)
        reas = get_rate(qno, "reasoning", reps)
        bb = get_rate(qno, "both_blind", reps)
        bo = get_rate(qno, "both_oracle", reps)
        best = max(ctrl, kb, ko, reas, bb, bo)
        if ctrl >= stoch_thresh:
            raw = "stochastic"
        elif best < hard_thresh:
            raw = "hard"
        elif (kb - ctrl) >= margin or (ko - ctrl) >= margin:
            raw = "knowledge" if not ((reas - ctrl) >= margin) else "hard"
        elif (reas - ctrl) >= margin:
            raw = "hard"
        elif (bb - ctrl) >= margin or (bo - ctrl) >= margin:
            raw = "knowledge" if ((kb - ctrl) >= margin or (ko - ctrl) >= margin) else "hard"
        else:
            raw = "hard"
        labels[qno] = raw
    return labels


labels = assign_labels(questions, split_a)


def bootstrap_ci(values, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


# Compute held-out rates and CIs
types_config = [
    ("stochastic", "Stochastic-recoverable", BLUE),
    ("knowledge", "Knowledge-limited", ORANGE),
    ("hard", "Hard / persistent", GREY),
]

results = {}
for lbl, label_name, color in types_config:
    qs = [q for q in questions if labels[q] == lbl]
    ctrl_rates = np.array([get_rate(q, "control", split_b) for q in qs])
    ko_rates = np.array([get_rate(q, "knowledge_oracle", split_b) for q in qs])
    ctrl_mean = np.mean(ctrl_rates)
    ko_mean = np.mean(ko_rates)
    ctrl_ci = bootstrap_ci(ctrl_rates)
    ko_ci = bootstrap_ci(ko_rates)
    results[lbl] = {
        "n": len(qs),
        "ctrl_mean": ctrl_mean, "ko_mean": ko_mean,
        "ctrl_ci": ctrl_ci, "ko_ci": ko_ci,
        "label": label_name, "color": color,
    }

# --- Figure ---
fig, ax = plt.subplots(figsize=(5.5, 5.5))

x_ctrl = 0
x_ko = 1

for lbl, label_name, color in types_config:
    r = results[lbl]
    ctrl_y = r["ctrl_mean"] * 100
    ko_y = r["ko_mean"] * 100
    ctrl_ci = r["ctrl_ci"] * 100
    ko_ci = r["ko_ci"] * 100

    lw = 2.8 if lbl != "hard" else 1.8
    ls = "-" if lbl != "hard" else "--"
    alpha = 1.0 if lbl != "hard" else 0.6

    # Line
    ax.plot([x_ctrl, x_ko], [ctrl_y, ko_y],
            color=color, linewidth=lw, linestyle=ls, alpha=alpha, zorder=3)

    # Error bars at each endpoint
    ax.errorbar(x_ctrl, ctrl_y,
                yerr=[[ctrl_y - ctrl_ci[0]], [ctrl_ci[1] - ctrl_y]],
                fmt="o", color=color, markersize=8, capsize=4,
                capthick=1.5, elinewidth=1.5, zorder=4)
    ax.errorbar(x_ko, ko_y,
                yerr=[[ko_y - ko_ci[0]], [ko_ci[1] - ko_y]],
                fmt="o", color=color, markersize=8, capsize=4,
                capthick=1.5, elinewidth=1.5, zorder=4)

    # Labels on right side — offset to avoid overlap at ~78-80%
    delta = ko_y - ctrl_y
    sign = "+" if delta > 0 else ""
    if lbl == "stochastic":
        offset_y = 6
    elif lbl == "knowledge":
        offset_y = -10
    else:
        offset_y = -4
    ax.annotate(
        f"{label_name} ({sign}{delta:.0f}pp, n={r['n']})",
        xy=(x_ko, ko_y), xytext=(x_ko + 0.06, ko_y + offset_y),
        fontsize=9, color=color, fontweight="bold",
        va="center", ha="left",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5)
            if abs(offset_y) > 5 else None,
    )

# Axis formatting
ax.set_xticks([x_ctrl, x_ko])
ax.set_xticklabels(["Control\n(debate only)", "Oracle knowledge\nscaffold"],
                    fontsize=11, fontweight="bold")
ax.set_xlim(-0.3, 2.35)
ax.set_ylim(-5, 105)
ax.set_ylabel("Recovery rate (%)", fontsize=12)
ax.set_title("Effect of knowledge scaffolding\nreverses across failure types",
             fontsize=13, fontweight="bold", pad=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

# Annotation: text box in the open middle region (no arrow)
interaction = (results["knowledge"]["ko_mean"] - results["knowledge"]["ctrl_mean"]) - \
              (results["stochastic"]["ko_mean"] - results["stochastic"]["ctrl_mean"])
box_text = (
    "Difference in treatment effects\n"
    f"+{interaction*100:.1f} pp\n"
    "95% CI [+71.7, +103.8]\n"
    r"$p_{\mathrm{perm}} < 2 \times 10^{-5}$"
)
ax.text(0.5, 45, box_text,
        fontsize=8.5, va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#aaaaaa", linewidth=0.8, alpha=0.95),
        linespacing=1.5)

# Subtitle
ax.text(0.5, -0.12,
        "Labels and outcomes estimated from disjoint repeat sets (0\u20133 vs. 4\u20137).\n"
        "Error bars: 95% CI via question-level bootstrap (10,000 resamples).",
        transform=ax.transAxes, fontsize=7.5, color="#666666",
        ha="center", va="top")

plt.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUT_DIR / f"fig1_interaction.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / f'fig1_interaction.{ext}'}")

plt.close()
print("Done.")
