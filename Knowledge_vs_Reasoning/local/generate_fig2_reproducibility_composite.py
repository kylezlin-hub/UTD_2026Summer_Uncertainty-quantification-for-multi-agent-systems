"""generate_fig2_reproducibility_composite.py -- Paper Figure 2:
Fixed-scaffold reproducibility of the 5-way response taxonomy. Two 5x5 Split-A -> Split-B
transition matrices side by side (Qwen | Llama), row-normalized (cell color = P(Split-B label |
Split-A label); the diagonal is reproducibility), with raw counts annotated. Single-brief
fixed-scaffold construct for BOTH models (matched), read from the stored taxonomy artifacts.

Data: interventions/taxonomy_nested_agreement_stats.json (Qwen),
      interventions_llama_fixed/taxonomy_nested_agreement_stats.json (Llama).
Output: interventions/results/fig2_reproducibility_composite.{png,pdf}
Usage: python generate_fig2_reproducibility_composite.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
OUT = HERE / "interventions" / "results"; OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Qwen2.5-7B": HERE / "interventions" / "taxonomy_nested_agreement_stats.json",
    "Llama-3.1-8B": HERE / "interventions_llama_fixed" / "taxonomy_nested_agreement_stats.json",
}
# stored label order in the JSONs
STORED = ["choice-conditioned responsive", "discordant information response",
          "information-responsive", "persistent", "reasoning-responsive"]
# desired display order + short labels
DESIRED = ["information-responsive", "choice-conditioned responsive",
           "discordant information response", "reasoning-responsive", "persistent"]
SHORT = ["Information", "Choice-cond.", "Discordant", "Reasoning", "Persistent"]
ORDER = [STORED.index(x) for x in DESIRED]

fig = plt.figure(figsize=(11.5, 5.2), layout="constrained")
gs = GridSpec(1, 2, figure=fig, wspace=0.12)
cmap = plt.get_cmap("Blues")

for k, (name, path) in enumerate(MODELS.items()):
    st = json.loads(Path(path).read_text(encoding="utf-8"))
    M = np.array(st["confusion_matrix"])[np.ix_(ORDER, ORDER)]        # reorder rows+cols
    n = int(st["n_real"]); diag = int(np.trace(M)); kap = st["cohen_kappa"]
    rown = M.sum(1, keepdims=True)
    R = np.divide(M, rown, out=np.zeros_like(M, float), where=rown > 0)  # row-normalized P(B|A)
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(R, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for i in range(5):
        for j in range(5):
            if M[i, j] == 0:
                txt = "·"
            else:
                txt = f"{M[i, j]}\n{R[i, j]*100:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if R[i, j] > 0.55 else "#222",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(SHORT, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_xlabel("Split-B label", fontsize=9)
    if k == 0:
        ax.set_ylabel("Split-A label", fontsize=9)
    letter = "AB"[k]
    ax.set_title(f"{letter}. {name} — Agreement {diag/n*100:.1f}%, κ = {kap:.3f}",
                 fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)

cb = fig.colorbar(im, ax=fig.axes, fraction=0.035, pad=0.02)
cb.set_label("Row-normalized transition probability", fontsize=8.5)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig2_reproducibility_composite.{ext}", dpi=200, bbox_inches="tight")
print(f"-> {OUT / 'fig2_reproducibility_composite.png'}")
for name, path in MODELS.items():
    st = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"   {name}: {int(np.trace(np.array(st['confusion_matrix'])))}/{st['n_real']} "
          f"= {st['raw_agreement_real']*100:.1f}%  kappa={st['cohen_kappa']:.3f}")
