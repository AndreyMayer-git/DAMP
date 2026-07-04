
import numpy as np, os, matplotlib.pyplot as plt, matplotlib as mpl
import damp_lca_model as M

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5b6472", "axes.linewidth": 0.9, "figure.dpi": 150, "savefig.dpi": 200})
INK = "#1b2a4a"; HOT = "#b8412f"; COOL = "#2e6e9e"; MUT = "#8a93a0"; GRID = "#e8ebf1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(OUT, exist_ok=True)

# ---- Panel A data: DCI vs gamma (LCA) and the DDM baseline ----
M.rng = np.random.default_rng(20260619)
gammas = np.linspace(0.0, 1.4, 15)
lca_mean = np.array([np.mean([M.lca_agent(gamma=g)["dci"] for _ in range(20)]) for g in gammas])
ddm_mean = np.mean([M.run_agent()["dci"] for _ in range(20)])

# ---- Panel B: an example high-conflict LCA trajectory with a visible (emergent) hook ----
M.rng = np.random.default_rng(7)
cands = [M.lca_trial("positive", "negative", gamma=0.6) for _ in range(500)]
for t in cands:
    e = t["h"][:int(len(t["h"]) * 0.6)]; t["_opp"] = (e * -np.sign(t["target"])).max()
cands = [t for t in cands if 0.20 <= t["_opp"] <= 0.55]
cands.sort(key=lambda t: t["auc"], reverse=True)
ex = cands[len(cands) // 2]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

axA.axhline(0, color=MUT, lw=1.0, zorder=1)
axA.plot(gammas, lca_mean, "-o", color=HOT, lw=2.4, ms=5, zorder=3,
         label="LCA (Model 2): salience-weighted competition")
axA.axhline(ddm_mean, color=COOL, ls="--", lw=2.0, zorder=2,
            label=f"minimal DDM (Model 1): DCI = {ddm_mean:+.2f}")
axA.set_xlabel("salience-capture weight  γ", color=INK)
axA.set_ylabel("Decision Conflict Index (DCI)", color=INK)
axA.set_title("Two models bracket the sign of DCI", fontsize=12.5, color=INK, pad=8)
axA.yaxis.grid(True, color=GRID, zorder=0); axA.set_axisbelow(True)
axA.legend(loc="upper left", fontsize=8.8, frameon=False)
axA.text(0.98, 0.03, "DDM predicts DCI ≤ 0;\nLCA predicts DCI > 0 once γ > 0",
         transform=axA.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUT)

h, v, tgt = ex["h"], ex["v"], ex["target"]
labels = {-1: ex["left"].capitalize(), 1: ex["right"].capitalize()}
for xpos in (-1, 1):
    chosen = (np.sign(xpos) == np.sign(tgt))
    axB.scatter([xpos], [1], s=760, marker="s", facecolor=(HOT if chosen else "#f3f5f9"),
                alpha=(0.16 if chosen else 1), edgecolor=(HOT if chosen else MUT), linewidth=1.5, zorder=2)
    axB.text(xpos, 1.15, labels[xpos], ha="center", fontsize=9.5,
             color=(HOT if chosen else MUT), fontweight=("bold" if chosen else "normal"))
axB.plot([0, tgt], [0, 1], "--", color=MUT, lw=1.1, zorder=1, label="ideal direct path")
axB.plot(h, v, "-", color=HOT, lw=2.6, zorder=3, label="cursor trajectory", solid_capstyle="round")
axB.scatter([0], [0], s=60, color=INK, zorder=4); axB.text(0, -0.1, "start", ha="center", fontsize=9, color=INK)
axB.set_xlim(-1.5, 1.5); axB.set_ylim(-0.2, 1.3); axB.set_xticks([]); axB.set_yticks([])
axB.set_title("High-conflict trajectory from the LCA dynamics", fontsize=12.5, color=INK, pad=8)
axB.text(0.97, 0.04, f"AUC = {ex['auc']:.2f}", transform=axB.transAxes, fontsize=9.5,
         color="#333", va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
axB.legend(loc="lower left", fontsize=8.5, frameon=False)
axB.text(0.5, -0.17, "Emergent (not injected): early excursion toward the aversive option, then correction.",
         transform=axB.transAxes, ha="center", fontsize=8.5, color=MUT)

fig.tight_layout()
fig.savefig(f"{OUT}/Figure6_model_comparison.png", facecolor="white"); plt.close()

print(f"Saved {OUT}/Figure6_model_comparison.png  | LCA example AUC={ex['auc']:.2f} hook={ex['_opp']:.2f}")
print("Power analysis: see damp_power_analysis.py")
