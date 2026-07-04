import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl, os
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "dampsimulation.py"), encoding="utf-8").read().split("if __name__")[0])

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5b6472", "axes.linewidth": 0.9,
    "figure.dpi": 150, "savefig.dpi": 200})
INK="#1b2a4a"; HOT="#b8412f"; COOL="#2e6e9e"; MUT="#8a93a0"; GRID="#e8ebf1"
OUT=os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)


def hypothetical_conflict_trial(c0=12.0, tc=0.22):
    """A Positive-Negative trial with an added transient pull
    toward the aversive option (decaying competition term). This term is NOT part
    of the minimal Section 2.2 model; it is injected here solely to render a
    plausible high-conflict trajectory for Figure 3."""
    left_cat, right_cat = "positive", "negative"
    vL, vR = VAL[left_cat], VAL[right_cat]
    D_eval = BETA * (vL - vR)
    toward = 1.0 if vL < vR else -1.0
    n = int(TMAX / DT); x = np.empty(n + 1); x[0] = encode(vL, vR)
    for t in range(n):
        cap = c0 * toward * np.exp(-(t * DT) / tc)
        dx = (D_eval + cap) * DT + SIGMA * np.sqrt(DT) * rng.standard_normal()
        x[t + 1] = x[t] + dx
    h, v, target = make_trajectory(x)
    a = auc(h, v, target)
    rt = max(0.05, 0) + MOTOR_T * (1 + 0.5 * a)
    return dict(left=left_cat, right=right_cat, h=h, v=v, target=target, auc=a, rt=rt)


globals()["rng"] = np.random.default_rng(11)

# Figure 3: hypothetical high-conflict trajectory (illustrative) with a moderate hook
cand = []
for _ in range(600):
    tr = hypothetical_conflict_trial()
    e = tr["h"][:int(len(tr["h"]) * 0.6)]
    tr["_opp"] = (e * -np.sign(tr["target"])).max()
    cand.append(tr)
cand = [t for t in cand if 0.25 <= t["_opp"] <= 0.55]
cand.sort(key=lambda t: t["auc"], reverse=True)
fig3 = cand[len(cand) // 2]

# Figures 4-5: genuine low-conflict trajectories from the minimal model
lo = []
for _ in range(800):
    cats = ("neutral", "positive") if rng.random() < 0.5 else ("neutral", "negative")
    lo.append(run_trial(*cats))
lo.sort(key=lambda t: t["auc"])
fig4 = next(t for t in lo if t["target"] > 0)
fig5 = next(t for t in lo if t["target"] < 0)


def draw(ax, tr, color, title, sub):
    h, v, tgt = tr["h"], tr["v"], tr["target"]
    labels = {-1: tr["left"].capitalize(), 1: tr["right"].capitalize()}
    for xpos in (-1, 1):
        chosen = (np.sign(xpos) == np.sign(tgt))
        ax.scatter([xpos], [1], s=900, marker="s",
                   facecolor=(color if chosen else "#f3f5f9"),
                   alpha=(0.16 if chosen else 1), edgecolor=(color if chosen else MUT),
                   linewidth=1.6, zorder=2)
        ax.text(xpos, 1.16, labels[xpos], ha="center", fontsize=9.5,
                color=(color if chosen else MUT), fontweight=("bold" if chosen else "normal"))
    ax.plot([0, tgt], [0, 1], "--", color=MUT, lw=1.1, alpha=0.9, zorder=1, label="ideal direct path")
    ax.plot(h, v, "-", color=color, lw=2.6, zorder=3, label="cursor trajectory", solid_capstyle="round")
    ax.scatter([0], [0], s=70, color=INK, zorder=4); ax.text(0, -0.1, "start", ha="center", fontsize=9, color=INK)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.2, 1.32); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12.5, color=INK, pad=8)
    ax.text(0.97, 0.04, f"AUC = {tr['auc']:.2f}\nnorm. RT = {tr['rt']/MOTOR_T:.2f}",
            transform=ax.transAxes, fontsize=9.5, color="#333", va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    ax.text(0.5, -0.16, sub, transform=ax.transAxes, ha="center", fontsize=8.5, color=MUT)

f, ax = plt.subplots(figsize=(5.5, 4.9)); draw(ax, fig3, HOT, "Hypothetical high-conflict trajectory",
    "ILLUSTRATIVE ONLY — not produced by the minimal Section 2.2 model.\n"
    "Illustrates what the AUC index is designed to detect.")
f.tight_layout(); f.savefig(f"{OUT}/Figure3_high_conflict_trajectory.png", facecolor="white"); plt.close()

f, ax = plt.subplots(figsize=(5.5, 4.9)); draw(ax, fig4, COOL, "Low-conflict trajectory in a SWAP trial",
    "Simulated (minimal Section 2.2 model): direct, near-linear movement —\npreference maintained under spatial inversion.")
f.tight_layout(); f.savefig(f"{OUT}/Figure4_swap_normal_trajectory.png", facecolor="white"); plt.close()

f, ax = plt.subplots(figsize=(5.5, 4.9)); draw(ax, fig5, COOL, "Low-conflict trajectory (standard trial)",
    "Simulated (minimal Section 2.2 model): rapid, unambiguous resolution —\nlow lateral deviation, short normalised RT.")
f.tight_layout(); f.savefig(f"{OUT}/Figure5_normal_trajectory.png", facecolor="white"); plt.close()

globals()["rng"] = np.random.default_rng(20260619)
ag = [run_agent(pos_bias=rng.uniform(0.05, 0.25)) for _ in range(24)]
rsis = np.sort([a["rsi"] for a in ag]); N = len(rsis)
f, ax = plt.subplots(figsize=(6.8, 4.4))
ax.bar(np.arange(N), rsis, color=[COOL if r > 0.5 else HOT for r in rsis],
       edgecolor="white", linewidth=0.6, width=0.82, zorder=3)
ax.axhline(0.5, color=HOT, ls="--", lw=1.6, zorder=4, label="chance / position-driven (RSI = 0.5)")
ax.axhline(rsis.mean(), color=INK, lw=1.4, alpha=0.75, zorder=4, label=f"group mean RSI = {rsis.mean():.2f}")
ax.set_ylim(0, 1.05); ax.set_xlim(-0.7, N - 0.3); ax.set_xticks([])
ax.set_xlabel("Simulated participants (n = 24)", color=INK)
ax.set_ylabel("Representational Stability Index (RSI)", color=INK)
ax.set_title("Stability of preference across SWAP trials", fontsize=12.5, color=INK, pad=10)
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.96, edgecolor=GRID)
f.text(0.5, 0.005, "Simulated output (minimal Section 2.2 model): RSI exceeds 0.5 for every agent — stimulus-driven,\n"
       "not position-driven, preference. Proof of concept for the index, not a test of any hypothesis.",
       ha="center", fontsize=8.5, color=MUT)
f.tight_layout(rect=(0, 0.06, 1, 1)); f.savefig(f"{OUT}/Figure2_swap_stability.png", facecolor="white"); plt.close()

print("OK -> %s" % OUT)
print("Fig3(hypothetical) AUC=%.2f hook=%.2f | Fig4 AUC=%.2f (%s|%s) | Fig5 AUC=%.2f (%s|%s) | group RSI=%.2f" % (
    fig3["auc"], fig3["_opp"], fig4["auc"], fig4["left"], fig4["right"],
    fig5["auc"], fig5["left"], fig5["right"], rsis.mean()))
