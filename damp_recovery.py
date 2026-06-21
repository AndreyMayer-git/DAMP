"""
Parameter recovery for the minimal DDM (Model 1), Wilson & Collins (2019), rule 5.
No human data. We generate synthetic datasets with KNOWN parameters (drift weight
beta, diffusion noise sigma), fit the model back from summary statistics, and check
that the recovered parameters track the true ones. Successful recovery shows the
model is identifiable from the observables the paradigm collects.

The minimal model has no leak, so the latent path is a drift-diffusion random walk
and can be simulated in closed vectorised form (exact, fast). Summary statistics
used for fitting: mean AUC, SD of AUC, mean normalised RT (these separate beta --
which speeds commitment and straightens paths -- from sigma -- which adds trajectory
variability). Fitting = nearest point on a (beta, sigma) grid in standardised
summary space.
"""
import numpy as np, os, matplotlib.pyplot as plt, matplotlib as mpl
import dampsimulation as ds

A_PHI, LAM, DT, TMAX, MOTOR_T, VAL, PAIR_TYPES = (
    ds.A_PHI, ds.LAM, ds.DT, ds.TMAX, ds.MOTOR_T, ds.VAL, ds.PAIR_TYPES)
N = int(TMAX / DT); SQ = np.sqrt(DT)

def encode(vL, vR):
    return (1 - LAM) * A_PHI * vL - A_PHI * vR

def latent_vec(vL, vR, beta, sigma):
    D = beta * (vL - vR)
    incr = D * DT + sigma * SQ * ds.rng.standard_normal(N)
    x = np.empty(N + 1); x[0] = encode(vL, vR); x[1:] = x[0] + np.cumsum(incr)
    return x

def trial(lc, rc, beta, sigma):
    x = latent_vec(VAL[lc], VAL[rc], beta, sigma)
    h, v, tg = ds.make_trajectory(x)
    a = ds.auc(h, v, tg)
    sgn = np.sign(x); flips = np.where(np.diff(sgn) != 0)[0]
    dt_dec = max(((flips[-1] + 1) if len(flips) else 0) * DT, 0.05)
    rt = dt_dec + MOTOR_T * (1 + 0.2 * ds.rng.standard_normal()) * (1 + 0.5 * a)
    return a, max(rt, 0.2) / MOTOR_T

def summary(beta, sigma, n_agents=6):
    aucs = []; rts = []
    for _ in range(n_agents):
        for cats in PAIR_TYPES:
            lc, rc = (cats if ds.rng.random() < 0.5 else cats[::-1])
            a, rt = trial(lc, rc, beta, sigma)
            aucs.append(a); rts.append(rt)
    return np.array([np.mean(aucs), np.std(aucs), np.mean(rts)])

# ---- build the (beta, sigma) grid in standardised summary space ----
betas = np.linspace(0.5, 1.4, 12); sigmas = np.linspace(0.15, 0.45, 9)
ds.rng = np.random.default_rng(20260619)
gp, gs = [], []
for b in betas:
    for s in sigmas:
        gp.append((b, s)); gs.append(np.mean([summary(b, s) for _ in range(3)], axis=0))
gp = np.array(gp); gs = np.array(gs)
mu, sd = gs.mean(0), gs.std(0)
gz = (gs - mu) / sd

# ---- recover from fresh synthetic datasets with known true parameters ----
ds.rng = np.random.default_rng(123)
true, rec = [], []
for _ in range(50):
    bt = ds.rng.uniform(0.55, 1.35); st = ds.rng.uniform(0.17, 0.43)
    z = (summary(bt, st) - mu) / sd
    j = int(np.argmin(((gz - z) ** 2).sum(1)))
    true.append((bt, st)); rec.append(gp[j])
true = np.array(true); rec = np.array(rec)

rb = np.corrcoef(true[:, 0], rec[:, 0])[0, 1]
rs = np.corrcoef(true[:, 1], rec[:, 1])[0, 1]
print(f"Parameter recovery (n=50 synthetic datasets, 6 agents each):")
print(f"  beta : r(true, recovered) = {rb:.2f}")
print(f"  sigma: r(true, recovered) = {rs:.2f}")

# ---- figure ----
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5b6472", "axes.linewidth": 0.9, "figure.dpi": 150, "savefig.dpi": 200})
INK = "#1b2a4a"; COOL = "#2e6e9e"; MUT = "#8a93a0"; GRID = "#e8ebf1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(OUT, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, i, name, r, lim in [(axes[0], 0, "β (drift weight)", rb, (0.45, 1.45)),
                            (axes[1], 1, "σ (diffusion noise)", rs, (0.13, 0.47))]:
    ax.plot(lim, lim, "--", color=MUT, lw=1.2, zorder=1)
    ax.scatter(true[:, i], rec[:, i], s=46, color=COOL, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(f"true {name}", color=INK); ax.set_ylabel(f"recovered {name}", color=INK)
    ax.set_title(f"{name.split()[0]}:  r = {r:.2f}", fontsize=12.5, color=INK, pad=8)
    ax.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
fig.suptitle("Parameter recovery — minimal DDM (Model 1)", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{OUT}/Figure7_parameter_recovery.png", facecolor="white"); plt.close()
print(f"Saved {OUT}/Figure7_parameter_recovery.png")
