"""
Supplementary checks backing reviewer responses (no human data):
  1. Is the sign of DCI under the minimal DDM robust to sigma? (it should be: the
     reviewer worried that poor sigma identifiability undercuts the DCI prediction)
  2. Is the high-conflict AUC distribution bimodal? (Freeman: averaging a mixture of
     straight + mid-flight-reversal trajectories yields a misleading smooth mean)
  3. Power via the DIFFERENCE contrast (AUC_high - AUC_low) rather than the fragile
     ratio DCI, expressed via a standardized effect size (decouples power from the
     model's self-chosen magnitude).
"""
import numpy as np
from math import sqrt, erf
import dampsimulation as ds
import damp_lca_model as M


print("1) Minimal DDM: mean DCI across sigma x beta (sign should stay <= 0)")
for beta in [0.70, 0.95, 1.20]:
    row = []
    for sigma in [0.18, 0.25, 0.32, 0.40]:
        ds.BETA = beta; ds.SIGMA = sigma; ds.rng = np.random.default_rng(20260619)
        d = np.mean([ds.run_agent()["dci"] for _ in range(24)])
        row.append("s=%.2f:%+.2f" % (sigma, d))
    print("   beta=%.2f | %s" % (beta, "  ".join(row)))
ds.BETA, ds.SIGMA = 0.95, 0.28  # restore


M.rng = np.random.default_rng(20260619)
hi = np.array([M.lca_trial("positive", "negative", gamma=0.6)["auc"] for _ in range(2000)])
n = len(hi); m = hi.mean(); s = hi.std(ddof=1)
g1 = np.mean(((hi - m) / s) ** 3)
g2 = np.mean(((hi - m) / s) ** 4) - 3.0
BC = (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
print("\n2) High-conflict AUC distribution (LCA, gamma=0.6): n=%d mean=%.3f sd=%.3f" % (n, m, s))
print("   Sarle bimodality coefficient BC = %.3f  (>0.555 suggests non-unimodal / mixture)" % BC)


M.rng = np.random.default_rng(20260619)
pool = [M.lca_agent(gamma=0.6) for _ in range(400)]
diff = np.array([a["auc_high"] - a["auc_low"] for a in pool])
ratio = np.array([a["dci"] for a in pool])
d_diff = diff.mean() / diff.std(ddof=1)
print("\n3) Per-participant contrasts under the LCA (gamma=0.6):")
print("   difference AUC_high-AUC_low: mean=%.3f sd=%.3f  Cohen's d=%.2f" % (diff.mean(), diff.std(ddof=1), d_diff))
print("   ratio DCI:                   mean=%.2f sd=%.2f  Cohen's d=%.2f  (heavier-tailed)" %
      (ratio.mean(), ratio.std(ddof=1), ratio.mean() / ratio.std(ddof=1)))

def power_for_d(d, N, reps=20000, rng=np.random.default_rng(1)):
    crit = 1.645

    hits = 0
    for _ in range(reps):
        x = rng.standard_normal(N) + d
        t = x.mean() / (x.std(ddof=1) / sqrt(N))
        if t > crit:
            hits += 1
    return hits / reps

print("   Power (one-sided, alpha=0.05) by N for a range of standardized effect sizes d:")
print("       d \\ N |  10    16    20    30")
for d in [0.3, 0.5, d_diff, 0.8]:
    ps = [power_for_d(d, N) for N in [10, 16, 20, 30]]
    tag = " (model-implied)" if abs(d - d_diff) < 1e-9 else ""
    print("      %.2f   | %s%s" % (d, "  ".join("%.2f" % p for p in ps), tag))
