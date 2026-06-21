"""
Simulation-based power analysis for DAMP (Wilson & Collins, 2019, rule 8). No human
data. Question: if the affective-competition hypothesis holds (here instantiated by
the LCA, Model 2, at gamma=0.6), how many participants are needed to detect DCI > 0
at the group level?

Method: simulate a large pool of synthetic participants once to obtain the per-
participant DCI distribution the model implies, then bootstrap-resample experiments
of size N and apply a one-sided one-sample t-test against DCI = 0 (alpha = 0.05).
Power = proportion of resampled experiments that reach significance.
"""
import numpy as np
from math import sqrt, erf
import damp_lca_model as M

GAMMA = 0.6
POOL = 400
BOOT = 4000

M.rng = np.random.default_rng(20260619)
dci_pool = np.array([M.lca_agent(gamma=GAMMA)["dci"] for _ in range(POOL)])
print(f"Per-participant DCI under the LCA (gamma={GAMMA}): "
      f"mean={dci_pool.mean():.2f}, sd={dci_pool.std(ddof=1):.2f}, P(DCI>0)={np.mean(dci_pool>0)*100:.0f}%\n")

def one_sided_t_p(x):
    n = len(x); m = x.mean(); s = x.std(ddof=1)
    if s == 0:
        return 0.0 if m > 0 else 1.0
    t = m / (s / sqrt(n))
    return 1.0 - 0.5 * (1 + erf(t / sqrt(2)))

rng = np.random.default_rng(1)
print("Power to detect DCI > 0 (one-sided one-sample t-test, alpha=0.05):")
for N in [8, 12, 16, 20, 25, 30, 40]:
    sig = 0
    for _ in range(BOOT):
        sample = dci_pool[rng.integers(0, POOL, size=N)]
        if one_sided_t_p(sample) < 0.05:
            sig += 1
    print(f"  N = {N:2d}  ->  power = {sig / BOOT:.2f}")
