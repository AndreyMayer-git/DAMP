"""
DAMP — SECOND, competing generative model: a leaky competing accumulator (LCA;
Usher & McClelland, 2001) with lateral inhibition and a salience-weighted input.
No human data is involved. This model is the principled counterpart to the minimal
drift-diffusion model in dampsimulation.py: the two make OPPOSITE, falsifiable
predictions about the sign of DCI, and empirical data are needed to arbitrate
(model comparison in the sense of Wilson & Collins, 2019).

Two rectified accumulators y_L, y_R (>=0) race to drive the cursor left/right:
    dy_i = ( I_i - k*y_i - w*y_{other} ) dt + sigma*dW ,   y_i = max(y_i, 0)
    I_i  = I0 + beta*v_i + gamma*|v_i|
        I0     baseline drive (a neutral option is still chosen sometimes)
        beta   evaluative/approach weight (positive valence -> more drive)
        gamma  SALIENCE-CAPTURE weight: the |v_i| term gives BOTH strongly
               valenced stimuli extra drive regardless of sign (Ohman & Mineka,
               2001; Pessoa, 2005). This is the explicit hypothesis knob.
        k      leak ; w lateral inhibition.

Why this predicts DCI>0 (for gamma>0). For a Positive-Negative pair both options
get the +gamma*|v| boost, so both accumulators are driven hard and fight through
mutual inhibition -> prolonged competition -> curved trajectory -> high AUC. For a
Neutral-X pair only one option is salient -> it wins quickly -> straight -> low AUC.
At gamma=0 the salience term vanishes and the model reduces to an approach race in
which the Positive-Negative pair (largest evaluative gap) resolves FASTEST -> DCI<=0,
exactly like the minimal DDM. Thus the sign of DCI is governed by gamma, and the two
models bracket the empirical question.
"""
import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "dampsimulation.py"), encoding="utf-8").read().split("if __name__")[0])

I0      = 0.5
BETA_A  = 0.15
KLEAK   = 1.5
WINH    = 2.0
SIG_L   = 0.35
GAMMA   = 0.6
LEAN_SCALE = 1.3

def lca_latent(vL, vR, gamma):
    # Input = baseline + evaluative drive + salience capture. The gamma*|v| term gives BOTH
    # valenced stimuli high drive, so a Positive-Negative pair has two high, near-equal inputs
    # -> a slow winner-take-all fight through lateral inhibition -> curved trajectory (high AUC).
    # A Neutral-X pair has only one high input -> fast, straight resolution (low AUC).
    I_L = I0 + BETA_A * vL + gamma * abs(vL)
    I_R = I0 + BETA_A * vR + gamma * abs(vR)
    n = int(TMAX / DT)
    yL = np.zeros(n + 1); yR = np.zeros(n + 1)
    for t in range(n):
        dL = (I_L - KLEAK * yL[t] - WINH * yR[t]) * DT + SIG_L * np.sqrt(DT) * rng.standard_normal()
        dR = (I_R - KLEAK * yR[t] - WINH * yL[t]) * DT + SIG_L * np.sqrt(DT) * rng.standard_normal()
        yL[t + 1] = max(0.0, yL[t] + dL); yR[t + 1] = max(0.0, yR[t] + dR)
    lean = LEAN_SCALE * (yL - yR)
    sgn = np.sign(lean); flips = np.where(np.diff(sgn) != 0)[0]
    commit = (flips[-1] + 1) if len(flips) else 0
    return lean, ("left" if lean[-1] >= 0 else "right"), max(commit * DT, 0.05)

def lca_trial(left_cat, right_cat, gamma=GAMMA):
    vL, vR = VAL[left_cat], VAL[right_cat]
    lean, choice, dt_dec = lca_latent(vL, vR, gamma)
    h, v, target = make_trajectory(lean)
    a = auc(h, v, target)
    chosen = left_cat if choice == "left" else right_cat
    rt = dt_dec + MOTOR_T * (1 + 0.2 * rng.standard_normal()) * (1 + 0.5 * a)
    return dict(left=left_cat, right=right_cat, choice=choice, chosen=chosen,
                h=h, v=v, target=target, auc=a, rt=max(rt, 0.2),
                conflict=("high" if {left_cat, right_cat} == {"positive", "negative"} else "low"))

def lca_agent(gamma=GAMMA, pos_bias=0.0):
    main = []
    for cats in PAIR_TYPES:
        lc, rc = (cats if rng.random() < 0.5 else cats[::-1])
        main.append(lca_trial(lc, rc, gamma))
    swap_idx = rng.choice(len(main), size=10, replace=False)
    swaps = []
    for i in swap_idx:
        o = main[i]; tr = lca_trial(o["right"], o["left"], gamma); kept = (tr["chosen"] == o["chosen"])
        if rng.random() < pos_bias: kept = rng.random() < 0.5
        swaps.append(kept)
    ah = np.mean([t["auc"] for t in main if t["conflict"] == "high"])
    al = np.mean([t["auc"] for t in main if t["conflict"] == "low"])
    return dict(main=main, rsi=np.mean(swaps), dci=(ah - al) / al, auc_high=ah, auc_low=al)

if __name__ == "__main__":
    ag = [lca_agent(gamma=GAMMA, pos_bias=rng.uniform(0.05, 0.25)) for _ in range(24)]
    rsis = np.array([a["rsi"] for a in ag]); dcis = np.array([a["dci"] for a in ag])
    errs = np.mean([t["chosen"] == "negative" for a in ag for t in a["main"] if t["conflict"] == "high"])
    print(f"LCA (gamma={GAMMA}): RSI={rsis.mean():.3f}  DCI={dcis.mean():.3f} (>0:{np.mean(dcis>0)*100:.0f}%)"
          f"  AUC high={np.mean([a['auc_high'] for a in ag]):.3f} low={np.mean([a['auc_low'] for a in ag]):.3f}"
          f"  Pos-Neg 'chose negative'={errs*100:.0f}%")
    print("\ngamma sweep (mean DCI over 24 agents):")
    for g in [0.0, 0.3, 0.6, 0.9, 1.2]:
        d = np.mean([lca_agent(gamma=g)["dci"] for _ in range(24)])
        print(f"  gamma={g:.1f}  ->  DCI={d:+.3f}")
