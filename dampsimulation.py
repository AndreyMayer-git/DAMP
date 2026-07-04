import numpy as np
rng = np.random.default_rng(20260619)

A_PHI, LAM = 0.12, 0.30
BETA       = 0.95
SIGMA      = 0.28
DT, TMAX   = 0.005, 1.10
MOTOR_T    = 0.55
VAL = {"neutral": 0.0, "positive": 2.0, "negative": -2.0}

def encode(vL, vR):
    x1 = A_PHI * vL
    return (1 - LAM) * x1 - A_PHI * vR

def simulate_latent(vL, vR, x0):
    D_eval = BETA * (vL - vR)
    n = int(TMAX / DT)
    x = np.empty(n + 1); x[0] = x0
    for t in range(n):
        dx = D_eval * DT + SIGMA * np.sqrt(DT) * rng.standard_normal()
        x[t + 1] = x[t] + dx
    sgn = np.sign(x); flips = np.where(np.diff(sgn) != 0)[0]
    commit = (flips[-1] + 1) if len(flips) else 0
    return x, ("left" if x[-1] >= 0 else "right"), max(commit * DT, 0.05)

def make_trajectory(x_latent):
    n = len(x_latent)
    commit = -np.tanh(0.8 * x_latent)
    pos = np.cumsum(commit) * DT
    pos -= pos[0]
    v = np.linspace(0, 1, n)
    target = -1.0 if x_latent[-1] >= 0 else 1.0
    h = pos / max(abs(pos[-1]), 1e-3)
    w_end = np.clip((v - 0.85) / 0.15, 0, 1)
    h = h * (1 - w_end) + target * w_end
    h = h + rng.normal(0, 0.008, n); h[0] = 0.0
    return h, v, target

def auc(h, v, target):
    return np.trapezoid(np.abs(h - target * v), v)

def run_trial(left_cat, right_cat):
    vL, vR = VAL[left_cat], VAL[right_cat]
    xl, choice, dt_dec = simulate_latent(vL, vR, encode(vL, vR))
    h, v, target = make_trajectory(xl)
    a = auc(h, v, target)
    chosen = left_cat if choice == "left" else right_cat
    rt = dt_dec + MOTOR_T * (1 + 0.2 * rng.standard_normal()) * (1 + 0.5 * a)
    return dict(left=left_cat, right=right_cat, choice=choice, chosen=chosen,
                h=h, v=v, target=target, auc=a, rt=max(rt, 0.2),
                conflict=("high" if {left_cat, right_cat} == {"positive", "negative"} else "low"))

PAIR_TYPES = ([("neutral", "positive")] * 17 + [("neutral", "negative")] * 17 +
              [("positive", "negative")] * 17)

def run_agent(pos_bias=0.0):
    rt_baseline = np.median([MOTOR_T * (1 + 0.2 * rng.standard_normal()) for _ in range(15)])
    main = []
    for cats in PAIR_TYPES:
        lc, rc = (cats if rng.random() < 0.5 else cats[::-1])
        main.append(run_trial(lc, rc))
    swap_idx = rng.choice(len(main), size=10, replace=False)
    swaps = []
    for i in swap_idx:
        o = main[i]; tr = run_trial(o["right"], o["left"])
        kept = (tr["chosen"] == o["chosen"])
        if rng.random() < pos_bias: kept = rng.random() < 0.5
        swaps.append(dict(orig=o, swap=tr, kept=kept))
    rsi = np.mean([s["kept"] for s in swaps])
    ah = np.mean([t["auc"] for t in main if t["conflict"] == "high"])
    al = np.mean([t["auc"] for t in main if t["conflict"] == "low"])
    return dict(main=main, swaps=swaps, rsi=rsi, dci=(ah - al) / al, auc_high=ah, auc_low=al,
                rt_baseline=rt_baseline)

if __name__ == "__main__":
    ag = [run_agent(pos_bias=rng.uniform(0.05, 0.25)) for _ in range(24)]
    rsis = np.array([a["rsi"] for a in ag]); dcis = np.array([a["dci"] for a in ag])
    errs = np.mean([t["chosen"] == "negative" for a in ag for t in a["main"] if t["conflict"] == "high"])
    print(f"RSI mean={rsis.mean():.3f} range={rsis.min():.2f}-{rsis.max():.2f} >0.5:{np.mean(rsis>0.5)*100:.0f}%")
    print(f"DCI mean={dcis.mean():.3f} >0:{np.mean(dcis>0)*100:.0f}%  (minimal model expects DCI<=0)")
    print(f"AUC high={np.mean([a['auc_high'] for a in ag]):.3f} low={np.mean([a['auc_low'] for a in ag]):.3f}")
    print(f"Pos-Neg 'chose negative' rate={errs*100:.0f}%")
