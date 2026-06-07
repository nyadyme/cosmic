#!/usr/bin/env python3
"""
Multiverse on a NONLINEAR multi-time-scale task (NARMA-10/20/30)
======================================================================
So far: linear memory capacity (delayed copy). Here the cross-check on a
NONLINEAR task with a selectable memory horizon — NARMA-n (standard
benchmark): y(t+1) depends nonlinearly on the last n inputs/outputs.
Higher order n = longer nonlinear memory required.

Hypothesis: If the modular time-scale isolation (multiverse) structurally brings
something, then it does so more strongly here than for linear MC — and the lead
should GROW with the order. Fair control as before: flat / mv2 / mv3 at the same
size (N=720), same edge count, IDENTICAL leak vector and input.

Metric: NRMSE (smaller = better; 1.0 = trivial mean predictor).
Reservoir building blocks from simulation_multiverse3.py.
"""

import numpy as np
import sys
import time
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

import simulation_multiverse3 as m3   # build, graded_leak
import simulation_insitu as si        # (spectral_scale via m3)

V_IN = 6
LEAF, NMOD, MG = 60, 12, 4            # N=720, 12 modules, mv3 = 3 groups of 4
N = LEAF * NMOD
ORDERS = [10, 20, 30]
N_SAMP, WASHOUT = 6000, 300
SEEDS = [0, 1, 2]
ALPHA = 1e-3
DISCLAIMER = "AI-assisted -- please reproduce independently"


def build_win_vec(n_mod, mod):
    rng = np.random.default_rng(0)             # fixed -> identical across topologies
    win = np.zeros(n_mod * mod)
    for k in range(n_mod):
        for s in range(V_IN):
            win[k * mod + 10 + s] = rng.choice([-1.0, 1.0])
    return win


def states(W, leak, win, sig):
    Wc = W.tocsr(); X = np.zeros((len(sig), len(win))); x = np.zeros(len(win))
    for t in range(len(sig)):
        x = (1 - leak) * x + leak * np.tanh(Wc.dot(x) + win * sig[t]); X[t] = x
    return X


def narma_target(u, order):
    L = len(u); y = np.zeros(L)
    for t in range(order, L - 1):
        y[t + 1] = np.tanh(0.3 * y[t] + 0.05 * y[t] * y[t - order + 1:t + 1].sum()
                           + 1.5 * u[t - order + 1] * u[t] + 0.1)
    return y


def nrmse(X, y):
    Xw, yw = X[WASHOUT:], y[WASHOUT:]
    mu, sd = Xw.mean(0), Xw.std(0) + 1e-8
    Xs = (Xw - mu) / sd
    n = int(0.7 * len(Xs))
    r = Ridge(alpha=ALPHA); r.fit(Xs[:n], yw[:n]); pred = r.predict(Xs[n:])
    return float(np.sqrt(np.mean((pred - yw[n:]) ** 2)) / (yw[n:].std() + 1e-12))


def evaluate(topo):
    leak = m3.graded_leak(NMOD, LEAF)
    win = build_win_vec(NMOD, LEAF)
    per = {o: [] for o in ORDERS}
    for sd in SEEDS:
        W = m3.build(topo, NMOD, LEAF, np.random.default_rng(100 + sd),
                     M=(MG if topo == "mv3" else None))
        u = np.random.default_rng(7 + sd).uniform(0.0, 0.5, N_SAMP)
        X = states(W, leak, win, u)            # once per (topo, seed)
        for o in ORDERS:
            per[o].append(nrmse(X, narma_target(u, o)))
    return {o: (float(np.mean(per[o])), float(np.std(per[o]))) for o in ORDERS}


def main():
    t0 = time.time()
    print("Multiverse on a nonlinear task (NARMA-10/20/30)")
    print(f"  N={N} ({NMOD}×{LEAF}), {len(SEEDS)} seeds, NRMSE (smaller=better)\n")
    res = {t: evaluate(t) for t in ("flat", "mv2", "mv3")}

    print(f"  {'Order':>8} {'flat':>14} {'mv2':>14} {'mv3':>14}")
    print(f"  {'-'*54}")
    for o in ORDERS:
        f = res['flat'][o]; m2 = res['mv2'][o]; m3v = res['mv3'][o]
        print(f"  NARMA-{o:<3} {f[0]:>7.3f}±{f[1]:<5.3f} {m2[0]:>7.3f}±{m2[1]:<5.3f} "
              f"{m3v[0]:>7.3f}±{m3v[1]:<5.3f}")
    print(f"  {'-'*54}")

    # Figure: grouped bars (NRMSE per order, per topology)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(ORDERS)); w = 0.26
    for i, (t, col) in enumerate([("flat", "C3"), ("mv2", "C2"), ("mv3", "C0")]):
        ax.bar(x + (i - 1) * w, [res[t][o][0] for o in ORDERS], w, label=t, color=col)
    ax.set_xticks(x); ax.set_xticklabels([f"NARMA-{o}" for o in ORDERS])
    ax.set_ylabel("NRMSE (smaller = better)")
    ax.set_title("Multiverse on a nonlinear task (N=720, 3 seeds)\n"
                 "flat vs mv2 vs mv3 — same size/edges/leak/input")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic", color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig_multiverse4.png", dpi=150)
    plt.close(fig)
    print("  Figure: fig_multiverse4.png")

    print(f"\n{'='*56}\nVERDICT\n{'='*56}")
    gaps = {o: res['flat'][o][0] - res['mv2'][o][0] for o in ORDERS}  # >0 = mv2 better
    print(f"  Lead of mv2 (NRMSE reduction vs flat) per order:")
    for o in ORDERS:
        print(f"    NARMA-{o}: {gaps[o]:+.3f}  (mv3 vs mv2: "
              f"{res['mv2'][o][0]-res['mv3'][o][0]:+.3f})")
    grow = gaps[ORDERS[-1]] - gaps[ORDERS[0]]
    if all(g > 0.02 for g in gaps.values()) and grow > 0.02:
        print("  -> Hierarchy helps, AND the lead GROWS with the order")
        print("     (structural multi-time-scale benefit — unlike for linear MC).")
    elif all(g > 0.02 for g in gaps.values()):
        print("  -> Hierarchy helps, but ~constant across the order (no")
        print("     specific long-term nonlinear bonus).")
    else:
        print("  -> No clear hierarchy advantage on the nonlinear task.")
    print(f"\n  Runtime: {time.time()-t0:.0f}s\n{'='*56}")


if __name__ == "__main__":
    main()
