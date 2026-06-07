#!/usr/bin/env python3
"""
Multiverse variant test -- how far can the approach be pushed?
====================================================================
Follow-up experiment to simulation_multiverse.py. Tests the "tuned" variants
that could theoretically help (multi-time-scales, input everywhere), AGAINST fair
baselines of the same size and the same time-scale spread.

Key fairness: a time-scale gradient (slow + fast nodes) raises the memory
capacity of EVERY reservoir — including the flat one. Hence the baseline
`flat_tau` (flat reservoir with the same leak spread). Only if the
hierarchy `mv_tuned` beats `flat_tau` does the macro structure itself contribute.

Variants (N=600, ~same edges, edge-of-chaos, 3 seeds):
  flat          -- 1 reservoir, fixed leak rate                 (reference)
  flat_tau      -- 1 reservoir, leak SPREAD (multi-time-scale)   (fair baseline)
  ensemble_tau  -- K blocks, input everywhere, leak spread, no loops
  mv_chain      -- K blocks + macro chain, input only block 0, fixed leak (naive)
  mv_tuned      -- K blocks + macro chain, input everywhere, time-scale GRADIENT
                   (block 0 fast ... block K-1 slow)             (best possible)
"""

import numpy as np
import scipy.sparse as sp
import sys
import time
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier

import simulation_insitu as si

V = si.V
RHO, LEAK, WASHOUT, ALPHA = 0.9, 0.2, 200, 1.0
LEAK_FAST, LEAK_SLOW = 0.6, 0.05          # time-scale range (fast..slow)
K, N_U = 6, 100
N = K * N_U
E_BLOCK, Q = 400, 4
LAGS = [1, 2, 3, 5, 8, 12, 16, 20, 25]
N_SAMP = 6000
SEEDS = [0, 1, 2]
chance = 1.0 / V
DISCLAIMER = "AI-assisted -- please reproduce independently"


def _add(R, C, D, lo, n, ne, rng, seen):
    cnt = tries = 0
    while cnt < ne and tries < ne * 30:
        i = lo + int(rng.integers(0, n)); j = lo + int(rng.integers(0, n)); tries += 1
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j)); R.append(i); C.append(j)
        D.append(float(rng.standard_normal())); cnt += 1


def build_reservoir(topo, rng):
    R, C, D, seen = [], [], [], set()
    if topo == "flat":
        _add(R, C, D, 0, N, K * E_BLOCK + (K - 1) * Q, rng, seen)
    else:
        for k in range(K):
            _add(R, C, D, k * N_U, N_U, E_BLOCK, rng, seen)
        if topo == "mv":
            for k in range(K - 1):
                for i in range(Q):
                    s = k * N_U + (N_U - 1 - i); d = (k + 1) * N_U + i
                    if (d, s) not in seen:
                        seen.add((d, s)); R.append(d); C.append(s)
                        D.append(float(rng.standard_normal()))
    return si.spectral_scale(sp.csr_matrix((D, (R, C)), shape=(N, N)), RHO)


def build_win(inp_kind):
    R, C, D = [], [], []
    blocks = range(K) if inp_kind == "all" else [0]
    for k in blocks:
        for s in range(V):
            R.append(k * N_U + 10 + s); C.append(s); D.append(1.0)
    return sp.csr_matrix((D, (R, C)), shape=(N, V))


def build_leak(kind, rng):
    if kind == "fixed":
        return np.full(N, LEAK)
    if kind == "spread":                  # per-node log-uniform, unstructured
        lo, hi = np.log(LEAK_SLOW), np.log(LEAK_FAST)
        return np.exp(rng.uniform(lo, hi, N))
    # "graded": block 0 fast ... block K-1 slow + slight spread
    bases = np.exp(np.linspace(np.log(LEAK_FAST), np.log(LEAK_SLOW), K))
    leak = np.empty(N)
    for k in range(K):
        jit = np.exp(rng.normal(np.log(bases[k]), 0.2, N_U))
        leak[k * N_U:(k + 1) * N_U] = np.clip(jit, LEAK_SLOW * 0.5, 0.95)
    return leak


def racc(X, tgt):
    Xw, yw = X[WASHOUT:], tgt[WASHOUT:]
    Xs = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-8)
    n = int(0.7 * len(Xs))
    clf = RidgeClassifier(alpha=ALPHA); clf.fit(Xs[:n], yw[:n])
    return clf.score(Xs[n:], yw[n:])


def mem_cap(X, inp):
    per = {}
    for lag in LAGS:
        t = np.zeros(len(inp), dtype=int); t[lag:] = inp[:-lag]
        per[lag] = max(racc(X, t) - chance, 0.0)
    return sum(per.values()) / (1 - chance), per


def evaluate(topo, inp_kind, leak_kind):
    totals, pers = [], []
    for sd in SEEDS:
        W = build_reservoir(topo, np.random.default_rng(100 + sd))
        Win = build_win(inp_kind)
        leak = build_leak(leak_kind, np.random.default_rng(500 + sd))
        inp = np.random.default_rng(7 + sd).integers(0, V, N_SAMP)
        X = si.esn_forward(W, leak, Win, inp, N)
        t, per = mem_cap(X, inp); totals.append(t); pers.append(per)
    per_mean = {lag: float(np.mean([p[lag] for p in pers])) for lag in LAGS}
    return float(np.mean(totals)), float(np.std(totals)), per_mean


CONFIGS = [
    ("flat",         "flat",   "single", "fixed"),    # reference
    ("flat_tau",     "flat",   "single", "spread"),   # only time-scale (6 inputs)
    ("flat_match",   "flat",   "all",    "graded"),   # = mv_tuned WITHOUT modular structure
    ("ensemble_tau", "blocks", "all",    "spread"),   # blocks, no loops
    ("mv_tuned",     "mv",     "all",    "graded"),   # tuned multiverse
]


def make_figure(res):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    sty = {"flat": ("o:", "0.5"), "flat_tau": ("o-", "C0"),
           "flat_match": ("D-", "C3"), "ensemble_tau": ("s-", "C1"),
           "mv_tuned": ("^-", "C2")}
    for name, (_, _, per) in res.items():
        fmt, col = sty[name]
        ax.plot(LAGS, [per[l] for l in LAGS], fmt, color=col, label=name)
    ax.set_xlabel("delay (lag)")
    ax.set_ylabel("reconstructable memory (acc − chance)")
    ax.set_title("Multiverse variant test: tuned vs fair baselines\n"
                 "(N=600, same size, 3 seeds) — fair reference: flat_tau")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic",
             color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig_multiverse2.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    print("Multiverse variant test (how far can it be pushed?)")
    print(f"  N={N} ({K}×{N_U}), RHO={RHO}, leak range [{LEAK_SLOW},{LEAK_FAST}], "
          f"{len(SEEDS)} Seeds\n")
    res = {n: evaluate(t, i, l) for (n, t, i, l) in CONFIGS}

    long_lags = [l for l in LAGS if l >= 16]
    print(f"  {'Variant':<14} {'MC total':>12}  {'long lags (≥16)':>18}")
    print(f"  {'-'*48}")
    for n, _, _, _ in CONFIGS:
        tot, sd, per = res[n]
        print(f"  {n:<14} {tot:>7.2f}±{sd:<4.2f}  "
              f"{sum(per[l] for l in long_lags):>18.3f}")
    print(f"  {'-'*48}")

    make_figure(res)
    print("  Figure: fig_multiverse2.png")

    fl = res["flat"][0]; ft = res["flat_tau"][0]
    fm = res["flat_match"][0]; mt = res["mv_tuned"][0]
    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"  Time-scale effect (applies to ALL): flat {fl:.2f} -> "
          f"flat_tau {ft:.2f} ({ft-fl:+.2f})")
    print(f"  Input fan-in + time-scale (flat): flat_match {fm:.2f}")
    print(f"  DECISIVE (same leak + same input, only connectivity):")
    print(f"     mv_tuned {mt:.2f}  vs  flat_match {fm:.2f}  ({mt-fm:+.2f})")
    if mt > fm + 0.3:
        print("  -> The modular structure contributes ITSELF (mv_tuned > flat_match).")
    elif mt >= fm - 0.3:
        print("  -> Tie: the gain comes from multi-time-scale + input fan-in,")
        print("     NOT from the hierarchy. Flat of the same size achieves the same.")
    else:
        print("  -> The hierarchy is even below the flat counterpart.")
    print(f"\n  Runtime: {time.time()-t0:.0f}s\n{'='*60}")


if __name__ == "__main__":
    main()
