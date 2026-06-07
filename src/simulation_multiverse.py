#!/usr/bin/env python3
"""
Multiverse test -- does recursive coupling (macro loops) bring more memory?
============================================================================
Tests the idea of coupling several small reservoirs ("universes") via a few
surface ports ("quasars") and thereby introducing a higher hierarchy level.
Honest test with a FAIR baseline (lesson from B-3):

  flat       -- ONE reservoir with N nodes (connected), input to V nodes.
  ensemble   -- K independent blocks (NO macro links), input to ALL blocks;
                states concatenated (= pure parallelism, upper bound without loops).
  multiverse -- K blocks + macro chain (quasar_k -> input_{k+1}), input only to block 0.

All three: SAME node count N and ~same edge count, same spectral radius
(edge of chaos), same leak rate. Metric: memory capacity over short AND long
lags (the claimed advantage would be LONG memory). Averaged over 3 seeds.

Intra-topology: random-sparse — because the fractal topology brings no
computational benefit (B-3); this isolates the macro-hierarchy effect at exactly
the same size. Uses the leaky-ESN mechanics from simulation_insitu.
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

import simulation_insitu as si   # esn_forward, spectral_scale, V, GAIN

V = si.V
RHO, LEAK, WASHOUT, ALPHA = 0.9, 0.2, 200, 1.0
K, N_U = 6, 100
N = K * N_U
E_BLOCK = 400                    # directed edges per block
Q = 4                            # macro links per bridge (quasar ports)
LAGS = [1, 2, 3, 5, 8, 12, 16, 20, 25]
N_SAMP = 6000
SEEDS = [0, 1, 2]
chance = 1.0 / V
DISCLAIMER = "AI-assisted -- please reproduce independently"


def _add_block(R, C, D, lo, n, ne, rng, seen):
    cnt = tries = 0
    while cnt < ne and tries < ne * 30:
        i = lo + int(rng.integers(0, n)); j = lo + int(rng.integers(0, n)); tries += 1
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j)); R.append(i); C.append(j)
        D.append(float(rng.standard_normal())); cnt += 1


def build_reservoir(kind, rng):
    R, C, D, seen = [], [], [], set()
    if kind == "flat":
        _add_block(R, C, D, 0, N, K * E_BLOCK + (K - 1) * Q, rng, seen)
    else:
        for k in range(K):
            _add_block(R, C, D, k * N_U, N_U, E_BLOCK, rng, seen)
        if kind == "multiverse":           # macro chain: quasar_k -> input_{k+1}
            for k in range(K - 1):
                for i in range(Q):
                    s = k * N_U + (N_U - 1 - i)
                    d = (k + 1) * N_U + i
                    if (d, s) not in seen:
                        seen.add((d, s)); R.append(d); C.append(s)
                        D.append(float(rng.standard_normal()))
    W = sp.csr_matrix((D, (R, C)), shape=(N, N))
    return si.spectral_scale(W, RHO)


def build_win(kind):
    R, C, D = [], [], []
    if kind == "flat":
        for s in range(V):
            R.append(10 + s); C.append(s); D.append(1.0)
    elif kind == "multiverse":             # input only in block 0
        for s in range(V):
            R.append(10 + s); C.append(s); D.append(1.0)
    else:                                  # ensemble: input in EVERY block
        for k in range(K):
            for s in range(V):
                R.append(k * N_U + 10 + s); C.append(s); D.append(1.0)
    return sp.csr_matrix((D, (R, C)), shape=(N, V))


def racc(X, tgt):
    Xw, yw = X[WASHOUT:], tgt[WASHOUT:]
    Xs = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-8)
    n = int(0.7 * len(Xs))
    clf = RidgeClassifier(alpha=ALPHA)
    clf.fit(Xs[:n], yw[:n])
    return clf.score(Xs[n:], yw[n:])


def mem_cap(X, inp):
    per = {}
    for lag in LAGS:
        t = np.zeros(len(inp), dtype=int); t[lag:] = inp[:-lag]
        per[lag] = max(racc(X, t) - chance, 0.0)
    return sum(per.values()) / (1 - chance), per


def evaluate(kind):
    leak = np.full(N, LEAK)
    totals, pers = [], []
    for sd in SEEDS:
        W = build_reservoir(kind, np.random.default_rng(100 + sd))
        Win = build_win(kind)
        inp = np.random.default_rng(7 + sd).integers(0, V, N_SAMP)
        X = si.esn_forward(W, leak, Win, inp, N)
        t, per = mem_cap(X, inp)
        totals.append(t); pers.append(per)
    per_mean = {lag: float(np.mean([p[lag] for p in pers])) for lag in LAGS}
    return float(np.mean(totals)), float(np.std(totals)), per_mean


def make_figure(results):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    styles = {"flat": ("o-", "C0"), "ensemble": ("s-", "C1"),
              "multiverse": ("^-", "C2")}
    for kind, (_, _, per) in results.items():
        fmt, col = styles[kind]
        ax.plot(LAGS, [per[l] for l in LAGS], fmt, color=col, label=kind)
    ax.set_xlabel("delay (lag)")
    ax.set_ylabel("reconstructable memory (acc − chance)")
    ax.set_title("Multiverse test: memory per lag (averaged over 3 seeds)\n"
                 "flat vs ensemble vs multiverse — same nodes/edges")
    ax.legend(); ax.grid(alpha=0.3)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic",
             color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig_multiverse.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    print("Multiverse test (hierarchical coupling vs flat vs ensemble)")
    print(f"  N={N} ({K}×{N_U}), edges≈{K*E_BLOCK}, RHO={RHO}, Leak={LEAK}, "
          f"V={V}, {len(SEEDS)} Seeds")
    print(f"  Lags={LAGS}\n")

    results = {k: evaluate(k) for k in ("flat", "ensemble", "multiverse")}

    print(f"{'='*58}")
    print(f"  {'Variant':<12} {'MC total':>12}  {'long lags (≥16)':>18}")
    print(f"  {'-'*46}")
    long_lags = [l for l in LAGS if l >= 16]
    for k, (tot, sd, per) in results.items():
        long = sum(per[l] for l in long_lags)
        print(f"  {k:<12} {tot:>7.2f}±{sd:<4.2f}  {long:>18.3f}")
    print(f"  {'-'*46}")

    make_figure(results)
    print("  Figure: fig_multiverse.png")

    f = results["flat"][0]; e = results["ensemble"][0]; m = results["multiverse"][0]
    fl = sum(results["flat"][2][l] for l in long_lags)
    el = sum(results["ensemble"][2][l] for l in long_lags)
    ml = sum(results["multiverse"][2][l] for l in long_lags)

    print(f"\n{'='*58}\nVERDICT\n{'='*58}")
    print(f"  Total MC:   flat {f:.2f} | ensemble {e:.2f} | multiverse {m:.2f}")
    print(f"  Long lags:  flat {fl:.2f} | ensemble {el:.2f} | multiverse {ml:.2f}")
    if m > max(f, e) + 0.3:
        print("  -> Macro loops bring measurably MORE total memory.")
    elif ml > max(fl, el) + 0.15:
        print("  -> No total advantage, but better LONG memory (macro time scale).")
    else:
        print("  -> NO advantage from macro loops: hierarchical coupling beats")
        print("     neither the flat reservoir nor the ensemble of the same size.")
    print("  (Order of magnitude, no exponential growth — cf. B-3/B-5.)")
    print(f"\n  Runtime: {time.time()-t0:.0f}s\n{'='*58}")


if __name__ == "__main__":
    main()
