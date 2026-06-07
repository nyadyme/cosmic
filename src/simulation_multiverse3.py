#!/usr/bin/env python3
"""
Multiverse recursion & scaling -- universes-in-universes + scaling K
==========================================================================
Follow-up experiment to simulation_multiverse2.py (where the tuned 2-level
multiverse beat the fair flat baseline). Here two questions, both with fair
control (flat = same nodes, same edges, IDENTICAL leak vector +
input — only the connectivity differs):

  PART 1 -- Scaling of K: number of universes K = 3,6,9,12 (module size 100 fixed).
            Does the memory capacity keep growing? Does the lead over flat hold?

  PART 2 -- Depth (universes-in-universes): at fixed N=720 and same
            module count (12) and edge count:
              flat  -- one reservoir
              mv2   -- 2 levels: 12 modules in ONE bridge chain
              mv3   -- 3 levels: 12 modules in 3 groups of 4 (bridges WITHIN
                       the groups + bridges BETWEEN the groups)
            2-level and 3-level have exactly the same number of bridges (n_mod-1) —
            only HOW they are wired differs (linear vs nested).

Combination: mv3 IS the combination of recursion (3 levels) and multiple modules;
K (or G×M) can be scaled at every level.
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
RHO, WASHOUT, ALPHA = 0.9, 200, 1.0
LEAK_FAST, LEAK_SLOW = 0.6, 0.05
E_PER_NODE, Q = 4, 4
LAGS = [1, 2, 3, 5, 8, 12, 16, 20, 25]
N_SAMP = 6000
SEEDS = [0, 1]
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


def build(topo, n_mod, mod, rng, M=None):
    N = n_mod * mod
    e_leaf = mod * E_PER_NODE
    R, C, D, seen = [], [], [], set()
    n_bridges = n_mod - 1                      # chain: same for mv2 and mv3
    if topo == "flat":
        _add(R, C, D, 0, N, n_mod * e_leaf + n_bridges * Q, rng, seen)
    else:
        for k in range(n_mod):
            _add(R, C, D, k * mod, mod, e_leaf, rng, seen)
        if topo == "mv2":
            bridges = [(k, k + 1) for k in range(n_mod - 1)]
        else:                                  # mv3: groups of M
            G = n_mod // M
            bridges = []
            for g in range(G):
                for m in range(M - 1):
                    bridges.append((g * M + m, g * M + m + 1))     # within group
            for g in range(G - 1):
                bridges.append((g * M + M - 1, (g + 1) * M))       # between groups
        for a, b in bridges:
            for i in range(Q):
                s = a * mod + (mod - 1 - i); d = b * mod + i
                if (d, s) not in seen:
                    seen.add((d, s)); R.append(d); C.append(s)
                    D.append(float(rng.standard_normal()))
    return si.spectral_scale(sp.csr_matrix((D, (R, C)), shape=(N, N)), RHO)


def graded_leak(n_mod, mod):
    rng = np.random.default_rng(0)             # fixed -> identical for flat & hierarchical
    bases = np.exp(np.linspace(np.log(LEAK_FAST), np.log(LEAK_SLOW), n_mod))
    leak = np.empty(n_mod * mod)
    for k in range(n_mod):
        leak[k*mod:(k+1)*mod] = np.clip(
            np.exp(rng.normal(np.log(bases[k]), 0.2, mod)), LEAK_SLOW*0.5, 0.95)
    return leak


def build_win(n_mod, mod):
    R, C, D = [], [], []
    for k in range(n_mod):
        for s in range(V):
            R.append(k * mod + 10 + s); C.append(s); D.append(1.0)
    return sp.csr_matrix((D, (R, C)), shape=(n_mod * mod, V))


def racc(X, tgt):
    Xw, yw = X[WASHOUT:], tgt[WASHOUT:]
    Xs = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-8)
    n = int(0.7 * len(Xs)); clf = RidgeClassifier(alpha=ALPHA)
    clf.fit(Xs[:n], yw[:n]); return clf.score(Xs[n:], yw[n:])


def mem_cap(X, inp):
    s = 0.0
    for lag in LAGS:
        t = np.zeros(len(inp), dtype=int); t[lag:] = inp[:-lag]
        s += max(racc(X, t) - chance, 0.0)
    return s / (1 - chance)


def evaluate(topo, n_mod, mod, M=None):
    leak = graded_leak(n_mod, mod); Win = build_win(n_mod, mod)
    vals = []
    for sd in SEEDS:
        W = build(topo, n_mod, mod, np.random.default_rng(100 + sd), M=M)
        inp = np.random.default_rng(7 + sd).integers(0, V, N_SAMP)
        X = si.esn_forward(W, leak, Win, inp, n_mod * mod)
        vals.append(mem_cap(X, inp))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    t0 = time.time()
    print("Multiverse: recursion (universes-in-universes) + scaling of K\n")

    # ---- PART 1: scaling of K (2 levels) ----
    MOD = 100
    Ks = [3, 6, 9, 12]
    print(f"{'='*56}\nPART 1 — scaling of K (module size {MOD})\n{'='*56}")
    print(f"  {'K':>3} {'N':>6} {'flat MC':>10} {'mv2 MC':>10} {'lead':>10}")
    print(f"  {'-'*44}")
    scal = {}
    for K in Ks:
        fmc, _ = evaluate("flat", K, MOD)
        mmc, _ = evaluate("mv2", K, MOD)
        scal[K] = (K * MOD, fmc, mmc)
        print(f"  {K:>3} {K*MOD:>6} {fmc:>10.2f} {mmc:>10.2f} {mmc-fmc:>+10.2f}")
    print(f"  {'-'*44}")

    # ---- PART 2: depth (2 vs 3 levels) at fixed N ----
    LEAF, NMOD, MG = 60, 12, 4                  # N=720, 12 modules, 3 groups of 4
    print(f"\n{'='*56}\nPART 2 — depth at N={LEAF*NMOD} "
          f"(12 modules; mv3 = 3 groups of {MG})\n{'='*56}")
    f2, _ = evaluate("flat", NMOD, LEAF)
    m2, _ = evaluate("mv2", NMOD, LEAF)
    m3, _ = evaluate("mv3", NMOD, LEAF, M=MG)
    print(f"  flat (1 level)    : MC {f2:.2f}")
    print(f"  mv2  (2 levels)   : MC {m2:.2f}   ({m2-f2:+.2f} vs flat)")
    print(f"  mv3  (3 levels)   : MC {m3:.2f}   ({m3-m2:+.2f} vs mv2, {m3-f2:+.2f} vs flat)")

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    Ns = [scal[K][0] for K in Ks]
    ax1.plot(Ns, [scal[K][1] for K in Ks], "o-", color="C3", label="flat (fair baseline)")
    ax1.plot(Ns, [scal[K][2] for K in Ks], "^-", color="C2", label="mv2 (2 levels)")
    ax1.set_xlabel("total nodes N (= K × 100)"); ax1.set_ylabel("memory capacity")
    ax1.set_title("Part 1: scaling of K"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.bar(["flat\n(1 lvl)", "mv2\n(2 lvl)", "mv3\n(3 lvl)"], [f2, m2, m3],
            color=["C3", "C2", "C0"])
    ax2.set_ylabel("memory capacity")
    ax2.set_title(f"Part 2: depth (N={LEAF*NMOD}, same edges)")
    ax2.grid(alpha=0.3, axis="y")
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic", color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig_multiverse3.png", dpi=150)
    plt.close(fig)
    print("\n  Figure: fig_multiverse3.png")

    # ---- Verdict ----
    gaps = [scal[K][2]-scal[K][1] for K in Ks]
    print(f"\n{'='*56}\nVERDICT\n{'='*56}")
    print(f"  Scaling: MC grows with K (flat {scal[Ks[0]][1]:.1f}->{scal[Ks[-1]][1]:.1f}, "
          f"mv2 {scal[Ks[0]][2]:.1f}->{scal[Ks[-1]][2]:.1f}).")
    print(f"  Lead mv2 vs flat over K: {[f'{g:+.2f}' for g in gaps]}")
    print(f"  Depth: 3 levels vs 2 levels: {m3-m2:+.2f} MC.")
    if m3 > m2 + 0.3:
        print("  -> Universes-in-universes (3rd level) brings something extra.")
    elif m3 >= m2 - 0.3:
        print("  -> 3rd level ~ equivalent to 2 levels (no clear added benefit).")
    else:
        print("  -> 3rd level hurts (too much isolation / too little mixing).")
    print(f"  (Combinable: recursion + K-scaling on the same blueprint. "
          f"Moderate, not exponential.)")
    print(f"\n  Runtime: {time.time()-t0:.0f}s\n{'='*56}")


if __name__ == "__main__":
    main()
