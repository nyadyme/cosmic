#!/usr/bin/env python3
"""
Multiverse: geometry of the UPPER level (macro topology of the universes)
======================================================================
So far the upper level (coupling between the universes) was a trivial CHAIN.
Question: does the memory capacity change when the upper level gets real geometry?
Sweep over the macro topology, everything else fixed (K=6 modules of 100,
graded leak, input everywhere, Q=4 ports per macro edge):

  chain     -- chain k→k+1            (reference, so far)
  ring      -- closed ring
  lattice   -- 2×3 grid (neighbors right/down)
  random    -- random macro edges (same count as chain)
  cosmic    -- universes placed in 3D, each connected to k-NN by proximity
               (= fractal/cosmic geometry on the upper level)
  alltoall  -- each universe with every other

Fair reference: flat_match (same size/leak/input). Prior from B-3 + Q-sweep:
the advantage comes from the time-scale-aligned modular organization, not
from the macro coupling pattern → expectation: MC ~unchanged. Being tested.
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

import simulation_insitu as si
import simulation_multiverse2 as m2

K, N_U, N = m2.K, m2.N_U, m2.N
E_BLOCK, RHO, V = m2.E_BLOCK, m2.RHO, m2.V
LAGS, N_SAMP = m2.LAGS, m2.N_SAMP
Q = 4
SEEDS = [0, 1, 2]
MACROS = ["chain", "ring", "lattice", "random", "cosmic", "alltoall"]
DISCLAIMER = "AI-assisted -- please reproduce independently"


def macro_pairs(macro):
    rng = np.random.default_rng(0)             # fixed macro topology
    if macro == "chain":
        return [(k, k + 1) for k in range(K - 1)]
    if macro == "ring":
        return [(k, (k + 1) % K) for k in range(K)]
    if macro == "lattice":                     # 2 rows × 3 columns
        p = []
        for r in range(2):
            for c in range(3):
                i = r * 3 + c
                if c < 2: p.append((i, i + 1))
                if r < 1: p.append((i, i + 3))
        return p
    if macro == "random":
        s = set()
        while len(s) < K - 1:
            a, b = int(rng.integers(0, K)), int(rng.integers(0, K))
            if a != b: s.add((a, b))
        return list(s)
    if macro == "cosmic":                      # 3D positions, k-NN by proximity
        pos = rng.normal(size=(K, 3)); p = []
        for i in range(K):
            d = np.linalg.norm(pos - pos[i], axis=1); d[i] = np.inf
            for j in np.argsort(d)[:2]:
                p.append((i, int(j)))
        return list(set(p))
    if macro == "alltoall":
        return [(i, j) for i in range(K) for j in range(K) if i < j]


def build(macro, wrng):
    R, C, D, seen = [], [], [], set()
    for k in range(K):
        m2._add(R, C, D, k * N_U, N_U, E_BLOCK, wrng, seen)
    for a, b in macro_pairs(macro):
        for i in range(Q):
            s = a * N_U + (N_U - 1 - i); d = b * N_U + i
            if (d, s) not in seen:
                seen.add((d, s)); R.append(d); C.append(s)
                D.append(float(wrng.standard_normal()))
    return si.spectral_scale(sp.csr_matrix((D, (R, C)), shape=(N, N)), RHO)


def evaluate(macro):
    leak = m2.build_leak("graded", np.random.default_rng(500))
    win = m2.build_win("all")
    vals = []
    for sd in SEEDS:
        W = build(macro, np.random.default_rng(100 + sd))
        inp = np.random.default_rng(7 + sd).integers(0, V, N_SAMP)
        X = si.esn_forward(W, leak, win, inp, N)
        t, _ = m2.mem_cap(X, inp); vals.append(t)
    return float(np.mean(vals)), float(np.std(vals))


def main():
    t0 = time.time()
    print("Multiverse: geometry of the upper level (macro topology)\n")
    f_tot, _, _ = m2.evaluate("flat", "all", "graded")
    print(f"  Reference flat_match: MC {f_tot:.2f}\n")
    print(f"  {'macro topology':<12} {'macro edges':>13} {'MC':>12}")
    print(f"  {'-'*40}")
    res = {}
    for m in MACROS:
        mc, sd = evaluate(m); res[m] = (mc, sd, len(macro_pairs(m)))
        print(f"  {m:<12} {len(macro_pairs(m)):>13} {mc:>7.2f}±{sd:<4.2f}")
    print(f"  {'-'*40}")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(MACROS, [res[m][0] for m in MACROS],
           yerr=[res[m][1] for m in MACROS], capsize=4, color="C2")
    ax.axhline(f_tot, color="C3", ls="--", lw=1.3, label=f"flat_match ({f_tot:.1f})")
    ax.set_ylabel("memory capacity")
    ax.set_title("Geometry of the upper multiverse level (N=600, 3 seeds)\n"
                 "does the macro topology change the capacity?")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(res[m][0] for m in MACROS) * 1.15)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic", color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig_multiverse6.png", dpi=150)
    plt.close(fig)
    print("  Figure: fig_multiverse6.png")

    mcs = [res[m][0] for m in MACROS]
    spread = max(mcs) - min(mcs); maxsd = max(res[m][1] for m in MACROS)
    best = max(MACROS, key=lambda m: res[m][0])
    print(f"\n{'='*48}\nVERDICT\n{'='*48}")
    print(f"  Range over macro topologies: {spread:.2f} (max. spread ±{maxsd:.2f}).")
    print(f"  Best: {best} ({res[best][0]:.2f}); chain {res['chain'][0]:.2f}; "
          f"cosmic {res['cosmic'][0]:.2f}; alltoall {res['alltoall'][0]:.2f}.")
    if spread <= 2 * maxsd:
        print("  -> Geometry of the upper level does NOT change the capacity (range within")
        print("     the noise). Confirmed: the advantage comes from the module organization,")
        print("     not from the macro coupling pattern (consistent with B-3 + Q-sweep).")
    else:
        print("  -> The macro topology makes a difference — see the numbers.")
    print(f"\n  Runtime: {time.time()-t0:.0f}s\n{'='*48}")


if __name__ == "__main__":
    main()
