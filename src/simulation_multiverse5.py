#!/usr/bin/env python3
"""
Multiverse: how many quasars (inter-universe ports) are optimal?
====================================================================
Question: What happens when each universe uses SEVERAL quasars in parallel for
coupling to its neighbors? Sweep over Q = number of bridge ports per
connection, on the tuned 2-level multiverse (graded leak, input everywhere, N=600).

Hypothesis: The multiverse advantage arises from modular ISOLATION (few
bridges protect the slow universes from over-mixing, cf. B-12). More quasars
= more mixing → the advantage should ERODE with growing Q, toward the
flat reservoir. Fair reference: flat_match (same size/leak/input).

Reservoir/metric building blocks from simulation_multiverse2.py.
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
import simulation_multiverse2 as m2     # _add, build_leak, build_win, racc, mem_cap, constants

K, N_U, N = m2.K, m2.N_U, m2.N
E_BLOCK, RHO = m2.E_BLOCK, m2.RHO
LAGS, N_SAMP, V = m2.LAGS, m2.N_SAMP, m2.V
SEEDS = [0, 1, 2]
QS = [1, 2, 4, 8, 16, 32, 64, 100]
LONG = [l for l in LAGS if l >= 16]
DISCLAIMER = "AI-assisted -- please reproduce independently"


def build_mv_q(q, rng):
    """2-level multiverse with q quasar ports per inter-module bridge."""
    R, C, D, seen = [], [], [], set()
    for k in range(K):
        m2._add(R, C, D, k * N_U, N_U, E_BLOCK, rng, seen)
    for k in range(K - 1):
        for i in range(q):
            s = k * N_U + (N_U - 1 - i)        # quasars = surface ports of the module
            d = (k + 1) * N_U + i
            if (d, s) not in seen:
                seen.add((d, s)); R.append(d); C.append(s)
                D.append(float(rng.standard_normal()))
    return si.spectral_scale(sp.csr_matrix((D, (R, C)), shape=(N, N)), RHO)


def evaluate_q(q):
    leak = m2.build_leak("graded", np.random.default_rng(500))
    win = m2.build_win("all")
    tot, lon = [], []
    for sd in SEEDS:
        W = build_mv_q(q, np.random.default_rng(100 + sd))
        inp = np.random.default_rng(7 + sd).integers(0, V, N_SAMP)
        X = si.esn_forward(W, leak, win, inp, N)
        t, per = m2.mem_cap(X, inp)
        tot.append(t); lon.append(sum(per[l] for l in LONG))
    return float(np.mean(tot)), float(np.std(tot)), float(np.mean(lon))


def main():
    t0 = time.time()
    print("Multiverse: sweep over number of quasar ports Q (2 levels, N=600)\n")

    # fair flat reference (Q-independent)
    f_tot, f_sd, f_per = m2.evaluate("flat", "all", "graded")
    f_long = sum(f_per[l] for l in LONG)
    print(f"  Reference flat_match: MC {f_tot:.2f} | long lags {f_long:.3f}\n")

    print(f"  {'Q':>4} {'bridge edges':>15} {'MC total':>14} {'long lags':>12}")
    print(f"  {'-'*48}")
    res = {}
    for q in QS:
        mc, sd, lon = evaluate_q(q)
        res[q] = (mc, sd, lon)
        print(f"  {q:>4} {(K-1)*q:>15} {mc:>8.2f}±{sd:<5.2f} {lon:>12.3f}")
    print(f"  {'-'*48}")

    # Figure
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(QS, [res[q][0] for q in QS], "^-", color="C2", label="multiverse MC")
    ax.axhline(f_tot, color="C3", ls="--", lw=1.3, label=f"flat_match ({f_tot:.1f})")
    ax.set_xscale("log", base=2); ax.set_xticks(QS); ax.set_xticklabels(QS)
    ax.set_xlabel("quasar ports per bridge  Q  (more = more mixing →)")
    ax.set_ylabel("memory capacity")
    ax.set_title("How many quasars are optimal? (2 levels, N=600, 2 seeds)\n"
                 "isolation (small Q) vs. mixing (large Q)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic", color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig_multiverse5.png", dpi=150)
    plt.close(fig)
    print("  Figure: fig_multiverse5.png")

    best_q = max(QS, key=lambda q: res[q][0])
    print(f"\n{'='*52}\nVERDICT\n{'='*52}")
    print(f"  Best Q = {best_q} (MC {res[best_q][0]:.2f}).")
    print(f"  Q=1: {res[1][0]:.2f} | Q={best_q}: {res[best_q][0]:.2f} | "
          f"Q=64: {res[64][0]:.2f} | flat: {f_tot:.2f}")
    if res[64][0] < res[best_q][0] - 0.3:
        print("  -> More quasars ERODE the advantage: too much mixing destroys")
        print("     the modular isolation, MC drops toward flat. Optimum at FEW.")
    else:
        print("  -> MC stays ~stable over Q (no clear isolation effect).")
    print(f"\n  Runtime: {time.time()-t0:.0f}s\n{'='*52}")


if __name__ == "__main__":
    main()
