#!/usr/bin/env python3
"""
External I/O bandwidth of the (multiverse) reservoir
====================================================
Complements the quasar sweep (simulation_multiverse5.py): quasars carry not only
inter-universe coupling but also EXTERNAL input/output. Two questions:

  PART A — input fan-in: ONE external stream distributed over P input ports
           (port weight ~ 1/√P, so the total drive stays constant).
           How many ports are worthwhile? -> saturation point = ideal port count.

  PART B — multi-channel: C INDEPENDENT external streams simultaneously (P ports each).
           Total capacity (Σ over channels) vs. C. Dambre bound: total ≤ N
           -> more channels share the same budget (bandwidth ceiling).

Reservoir: tuned 2-level multiverse (Q=1, graded leak, N=600).
Metric: linear MC = Σ_k corr²(û(t−k), u(t−k)).
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

import simulation_multiverse5 as m5   # build_mv_q
import simulation_multiverse2 as m2   # build_leak, N

N = m2.N
KMAX, WASHOUT, T = 80, 200, 4000
SEEDS = [0, 1]
DISCLAIMER = "AI-assisted / AI-assisted -- please reproduce independently"


def ports(count, offset):
    return [(offset + int(i * N / count)) % N for i in range(count)]


def make_Win(port_lists, signs):
    C = len(port_lists); W = np.zeros((N, C))
    for c, (nodes, sg) in enumerate(zip(port_lists, signs)):
        scale = 1.0 / np.sqrt(len(nodes))
        for nd, s in zip(nodes, sg):
            W[nd, c] = s * scale
    return W


def states(W, leak, Win, U):
    Wc = W.tocsr(); X = np.zeros((len(U), N)); x = np.zeros(N)
    for t in range(len(U)):
        x = (1 - leak) * x + leak * np.tanh(Wc.dot(x) + Win.dot(U[t])); X[t] = x
    return X


def chan_mc(Xs, u, idx, n):
    tot = 0.0
    for k in range(1, KMAX + 1):
        y = u[idx - k]
        r = Ridge(alpha=1e-3); r.fit(Xs[:n], y[:n]); p = r.predict(Xs[n:])
        c = np.corrcoef(p, y[n:])[0, 1]
        tot += max(float(c) ** 2, 0.0) if np.isfinite(c) else 0.0
    return tot


def run(P_list_per_channel, sign_seed):
    """Build reservoir+input, return total MC and per-channel MC (averaged over seeds)."""
    leak = m2.build_leak("graded", np.random.default_rng(500))
    C = len(P_list_per_channel)
    tots, perc = [], []
    for sd in SEEDS:
        W = m5.build_mv_q(1, np.random.default_rng(100 + sd))
        rng = np.random.default_rng(sign_seed + sd)
        plists = [ports(P, c * 13 + 5) for c, P in enumerate(P_list_per_channel)]
        signs = [rng.choice([-1.0, 1.0], P) for P in P_list_per_channel]
        Win = make_Win(plists, signs)
        U = rng.uniform(-0.8, 0.8, (T, C))
        X = states(W, leak, Win, U)
        idx = np.arange(WASHOUT, T)
        Xs = (X[idx] - X[idx].mean(0)) / (X[idx].std(0) + 1e-9)
        n = int(0.7 * len(idx))
        mcs = [chan_mc(Xs, U[:, c], idx, n) for c in range(C)]
        tots.append(sum(mcs)); perc.append(np.mean(mcs))
    return float(np.mean(tots)), float(np.mean(perc))


def main():
    t0 = time.time()
    print("External I/O bandwidth (multiverse reservoir, N=600)\n")

    # ---- PART A: input fan-in (1 channel, P ports) ----
    Ps = [1, 2, 4, 8, 16, 32, 64]
    print(f"{'='*46}\nPART A — input fan-in (1 external stream)\n{'='*46}")
    print(f"  {'P Ports':>8} {'MC':>10}")
    a = {}
    for P in Ps:
        mc, _ = run([P], sign_seed=300)
        a[P] = mc
        print(f"  {P:>8} {mc:>10.1f}")

    # ---- PART B: multi-channel (C streams, 8 ports each) ----
    Cs = [1, 2, 4, 8]
    PP = 8
    print(f"\n{'='*46}\nPART B — multi-channel bandwidth ({PP} ports each)\n{'='*46}")
    print(f"  {'C Channels':>9} {'MC total':>11} {'MC/channel':>10}")
    b = {}
    for C in Cs:
        tot, per = run([PP] * C, sign_seed=700)
        b[C] = (tot, per)
        print(f"  {C:>9} {tot:>11.1f} {per:>10.2f}")

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax1.plot(Ps, [a[P] for P in Ps], "o-", color="C0")
    ax1.set_xscale("log", base=2); ax1.set_xticks(Ps); ax1.set_xticklabels(Ps)
    ax1.set_xlabel("input ports P (1 stream)"); ax1.set_ylabel("Memory Capacity")
    ax1.set_title("Part A: input fan-in (saturation)"); ax1.grid(alpha=0.3)
    ax2.plot(Cs, [b[C][0] for C in Cs], "s-", color="C2", label="MC total")
    ax2.plot(Cs, [b[C][1] for C in Cs], "^--", color="C3", label="MC per channel")
    ax2.set_xlabel("channels C (8 ports each)"); ax2.set_ylabel("Memory Capacity")
    ax2.set_title("Part B: multi-channel (shared budget)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic", color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig_io.png", dpi=150)
    plt.close(fig)
    print("\n  Figure: fig_io.png")

    # ---- Verdict ----
    print(f"\n{'='*46}\nVERDICT\n{'='*46}")
    sat = next((P for P in Ps if a[P] > 0.9 * a[Ps[-1]]), Ps[-1])
    print(f"  Part A: fan-in saturates ~from P≈{sat} (MC {a[sat]:.1f} vs P=1 {a[1]:.1f}, "
          f"P=64 {a[64]:.1f}). More ports = soon no gain.")
    print(f"  Part B: total MC {b[1][0]:.1f}→{b[Cs[-1]][0]:.1f} over C "
          f"(shared N budget), MC/channel {b[1][1]:.1f}→{b[Cs[-1]][1]:.1f} (drops).")
    print(f"  -> External I/O bandwidth is capped by N: more ports/channels")
    print(f"     share the same fixed capacity. Decorrelation/N raises the ceiling (B-13).")
    print(f"\n  Runtime: {time.time()-t0:.0f}s\n{'='*46}")


if __name__ == "__main__":
    main()
