#!/usr/bin/env python3
"""
Capacity levers: (1) best reservoir for linear memory, (2) external memory
=========================================================================
Follow-up to the multiverse tests: there, structure was maxed out. Here are the
two honest levers to genuinely raise the total capacity.

PART 1 — Reservoir-internal upper bound (linear Memory Capacity, MC ≤ N):
  random      -- standard ESN coupling
  orthogonal  -- W = orthogonal matrix × ρ
  cycle       -- simple ring (Rodan & Tiňo, SCR) — theoretically reaches MC ≈ N
  Linear reservoir, MC = Σ_k corr²(û(t−k), u(t−k)). Shows how close one gets to
  the maximum N (random does NOT exhaust N).

PART 2 — external digital memory (delay taps) on a NONLINEAR task (NARMA):
  res(N)         -- reservoir N, readout from states
  res(N)+taps(D) -- readout from [states, u(t−1..D)]  (digital buffer)
  res(N+D)       -- larger reservoir of equal readout dimension (fair control)
  Question: do D digital taps help more than D additional reservoir nodes?
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

DISCLAIMER = "AI-assisted / AI-assisted -- please reproduce independently"
SEEDS = [0, 1]


# ----------------------------- PART 1 -----------------------------
def W_random(N, rng, rho=0.9):
    W = rng.standard_normal((N, N)) / np.sqrt(N)
    ev = np.max(np.abs(np.linalg.eigvals(W)))
    return W * (rho / ev)


def W_orth(N, rng, rho=0.9):
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    return rho * Q


def W_cycle(N, r=0.9):
    W = np.zeros((N, N))
    for i in range(N):
        W[i, (i - 1) % N] = r
    return W


def run_linear(W, win, u):
    N = W.shape[0]; X = np.zeros((len(u), N)); x = np.zeros(N)
    for t in range(len(u)):
        x = W @ x + win * u[t]; X[t] = x
    return X


def mc_total(X, u, kmax, washout):
    idx = np.arange(washout, len(u))
    Xw = X[idx]; Xs = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-9)
    n = int(0.7 * len(idx)); tot = 0.0; per = []
    for k in range(1, kmax + 1):
        y = u[idx - k]
        r = Ridge(alpha=1e-6); r.fit(Xs[:n], y[:n]); p = r.predict(Xs[n:])
        c = np.corrcoef(p, y[n:])[0, 1]
        mck = max(float(c) ** 2, 0.0) if np.isfinite(c) else 0.0
        per.append(mck); tot += mck
    return tot, per


def part1(N=100, kmax=200, washout=250, L=3500):
    builders = {"random": lambda rng: W_random(N, rng),
                "orthogonal": lambda rng: W_orth(N, rng),
                "cycle": lambda rng: W_cycle(N)}
    win0 = np.random.default_rng(0).choice([-1.0, 1.0], N) * 0.5
    res = {}
    curves = {}
    for name, bf in builders.items():
        tots, pers = [], []
        for sd in SEEDS:
            u = np.random.default_rng(20 + sd).uniform(-0.8, 0.8, L)
            W = bf(np.random.default_rng(40 + sd))
            X = run_linear(W, win0, u)
            t, per = mc_total(X, u, kmax, washout)
            tots.append(t); pers.append(per)
        res[name] = (float(np.mean(tots)), float(np.std(tots)))
        curves[name] = np.mean(pers, axis=0)
    return N, res, curves


# ----------------------------- PART 2 -----------------------------
def run_tanh(W, win, u, leak=0.3):
    N = W.shape[0]; X = np.zeros((len(u), N)); x = np.zeros(N)
    for t in range(len(u)):
        x = (1 - leak) * x + leak * np.tanh(W @ x + win * u[t]); X[t] = x
    return X


def narma_target(u, order):
    L = len(u); y = np.zeros(L)
    for t in range(order, L - 1):
        y[t + 1] = np.tanh(0.3 * y[t] + 0.05 * y[t] * y[t - order + 1:t + 1].sum()
                           + 1.5 * u[t - order + 1] * u[t] + 0.1)
    return y


def delay_taps(u, D):
    T = len(u); U = np.zeros((T, D))
    for d in range(1, D + 1):
        U[d:, d - 1] = u[:-d]
    return U


def nrmse_feat(F, y, washout=300, alpha=1e-3):
    Fw, yw = F[washout:], y[washout:]
    mu, sd = Fw.mean(0), Fw.std(0) + 1e-9
    Fs = (Fw - mu) / sd
    n = int(0.7 * len(Fs))
    r = Ridge(alpha=alpha); r.fit(Fs[:n], yw[:n]); p = r.predict(Fs[n:])
    return float(np.sqrt(np.mean((p - yw[n:]) ** 2)) / (yw[n:].std() + 1e-12))


def part2(N=100, D=20, orders=(10, 20), L=6000):
    win_rng = np.random.default_rng(0)
    win_N = win_rng.choice([-1.0, 1.0], N) * 0.5
    win_ND = np.random.default_rng(1).choice([-1.0, 1.0], N + D) * 0.5
    out = {o: {"res(N)": [], "res(N)+taps": [], "res(N+D)": []} for o in orders}
    for sd in SEEDS:
        u = np.random.default_rng(7 + sd).uniform(0.0, 0.5, L)
        WN = W_random(N, np.random.default_rng(40 + sd))
        WND = W_random(N + D, np.random.default_rng(60 + sd))
        XN = run_tanh(WN, win_N, u)
        XND = run_tanh(WND, win_ND, u)
        taps = delay_taps(u, D)
        XN_taps = np.hstack([XN, taps])
        for o in orders:
            y = narma_target(u, o)
            out[o]["res(N)"].append(nrmse_feat(XN, y))
            out[o]["res(N)+taps"].append(nrmse_feat(XN_taps, y))
            out[o]["res(N+D)"].append(nrmse_feat(XND, y))
    return {o: {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()}
            for o, d in out.items()}, N, D


def main():
    t0 = time.time()
    print("Capacity levers: best reservoir + external memory\n")

    # ---- PART 1 ----
    N1, r1, curves = part1()
    print(f"{'='*52}\nPART 1 — linear Memory Capacity (N={N1}, MC≤N)\n{'='*52}")
    for name in ("random", "orthogonal", "cycle"):
        m, s = r1[name]
        print(f"  {name:<12} MC = {m:6.1f} ± {s:4.1f}   ({m/N1*100:4.0f}% of N)")
    print(f"  {'-'*40}")

    # ---- PART 2 ----
    r2, N2, D2 = part2()
    print(f"\n{'='*52}\nPART 2 — reservoir + digital delay taps (N={N2}, D={D2})\n{'='*52}")
    print(f"  NRMSE (smaller=better):")
    print(f"  {'Task':>9} {'res(N)':>12} {'res(N)+taps':>14} {'res(N+D)':>12}")
    for o in r2:
        a = r2[o]["res(N)"]; b = r2[o]["res(N)+taps"]; c = r2[o]["res(N+D)"]
        print(f"  NARMA-{o:<3} {a[0]:>11.3f} {b[0]:>14.3f} {c[0]:>12.3f}")

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.bar(["random", "orthogonal", "cycle"],
            [r1[n][0] for n in ("random", "orthogonal", "cycle")],
            color=["C3", "C0", "C2"])
    ax1.axhline(N1, color="k", ls="--", lw=1, label=f"theoret. max = N ({N1})")
    ax1.set_ylabel("linear Memory Capacity"); ax1.legend(fontsize=8)
    ax1.set_title("Part 1: best reservoir for linear memory")
    ax1.grid(alpha=0.3, axis="y")

    orders = list(r2.keys()); x = np.arange(len(orders)); w = 0.26
    for i, (k, col) in enumerate([("res(N)", "C3"), ("res(N)+taps", "C2"),
                                  ("res(N+D)", "C0")]):
        ax2.bar(x + (i - 1) * w, [r2[o][k][0] for o in orders], w, label=k, color=col)
    ax2.set_xticks(x); ax2.set_xticklabels([f"NARMA-{o}" for o in orders])
    ax2.set_ylabel("NRMSE (smaller=better)"); ax2.legend(fontsize=8)
    ax2.set_title(f"Part 2: + digital delay taps (N={N2}, D={D2})")
    ax2.grid(alpha=0.3, axis="y")
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic", color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig_capacity.png", dpi=150)
    plt.close(fig)
    print("\n  Figure: fig_capacity.png")

    # ---- Verdict ----
    print(f"\n{'='*52}\nVERDICT\n{'='*52}")
    print(f"  Part 1: cycle/orthogonal exhaust N much better than random "
          f"({r1['cycle'][0]/N1*100:.0f}% vs {r1['random'][0]/N1*100:.0f}% of N).")
    o = orders[-1]
    print(f"  Part 2 (NARMA-{o}): res+taps NRMSE {r2[o]['res(N)+taps'][0]:.3f} vs "
          f"res(N+D) {r2[o]['res(N+D)'][0]:.3f} vs res(N) {r2[o]['res(N)'][0]:.3f}.")
    if r2[o]["res(N)+taps"][0] < r2[o]["res(N+D)"][0] - 0.01:
        print("  -> Digital taps help MORE than the same number of reservoir nodes:")
        print("     external memory is the more efficient capacity lever (ADR-3).")
    else:
        print("  -> Taps ~ equivalent to more nodes (in this regime).")
    print(f"\n  Runtime: {time.time()-t0:.0f}s\n{'='*52}")


if __name__ == "__main__":
    main()
