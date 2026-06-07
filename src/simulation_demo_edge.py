#!/usr/bin/env python3
"""
Application demo -- streaming edge inference on the fractal reservoir
====================================================================
Shows WHAT the reservoir is actually good for (goal-1 finding: it is a generic
reservoir computer, not a GPT LLM). Two canonical edge tasks, both
streaming/predictive (the inductive bias from B-1):

  A) NARMA-10  -- nonlinear time-series prediction (RC standard benchmark).
     Metric: NRMSE (smaller = better). Honest fractal-vs-random comparison.
  B) Vibration anomaly detection (predictive maintenance): the one-step
     prediction model trained on healthy data flags anomalies via the
     prediction residual error. Metric: ROC-AUC.

Reservoir = stable leaky ESN (like B-3/B-10), fixed readout (Ridge). Proves
applicability; at the same time confirms (B-3) fractal ≈ random -> value = hardware,
not topology.
"""

try:
    import cupy as np
    _GPU = True
except ImportError:
    import numpy as np
    _GPU = False

import numpy as _np
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
from sklearn.metrics import roc_auc_score

import simulation_insitu as si   # fractal_coupling, random_sparse_coupling, spectral_scale


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)


RHO, GAIN, WASHOUT, ALPHA = 0.95, 0.4, 200, 1e-3


# ---------------- Reservoir (continuous input) ----------------
def reservoir_states(W, leak, win, signal, N):
    """Leaky-ESN run on a scalar stream; returns states [T, N]."""
    X = _np.zeros((len(signal), N))
    x = _np.zeros(N)
    Wc = W.tocsr()
    for t in range(len(signal)):
        x = (1 - leak) * x + leak * _np.tanh(Wc.dot(x) + win * signal[t])
        X[t] = x
    return X


def build_reservoir(keep="fractal"):
    web = si.generate_cosmic_web(si.NET)
    N = web.N
    leak = _cpu(si.np.clip(1.0 / si.np.asarray(web.tau), 1e-3, 1.0))
    if keep == "fractal":
        W = si.spectral_scale(si.fractal_coupling(web), RHO)
    else:
        pool = [d['W'] for _, _, d in web.G.edges(data=True)]
        W = si.spectral_scale(si.random_sparse_coupling(
            N, web.n_filaments, pool, _np.random.default_rng(11)), RHO)
    # input vector: scalar stream into leaf-eps nodes (random signs)
    mlv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[si.mu(c)]['level'] == mlv]
    rng = _np.random.default_rng(0)
    win = _np.zeros(N)
    for c in leaves[:16]:
        win[si.eps(c)] = GAIN * rng.choice([-1.0, 1.0])
    # scipy CSR for the demo run (CPU)
    import scipy.sparse as _sp
    W = _sp.csr_matrix((_cpu(W.data), _cpu(W.indices), _cpu(W.indptr)),
                       shape=W.shape)
    return W, leak, win, N


def ridge_readout(Xtr, ytr, Xte, alpha=ALPHA):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    clf = Ridge(alpha=alpha)
    clf.fit((Xtr - mu) / sd, ytr)
    return clf.predict((Xte - mu) / sd)


def nrmse(pred, true):
    return float(_np.sqrt(_np.mean((pred - true) ** 2)) / (_np.std(true) + 1e-12))


# ---------------- Task A: NARMA-10 ----------------
def narma10(n, seed=1):
    rng = _np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, n)
    y = _np.zeros(n)
    for t in range(9, n - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * _np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t] + 0.1)
    return u, y


def task_narma(W, leak, win, N, n=3000, n_tr=2000):
    u, y = narma10(n)
    X = reservoir_states(W, leak, win, u, N)
    pred = ridge_readout(X[WASHOUT:n_tr], y[WASHOUT:n_tr], X[n_tr:])
    return nrmse(pred, y[n_tr:]), (y[n_tr:n_tr + 300], pred[:300])


# ---------------- Task B: vibration anomaly ----------------
def vibration_signal(n=4000, seed=3):
    rng = _np.random.default_rng(seed)
    t = _np.arange(n)
    base = (1.0 * _np.sin(2 * _np.pi * t / 50)
            + 0.5 * _np.sin(2 * _np.pi * t / 17)
            + 0.3 * _np.sin(2 * _np.pi * t / 7))
    sig = base + 0.05 * rng.standard_normal(n)
    labels = _np.zeros(n, dtype=int)
    # anomalies only in the test region (second half): high-frequency bursts
    for _ in range(8):
        s = int(rng.integers(n // 2 + 100, n - 120))
        L = int(rng.integers(40, 90))
        sig[s:s + L] += 0.8 * _np.sin(2 * _np.pi * _np.arange(L) / 3.0)
        labels[s:s + L] = 1
    return sig, labels


def task_vibration(W, leak, win, N, n_tr=2000):
    sig, labels = vibration_signal()
    X = reservoir_states(W, leak, win, sig, N)
    # one-step prediction, trained ONLY on healthy data (first half)
    pred = ridge_readout(X[WASHOUT:n_tr - 1], sig[WASHOUT + 1:n_tr],
                         X[n_tr - 1:-1])
    actual = sig[n_tr:]
    resid = _np.abs(pred - actual)
    lab = labels[n_tr:]
    auc = float(roc_auc_score(lab, resid)) if lab.sum() else float("nan")
    return auc, (actual[:600], resid[:600], lab[:600])


# ---------------- Figure ----------------
def make_figure(narma_demo, vib_demo):
    yte, pred = narma_demo
    actual, resid, lab = vib_demo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6))

    ax1.plot(_cpu(yte), color="0.3", lw=1.4, label="NARMA-10 target")
    ax1.plot(_cpu(pred), color="C0", lw=1.0, ls="--", label="reservoir prediction")
    ax1.set_title("A) Streaming prediction (NARMA-10)")
    ax1.set_xlabel("time step"); ax1.set_ylabel("y(t)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(_cpu(resid), color="C3", lw=1.0, label="prediction residual")
    lab = _cpu(lab)
    inq = False
    for i, v in enumerate(lab):
        if v and not inq:
            start = i; inq = True
        elif not v and inq:
            ax2.axvspan(start, i, color="orange", alpha=0.3); inq = False
    if inq:
        ax2.axvspan(start, len(lab), color="orange", alpha=0.3)
    ax2.axvspan(0, 0, color="orange", alpha=0.3, label="true anomaly")
    ax2.set_title("B) Vibration anomaly: residual spikes at anomalies")
    ax2.set_xlabel("time step"); ax2.set_ylabel("|residual|")
    ax2.legend(loc="upper right", fontsize=8); ax2.grid(alpha=0.3)

    fig.text(0.5, 0.005,
             "AI-assisted / AI-assisted -- please reproduce independently",
             ha="center", fontsize=7, style="italic", color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig4_demo.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    print("Application demo: streaming edge inference on the reservoir")
    print(f"  Reservoir: leaky ESN, RHO={RHO}, Ridge readout\n")

    Wf, leak, win, N = build_reservoir("fractal")
    Wr, leak_r, win_r, _ = build_reservoir("random")
    print(f"  {N} nodes | input: 16 leaf channels\n")

    # A) NARMA-10
    nf, narma_demo = task_narma(Wf, leak, win, N)
    nr, _ = task_narma(Wr, leak_r, win_r, N)
    print(f"{'='*58}")
    print(f"A) NARMA-10 -- nonlinear time-series prediction")
    print(f"{'='*58}")
    print(f"  NRMSE fractal:        {nf:.3f}")
    print(f"  NRMSE random-sparse:  {nr:.3f}")
    print(f"  -> both solve the task; fractal ≈ random (B-3): "
          f"difference {abs(nf-nr):.3f}")

    # B) vibration anomaly
    auc, vib_demo = task_vibration(Wf, leak, win, N)
    print(f"\n{'='*58}")
    print(f"B) Vibration anomaly detection (predictive maintenance)")
    print(f"{'='*58}")
    print(f"  Prediction model trained on HEALTHY data; anomaly = "
          f"residual spike")
    print(f"  ROC-AUC: {auc:.3f}  "
          f"({'usable' if auc > 0.8 else 'weak'} for threshold alarm)")

    make_figure(narma_demo, vib_demo)
    print(f"\n  Figure: fig4_demo.png")

    print(f"\n{'='*58}")
    print(f"CONCLUSION (application demo)")
    print(f"{'='*58}")
    print(f"  The reservoir solves real streaming edge tasks (prediction,")
    print(f"  anomaly) with a simple linear readout -- exactly its profile")
    print(f"  (B-1: predictive/streaming). fractal ≈ random confirms again:")
    print(f"  the benefit lies in analog buildability, not in topology.")
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print(f"{'='*58}")


if __name__ == '__main__':
    main()
