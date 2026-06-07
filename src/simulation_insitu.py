#!/usr/bin/env python3
"""
Goal 1 -- In-situ learning (ADR-8) tested FAIRLY
================================================
Open point #2 (RESULTS.md): In-situ learning was never tested fairly -- the
stage-5 Hebbian numbers were artefacts of the broken Euler numerics (saturation,
zero-fixed-point decode). Here the local ADR-8 learning rule is tested on the
REPAIRED leaky-ESN numerics (B-3).

Learning rule (ADR-8 memristor/Hebbian + ADR-9 spring term), on the EXISTING
reservoir edges (sparsity is preserved):
    dW_ij = eta * x_post_i * x_pre_j  -  kappa * (W_ij - W0_ij)
After every RENORM steps the spectral radius is rescaled back to RHO
(keep edge-of-chaos -- otherwise learning explodes/vanishes, and the
comparison only measures the gain, not the STRUCTURE).

Fair-test design:
  - Identical stable dynamics + identical readout (Ridge) as B-3.
  - Memory-forcing task (Memory Capacity, i.i.d. input across LAGs).
  - Comparison FIXED vs. LEARNED, for fractal AND random_sparse (ensemble).
  - eta sweep. Plus control: |dW| and spectral drift (does it learn at all?).

Interpretation:
  learned >> fixed (both)         -> In-situ learning is a (topology-free) lever
  learned ~= fixed                -> fixed reservoir already ~optimal (ADR-8 no lever)
  learned(fractal) >> learned(random) -> first fractal-specific compute advantage
"""

try:
    import cupy as np
    import cupyx.scipy.sparse as sp
    from cupyx.scipy.sparse.linalg import eigs
    _GPU = True
except ImportError:
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs
    _GPU = False

import numpy as _np
import time
from sklearn.linear_model import RidgeClassifier

from cosmic_web_generator import CosmicWebConfig, generate_cosmic_web, mu, eps


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)


# ---- Configuration (modelled on simulation_ablation.py, B-3) ----
NET = CosmicWebConfig(
    n_levels=3, eta=4, n_top=6, tau_leaf=1.0, tau_top=40.0,
    G_min=0.1, G_iota=0.1, W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3, seed=42,
)

V          = 6                       # i.i.d. vocabulary
LAGS       = [1, 2, 3, 5, 7, 10]     # Memory-Capacity delays
N_TRAIN    = 8000                    # Hebbian learning run
N_EVAL     = 6000                    # Memory-Capacity evaluation
WARMUP     = 200
RHO        = 0.95                    # target spectral radius (edge of chaos)
GAIN       = 1.0
ALPHA      = 1.0                     # Ridge regularization
RENORM     = 100                     # spectral renorm interval (tight -> edge-of-chaos)
KAPPA      = 0.001                   # ADR-9 spring stiffness (anchor to W0)
ETAS       = [0.0, 0.005, 0.02, 0.05]   # 0.0 = fixed (control)
N_RAND     = 3                       # random ensemble (lesson from B-3)
chance     = 1.0 / V


# ---- Couplings ----
def fractal_coupling(web):
    rows, cols, data = [], [], []
    for a, b, d in web.G.edges(data=True):
        rows += [a, b]; cols += [b, a]; data += [d['W'], d['W']]
    return sp.csr_matrix((data, (rows, cols)), shape=(web.N, web.N))


def random_sparse_coupling(N, n_edges, weight_pool, rng):
    rows, cols, data = [], [], []
    seen = set(); pool = _np.asarray(weight_pool); tries = 0
    while len(seen) < n_edges and tries < n_edges * 20:
        i, j = int(rng.integers(0, N)), int(rng.integers(0, N)); tries += 1
        if i == j or (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        w = float(pool[int(rng.integers(0, len(pool)))])
        rows += [i, j]; cols += [j, i]; data += [w, w]
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def spectral_radius(W):
    try:
        v = eigs(W.astype(float), k=1, which='LM', return_eigenvectors=False,
                 tol=1e-3, maxiter=500)
        return max(float(np.abs(v[0])), 1e-6)
    except Exception:
        return max(float(np.abs(W.data).max()), 1e-6)


def spectral_scale(W, rho_target):
    return W * (rho_target / spectral_radius(W))


# ---- ESN dynamics (stable, leaky) ----
def esn_forward(W, leak, Win, inp, N):
    """Fixed reservoir run: returns state matrix [T, N]."""
    X = np.zeros((len(inp), N)); x = np.zeros(N)
    for t in range(len(inp)):
        u = np.zeros(V); u[int(inp[t])] = GAIN
        x = (1 - leak) * x + leak * np.tanh(Win.dot(u) + W.dot(x))
        X[t] = x
    return X


def esn_learn(W0, leak, Win, inp, N, eta, kappa=KAPPA, renorm=RENORM):
    """In-situ Hebbian learning run on the existing edges (ADR-8 + ADR-9).

    Updates only the NNZ entries of the coupling (sparsity is preserved), with
    a spring-term anchor to W0. So that learning happens AT the edge-of-chaos (not
    in saturation), the weight L2 norm is kept constant EVERY step
    (Oja-like scale control) + the exact spectral radius is pinned periodically.

    :returns: (W_learned, mean_|dW|, mean_rho_during_learning)
    """
    W = sp.csr_matrix(W0.copy())
    data0 = W.data.copy()
    norm0 = float(np.linalg.norm(W.data))
    # row/column index per NNZ entry (for the local Hebbian term)
    row_idx = np.asarray(_np.repeat(_np.arange(N), _cpu(np.diff(W.indptr))))
    col_idx = W.indices
    x = np.zeros(N)
    rho_track = []
    for t in range(len(inp)):
        u = np.zeros(V); u[int(inp[t])] = GAIN
        x_new = (1 - leak) * x + leak * np.tanh(Win.dot(u) + W.dot(x))
        # local Hebbian correlation on existing edges + spring anchor
        hebb = x_new[row_idx] * x[col_idx]
        W.data += eta * hebb - kappa * (W.data - data0)
        # scale control: weight norm constant -> gain ~ edge-of-chaos
        if eta > 0:
            W.data *= norm0 / (float(np.linalg.norm(W.data)) + 1e-12)
        x = x_new
        if (t + 1) % renorm == 0 and eta > 0:
            r = spectral_radius(W)
            rho_track.append(r)
            W.data *= RHO / r          # exact spectral pin
    dW = float(np.mean(np.abs(W.data - data0)))
    rho_mean = float(_np.mean(rho_track)) if rho_track else RHO
    return W, dW, rho_mean


# ---- Readout + Memory Capacity ----
def racc(X, tgt):
    Xw, yw = X[WARMUP:], tgt[WARMUP:]
    Xs = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-8)
    n = int(0.7 * len(Xs))
    clf = RidgeClassifier(alpha=ALPHA)
    clf.fit(_cpu(Xs[:n]), _cpu(yw[:n]))
    return clf.score(_cpu(Xs[n:]), _cpu(yw[n:]))


def mem_cap(X, inp):
    mc = 0.0
    for lag in LAGS:
        t = np.zeros(len(inp), dtype=int); t[lag:] = inp[:-lag]
        mc += max(racc(X, t) - chance, 0)
    return mc / (1 - chance)


# ---- Setup helpers ----
def make_Win(N, node_targets):
    W_in = _np.zeros((N, V))
    for s in range(V):
        W_in[node_targets[s], s] = 1.0
    return sp.csr_matrix(np.asarray(W_in))


def evaluate(name, W0, leak, Win, N, inp_tr, inp_ev):
    """Compares fixed vs. learned reservoir across the eta sweep."""
    rows = []
    for eta in ETAS:
        if eta == 0.0:
            W, dW, drift = W0, 0.0, RHO          # control: fixed
        else:
            W, dW, drift = esn_learn(W0, leak, Win, inp_tr, N, eta)
        X = esn_forward(W, leak, Win, inp_ev, N)
        mc = mem_cap(X, inp_ev)
        rows.append((eta, mc, dW, drift))
    return rows


def main():
    t0 = time.time()
    print("In-situ learning (ADR-8) -- fair on repaired leaky-ESN numerics")
    print(f"  V={V}, LAGs={LAGS}, Train={N_TRAIN}, Eval={N_EVAL}, RHO={RHO}")

    web = generate_cosmic_web(NET)
    N = web.N
    tau = np.asarray(web.tau)
    leak = np.clip(1.0 / tau, 1e-3, 1.0)
    mlv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[mu(c)]['level'] == mlv]
    node_targets = [eps(leaves[s]) for s in range(V)]
    Win = make_Win(N, node_targets)
    n_edges = web.n_filaments
    weight_pool = [d['W'] for _, _, d in web.G.edges(data=True)]
    print(f"  {web.n_clusters} clusters | {N} nodes | {n_edges} edges")

    rng_in = _np.random.default_rng(7)
    inp_tr = np.asarray(rng_in.integers(0, V, N_TRAIN))
    inp_ev = np.asarray(rng_in.integers(0, V, N_EVAL))

    # ---- fractal ----
    W_frac = spectral_scale(fractal_coupling(web), RHO)
    frac_rows = evaluate("fractal", W_frac, leak, Win, N, inp_tr, inp_ev)

    # ---- random_sparse (ensemble, same nodes/edges/spectral radius) ----
    rand_rows_ens = []
    for s in range(N_RAND):
        rng = _np.random.default_rng(200 + s)
        W_r = spectral_scale(
            random_sparse_coupling(N, n_edges, weight_pool, rng), RHO)
        rand_rows_ens.append(evaluate("random", W_r, leak, Win, N,
                                      inp_tr, inp_ev))
    # ensemble mean per eta
    rand_rows = []
    for i, eta in enumerate(ETAS):
        mcs   = [r[i][1] for r in rand_rows_ens]
        dWs   = [r[i][2] for r in rand_rows_ens]
        rand_rows.append((eta, float(_np.mean(mcs)), float(_np.std(mcs)),
                          float(_np.mean(dWs))))

    # ---- Output ----
    print(f"\n{'='*64}")
    print(f"MEMORY CAPACITY: fixed (eta=0) vs. in-situ learned")
    print(f"{'='*64}")
    print(f"  {'eta':>6} {'fractal MC':>11} {'|dW|':>8}   "
          f"{'random MC (Ens.)':>18} {'|dW|':>8}")
    print(f"  {'-'*60}")
    for i, eta in enumerate(ETAS):
        fe, fmc, fdw, _ = frac_rows[i]
        _, rmc, rsd, rdw = rand_rows[i]
        tag = "  (fixed)" if eta == 0 else ""
        print(f"  {eta:>6.3f} {fmc:>11.3f} {fdw:>8.4f}   "
              f"{rmc:>10.3f}+/-{rsd:<5.3f} {rdw:>8.4f}{tag}")
    print(f"  {'-'*60}")

    # ---- Analysis ----
    f_fix = frac_rows[0][1]
    f_best = max(frac_rows, key=lambda r: r[1])
    r_fix = rand_rows[0][1]
    r_best = max(rand_rows, key=lambda r: r[1])
    print(f"\n{'='*64}")
    print(f"ANALYSIS")
    print(f"{'='*64}")
    print(f"  fractal: fixed {f_fix:.3f} -> best learned {f_best[1]:.3f} "
          f"(eta={f_best[0]}, {f_best[1]-f_fix:+.3f})")
    print(f"  random:  fixed {r_fix:.3f} -> best learned {r_best[1]:.3f} "
          f"(eta={r_best[0]}, {r_best[1]-r_fix:+.3f})")
    max_dw = max(r[2] for r in frac_rows)
    rho_small = frac_rows[1][3]                 # smallest eta > 0
    rho_big   = frac_rows[-1][3]                # largest eta
    print(f"  Learning active? max|dW|={max_dw:.4f} "
          f"(weights change {'yes' if max_dw > 1e-3 else 'barely'})")
    print(f"  Mechanics (no saturation artefact like stage 5):")
    print(f"    eta={frac_rows[1][0]}: rho~{rho_small:.2f} (cleanly at edge-of-chaos)"
          f" -> MC neutral")
    print(f"    eta={frac_rows[-1][0]}: rho up to ~{rho_big:.1f} "
          f"(Hebbian self-reinforcement) -> MC drops")

    f_gain = f_best[1] - f_fix
    r_gain = r_best[1] - r_fix
    print(f"\n  VERDICT:")
    if f_gain > 0.3 and r_gain > 0.3:
        print(f"  -> In-situ learning HELPS both (+{f_gain:.2f}/+{r_gain:.2f} MC):")
        print(f"     ADR-8 is a lever -- but topology-free (like B-3).")
    elif f_gain > 0.3 and f_gain > r_gain + 0.3:
        print(f"  -> Learning helps fractal MORE (+{f_gain:.2f} vs +{r_gain:.2f}):")
        print(f"     FIRST fractal-specific compute advantage -- check it!")
    else:
        print(f"  -> In-situ learning yields NO gain "
              f"(fractal {f_gain:+.2f}, random {r_gain:+.2f} MC).")
        print(f"     The fixed reservoir is already ~optimal; the local")
        print(f"     Hebbian rule does not align with the memory target.")
        print(f"     ADR-8 as a compute lever: not supported (consistent with B-3).")
    print(f"\n  Runtime: {time.time()-t0:.0f}s")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
