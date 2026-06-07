#!/usr/bin/env python3
"""
Path B (2) -- Memory-capacity curve: Does the FRACTAL hierarchy
have an advantage for LONG memory over a random graph?
==============================================================
LAG=3 was too easy (fractal=random=100%, ceiling effect). Now a LAG sweep:
for each delay k the decode accuracy of input[t-k].
The LAG at which the curve drops = memory capacity.

Efficient: reservoir states do NOT depend on the target -> ONE ESN run
per reservoir, then many Ridge fits across different LAGs.

Standard metric: Memory Capacity (Jaeger 2001), here as classification
accuracy over the i.i.d. symbol sequence.

Comparison (identical tau distribution, identical readout):
  fractal      : cosmic hierarchy coupling
  random_sparse: random graph, same nodes/edges/spectral radius/tau
  leaky_only   : no graph (floor)
  delayline    : explicit buffer of the last LAG_MAX tokens (ceiling, reference only)
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


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)


import time

from cosmic_web_generator import (
    CosmicWebConfig, generate_cosmic_web, mu, eps
)

try:
    from sklearn.linear_model import RidgeClassifier
    HAS_SK = True
except ImportError:
    HAS_SK = False

# ---------------------------------------------------------------
NET = CosmicWebConfig(
    n_levels=3, eta=4, n_top=6,
    tau_leaf=1.0, tau_top=40.0,
    G_min=0.1, G_iota=0.1,
    W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3, seed=42,
)

V          = 6
LAGS       = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
N_SAMPLES  = 12000
WARMUP     = 300
RHO        = 0.97      # higher = more memory (closer to edge of chaos)
INPUT_GAIN = 0.6       # smaller -> less saturation -> more linear memory
RIDGE_A    = 1.0
SEED       = 7


def fractal_coupling(web):
    rows, cols, data = [], [], []
    for a, b, d in web.G.edges(data=True):
        rows += [a, b]; cols += [b, a]; data += [d['W'], d['W']]
    return sp.csr_matrix((data, (rows, cols)), shape=(web.N, web.N))


def random_sparse_coupling(N, n_edges, pool, rng):
    rows, cols, data = [], [], []
    seen, tries = set(), 0
    pool = np.array(pool)
    while len(seen) < n_edges and tries < n_edges*20:
        i, j = int(rng.integers(0,N)), int(rng.integers(0,N))
        tries += 1
        if i==j or (i,j) in seen or (j,i) in seen: continue
        seen.add((i,j)); w = float(pool[rng.integers(0,len(pool))])
        rows += [i,j]; cols += [j,i]; data += [w,w]
    return sp.csr_matrix((data,(rows,cols)), shape=(N,N))


def spectral_scale(W, rho):
    try:
        vals = eigs(W.astype(float), k=1, which='LM',
                    return_eigenvectors=False, tol=1e-3, maxiter=500)
        rho0 = max(float(np.abs(vals[0])), 1e-6)
    except Exception:
        rho0 = max(float(np.abs(W.data).max()), 1e-6)
    return W * (rho/rho0)


def run_esn(W_res, leak, W_in, inputs, N):
    T = len(inputs)
    X = np.zeros((T,N)); x = np.zeros(N)
    use = W_res is not None and W_res.nnz > 0
    for t in range(T):
        u = np.zeros(V); u[inputs[t]] = INPUT_GAIN
        drive = W_in.dot(u)
        if use: drive = drive + W_res.dot(x)
        x = (1.0-leak)*x + leak*np.tanh(drive)
        X[t] = x
    return X


def ridge_acc(X, targets, warmup=WARMUP):
    Xw, yw = X[warmup:], targets[warmup:]
    mu_, sd_ = Xw.mean(0), Xw.std(0)+1e-8
    Xs = (Xw-mu_)/sd_
    n = int(0.7*len(Xs))
    if HAS_SK:
        clf = RidgeClassifier(alpha=RIDGE_A); clf.fit(_cpu(Xs[:n]), _cpu(yw[:n]))
        return clf.score(_cpu(Xs[n:]), _cpu(yw[n:]))
    Yoh = np.eye(V)[yw[:n]]
    W = np.linalg.lstsq(Xs[:n].T@Xs[:n]+RIDGE_A*np.eye(Xs.shape[1]),
                        Xs[:n].T@Yoh, rcond=None)[0]
    return float(np.mean(np.argmax(Xs[n:]@W,1)==yw[n:]))


def main():
    t0 = time.time()
    rng = _np.random.default_rng(SEED)
    inputs = np.asarray(rng.integers(0, V, N_SAMPLES))
    chance = 1.0/V

    print(f"Memory-Capacity sweep: V={V}, LAGs={LAGS}, {N_SAMPLES} samples")
    print(f"  RHO={RHO}, INPUT_GAIN={INPUT_GAIN}, chance={chance:.1%}")

    web = generate_cosmic_web(NET)
    N = web.N
    leak = np.clip(1.0/np.asarray(web.tau), 1e-3, 1.0)
    max_lv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[mu(c)]['level']==max_lv]
    W_in = np.zeros((N,V))
    for s in range(V): W_in[eps(leaves[s]), s] = 1.0
    W_in = sp.csr_matrix(W_in)

    pool = [d['W'] for _,_,d in web.G.edges(data=True)]
    print(f"  {web.n_clusters} clusters | {N} nodes | {web.n_filaments} edges")

    # ONE ESN run per reservoir (states independent of the target)
    print("\nESN runs (once each) ...")
    W_frac = spectral_scale(fractal_coupling(web), RHO)
    X_frac = run_esn(W_frac, leak, W_in, inputs, N)
    print("  fractal done")
    W_rand = spectral_scale(random_sparse_coupling(N, web.n_filaments, pool, rng), RHO)
    X_rand = run_esn(W_rand, leak, W_in, inputs, N)
    print("  random_sparse done")
    X_leak = run_esn(None, leak, W_in, inputs, N)
    print("  leaky_only done")

    # delayline features (explicit buffer)
    buf = max(LAGS)+2
    X_dl = np.zeros((N_SAMPLES, V*buf))
    for t in range(N_SAMPLES):
        for k in range(buf):
            if t-k >= 0: X_dl[t, k*V+inputs[t-k]] = 1.0

    # sweep across LAGs
    print("\nRidge fits across LAGs ...")
    rows = []
    for lag in LAGS:
        tgt = np.zeros(N_SAMPLES, dtype=int)
        tgt[lag:] = inputs[:-lag]
        a_f = ridge_acc(X_frac, tgt)
        a_r = ridge_acc(X_rand, tgt)
        a_l = ridge_acc(X_leak, tgt)
        a_d = ridge_acc(X_dl,   tgt)
        rows.append((lag, a_f, a_r, a_l, a_d))

    dt = time.time()-t0
    print(f"\n{'='*64}")
    print(f"MEMORY-CAPACITY CURVE  ({dt:.0f}s)")
    print(f"{'='*64}")
    print(f"  {'LAG':>4}  {'fractal':>8}  {'random':>8}  {'leaky':>8}  {'delayln':>8}")
    print(f"  {'-'*46}")
    for lag, af, ar, al, ad in rows:
        mark = ' <-' if abs(af-ar) > 0.05 else ''
        print(f"  {lag:>4}  {af:>8.1%}  {ar:>8.1%}  {al:>8.1%}  {ad:>8.1%}{mark}")
    print(f"  {'-'*46}")
    print(f"  chance = {chance:.1%}")

    # Memory Capacity (sum of accuracy-above-chance, normalized)
    def mc(col):
        return sum(max(r[col]-chance,0) for r in rows) / (1-chance)
    mc_f, mc_r, mc_l = mc(1), mc(2), mc(3)
    print(f"\n  Memory Capacity (sum of acc-above-chance):")
    print(f"    fractal      : {mc_f:.2f}")
    print(f"    random_sparse: {mc_r:.2f}")
    print(f"    leaky_only   : {mc_l:.2f}")

    diff = (mc_f - mc_r) / max(mc_r, 1e-6) * 100
    print(f"\n  INTERPRETATION:")
    print(f"  fractal vs random_sparse: {diff:+.1f}% Memory Capacity")
    if mc_f > mc_r * 1.10:
        print(f"  -> FRACTAL hierarchy has MORE memory (Goal 1 specifically supported)")
    elif mc_f < mc_r * 0.90:
        print(f"  -> random graph has MORE memory (fractal is even disadvantageous)")
    else:
        print(f"  -> tie: cosmic structure gives NO memory advantage")
        print(f"     Honest finding: reservoir mechanics carry it, 'cosmic' is decoration.")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
