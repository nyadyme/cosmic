#!/usr/bin/env python3
"""
Path B -- Controlled ablation: Does the FRACTAL topology carry memory?
======================================================================
Honest core question (Goal 1): Does the cosmic-fractal topology achieve something
that a trivial baseline does NOT also achieve?

TASK (memory-forcing): Delayed-Copy.
  input[t] = i.i.d. random from V symbols
  target[t] = input[t - LAG]          (LAG=3)
  -> current token says NOTHING about the target. Memoryless = chance (1/V).
  -> only genuine reservoir memory can decode input[t-LAG].

DYNAMICS (standard leaky echo-state, unconditionally stable -- replaces broken Euler):
  x <- (1-a) * x + a * tanh(W_res @ x + W_in @ onehot(u))
  a_i = leak rate from tau gradient: a_i = 1/tau_i  (core slow, leaf fast)
  state x is carried across tokens (echo state) -> memory in the slow core.

READOUT: Ridge on the CURRENT state x(t) (no external delay buffer!),
z-standardized. Memory MUST come from the reservoir dynamics.

FIVE CONTROLS (identical tau distribution, identical readout, only the coupling varies):
  1. memoryless   : features = onehot(u(t)) only          -> chance (task needs memory)
  2. delayline    : features = [u(t),u(t-1),...,u(t-LAG-1)] -> ~100% (trivial memory solves it)
  3. fractal      : W_res = cosmic fractal coupling
  4. random_sparse: W_res = random graph, SAME nodes/edges/spectral radius
  5. leaky_only   : W_res = 0 (pure leaky integrators, no graph)

INTERPRETATION:
  fractal >> random_sparse  -> the HIERARCHICAL structure contributes (Goal 1 supported)
  fractal ~= random_sparse  -> only sparsity/spectral radius matters, not "cosmic"
  fractal ~= leaky_only     -> only fading memory, the graph is irrelevant
  all ~= memoryless         -> no memory -> architecture contributes nothing
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

from cosmic_web_generator import (
    CosmicWebConfig, generate_cosmic_web, mu, eps, iota
)

try:
    from sklearn.linear_model import RidgeClassifier
    HAS_SK = True
except ImportError:
    HAS_SK = False


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

NET = CosmicWebConfig(
    n_levels=3, eta=4, n_top=6,
    tau_leaf=1.0, tau_top=40.0,
    G_min=0.1, G_iota=0.1,
    W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3, seed=42,
)

V         = 6        # vocabulary (i.i.d. uniform)
LAG       = 3        # target[t] = input[t-LAG]
N_SAMPLES = 9000
WARMUP    = 200
RHO       = 0.95     # spectral radius (edge of chaos)
INPUT_GAIN= 1.0
RIDGE_A   = 1.0
SEED      = 7


# ---------------------------------------------------------------
# Reservoir coupling matrices
# ---------------------------------------------------------------

def fractal_coupling(web):
    """Symmetric coupling matrix W_ij from the generator edges."""
    rows, cols, data = [], [], []
    for a, b, d in web.G.edges(data=True):
        rows += [a, b]; cols += [b, a]; data += [d['W'], d['W']]
    W = sp.csr_matrix((data, (rows, cols)), shape=(web.N, web.N))
    return W


def random_sparse_coupling(N, n_edges, weight_pool, rng):
    """Random graph: same node/edge count, weights from the same pool."""
    rows, cols, data = [], [], []
    seen = set()
    pool = np.asarray(weight_pool)
    tries = 0
    while len(seen) < n_edges and tries < n_edges * 20:
        i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
        tries += 1
        if i == j or (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        w = float(pool[rng.integers(0, len(pool))])
        rows += [i, j]; cols += [j, i]; data += [w, w]
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def spectral_scale(W, rho_target):
    """Scale W so that spectral radius = rho_target."""
    try:
        vals = eigs(W.astype(float), k=1, which='LM',
                    return_eigenvectors=False, tol=1e-3, maxiter=500)
        rho0 = max(float(np.abs(vals[0])), 1e-6)
    except Exception:
        rho0 = max(float(np.abs(W.data).max()), 1e-6)
    return W * (rho_target / rho0)


# ---------------------------------------------------------------
# Leaky-ESN run
# ---------------------------------------------------------------

def run_esn(W_res, leak, W_in, inputs, N):
    """
    x <- (1-a)*x + a*tanh(W_res@x + W_in@onehot(u))
    Returns state matrix [T, N].
    """
    T = len(inputs)
    X = np.zeros((T, N))
    x = np.zeros(N)
    use_res = W_res is not None and W_res.nnz > 0
    for t in range(T):
        u_oh = np.zeros(V); u_oh[inputs[t]] = INPUT_GAIN
        drive = W_in.dot(u_oh)
        if use_res:
            drive = drive + W_res.dot(x)
        x = (1.0 - leak) * x + leak * np.tanh(drive)
        X[t] = x
    return X


def make_W_in(N, node_targets, rng):
    """W_in [N, V]: each symbol injects into a dedicated node."""
    W_in = np.zeros((N, V))
    for s in range(V):
        W_in[node_targets[s], s] = 1.0
    return sp.csr_matrix(W_in)


# ---------------------------------------------------------------
# Readout (Ridge on z-standardized state)
# ---------------------------------------------------------------

def ridge_eval(X, targets, warmup=WARMUP):
    Xw = X[warmup:]
    yw = targets[warmup:]
    # z-standardization (important: tiny core signal vs large leaf values)
    mu_ = Xw.mean(0); sd_ = Xw.std(0) + 1e-8
    Xs = (Xw - mu_) / sd_
    n_tr = int(0.7 * len(Xs))
    Xtr, ytr = Xs[:n_tr], yw[:n_tr]
    Xte, yte = Xs[n_tr:], yw[n_tr:]
    if HAS_SK:
        clf = RidgeClassifier(alpha=RIDGE_A)
        clf.fit(_cpu(Xtr), _cpu(ytr))
        return clf.score(_cpu(Xtr), _cpu(ytr)), clf.score(_cpu(Xte), _cpu(yte))
    Yoh = np.eye(V)[ytr]
    Wout = np.linalg.lstsq(Xtr.T@Xtr + RIDGE_A*np.eye(Xtr.shape[1]),
                           Xtr.T@Yoh, rcond=None)[0]
    tr = float(np.mean(np.argmax(Xtr@Wout,1)==ytr))
    te = float(np.mean(np.argmax(Xte@Wout,1)==yte))
    return tr, te


# ---------------------------------------------------------------
# Main program
# ---------------------------------------------------------------

def main():
    t0 = time.time()
    rng = _np.random.default_rng(SEED)

    # data: i.i.d. + delayed-copy target
    inputs  = np.asarray(rng.integers(0, V, N_SAMPLES))
    targets = np.empty(N_SAMPLES, dtype=int)
    targets[LAG:] = inputs[:-LAG]
    targets[:LAG] = 0
    chance = 1.0 / V

    print(f"Task: Delayed-Copy LAG={LAG}, V={V}, {N_SAMPLES} samples")
    print(f"  chance={chance:.1%}  |  i.i.d. input -> current token carries 0 info")

    # topology
    print("\nTopology ...")
    web = generate_cosmic_web(NET)
    N   = web.N
    tau = np.asarray(web.tau.copy())
    leak = np.clip(1.0 / tau, 1e-3, 1.0)   # a_i = 1/tau_i (core slow)
    print(f"  {web.n_clusters} clusters | {N} nodes | "
          f"leak: {leak.min():.3f}..{leak.max():.3f} "
          f"(core slow, leaf fast)")

    # input nodes: V leaf eps nodes
    max_lv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[mu(c)]['level'] == max_lv]
    node_targets = [eps(leaves[s]) for s in range(V)]
    W_in = make_W_in(N, node_targets, rng)

    # fractal coupling
    W_frac = spectral_scale(fractal_coupling(web), RHO)
    n_edges = web.n_filaments
    weight_pool = [d['W'] for _,_,d in web.G.edges(data=True)]

    results = {}

    # 1. memoryless (only current input)
    print("\n[1] memoryless ...")
    Xml = np.eye(V)[inputs]
    results['1 memoryless'] = ridge_eval(Xml, targets)

    # 2. delayline (explicit buffer, reference ceiling)
    print("[2] delayline (reference ceiling) ...")
    buf = LAG + 2
    Xdl = np.zeros((N_SAMPLES, V*buf))
    for t in range(N_SAMPLES):
        for k in range(buf):
            if t-k >= 0:
                Xdl[t, k*V + inputs[t-k]] = 1.0
    results['2 delayline'] = ridge_eval(Xdl, targets)

    # 3. fractal
    print("[3] fractal reservoir ...")
    Xf = run_esn(W_frac, leak, W_in, inputs, N)
    results['3 fractal'] = ridge_eval(Xf, targets)

    # 4. random_sparse (same tau, same edges/spectral radius)
    print("[4] random_sparse (matched) ...")
    W_rand = spectral_scale(
        random_sparse_coupling(N, n_edges, weight_pool, rng), RHO)
    Xr = run_esn(W_rand, leak, W_in, inputs, N)
    results['4 random_sparse'] = ridge_eval(Xr, targets)

    # 5. leaky_only (W_res = 0)
    print("[5] leaky_only (no graph) ...")
    Xl = run_esn(None, leak, W_in, inputs, N)
    results['5 leaky_only'] = ridge_eval(Xl, targets)

    # result
    dt = time.time() - t0
    print(f"\n{'='*58}")
    print(f"ABLATION -- Delayed-Copy LAG={LAG}, V={V}  ({dt:.0f}s)")
    print(f"{'='*58}")
    print(f"  {'Condition':<20}  {'Train':>7}  {'Test':>7}  {'vs chance':>10}")
    print(f"  {'-'*52}")
    print(f"  {'chance':<20}  {'--':>7}  {chance:>7.1%}  {'0.0 PP':>10}")
    for name in sorted(results):
        tr, te = results[name]
        print(f"  {name:<20}  {tr:>7.1%}  {te:>7.1%}  "
              f"{(te-chance)*100:>+8.1f} PP")
    print(f"  {'-'*52}")

    # interpretation
    te = {k: v[1] for k, v in results.items()}
    f  = te['3 fractal']
    r  = te['4 random_sparse']
    l  = te['5 leaky_only']
    m  = te['1 memoryless']
    c  = te['2 delayline']
    print(f"\n  INTERPRETATION:")
    print(f"  - memoryless={m:.1%} (≈chance={chance:.1%}?) -> "
          f"{'task needs memory, good' if m < chance+0.05 else 'WARNING: task leaks info'}")
    print(f"  - delayline={c:.1%} -> ceiling (trivial memory solves the task)")
    print(f"  - fractal={f:.1%} vs random_sparse={r:.1%}: "
          f"diff {(f-r)*100:+.1f} PP")
    if f > r + 3:
        print(f"    -> HIERARCHICAL structure contributes (Goal 1 supported)")
    elif abs(f - r) <= 3:
        print(f"    -> topology irrelevant; only sparsity+spectral radius matter")
    print(f"  - fractal={f:.1%} vs leaky_only={l:.1%}: diff {(f-l)*100:+.1f} PP")
    if f > l + 5:
        print(f"    -> the GRAPH (coupling) contributes, not just fading memory")
    else:
        print(f"    -> pure fading memory suffices; coupling irrelevant")
    print(f"{'='*58}")


if __name__ == '__main__':
    main()
