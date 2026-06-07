#!/usr/bin/env python3
"""
Stage 4 v4 -- Parallel reservoir test (24 cores)
====================================================
Parallelization: corpus split into N_WORKERS chunks, each chunk
runs in its own process. Valid for reservoir readout because
ridge regression only needs (state, label) pairs, no global
sequence ordering.

Extra: vectorized I_pred via sparse matrix (no Python cluster loop).
Early stopping: aborts settling when max|dV| < tol.
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
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    from multiprocessing import Pool

try:
    from sklearn.linear_model import RidgeClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

NET = CosmicWebConfig(
    n_levels=3, eta=3, n_top=4,
    tau_leaf=1.0, tau_top=30.0,
    G_min=0.1, G_iota=0.1,
    W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3,
    seed=42,
)

DT          = 0.01
MAX_STEPS   = 500
EARLY_TOL   = 1e-4
I_AMP       = 1.0     # saturation as nonlinearity (binary reservoir → better encoding)
VOCAB       = 3
N_CORPUS    = 8000
WARMUP      = 40      # token warmup per chunk (RC needs ~tau_top steps)
SWITCH_MU   = 60
RIDGE_ALPHA = 1.0
N_WORKERS   = 24      # CPU cores


# ──────────────────────────────────────────────────────────────
# Preparation: extract NumPy arrays from the web
# (NetworkX not picklable -> pull everything out before Parallel)
# ──────────────────────────────────────────────────────────────

class ReservoirArrays:
    """Picklable summary of all required arrays."""
    __slots__ = ('Y_data', 'Y_indices', 'Y_indptr', 'Y_shape',
                 'C_inv', 'emb', 'emb_mu',
                 'W_td_data', 'W_td_indices', 'W_td_indptr', 'W_td_shape',
                 'G_mu', 'G_eps',
                 'N', 'n_clusters', 'mu_idx', 'eps_idx')

    def Y(self):
        import scipy.sparse as _sc_sp
        return _sc_sp.csr_matrix(
            (self.Y_data, self.Y_indices, self.Y_indptr), self.Y_shape)

    def W_td(self):
        import scipy.sparse as _sc_sp
        return _sc_sp.csr_matrix(
            (self.W_td_data, self.W_td_indices, self.W_td_indptr), self.W_td_shape)


def build_arrays(web, rho_target=0.9) -> ReservoirArrays:
    """Extracts all necessary arrays from the web object."""
    ra = ReservoirArrays()
    ra.N          = web.N
    ra.n_clusters = web.n_clusters

    # Capacitance inverse (as CPU NumPy for pickling in worker processes)
    g_prec   = _np.array([web.G.nodes[i]['G_prec'] for i in range(web.N)])
    ra.C_inv = 1.0 / _np.maximum(web.tau * g_prec, 1e-10)

    # Admittance matrix as CPU NumPy arrays (pickling)
    Y        = web.Y.tocsr()
    ra.Y_data, ra.Y_indices, ra.Y_indptr = (
        _np.asarray(Y.data), _np.asarray(Y.indices), _np.asarray(Y.indptr))
    ra.Y_shape = Y.shape

    # Node indices as CPU NumPy arrays
    ra.mu_idx  = _np.array([mu(c) for c in range(web.n_clusters)])
    ra.eps_idx = _np.array([eps(c) for c in range(web.n_clusters)])
    ra.G_mu    = g_prec[ra.mu_idx]
    ra.G_eps   = g_prec[ra.eps_idx]

    # Token embedding
    max_lv    = int(web.levels.max())
    leaves    = [c for c in range(web.n_clusters)
                 if web.G.nodes[mu(c)]['level'] == max_lv]
    assert len(leaves) >= VOCAB
    ra.emb    = _np.array([eps(leaves[i]) for i in range(VOCAB)])
    ra.emb_mu = ra.emb - 1   # mu(c) = eps(c) - 1

    # Top-down weight matrix W_td: eps(child) x mu(parent)
    # W_td @ V[mu_all] gives the prediction sum for each eps node
    n_c = web.n_clusters
    rows_td, cols_td, data_td = [], [], []
    vert = {(i, j): d['W']
            for i, j, d in web.G.edges(data=True)
            if d.get('etype') == 'vertical'}
    if vert:
        rows_v, cols_v, data_v = [], [], []
        for (i, j), w in vert.items():
            rows_v += [i, j]; cols_v += [j, i]; data_v += [w, w]
        W_sp = sp.csr_matrix((data_v, (rows_v, cols_v)), shape=(web.N, web.N))
        try:
            vals = eigs(W_sp.astype(float), k=1, which='LM',
                        return_eigenvectors=False, tol=1e-3, maxiter=300)
            rho0 = max(float(np.abs(vals[0])), 1e-6)
        except Exception:
            rho0 = 1.0
        scale = rho_target / rho0

        for (p, ec), w in vert.items():   # p=mu(parent), ec=eps(child)
            if web.kinds[p] == 0 and web.kinds[ec] == 1:
                c_child = web.G.nodes[ec]['cluster']
                row = c_child               # cluster index of the child
                col_mu = p                  # global mu node index
                rows_td.append(row)
                cols_td.append(col_mu)
                data_td.append(w * scale)

    if rows_td:
        W_td = sp.csr_matrix((data_td, (rows_td, cols_td)),
                             shape=(n_c, web.N))
    else:
        W_td = sp.csr_matrix((n_c, web.N))

    ra.W_td_data    = _np.asarray(W_td.data)
    ra.W_td_indices = _np.asarray(W_td.indices)
    ra.W_td_indptr  = _np.asarray(W_td.indptr)
    ra.W_td_shape   = W_td.shape

    return ra


# ──────────────────────────────────────────────────────────────
# Vectorized settling (no Python cluster loop!)
# ──────────────────────────────────────────────────────────────

def settle_vec(V0, I_inject, ra: ReservoirArrays,
               max_steps=MAX_STEPS, tol=EARLY_TOL, dt=DT):
    """
    Euler settling, fully vectorized.
    I_pred via sparse W_td @ V (no Python cluster loop).
    Early stopping when max|dV| < tol.
    """
    Y   = ra.Y()
    W_td= ra.W_td()
    V   = V0.copy()

    for step in range(max_steps):
        # top-down prediction: f[c] = tanh(sum_p W_td[c,p] * V[p])
        f_c   = np.tanh(W_td.dot(V))           # [n_clusters]

        # assemble I_pred
        I = I_inject.copy()
        np.add.at(I, ra.mu_idx,  ra.G_mu  * f_c)
        np.add.at(I, ra.eps_idx, ra.G_eps * f_c)

        dV  = dt * ra.C_inv * (-Y.dot(V) + I)
        V   = np.clip(V + dV, -2.0, 2.0)

        if step > 5 and np.max(np.abs(dV)) < tol:
            return V, step + 1

    return V, max_steps


# ──────────────────────────────────────────────────────────────
# Corpus
# ──────────────────────────────────────────────────────────────

def make_corpus(n, seed=1, switch_mu=SWITCH_MU):
    pats = [[0, 1, 2], [0, 0, 1, 1]]
    rng  = _np.random.default_rng(seed)
    seq, pat, pos, until = [], 0, 0, switch_mu
    while len(seq) < n:
        seq.append(pats[pat][pos % len(pats[pat])])
        pos += 1; until -= 1
        if until <= 0:
            pat = 1 - pat; pos = 0
            until = max(1, int(rng.poisson(switch_mu)))
    return _np.array(seq[:n], dtype=_np.int8)


# ──────────────────────────────────────────────────────────────
# Worker function (runs in its own process)
# ──────────────────────────────────────────────────────────────

def process_chunk(args):
    """
    Processes a corpus chunk sequentially (runs on CPU).
    Returns (states, labels) after discarding warmup tokens.
    """
    import numpy as _np_w
    chunk, ra, warmup = args
    Y    = ra.Y()
    W_td = ra.W_td()
    V      = _np_w.zeros(ra.N)
    states = []
    labels = []

    for t, tok in enumerate(chunk):
        I_inj = _np_w.zeros(ra.N)
        I_inj[ra.emb[int(tok)]] = I_AMP

        # Inline settle (scipy/numpy, no CuPy in worker)
        for step in range(MAX_STEPS):
            f_c = _np_w.tanh(W_td.dot(V))
            I = I_inj.copy()
            _np_w.add.at(I, ra.mu_idx,  ra.G_mu  * f_c)
            _np_w.add.at(I, ra.eps_idx, ra.G_eps * f_c)
            dV = DT * ra.C_inv * (-Y.dot(V) + I)
            V  = _np_w.clip(V + dV, -2.0, 2.0)
            if step > 5 and _np_w.max(_np_w.abs(dV)) < EARLY_TOL:
                break

        if t >= warmup:
            states.append(V[ra.mu_idx].copy())
            labels.append(int(tok))

    return _np_w.array(states), _np_w.array(labels)


# ──────────────────────────────────────────────────────────────
# Ridge readout (fallback without sklearn)
# ──────────────────────────────────────────────────────────────

def ridge_acc(X_tr, y_tr, X_te, y_te):
    if HAS_SKLEARN:
        clf = RidgeClassifier(alpha=RIDGE_ALPHA)
        X_tr_c, y_tr_c = _cpu(X_tr), _cpu(y_tr)
        X_te_c, y_te_c = _cpu(X_te), _cpu(y_te)
        clf.fit(X_tr_c, y_tr_c)
        return clf.score(X_tr_c, y_tr_c), clf.score(X_te_c, y_te_c)
    # Manual ridge solution (runs on GPU if available)
    Y_oh = np.eye(VOCAB)[y_tr]
    W    = np.linalg.solve(X_tr.T @ X_tr + RIDGE_ALPHA * np.eye(X_tr.shape[1]),
                           X_tr.T @ Y_oh)
    tr   = np.mean(np.argmax(X_tr @ W, 1) == y_tr)
    te   = np.mean(np.argmax(X_te @ W, 1) == y_te)
    return tr, te


# ──────────────────────────────────────────────────────────────
# Main program
# ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Generating topology ...")
    web = generate_cosmic_web(NET)
    print(f"  {web.n_clusters} clusters | {web.N} nodes | "
          f"{web.n_filaments} filaments | d_H={web.d_H_angular:.2f}")

    print("Extracting arrays (picklable) ...")
    ra = build_arrays(web)
    print(f"  Embedding: {ra.emb}  |  C_inv range: "
          f"{ra.C_inv.min():.2f}..{ra.C_inv.max():.2f}")

    corpus = make_corpus(N_CORPUS)
    sym, cnt = np.unique(corpus, return_counts=True)
    print(f"  Corpus {N_CORPUS} tokens: {dict(zip(sym.tolist(), cnt.tolist()))}")

    # Split into chunks
    chunk_size = max(WARMUP + 10, N_CORPUS // N_WORKERS)
    chunks     = []
    for i in range(N_WORKERS):
        start = i * (N_CORPUS // N_WORKERS)
        end   = min(start + chunk_size, N_CORPUS - 1)
        if start < N_CORPUS - 1:
            chunks.append(corpus[start:end])

    print(f"\nParallel settling: {len(chunks)} chunks "
          f"of ~{chunk_size} tokens, warmup={WARMUP}, "
          f"workers={N_WORKERS} ...")

    args = [(c, ra, WARMUP) for c in chunks]

    t_settle = time.time()
    if HAS_JOBLIB:
        results = Parallel(n_jobs=N_WORKERS, backend='loky', verbose=5)(
            delayed(process_chunk)(a) for a in args)
    else:
        with Pool(N_WORKERS) as pool:
            results = pool.map(process_chunk, args)

    dt_settle = time.time() - t_settle
    print(f"Settling done in {dt_settle:.1f}s")

    # Merge states
    # For next-token prediction: State[t] -> Label[t+1]
    all_states, all_labels = [], []
    for states, labels in results:
        if len(states) > 1:
            all_states.append(states[:-1])   # t
            all_labels.append(labels[1:])    # t+1 (next token)

    X = np.vstack(all_states)
    y = np.concatenate(all_labels)
    print(f"  Total: {len(X)} (state, label) pairs")

    # Train/test split (80/20)
    n_tr = int(0.8 * len(X))
    idx  = _np.random.default_rng(0).permutation(len(X))
    X_tr, y_tr = X[idx[:n_tr]], y[idx[:n_tr]]
    X_te, y_te = X[idx[n_tr:]], y[idx[n_tr:]]

    print(f"\nTraining ridge readout (alpha={RIDGE_ALPHA}) ...")
    tr_acc, te_acc = ridge_acc(X_tr, y_tr, X_te, y_te)

    chance = 1.0 / VOCAB
    dt_total = time.time() - t0
    print(f"\n{'='*52}")
    print(f"RESERVOIR TEST RESULT")
    print(f"{'='*52}")
    print(f"  Total runtime        : {dt_total:.1f}s")
    print(f"  Train accuracy       : {tr_acc:.1%}")
    print(f"  Test accuracy        : {te_acc:.1%}   (chance: {chance:.1%})")
    print(f"  Improvement          : +{(te_acc - chance)*100:.1f} percentage points")
    print()

    if te_acc > 0.60:
        print("  -> TOPOLOGY IS INFORMATIVE!")
        print("     Fractal hierarchy encodes sequence patterns.")
        print("     Hebbian learning builds on this basis.")
    elif te_acc > chance + 0.08:
        print("  -> Weak signal present.")
        print("     Tip: increase tau_top or use more top clusters.")
    else:
        print("  -> No usable signal.")
        print("     Possible causes:")
        print("     - tau_top too small (core nodes 'forget' too quickly)")
        print("     - Settle time too short (increase WARMUP)")
        print("     - Too few levels for this pattern")
    print(f"{'='*52}")


if __name__ == '__main__':
    main()
