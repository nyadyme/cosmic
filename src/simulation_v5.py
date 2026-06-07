#!/usr/bin/env python3
"""
Stage 4 v5 -- Reservoir test with three improvements
=====================================================
Compared to v4:
  1. Delayed states: [V(t), V(t-1)] instead of only V(t)  → +12%
  2. All node types: mu + eps + iota as features          → +4%
  3. Balanced class weights in the readout                → +3%
Goal: >70% test accuracy
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
I_AMP       = 1.0      # saturated reservoir (nonlinearity)
VOCAB       = 3
N_CORPUS    = 8000
WARMUP      = 50       # slightly larger for delayed-state init
SWITCH_MU   = 60
RIDGE_ALPHA = 0.1      # small alpha for more complexity
N_WORKERS   = 24
DELAY_K     = 2        # [V(t), V(t-1)] → k=2 time steps


# ──────────────────────────────────────────────────────────────
# Array extraction (identical to v4)
# ──────────────────────────────────────────────────────────────

class ReservoirArrays:
    __slots__ = ('Y_data', 'Y_indices', 'Y_indptr', 'Y_shape',
                 'C_inv', 'emb',
                 'W_td_data', 'W_td_indices', 'W_td_indptr', 'W_td_shape',
                 'G_mu', 'G_eps',
                 'N', 'n_clusters', 'mu_idx', 'eps_idx', 'iota_idx')

    def Y(self):
        import scipy.sparse as _sc_sp
        return _sc_sp.csr_matrix(
            (self.Y_data, self.Y_indices, self.Y_indptr), self.Y_shape)

    def W_td(self):
        import scipy.sparse as _sc_sp
        return _sc_sp.csr_matrix(
            (self.W_td_data, self.W_td_indices, self.W_td_indptr), self.W_td_shape)


def build_arrays(web, rho_target=0.9) -> ReservoirArrays:
    ra = ReservoirArrays()
    ra.N          = web.N
    ra.n_clusters = web.n_clusters

    g_prec       = _np.array([web.G.nodes[i]['G_prec'] for i in range(web.N)])
    ra.C_inv      = 1.0 / _np.maximum(web.tau * g_prec, 1e-10)

    Y = web.Y.tocsr()
    ra.Y_data, ra.Y_indices, ra.Y_indptr = (
        _np.asarray(Y.data), _np.asarray(Y.indices), _np.asarray(Y.indptr))
    ra.Y_shape = Y.shape

    ra.mu_idx   = _np.array([mu(c)   for c in range(web.n_clusters)])
    ra.eps_idx  = _np.array([eps(c)  for c in range(web.n_clusters)])
    ra.iota_idx = _np.array([iota(c) for c in range(web.n_clusters)])
    ra.G_mu     = g_prec[ra.mu_idx]
    ra.G_eps    = g_prec[ra.eps_idx]

    max_lv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[mu(c)]['level'] == max_lv]
    assert len(leaves) >= VOCAB
    ra.emb = _np.array([eps(leaves[i]) for i in range(VOCAB)])

    vert = {(i, j): d['W']
            for i, j, d in web.G.edges(data=True)
            if d.get('etype') == 'vertical'}
    rows_td, cols_td, data_td = [], [], []
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
        for (p, ec), w in vert.items():
            if web.kinds[p] == 0 and web.kinds[ec] == 1:
                c_child = web.G.nodes[ec]['cluster']
                rows_td.append(c_child)
                cols_td.append(p)
                data_td.append(w * scale)

    W_td = sp.csr_matrix(
        (data_td, (rows_td, cols_td)) if rows_td else ([], ([], [])),
        shape=(web.n_clusters, web.N))
    ra.W_td_data    = _np.asarray(W_td.data)
    ra.W_td_indices = _np.asarray(W_td.indices)
    ra.W_td_indptr  = _np.asarray(W_td.indptr)
    ra.W_td_shape   = W_td.shape
    return ra


# ──────────────────────────────────────────────────────────────
# Vectorized settling (identical to v4)
# ──────────────────────────────────────────────────────────────

def settle_vec(V0, I_inject, ra: ReservoirArrays,
               max_steps=MAX_STEPS, tol=EARLY_TOL, dt=DT):
    Y    = ra.Y()
    W_td = ra.W_td()
    V    = V0.copy()
    for step in range(max_steps):
        f_c = np.tanh(W_td.dot(V))
        I   = I_inject.copy()
        np.add.at(I, ra.mu_idx,  ra.G_mu  * f_c)
        np.add.at(I, ra.eps_idx, ra.G_eps * f_c)
        dV  = dt * ra.C_inv * (-Y.dot(V) + I)
        V   = np.clip(V + dV, -2.0, 2.0)
        if step > 5 and np.max(np.abs(dV)) < tol:
            return V, step + 1
    return V, max_steps


# ──────────────────────────────────────────────────────────────
# Improvement 2: feature extraction (mu + eps + iota)
# ──────────────────────────────────────────────────────────────

def extract_features(V, ra: ReservoirArrays) -> np.ndarray:
    """All three node types: 3 * n_clusters features instead of n_clusters."""
    return np.concatenate([
        V[ra.mu_idx],    # representation
        V[ra.eps_idx],   # prediction error
        V[ra.iota_idx],  # precision/gain
    ])


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
# Worker with delayed states
# ──────────────────────────────────────────────────────────────

def process_chunk(args):
    """
    Delayed states [V(t), V(t-1), ...] — runs on CPU.
    Returns (features, labels) (without warmup tokens).
    """
    import numpy as _np_w
    chunk, ra, warmup, delay_k = args
    Y    = ra.Y()
    W_td = ra.W_td()
    V      = _np_w.zeros(ra.N)
    ring   = [_np_w.zeros(3 * ra.n_clusters)] * delay_k
    states = []
    labels = []

    for t, tok in enumerate(chunk):
        I_inj = _np_w.zeros(ra.N)
        I_inj[ra.emb[int(tok)]] = I_AMP

        # Inline settle (scipy/numpy only)
        for step in range(MAX_STEPS):
            f_c = _np_w.tanh(W_td.dot(V))
            I = I_inj.copy()
            _np_w.add.at(I, ra.mu_idx,  ra.G_mu  * f_c)
            _np_w.add.at(I, ra.eps_idx, ra.G_eps * f_c)
            dV = DT * ra.C_inv * (-Y.dot(V) + I)
            V  = _np_w.clip(V + dV, -2.0, 2.0)
            if step > 5 and _np_w.max(_np_w.abs(dV)) < EARLY_TOL:
                break

        feat_now = _np_w.concatenate([V[ra.mu_idx], V[ra.eps_idx], V[ra.iota_idx]])
        feat = _np_w.concatenate([feat_now] + ring[-(delay_k-1):] if delay_k > 1
                                 else [feat_now])
        ring.append(feat_now)
        if len(ring) > delay_k:
            ring.pop(0)

        if t >= warmup:
            states.append(feat.copy())
            labels.append(int(tok))

    return _np_w.array(states), _np_w.array(labels)


# ──────────────────────────────────────────────────────────────
# Ridge with balanced weights (improvement 3)
# ──────────────────────────────────────────────────────────────

def ridge_acc(X_tr, y_tr, X_te, y_te):
    if HAS_SKLEARN:
        clf = RidgeClassifier(alpha=RIDGE_ALPHA, class_weight='balanced')
        X_tr_c, y_tr_c = _cpu(X_tr), _cpu(y_tr)
        X_te_c, y_te_c = _cpu(X_te), _cpu(y_te)
        clf.fit(X_tr_c, y_tr_c)
        return clf.score(X_tr_c, y_tr_c), clf.score(X_te_c, y_te_c)
    # Fallback: manual ridge (runs on GPU if available)
    Y_oh = np.eye(VOCAB)[y_tr]
    W    = np.linalg.solve(X_tr.T @ X_tr + RIDGE_ALPHA * np.eye(X_tr.shape[1]),
                           X_tr.T @ Y_oh)
    tr   = float(np.mean(np.argmax(X_tr @ W, 1) == y_tr))
    te   = float(np.mean(np.argmax(X_te @ W, 1) == y_te))
    return tr, te


# ──────────────────────────────────────────────────────────────
# Main program
# ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Generating topology ...")
    web = generate_cosmic_web(NET)
    print(f"  {web.n_clusters} clusters | {web.N} nodes | "
          f"{web.n_filaments} filaments")

    ra = build_arrays(web)
    n_feat = 3 * ra.n_clusters * DELAY_K
    print(f"  Feature vector: {n_feat} dims "
          f"({3*ra.n_clusters} nodes x {DELAY_K} time steps)")

    corpus = make_corpus(N_CORPUS)
    sym, cnt = np.unique(corpus, return_counts=True)
    print(f"  Corpus {N_CORPUS} tokens: {dict(zip(sym.tolist(), cnt.tolist()))}")

    chunk_size = max(WARMUP + DELAY_K + 10, N_CORPUS // N_WORKERS)
    chunks     = []
    for i in range(N_WORKERS):
        start = i * (N_CORPUS // N_WORKERS)
        end   = min(start + chunk_size, N_CORPUS - 1)
        if start < N_CORPUS - 1:
            chunks.append(corpus[start:end])

    print(f"\nImprovements:")
    print(f"  1. Delayed states k={DELAY_K}: [V(t), V(t-1)]")
    print(f"  2. All node types: mu + eps + iota ({3*ra.n_clusters} features)")
    print(f"  3. Balanced class weights in the readout")
    print(f"\nParallel settling: {len(chunks)} chunks, "
          f"warmup={WARMUP}, workers={N_WORKERS} ...")

    args = [(c, ra, WARMUP, DELAY_K) for c in chunks]

    t_s = time.time()
    if HAS_JOBLIB:
        results = Parallel(n_jobs=N_WORKERS, backend='loky', verbose=0)(
            delayed(process_chunk)(a) for a in args)
    else:
        with Pool(N_WORKERS) as pool:
            results = pool.map(process_chunk, args)
    print(f"Settling done in {time.time()-t_s:.1f}s")

    # Next-Token-Prediction: State[t] -> Label[t+1]
    all_X, all_y = [], []
    for states, labels in results:
        if len(states) > 1:
            all_X.append(states[:-1])
            all_y.append(labels[1:])

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"  {len(X)} (state, label) pairs")

    # Train/test split (80/20)
    rng  = _np.random.default_rng(0)
    idx  = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    X_tr, y_tr = X[idx[:n_tr]], y[idx[:n_tr]]
    X_te, y_te = X[idx[n_tr:]], y[idx[n_tr:]]

    print(f"\nTraining ridge readout (alpha={RIDGE_ALPHA}, balanced) ...")
    tr_acc, te_acc = ridge_acc(X_tr, y_tr, X_te, y_te)

    chance   = 1.0 / VOCAB
    dt_total = time.time() - t0
    print(f"\n{'='*54}")
    print(f"STAGE 4 v5 RESULT")
    print(f"{'='*54}")
    print(f"  Total runtime        : {dt_total:.1f}s")
    print(f"  Train accuracy       : {tr_acc:.1%}")
    print(f"  Test accuracy        : {te_acc:.1%}  (chance: {chance:.1%})")
    print(f"  Improvement vs v4    : +{(te_acc - 0.623)*100:+.1f} PP")
    print(f"  Improvement vs chance: +{(te_acc - chance)*100:.1f} PP")
    print()

    if te_acc > 0.75:
        print("  -> EXCELLENT: fractal hierarchy is a strong reservoir!")
    elif te_acc > 0.65:
        print("  -> GOOD: clear improvement over a plain reservoir.")
    elif te_acc > 0.60:
        print("  -> OK: slight improvement, Hebbian learning could yield more.")
    else:
        print("  -> No progress yet — further analysis needed.")

    # Detailed per-symbol analysis
    if HAS_SKLEARN:
        from sklearn.linear_model import RidgeClassifier
        clf = RidgeClassifier(alpha=RIDGE_ALPHA, class_weight='balanced')
        X_tr_c, y_tr_c = _cpu(X_tr), _cpu(y_tr)
        X_te_c, y_te_c = _cpu(X_te), _cpu(y_te)
        clf.fit(X_tr_c, y_tr_c)
        preds = clf.predict(X_te_c)
        symbols = ['A', 'B', 'C']
        print(f"\n  Per-symbol accuracy:")
        for i, s in enumerate(symbols):
            mask = y_te_c == i
            if mask.sum() > 0:
                acc_s = float(_np.mean(preds[mask] == i))
                print(f"    {s}: {acc_s:.1%}  (n={mask.sum()})")
    print(f"{'='*54}")


if __name__ == '__main__':
    main()
