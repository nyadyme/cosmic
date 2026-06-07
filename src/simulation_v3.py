#!/usr/bin/env python3
"""
Stage 4 v3 — Reservoir test with early stopping
================================================
Answers the core question: Does the fractal topology carry
information about sequences? (independent of the learning algorithm)

Approach: weights frozen (W_init), Euler settling with
early stopping, then linear readout (ridge regression).

Early stopping: aborts settling when ||dV|| < tol
(saves 50-90% of the Euler steps per token).
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

from cosmic_web_generator import (
    CosmicWebConfig, generate_cosmic_web, mu, eps, iota
)

# sklearn optional — fall back to manual ridge regression
try:
    from sklearn.linear_model import RidgeClassifier
    SKLEARN = True
except ImportError:
    SKLEARN = False


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

NET = CosmicWebConfig(
    n_levels=3, eta=3, n_top=4,    # 52 clusters, 156 nodes
    tau_leaf=1.0, tau_top=30.0,    # longer memory
    G_min=0.1,
    G_iota=0.1, W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3,
    seed=42,
)

DT          = 0.01     # Euler time step (stable: dt < 2/lambda_max ~ 0.02)
MAX_STEPS   = 500      # maximum Euler steps per token
EARLY_TOL   = 1e-4     # early stopping: ||dV|| < tol → settled
I_AMP       = 1.0      # token injection amplitude
VOCAB       = 3        # A=0, B=1, C=2
N_CORPUS    = 4000     # total corpus
N_TRAIN     = 3000     # training length
SWITCH_MU   = 60       # mean pattern-switch length
RIDGE_ALPHA = 1.0      # ridge regularization


# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────

def build_system(web):
    """Returns C_inv, Y, emb, W_init."""
    g_prec = np.array([web.G.nodes[i]['G_prec'] for i in range(web.N)])
    C_inv  = 1.0 / np.maximum(np.asarray(web.tau) * g_prec, 1e-10)

    # Spectrally scaled vertical weights
    vert = {(i, j): d['W']
            for i, j, d in web.G.edges(data=True)
            if d.get('etype') == 'vertical'}
    if vert:
        rows, cols, data = [], [], []
        for (i, j), w in vert.items():
            rows += [i, j]; cols += [j, i]; data += [w, w]
        W_sp = sp.csr_matrix((data, (rows, cols)), shape=(web.N, web.N))
        try:
            vals = eigs(W_sp.astype(float), k=1, which='LM',
                        return_eigenvectors=False, tol=1e-3, maxiter=300)
            rho0 = max(float(np.abs(vals[0])), 1e-6)
        except Exception:
            rho0 = 1.0
        scale = 0.9 / rho0
        W_init = {k: float(np.clip(v * scale, 1e-3, 0.99))
                  for k, v in vert.items()}
    else:
        W_init = {}

    # Token embedding: one leaf eps node per token
    max_lv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[mu(c)]['level'] == max_lv]
    assert len(leaves) >= VOCAB, f"Too few leaf clusters ({len(leaves)})"
    emb = np.array([eps(leaves[i]) for i in range(VOCAB)], dtype=int)

    return C_inv, sp.csr_matrix(web.Y.tocsr()), emb, W_init


# ─────────────────────────────────────────────────────────────
# Euler settling with early stopping
# ─────────────────────────────────────────────────────────────

def settle(V0, I_inject, C_inv, Y, web, W_init,
           max_steps=MAX_STEPS, tol=EARLY_TOL, dt=DT):
    """
    Euler settling with early stopping.
    Aborts when max(|dV|) < tol (activity change negligible).
    Returns (V_settled, steps_used).
    """
    V = V0.copy()
    for step in range(max_steps):
        # Nonlinear I_pred (top-down prediction, simplified)
        I = I_inject.copy()
        for c in range(web.n_clusters):
            ps = 0.0
            for p, ed in web.G[eps(c)].items():
                if ed.get('etype') == 'vertical' and web.kinds[p] == 0:
                    w = W_init.get((p, eps(c)),
                            W_init.get((eps(c), p), ed['W']))
                    ps += w * V[p]
            f_c = float(np.tanh(ps))
            g   = web.G.nodes[mu(c)]['G_prec']
            I[mu(c)]  += g * f_c
            I[eps(c)] += g * f_c

        dV = dt * C_inv * (-Y.dot(V) + I)
        V  = np.clip(V + dV, -2.0, 2.0)

        # Early stopping
        if np.max(np.abs(dV)) < tol:
            return V, step + 1

    return V, max_steps


# ─────────────────────────────────────────────────────────────
# Corpus generator
# ─────────────────────────────────────────────────────────────

def make_corpus(n, seed=1, switch_mu=SWITCH_MU):
    pats  = [[0, 1, 2], [0, 0, 1, 1]]
    rng   = _np.random.default_rng(seed)
    seq, pat, pos, until = [], 0, 0, switch_mu
    while len(seq) < n:
        seq.append(pats[pat][pos % len(pats[pat])])
        pos += 1; until -= 1
        if until <= 0:
            pat = 1 - pat; pos = 0
            until = max(1, int(rng.poisson(switch_mu)))
    return np.asarray(_np.array(seq[:n], dtype=_np.int8))


# ─────────────────────────────────────────────────────────────
# Ridge regression (fallback without sklearn)
# ─────────────────────────────────────────────────────────────

def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    """Closed-form ridge solution: W = (X'X + alpha*I)^{-1} X'Y_onehot."""
    n_cls = VOCAB
    Y_oh  = np.eye(n_cls)[y]
    A     = X.T @ X + alpha * np.eye(X.shape[1])
    W     = np.linalg.solve(A, X.T @ Y_oh)
    return W


def ridge_predict(X, W):
    return np.argmax(X @ W, axis=1)


# ─────────────────────────────────────────────────────────────
# Main program
# ─────────────────────────────────────────────────────────────

def main():
    print("Generating topology ...")
    web = generate_cosmic_web(NET)
    print(f"  {web.n_clusters} clusters | {web.N} nodes | "
          f"{web.n_filaments} filaments | d_H={web.d_H_angular:.2f}")

    C_inv, Y, emb, W_init = build_system(web)
    print(f"  Token eps nodes: {emb}  | tau: {web.tau.min():.1f}..{web.tau.max():.1f}")
    print(f"  Learnable edges: {len(W_init)}")

    corpus = make_corpus(N_CORPUS)
    sym, cnt = np.unique(corpus, return_counts=True)
    print(f"  Corpus {N_CORPUS} tokens: {dict(zip(sym.tolist(), cnt.tolist()))}")
    print(f"\nCollecting reservoir states (early stopping tol={EARLY_TOL}) ...")

    states      = []
    steps_log   = []
    V           = np.zeros(web.N)

    for t, tok in enumerate(corpus):
        I_inj       = np.zeros(web.N)
        I_inj[emb[int(tok)]] = I_AMP

        V, steps    = settle(V, I_inj, C_inv, Y, web, W_init)
        steps_log.append(steps)

        # Only mu nodes as feature vector (representation states)
        states.append(V[[mu(c) for c in range(web.n_clusters)]].copy())

        if (t + 1) % 500 == 0 or t < 5:
            avg_steps = np.mean(steps_log[-min(20, len(steps_log)):])
            vmax      = float(np.max(np.abs(V)))
            print(f"  Token {t+1:>4}: steps={steps:>3}  "
                  f"avg={avg_steps:.0f}  V_max={vmax:.3f}")

    avg_all = np.mean(steps_log)
    print(f"\nAverage early stopping: {avg_all:.0f}/{MAX_STEPS} steps "
          f"({100*avg_all/MAX_STEPS:.0f}% of the maximum)")

    # ── Ridge readout ──────────────────────────────────────────────────────
    print("\nTraining linear readout (ridge) ...")
    X = np.array(states)
    y = corpus.astype(int)

    X_tr, y_tr = X[:N_TRAIN - 1],   y[1:N_TRAIN]      # t → t+1
    X_te, y_te = X[N_TRAIN:-1],     y[N_TRAIN + 1:]

    if SKLEARN:
        clf = RidgeClassifier(alpha=RIDGE_ALPHA)
        clf.fit(_cpu(X_tr), _cpu(y_tr))
        tr_acc = clf.score(_cpu(X_tr), _cpu(y_tr))
        te_acc = clf.score(_cpu(X_te), _cpu(y_te))
    else:
        W_out  = ridge_fit(X_tr, y_tr)
        tr_acc = float(np.mean(ridge_predict(X_tr, W_out) == y_tr))
        te_acc = float(np.mean(ridge_predict(X_te, W_out) == y_te))

    chance = 1.0 / VOCAB
    print(f"\n{'='*50}")
    print(f"RESERVOIR TEST RESULT")
    print(f"{'='*50}")
    print(f"  Train accuracy : {tr_acc:.1%}")
    print(f"  Test accuracy  : {te_acc:.1%}   (chance: {chance:.1%})")
    print(f"  Improvement    : +{(te_acc - chance)*100:.1f} percentage points")
    print()
    if te_acc > 0.60:
        print("  -> TOPOLOGY IS INFORMATIVE: sequence patterns are encoded!")
        print("     Hebbian learning can build on this basis.")
    elif te_acc > chance + 0.05:
        print("  -> Weak signal: topology carries something,")
        print("     but tau_top or network size needs to increase.")
    else:
        print("  -> No usable signal. Possible causes:")
        print("     - tau_top too small (top nodes 'forget' too quickly)")
        print("     - Too few top clusters (n_top=4 for 3 token types)")
        print("     - Settle time too short")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
