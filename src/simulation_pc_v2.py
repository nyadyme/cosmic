#!/usr/bin/env python3
"""
Full PC v2 -- correct two-phase loop
============================================
Free run -> Clamp -> Contrastive Hebbian (Equilibrium Propagation)

Algorithm per token t:
  1. FREE RUN  (no clamping, context from t-1)
     -> prediction state V_free
     -> Decode: pred = argmax(V_free[leaf_mu])  <- CORRECT location

  2. CLAMPED RUN (clamp true token t, same context)
     -> correction state V_clamp

  3. CONTRASTIVE ERROR:
     delta_eps[c] = V_eps_clamp[c] - V_eps_free[c]

  4. HEBBIAN  (EqProp approximation):
     dW[c_child, p_mu] = eta * delta_eps[c_child] * V_mu_free[p_mu]

  5. CARRY CONTEXT FORWARD:
     V_ctx = V_clamp

Reference: Contrastive Hebbian Learning / Equilibrium Propagation
(Scellier & Bengio 2017 -- exactly what chat.md proposes as fallback).
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
from collections import Counter, defaultdict
from pathlib import Path
import re, time, sys


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

sys.path.insert(0, str(Path(__file__).parent))
from cosmic_web_generator import (
    CosmicWebConfig, generate_cosmic_web, mu, eps, iota
)

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

NET = CosmicWebConfig(
    n_levels=3, eta=4, n_top=6,
    tau_leaf=1.0, tau_top=40.0,
    G_min=0.1, G_iota=0.1,
    W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3, seed=42,
)

I_AMP       = 1.0
DT          = 0.01
FREE_STEPS  = 300    # 3*tau_leaf / dt  (short prediction phase)
CLAMP_STEPS = 500    # 5*tau_leaf / dt  (full correction phase)
TOL         = 1e-4

ETA         = 0.15   # contrastive Hebbian learning rate
KAPPA       = 0.002  # spring to W_init (ADR-9)
RHO_TARGET  = 0.9

N_CORPUS    = 80_000
N_TRAIN     = 50_000
EVAL_WIN    = 1_000
PRINT_EVERY = 5_000
N_TEST_EVAL = 2_000


# ----------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------

def make_corpus(seeds, n=N_CORPUS, order=3, min_freq=15, rng_seed=7):
    raw = []
    for p in seeds:
        try:
            t = Path(p).read_text(encoding='utf-8', errors='replace')
            t = re.sub(r'[^\x20-\x7eaouAOUss\n]', ' ', t)
            t = re.sub(r' {2,}', ' ', t)
            raw.append(t)
        except FileNotFoundError:
            pass
    text = '\n'.join(raw)
    counts = Counter(text)
    vocab  = sorted(c for c, cnt in counts.items() if cnt >= min_freq)
    c2i    = {c: i for i, c in enumerate(vocab)}
    V      = len(vocab)

    table = defaultdict(Counter)
    for i in range(len(text) - order):
        ctx, nxt = text[i:i+order], text[i+order]
        if nxt in c2i and all(ch in c2i for ch in ctx):
            table[ctx][nxt] += 1
    probs = {ctx: {c: cnt/sum(d.values()) for c, cnt in d.items()}
             for ctx, d in table.items()}

    rng = _np.random.default_rng(rng_seed)
    ctx = text[:order]
    while not all(ch in c2i for ch in ctx):
        s = int(rng.integers(0, max(1, len(text)-order)))
        ctx = text[s:s+order]
    out = list(ctx)
    for _ in range(n - order):
        key = ''.join(out[-order:])
        if key in probs:
            chs = list(probs[key].keys())
            ps  = [probs[key][c] for c in chs]
            out.append(rng.choice(chs, p=ps))
        else:
            out.append(vocab[int(rng.integers(0, V))])
    corpus = np.asarray(_np.array([c2i.get(c, 0) for c in out], dtype=_np.int32))
    print(f"  Markov-{order} | V={V} | {n} characters")
    return corpus, V


# ----------------------------------------------------------------
# Reservoir setup
# ----------------------------------------------------------------

def setup(web, V_vocab):
    g_prec   = np.array([web.G.nodes[i]['G_prec'] for i in range(web.N)])
    C_inv    = 1.0 / np.maximum(np.asarray(web.tau) * g_prec, 1e-10)
    Y        = sp.csr_matrix(web.Y.tocsr())
    mu_idx   = np.array([mu(c) for c in range(web.n_clusters)])
    eps_idx  = np.array([eps(c) for c in range(web.n_clusters)])
    iota_idx = np.array([iota(c) for c in range(web.n_clusters)])
    G_mu     = g_prec[mu_idx]
    G_eps    = g_prec[eps_idx]

    max_lv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[mu(c)]['level'] == max_lv]
    assert len(leaves) >= V_vocab, \
        f"Too few leaf clusters ({len(leaves)}) for V={V_vocab}"
    emb = np.array([eps(leaves[i]) for i in range(V_vocab)])
    # mu nodes of leaf token clusters (mu = eps - 1 since mu(c)=3c, eps(c)=3c+1)
    emb_mu = emb - 1

    # spectrally scaled vertical weights
    vert = {(i, j): d['W']
            for i, j, d in web.G.edges(data=True)
            if d.get('etype') == 'vertical'}
    rows, cols, data = [], [], []
    if vert:
        rv, cv, dv = [], [], []
        for (i, j), w in vert.items():
            rv += [i, j]; cv += [j, i]; dv += [w, w]
        W_sp = sp.csr_matrix((dv, (rv, cv)), shape=(web.N, web.N))
        try:
            vals = eigs(W_sp.astype(float), k=1, which='LM',
                        return_eigenvectors=False, tol=1e-3, maxiter=300)
            rho0 = max(float(np.abs(vals[0])), 1e-6)
        except Exception:
            rho0 = 1.0
        scale = RHO_TARGET / rho0
        for (p_mu_, ec), w in vert.items():
            if web.kinds[p_mu_] == 0 and web.kinds[ec] == 1:
                c_child = web.G.nodes[ec]['cluster']
                rows.append(c_child); cols.append(p_mu_)
                data.append(w * scale)

    W_td0   = sp.csr_matrix(
        (data, (rows, cols)) if rows else ([], ([], [])),
        shape=(web.n_clusters, web.N))
    W_init  = dict(zip(zip(rows, cols), data))
    W0_coo  = W_td0.tocoo()

    return dict(C_inv=C_inv, Y=Y, emb=emb, emb_mu=emb_mu,
                W_td0=W_td0, W_init=W_init, W0_coo=W0_coo,
                mu_idx=mu_idx, eps_idx=eps_idx, iota_idx=iota_idx,
                G_mu=G_mu, G_eps=G_eps,
                N=web.N, n_cl=web.n_clusters)


def w_td_from_dict(W_learn, res):
    W0 = res['W0_coo']
    new_d = np.array([W_learn.get((int(r), int(c)), float(d))
                      for r, c, d in zip(W0.row, W0.col, W0.data)])
    return sp.csr_matrix(
        (new_d, (W0.row, W0.col)), shape=res['W_td0'].shape).tocsr()


# ----------------------------------------------------------------
# Euler settling  (separate mu/eps/iota)
# ----------------------------------------------------------------

def euler_settle(V_mu, V_eps, V_iota, I_inject,
                 res, W_td_mat, n_steps, tol=TOL, dt=DT):
    """n_steps Euler steps with early stopping."""
    N   = res['N']
    Y   = res['Y']
    Cinv= res['C_inv']
    mi, ei, ii = res['mu_idx'], res['eps_idx'], res['iota_idx']
    Gmu, Geps  = res['G_mu'], res['G_eps']

    V = np.zeros(N)
    V[mi] = V_mu; V[ei] = V_eps; V[ii] = V_iota

    for step in range(n_steps):
        f_c = np.tanh(W_td_mat.dot(V))
        I   = I_inject.copy()
        np.add.at(I, mi, Gmu * f_c)
        np.add.at(I, ei, Geps * f_c)
        dV  = dt * Cinv * (-Y.dot(V) + I)
        V   = np.clip(V + dV, -2.0, 2.0)
        if step > 5 and np.max(np.abs(dV)) < tol:
            break

    return V[mi].copy(), V[ei].copy(), V[ii].copy()


# ----------------------------------------------------------------
# Decode: periphery mu  (no Ridge)
# ----------------------------------------------------------------

def decode_mu(V_mu, res, V_vocab):
    """argmax over leaf-mu voltages of the token clusters."""
    # emb_mu[i] = global index of mu node for token i
    # mu nodes are in mu_idx, so we need the POSITION in mu_idx
    # mu(c) = 3c, and mu_idx[c] = 3c, so position in mu_idx = cluster index c
    # leaf cluster c_i has mu at position c_i in mu_idx
    # emb_mu[i] = 3 * leaves[i], which equals mu_idx[leaves[i]]
    # So: score[i] = V_mu[leaves[i]]
    # We can get cluster index from emb_mu:  c_i = emb_mu[i] // 3
    cluster_ids = res['emb_mu'] // 3   # cluster index for each token
    cluster_ids = np.clip(cluster_ids, 0, res['n_cl'] - 1)
    scores = V_mu[cluster_ids[:V_vocab]]
    return int(np.argmax(scores)), scores


# ----------------------------------------------------------------
# Contrastive Hebbian update (EqProp)
# ----------------------------------------------------------------

def contrastive_hebbian(V_mu_free, V_eps_free,
                        V_mu_clamp, V_eps_clamp,
                        W_learn, res, eta, kappa):
    """
    dW[c_child, p_mu] = eta * delta_eps[c_child] * V_mu_free[c_parent]
    delta_eps = V_eps_clamp - V_eps_free  (contrastive error)
    """
    delta_eps = V_eps_clamp - V_eps_free
    W0 = res['W0_coo']

    for idx in range(len(W0.data)):
        c_child = int(W0.row[idx])
        p_mu_g  = int(W0.col[idx])

        # Cluster index of parent mu (p_mu_g = 3*c_parent)
        c_parent = p_mu_g // 3
        if c_parent >= res['n_cl'] or c_child >= res['n_cl']:
            continue

        d_eps = float(delta_eps[c_child])
        mu_p  = float(V_mu_free[c_parent])
        w      = W_learn.get((c_child, p_mu_g), float(W0.data[idx]))
        w_init = float(W0.data[idx])

        dW = eta * d_eps * mu_p - kappa * (w - w_init)
        W_learn[(c_child, p_mu_g)] = float(np.clip(w + dW, 1e-3, 0.99))

    return W_learn


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------

def train(corpus_tr, corpus_te, res, V_vocab):
    W_learn = dict(res['W_init'])
    W_td    = w_td_from_dict(W_learn, res)

    V_mu    = np.zeros(res['n_cl'])
    V_eps   = np.zeros(res['n_cl'])
    V_iota  = np.zeros(res['n_cl'])

    recent, curve = [], []
    t0 = time.time()

    print(f"  {'Step':>8}  {'Roll-Acc':>9}  {'Time':>5}")
    print(f"  {'-'*28}")

    n = min(N_TRAIN, len(corpus_tr) - 1)
    for t in range(n):
        tok     = int(corpus_tr[t])     % V_vocab
        tok_nxt = int(corpus_tr[t+1])  % V_vocab

        # Phase 1: FREE RUN -> prediction
        I_free = np.zeros(res['N'])
        mu_f, eps_f, iota_f = euler_settle(
            V_mu, V_eps, V_iota, I_free, res, W_td, FREE_STEPS)

        pred, _ = decode_mu(mu_f, res, V_vocab)
        recent.append(int(pred == tok_nxt))

        # Phase 2: CLAMPED RUN -> correction
        I_clamp = np.zeros(res['N'])
        I_clamp[res['emb'][tok]] = I_AMP
        mu_c, eps_c, iota_c = euler_settle(
            V_mu, V_eps, V_iota, I_clamp, res, W_td, CLAMP_STEPS)

        # Phase 3: contrastive Hebbian
        W_learn = contrastive_hebbian(
            mu_f, eps_f, mu_c, eps_c, W_learn, res, ETA, KAPPA)

        # carry context forward
        V_mu, V_eps, V_iota = mu_c, eps_c, iota_c

        # update W_td & logging
        if (t + 1) % PRINT_EVERY == 0:
            W_td    = w_td_from_dict(W_learn, res)
            rolling = float(np.mean(recent[-EVAL_WIN:]))
            curve.append((t + 1, rolling))
            dt_ = time.time() - t0
            print(f"  Step {t+1:>6}:  {rolling:.1%}  ({dt_:.0f}s)")

    # test evaluation
    W_td_final = w_td_from_dict(W_learn, res)
    mu_e = np.zeros(res['n_cl'])
    eps_e = np.zeros(res['n_cl'])
    iota_e = np.zeros(res['n_cl'])
    correct_te = 0
    n_te = min(N_TEST_EVAL, len(corpus_te) - 1)

    for t in range(n_te):
        tok     = int(corpus_te[t])     % V_vocab
        tok_nxt = int(corpus_te[t+1]) % V_vocab
        I_free  = np.zeros(res['N'])
        mu_e, eps_e, iota_e = euler_settle(
            mu_e, eps_e, iota_e, I_free, res, W_td_final, FREE_STEPS)
        pred, _ = decode_mu(mu_e, res, V_vocab)
        correct_te += (pred == tok_nxt)
        # Carry clamped state for context
        I_cl = np.zeros(res['N'])
        I_cl[res['emb'][tok]] = I_AMP
        mu_e, eps_e, iota_e = euler_settle(
            mu_e, eps_e, iota_e, I_cl, res, W_td_final, CLAMP_STEPS)

    acc_te = correct_te / n_te
    return curve, acc_te


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    t0 = time.time()

    print("Corpus ...")
    seeds  = ['CONCEPT.md', 'fractal_llm.md', 'STAGE2_FRACTALITY.md']
    corpus, V = make_corpus(seeds)
    n_tr   = int(0.8 * len(corpus))
    corpus_tr, corpus_te = corpus[:n_tr], corpus[n_tr:]
    chance = 1.0 / V

    print("\nTopology ...")
    web = generate_cosmic_web(NET)
    res = setup(web, V)
    print(f"  {web.n_clusters} clusters | {web.N} nodes | V={V}")
    print(f"  Free={FREE_STEPS} steps | Clamp={CLAMP_STEPS} steps")
    print(f"  eta={ETA} | kappa={KAPPA} | RHO={RHO_TARGET}")

    print(f"\nTraining (Free->Clamp->EqProp, {N_TRAIN} steps) ...")
    curve, acc_te = train(corpus_tr, corpus_te, res, V)

    dt = time.time() - t0
    acc_start = curve[0][1]  if curve else 0.0
    acc_end   = curve[-1][1] if curve else 0.0
    gain      = (acc_end - acc_start) * 100

    print(f"\n{'='*52}")
    print(f"PC v2 RESULT  (V={V}, {dt:.0f}s)")
    print(f"{'='*52}")
    print(f"  Chance:          {chance:.1%}")
    print(f"  Curve start:     {acc_start:.1%}")
    print(f"  Curve end:       {acc_end:.1%}")
    print(f"  Hebbian gain:    {gain:+.1f} PP")
    print(f"  Test accuracy:   {acc_te:.1%}  (mu-decode, no Ridge)")
    print(f"  Reservoir-Ridge: 27.5%  (stage 5 reference)")

    if gain > 3.0:
        print(f"\n  -> EqProp / contrastive Hebbian WORKS!")
        print(f"     ADR-8 confirmed with correct two-phase loop.")
    elif gain > 0.5:
        print(f"\n  -> Weak learning signal ({gain:+.1f} PP).")
        print(f"     Test more steps or larger eta.")
    else:
        print(f"\n  -> No learning effect. Next step: network size or")
        print(f"     FORCE learning as a stronger alternative.")
    print(f"{'='*52}")


if __name__ == '__main__':
    main()
