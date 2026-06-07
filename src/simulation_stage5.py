#!/usr/bin/env python3
"""
Stage 5 -- Fractal LLM on real text with Hebbian learning
============================================================
Three phases:
  Phase 1: reservoir baseline  (fixed weights)
  Phase 2: Hebbian learning curve   (sequential, W adapts)
  Phase 3: RNN baseline        (numpy backprop, same parameter count)

Corpus: Markov-3 model on seed texts (small texts + synthetic)
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
import re, time

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
    """CuPy array → CPU NumPy; NumPy unchanged."""
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

# ── Configuration ────────────────────────────────────────────

NET = CosmicWebConfig(
    n_levels=3, eta=4, n_top=6,      # 126 clusters, 96 leaves ≥ V
    tau_leaf=1.0, tau_top=40.0,
    G_min=0.1, G_iota=0.1,
    W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3, seed=42,
)

DT, MAX_STEPS, TOL = 0.01, 500, 1e-4
I_AMP    = 1.0
N_WORKERS= 24
DELAY_K  = 2
WARMUP   = 50
RIDGE_A  = 0.1

ETA_HEBB = 0.05
KAPPA    = 0.005
RHO_TGT  = 0.9

N_CORPUS     = 80_000
N_HEBB_TRAIN = 20_000   # sequential Hebbian path
N_HEBB_EVAL  = 500      # mini-eval per checkpoint
HEBB_CKPT    = 2_000    # evaluation every N steps


# ── Markov corpus ────────────────────────────────────────────

def make_markov_corpus(seed_paths, n=N_CORPUS, order=3, min_freq=15, seed=7):
    raw = []
    for p in seed_paths:
        try:
            t = Path(p).read_text(encoding='utf-8', errors='replace')
            t = re.sub(r'[^\x20-\x7eäöüÄÖÜß\n]', ' ', t)
            t = re.sub(r' {2,}', ' ', t)
            raw.append(t)
        except FileNotFoundError:
            pass
    text = '\n'.join(raw)
    counts  = Counter(text)
    vocab   = sorted(c for c, cnt in counts.items() if cnt >= min_freq)
    c2i     = {c: i for i, c in enumerate(vocab)}
    i2c     = {i: c for c, i in c2i.items()}
    V       = len(vocab)

    table = defaultdict(Counter)
    for i in range(len(text) - order):
        ctx = text[i:i+order]
        nxt = text[i+order]
        if nxt in c2i and all(c in c2i for c in ctx):
            table[ctx][nxt] += 1
    probs = {ctx: {c: n/sum(d.values()) for c, n in d.items()}
             for ctx, d in table.items()}

    rng = _np.random.default_rng(seed)
    ctx = text[:order]
    while not all(c in c2i for c in ctx):
        s = int(rng.integers(0, max(1, len(text)-order)))
        ctx = text[s:s+order]
    out = list(ctx)
    for _ in range(n - order):
        key = ''.join(out[-order:])
        if key in probs:
            ch = list(probs[key].keys())
            p  = [probs[key][c] for c in ch]
            nxt = rng.choice(ch, p=p)
        else:
            nxt = vocab[int(rng.integers(0, V))]
        out.append(nxt)
    corpus = np.asarray(_np.array([c2i.get(c, 0) for c in out], dtype=_np.int32))
    print(f"  Markov-{order} | V={V} | {n} characters | "
          f"seed: {[Path(p).name for p in seed_paths if Path(p).exists()]}")
    return corpus, V, i2c


# ── Reservoir setup ──────────────────────────────────────────

def build_reservoir(web, V):
    g_prec   = np.array([web.G.nodes[i]['G_prec'] for i in range(web.N)])
    C_inv    = 1.0 / np.maximum(np.asarray(web.tau) * g_prec, 1e-10)
    Y        = sp.csr_matrix(web.Y.tocsr())
    mu_idx   = np.array([mu(c) for c in range(web.n_clusters)])
    eps_idx  = np.array([eps(c) for c in range(web.n_clusters)])
    iota_idx = np.array([iota(c) for c in range(web.n_clusters)])
    G_mu     = g_prec[mu_idx]
    G_eps    = g_prec[eps_idx]

    max_lv   = int(web.levels.max())
    leaves   = [c for c in range(web.n_clusters)
                if web.G.nodes[mu(c)]['level'] == max_lv]
    assert len(leaves) >= V, f"leaf clusters ({len(leaves)}) < V ({V})"
    emb = np.array([eps(leaves[i]) for i in range(V)])

    # vertical weights + spectral scaling
    vert = {(i, j): d['W']
            for i, j, d in web.G.edges(data=True)
            if d.get('etype') == 'vertical'}
    coo_r, coo_c, coo_d = [], [], []
    if vert:
        rv, cv, dv = [], [], []
        for (i,j), w in vert.items(): rv+=[i,j]; cv+=[j,i]; dv+=[w,w]
        W_sp = sp.csr_matrix((dv,(rv,cv)), shape=(web.N, web.N))
        try:
            vals = eigs(W_sp.astype(float), k=1, which='LM',
                        return_eigenvectors=False, tol=1e-3, maxiter=300)
            rho0 = max(float(np.abs(vals[0])), 1e-6)
        except Exception:
            rho0 = 1.0
        scale = RHO_TGT / rho0
        for (p_mu, ec), w in vert.items():
            if web.kinds[p_mu] == 0 and web.kinds[ec] == 1:
                c_child = web.G.nodes[ec]['cluster']
                coo_r.append(c_child); coo_c.append(p_mu)
                coo_d.append(w * scale)

    W_td0 = sp.csr_matrix(
        (coo_d, (coo_r, coo_c)) if coo_r else ([],([],[])),
        shape=(web.n_clusters, web.N))
    W_init = dict(zip(zip(coo_r, coo_c), coo_d))

    return dict(C_inv=C_inv, Y=Y, emb=emb, W_td0=W_td0, W_init=W_init,
                mu_idx=mu_idx, eps_idx=eps_idx, iota_idx=iota_idx,
                G_mu=G_mu, G_eps=G_eps, N=web.N, n_cl=web.n_clusters)


def w_td_from_learn(res, W_learn):
    """Builds W_td CSR matrix from learned W_learn dict."""
    W0 = res['W_td0']
    W0_coo = W0.tocoo()
    new_d = np.array([W_learn.get((int(r), int(c)), float(d))
                      for r, c, d in zip(W0_coo.row, W0_coo.col, W0_coo.data)])
    return sp.csr_matrix((new_d, (W0_coo.row, W0_coo.col)),
                         shape=W0.shape).tocsr()


# ── Euler settling ───────────────────────────────────────────

def settle(V0, I_inj, res, W_td):
    Y, C_inv = res['Y'], res['C_inv']
    mu_idx, eps_idx = res['mu_idx'], res['eps_idx']
    G_mu, G_eps = res['G_mu'], res['G_eps']
    V = V0.copy()
    for step in range(MAX_STEPS):
        f_c = np.tanh(W_td.dot(V))
        I   = I_inj.copy()
        np.add.at(I, mu_idx,  G_mu  * f_c)
        np.add.at(I, eps_idx, G_eps * f_c)
        dV  = DT * C_inv * (-Y.dot(V) + I)
        V   = np.clip(V + dV, -2.0, 2.0)
        if step > 5 and np.max(np.abs(dV)) < TOL:
            return V
    return V


def feat(V, res):
    return np.concatenate([V[res['mu_idx']], V[res['eps_idx']],
                           V[res['iota_idx']]])


# ── Parallel chunk processing (Phase 1) ───────────────────

def _chunk_worker(args):
    import scipy.sparse as _sp
    import numpy as _np_w
    chunk, res_pack, warmup = args
    C_inv   = res_pack['C_inv']
    Y       = _sp.csr_matrix((res_pack['Y_d'], res_pack['Y_i'], res_pack['Y_p']),
                              res_pack['Y_s'])
    emb     = res_pack['emb']
    W_td    = _sp.csr_matrix((res_pack['W_d'], res_pack['W_i'], res_pack['W_p']),
                              res_pack['W_s'])
    mu_idx  = res_pack['mu_idx']
    eps_idx = res_pack['eps_idx']
    iota_idx= res_pack['iota_idx']
    G_mu    = res_pack['G_mu']
    G_eps   = res_pack['G_eps']
    N, n_cl = res_pack['N'], res_pack['n_cl']

    res_w = dict(C_inv=C_inv, Y=Y, emb=emb, W_td0=W_td,
                 mu_idx=mu_idx, eps_idx=eps_idx, iota_idx=iota_idx,
                 G_mu=G_mu, G_eps=G_eps, N=N, n_cl=n_cl)

    V    = _np_w.zeros(N)
    ring = [_np_w.zeros(3*n_cl)] * DELAY_K
    sts, lbs = [], []
    for t, tok in enumerate(chunk):
        if tok >= len(emb): continue
        I_inj = _np_w.zeros(N); I_inj[emb[tok]] = I_AMP

        # --- Inline settle (scipy/numpy only, no CuPy in worker) ---
        V2 = V.copy()
        for _ in range(MAX_STEPS):
            f_c = _np_w.tanh(W_td.dot(V2))
            I   = I_inj.copy()
            _np_w.add.at(I, mu_idx,  G_mu  * f_c)
            _np_w.add.at(I, eps_idx, G_eps * f_c)
            dV  = DT * C_inv * (-Y.dot(V2) + I)
            V2  = _np_w.clip(V2 + dV, -2.0, 2.0)
            if _ > 5 and _np_w.max(_np_w.abs(dV)) < TOL:
                break
        V = V2

        fn = _np_w.concatenate([V[mu_idx], V[eps_idx], V[iota_idx]])
        ft = _np_w.concatenate([fn] + ring[-(DELAY_K-1):] if DELAY_K > 1 else [fn])
        ring.append(fn)
        if len(ring) > DELAY_K: ring.pop(0)
        if t >= warmup: sts.append(ft.copy()); lbs.append(tok)
    return _np_w.array(sts), _np_w.array(lbs)


def parallel_collect(corpus, res, W_learn=None):
    """Collect reservoir states in parallel (24 cores)."""
    W_td   = w_td_from_learn(res, W_learn) if W_learn else res['W_td0']
    W0     = W_td.tocsr()
    # CuPy arrays → CPU NumPy before serializing for worker processes
    pack   = dict(
        C_inv=_cpu(res['C_inv']),
        Y_d=_cpu(res['Y'].data), Y_i=_cpu(res['Y'].indices),
        Y_p=_cpu(res['Y'].indptr), Y_s=res['Y'].shape,
        emb=_cpu(res['emb']),
        W_d=_cpu(W0.data), W_i=_cpu(W0.indices), W_p=_cpu(W0.indptr),
        W_s=W0.shape,
        mu_idx=_cpu(res['mu_idx']), eps_idx=_cpu(res['eps_idx']),
        iota_idx=_cpu(res['iota_idx']),
        G_mu=_cpu(res['G_mu']), G_eps=_cpu(res['G_eps']),
        N=res['N'], n_cl=res['n_cl'])

    chunk_size = max(WARMUP+DELAY_K+5, len(corpus)//N_WORKERS)
    chunks = [corpus[i*(len(corpus)//N_WORKERS):
                     min(i*(len(corpus)//N_WORKERS)+chunk_size, len(corpus)-1)]
              for i in range(N_WORKERS)
              if i*(len(corpus)//N_WORKERS) < len(corpus)-1]

    args = [(c, pack, WARMUP) for c in chunks]
    if HAS_JOBLIB:
        results = Parallel(n_jobs=N_WORKERS, backend='loky', verbose=0)(
            delayed(_chunk_worker)(a) for a in args)
    else:
        with Pool(N_WORKERS) as pool:
            results = pool.map(_chunk_worker, args)

    X_all, y_all = [], []
    for s, l in results:
        if len(s) > 1: X_all.append(s[:-1]); y_all.append(l[1:])
    return np.vstack(X_all), np.concatenate(y_all)


def fit_ridge(X_tr, y_tr, X_te, y_te, V):
    y_tr = np.clip(y_tr, 0, V-1)
    y_te = np.clip(y_te, 0, V-1)
    if HAS_SKLEARN:
        clf = RidgeClassifier(alpha=RIDGE_A, class_weight='balanced')
        clf.fit(_cpu(X_tr), _cpu(y_tr))
        return clf.score(_cpu(X_tr), _cpu(y_tr)), clf.score(_cpu(X_te), _cpu(y_te))
    A = X_tr.T @ X_tr + RIDGE_A * np.eye(X_tr.shape[1])
    W = np.linalg.lstsq(A, X_tr.T @ np.eye(V)[y_tr], rcond=None)[0]
    tr = float(np.mean(np.argmax(X_tr @ W, 1) == y_tr))
    te = float(np.mean(np.argmax(X_te @ W, 1) == y_te))
    return tr, te


# ── Phase 2: Sequential Hebbian ───────────────────────────

def hebbian_train(corpus_tr, corpus_te, res, V):
    """
    Sequential Hebbian path: W adapts token-by-token.
    Returns learning curve (accuracy vs. training step).
    """
    W_learn = dict(res['W_init'])
    W_td    = res['W_td0']
    V_state = np.zeros(res['N'])
    emb     = res['emb']

    # eval set: first N_HEBB_EVAL tokens of the test corpus
    eval_corpus = corpus_te[:N_HEBB_EVAL + 1]

    curve = []   # (step, accuracy)

    for t in range(min(N_HEBB_TRAIN, len(corpus_tr) - 1)):
        tok = int(corpus_tr[t])
        if tok >= len(emb): continue
        I_inj = np.zeros(res['N']); I_inj[emb[tok]] = I_AMP
        V_pre = V_state.copy()
        V_state = settle(V_state, I_inj, res, W_td)

        # Hebbian + spring (ADR-8, ADR-9)
        W0 = res['W_td0'].tocoo()
        for r_idx in range(len(W0.data)):
            c_child, p_mu = int(W0.row[r_idx]), int(W0.col[r_idx])
            if eps(c_child) >= res['N'] or p_mu >= res['N']: continue
            w     = W_learn.get((c_child, p_mu), float(W0.data[r_idx]))
            dV_e  = float(V_state[eps(c_child)] - V_pre[eps(c_child)])
            V_m   = float(V_pre[p_mu])
            f_w   = w * (1.0 - w)
            dW    = ETA_HEBB * V_m * dV_e * f_w \
                    - KAPPA * (w - float(W0.data[r_idx]))
            W_learn[(c_child, p_mu)] = min(0.99, max(1e-3, w + dW))

        # checkpoint: mini-eval with greedy decode
        if (t + 1) % HEBB_CKPT == 0:
            W_td = w_td_from_learn(res, W_learn)
            V_e  = np.zeros(res['N'])
            correct = 0
            for s in range(min(N_HEBB_EVAL, len(eval_corpus)-1)):
                tok_e = int(eval_corpus[s])
                nxt_e = int(eval_corpus[s+1])
                if tok_e >= len(emb): continue
                I_e = np.zeros(res['N']); I_e[emb[tok_e]] = I_AMP
                V_e = settle(V_e, I_e, res, W_td)
                # greedy decode: highest mu value among the token nodes
                scores = np.array([float(V_e[emb[i] - 1])
                                   for i in range(min(V, len(emb)))])
                pred = int(np.argmax(scores))
                correct += (pred == nxt_e % V)
            acc = correct / min(N_HEBB_EVAL, len(eval_corpus)-1)
            curve.append((t+1, acc))
            print(f"    Step {t+1:>6}: acc={acc:.1%}")

    return curve, W_learn


# ── Phase 3: RNN baseline ────────────────────────────────────

def rnn_baseline(corpus_tr, corpus_te, V, H=None, n_steps=5000):
    """Elman RNN, 1-step BPTT (always runs on CPU)."""
    corpus_tr_cpu = _cpu(corpus_tr)
    corpus_te_cpu = _cpu(corpus_te)
    if H is None:
        H = min(128, 3 * 52)
    rng = _np.random.default_rng(1)
    Wxh = rng.normal(0, 0.01, (H, V))
    Whh = rng.normal(0, 0.01, (H, H))
    Why = rng.normal(0, 0.01, (V, H))
    bh  = _np.zeros(H)
    by  = _np.zeros(V)
    eta = 0.01

    h = _np.zeros(H)
    correct_tr = 0
    n_tr = min(n_steps, len(corpus_tr_cpu)-1)

    for t in range(n_tr):
        tok = int(corpus_tr_cpu[t])   % V
        nxt = int(corpus_tr_cpu[t+1]) % V
        x   = _np.zeros(V); x[tok] = 1

        h_new = _np.tanh(Wxh @ x + Whh @ h + bh)
        logits = Why @ h_new + by
        logits -= logits.max()
        probs  = _np.exp(logits) / _np.exp(logits).sum()
        correct_tr += (int(_np.argmax(probs)) == nxt)

        dy   = probs.copy(); dy[nxt] -= 1
        dWhy = _np.outer(dy, h_new)
        dby  = dy.copy()
        dh   = (Why.T @ dy) * (1 - h_new**2)
        dWxh = _np.outer(dh, x)
        dWhh = _np.outer(dh, h)
        dbh  = dh.copy()

        for p, dp in [(Wxh,dWxh),(Whh,dWhh),(Why,dWhy),(bh,dbh),(by,dby)]:
            p -= eta * dp
        h = h_new

    # Test
    h, correct_te = _np.zeros(H), 0
    n_te = len(corpus_te_cpu) - 1
    for t in range(n_te):
        tok = int(corpus_te_cpu[t])   % V
        nxt = int(corpus_te_cpu[t+1]) % V
        x   = _np.zeros(V); x[tok] = 1
        h   = _np.tanh(Wxh @ x + Whh @ h + bh)
        logits = Why @ h + by
        logits -= logits.max()
        probs  = _np.exp(logits) / _np.exp(logits).sum()
        correct_te += (int(_np.argmax(probs)) == nxt)

    return correct_tr / n_tr, correct_te / n_te


# ── Main program ────────────────────────────────────────────

def main():
    t0 = time.time()

    # corpus
    print("Corpus (Markov-3) ...")
    seeds = ['CONCEPT.md', 'fractal_llm.md', 'STAGE2_FRACTALITY.md']
    corpus, V, i2c = make_markov_corpus(seeds, n=N_CORPUS)
    n_tr = int(0.8 * len(corpus))
    corpus_tr, corpus_te = corpus[:n_tr], corpus[n_tr:]

    # topology
    print("\nTopology ...")
    web = generate_cosmic_web(NET)
    print(f"  {web.n_clusters} clusters | {web.N} nodes | "
          f"{web.n_filaments} filaments")
    res = build_reservoir(web, V)
    print(f"  leaf clusters: {len([c for c in range(web.n_clusters) if web.G.nodes[mu(c)]['level']==int(web.levels.max())])} | V={V}")
    print(f"  feature dim: {3*web.n_clusters*DELAY_K}")

    chance = 1.0 / V

    # ── Phase 1 ──────────────────────────────────────────────
    print(f"\nPhase 1: reservoir baseline (24 cores) ...")
    t1 = time.time()
    X_tr1, y_tr1 = parallel_collect(corpus_tr, res)
    X_te1, y_te1 = parallel_collect(corpus_te, res)
    p1_tr, p1_te = fit_ridge(X_tr1, y_tr1, X_te1, y_te1, V)
    print(f"  {time.time()-t1:.0f}s | Train:{p1_tr:.1%} Test:{p1_te:.1%} "
          f"(chance:{chance:.1%})")

    # ── Phase 2 ──────────────────────────────────────────────
    print(f"\nPhase 2: Hebbian learning curve (sequential, {N_HEBB_TRAIN} steps)")
    print(f"  eta={ETA_HEBB}, kappa={KAPPA}, checkpoint every {HEBB_CKPT} steps")
    t2 = time.time()
    curve, W_learned = hebbian_train(corpus_tr, corpus_te, res, V)
    p2_baseline = curve[0][1] if curve else 0
    p2_final    = curve[-1][1] if curve else 0
    gain        = (p2_final - p2_baseline) * 100
    print(f"  {time.time()-t2:.0f}s | start:{p2_baseline:.1%} -> end:{p2_final:.1%} "
          f"(Hebbian gain: {gain:+.1f} PP)")

    # ── Phase 3 ──────────────────────────────────────────────
    print(f"\nPhase 3: RNN baseline (numpy, {5000} steps) ...")
    t3 = time.time()
    p3_tr, p3_te = rnn_baseline(corpus_tr, corpus_te, V)
    print(f"  {time.time()-t3:.0f}s | Train:{p3_tr:.1%} Test:{p3_te:.1%}")

    # ── Result ─────────────────────────────────────────────
    dt = time.time() - t0
    print(f"\n{'='*62}")
    print(f"STAGE 5 RESULT  (V={V}, {len(corpus)} characters, {dt:.0f}s)")
    print(f"{'='*62}")
    print(f"  {'Method':<32}  {'Test':>7}  {'vs chance':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Chance':<32}  {chance:>7.1%}  {'0.0 PP':>10}")
    print(f"  {'Reservoir (Phase 1)':<32}  {p1_te:>7.1%}  "
          f"{(p1_te-chance)*100:>+8.1f} PP")
    print(f"  {'Hebbian start (greedy)':<32}  {p2_baseline:>7.1%}  "
          f"{(p2_baseline-chance)*100:>+8.1f} PP")
    print(f"  {'Hebbian end   (greedy)':<32}  {p2_final:>7.1%}  "
          f"{(p2_final-chance)*100:>+8.1f} PP")
    print(f"  {'RNN baseline B-1 (numpy)':<32}  {p3_te:>7.1%}  "
          f"{(p3_te-chance)*100:>+8.1f} PP")
    print(f"  {'-'*52}")
    print(f"  Hebbian gain: {gain:+.1f} PP "
          f"({'ADR-8 confirmed!' if gain > 2 else 'minimal effect'})")
    print(f"  Reservoir vs. RNN: {(p1_te-p3_te)*100:+.1f} PP")
    print(f"\n  Finding: The fractal topology encodes Markov-3 text with")
    print(f"  {(p1_te/chance):.1f}x more accuracy than chance,")
    print(f"  {('exceeds' if p1_te > p3_te else 'does not yet reach')} "
          f"the numpy RNN baseline.")
    print(f"{'='*62}")


if __name__ == '__main__':
    main()
