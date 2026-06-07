#!/usr/bin/env python3
"""
Stage 5b -- Hebbian vs. reservoir: Ridge-fair comparison
==========================================================
Direct comparison with identical readout (Ridge):
  A) Reservoir with W_init  -> Ridge accuracy (baseline)
  B) Reservoir with W_hebb  -> Ridge accuracy after Hebbian training

If B > A: Hebbian improves the reservoir representations (ADR-8 confirmed).
If B ~ A: weight changes too small or direction wrong.

Three Hebbian variants: weak / medium / strong (eta sweep).
"""

try:
    import cupy as np
    import cupyx.scipy.sparse as sp
    _GPU = True
except ImportError:
    import numpy as np
    import scipy.sparse as sp
    _GPU = False

import numpy as _np
import time
import sys
from pathlib import Path


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

sys.path.insert(0, str(Path(__file__).parent))
from simulation_stage5 import (
    make_markov_corpus, build_reservoir, parallel_collect,
    w_td_from_learn, settle, fit_ridge,
    NET, N_CORPUS, WARMUP, DELAY_K, RIDGE_A, I_AMP,
)
from cosmic_web_generator import generate_cosmic_web, mu, eps


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

N_HEBB       = 30_000   # Hebbian training steps
EVAL_SPLIT   = 0.6      # fraction of training corpus for Hebbian

VARIANTS = [
    dict(name="weak",   eta=0.02, kappa=0.005),
    dict(name="medium", eta=0.10, kappa=0.002),
    dict(name="strong", eta=0.30, kappa=0.001),
]


# ----------------------------------------------------------------
# Sequential Hebbian training
# ----------------------------------------------------------------

def hebbian_train(corpus_hebb, res, eta, kappa, n_max=None):
    """
    Trains W_learn sequentially on corpus_hebb.
    Returns (W_learned, mean_weight_change).
    """
    W_learn = dict(res['W_init'])
    W_td    = res['W_td0']
    W0_coo  = W_td.tocoo()
    V_state = np.zeros(res['N'])
    emb     = res['emb']
    n       = min(n_max or len(corpus_hebb), len(corpus_hebb) - 1)

    for t in range(n):
        tok = int(corpus_hebb[t])
        if tok >= len(emb):
            continue
        I = np.zeros(res['N'])
        I[emb[tok]] = I_AMP
        V_pre   = V_state.copy()
        V_state = settle(V_state, I, res, W_td)

        for idx in range(len(W0_coo.data)):
            c_child = int(W0_coo.row[idx])
            p_mu_   = int(W0_coo.col[idx])
            eps_c   = eps(c_child)
            if eps_c >= res['N'] or p_mu_ >= res['N']:
                continue
            w      = W_learn.get((c_child, p_mu_), float(W0_coo.data[idx]))
            dV_e   = float(V_state[eps_c] - V_pre[eps_c])
            V_m    = float(V_pre[p_mu_])
            f_w    = w * (1.0 - w)
            w_init = float(W0_coo.data[idx])
            dW     = eta * V_m * dV_e * f_w - kappa * (w - w_init)
            W_learn[(c_child, p_mu_)] = min(0.99, max(1e-3, w + dW))

    delta = float(_np.mean([
        abs(W_learn.get(k, v) - v)
        for k, v in res['W_init'].items()
    ]))
    return W_learn, delta


# ----------------------------------------------------------------
# Main program
# ----------------------------------------------------------------

def main():
    t0 = time.time()

    # corpus
    print("Corpus (Markov-3) ...")
    seeds = ['CONCEPT.md', 'fractal_llm.md', 'STAGE2_FRACTALITY.md']
    corpus, V, i2c = make_markov_corpus(seeds, n=N_CORPUS)

    n_hebb  = int(EVAL_SPLIT * 0.8 * len(corpus))
    n_tr    = int(0.8 * len(corpus))
    c_hebb  = corpus[:n_hebb]
    c_ridtr = corpus[n_hebb:n_tr]
    c_ridte = corpus[n_tr:]
    print(f"  V={V} | Hebb-train={len(c_hebb)} | "
          f"Ridge-train={len(c_ridtr)} | Ridge-test={len(c_ridte)}")

    # topology
    print("\nTopology ...")
    web = generate_cosmic_web(NET)
    res = build_reservoir(web, V)
    n_leaves = len([c for c in range(web.n_clusters)
                    if web.G.nodes[mu(c)]['level'] == int(web.levels.max())])
    print(f"  {web.n_clusters} clusters | {web.N} nodes | "
          f"leaf={n_leaves} | feature dim={3*web.n_clusters*DELAY_K}")

    chance = 1.0 / V

    # baseline: W_init
    print("\nBaseline (W_init, 24 cores) ...")
    t1 = time.time()
    X_tr0, y_tr0 = parallel_collect(c_ridtr, res)
    X_te0, y_te0 = parallel_collect(c_ridte, res)
    base_tr, base_te = fit_ridge(X_tr0, y_tr0, X_te0, y_te0, V)
    print(f"  {time.time()-t1:.0f}s | Train:{base_tr:.1%}  Test:{base_te:.1%}")

    results = [dict(name="Baseline (W_init)",
                    delta=0.0, tr=base_tr, te=base_te, gain=0.0)]

    # Hebbian variants
    for var in VARIANTS:
        name, eta, kappa = var['name'], var['eta'], var['kappa']
        print(f"\nHebbian {name}  (eta={eta}, kappa={kappa}, "
              f"{N_HEBB} steps) ...")

        t2 = time.time()
        W_learned, delta = hebbian_train(c_hebb, res, eta, kappa, n_max=N_HEBB)
        print(f"  Training: {time.time()-t2:.0f}s | "
              f"mean weight change: {delta:.5f}")

        t3 = time.time()
        X_tr_h, y_tr_h = parallel_collect(c_ridtr, res, W_learn=W_learned)
        X_te_h, y_te_h = parallel_collect(c_ridte, res, W_learn=W_learned)
        h_tr, h_te = fit_ridge(X_tr_h, y_tr_h, X_te_h, y_te_h, V)
        gain = (h_te - base_te) * 100
        print(f"  Eval: {time.time()-t3:.0f}s | "
              f"Train:{h_tr:.1%}  Test:{h_te:.1%}  "
              f"(gain: {gain:+.2f} PP)")

        results.append(dict(name=f"Hebbian {name}",
                            delta=delta, tr=h_tr, te=h_te, gain=gain))

    # result
    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"STAGE 5b - ADR-8 VALIDATION  (V={V}, {dt:.0f}s)")
    print(f"{'='*60}")
    print(f"  {'Method':<28}  {'Test':>7}  {'Gain':>9}  {'|dW|':>8}")
    print(f"  {'-'*55}")
    for r in results:
        bar = '*' * min(int(max(r['gain'], 0) * 3), 15)
        print(f"  {r['name']:<28}  {r['te']:>7.1%}  "
              f"{r['gain']:>+8.2f}PP  {r['delta']:>8.5f}  {bar}")
    print(f"  {'-'*55}")
    print(f"  Chance: {chance:.1%}")

    best = max(results, key=lambda r: r['te'])
    gain_best = best['gain']
    print(f"\n  Best variant: {best['name']}  ({best['te']:.1%})")

    if gain_best > 1.0:
        print(f"  -> ADR-8 CONFIRMED: Hebbian improves "
              f"reservoir by {gain_best:+.1f} PP!")
    elif gain_best > 0.2:
        print(f"  -> Weak effect ({gain_best:+.1f} PP). "
              f"Test more steps or larger eta.")
    else:
        print(f"  -> No measurable effect. Findings:")
        print(f"     - Reservoir with W_init already near optimal")
        print(f"     - Or: Hebbian direction not aligned with global gradient")
        print(f"     - Or: |dW| too small due to Joglekar window + spring")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
