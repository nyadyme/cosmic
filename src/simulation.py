#!/usr/bin/env python3
"""
Stage 4 - Software simulation of the fractal PC network (v2)
===========================================================
Simplifications for clear debugging:
  - Euler integrator (fast, sufficient for learning test)
  - G_min=0.1 (stronger signals, measurable learning)
  - G_eff = G_min (iota dynamics deactivated for now)
  - Free-Run decode (PC-compliant)
  - PC correlation: error improvement after update

Specification: CONCEPT.md questions 30-36, ADR-1 to ADR-12
Toy problem: next-symbol prediction, V={A,B,C}, synthetic grammar
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
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

from cosmic_web_generator import (
    CosmicWebConfig, CosmicWebGraph, generate_cosmic_web,
    mu, eps, iota,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    net: CosmicWebConfig = field(default_factory=lambda: CosmicWebConfig(
        n_levels=3, eta=3, n_top=3,        # 39 clusters, 117 nodes
        tau_leaf=1.0, tau_top=20.0,
        G_min=0.1,                          # larger signals
        G_iota=0.1,
        W_eps_iota=0.5, W_iota_mu=0.3,
        W_intra=1.0,
        seed=42,
    ))

    # Euler integrator  (dt < 2/lambda_max ~0.038 for stability)
    dt: float              = 0.005   # time step (safely below stability limit)
    n_settle: int          = 600     # steps clamp phase (3*tau_leaf / dt)
    n_free: int            = 200     # steps Free-Run   (1*tau_leaf / dt)

    # Spectral init (question 19/G5)
    rho_target: float      = 0.9

    # Learning (ADR-8, ADR-9)
    eta_hebb: float        = 0.05    # Hebbian learning rate
    kappa_top: float       = 0.001   # spring stiffness core
    kappa_leaf: float      = 0.0001  # spring stiffness leaf

    # Toy problem
    vocab_size: int        = 3       # A=0, B=1, C=2
    seq_len: int           = 5000
    pattern_switch_mu: int = 60
    eval_every: int        = 500
    n_eval: int            = 100

    # Injection
    I_amplitude: float     = 1.0     # strength of token injection


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def build_C_inv(web: CosmicWebGraph) -> np.ndarray:
    """C_i = tau_i * G_prec_i  (question 30/G31)."""
    g_prec = np.array([web.G.nodes[i]['G_prec'] for i in range(web.N)])
    C = np.maximum(np.asarray(web.tau) * g_prec, 1e-10)
    return 1.0 / C


def spectral_init(web: CosmicWebGraph, cfg: SimConfig) -> dict:
    """W_init = (rho_target/rho0)*W(r) on vertical edges (question 19/G5)."""
    vert = {(i, j): d['W']
            for i, j, d in web.G.edges(data=True)
            if d.get('etype') == 'vertical'}
    if not vert:
        return {}
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
    scale = cfg.rho_target / rho0
    return {k: float(np.clip(v * scale, 1e-3, 0.99)) for k, v in vert.items()}


def leaf_clusters(web: CosmicWebGraph) -> list[int]:
    max_lv = int(web.levels.max())
    return [c for c in range(web.n_clusters)
            if web.G.nodes[mu(c)]['level'] == max_lv]


def build_embedding(web: CosmicWebGraph, vocab: int) -> np.ndarray:
    lv = leaf_clusters(web)
    assert len(lv) >= vocab
    return np.array([eps(lv[i]) for i in range(vocab)], dtype=int)


def build_C_inv_diag(web: CosmicWebGraph) -> sp.diags:
    return sp.diags(build_C_inv(web))


def precompute_parent_mu(web: CosmicWebGraph) -> list[list[int]]:
    parents = []
    for c in range(web.n_clusters):
        plist = [nbr for nbr, ed in web.G[eps(c)].items()
                 if ed.get('etype') == 'vertical' and web.kinds[nbr] == 0]
        parents.append(plist)
    return parents


# ─────────────────────────────────────────────────────────────────────────────
# ODE - Euler step
# ─────────────────────────────────────────────────────────────────────────────

def compute_I_pred(V: np.ndarray,
                   web: CosmicWebGraph,
                   W_learn: dict,
                   parent_mu: list[list[int]],
                   G_eff_val: float) -> np.ndarray:
    """
    Nonlinear injection currents (question 31/G33, simplified without Y_lm).
    G_eff = constant (iota dynamics stage 5).
    """
    I = np.zeros(web.N)
    for c in range(web.n_clusters):
        ps = 0.0
        for p in parent_mu[c]:
            key = (p, eps(c))
            w = W_learn.get(key, W_learn.get((eps(c), p),
                    web.G[p][eps(c)]['W'] if web.G.has_edge(p, eps(c)) else 0.01))
            ps += w * V[p]
        f_c = float(np.tanh(ps))
        I[mu(c)]  += G_eff_val * f_c
        I[eps(c)] += G_eff_val * f_c
        # iota: deactivated for stage 4 (basic test)
    return I


def euler_settle(V0: np.ndarray,
                 I_inject: np.ndarray,
                 n_steps: int,
                 web: CosmicWebGraph,
                 W_learn: dict,
                 C_inv: np.ndarray,
                 parent_mu: list[list[int]],
                 G_eff_val: float,
                 dt: float) -> np.ndarray:
    """Euler integration for n_steps steps."""
    V = V0.copy()
    Y = sp.csr_matrix(web.Y.tocsr())
    for _ in range(n_steps):
        I = compute_I_pred(V, web, W_learn, parent_mu, G_eff_val) + I_inject
        dV = C_inv * (-Y.dot(V) + I)
        V = np.clip(V + dt * dV, -2.0, 2.0)
    return V


def run_token(V0: np.ndarray,
              token: int,
              emb: np.ndarray,
              web: CosmicWebGraph,
              W_learn: dict,
              C_inv: np.ndarray,
              parent_mu: list[list[int]],
              cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Clamp phase + Free-Run for one token.
    Returns (V_clamped, V_free).
    """
    G_eff_val = web.G.nodes[mu(0)]['G_prec']   # = G_min

    # Clamp phase: inject token
    I_inject = np.zeros(web.N)
    if token >= 0:
        I_inject[emb[token]] = cfg.I_amplitude
    V_clamped = euler_settle(V0, I_inject, cfg.n_settle,
                             web, W_learn, C_inv, parent_mu, G_eff_val, cfg.dt)

    # Free-Run: injection removed, top-down prediction visible
    V_free = euler_settle(V_clamped, np.zeros(web.N), cfg.n_free,
                          web, W_learn, C_inv, parent_mu, G_eff_val, cfg.dt)
    return V_clamped, V_free


# ─────────────────────────────────────────────────────────────────────────────
# Decode
# ─────────────────────────────────────────────────────────────────────────────

def decode_greedy(V_free: np.ndarray, emb: np.ndarray) -> int:
    """
    Read mu nodes of the embedding clusters from the Free-Run.
    mu(c) = eps(c) - 1 = emb[i] - 1.
    """
    scores = np.array([V_free[int(n) - 1] for n in emb])
    return int(np.argmax(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Learning (ADR-8, ADR-9)
# ─────────────────────────────────────────────────────────────────────────────

def _kappa(c: int, web: CosmicWebGraph, cfg: SimConfig) -> float:
    lv = web.G.nodes[mu(c)]['level']
    max_lv = int(web.levels.max())
    t = (lv - 1) / max(max_lv - 1, 1)
    return cfg.kappa_top * (1 - t) + cfg.kappa_leaf * t


def update_weights(V_pre: np.ndarray,
                   V_post: np.ndarray,
                   web: CosmicWebGraph,
                   W_learn: dict,
                   W_init: dict,
                   cfg: SimConfig) -> dict:
    """
    Hebbian + spring term (ADR-8, ADR-9).
    dW = eta * V_mu_parent * V_eps_child * f_w(W)  -  kappa*(W-W_init)
    """
    W_new = {}
    for (i, j), w in W_learn.items():
        if web.kinds[i] == 0 and web.kinds[j] == 1:
            c      = web.G.nodes[j]['cluster']
            V_mu_p = float(V_pre[i])
            V_ep_c = float(V_post[j])
            f_w    = w * (1.0 - w)
            dW_h   = cfg.eta_hebb * V_mu_p * V_ep_c * f_w
            w_i    = W_init.get((i, j), W_init.get((j, i), w))
            dW_s   = -_kappa(c, web, cfg) * (w - w_i)
            w_new  = float(np.clip(w + dW_h + dW_s, 1e-3, 0.99))
        else:
            w_new = w
        W_new[(i, j)] = w_new
    return W_new


# ─────────────────────────────────────────────────────────────────────────────
# Toy problem
# ─────────────────────────────────────────────────────────────────────────────

def generate_corpus(n: int, rng: np.random.Generator,
                    switch_mu: int = 60) -> np.ndarray:
    """Switches between ABCABC (p=3) and AABBAABB (p=4)."""
    patterns = [[0, 1, 2], [0, 0, 1, 1]]
    seq, pat, pos, until = [], 0, 0, int(rng.poisson(switch_mu))
    while len(seq) < n:
        seq.append(patterns[pat][pos % len(patterns[pat])])
        pos += 1
        until -= 1
        if until <= 0:
            pat = 1 - pat; pos = 0
            until = max(1, int(rng.poisson(switch_mu)))
    return np.array(seq[:n], dtype=np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(V0: np.ndarray, corpus: np.ndarray,
             emb: np.ndarray, web: CosmicWebGraph,
             W_learn: dict, C_inv: np.ndarray,
             parent_mu: list[list[int]], cfg: SimConfig) -> dict:
    V, correct = V0.copy(), 0
    n = min(cfg.n_eval, len(corpus) - 1)
    for i in range(n):
        _, V_free = run_token(V, int(corpus[i]), emb, web,
                              W_learn, C_inv, parent_mu, cfg)
        V, _ = run_token(V, int(corpus[i]), emb, web,
                         W_learn, C_inv, parent_mu, cfg)
        pred = decode_greedy(V_free, emb)
        if pred == int(corpus[i + 1]):
            correct += 1
    return {
        'accuracy': correct / n,
        'V_max'   : float(np.max(np.abs(V))),
        'V_stable': float(np.max(np.abs(V))) <= 2.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: SimConfig) -> dict:
    rng = _np.random.default_rng(cfg.net.seed)

    print("Generating topology ...")
    web = generate_cosmic_web(cfg.net)
    print(f"  {web.n_clusters} clusters | {web.N} nodes | "
          f"{web.n_filaments} filaments")

    C_inv     = build_C_inv(web)
    W_init    = spectral_init(web, cfg)
    W_learn   = dict(W_init)
    emb       = build_embedding(web, cfg.vocab_size)
    parent_mu = precompute_parent_mu(web)

    rho0_info = f"edges learnable: {len(W_init)}"
    print(f"  token-eps-nodes: {emb}  |  {rho0_info}")
    print(f"  tau range: {web.tau.min():.1f} .. {web.tau.max():.1f}")
    print(f"  G_min={cfg.net.G_min} | eta={cfg.eta_hebb} | "
          f"dt={cfg.dt} | settle={cfg.n_settle}steps | free={cfg.n_free}steps")

    corpus = generate_corpus(cfg.seq_len, rng, cfg.pattern_switch_mu)
    sym, cnt = np.unique(corpus, return_counts=True)
    print(f"  corpus: {len(corpus)} tokens | {dict(zip(sym.tolist(), cnt.tolist()))}")

    V = np.zeros(web.N)
    err_before, err_after = [], []
    steps, accs, vmaxs, corrs = [], [], [], []

    print(f"\n{'Step':>7}  {'Accuracy':>12}  {'V_max':>6}  "
          f"{'PC-Corr':>8}  Status")
    print("-" * 55)

    for t in range(len(corpus) - 1):
        tok    = int(corpus[t])
        target = int(corpus[t + 1])

        # error BEFORE update (proxy global gradient)
        _, V_free_pre = run_token(V, tok, emb, web, W_learn,
                                  C_inv, parent_mu, cfg)
        scores_pre = np.array([V_free_pre[int(n) - 1] for n in emb])
        err_before.append(float(scores_pre[target] - scores_pre.max()))

        # clamp phase (receives learning update)
        V_pre = V.copy()
        V, _ = run_token(V, tok, emb, web, W_learn, C_inv, parent_mu, cfg)

        # learning
        W_learn = update_weights(V_pre, V, web, W_learn, W_init, cfg)

        # error AFTER update
        _, V_free_post = run_token(V, tok, emb, web, W_learn,
                                   C_inv, parent_mu, cfg)
        scores_post = np.array([V_free_post[int(n) - 1] for n in emb])
        err_after.append(float(scores_post[target] - scores_post.max()))

        # Evaluation
        if (t + 1) % cfg.eval_every == 0:
            n_w = cfg.eval_every
            eb = np.array(err_before[-n_w:])
            ea = np.array(err_after[-n_w:])
            # PC-Corr: fraction of steps where error decreased
            corr = float(np.mean(ea > eb))   # 0..1, >0.5 = improvement

            met = evaluate(V, corpus[t:], emb, web, W_learn,
                           C_inv, parent_mu, cfg)
            steps.append(t + 1)
            accs.append(met['accuracy'])
            vmaxs.append(met['V_max'])
            corrs.append(corr)

            ok = "OK" if (met['accuracy'] > .60
                          and met['V_stable']
                          and corr > .5) else "."
            print(f"{t+1:>7}  {met['accuracy']:>12.1%}  "
                  f"{met['V_max']:>6.3f}  {corr:>8.3f}  {ok}")

    return {'steps': steps, 'accs': accs,
            'vmaxs': vmaxs, 'corrs': corrs}


# ─────────────────────────────────────────────────────────────────────────────
# Report & Plot
# ─────────────────────────────────────────────────────────────────────────────

def report_gate(steps, accs, vmaxs, corrs) -> bool:
    print("\n" + "=" * 52)
    print("STAGE 4 - GATE (question 36 / G36)")
    print("=" * 52)
    if not steps:
        print("  No measurement points.")
        return False
    acc, vmax, corr = accs[-1], vmaxs[-1], corrs[-1]
    g_acc    = acc  > 0.60
    g_stable = vmax <= 2.00
    g_corr   = corr > 0.50
    print(f"  Accuracy      : {acc:.1%}   {'OK' if g_acc else 'FAIL'}  (>60%)")
    print(f"  V-stability   : {vmax:.3f}  {'OK' if g_stable else 'FAIL'}  (<=2.0)")
    print(f"  PC-impr.rate  : {corr:.3f}  {'OK' if g_corr else 'FAIL'}  (>0.5)")
    passed = g_acc and g_stable and g_corr
    print(f"\n  -> {'PASSED' if passed else 'Keep tuning'}")
    return passed


def plot_results(steps, accs, vmaxs, corrs) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, [a * 100 for a in accs], 'b-o', ms=4)
    axes[0].axhline(60, color='g', ls='--', label='Gate 60%')
    axes[0].axhline(33, color='r', ls='--', label='Chance 33%')
    axes[0].set(xlabel='Token', ylabel='Accuracy %',
                title='Next-Symbol Accuracy (A,B,C)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, vmaxs, 'r-o', ms=4)
    axes[1].axhline(2.0, color='r', ls='--', label='Limit 2.0')
    axes[1].set(xlabel='Token', ylabel='|V|_max',
                title='Voltage Stability')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(steps, [c * 100 for c in corrs], 'g-o', ms=4)
    axes[2].axhline(50, color='g', ls='--', label='Gate 50%')
    axes[2].set(xlabel='Token', ylabel='Improvement rate %',
                title='PC learning direction (% steps with error decrease)')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/stage4_results.png', dpi=150)
    print("Plot saved: stage4_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg     = SimConfig()
    results = train(cfg)
    report_gate(results['steps'], results['accs'],
                results['vmaxs'], results['corrs'])
    if results['steps']:
        plot_results(results['steps'], results['accs'],
                     results['vmaxs'], results['corrs'])
