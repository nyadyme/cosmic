#!/usr/bin/env python3
"""
Full PC - Predictive Coding, proper implementation
===================================================
Spec (user):
  1. Separate mu / eps updates per step (not fused V)
  2. Clamp true token at periphery -> compute eps -> propagate
  3. Settle until max|dV| < tol  (or t = 5*tau_leaf)
  4. Decode from periphery mu (not Ridge)
  5. One Hebbian step on vertical theta from settled eps

Training loop:
  for each token t:
    clamp token t at leaf-eps  -> settle  -> decode pred for t+1
    one Hebbian update on vertical W from settled eps
    carry settled V as init for token t+1

Evaluation:
  accuracy = fraction(pred == true_next_token)
  plotted as learning curve (rolling window)
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
from pathlib import Path
import sys


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

sys.path.insert(0, str(Path(__file__).parent))
from simulation_stage5 import (
    make_markov_corpus, build_reservoir, w_td_from_learn,
    NET, N_CORPUS, I_AMP,
)
from cosmic_web_generator import generate_cosmic_web, mu, eps, iota

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------

DT         = 0.01
MAX_STEPS  = 500
TOL        = 1e-4
TAU_LEAF   = NET.tau_leaf

ETA        = 0.08    # Hebbian learning rate
KAPPA      = 0.002   # spring back to W_init

N_TRAIN    = 40_000
EVAL_WIN   = 500     # rolling window for accuracy
PRINT_EVERY= 2_000

# ----------------------------------------------------------------
# Settling (Euler, early-stop, separate mu/eps tracking)
# ----------------------------------------------------------------

def settle_step(V_mu, V_eps, V_iota,
                I_clamp_eps,        # clamped injection at peripheral eps
                res, W_td_mat, dt):
    """
    One Euler step with separate mu/eps/iota tracking.
    I_clamp_eps: additional current at clamped peripheral eps node.
    Returns updated (V_mu, V_eps, V_iota, dV_max).
    """
    N_cl = res['n_cl']

    # Top-down prediction for each cluster: f_c = tanh(W_td @ V_mu_global)
    # V_mu_global: place mu voltages at their node indices
    V_full = np.zeros(res['N'])
    V_full[res['mu_idx']]   = V_mu
    V_full[res['eps_idx']]  = V_eps
    V_full[res['iota_idx']] = V_iota

    f_c = np.tanh(W_td_mat.dot(V_full))   # [n_clusters]

    # Build full I vector
    G_mu  = res['G_mu']
    G_eps = res['G_eps']
    I_full = np.zeros(res['N'])
    I_full[res['mu_idx']]  += G_mu  * f_c
    I_full[res['eps_idx']] += G_eps * f_c
    I_full                 += I_clamp_eps   # clamped injection

    # Full ODE step
    dV_full = dt * res['C_inv'] * (-res['Y'].dot(V_full) + I_full)
    V_new   = np.clip(V_full + dV_full, -2.0, 2.0)

    dV_max  = float(np.max(np.abs(dV_full)))
    return (V_new[res['mu_idx']],
            V_new[res['eps_idx']],
            V_new[res['iota_idx']],
            dV_max)


def settle(V_mu, V_eps, V_iota, I_clamp_eps, res, W_td_mat,
           max_steps=MAX_STEPS, tol=TOL, dt=DT):
    """
    Settle until max|dV| < tol or t = 5*tau_leaf.
    Returns settled (V_mu, V_eps, V_iota).
    """
    for step in range(max_steps):
        V_mu, V_eps, V_iota, dV_max = settle_step(
            V_mu, V_eps, V_iota, I_clamp_eps, res, W_td_mat, dt)
        if step > 5 and dV_max < tol:
            break
    return V_mu, V_eps, V_iota


# ----------------------------------------------------------------
# Decode from periphery mu  (not Ridge)
# ----------------------------------------------------------------

def decode_from_periphery_mu(V_mu, res, V_vocab):
    """
    Decode next-token prediction from leaf mu node voltages.
    emb[i] = eps index of token i's leaf cluster.
    mu of same cluster = emb[i] - 1  (since mu(c)=3c, eps(c)=3c+1).
    Returns (pred_token, scores).
    """
    leaf_mu_idx = res['emb'] - 1   # mu indices of embedding clusters
    scores = V_mu[res['mu_idx_inv'][leaf_mu_idx[:V_vocab]]]
    return int(np.argmax(scores)), scores


def build_mu_inv(res):
    """Build inverse map: global_node_idx -> position in mu_idx array."""
    inv = np.full(res['N'], -1, dtype=int)
    for pos, global_i in enumerate(res['mu_idx']):
        inv[global_i] = pos
    res['mu_idx_inv'] = inv


# ----------------------------------------------------------------
# Hebbian update on vertical theta from settled eps
# ----------------------------------------------------------------

def hebbian_step(V_mu, V_eps, W_learn, W0_coo, eta, kappa):
    """
    One Hebbian step: dtheta = eta * eps_child * J_f * V_mu_parent
    where J_f = 1 - tanh(V_mu_parent)^2  (tanh derivative).
    Spring: -kappa * (W - W_init).
    """
    for idx in range(len(W0_coo.data)):
        c_child = int(W0_coo.row[idx])
        p_pos   = int(W0_coo.col[idx])   # position in mu_idx of parent mu node

        # Get voltages
        eps_c = float(V_eps[c_child])        # error signal at child cluster
        mu_p  = float(V_mu[p_pos])           # parent mu voltage (if p_pos is valid)

        # Hebbian: dW = eta * eps_c * J_f * mu_parent
        f_p = float(np.tanh(mu_p))
        J_f = 1.0 - f_p ** 2                 # tanh Jacobian
        w       = W_learn.get((c_child, p_pos), float(W0_coo.data[idx]))
        w_init  = float(W0_coo.data[idx])
        dW      = eta * eps_c * J_f * mu_p   - kappa * (w - w_init)
        W_learn[(c_child, p_pos)] = float(np.clip(w + dW, 1e-3, 0.99))

    return W_learn


# ----------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------

def run_pc(corpus_tr, corpus_te, res, V_vocab,
           n_train=N_TRAIN, eta=ETA, kappa=KAPPA):
    """
    Full PC training loop per the spec.
    Returns (accuracy_curve, W_final).
    """
    W_learn = dict(res['W_init'])
    W_td    = w_td_from_learn(res, W_learn)
    W0_coo  = res['W_td0'].tocoo()

    # Remap col indices of W0_coo to positions in mu_idx
    mu_global_to_pos = {int(g): pos for pos, g in enumerate(res['mu_idx'])}
    # Rebuild coo with remapped col
    new_rows, new_cols, new_data = [], [], []
    for i in range(len(W0_coo.data)):
        c_child = int(W0_coo.row[i])
        g_mu    = int(W0_coo.col[i])
        pos     = mu_global_to_pos.get(g_mu, -1)
        if pos >= 0:
            new_rows.append(c_child)
            new_cols.append(pos)
            new_data.append(float(W0_coo.data[i]))
    from scipy.sparse import coo_matrix
    W0_remapped = coo_matrix(
        (new_data, (new_rows, new_cols)),
        shape=(res['n_cl'], len(res['mu_idx'])))

    # Initial state
    V_mu   = np.zeros(res['n_cl'])
    V_eps  = np.zeros(res['n_cl'])
    V_iota = np.zeros(res['n_cl'])

    # Rolling accuracy window
    recent    = []
    curve     = []      # (step, rolling_acc)
    t0        = time.time()

    for t in range(min(n_train, len(corpus_tr) - 1)):
        tok     = int(corpus_tr[t]) % V_vocab
        tok_nxt = int(corpus_tr[t+1]) % V_vocab

        # Build clamped injection at peripheral eps of true token
        I_clamp = np.zeros(res['N'])
        I_clamp[res['emb'][tok]] = I_AMP

        # 2. Clamp -> settle
        V_mu, V_eps, V_iota = settle(
            V_mu, V_eps, V_iota, I_clamp, res, W_td)

        # 3. Decode from periphery mu
        leaf_mu_pos = res['mu_idx_inv'][res['emb'][:V_vocab] - 1]
        valid       = leaf_mu_pos >= 0
        scores      = np.full(V_vocab, -99.0)
        scores[valid] = V_mu[leaf_mu_pos[valid]]
        pred = int(np.argmax(scores))

        recent.append(int(pred == tok_nxt))

        # 5. One Hebbian step on vertical theta from settled eps
        W_learn = hebbian_step(V_mu, V_eps, W_learn, W0_remapped, eta, kappa)

        # Update W_td every PRINT_EVERY steps (cheap rebuild)
        if (t + 1) % PRINT_EVERY == 0:
            W_td = w_td_from_learn(res, W_learn)
            rolling = float(np.mean(recent[-EVAL_WIN:]))
            curve.append((t + 1, rolling))
            elapsed = time.time() - t0
            print(f"  Step {t+1:>6}:  acc={rolling:.1%}  "
                  f"({elapsed:.0f}s)")

    # Final evaluation on test corpus
    correct_te = 0
    V_mu_e = np.zeros(res['n_cl'])
    V_eps_e = np.zeros(res['n_cl'])
    V_io_e  = np.zeros(res['n_cl'])
    n_te = min(2000, len(corpus_te) - 1)
    W_td_final = w_td_from_learn(res, W_learn)

    for t in range(n_te):
        tok     = int(corpus_te[t])  % V_vocab
        tok_nxt = int(corpus_te[t+1]) % V_vocab
        I_clamp = np.zeros(res['N'])
        I_clamp[res['emb'][tok]] = I_AMP
        V_mu_e, V_eps_e, V_io_e = settle(
            V_mu_e, V_eps_e, V_io_e, I_clamp, res, W_td_final)
        leaf_mu_pos = res['mu_idx_inv'][res['emb'][:V_vocab] - 1]
        scores      = np.full(V_vocab, -99.0)
        valid       = leaf_mu_pos >= 0
        scores[valid] = V_mu_e[leaf_mu_pos[valid]]
        correct_te += (int(np.argmax(scores)) == tok_nxt)

    acc_te = correct_te / n_te
    return curve, acc_te, W_learn


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    t0 = time.time()

    print("Corpus (Markov-3) ...")
    seeds = ['CONCEPT.md', 'fractal_llm.md', 'STAGE2_FRACTALITY.md']
    corpus, V, _ = make_markov_corpus(seeds, n=N_CORPUS)
    n_tr = int(0.8 * len(corpus))
    corpus_tr, corpus_te = corpus[:n_tr], corpus[n_tr:]
    print(f"  V={V} | {n_tr} train characters")

    print("\nTopology ...")
    web = generate_cosmic_web(NET)
    res = build_reservoir(web, V)
    build_mu_inv(res)
    print(f"  {web.n_clusters} clusters | {web.N} nodes")

    chance = 1.0 / V

    print(f"\nFull PC Training  (eta={ETA}, kappa={KAPPA}, "
          f"{N_TRAIN} steps) ...")
    print(f"  Decode: periphery mu  |  Learn: Hebbian on vertical theta")
    print(f"  {'Step':>8}  {'Roll-Acc':>10}  {'Time':>6}")
    print(f"  {'-'*28}")

    curve, acc_te, W_final = run_pc(
        corpus_tr, corpus_te, res, V)

    # Summary
    dt = time.time() - t0
    acc_start = curve[0][1]  if curve else 0.0
    acc_end   = curve[-1][1] if curve else 0.0
    gain      = (acc_end - acc_start) * 100

    print(f"\n{'='*56}")
    print(f"FULL PC RESULT  (V={V}, {dt:.0f}s)")
    print(f"{'='*56}")
    print(f"  Chance:          {chance:.1%}")
    print(f"  Learning curve start: {acc_start:.1%}  (step {curve[0][0] if curve else 0})")
    print(f"  Learning curve end:  {acc_end:.1%}  (step {curve[-1][0] if curve else 0})")
    print(f"  Hebbian gain:    {gain:+.1f} PP")
    print(f"  Test accuracy:   {acc_te:.1%}  (decode from mu, no Ridge)")

    if gain > 2.0:
        print(f"\n  -> ADR-8 CONFIRMED: Full PC learns via local rule!")
    elif gain > 0.5:
        print(f"\n  -> Weak learning effect ({gain:+.1f} PP).")
        print(f"     Test more steps or larger eta.")
    else:
        print(f"\n  -> No learning effect in {N_TRAIN} steps.")
        print(f"     Larger network or longer training needed.")
    print(f"{'='*56}")


if __name__ == '__main__':
    main()
