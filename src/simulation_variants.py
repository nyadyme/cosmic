#!/usr/bin/env python3
"""
Goal 2 -- test two variants
===========================
A) iota removal: does dropping the third node type (ADR-12) bring
   hardware savings without compute loss?
B) Multifractal: heterogeneous branching instead of fixed eta -- does it
   change hardware/compute compared to monofractal?

Metrics: Memory Capacity (delayed-copy ESN), node/REDAC demand,
wire length, tile cut (balanced), box-counting d_H.
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


from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier
import time

from cosmic_web_generator import (
    CosmicWebConfig, generate_cosmic_web, mu, eps, iota,
    _perturb_direction, _level_radius, _level_tau_mu, _level_tau_iota,
)

V, LAGS, N_SAMP = 6, [1, 2, 3, 5, 8, 12, 16], 8000
WARMUP, RHO, GAIN, ALPHA = 250, 0.97, 0.6, 1.0
REDAC_INT, REDAC_MULT = 864, 432
chance = 1.0 / V


# ---------- ESN / Memory Capacity ----------
def sscale(W, rho):
    try:
        v = eigs(W.astype(float), k=1, which='LM', return_eigenvectors=False,
                 tol=1e-3, maxiter=500); rho0 = max(float(np.abs(v[0])), 1e-6)
    except Exception:
        rho0 = max(float(np.abs(W.data).max()), 1e-6)
    return W * (rho / rho0)


def esn(W, leak, Win, inp, N):
    X = np.zeros((len(inp), N)); x = np.zeros(N)
    for t in range(len(inp)):
        u = np.zeros(V); u[inp[t]] = GAIN
        x = (1 - leak) * x + leak * np.tanh(Win.dot(u) + W.dot(x)); X[t] = x
    return X


def racc(X, tgt):
    Xw, yw = X[WARMUP:], tgt[WARMUP:]
    Xs = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-8); n = int(0.7 * len(Xs))
    clf = RidgeClassifier(alpha=ALPHA); clf.fit(_cpu(Xs[:n]), _cpu(yw[:n]))
    return clf.score(_cpu(Xs[n:]), _cpu(yw[n:]))


def mem_cap(W, leak, Win, inp, N):
    X = esn(W, leak, Win, inp, N); mc = 0.0
    for lag in LAGS:
        t = np.zeros(len(inp), dtype=int); t[lag:] = inp[:-lag]
        mc += max(racc(X, t) - chance, 0)
    return mc / (1 - chance)


def box_dH(pos):
    p = (pos - pos.min(0)) / (np.ptp(pos, axis=0) + 1e-9)
    sizes = [2, 4, 8, 16, 32]; counts = []
    for nb in sizes:
        idx = _cpu(np.floor(p * nb).astype(int).clip(0, nb - 1))
        counts.append(len({tuple(r) for r in idx}))
    return float(_np.polyfit(_np.log(sizes), _np.log(counts), 1)[0])


def hardware(pos, edges):
    a = pos[[e[0] for e in edges]]; b = pos[[e[1] for e in edges]]
    wire = float(np.linalg.norm(a - b, axis=1).sum())
    N = len(pos); k = 6 * int(np.ceil(N / REDAC_INT)); cap = int(np.ceil(N / k))
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(_cpu(pos))
    dist = np.linalg.norm(pos[:, None, :] - np.asarray(km.cluster_centers_)[None, :, :], axis=2)
    order = _np.argsort(_cpu(dist.min(1))); lab = -_np.ones(N, int); cnt = _np.zeros(k, int)
    dist_cpu = _cpu(dist)
    for n in order:
        for c in _np.argsort(dist_cpu[n]):
            if cnt[c] < cap: lab[n] = c; cnt[c] += 1; break
    cut = sum(1 for i, j in edges if lab[i] != lab[j]) / max(len(edges), 1)
    return wire, cut, k


def Win_for(node_ids, leaves_eps, N):
    W = np.zeros((N, V))
    for s in range(V): W[node_ids[leaves_eps[s]], s] = 1.0
    return sp.csr_matrix(W)


# ============================================================
# VARIANT A: iota removal
# ============================================================
def variant_A(inp):
    print(f"\n{'='*60}\nVARIANT A -- iota removal\n{'='*60}")
    web = generate_cosmic_web(CosmicWebConfig(
        n_levels=4, eta=4, n_top=6, tau_leaf=1.0, tau_top=40.0,
        G_min=0.1, G_iota=0.1, W_eps_iota=0.5, W_iota_mu=0.3,
        W_intra=1.0, k_neighbors=3, seed=42))
    pos = np.asarray(web.positions); tau = np.asarray(web.tau); kinds = web.kinds
    edges_all = [(a, b) for a, b, _ in web.G.edges(data=True)]
    wdict = {(a, b): d['W'] for a, b, d in web.G.edges(data=True)}
    mlv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[mu(c)]['level'] == mlv]

    rows = []
    for label, keep_iota in [("with iota (mu/eps/iota)", True),
                             ("without iota (mu/eps)", False)]:
        if keep_iota:
            mask = np.ones(web.N, bool)
        else:
            mask = kinds != 2
        old2new = -np.ones(web.N, int); old2new[mask] = np.arange(mask.sum())
        Np = int(mask.sum())
        e = [(old2new[a], old2new[b]) for (a, b) in edges_all
             if mask[a] and mask[b]]
        # coupling
        r, c, d = [], [], []
        for (a, b) in edges_all:
            if mask[a] and mask[b]:
                w = wdict[(a, b)]
                r += [old2new[a], old2new[b]]; c += [old2new[b], old2new[a]]
                d += [w, w]
        W = sscale(sp.csr_matrix((d, (r, c)), shape=(Np, Np)), RHO)
        leak = np.clip(1.0 / np.asarray(tau[mask]), 1e-3, 1.0)
        # Win: eps of the leaf clusters, in new indices
        eps_ids = {c: old2new[eps(c)] for c in leaves}
        Win = np.zeros((Np, V))
        for s in range(V): Win[eps_ids[leaves[s]], s] = 1.0
        Win = sp.csr_matrix(Win)
        mc = mem_cap(W, leak, Win, inp, Np)
        wire, cut, k = hardware(pos[mask], e)
        n_mult = int((kinds[mask] == 2).sum())
        n_redac = max(int(np.ceil(Np / REDAC_INT)),
                      int(np.ceil(max(n_mult, 1) / REDAC_MULT)) if n_mult else 1)
        rows.append((label, Np, mc, wire, cut, n_mult, n_redac))

    print(f"  {'Variant':<24} {'Nodes':>7} {'MC':>6} {'Wire':>7} "
          f"{'Tile%':>6} {'Mult':>5} {'REDAC':>6}")
    print(f"  {'-'*60}")
    for la, Np, mc, w, cut, nm, nr in rows:
        print(f"  {la:<24} {Np:>7} {mc:>6.2f} {w:>7.0f} {cut:>6.1%} "
              f"{nm:>5} {nr:>6}")
    full, drop = rows[0], rows[1]
    print(f"\n  MC loss from removal: {full[2]-drop[2]:+.2f} "
          f"({(drop[2]-full[2])/full[2]*100:+.1f}%)")
    print(f"  Hardware savings: {full[1]-drop[1]} nodes "
          f"({(1-drop[1]/full[1])*100:.0f}%), {full[5]-drop[5]} multipliers,"
          f" REDAC {full[6]}->{drop[6]}")
    if abs(drop[2]-full[2]) < 0.3 and drop[1] < full[1]:
        print(f"  -> WORTH IT: ~same compute power, ~1/3 less hardware.")
    else:
        print(f"  -> Tradeoff: hardware cheaper, but mind the MC loss.")


# ============================================================
# VARIANT B: multifractal (variable branching)
# ============================================================
def gen_multifractal(seed, eta_choices, n_levels=4, n_top=6):
    cfg = CosmicWebConfig(n_levels=n_levels, eta=4, n_top=n_top,
        tau_leaf=1.0, tau_top=40.0, G_min=0.1, G_iota=0.1,
        W_eps_iota=0.5, W_iota_mu=0.3, W_intra=1.0, k_neighbors=3, seed=seed)
    rng = _np.random.default_rng(seed)
    centers, levels, parents = [], [], []

    def place(direction, level, parent):
        idx = len(centers)
        radius = _level_radius(level, cfg)
        if level == n_levels:
            radius += rng.uniform(-0.5, 0.5) * cfg.shell_thickness
        centers.append(direction * radius); levels.append(level); parents.append(parent)
        if level == n_levels: return
        eta_here = int(rng.choice(eta_choices))   # MULTIFRACTAL: variable branching
        jit = cfg.base_jitter * cfg.lambda_scale ** (level - 1)
        for _ in range(eta_here):
            place(_perturb_direction(direction, jit, rng), level + 1, idx)

    for _ in range(n_top):
        d = rng.normal(size=3); d /= _np.linalg.norm(d); place(d, 1, -1)

    ncl = len(centers); cen = np.array(centers); lev = np.array(levels)
    par = np.array(parents)
    # mu/eps/iota expansion + edges
    N = 3 * ncl
    pos = np.zeros((N, 3)); tau = np.zeros(N); kinds = np.zeros(N, int)
    for c in range(ncl):
        for k, t in [(0, _level_tau_mu(lev[c], cfg)), (1, cfg.tau_leaf),
                     (2, _level_tau_iota(lev[c], cfg))]:
            nd = 3 * c + k; pos[nd] = cen[c]; tau[nd] = t; kinds[nd] = k
    edges = []
    dH_form = np.log(np.mean([np.bincount(par[par >= 0]).mean() if (par >= 0).any() else 1])) # dummy
    for c in range(ncl):
        edges += [(3*c, 3*c+1), (3*c+1, 3*c+2), (3*c+2, 3*c)]
        if par[c] >= 0: edges.append((3*par[c], 3*c+1))
    for lv in range(1, n_levels+1):
        mem = [c for c in range(ncl) if lev[c] == lv]
        if len(mem) < 2: continue
        pts = cen[mem]; dist = np.linalg.norm(pts[:,None]-pts[None], axis=2)
        np.fill_diagonal(dist, np.inf)
        for li, c in enumerate(mem):
            for lj in np.argsort(dist[li])[:min(3, len(mem)-1)]:
                edges.append((3*c, 3*mem[lj]))
    # dedupe
    seen = set(); ue = []
    for a, b in edges:
        if a != b and (a, b) not in seen and (b, a) not in seen:
            seen.add((a, b)); ue.append((a, b))
    return pos, tau, kinds, ue, ncl, lev


def reservoir_from(pos, tau, edges, N, leaves_eps):
    r, c, d = [], [], []
    for a, b in edges:
        L = float(np.linalg.norm(pos[a]-pos[b])); w = max(L**(-1.0), 1e-4) if L > 1e-9 else 1.0
        r += [a, b]; c += [b, a]; d += [w, w]
    W = sscale(sp.csr_matrix((d, (r, c)), shape=(N, N)), RHO)
    leak = np.clip(1.0/np.asarray(tau), 1e-3, 1.0)
    Win = np.zeros((N, V))
    for s in range(V): Win[leaves_eps[s], s] = 1.0
    return W, leak, sp.csr_matrix(Win)


def variant_B(inp):
    print(f"\n{'='*60}\nVARIANT B -- multifractal vs monofractal\n{'='*60}")
    rows = []
    for label, etas in [("monofractal (eta=4)", [4]),
                        ("multifractal (eta 2..6)", [2, 3, 4, 5, 6])]:
        pos, tau, kinds, edges, ncl, lev = gen_multifractal(42, etas)
        N = len(pos)
        leaves = [c for c in range(ncl) if lev[c] == lev.max()]
        leaves_eps = {s: 3*leaves[s]+1 for s in range(V)}
        W, leak, Win = reservoir_from(pos, tau, edges, N, leaves_eps)
        mc = mem_cap(W, leak, Win, inp, N)
        wire, cut, k = hardware(pos, edges)
        dH = box_dH(pos[kinds == 0])  # only mu nodes for d_H
        rows.append((label, ncl, N, mc, wire, cut, dH))
    print(f"  {'Variant':<26} {'Cluster':>7} {'Nodes':>7} {'MC':>6} "
          f"{'Wire':>7} {'Tile%':>6} {'d_H':>5}")
    print(f"  {'-'*64}")
    for la, ncl, N, mc, w, cut, dH in rows:
        print(f"  {la:<26} {ncl:>7} {N:>7} {mc:>6.2f} {w:>7.0f} "
              f"{cut:>6.1%} {dH:>5.2f}")
    mono, multi = rows[0], rows[1]
    print(f"\n  MC: mono {mono[3]:.2f} vs multi {multi[3]:.2f} "
          f"({multi[3]-mono[3]:+.2f})")
    print(f"  Tile cut: mono {mono[5]:.1%} vs multi {multi[5]:.1%} "
          f"({(multi[5]-mono[5])*100:+.1f} PP)")
    print(f"  d_H: mono {mono[6]:.2f} vs multi {multi[6]:.2f}")
    if abs(multi[3]-mono[3]) < 0.3 and abs(multi[5]-mono[5]) < 0.03:
        print(f"  -> No significant difference: multifractality brings")
        print(f"     neither compute nor hardware advantage (as B-3 predicts).")
    elif multi[5] > mono[5] + 0.03:
        print(f"  -> Multifractal worsens tile balance (heterogeneous subtrees).")
    else:
        print(f"  -> Difference present -- see the numbers.")


def main():
    t0 = time.time()
    inp = np.asarray(_np.random.default_rng(7).integers(0, V, N_SAMP))
    print(f"Two-variant test (V={V}, {N_SAMP} samples, LAGs={LAGS})")
    variant_A(inp)
    variant_B(inp)
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
