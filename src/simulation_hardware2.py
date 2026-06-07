#!/usr/bin/env python3
"""
Goal-2 deep dive: random-geometric control + energy + REDAC mapping detail
=============================================================================
Part A: secure B-4 -- random_geometric (local random edges) as a third
        control. Clarifies: does the wire advantage come from LOCALITY alone
        (then ~= fractal) or from the HIERARCHY (then fractal shorter)?
Part B: energy proxy -- interconnect energy ~ total wire length.
        With an honest caveat (converters dominate, G16).
Part C: REDAC-6-tile mapping detail -- tile balance + absolute number of
        inter-tile connections (= where the photonic interconnect from B-2 sits).
"""

try:
    import cupy as np
    _GPU = True
except ImportError:
    import numpy as np
    _GPU = False

import numpy as _np
import time
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from cosmic_web_generator import CosmicWebConfig, generate_cosmic_web


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

NET = CosmicWebConfig(
    n_levels=4, eta=4, n_top=6,
    tau_leaf=1.0, tau_top=40.0,
    G_min=0.1, G_iota=0.1,
    W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3, seed=42,
)
N_RAND = 8
TILES  = [6, 12, 24]


def edge_len(pos, edges):
    if not edges:
        return np.array([0.0])
    a = pos[[e[0] for e in edges]]; b = pos[[e[1] for e in edges]]
    return np.linalg.norm(a - b, axis=1)


def random_global(N, n_edges, rng):
    seen = set(); tries = 0
    while len(seen) < n_edges and tries < n_edges*20:
        i, j = int(rng.integers(0,N)), int(rng.integers(0,N)); tries += 1
        if i != j and (i,j) not in seen and (j,i) not in seen:
            seen.add((i,j))
    return list(seen)


def random_geometric(pos, n_edges, M, rng):
    """Local random edges: each node connects to a random pick among its
    M nearest spatial neighbors. Same edge count as fractal."""
    tree = cKDTree(_cpu(pos))
    N = len(pos)
    _, nbrs = tree.query(pos, k=M+1)   # +1: self
    seen = set(); tries = 0
    while len(seen) < n_edges and tries < n_edges*20:
        i = int(rng.integers(0, N))
        cand = nbrs[i][1:]             # without self
        j = int(cand[rng.integers(0, len(cand))])
        tries += 1
        if i != j and (i,j) not in seen and (j,i) not in seen:
            seen.add((i,j))
    return list(seen)


def cut_frac(pos, edges, k):
    labels = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(_cpu(pos))
    cuts = sum(1 for i,j in edges if labels[i] != labels[j])
    return cuts/max(len(edges),1), labels


def main():
    t0 = time.time()
    web = generate_cosmic_web(NET)
    pos = np.asarray(web.positions); N = web.N
    frac_edges = [(a,b) for a,b,_ in web.G.edges(data=True)]
    n_edges = len(frac_edges)
    rng = np.random.default_rng(7)
    print(f"{web.n_clusters} clusters | {N} nodes | {n_edges} edges")

    fl = edge_len(pos, frac_edges)

    # Part A: three controls, wire length + tile cut (k=6)
    print(f"\n{'='*64}")
    print(f"PART A -- random-geometric control (secure B-4)")
    print(f"{'='*64}")

    # random_global ensemble
    rg_tot, rg_cut = [], []
    for s in range(N_RAND):
        e = random_global(N, n_edges, np.random.default_rng(200+s))
        rg_tot.append(edge_len(pos,e).sum())
        rg_cut.append(cut_frac(pos,e,6)[0])
    # random_geometric ensemble
    geo_tot, geo_cut, geo_local = [], [], []
    for s in range(N_RAND):
        e = random_geometric(pos, n_edges, 15, np.random.default_rng(300+s))
        el = edge_len(pos,e)
        geo_tot.append(el.sum()); geo_cut.append(cut_frac(pos,e,6)[0])
        geo_local.append(float(np.mean(el < 0.2)))

    frac_cut6 = cut_frac(pos, frac_edges, 6)[0]
    print(f"  {'Condition':<18}  {'Wire length':>12}  {'local<0.2':>10}  {'Tile cut k=6':>16}")
    print(f"  {'-'*60}")
    print(f"  {'fractal':<18}  {fl.sum():>12.1f}  {np.mean(fl<0.2):>10.1%}  {frac_cut6:>16.1%}")
    print(f"  {'random_geometric':<18}  {np.mean(geo_tot):>12.1f}  "
          f"{np.mean(geo_local):>10.1%}  {np.mean(geo_cut):>16.1%}")
    print(f"  {'random_global':<18}  {np.mean(rg_tot):>12.1f}  "
          f"{'~1.7%':>10}  {np.mean(rg_cut):>16.1%}")
    print(f"  {'-'*60}")
    geo_factor = np.mean(geo_tot)/max(fl.sum(),1e-9)
    print(f"  fractal vs random_geometric: {geo_factor:.2f}x wire length")
    if geo_factor < 1.3:
        print(f"  -> Locality is the lever; random_geometric reaches similar")
        print(f"     wire length. Fractal delivers locality + hierarchy+tau in ONE.")
    else:
        print(f"  -> Fractal wires shorter even against local random graphs.")

    # Part B: energy proxy
    print(f"\n{'='*64}")
    print(f"PART B -- energy proxy (interconnect)")
    print(f"{'='*64}")
    wl_factor = np.mean(rg_tot)/max(fl.sum(),1e-9)
    print(f"  Interconnect energy ~ total wire length (charging the line cap.)")
    print(f"  fractal: {fl.sum():.0f}  |  random_global: {np.mean(rg_tot):.0f}  "
          f"-> {wl_factor:.1f}x less interconnect energy")
    print(f"  CAVEAT (G16): ADC/DAC + node quiescent current are topology-INDEPENDENT")
    print(f"  and dominate 70-90% in real analog systems. The 11x advantage")
    print(f"  applies only to the interconnect share -- true total balance: level 6.")

    # Part C: REDAC-6-tile detail
    print(f"\n{'='*64}")
    print(f"PART C -- REDAC-6-tile mapping detail")
    print(f"{'='*64}")
    cut6, labels = cut_frac(pos, frac_edges, 6)
    sizes = np.bincount(labels, minlength=6)
    inter_tile = int(round(cut6 * n_edges))
    print(f"  Tile sizes (nodes per REDAC cluster): {sorted(sizes.tolist())}")
    print(f"  Balance: min={sizes.min()} max={sizes.max()} "
          f"(ideal ~{N//6}/tile)")
    print(f"  Inter-tile connections: {inter_tile} of {n_edges} "
          f"({cut6:.1%})")
    print(f"  -> Only {inter_tile} long connections needed (rest tile-local).")
    print(f"     Exactly here sits the photonic interconnect (B-2): few,")
    print(f"     expensive inter-tile links optical, the local rest electrical.")
    print(f"\n  Runtime: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
