#!/usr/bin/env python3
"""
Goal 2 / Strand 1 -- REDAC mapping feasibility
===============================================
Maps the fractal topology onto the REAL REDAC resources:
  per REDAC: 6 clusters, 432 multipliers, 864 integrators,
             1728 summation paths, 3456 scaling elements, 124416 switches.

Mapping logic (what -> which resource):
  - each node (mu/eps/iota) has dynamics dV/dt          -> 1 integrator
  - each node sums Kirchhoff currents                   -> 1 summation path
  - each edge (weight W_ij)                             -> 1 scaling element
  - each iota node computes V_eps^2                     -> 1 multiplier
  (f=tanh, J: digital per ADR-4 -> no analog multipliers)

Delivers:
  1. Resource budget: which resource is the bottleneck, how many REDACs cascaded
  2. Balanced tiling (each tile <= REDAC cluster capacity) + cut cost
     vs. unbalanced KMeans optimum -> price of balance
  3. Inter-REDAC/inter-cluster link budget (= photonic interconnect demand, B-2)
"""

try:
    import cupy as np
    _GPU = True
except ImportError:
    import numpy as np
    _GPU = False

import numpy as _np
from sklearn.cluster import KMeans
from cosmic_web_generator import CosmicWebConfig, generate_cosmic_web, mu, eps, iota


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

# REDAC specification (per unit)
REDAC = dict(clusters=6, multipliers=432, integrators=864,
             summation=1728, scaling=3456, switches=124416)


def cfg(L, eta=4):
    return CosmicWebConfig(n_levels=L, eta=eta, n_top=6,
        tau_leaf=1.0, tau_top=40.0, G_min=0.1, G_iota=0.1,
        W_eps_iota=0.5, W_iota_mu=0.3, W_intra=1.0, k_neighbors=3, seed=42)


def resource_needs(web):
    N = web.N                       # integrators + summation paths
    n_edges = web.n_filaments       # scaling elements
    n_iota = (np.asarray(web.kinds) == 2).sum()
    return dict(integrators=N, summation=N, scaling=n_edges,
                multipliers=int(n_iota))


def redacs_needed(needs):
    return {r: int(np.ceil(needs[r] / REDAC[r])) for r in needs}


def balanced_partition(pos, k):
    """Capacity-constrained k-means: each tile <= ceil(N/k) nodes.
    Greedy assignment by proximity, overflow to nearest free center."""
    N = len(pos)
    cap = int(np.ceil(N / k))
    pos_cpu = _cpu(pos)
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pos_cpu)
    cent = np.asarray(km.cluster_centers_)
    dist = np.linalg.norm(pos[:, None, :] - cent[None, :, :], axis=2)  # [N,k]
    order = np.argsort(dist.min(axis=1))  # nodes with clearest preference first
    labels = -_np.ones(N, dtype=int)
    counts = _np.zeros(k, dtype=int)
    for n in _cpu(order).tolist():
        for c in _cpu(np.argsort(dist[n])).tolist():
            if counts[c] < cap:
                labels[n] = c; counts[c] += 1; break
    return labels, counts


def cut_fraction(edges, labels):
    cuts = sum(1 for i, j in edges if labels[i] != labels[j])
    return cuts / max(len(edges), 1), cuts


def analyse(L):
    web = generate_cosmic_web(cfg(L))
    pos = np.asarray(web.positions)
    edges = [(a, b) for a, b, _ in web.G.edges(data=True)]
    needs = resource_needs(web)
    nreq = redacs_needed(needs)
    n_redac = max(nreq.values())
    bottleneck = max(nreq, key=lambda r: needs[r] / REDAC[r])

    print(f"\n{'#'*70}")
    print(f"# CONFIG n_levels={L}: {web.n_clusters} clusters, {web.N} nodes, "
          f"{web.n_filaments} edges")
    print(f"{'#'*70}")
    print(f"  Resource demand vs. 1 REDAC:")
    for r in ['integrators', 'summation', 'scaling', 'multipliers']:
        frac = needs[r] / REDAC[r]
        print(f"    {r:<12}: {needs[r]:>5} / {REDAC[r]:>5}  "
              f"= {frac:>5.2f} REDAC  -> {nreq[r]} REDAC(s)")
    print(f"  -> Bottleneck: {bottleneck}  |  Needed: {n_redac} REDAC(s) cascaded")

    # Tiling: k = 6 clusters per REDAC * n_redac
    k = REDAC['clusters'] * n_redac
    # unbalanced (KMeans, locally optimal)
    lab_u = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(_cpu(pos))
    cut_u, ncut_u = cut_fraction(edges, lab_u)
    # balanced (capacity cap)
    lab_b, counts = balanced_partition(pos, k)
    cut_b, ncut_b = cut_fraction(edges, lab_b)

    print(f"\n  Tiling onto {k} REDAC clusters ({n_redac} REDAC x 6):")
    print(f"    unbalanced (KMeans): cut {cut_u:>5.1%} "
          f"({ncut_u} links), tile sizes {sorted(np.bincount(lab_u,minlength=k).tolist())}")
    print(f"    balanced  (Cap={int(np.ceil(web.N/k))}): cut {cut_b:>5.1%} "
          f"({ncut_b} links), tile sizes {sorted(counts.tolist())}")
    print(f"    Price of balance: +{(cut_b-cut_u)*100:.1f} PP cut")
    print(f"    -> {ncut_b} inter-cluster links (photonic interconnect demand, B-2)")
    return dict(L=L, N=web.N, n_redac=n_redac, bottleneck=bottleneck,
                cut_u=cut_u, cut_b=cut_b, links_b=ncut_b)


def main():
    print("REDAC mapping feasibility")
    print(f"REDAC spec/unit: {REDAC}")
    rows = [analyse(L) for L in [3, 4]]

    print(f"\n{'='*70}")
    print(f"CONCLUSION (Goal 2 / Strand 1)")
    print(f"{'='*70}")
    for r in rows:
        fit = "fits" if r['n_redac'] <= 2 else f"{r['n_redac']} REDACs"
        print(f"  n_levels={r['L']} ({r['N']} nodes): {r['n_redac']} REDAC(s) "
              f"[{fit}], bottleneck={r['bottleneck']}, "
              f"balanced {r['cut_b']:.1%} cut = {r['links_b']} inter-links")
    big = rows[-1]
    print(f"\n  The full n_levels=4 configuration ({big['N']} nodes) "
          f"fits into {big['n_redac']} cascaded REDACs.")
    print(f"  Bottleneck is '{big['bottleneck']}'. Balanced tiling costs")
    print(f"  {(big['cut_b']-big['cut_u'])*100:.1f} PP more cut than the optimum,")
    print(f"  but remains well manageable at {big['cut_b']:.1%} ({big['links_b']} links).")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
