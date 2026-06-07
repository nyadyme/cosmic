#!/usr/bin/env python3
"""
Goal 2 / Strand 2 -- Filament refinement
========================================
Tests the intuition "real 1D filaments instead of direct edges": subdivide
long inter-cluster edges into SEGMENTS with relay nodes (1D filament).
Measures the hardware effect + the node price.

Important physics first:
  - Total wire length is INVARIANT under subdivision (splitting a line into
    pieces does not change the sum). Subdivision does NOT improve wire length.
  - What changes: max. segment length (shorter) and tile cut -- one long
    crossing edge becomes a chain that is absorbed locally tile by tile.
  - Price: each relay node = +1 integrator (REDAC bottleneck!).
Question: is the partition gain worth the integrator price?
"""

try:
    import cupy as np
    _GPU = True
except ImportError:
    import numpy as np
    _GPU = False

import numpy as _np
from sklearn.cluster import KMeans
from cosmic_web_generator import CosmicWebConfig, generate_cosmic_web


def _cpu(a):
    """CuPy array → CPU NumPy; NumPy array → unchanged."""
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

REDAC_INT = 864  # integrators per REDAC (bottleneck from Strand 1)


def cfg(L=4):
    return CosmicWebConfig(n_levels=L, eta=4, n_top=6,
        tau_leaf=1.0, tau_top=40.0, G_min=0.1, G_iota=0.1,
        W_eps_iota=0.5, W_iota_mu=0.3, W_intra=1.0, k_neighbors=3, seed=42)


def subdivide(pos, edges, seg_len):
    """Subdivides each edge > seg_len into segments with relay nodes.
    Returns (new_pos, new_edges)."""
    pos = list(pos)
    new_edges = []
    for a, b in edges:
        pa, pb = np.array(pos[a]), np.array(pos[b])
        L = np.linalg.norm(pa - pb)
        if seg_len <= 0 or L <= seg_len:
            new_edges.append((a, b))
            continue
        n_seg = int(np.ceil(L / seg_len))
        prev = a
        for s in range(1, n_seg):
            p = pa + (pb - pa) * (s / n_seg)
            idx = len(pos); pos.append(p)
            new_edges.append((prev, idx)); prev = idx
        new_edges.append((prev, b))
    return np.array(pos), new_edges


def wire(pos, edges):
    if not edges:
        return 0.0, 0.0
    a = pos[[e[0] for e in edges]]; b = pos[[e[1] for e in edges]]
    L = np.linalg.norm(a - b, axis=1)
    return float(L.sum()), float(L.max())


def balanced_cut(pos, edges, k):
    N = len(pos); cap = int(np.ceil(N/k))
    pos_cpu = _cpu(pos)
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pos_cpu)
    cent = np.asarray(km.cluster_centers_)
    dist = np.linalg.norm(pos[:,None,:]-cent[None,:,:],axis=2)
    order = np.argsort(dist.min(1)); lab = -_np.ones(N,dtype=int); cnt=_np.zeros(k,int)
    for n in _cpu(order).tolist():
        for c in _cpu(np.argsort(dist[n])).tolist():
            if cnt[c]<cap: lab[n]=c; cnt[c]+=1; break
    cuts = sum(1 for i,j in edges if lab[i]!=lab[j])
    return cuts/max(len(edges),1), cuts


def analyse(seg_len, web, base_pos, base_edges):
    if seg_len <= 0:
        pos, edges, label = base_pos, base_edges, "v1 (no subdivision)"
    else:
        pos, edges = subdivide(base_pos, base_edges, seg_len)
        label = f"seg_len={seg_len}"
    tot, mx = wire(pos, edges)
    N = len(pos)
    n_redac = int(np.ceil(N / REDAC_INT))
    k = 6 * n_redac
    cut_b, links_b = balanced_cut(pos, edges, k)
    return dict(label=label, N=N, edges=len(edges), wire=tot, maxedge=mx,
                n_redac=n_redac, cut_b=cut_b, links_b=links_b, k=k)


def main():
    web = generate_cosmic_web(cfg(4))
    base_pos = np.asarray(web.positions)
    base_edges = [(a, b) for a, b, _ in web.G.edges(data=True)]
    print(f"Base: {web.N} nodes, {len(base_edges)} edges\n")

    print(f"{'Variant':<24} {'Nodes':>7} {'Edges':>7} {'Wire':>8} "
          f"{'maxEdge':>8} {'REDACs':>7} {'balCut':>7} {'Links':>6}")
    print("-"*78)
    rows = []
    for sl in [0.0, 0.2, 0.1, 0.05]:
        r = analyse(sl, web, base_pos, base_edges)
        rows.append(r)
        print(f"{r['label']:<24} {r['N']:>7} {r['edges']:>7} {r['wire']:>8.1f} "
              f"{r['maxedge']:>8.3f} {r['n_redac']:>7} {r['cut_b']:>7.1%} "
              f"{r['links_b']:>6}")
    print("-"*78)

    v1 = rows[0]
    wire_list = ", ".join(f"{r['wire']:.0f}" for r in rows)
    wire_spread = max(r['wire'] for r in rows) - min(r['wire'] for r in rows)
    print("\nANALYSIS:")
    print(f"  Wire length: [{wire_list}] -> "
          f"{'invariant (as expected)' if wire_spread < 1 else 'varies'}")
    print(f"  max. edge length drops: "
          f"{v1['maxedge']:.3f} -> {rows[-1]['maxedge']:.3f}")
    print("  Tile cut (balanced): "
          + " -> ".join(f"{r['cut_b']:.1%}" for r in rows))
    print(f"  Node price: {v1['N']} -> {rows[-1]['N']} "
          f"(+{rows[-1]['N']-v1['N']} relays = +integrators)")
    print(f"  REDAC demand: {v1['n_redac']} -> {rows[-1]['n_redac']}")

    # Verdict
    best = min(rows, key=lambda r: r['cut_b'])
    print(f"\n  CONCLUSION (Strand 2):")
    print(f"  Filament subdivision lowers the tile cut "
          f"({v1['cut_b']:.1%} -> {best['cut_b']:.1%} at {best['label']}),")
    print(f"  but does NOT change the total wire length (physics). Price: "
          f"+{best['N']-v1['N']} relay nodes -> {best['n_redac']} instead of "
          f"{v1['n_redac']} REDACs.")
    if best['cut_b'] < v1['cut_b'] - 0.02 and best['n_redac'] <= v1['n_redac']+1:
        print(f"  -> Worth it: better partition at acceptable node price.")
    elif best['cut_b'] < v1['cut_b'] - 0.02:
        print(f"  -> Tradeoff: better partition, but expensive (more REDACs needed).")
    else:
        print(f"  -> Not worth it: little partition gain for the node price.")


if __name__ == '__main__':
    main()
