#!/usr/bin/env python3
"""
Goal 2 -- Hardware layout ablation: is the fractal geometry
physically more wireable than a random graph?
==============================================================
B-3 showed: computationally no advantage. But analog hardware costs depend
on the SPATIAL structure (wire length -> parasitic R/C, energy, area;
tile partitionability -> crossbar mapping like REDAC).

Fair control: random_sparse uses the SAME 3D node positions as fractal,
but connects random pairs (same node/edge count). This isolates the
comparison exactly: does the fractal topology connect LOCALLY (short wires) or not?

Metrics (all from 3D positions + edge set):
  1. Wire length (Euclidean): mean / median / p90 / max / sum
  2. Locality: % edges shorter than threshold
  3. Tile partitioning (k-means on positions, k=6/12/24 like REDAC clusters):
     cut-edge fraction = fraction of edges between different tiles
     (low = maps well onto crossbar, little inter-tile wiring)
  4. random over ensemble (8 graphs) -> mean +/- spread
"""

try:
    import cupy as np
    _GPU = True
except ImportError:
    import numpy as np
    _GPU = False

import numpy as _np
import time

from cosmic_web_generator import CosmicWebConfig, generate_cosmic_web

try:
    from sklearn.cluster import KMeans
    HAS_SK = True
except ImportError:
    HAS_SK = False


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)

NET = CosmicWebConfig(
    n_levels=4, eta=4, n_top=6,    # larger for meaningful statistics
    tau_leaf=1.0, tau_top=40.0,
    G_min=0.1, G_iota=0.1,
    W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3, seed=42,
)

N_RAND   = 8
TILES    = [6, 12, 24]
LOCAL_TH = [0.05, 0.1, 0.2]


def edge_lengths(pos, edges):
    """Euclidean length of each edge."""
    if len(edges) == 0:
        return np.array([0.0])
    a = pos[[e[0] for e in edges]]
    b = pos[[e[1] for e in edges]]
    return np.linalg.norm(a - b, axis=1)


def random_edges(N, n_edges, rng):
    """n_edges random undirected pairs (no self, no duplicates)."""
    seen = set()
    tries = 0
    while len(seen) < n_edges and tries < n_edges * 20:
        i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
        tries += 1
        if i == j or (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
    return list(seen)


def cut_fraction(pos, edges, k, rng):
    """Fraction of edges between different tiles (k-means on positions)."""
    if HAS_SK:
        labels = KMeans(n_clusters=k, n_init=3,
                        random_state=0).fit_predict(_cpu(pos))
    else:
        # simple fallback: random assignment (only if sklearn is missing)
        labels = rng.integers(0, k, len(pos))
    cuts = sum(1 for i, j in edges if labels[i] != labels[j])
    return cuts / max(len(edges), 1)


def stats(lengths):
    return dict(
        mean=float(np.mean(lengths)),
        median=float(np.median(lengths)),
        p90=float(np.percentile(lengths, 90)),
        mx=float(np.max(lengths)),
        total=float(np.sum(lengths)),
    )


def main():
    t0 = time.time()
    print("Goal-2 hardware layout ablation")
    print(f"  sklearn: {HAS_SK}")

    web = generate_cosmic_web(NET)
    pos = np.asarray(web.positions)
    N = web.N
    frac_edges = [(a, b) for a, b, _ in web.G.edges(data=True)]
    n_edges = len(frac_edges)
    print(f"  {web.n_clusters} clusters | {N} nodes | {n_edges} edges")
    print(f"  Volume: radius {np.linalg.norm(pos,axis=1).min():.2f}"
          f"..{np.linalg.norm(pos,axis=1).max():.2f}")

    # Fractal
    fl = edge_lengths(pos, frac_edges)
    fs = stats(fl)

    # Random ensemble (same positions, random pairs)
    rng = np.random.default_rng(7)
    rand_stats = []
    rand_local = {th: [] for th in LOCAL_TH}
    rand_cuts  = {k: [] for k in TILES}
    frac_local = {th: float(np.mean(fl < th)) for th in LOCAL_TH}
    frac_cuts  = {k: cut_fraction(pos, frac_edges, k, rng) for k in TILES}

    for s in range(N_RAND):
        rr = np.random.default_rng(200 + s)
        re = random_edges(N, n_edges, rr)
        rl = edge_lengths(pos, re)
        rand_stats.append(stats(rl))
        for th in LOCAL_TH:
            rand_local[th].append(float(np.mean(rl < th)))
        for k in TILES:
            rand_cuts[k].append(cut_fraction(pos, re, k, rr))

    def rmean(key):
        return np.mean([r[key] for r in rand_stats])
    def rstd(key):
        return np.std([r[key] for r in rand_stats])

    # Output
    print(f"\n{'='*62}")
    print(f"WIRE LENGTH (Euclidean in 3D)  ({time.time()-t0:.0f}s)")
    print(f"{'='*62}")
    print(f"  {'Metric':<10}  {'fractal':>10}  {'random mean+/-std':>22}  {'Factor':>7}")
    print(f"  {'-'*54}")
    for key, label in [('mean','Mean'),('median','Median'),
                       ('p90','p90'),('mx','Max'),('total','Sum')]:
        rm, rs = rmean(key), rstd(key)
        factor = rm / max(fs[key], 1e-9)
        print(f"  {label:<10}  {fs[key]:>10.3f}  "
              f"{rm:>9.3f} +/- {rs:>6.3f}     {factor:>5.1f}x")
    print(f"  {'-'*54}")
    print(f"  Factor = random / fractal (>1 = fractal shorter wires)")

    print(f"\n{'='*62}")
    print(f"LOCALITY (% edges shorter than threshold)")
    print(f"{'='*62}")
    print(f"  {'Threshold':<10}  {'fractal':>10}  {'random mean+/-std':>22}")
    print(f"  {'-'*46}")
    for th in LOCAL_TH:
        rm = np.mean(rand_local[th]); rs = np.std(rand_local[th])
        print(f"  <{th:<8}  {frac_local[th]:>10.1%}  "
              f"{rm:>9.1%} +/- {rs:>6.1%}")

    print(f"\n{'='*62}")
    print(f"CROSSBAR MAPPING: cut-edge fraction at k-tile partition")
    print(f"  (low = maps well, little inter-tile wiring)")
    print(f"{'='*62}")
    print(f"  {'k Tiles':<10}  {'fractal':>10}  {'random mean+/-std':>22}")
    print(f"  {'-'*46}")
    for k in TILES:
        rm = np.mean(rand_cuts[k]); rs = np.std(rand_cuts[k])
        print(f"  {k:<10}  {frac_cuts[k]:>10.1%}  "
              f"{rm:>9.1%} +/- {rs:>6.1%}")

    # Conclusion
    wire_factor = rmean('total') / max(fs['total'], 1e-9)
    cut6_frac = frac_cuts[6]; cut6_rand = np.mean(rand_cuts[6])
    print(f"\n{'='*62}")
    print(f"CONCLUSION (Goal 2 -- hardware usability)")
    print(f"{'='*62}")
    print(f"  Total wire length: fractal {wire_factor:.1f}x shorter than random")
    print(f"  Crossbar cut (6 tiles): fractal {cut6_frac:.1%} vs "
          f"random {cut6_rand:.1%}")
    if wire_factor > 2 and cut6_frac < cut6_rand - 0.1:
        print(f"  -> GOAL 2 SUPPORTED: the fractal geometry is markedly")
        print(f"     more hardware-friendly (short wires, well partitionable).")
        print(f"     Even if computationally neutral (B-3), it is PHYSICALLY")
        print(f"     superior -- the actual value of the cosmic structure.")
    elif wire_factor > 1.3:
        print(f"  -> Partially supported: shorter wires, partition mixed.")
    else:
        print(f"  -> No clear hardware advantage.")
    print(f"{'='*62}")


if __name__ == '__main__':
    main()
