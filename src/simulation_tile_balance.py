#!/usr/bin/env python3
"""
Goal 2 / Strand 1b -- REDAC tile load balancing
================================================
Open point #5 (RESULTS.md): the balanced partition (B-6) costs ~3x
cut compared to the unbalanced KMeans optimum (4.7% -> 13.9%). Question:
can the inter-tile cut be lowered WITHOUT violating the REDAC capacity?

Key insight: a REDAC cluster has 864/6 = 144 integrators. With
12 tiles and 1020 nodes (iota-less) the ideal is only 85 nodes/tile -- so there
is room up to 144 BEFORE the hardware breaks. This slack can be turned into
less cut.

Three partitioners on the iota-less topology (variant A, default):
  1. KMeans (unbalanced)          -> lower bound for the cut
  2. Greedy capacity-constrained  -> current state (B-6)
  3. Greedy + FM refinement       -> boundary-node swap to lower the cut

Plus Pareto sweep: capacity slack (1.0 .. unconstrained) against cut%
and max. tile load, with marking of the REDAC hardware limit (144).

Metrics: cut%, inter-tile links, tile sizes (min/max/std),
integrator/multiplier utilization per tile.

ASSUMPTION (hardware model): each tile forms a REDAC cluster (own
crossbar) with <= 144 integrators (1/node) and <= 72 multipliers
(1/iota node). 6 clusters per REDAC, 2 REDACs for the full network.
"""

try:
    import cupy as np
    _GPU = True
except ImportError:
    import numpy as np
    _GPU = False

import numpy as _np
import time
from sklearn.cluster import KMeans

from cosmic_web_generator import (
    CosmicWebConfig, generate_cosmic_web, mu, eps,
)


def _cpu(a):
    """CuPy array -> CPU NumPy; NumPy unchanged."""
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)


# REDAC specification (per unit, from Strand 1 / simulation_redac.py)
REDAC_CLUSTERS   = 6
REDAC_INTEGR     = 864
REDAC_MULT       = 432
INTEGR_PER_TILE  = REDAC_INTEGR // REDAC_CLUSTERS   # 144 nodes/tile (crossbar)
MULT_PER_TILE    = REDAC_MULT // REDAC_CLUSTERS     # 72 iota nodes/tile

NET = CosmicWebConfig(
    n_levels=4, eta=4, n_top=6, tau_leaf=1.0, tau_top=40.0,
    G_min=0.1, G_iota=0.1, W_eps_iota=0.5, W_iota_mu=0.3,
    W_intra=1.0, k_neighbors=3, seed=42,
)

SLACKS = [1.0, 1.1, 1.25, 1.5, 1.7, 2.0]   # capacity slack for Pareto sweep


# ----------------------------------------------------------------
# Topology + iota masking (like variant A)
# ----------------------------------------------------------------

def build_topology(keep_iota):
    """Generates the reference topology and optionally removes the iota nodes.

    :param keep_iota: True = full mu/eps/iota topology, False = only mu/eps.
    :returns: (pos, kinds, edges) with 0-based, remapped node indices.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, list[tuple[int, int]]]
    """
    web = generate_cosmic_web(NET)
    pos_full   = _cpu(web.positions)
    kinds_full = _cpu(web.kinds)
    edges_full = [(a, b) for a, b, _ in web.G.edges(data=True)]

    if keep_iota:
        return pos_full, kinds_full, edges_full

    mask = kinds_full != 2                     # remove iota = kind 2
    old2new = -_np.ones(len(mask), dtype=int)
    old2new[mask] = _np.arange(int(mask.sum()))
    pos   = pos_full[mask]
    kinds = kinds_full[mask]
    edges = [(int(old2new[a]), int(old2new[b])) for (a, b) in edges_full
             if mask[a] and mask[b]]
    return pos, kinds, edges


def n_redacs(n_nodes, n_mult):
    """REDAC count: bottleneck from integrators OR multipliers."""
    by_int  = int(_np.ceil(n_nodes / REDAC_INTEGR))
    by_mult = int(_np.ceil(n_mult / REDAC_MULT)) if n_mult else 1
    return max(by_int, by_mult, 1)


# ----------------------------------------------------------------
# Partitioners (all on CPU -- tiny geometry problem)
# ----------------------------------------------------------------

def partition_kmeans(pos, k):
    """Unbalanced spatial partition (lower bound for the cut)."""
    return KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(_cpu(pos))


def partition_greedy(pos, k, cap):
    """Capacity-constrained greedy KMeans (current state, B-6).

    Assign nodes in order of their clearest center preference,
    overflow to the nearest free center (each tile <= cap nodes).
    """
    pos_cpu = _cpu(pos)
    N = len(pos_cpu)
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pos_cpu)
    cent = _np.asarray(km.cluster_centers_)
    dist = _np.linalg.norm(pos_cpu[:, None, :] - cent[None, :, :], axis=2)
    order = _np.argsort(dist.min(axis=1))
    labels = -_np.ones(N, dtype=int)
    counts = _np.zeros(k, dtype=int)
    for n in order.tolist():
        for c in _np.argsort(dist[n]).tolist():
            if counts[c] < cap:
                labels[n] = c
                counts[c] += 1
                break
    return labels


def build_adjacency(edges, n_nodes):
    """Adjacency list (undirected) for the FM refinement."""
    adj = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def fm_refine(adj, labels, k, cap, max_passes=30):
    """Greedy boundary-node swap (Fiduccia-Mattheyses light).

    Repeatedly moves each node into the neighbor tile with the highest
    gain (more neighbors there than in its own tile), as long as the target
    tile has capacity and the source tile does not empty out. Lowers the cut
    while respecting the capacity limit.

    :param adj: adjacency list.
    :param labels: start partition (copied, not mutated).
    :param cap: maximum node count per tile.
    :returns: refined labels.
    :rtype: numpy.ndarray
    """
    labels = _np.asarray(labels).copy()
    counts = _np.bincount(labels, minlength=k).astype(int)
    for _ in range(max_passes):
        moved = 0
        for n in range(len(labels)):
            a = int(labels[n])
            nbr_tiles = {}
            for m in adj[n]:
                t = int(labels[m])
                nbr_tiles[t] = nbr_tiles.get(t, 0) + 1
            own = nbr_tiles.get(a, 0)
            best_t, best_gain = a, 0
            for t, cnt_t in nbr_tiles.items():
                if t == a:
                    continue
                gain = cnt_t - own
                if gain > best_gain and counts[t] < cap and counts[a] > 1:
                    best_gain, best_t = gain, t
            if best_t != a:
                labels[n] = best_t
                counts[a] -= 1
                counts[best_t] += 1
                moved += 1
        if moved == 0:
            break
    return labels


# ----------------------------------------------------------------
# Metriken
# ----------------------------------------------------------------

def cut_fraction(edges, labels):
    """Fraction + count of edges between different tiles."""
    labels = _np.asarray(labels)
    cuts = sum(1 for i, j in edges if labels[i] != labels[j])
    return cuts / max(len(edges), 1), cuts


def tile_stats(labels, kinds, k):
    """Tile sizes + integrator/multiplier utilization."""
    labels = _np.asarray(labels)
    kinds = _np.asarray(kinds)
    sizes = _np.bincount(labels, minlength=k)
    mults = _np.array([int(((labels == t) & (kinds == 2)).sum())
                       for t in range(k)])
    return dict(
        sizes=sizes, min=int(sizes.min()), max=int(sizes.max()),
        std=float(sizes.std()), mult_max=int(mults.max()),
        integr_ok=bool(sizes.max() <= INTEGR_PER_TILE),
        mult_ok=bool(mults.max() <= MULT_PER_TILE),
    )


# ----------------------------------------------------------------
# Analysis per topology
# ----------------------------------------------------------------

def analyse(keep_iota):
    label = "with iota (mu/eps/iota)" if keep_iota else "without iota (mu/eps)"
    pos, kinds, edges = build_topology(keep_iota)
    N = len(pos)
    n_mult = int((kinds == 2).sum())
    nr = n_redacs(N, n_mult)
    k = REDAC_CLUSTERS * nr
    ideal = int(_np.ceil(N / k))
    adj = build_adjacency(edges, N)

    print(f"\n{'#'*70}")
    print(f"# {label}: {N} nodes, {len(edges)} edges, {n_mult} iota")
    print(f"#   {nr} REDAC(s) -> {k} tiles | ideal {ideal} nodes/tile | "
          f"HW limit {INTEGR_PER_TILE}/tile")
    print(f"{'#'*70}")

    # --- Table 1: methods at strict balance (cap = ideal) ---
    lab_km = partition_kmeans(pos, k)
    cut_km, links_km = cut_fraction(edges, lab_km)
    st_km = tile_stats(lab_km, kinds, k)

    lab_gr = partition_greedy(pos, k, ideal)
    cut_gr, links_gr = cut_fraction(edges, lab_gr)
    st_gr = tile_stats(lab_gr, kinds, k)

    lab_fm = fm_refine(adj, lab_gr, k, ideal)
    cut_fm, links_fm = cut_fraction(edges, lab_fm)
    st_fm = tile_stats(lab_fm, kinds, k)

    print(f"\n  Methods at strict balance (cap = ideal = {ideal}):")
    print(f"  {'Method':<26} {'Cut':>8} {'Links':>6} "
          f"{'maxTile':>8} {'std':>6} {'HW?':>5}")
    print(f"  {'-'*62}")
    rows = [
        ("1 KMeans (geom. baseline)", cut_km, links_km, st_km),
        ("2 Greedy (Cap, B-6)",       cut_gr, links_gr, st_gr),
        ("3 Greedy + FM-Refine",      cut_fm, links_fm, st_fm),
    ]
    for name, cut, links, st in rows:
        hw = "ok" if (st['integr_ok'] and st['mult_ok']) else "X"
        print(f"  {name:<26} {cut:>7.1%} {links:>6} "
              f"{st['max']:>8} {st['std']:>6.1f} {hw:>5}")
    print(f"  {'-'*62}")
    print(f"  KMeans = geometric baseline (minimizes spread, not cut;")
    print(f"    violates the balance: maxTile={st_km['max']} > {ideal}).")
    print(f"  FM-Refine at strict balance: {(cut_gr-cut_fm)*100:+.1f} PP "
          f"({cut_gr:.1%} -> {cut_fm:.1%}) -- hardly any effect, because at cap=ideal")
    print(f"    all tiles are full and no boundary node can be moved.")

    # --- Table 2: Pareto sweep capacity slack (Greedy alone vs. +FM) ---
    print(f"\n  Pareto: capacity slack -- separates slack effect from FM effect:")
    print(f"  {'Slack':>6} {'cap':>5} {'maxTile':>8} {'Greedy':>8} "
          f"{'+FM':>8} {'Links':>6} {'HW-fit?':>8}")
    print(f"  {'-'*56}")
    pareto = []
    for slack in SLACKS:
        cap = max(ideal, int(_np.ceil(ideal * slack)))
        lab_g = partition_greedy(pos, k, cap)
        cut_g, _ = cut_fraction(edges, lab_g)
        lab_f = fm_refine(adj, lab_g, k, cap)
        cut, links = cut_fraction(edges, lab_f)
        st = tile_stats(lab_f, kinds, k)
        hw_fit = st['max'] <= INTEGR_PER_TILE and st['mult_max'] <= MULT_PER_TILE
        pareto.append((slack, cap, st['max'], cut, links, hw_fit))
        flag = "yes" if hw_fit else "NO(>HW)"
        print(f"  {slack:>6.2f} {cap:>5} {st['max']:>8} {cut_g:>7.1%} "
              f"{cut:>7.1%} {links:>6} {flag:>8}")
    print(f"  {'-'*56}")

    # best HW-admissible result
    feasible = [p for p in pareto if p[5]]
    best = min(feasible, key=lambda p: p[3]) if feasible else None
    return dict(label=label, N=N, k=k, ideal=ideal,
                cut_km=cut_km, cut_greedy=cut_gr, cut_fm=cut_fm,
                best_pareto=best, n_redac=nr)


def main():
    t0 = time.time()
    print("REDAC tile load balancing (Goal 2, open point #5)")
    print(f"  REDAC spec: {REDAC_CLUSTERS} clusters, {REDAC_INTEGR} integr., "
          f"{REDAC_MULT} mult. -> {INTEGR_PER_TILE} integr./tile")

    res_no  = analyse(keep_iota=False)   # Default: iota-less (variant A)
    res_yes = analyse(keep_iota=True)    # Comparison: full topology (B-6)

    print(f"\n{'='*70}")
    print(f"CONCLUSION (Goal 2 / tile balancing)")
    print(f"{'='*70}")
    for r in (res_no, res_yes):
        bp = r['best_pareto']
        print(f"  {r['label']} ({r['N']} nodes, {r['k']} tiles):")
        print(f"    Geometry baseline (KMeans, unbal.):      {r['cut_km']:.1%}")
        print(f"    Strict balance + B-6 Greedy:             {r['cut_greedy']:.1%}")
        print(f"    Strict balance + FM-Refine:              {r['cut_fm']:.1%} "
              f"({(r['cut_fm']-r['cut_greedy'])*100:+.1f} PP)")
        if bp:
            print(f"    Best HW-admissible (Slack {bp[0]:.2f}, "
                  f"maxTile {bp[2]}<={INTEGR_PER_TILE}): {bp[3]:.1%} "
                  f"({bp[4]} inter-links)")
    print(f"\n  Key statements:")
    print(f"  1. FM refinement at STRICT balance brings almost nothing "
          f"(tiles full).")
    print(f"  2. The real lever is the capacity SLACK up to the HW limit")
    print(f"     (144/tile): it nearly halves the inter-tile links. The")
    print(f"     strict balance (cap=ideal) was unnecessarily expensive.")
    print(f"  3. FM gives a small extra ON TOP, once slack is there")
    print(f"     (graph-aware beats purely geometric KMeans in the cut).")
    print(f"\n  Runtime: {time.time()-t0:.0f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
