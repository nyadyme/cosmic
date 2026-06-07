#!/usr/bin/env python3
"""
Fractal Cosmic-Web topology generator (ADR-12 — trisynaptic ganglion)
=============================================================================
Generates the hierarchical, sparse PC topology per CONCEPT.md:

  - ADR-2:  tau gradient (fast leaf, slow core)
  - ADR-5:  radial = hierarchical (top clusters in the core)
  - ADR-6:  mu/eps node pairs (representation + error)
  - ADR-7:  thick peripheral shell (token embedding space)
  - ADR-9:  elastically anchored weights (spring term)
  - ADR-12: third node type iota (interneuron/gain control) per cluster

Node indexing:
  mu(c)   = 3*c      representation
  eps(c)  = 3*c + 1  error
  iota(c) = 3*c + 2  interneuron (analog precision estimator)

Intra-cluster edges:
  mu  <-> eps    (prediction-error coupling, ADR-6)
  eps <-> iota   (error signal drives interneuron)
  iota <-> mu    (gain modulation)

Output:
  - NetworkX graph with node attributes (kind, tau, radius, cluster, G_prec)
  - Sparse admittance matrix Y = L_W + diag(G) over 3*n_cluster nodes

Usage:
  python cosmic_web_generator.py
"""

import numpy as np  # always CPU NumPy (NetworkX requires it)
import networkx as nx
try:
    import cupyx.scipy.sparse as sp
except ImportError:
    import scipy.sparse as sp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CosmicWebConfig:
    n_levels: int          = 4       # L hierarchy levels (1=top/core, L=leaf)
    eta: int               = 4       # sub-clusters per parent cluster
    lambda_scale: float    = 0.5     # angular self-similarity (jitter shrink)
    n_top: int             = 6       # number of top-level clusters in the core

    # ADR-5: radial geometry
    core_radius: float     = 0.15    # radius top level (core)
    leaf_radius: float     = 1.0     # radius leaf level (periphery)
    shell_thickness: float = 0.20    # ADR-7: thickness of peripheral shell
    base_jitter: float     = 1.0     # angular spread top level (rad)

    # ADR-2: tau gradient
    tau_leaf: float        = 1.0     # fast (periphery, inference)
    tau_top: float         = 100.0   # slow (core, memory)

    # Conductances mu/eps
    G_min: float           = 0.01    # minimal node conductance (precision)
    W_min: float           = 1e-4    # minimal filament conductance
    W_intra: float         = 1.0     # mu<->eps coupling
    k_neighbors: int       = 3       # lateral k-NN filaments per level (mu-mu)

    # ADR-12: interneuron iota
    G_iota: float          = 0.05    # iota shunt conductance (smaller than G_min)
    W_eps_iota: float      = 0.5     # eps->iota coupling conductance
    W_iota_mu: float       = 0.3     # iota->mu gain modulation conductance

    seed: Optional[int]    = 42


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class CosmicWebGraph:
    G:            nx.Graph       # NetworkX graph (mu/eps/iota + attributes)
    Y:            sp.csr_matrix  # admittance matrix over 3*n_cluster nodes
    positions:    np.ndarray     # 3D positions [3N, 3]
    levels:       np.ndarray     # hierarchy level per node [3N]
    kinds:        np.ndarray     # 0=mu, 1=eps, 2=iota [3N]
    cluster_id:   np.ndarray     # cluster ID per node [3N]
    tau:          np.ndarray     # time constant per node [3N]
    radius:       np.ndarray     # distance from center per node [3N]
    d_H_angular:  float          # angular d_H reference value
    n_clusters:   int            # number of clusters
    N:            int            # total number of nodes (= 3 * n_clusters)
    n_filaments:  int            # number of edges
    fill_factor:  float          # fill ratio of the Y matrix


# ---------------------------------------------------------------------------
# Node-index helpers
# ---------------------------------------------------------------------------

def mu(c: int)   -> int: return 3 * c
def eps(c: int)  -> int: return 3 * c + 1
def iota(c: int) -> int: return 3 * c + 2


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _perturb_direction(d: np.ndarray, angle: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Perturb unit vector d by a random angle <= angle."""
    tangent = rng.normal(size=3)
    tangent = tangent - np.dot(tangent, d) * d
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return d
    tangent = tangent / norm
    a = rng.uniform(0.0, angle)
    return np.cos(a) * d + np.sin(a) * tangent


def _level_radius(level: int, config: CosmicWebConfig) -> float:
    """Target radius of a hierarchy level (ADR-5)."""
    if config.n_levels == 1:
        return config.core_radius
    frac = (level - 1) / (config.n_levels - 1)
    return config.core_radius + frac * (config.leaf_radius - config.core_radius)


def _level_tau_mu(level: int, config: CosmicWebConfig) -> float:
    """tau gradient mu: fast at periphery, slow in the core (ADR-2)."""
    if config.n_levels == 1:
        return config.tau_top
    frac_leaf = (level - 1) / (config.n_levels - 1)
    log_tau = ((1.0 - frac_leaf) * np.log(config.tau_top)
               + frac_leaf * np.log(config.tau_leaf))
    return float(np.exp(log_tau))


def _level_tau_iota(level: int, config: CosmicWebConfig) -> float:
    """tau of the interneuron: geometric mean of tau_leaf and tau_mu.

    Positions iota between the fast eps and the slow mu,
    so that iota can smooth the error trajectory without becoming too sluggish.
    Explanation at the end of the module (section EXPLANATION TAU_IOTA).
    """
    tau_mu = _level_tau_mu(level, config)
    return float(np.sqrt(config.tau_leaf * tau_mu))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_cosmic_web(config: CosmicWebConfig) -> CosmicWebGraph:
    """Generate the ADR-12-compliant trisynaptic PC topology."""
    rng = np.random.default_rng(config.seed)
    d_H_angular = np.log(config.eta) / np.log(1.0 / config.lambda_scale)

    # ------------------------------------------------------------------
    # 1. Place cluster centers radially-hierarchically (ADR-5)
    # ------------------------------------------------------------------
    centers:  list[np.ndarray] = []
    levels:   list[int]        = []
    parents:  list[int]        = []

    def place(direction: np.ndarray, level: int, parent_idx: int) -> None:
        idx = len(centers)
        radius = _level_radius(level, config)
        if level == config.n_levels:
            radius += rng.uniform(-0.5, 0.5) * config.shell_thickness
        centers.append(direction * radius)
        levels.append(level)
        parents.append(parent_idx)
        if level == config.n_levels:
            return
        jitter = config.base_jitter * config.lambda_scale ** (level - 1)
        for _ in range(config.eta):
            child_dir = _perturb_direction(direction, jitter, rng)
            place(child_dir, level + 1, idx)

    for _ in range(config.n_top):
        top_dir = rng.normal(size=3)
        top_dir /= np.linalg.norm(top_dir)
        place(top_dir, 1, -1)

    n_clusters   = len(centers)
    centers_arr  = np.array(centers)
    clev_arr     = np.array(levels)
    parent_arr   = np.array(parents)

    # ------------------------------------------------------------------
    # 2. Expand each cluster into a mu/eps/iota triplet (ADR-12)
    # ------------------------------------------------------------------
    N = 3 * n_clusters
    pos        = np.zeros((N, 3))
    lvl        = np.zeros(N, dtype=int)
    kind       = np.zeros(N, dtype=int)    # 0=mu, 1=eps, 2=iota
    cid        = np.zeros(N, dtype=int)
    tau_arr    = np.zeros(N)
    rad_arr    = np.zeros(N)
    g_prec_arr = np.zeros(N)

    G = nx.Graph()
    for c in range(n_clusters):
        lv = clev_arr[c]
        r  = float(np.linalg.norm(centers_arr[c]))
        t_mu   = _level_tau_mu(lv, config)
        t_iota = _level_tau_iota(lv, config)
        t_eps  = config.tau_leaf        # eps always fast (error reacts immediately)

        node_specs = [
            (mu(c),   0, "mu",   t_mu,   config.G_min),
            (eps(c),  1, "eps",  t_eps,  config.G_min),
            (iota(c), 2, "iota", t_iota, config.G_iota),
        ]
        for node, k, kname, t, gp in node_specs:
            pos[node]        = centers_arr[c]
            lvl[node]        = lv
            kind[node]       = k
            cid[node]        = c
            tau_arr[node]    = t
            rad_arr[node]    = r
            g_prec_arr[node] = gp
            G.add_node(node, pos=centers_arr[c], level=int(lv), kind=kname,
                       cluster=c, tau=t, radius=r, V=0.0, G_prec=gp)

    # ------------------------------------------------------------------
    # 3. Build edges
    # ------------------------------------------------------------------

    def add_edge(a: int, b: int, kind_label: str, W_override: float = 0.0
                 ) -> None:
        r = float(np.linalg.norm(pos[a] - pos[b]))
        if W_override > 0:
            W = W_override
        elif r < 1e-9:
            W = config.W_intra
        else:
            W = max(r ** (-(d_H_angular - 1)), config.W_min)
        if not G.has_edge(a, b):
            G.add_edge(a, b, W=W, r=r, etype=kind_label)

    # 3a. Intra-cluster: mu <-> eps  (ADR-6)
    for c in range(n_clusters):
        add_edge(mu(c),   eps(c),  "intra_mu_eps",  config.W_intra)

    # 3b. Intra-cluster: eps <-> iota  (ADR-12: eps drives interneuron)
    for c in range(n_clusters):
        add_edge(eps(c),  iota(c), "intra_eps_iota", config.W_eps_iota)

    # 3c. Intra-cluster: iota <-> mu  (ADR-12: gain modulation)
    for c in range(n_clusters):
        add_edge(iota(c), mu(c),   "intra_iota_mu",  config.W_iota_mu)

    # 3d. Vertical: parent-mu <-> child-eps  (top-down/bottom-up, ADR-5/6)
    for c in range(n_clusters):
        p = parent_arr[c]
        if p >= 0:
            add_edge(mu(p), eps(c), "vertical")

    # 3e. Lateral: mu <-> mu k-NN within level  (ADR-6)
    for level in range(1, config.n_levels + 1):
        members = [c for c in range(n_clusters) if clev_arr[c] == level]
        if len(members) < 2:
            continue
        pts  = centers_arr[members]
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        k_act = min(config.k_neighbors, len(members) - 1)
        for li, c in enumerate(members):
            for lj in np.argsort(dist[li])[:k_act]:
                add_edge(mu(c), mu(members[lj]), "lateral")

    # ------------------------------------------------------------------
    # 4. Admittance matrix Y = L_W + diag(G_prec)  (sparse, 3N x 3N)
    # ------------------------------------------------------------------
    rows, cols, data = [], [], []
    for a, b, d in G.edges(data=True):
        W = d["W"]
        rows += [a, b, a, b]
        cols += [b, a, a, b]
        data += [-W, -W, W, W]

    Y = (sp.csr_matrix((data, (rows, cols)), shape=(N, N))
         + sp.diags(g_prec_arr))

    n_fil = G.number_of_edges()
    return CosmicWebGraph(
        G=G, Y=Y, positions=pos, levels=lvl, kinds=kind,
        cluster_id=cid, tau=tau_arr, radius=rad_arr,
        d_H_angular=d_H_angular, n_clusters=n_clusters,
        N=N, n_filaments=n_fil, fill_factor=n_fil / (N * N),
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(web: CosmicWebGraph) -> None:
    print("=" * 60)
    print("ADR-12 trisynaptic PC topology — characteristics")
    print("=" * 60)
    print(f"  Clusters             : {web.n_clusters}")
    print(f"  Nodes N (mu+eps+iota): {web.N}")
    n_mu   = (web.kinds == 0).sum()
    n_eps  = (web.kinds == 1).sum()
    n_iota = (web.kinds == 2).sum()
    print(f"  of which mu/eps/iota : {n_mu} / {n_eps} / {n_iota}")
    print(f"  Filaments |E|        : {web.n_filaments}")
    print(f"  d_H (angular, ref.)  : {web.d_H_angular:.4f}")
    print(f"  Fill ratio           : {web.fill_factor * 100:.4f} %")

    print("  --- Levels (core -> periphery) ---")
    for lv in range(1, int(web.levels.max()) + 1):
        sel = web.levels == lv
        r   = web.radius[sel]
        # tau per kind
        sel_mu   = sel & (web.kinds == 0)
        sel_iota = sel & (web.kinds == 2)
        t_mu   = web.tau[sel_mu][0]   if sel_mu.any()   else float("nan")
        t_iota = web.tau[sel_iota][0] if sel_iota.any() else float("nan")
        print(f"    Level {lv}: {sel.sum():4d} nodes | "
              f"r={r.min():.3f}-{r.max():.3f} | "
              f"tau_mu={t_mu:.2f}  tau_iota={t_iota:.2f}")

    degs = np.array([web.G.degree(i) for i in range(web.N)])
    print(f"  Degree min/mean/max  : {degs.min()} / {degs.mean():.2f} / "
          f"{degs.max()}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(web: CosmicWebGraph, max_edges: int = 3000,
              show: bool = False) -> None:
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")
    cmap   = plt.cm.plasma
    lv_max = int(web.levels.max())

    # Draw only mu nodes (eps/iota lie coincident)
    for lv in range(1, lv_max + 1):
        sel = (web.levels == lv) & (web.kinds == 0)
        pts = web.positions[sel]
        if len(pts) == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[cmap(lv / lv_max)], s=max(4, 55 - lv * 12),
                   alpha=0.85, label=f"Level {lv}")

    color_map = {
        "vertical":       "cyan",
        "lateral":        "white",
        "intra_mu_eps":   "orange",
        "intra_eps_iota": "yellow",
        "intra_iota_mu":  "magenta",
    }
    edges = list(web.G.edges(data=True))
    if len(edges) > max_edges:
        rng_v = np.random.default_rng(0)
        edges = [edges[i] for i in
                 rng_v.choice(len(edges), max_edges, replace=False)]
    for a, b, d in edges:
        p0, p1 = web.positions[a], web.positions[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=color_map.get(d.get("etype"), "gray"),
                alpha=min(0.4, float(d["W"]) * 0.5), linewidth=0.4)

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.tick_params(colors="gray")
    ax.legend(facecolor="black", labelcolor="white", loc="upper left")
    ax.set_title(
        f"ADR-12 trisynaptic | clusters={web.n_clusters}  N={web.N} "
        f"| mu/eps/iota | core->periphery",
        color="white")
    plt.tight_layout()
    plt.savefig("figures/cosmic_web.png", dpi=150, facecolor="black")
    print("Visualization saved: cosmic_web.png")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = CosmicWebConfig()
    print("Generating ADR-12-compliant trisynaptic topology ...")
    web = generate_cosmic_web(cfg)
    analyse(web)
    visualize(web, show=False)

    # ------------------------------------------------------------------
    # EXPLANATION TAU_IOTA (geometric mean)
    # ------------------------------------------------------------------
    print()
    print("EXPLANATION: Why tau_iota = sqrt(tau_leaf * tau_mu)?")
    print("-" * 55)
    print("The interneuron iota has three tasks:")
    print("  1. It tracks the error trajectory of eps (must be fast enough)")
    print("  2. It smooths over multiple time steps (must be sluggish enough)")
    print("  3. It must not overwhelm mu prematurely (must stay below tau_mu)")
    print()
    print("The geometric mean sqrt(tau_eps * tau_mu) solves all three:")
    print("  - tau_iota lies exactly BETWEEN tau_eps and tau_mu")
    print("  - it scales logarithmically evenly (no preferred endpoint)")
    print("  - at each level the distance to both neighbors is the same number of")
    print("    orders of magnitude -> symmetric time-scale separation")
    print()
    for lv in range(1, cfg.n_levels + 1):
        t_mu   = _level_tau_mu(lv, cfg)
        t_iota = _level_tau_iota(lv, cfg)
        t_eps  = cfg.tau_leaf
        print(f"  Level {lv}: tau_eps={t_eps:.2f}  tau_iota={t_iota:.2f}"
              f"  tau_mu={t_mu:.2f}")
