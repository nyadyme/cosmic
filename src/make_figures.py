#!/usr/bin/env python3
"""
Figures for BERICHT.md / REPORT.md -- from live runs of the experiments.
============================================================================
Generates four PNGs from the real experiment building blocks (deterministic):
  fig0_model.png    -- The model: fractal Cosmic-Web topology (3D)
  fig1_compute.png  -- Goal 1: in-situ learning yields no MC gain (B-10)
  fig2_tiling.png   -- Goal 2: tile cut vs. capacity, HW limit (B-8)
  fig3_energy.png   -- Energy breakdown fractal vs random (B-9)

Invocation:  python make_figures.py
"""

import numpy as _np
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt

import simulation_insitu as si
import simulation_tile_balance as tb
import simulation_energy_spice as es

DISCLAIMER = "AI-assisted -- please reproduce independently"


def _footer(fig):
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", va="bottom",
             fontsize=7, style="italic", color="0.45")


# ------------------------------------------------------------------
# Fig 0 -- The model: fractal Cosmic-Web topology (3D)
# ------------------------------------------------------------------
def _render_topology(elev, azim, fname, alt=False, edges=False):
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    cfg = si.CosmicWebConfig(
        n_levels=4, eta=4, n_top=6, tau_leaf=1.0, tau_top=40.0,
        G_min=0.1, G_iota=0.1, W_eps_iota=0.5, W_iota_mu=0.3,
        W_intra=1.0, k_neighbors=3, seed=42)
    web = si.generate_cosmic_web(cfg)
    pos = si._cpu(web.positions)
    lvl = si._cpu(web.levels)
    cid = si._cpu(web.cluster_id)
    ncl = web.n_clusters

    cidx = [3 * c for c in range(ncl)]                 # mu node = cluster center
    cpos = pos[cidx]
    clvl = lvl[cidx]

    pairs = [(a, b) for a, b, _ in web.G.edges(data=True) if cid[a] != cid[b]]
    segs = [[pos[a], pos[b]] for a, b in pairs]

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    if edges:
        # Emphasize filaments: color-coded by length, nodes small/neutral
        lens = _np.array([float(_np.linalg.norm(pos[a] - pos[b]))
                          for a, b in pairs])
        lc = Line3DCollection(segs, linewidths=0.7, alpha=0.6, cmap="viridis")
        lc.set_array(lens)
        ax.add_collection3d(lc)
        ax.scatter(cpos[:, 0], cpos[:, 1], cpos[:, 2], c="0.15", s=5,
                   alpha=0.5, depthshade=True)
        cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("Filament length (inter-cluster)")
        ax.set_title(f"Cosmic-Web topology — filament connections\n"
                     f"{ncl} clusters · {web.n_filaments} edges "
                     f"(length color-coded)")
    else:
        ax.add_collection3d(Line3DCollection(segs, colors="0.6",
                                             linewidths=0.3, alpha=0.30))
        sizes = (cfg.n_levels - clvl + 1) * 14 + 6     # core large, leaf small
        sc = ax.scatter(cpos[:, 0], cpos[:, 1], cpos[:, 2], c=clvl,
                        cmap="plasma_r", s=sizes, edgecolors="k",
                        linewidths=0.2, depthshade=True)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02,
                          ticks=range(1, cfg.n_levels + 1))
        cb.set_label("Hierarchy level (1 = core, 4 = leaf)")
        if alt:
            ax.set_title("Cosmic-Web topology — oblique from above\n"
                         "radial hierarchy: core inside, leaves outside")
        else:
            ax.set_title(f"The model: fractal Cosmic-Web topology\n"
                         f"{ncl} μ/ε/ι clusters · {web.N} nodes · "
                         f"{web.n_filaments} filaments · τ gradient (ADR-2/12)")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("  " + fname)


def fig_model():
    _render_topology(18, 35, "figures/fig0_model.png", alt=False)
    _render_topology(52, -120, "figures/fig0b_model.png", alt=True)
    _render_topology(32, -150, "figures/fig0_edges.png", edges=True)


def fig_triplet():
    """Schematic of the μ/ε/ι node triplet per cluster (ADR-6/-12)."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_aspect("equal")          # round circles instead of ellipses

    nodes = {
        "μ": (0.30, 0.34, "#4c72b0", "Representation\nτ large (slow)"),
        "ε": (0.70, 0.34, "#c44e52", "Error\nτ = τ_leaf (fast)"),
        "ι": (0.50, 0.72, "#55a868", "Interneuron\nτ medium (gain)"),
    }
    R = 0.085
    circ = {}
    for name, (x, y, c, role) in nodes.items():
        p = Circle((x, y), R, fc=c, ec="k", lw=1.3, zorder=4)
        ax.add_patch(p); circ[name] = p
        ax.text(x, y, name, ha="center", va="center", color="w",
                fontsize=17, fontweight="bold", zorder=5)
        dy = -R - 0.045 if name != "ι" else R + 0.015
        va = "top" if name != "ι" else "bottom"
        ax.text(x, y + dy, role, ha="center", va=va, fontsize=7.5)

    def edge(a, b, label, lx, ly, rad=0.0):
        arr = FancyArrowPatch(nodes[a][:2], nodes[b][:2],
                              connectionstyle=f"arc3,rad={rad}",
                              arrowstyle="<|-|>", mutation_scale=11, lw=1.3,
                              color="0.30", patchA=circ[a], patchB=circ[b],
                              zorder=2)
        ax.add_patch(arr)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.2,
                color="0.15")

    edge("μ", "ε", "Prediction ↔ error\n(ADR-6)", 0.50, 0.27)
    edge("ε", "ι", "Error → gain", 0.655, 0.56)
    edge("ι", "μ", "Gain modulation", 0.345, 0.56)

    # Vertical top-down link from parent-μ into ε (learnable, ADR-8/9)
    box = FancyBboxPatch((0.04, 0.82), 0.30, 0.12, boxstyle="round,pad=0.01",
                         fc="#eaeaf2", ec="0.4", lw=1.0, zorder=3)
    ax.add_patch(box)
    ax.text(0.19, 0.88, "Parent-μ\n(level l−1)", ha="center", va="center",
            fontsize=8)
    arr = FancyArrowPatch((0.30, 0.82), nodes["ε"][:2],
                          connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
                          mutation_scale=13, lw=1.6, color="#8172b3",
                          patchB=circ["ε"], zorder=2)
    ax.add_patch(arr)
    ax.text(0.74, 0.62, "Top-down prediction\nf(θ)·μ  (θ learnable, ADR-8/9)",
            ha="left", va="center", fontsize=7.2, color="#5b4a86")

    ax.set_title("μ/ε/ι cluster motif (ADR-6/-12): the node triplet per cluster",
                 fontsize=10.5)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig0c_triplet.png", dpi=150)
    plt.close(fig)
    print("  fig0c_triplet.png")


# ------------------------------------------------------------------
# Fig 1 -- Goal 1: in-situ learning (B-10)
# ------------------------------------------------------------------
def fig_compute():
    web = si.generate_cosmic_web(si.NET)
    N = web.N
    leak = si.np.clip(1.0 / si.np.asarray(web.tau), 1e-3, 1.0)
    mlv = int(web.levels.max())
    leaves = [c for c in range(web.n_clusters)
              if web.G.nodes[si.mu(c)]['level'] == mlv]
    Win = si.make_Win(N, [si.eps(leaves[s]) for s in range(si.V)])
    n_edges = web.n_filaments
    pool = [d['W'] for _, _, d in web.G.edges(data=True)]

    rng_in = _np.random.default_rng(7)
    inp_tr = si.np.asarray(rng_in.integers(0, si.V, si.N_TRAIN))
    inp_ev = si.np.asarray(rng_in.integers(0, si.V, si.N_EVAL))

    W_frac = si.spectral_scale(si.fractal_coupling(web), si.RHO)
    frac = si.evaluate("fractal", W_frac, leak, Win, N, inp_tr, inp_ev)

    rand_ens = []
    for s in range(si.N_RAND):
        rng = _np.random.default_rng(200 + s)
        W_r = si.spectral_scale(
            si.random_sparse_coupling(N, n_edges, pool, rng), si.RHO)
        rand_ens.append(si.evaluate("random", W_r, leak, Win, N,
                                    inp_tr, inp_ev))
    etas = [r[0] for r in frac]
    f_mc = [r[1] for r in frac]
    r_mc = [float(_np.mean([e[i][1] for e in rand_ens]))
            for i in range(len(etas))]
    r_sd = [float(_np.std([e[i][1] for e in rand_ens]))
            for i in range(len(etas))]

    x = _np.arange(len(etas))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(f_mc[0], color="C0", ls=":", lw=1, alpha=0.6)
    ax.axhline(r_mc[0], color="C1", ls=":", lw=1, alpha=0.6)
    ax.plot(x, f_mc, "o-", color="C0", label="fractal")
    ax.errorbar(x, r_mc, yerr=r_sd, fmt="s-", color="C1",
                capsize=3, label="random-sparse (ensemble)")
    ax.set_xticks(x)
    ax.set_xticklabels([("fixed" if e == 0 else f"{e:g}") for e in etas])
    ax.set_xlabel("Learning rate η  (0 = fixed reservoir)")
    ax.set_ylabel("Memory Capacity")
    ax.set_title("Goal 1: in-situ learning yields no compute advantage (B-10)\n"
                 "random ≥ fractal throughout; learning neutral→harmful")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig1_compute.png", dpi=150)
    plt.close(fig)
    print("  fig1_compute.png")


# ------------------------------------------------------------------
# Fig 2 -- Goal 2: tile cut vs. capacity (B-8)
# ------------------------------------------------------------------
def fig_tiling():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    slacks = [1.0, 1.1, 1.25, 1.5, 1.7, 2.0]
    styles = {False: ("o-", "C0", "ι-less (μ/ε, 1020 nodes)"),
              True:  ("s-", "C2", "ι-full (μ/ε/ι, 1530 nodes)")}
    for keep in (False, True):
        pos, kinds, edges = tb.build_topology(keep)
        Nn = len(pos)
        n_mult = int((si._cpu(kinds) == 2).sum())
        k = tb.REDAC_CLUSTERS * tb.n_redacs(Nn, n_mult)
        ideal = int(_np.ceil(Nn / k))
        adj = tb.build_adjacency(edges, Nn)
        xs, ys, feasible = [], [], []
        for sl in slacks:
            cap = max(ideal, int(_np.ceil(ideal * sl)))
            lab = tb.fm_refine(adj, tb.partition_greedy(pos, k, cap), k, cap)
            st = tb.tile_stats(lab, kinds, k)
            cut, _ = tb.cut_fraction(edges, lab)
            xs.append(st['max']); ys.append(cut * 100)
            feasible.append(st['max'] <= tb.INTEGR_PER_TILE)
        fmt, col, lab = styles[keep]
        ax.plot(xs, ys, fmt, color=col, label=lab)
        # mark HW-infeasible points
        for xi, yi, ok in zip(xs, ys, feasible):
            if not ok:
                ax.plot(xi, yi, "x", color="red", ms=9, mew=2)
    ax.axvline(tb.INTEGR_PER_TILE, color="red", ls="--", lw=1.2,
               label=f"REDAC limit ({tb.INTEGR_PER_TILE} nodes/tile)")
    ax.set_xlabel("max. nodes per tile (capacity slack →)")
    ax.set_ylabel("Inter-tile cut  [%]")
    ax.set_title("Goal 2: strict balance was needlessly expensive (B-8)\n"
                 "slack up to the HW limit halves the cut; ✗ = HW-infeasible")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig2_tiling.png", dpi=150)
    plt.close(fig)
    print("  fig2_tiling.png")


# ------------------------------------------------------------------
# Fig 3 -- Energy breakdown (B-9)
# ------------------------------------------------------------------
def fig_energy():
    m = es.topo_metrics(keep_iota=False)
    eb_f = es.energy_breakdown(m['N'], m['caps'], m['wire'], m['inter_links'])
    wire_rand = m['wire'] * es.WIRE_RAND_FACTOR
    n_inter_rand = int(round(es.CUT_RAND_PHOT * m['n_edges']))
    eb_r = es.energy_breakdown(m['N'], m['caps'], wire_rand, n_inter_rand)

    comps = [('compute', 'Analog compute'), ('wire', 'Wire local'),
             ('photonic', 'Photonics E/O/E'), ('converter', 'ADC/DAC'),
             ('static', 'Quiescent current'), ('digital', 'Digital f/J')]
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3']

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for col_i, (eb, label) in enumerate([(eb_f, "fractal"),
                                         (eb_r, "random-global")]):
        bottom = 0.0
        for (key, name), c in zip(comps, colors):
            val = eb[key] / 1000.0      # pJ -> nJ
            ax.bar(col_i, val, bottom=bottom, color=c,
                   label=name if col_i == 0 else None, width=0.6)
            bottom += val
        ax.text(col_i, bottom + 0.8, f"{eb['total']/1000:.1f} nJ",
                ha="center", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fractal", "random-global"])
    ax.set_ylabel("Energy per token  [nJ]")
    ax.set_title("Energy: advantage only 1.07× — converters+quiescent current "
                 "dominate 83% (B-9)\nSPICE proxy, ι-less, 1020 nodes")
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    ax.set_ylim(0, max(eb_f['total'], eb_r['total']) / 1000 * 1.25)
    ax.grid(alpha=0.3, axis="y")
    _footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/fig3_energy.png", dpi=150)
    plt.close(fig)
    print("  fig3_energy.png")


def main():
    print("Generating figures (live runs, deterministic) ...")
    fig_model()
    fig_triplet()
    fig_compute()
    fig_tiling()
    fig_energy()
    print("Done: fig0_model, fig1_compute, fig2_tiling, fig3_energy")


if __name__ == '__main__':
    main()
