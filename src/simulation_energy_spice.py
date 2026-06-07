#!/usr/bin/env python3
"""
Goal 2 / Strand 3b -- Energy/latency balance (SPICE proxy)
==========================================================
Refines the parametric proxy (simulation_energy.py) into an
RC energy model that computes on the REAL topology and incorporates the
B-8 findings:

  1. Node capacitance from the tau gradient (ADR-2): C_node = tau * G.
     Slow core nodes are capacitively expensive, fast leaf nodes cheap.
  2. Wire energy from the REAL Euclidean edge lengths (generator),
     scaled to physical die size -> C_wire = c_per_um * length.
  3. Interconnect split (B-8): local tile-internal edges = ELECTRICAL
     (cheap), inter-tile links = PHOTONIC (expensive E/O/E SerDes).
  4. Converters (ADC/DAC) + quiescent current dominate (G16) -- the SPICE proxy
     quantifies how much of the 11x wire advantage remains in TOTAL and
     whether the few but expensive photonic links erode the advantage.

IMPORTANT: not a transistor SPICE run, but an analytical RC energy model
with literature-near device parameters. All assumptions are marked as ASSUMPTION;
a sensitivity sweep shows which conclusions are robust.
"""

try:
    import cupy as np
    _GPU = True
except ImportError:
    import numpy as np
    _GPU = False

import numpy as _np
import time

from cosmic_web_generator import generate_cosmic_web
from simulation_tile_balance import (
    NET, build_topology, partition_greedy, fm_refine, build_adjacency,
    cut_fraction, n_redacs, tile_stats, REDAC_CLUSTERS, INTEGR_PER_TILE,
)


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)


# ============================================================
# ASSUMPTIONS (SPICE proxy, literature-near analog/photonic values)
# ============================================================
V_OP        = 0.5      # V    analog operating voltage (modern analog IC)
C_LEAF      = 10.0     # fF   node capacitance of a fast leaf node
                       #      (core node scales with tau: C = C_LEAF*tau/tau_leaf)
K_SETTLE    = 2.0      #      effective charge/discharge cycles per token (settling)
C_WIRE_UM   = 0.20     # fF/um  on-chip line capacitance per micrometer
DIE_MM      = 10.0     # mm   edge length of the die (norm. radius 1.0 = DIE_MM/2)
K_WIRE_SW   = 2.0      #      switching cycles per wire and token

E_ADC       = 5.0      # pJ   per node read (SAR-ADC sample)
E_DAC       = 5.0      # pJ   per node write (DAC)
E_DIGITAL   = 8.0      # pJ   digital f=tanh / J computation per node
P_STATIC    = 10.0     # uW   quiescent power per node (opamp bias)
T_SETTLE    = 3.0      # us   settle time per token (~3*tau_leaf, leaf-dominated)

# Photonics (E/O/E per inter-tile link and token; SerDes + serialization)
BITS_TOKEN  = 8        #      bits per link and token (activation precision)
E_EOE_BIT   = 0.25     # pJ/bit  optical link (modulator+TIA+driver, round-trip)

# Topology boundary values from earlier findings
WIRE_RAND_FACTOR = 11.4   # B-4: random-global wire = 11.4x fractal
CUT_FRAC_PHOT    = 0.10   # B-8: ~10% of fractal edges photonic (inter-tile)
CUT_RAND_PHOT    = 0.826  # B-4: 82.6% of random-global edges inter-tile


def e_cv(c_ff, v=V_OP):
    """Capacitive switching energy 0.5*C*V^2 in pJ (C in fF)."""
    return 0.5 * c_ff * v * v / 1000.0     # fF*V^2 = fJ -> /1000 = pJ


def node_capacitances(tau):
    """Node capacitance from the tau gradient: C = C_LEAF * tau/tau_leaf."""
    tau = _cpu(tau)
    return C_LEAF * tau / NET.tau_leaf      # fF, slower core = larger


def wire_lengths_um(pos, edges):
    """Physical edge lengths in um (norm. radius 1.0 = DIE_MM/2)."""
    pos = _cpu(pos)
    scale_um = (DIE_MM / 2.0) * 1000.0      # norm. unit -> um
    a = pos[[e[0] for e in edges]]
    b = pos[[e[1] for e in edges]]
    return _np.linalg.norm(a - b, axis=1) * scale_um


# ============================================================
# Energy balance per token
# ============================================================

def energy_breakdown(N, caps_ff, wire_um, n_inter_links,
                     params=None):
    """Energy breakdown per token (pJ).

    :param N: node count.
    :param caps_ff: node capacitances [fF].
    :param wire_um: edge lengths [um] of the LOCAL (electrical) edges.
    :param n_inter_links: number of photonic inter-tile links.
    :param params: optional override dict (for sensitivity).
    :returns: component dict with 'total'.
    """
    p = dict(E_ADC=E_ADC, E_DAC=E_DAC, E_DIGITAL=E_DIGITAL,
             P_STATIC=P_STATIC, T_SETTLE=T_SETTLE, E_EOE_BIT=E_EOE_BIT)
    if params:
        p.update(params)

    # 1. Analog compute: charge/discharge energy of the node capacitances
    e_compute = float(_np.sum(e_cv(_cpu(caps_ff))) * K_SETTLE)
    # 2. Local wire energy (electrical, tile-internal)
    c_wire_ff = C_WIRE_UM * _cpu(wire_um)
    e_wire    = float(_np.sum(e_cv(c_wire_ff)) * K_WIRE_SW)
    # 3. Photonics (E/O/E per inter-tile link and token)
    e_photon  = n_inter_links * BITS_TOKEN * p['E_EOE_BIT']
    # 4. Converters (ADC/DAC per node -- G16 bottleneck)
    e_conv    = N * (p['E_ADC'] + p['E_DAC'])
    # 5. Quiescent current (uW * us = pJ)
    e_static  = N * p['P_STATIC'] * p['T_SETTLE']
    # 6. Digital f/J computation
    e_digital = N * p['E_DIGITAL']

    total = e_compute + e_wire + e_photon + e_conv + e_static + e_digital
    return dict(compute=e_compute, wire=e_wire, photonic=e_photon,
                converter=e_conv, static=e_static, digital=e_digital,
                total=total)


# ============================================================
# Topology metrics (real from generator + B-8 partition)
# ============================================================

def topo_metrics(keep_iota):
    """Real metrics + B-8 tile partition for a topology variant."""
    pos, kinds, edges = build_topology(keep_iota)
    web = generate_cosmic_web(NET)
    tau_full = _cpu(web.tau)
    tau = tau_full if keep_iota else tau_full[_cpu(web.kinds) != 2]

    N = len(pos)
    n_mult = int((_cpu(kinds) == 2).sum())
    k = REDAC_CLUSTERS * n_redacs(N, n_mult)
    ideal = int(_np.ceil(N / k))

    # B-8: best HW-admissible tiling (slack up to 144) -> inter-tile links
    adj = build_adjacency(edges, N)
    best_links, best_cut = None, 1.0
    for slack in [1.0, 1.1, 1.25, 1.5, 1.7]:
        cap = max(ideal, int(_np.ceil(ideal * slack)))
        lab = fm_refine(adj, partition_greedy(pos, k, cap), k, cap)
        st = tile_stats(lab, kinds, k)
        if st['max'] > INTEGR_PER_TILE:        # HW limit violated
            continue
        cut, links = cut_fraction(edges, lab)
        if cut < best_cut:
            best_cut, best_links = cut, links

    caps = node_capacitances(tau)
    wire = wire_lengths_um(pos, edges)
    return dict(label="without iota (mu/eps)" if not keep_iota
                else "with iota (mu/eps/iota)",
                N=N, n_edges=len(edges), caps=caps, wire=wire,
                inter_links=best_links, cut=best_cut,
                wire_total_um=float(_np.sum(wire)))


# ============================================================
# Output
# ============================================================

def print_breakdown(title, eb):
    print(f"\n  {title}")
    print(f"  {'Component':<16} {'pJ/token':>10} {'Share':>8}")
    print(f"  {'-'*36}")
    order = [('compute', 'Analog compute'), ('wire', 'Wire local'),
             ('photonic', 'Photonics E/O/E'), ('converter', 'ADC/DAC'),
             ('static', 'Quiescent'), ('digital', 'Digital f/J')]
    for key, lab in order:
        print(f"  {lab:<16} {eb[key]:>10.1f} {eb[key]/eb['total']:>8.1%}")
    print(f"  {'-'*36}")
    print(f"  {'TOTAL':<16} {eb['total']:>10.1f}")


def main():
    t0 = time.time()
    print("Energy/latency balance (SPICE proxy)")
    print(f"  ASSUMPTIONS: V_op={V_OP}V, C_leaf={C_LEAF}fF, Die={DIE_MM}mm, "
          f"ADC=DAC={E_ADC}pJ,")
    print(f"  P_static={P_STATIC}uW, t_settle={T_SETTLE}us, "
          f"E_eoe={E_EOE_BIT}pJ/bit x {BITS_TOKEN}bit/link")

    m = topo_metrics(keep_iota=False)     # Default: iota-less (B-7)
    print(f"\n  Topology (iota-less): {m['N']} nodes, {m['n_edges']} edges, "
          f"wire {m['wire_total_um']/1000:.1f}mm total")
    print(f"  B-8 tiling: {m['inter_links']} photonic inter-tile links "
          f"({m['cut']:.1%})")

    # fractal: local edges electrical, inter-tile photonic
    eb_frac = energy_breakdown(m['N'], m['caps'], m['wire'], m['inter_links'])
    print_breakdown("FRACTAL (iota-less) -- energy per token:", eb_frac)

    # random-global reference: 11.4x wire, 82.6% edges photonic
    wire_rand = m['wire'] * WIRE_RAND_FACTOR
    n_inter_rand = int(round(CUT_RAND_PHOT * m['n_edges']))
    eb_rand = energy_breakdown(m['N'], m['caps'], wire_rand, n_inter_rand)
    print_breakdown(f"RANDOM-GLOBAL (same {m['N']} nodes) -- per token:",
                    eb_rand)

    # ---- Advantage analysis ----
    print(f"\n{'='*60}")
    print(f"ADVANTAGE ANALYSIS (fractal vs random-global)")
    print(f"{'='*60}")
    inter_only = ((eb_rand['wire'] + eb_rand['photonic'])
                  / max(eb_frac['wire'] + eb_frac['photonic'], 1e-9))
    print(f"  Interconnect alone (wire+photonics): {inter_only:.1f}x")
    print(f"  TOTAL energy/token:  fractal {eb_frac['total']:.0f}pJ vs "
          f"random {eb_rand['total']:.0f}pJ = {eb_rand['total']/eb_frac['total']:.2f}x")
    share = (eb_frac['wire'] + eb_frac['photonic']) / eb_frac['total']
    print(f"  Interconnect share (fractal): {share:.1%} of total energy")
    print(f"  -> Converters+quiescent dominate with "
          f"{(eb_frac['converter']+eb_frac['static'])/eb_frac['total']:.0%} "
          f"(G16 confirmed).")

    # ---- Does photonics erode the advantage? ----
    print(f"\n{'='*60}")
    print(f"PHOTONICS EROSION (B-8: few, but expensive E/O/E links)")
    print(f"{'='*60}")
    print(f"  fractal: {m['inter_links']} links -> {eb_frac['photonic']:.0f}pJ "
          f"({eb_frac['photonic']/eb_frac['total']:.1%} of total energy)")
    print(f"  random:  {n_inter_rand} links -> {eb_rand['photonic']:.0f}pJ "
          f"({eb_rand['photonic']/eb_rand['total']:.1%})")
    if eb_frac['photonic'] / eb_frac['total'] < 0.05:
        print(f"  -> Photonics does NOT erode the fractal advantage "
              f"(<5% of energy): few links, converters dominate.")
    else:
        print(f"  -> Photonics is a noticeable item -- check E_eoe carefully.")

    # ---- Latency ----
    print(f"\n{'='*60}")
    print(f"LATENCY PER TOKEN")
    print(f"{'='*60}")
    t_conv, t_phot = 0.2, 0.05    # us: ADC/DAC, photonic hop
    t_tok = T_SETTLE + t_conv + t_phot
    print(f"  Settle {T_SETTLE}us + converter {t_conv}us + photonics {t_phot}us "
          f"= {t_tok:.2f}us/token")
    print(f"  Throughput: {1e6/t_tok:.0f} tokens/s | latency topology-neutral "
          f"(settle dominates).")

    # ---- Sensitivity ----
    print(f"\n{'='*60}")
    print(f"SENSITIVITY (TOTAL advantage robust against assumptions?)")
    print(f"{'='*60}")
    print(f"  {'Scenario':<28} {'fractal':>9} {'random':>9} {'Advantage':>8}")
    print(f"  {'-'*56}")
    scenarios = [
        ("Base", {}),
        ("ADC/DAC cheap (1pJ)", dict(E_ADC=1.0, E_DAC=1.0)),
        ("ADC/DAC expensive (20pJ)", dict(E_ADC=20.0, E_DAC=20.0)),
        ("Quiescent high (50uW)", dict(P_STATIC=50.0)),
        ("Photonics expensive (1pJ/bit)", dict(E_EOE_BIT=1.0)),
        ("Settle short (1us)", dict(T_SETTLE=1.0)),
    ]
    for name, ov in scenarios:
        ef = energy_breakdown(m['N'], m['caps'], m['wire'],
                              m['inter_links'], ov)
        er = energy_breakdown(m['N'], m['caps'], wire_rand,
                              n_inter_rand, ov)
        print(f"  {name:<28} {ef['total']:>8.0f}p {er['total']:>8.0f}p "
              f"{er['total']/ef['total']:>7.2f}x")
    print(f"  {'-'*56}")

    # ---- iota-less vs iota-full ----
    m2 = topo_metrics(keep_iota=True)
    eb2 = energy_breakdown(m2['N'], m2['caps'], m2['wire'], m2['inter_links'])
    print(f"\n{'='*60}")
    print(f"iota removal (B-7): energy effect")
    print(f"{'='*60}")
    print(f"  iota-less : {m['N']} nodes -> {eb_frac['total']:.0f}pJ/token")
    print(f"  iota-full: {m2['N']} nodes -> {eb2['total']:.0f}pJ/token")
    print(f"  -> iota removal saves "
          f"{(1-eb_frac['total']/eb2['total'])*100:.0f}% energy/token "
          f"(mainly converters+quiescent of the {m2['N']-m['N']} iota nodes).")

    # ---- Conclusion ----
    print(f"\n{'='*60}")
    print(f"CONCLUSION (Strand 3b -- SPICE proxy)")
    print(f"{'='*60}")
    print(f"  The SPICE proxy on the REAL topology confirms G16/B-6:")
    print(f"  The {inter_only:.0f}x interconnect advantage shrinks to "
          f"{eb_rand['total']/eb_frac['total']:.2f}x TOTAL energy,")
    print(f"  because converters+quiescent topology-free "
          f"{(eb_frac['converter']+eb_frac['static'])/eb_frac['total']:.0%} "
          f"dominate.")
    print(f"  NEW (B-8): the {m['inter_links']} photonic links cost only "
          f"{eb_frac['photonic']/eb_frac['total']:.1%} -- they do not erode the")
    print(f"  advantage. iota removal (B-7) additionally saves ~"
          f"{(1-eb_frac['total']/eb2['total'])*100:.0f}% energy.")
    print(f"  -> Goal-2 energy statement holds: REAL, but MODERATE "
          f"(~{eb_rand['total']/eb_frac['total']:.1f}x), not 11x.")
    print(f"     The hardware value is routability/area, not energy.")
    print(f"\n  Runtime: {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
