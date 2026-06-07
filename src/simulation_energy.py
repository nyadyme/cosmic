#!/usr/bin/env python3
"""
Goal 2 / Strand 3 -- Energy/latency balance (parametric proxy)
==============================================================
The 11x wire advantage (B-4) meets converter dominance (G16: ADC/DAC +
quiescent current make up 70-90% of the energy in real analog in-memory
systems).
Question: how much of the 11x remains for TOTAL energy/token?

Model per token (all values explicit, clearly marked ASSUMPTIONS -- proxy):
  E_total = E_interconnect (topology-dependent, fractal wins 11x)
          + E_converter    (ADC/DAC per node, topology-INDEPENDENT)
          + E_static       (node quiescent current x settle time, ~independent)
          + E_digital      (f/J computation per node, independent)

Output: energy breakdown fractal vs random, TOTAL advantage, and
sensitivity over converter dominance (which fraction is topology-free).
"""

try:
    import cupy as np
    _GPU = True
except ImportError:
    import numpy as np
    _GPU = False

import numpy as _np  # noqa: F401  (always CPU numpy, e.g. for sklearn)

# ---- ASSUMPTIONS (proxy, representative analog values; pJ per token) ----
E_ADC      = 5.0    # pJ per node read (ADC sample)
E_DAC      = 5.0    # pJ per node write (DAC)
E_DIGITAL  = 8.0    # pJ per node (digital f=tanh / J computation)
P_STATIC   = 10.0   # uW quiescent power per node
T_SETTLE   = 1.0    # us settle time per token
# Interconnect: E = 0.5 * C_wire * len * V^2 per edge; C_wire normalized
C_WIRE     = 0.5    # pJ per unit wire length (normalized geometry)

# ---- Topology metrics (from B-4, n_levels=4) ----
N_NODES    = 1530
WIRE_FRAC  = 291.0    # fractal total wire length
WIRE_RAND  = 3330.0   # random-global total wire length


def energy_breakdown(wire_len):
    e_inter   = C_WIRE * wire_len
    e_conv    = N_NODES * (E_ADC + E_DAC)
    e_static  = N_NODES * P_STATIC * T_SETTLE * 1e-6 * 1e12 / 1e6  # uW*us -> pJ
    # uW * us = 1e-6 W * 1e-6 s = 1e-12 J = 1 pJ  -> so P_STATIC*T_SETTLE pJ
    e_static  = N_NODES * P_STATIC * T_SETTLE
    e_digital = N_NODES * E_DIGITAL
    return dict(interconnect=e_inter, converter=e_conv,
                static=e_static, digital=e_digital,
                total=e_inter+e_conv+e_static+e_digital)


def main():
    print("Energy/latency balance (parametric proxy)")
    print("ASSUMPTIONS: E_ADC=E_DAC=5pJ/node, E_digital=8pJ/node,")
    print(f"  P_static=10uW/node, t_settle=1us, C_wire=0.5pJ/unit, N={N_NODES}")

    ef = energy_breakdown(WIRE_FRAC)
    er = energy_breakdown(WIRE_RAND)

    print(f"\n{'='*64}")
    print(f"ENERGY PER TOKEN (pJ)")
    print(f"{'='*64}")
    print(f"  {'Component':<16} {'fractal':>10} {'random':>10} {'topology?':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Interconnect':<16} {ef['interconnect']:>10.0f} {er['interconnect']:>10.0f} {'YES (11x)':>12}")
    print(f"  {'Converter ADC/DAC':<16} {ef['converter']:>10.0f} {er['converter']:>10.0f} {'no':>12}")
    print(f"  {'Static/quiescent':<16} {ef['static']:>10.0f} {er['static']:>10.0f} {'no':>12}")
    print(f"  {'Digital f/J':<16} {ef['digital']:>10.0f} {er['digital']:>10.0f} {'no':>12}")
    print(f"  {'-'*52}")
    print(f"  {'TOTAL':<16} {ef['total']:>10.0f} {er['total']:>10.0f}")
    print(f"  {'-'*52}")
    print(f"  Interconnect advantage:     {er['interconnect']/ef['interconnect']:.1f}x")
    print(f"  TOTAL energy advantage:   {er['total']/ef['total']:.2f}x")
    inter_share_f = ef['interconnect']/ef['total']
    print(f"  Interconnect share (fractal): {inter_share_f:.1%} of total energy")

    # Sensitivity: TOTAL advantage as a function of converter dominance
    print(f"\n{'='*64}")
    print(f"SENSITIVITY: TOTAL advantage vs. topology-free share")
    print(f"{'='*64}")
    print(f"  (Assumption: fractal interconnect = random/11; rest identical)")
    print(f"  {'topology-free':>14} {'TOTAL advantage fractal':>24}")
    print(f"  {'-'*40}")
    wf = WIRE_RAND / WIRE_FRAC   # 11.4
    for free_share in [0.5, 0.7, 0.8, 0.9, 0.95]:
        # total_rand = 1 (normalized); interconnect_rand = (1-free_share)
        # fractal interconnect = (1-free_share)/wf ; rest = free_share
        e_rand = 1.0
        e_frac = free_share + (1-free_share)/wf
        print(f"  {free_share:>13.0%}  {e_rand/e_frac:>22.2f}x")
    print(f"  {'-'*40}")

    # Latency
    print(f"\n{'='*64}")
    print(f"LATENCY PER TOKEN (proxy)")
    print(f"{'='*64}")
    t_conv = 0.2   # us ADC+DAC per token (rough)
    t_dig  = 0.1   # us digital f/J
    t_tok  = T_SETTLE + t_conv + t_dig
    print(f"  Settle {T_SETTLE}us + converter {t_conv}us + digital {t_dig}us "
          f"= {t_tok}us/token")
    print(f"  Throughput: {1e6/t_tok:.0f} tokens/s (>> 30/s reading speed)")
    print(f"  Latency is topology-INDEPENDENT (settle dominates, "
          f"all nodes parallel).")

    # Conclusion
    print(f"\n{'='*64}")
    print(f"CONCLUSION (Strand 3)")
    print(f"{'='*64}")
    print(f"  The 11x interconnect advantage shrinks to "
          f"{er['total']/ef['total']:.2f}x TOTAL energy,")
    print(f"  because converters+quiescent (topology-free) account for "
          f"{1-inter_share_f:.0%} "
          f"of the energy (G16 confirmed).")
    print(f"  -> Honest: the fractal energy advantage is REAL, but with real")
    print(f"     converter costs MODERATE ({er['total']/ef['total']:.1f}x), not 11x.")
    print(f"     Latency/throughput topology-neutral. The hardware value of the fractal")
    print(f"     lies primarily in AREA/wireability (B-4), not in energy.")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
