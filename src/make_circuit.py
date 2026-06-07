#!/usr/bin/env python3
"""
Electrical circuit diagrams for the model (schemdraw).
================================================
circuit_node.png -- Analog realization of ONE PC node (Kirchhoff cell):
  Memristor weights W_cj bring currents from neighbor voltages into the node,
  integrator capacitor C (= τ·G) + leak/precision conductance G_eff(ι) to
  ground, DAC current source I_pred, ADC tap. Realizes
      C·dV_μ/dt = Σ_j W_cj·(V_j − V_μ) − G_eff·V_μ + I_pred.

Invocation:  python make_circuit.py
"""

import matplotlib
matplotlib.use("Agg")
import schemdraw
from schemdraw import elements as elm

DISCLAIMER = "AI-assisted -- please reproduce independently"


def circuit_node():
    d = schemdraw.Drawing(unit=2.4, fontsize=13)

    # --- central node + vertical busbar ---
    d += elm.Dot(radius=0.12).at((0, 0))
    d += elm.Line().at((0, 0)).up().length(1.6)          # node bus
    d += elm.Label().at((0.3, -0.05)).label(r"$V_\mu$", halign="left")

    # --- two memristor inputs (neighbor voltages) ---
    d += elm.Memristor().at((-3.4, 1.6)).right().length(2.6).label(r"$W_{c1}$")
    d += elm.Dot(open=True).at((-3.4, 1.6))
    d += elm.Label().at((-3.7, 1.6)).label(r"$V_1$", halign="right")
    d += elm.Line().at((-0.8, 1.6)).to((0, 1.6))

    d += elm.Memristor().at((-3.4, 0.7)).right().length(2.6).label(r"$W_{c2}$")
    d += elm.Dot(open=True).at((-3.4, 0.7))
    d += elm.Label().at((-3.7, 0.7)).label(r"$V_2$", halign="right")
    d += elm.Line().at((-0.8, 0.7)).to((0, 0.7))

    # --- shunt: integrator capacitor C to ground ---
    d += elm.Capacitor().at((0, 0)).down().length(2.4).label(r"$C=\tau G$", loc="left")
    d += elm.Ground().at((0, -2.4))

    # --- parallel leak/precision conductance G_eff(ι) ---
    d += elm.Line().at((0, 0)).right().length(1.8)
    d += elm.ResistorIEC().at((1.8, 0)).down().length(2.4).label(r"$G_{eff}(\iota)$", loc="right")
    d += elm.Line().at((1.8, -2.4)).to((0, -2.4))

    # --- DAC current source I_pred (prediction, top-down injected) ---
    d += elm.Line().at((0, 0)).right().length(3.6)
    d += elm.SourceI().at((3.6, -2.4)).up().length(2.4).label(r"$I_{pred}$", loc="right")
    d += elm.Ground().at((3.6, -2.4))
    d += elm.Label().at((3.6, 0.45)).label("DAC", halign="center", fontsize=10)

    # --- ADC tap (read state) ---
    d += elm.Line().at((0, 1.6)).up().length(0.7)
    d += elm.Arrow().at((0, 2.3)).up().length(0.5)
    d += elm.Label().at((0, 3.05)).label("ADC (read)")

    # --- title + equation + disclaimer ---
    d += elm.Label().at((1.6, 3.9)).label(
        "Analog node cell (μ): Kirchhoff summation", halign="center")
    d += elm.Label().at((1.6, -4.0)).label(
        r"$C\,\dot V_\mu = \sum_j W_{cj}(V_j-V_\mu) - G_{eff}V_\mu + I_{pred}$",
        halign="center")
    d += elm.Label().at((1.6, -4.8)).label(DISCLAIMER, halign="center",
                                            fontsize=8, color="0.45")
    d.save("figures/circuit_node.png", dpi=150, transparent=False)
    print("  circuit_node.png")


def circuit_crossbar():
    """Memristor crossbar tile: W=G as crossbar, I_j = Σ_i G_ij V_i."""
    d = schemdraw.Drawing(unit=1.0, fontsize=13)
    rows = [6.0, 4.5, 3.0]
    cols = [2.0, 4.0, 6.0]
    for i, y in enumerate(rows):                      # rows = input voltages
        d += elm.Line().at((0.6, y)).to((7.0, y)).color("gray")
        d += elm.Dot(open=True).at((0.6, y))
        d += elm.Label().at((0.1, y)).label(rf"$V_{{{i+1}}}$", halign="right")
    for x in cols:                                    # columns = summed-current lines
        d += elm.Line().at((x, 6.7)).to((x, 1.4)).color("gray")
    for y in rows:                                    # memristor at each crossing point
        for x in cols:
            d += elm.Memristor().at((x - 0.5, y - 0.5)).theta(45).length(1.4)
            d += elm.Dot(radius=0.07).at((x, y))
    for j, x in enumerate(cols):                      # column outputs -> ADC/integrator
        d += elm.Arrow().at((x, 1.4)).down().length(0.8)
        d += elm.Label().at((x, 0.3)).label(rf"$I_{{{j+1}}}$")
    d += elm.Label().at((3.5, 7.4)).label(
        "Memristor crossbar tile (one REDAC cluster)", halign="center")
    d += elm.Label().at((3.5, -0.6)).label(
        r"$I_j=\sum_i G_{ij}\,V_i$   (Ohm/Kirchhoff; weight $W\!=\!G$)",
        halign="center")
    d += elm.Label().at((3.5, -1.3)).label(DISCLAIMER, halign="center",
                                            fontsize=8, color="0.45")
    d.save("figures/circuit_crossbar.png", dpi=150, transparent=False)
    print("  circuit_crossbar.png")


def _box(ax, x, y, w, h, text, fc="#eaeaf2"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02", fc=fc, ec="0.25", lw=1.3, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, zorder=4)


def _arrow(ax, p, q, color="0.2", rad=0.0, label=None, lx=0, ly=0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color, zorder=2))
    if label:
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.5, color="0.15")


def circuit_triplet_opamp():
    """μ/ε/ι cluster as an analog signal chain (op-amp level)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(-3.4, 3.0); ax.axis("off")

    _box(ax, 1.6, 1.4, 2.4, 1.2, "Difference op-amp\n$\\varepsilon = V_\\mu - \\hat{V}$",
         fc="#d9e4f0")
    _box(ax, 4.8, 1.4, 2.0, 1.0, "Squarer\n$(\\cdot)^2$", fc="#f0e0e0")
    _box(ax, 7.8, 1.4, 2.4, 1.2, "ι-integrator\n$C_\\iota,\\,G_\\iota \\to V_\\iota$",
         fc="#e0efe0")
    _box(ax, 10.4, 1.4, 2.2, 1.2,
         "Gain\n$G_{eff}=G_s/(V_\\iota+\\varepsilon_0)$", fc="#e0efe0")
    _box(ax, 7.8, -1.6, 3.0, 1.2,
         "μ-integrator (op-amp+$C_\\mu$)\n$C_\\mu\\dot V_\\mu = -G_{eff}\\,\\varepsilon$",
         fc="#d9e4f0")

    _arrow(ax, (2.8, 1.4), (3.8, 1.4))                       # diff -> squarer
    _arrow(ax, (5.8, 1.4), (6.6, 1.4))                       # squarer -> iota
    _arrow(ax, (9.0, 1.4), (9.3, 1.4))                       # iota -> gain
    _arrow(ax, (10.4, 0.8), (9.0, -1.0), label="$G_{eff}$ (gain modulation)",
           lx=10.9, ly=-0.2)
    # epsilon branch (from the diff->squarer line) into the mu integrator
    _arrow(ax, (3.3, 1.1), (6.5, -1.6), color="#a05050",
           label="$\\varepsilon$", lx=4.6, ly=-0.7)
    # V_mu output + feedback into the difference
    _arrow(ax, (9.3, -1.6), (11.2, -1.6))
    ax.text(11.4, -1.6, r"$V_\mu$", fontsize=12, va="center")
    _arrow(ax, (11.3, -2.1), (1.6, -2.6), color="0.45", rad=-0.06)
    _arrow(ax, (1.6, -2.6), (1.6, 0.8), color="0.45",
           label="$V_\\mu$ (feedback)", lx=1.0, ly=-1.4)
    # V_hat input (DAC) into the difference
    _arrow(ax, (1.6, 2.6), (1.6, 2.0), color="#5b4a86")
    ax.text(1.6, 2.8, "$\\hat{V}$ (DAC, top-down)", ha="center", fontsize=8,
            color="#5b4a86")
    # ADC tap from V_mu
    _arrow(ax, (10.6, -1.6), (10.6, -2.8))
    ax.text(10.6, -3.05, "ADC (read)", ha="center", fontsize=8)

    ax.set_title("μ/ε/ι cluster — analog signal chain (ADR-12)", fontsize=11)
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=7, style="italic",
             color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/circuit_triplet.png", dpi=150)
    plt.close(fig)
    print("  circuit_triplet.png")


def circuit_hybrid():
    """Hybrid loop: analog ↔ digital per token (ADC/DAC, ADR-4)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

    _box(ax, 2.4, 6.2, 3.6, 1.8,
         "Analog core\nY = L_W + diag(G)\nMemristor crossbar\nC·dV/dt = −Y·V + I_pred",
         fc="#d9e4f0")
    _box(ax, 7.4, 6.2, 2.6, 1.4, "ADC\nread V, ΔV", fc="#efe8d8")
    _box(ax, 7.4, 2.0, 3.0, 1.8,
         "Digital (state machine)\nf, J, precision G\nLUT, O(k) — no O(n²)", fc="#e6e0ee")
    _box(ax, 2.4, 2.0, 2.6, 1.4, "DAC\nwrite $\\hat{V}$, J, G", fc="#efe8d8")

    _arrow(ax, (4.2, 6.2), (6.1, 6.2))                       # core -> ADC
    _arrow(ax, (7.4, 5.5), (7.4, 2.9))                       # ADC -> digital
    _arrow(ax, (5.9, 2.0), (3.7, 2.0))                       # digital -> DAC
    _arrow(ax, (2.4, 2.7), (2.4, 5.3))                       # DAC -> core
    ax.text(5.1, 6.5, "analog→digital", ha="center", fontsize=7.5, color="0.3")
    ax.text(5.0, 1.7, "digital→analog", ha="center", fontsize=7.5, color="0.3")
    ax.text(5.0, 4.1, "1 token =\n1 settling cycle\n(~3.25 µs)", ha="center",
            fontsize=8, color="0.25")

    ax.set_title("Hybrid loop: analog ↔ digital per token (ADR-4)", fontsize=11)
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=7, style="italic",
             color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/circuit_hybrid.png", dpi=150)
    plt.close(fig)
    print("  circuit_hybrid.png")


def main():
    print("Generating circuit diagrams (schemdraw + matplotlib) ...")
    circuit_node()
    circuit_crossbar()
    circuit_triplet_opamp()
    circuit_hybrid()
    print("Done: circuit_node, circuit_crossbar, circuit_triplet, circuit_hybrid")


if __name__ == "__main__":
    main()
