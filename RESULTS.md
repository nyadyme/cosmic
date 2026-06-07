# Cosmic-Fractal Model — Results & Interim Assessment

> **As of:** 2026-06-05 | **Status:** Milestone after Stages 1–5 + Goal-2 layout analysis
> **Purpose:** Evidence-based overall assessment. What is proven, what is refuted,
> what is open. Counterpart to the visionary `CONCEPT.md`.

---

## Core statement in three sentences

A neural network with cosmic-fractal topology works as a
reservoir computer — but it **does not compute better** than a generic
random reservoir of the same size (no cosmic computational advantage). Its real
value lies in **physical buildability**: ~11× shorter wiring and
95% local crossbar mappability make it suitable for analog hardware, where an
equivalent *globally* connected reservoir would be a wiring nightmare.
**Refinement:** This 11× advantage is primarily a **locality effect** (any
local network achieves it); the specifically *fractal* aspect contributes only
~1.3× on top of that. Thus the original thesis (Pillar 2: analog hybrid hardware
needs local wiring) is supported, while the "cosmic-magic" computational claim
is rejected.

---

## The four findings

### B-1 — Expressiveness (theoretical)
Qualitatively universal (sparsity not fatal: small-world, settling depth;
universality inherited from CT-RNN/reservoir/PC). Different inductive bias than
transformers: strong at predictive/streaming, weak at long-range retrieval.

### B-2 — Photonics rejected (compute), survives as interconnect
Photonic processors do not fit the recurrent/in-situ-learning design
(E/O/E latency per iteration, no optical memory/learning). Realistic
path: analog flash/memristor compute. Photonics only as a multi-chip interconnect.

### B-3 — Computationally: "cosmic" is neutral *(Goal 1)*
Controlled ablation, memory-forcing task (delayed copy, i.i.d.):
- **Coupling carries:** coupled reservoir (fractal/random) ≫ pure
  leaky integrators (≈ chance). The reservoir principle works.
- **"Cosmic" irrelevant:** fractal ≈ random-sparse (8-ensemble) at short
  memory, **worse** at long (LAG 8: 58% vs 99%). Memory Capacity
  fractal 5.14 vs random 5.47.
- *Lesson:* an earlier single run made fractal appear superior — an outlier,
  refuted by the ensemble. **Always check against an ensemble.**

### B-4 — Physically: "cosmic" is superior *(Goal 2)*
Hardware layout ablation, identical 3D positions:
- Total wire length: fractal **11.4× shorter** than random-global.
- Locality: 79.5% of edges local vs. 1.7% (random-global).
- REDAC-6-tile mapping: **4.7% inter-tile cut** (vs. 82.6% random-global)
  → **95.3% of fractal edges tile-local**, only **141 long connections** needed
  (= photonics interconnect niche, B-2).
- **Calibration (random-geometric control):** The 11× advantage is *primarily
  locality*; the fractal hierarchy gives only a *moderate* additional
  advantage on top (1.3× wire, ~1.7× fewer tile cuts) over local-random.
  *Remarkable (the most revealing detail):* random-geometric is even **more local**
  (86.2% of edges <0.2 vs. fractal 79.5%) — fractal still wires shorter and
  partitions more cleanly. The hierarchy/τ-gradient contribution is therefore real, albeit small.

---

### B-5 — Resolution scaling: refinement helps hardware, not computation
**Experiment (simulation_resolution.py):** Sweep over depth (n_levels 2/3/4) and
branching (eta 3/4/5), each measuring Memory Capacity + hardware + fractal-vs-random gap.

| Axis | Computational ability (MC) | Hardware |
|-------|---------------------|----------|
| Depth ↑ (L2→L3→L4) | 3.82 → 4.91 → 4.74 (**plateau from L3**, despite 4× nodes) | Wire advantage 3.9× → 7.2× → **11.3×**; tile cut 22.8% → **4.7%** |
| fractal vs random | random wins at **every** resolution (~−1 MC, gap never closes) | — |

**Finding:** Higher resolution has **opposite** effects on the two goals.
*Computational ability* saturates early (~L3) and never becomes fractal-specific (generic
scaling, B-3 confirmed at all scales). *Hardware suitability*, by contrast,
**improves with resolution** — the locality advantage compounds with
size (B-4 reinforced). Consequence: refining the generator (more depth, true
1D filaments, multifractality) pays off for **Goal 2**, not for Goal 1.

---

### B-6 — Goal-2 deep dive (REDAC mapping, filament, energy): refined
**Three strands** (simulation_redac / _filament / _energy.py):

| Strand | Finding |
|--------|--------|
| **1 REDAC mapping** | Full network (1530 nodes) **fits into 2 cascaded REDACs**. Bottleneck: integrators (1/node), driven by the ι node (ADR-12). **Balanced** tiling costs ~3× cut (4.7% unbal. → 13.9% bal., 421 inter-links) — the B-4 value was unbalanced. |
| **2 Filament refinement** | Subdividing edges into 1D segments **loses**: wire length invariant (physics), tile cut *worse* (13.9%→17.8%), node explosion (1530→6607, 2→8 REDACs). Useful refinement only at the *cluster* level (B-5), not at the wire level. |
| **3 Energy/latency** | 11× interconnect advantage → only **~1.1–1.2× total energy** (converters/quiescent current dominate 80–90%, G16). Latency topology-neutral (~1.3 µs/token). |

**Refined Goal-2 statement (revised vs. the B-4 headline):**
> The hardware value of the fractal lies in **routability/area**, not in energy
> or speed. An equivalent random reservoir would need 11× more wiring —
> possibly physically *unroutable*. The fractal is what makes analog construction
> **feasible and compact** in the first place. A real but *narrow* advantage — not a blanket efficiency win.
> **Design consequence:** ι (ADR-12) is hardware-expensive (drives the integrator bottleneck +
> 510 multipliers); since learning is not the lever anyway (B-3), ι is a
> deletion candidate for a pure inference reservoir → saves ~⅓ of the nodes.

---

### B-7 — Variant test: ι deletable, multifractal useless
**Experiment (simulation_variants.py):** Two architecture variants against the
reference topology (n_levels=4, eta=4, 1530 nodes), each measuring Memory Capacity
(delayed-copy ESN, LAGs 1–16) + hardware (wire, balanced tile cut, REDAC).

| Variant | Nodes | MC | Wire | Tile% | Mult | REDAC | Finding |
|----------|-------:|----:|------:|------:|-----:|------:|--------|
| **A** with ι (μ/ε/ι) | 1530 | 4.63 | 291 | 13.9% | 510 | 2 | Reference |
| **A** without ι (μ/ε) | 1020 | 4.61 | 291 | 20.2% | 0 | 2 | **−⅓ nodes, MC −0.4%** |
| **B** monofractal (eta=4) | 1530 | 4.64 | 291 | 13.9% | — | — | d_H 1.49 |
| **B** multifractal (eta 2..6) | 1977 | 4.67 | 362 | 15.1% | — | — | d_H 1.57 |

**Finding A (ι deletion):** Confirms the B-6 design recommendation **empirically**.
ι (ADR-12) costs 510 nodes + 510 multipliers, but delivers practically no
computational power (MC 4.63 → 4.61, −0.4%). For a **pure inference reservoir**
(learning is not the lever anyway, B-3), ι is **deletable → ⅓ less hardware**.
*Caveat:* the tile cut rises (13.9% → 20.2%) — the ι nodes loosened the
partition slightly. Routability marginally worse, but remains well manageable.

**Finding B (multifractal):** Confirms B-3 again. Heterogeneous branching
(eta 2..6 instead of fixed 4) yields **neither a computational (MC +0.03) nor a hardware advantage**
(tile +1.2 pp, more wire). Multifractality is not a lever.

---

### B-8 — REDAC tile balancing: strict balance was unnecessarily expensive
**Experiment (simulation_tile_balance.py):** Addresses open point #5. Question:
can the balanced tile cut (B-6: 13.9%) be lowered without violating the
REDAC capacity? Hardware model: each tile = one REDAC cluster
(crossbar) with ≤ **144 integrators** (864/6) and ≤ 72 multipliers.
Three partitioners (KMeans / Greedy-Cap / Greedy+FM refinement) + capacity-slack sweep.

| Topology | strict balance (B-6 Greedy) | + FM refine | **best HW-admissible (slack→144)** |
|-----------|------------------------------:|------------:|-------------------------------------:|
| ι-less (1020 nodes, μ/ε) | 20.2% (406 links) | 20.2% | **10.1%** (203 links, cap 131) |
| ι-full (1530 nodes) | 13.9% (421 links) | 13.8% | **10.6%** (322 links, cap 141) |

**Three findings:**
1. **FM refinement under strict balance ≈ ineffective** (−0.0 to −0.2 pp): at
   `cap = ideal` all tiles are full, no boundary node can be moved.
2. **The real lever is capacity slack.** A REDAC cluster holds 144
   nodes, but the ideal is only 85 (ι-less) or 128 (ι-full). This
   hardware headroom, converted into less cut, **nearly halves the
   inter-tile links** (ι-less 406→203, ι-full 421→322) — within the HW limit.
   The strict balance of the B-6 value was therefore unnecessarily pessimistic.
3. **FM adds a small extra on top** once slack is available
   (graph-aware refinement beats purely geometric KMeans on cut;
   ι-less reaches 10.1% < KMeans baseline 11.3%).

**Consequence:** The realistic inter-tile wiring requirement (photonics niche,
B-2) is **~10%** instead of the conservative 13.9% — for the ι-less inference topology
(B-7) only **203 long links** for the entire network. Routability (the actual
Goal-2 advantage, B-6) is thus even more favorable than thought.

---

### B-9 — Energy budget (SPICE proxy): advantage moderate, photonics uncritical
**Experiment (simulation_energy_spice.py):** Refines the parametric proxy
into an RC energy model on the **real** topology: node capacitance from the
τ-gradient (C = τ·G), wire energy from real edge lengths (die 10 mm), and —
new — the B-8 inter-tile links as a separate **photonic E/O/E load**. No
transistor SPICE, but a device-level analytical model with sensitivity sweep.

**Energy/token (ι-less, 1020 nodes):**

| Component | fractal | random-global | Share (fractal) |
|------------|--------:|--------------:|-----------------:|
| Analog compute | 4 pJ | 4 pJ | 0.0% |
| Wire local | 73 pJ | 829 pJ | 0.1% |
| Photonics E/O/E | 406 pJ | 3312 pJ | 0.8% |
| ADC/DAC | 10200 pJ | 10200 pJ | 20.6% |
| Quiescent current | 30600 pJ | 30600 pJ | 61.9% |
| Digital f/J | 8160 pJ | 8160 pJ | 16.5% |
| **TOTAL** | **49443 pJ** | **53106 pJ** | — |

**Three findings:**
1. **G16/B-6 confirmed on the real topology:** The 8.7× interconnect advantage
   shrinks to **1.07× total energy** — converters + quiescent current dominate
   topology-independently at **83%**. Robust across the sensitivity sweep (1.02×–1.24×).
2. **Photonics (B-8) does NOT erode the advantage:** The 203 inter-tile links cost
   only **0.8%** of the energy (random: 1656 links = 6.2%). *Counterintuitively:* with
   expensive photonics (1 pJ/bit) the fractal advantage **grows** to **1.24×** — the
   locality (few links) then pays off more strongly.
3. **ι deletion (B-7) saves ~33% energy/token** (49 → 74 nJ ι-full→ι-less;
   linear in node count, since converters + quiescent current scale with N).

**Latency:** ~3.25 µs/token (settle-dominated), ~308k tokens/s, topology-neutral.

**Consequence:** Finally confirms the revised Goal-2 statement — the hardware value
of the fractal is **routability/area**, **not energy** (~1.1×) or speed.

---

### B-10 — In-situ learning (ADR-8) fairly tested: no computational lever
**Experiment (simulation_insitu.py):** Closes open point #2. The
Stage-5 Hebbian numbers were artifacts of the broken Euler numerics. Here the
local ADR-8 rule (Hebbian + ADR-9 spring anchor) runs on the **repaired** leaky-ESN
numerics (B-3), on the existing edges (sparsity preserved), with scale control
+ spectral pin (learning **at** the edge of chaos, not in saturation). Measured: Memory
Capacity **fixed vs. learned**, for fractal **and** random (ensemble), η sweep.

| | fixed (η=0) | best learned | Δ |
|--|----------:|---------------:|--:|
| fractal | 4.885 | 4.916 (η=0.005) | +0.03 (noise) |
| random (ens.) | 5.540 | 5.540 (η=0) | +0.00 |

**Finding:** In-situ learning yields **no** MC gain. Not a numerics artifact:
at η=0.005 it runs cleanly at ρ≈0.95 → **neutral**; at η=0.05 Hebbian
self-amplification drives ρ→~2.5 → MC **drops**. The fixed reservoir is already
~optimal; the unsupervised local rule does not align with the memory goal
(the ridge readout does the work). Random > fractal here too (B-3 confirmed).

**Lesson:** The Stage-5 failure was **not only** due to the numerics — even repaired,
ADR-8 is no computational lever. *Caveat:* what is tested is the Hebbian+anchor rule; a
universal statement about *any* local learning rule this is not.

---

### B-11 — Application demo: the reservoir solves real edge tasks
**Experiment (simulation_demo_edge.py):** Demonstrates the *practical applicability* of the
reservoir (not as an LLM, but as a streaming edge reservoir computer, B-1) on
two canonical tasks with a linear ridge readout.

| Task | Metric | Result |
|---------|--------|----------|
| NARMA-10 (nonlinear time-series prediction) | NRMSE | 0.47 (fractal) / 0.39 (random) — both solve it |
| Vibration anomaly (predictive maintenance, residual-based) | ROC-AUC | **0.99** |

**Finding:** The reservoir solves streaming prediction and anomaly tasks well —
exactly its inductive profile (B-1). fractal ≈ random confirms B-3 again: the
benefit lies in analog buildability, not in the topology. Concretizes the
use cases (edge sensing, time series, predictive maintenance) and provides
a transferable hardware design guideline (see BERICHT/REPORT §5).

---

### B-12 — "Multiverse" approach: macro loops bring no computational advantage
**Experiment (simulation_multiverse.py):** Tests the idea of coupling several small reservoirs
("universes") via a few surface ports ("quasars") into a higher
hierarchy level. Fair comparison at equal node/edge count,
equal spectral radius, 3 seeds; Memory Capacity over short and long lags.

| Variant | MC total | long lags (≥16) |
|----------|----------:|-----------------:|
| flat (1 reservoir, N=600) | 3.62 | 0.634 |
| ensemble (6 blocks, no loops) | 3.59 | 0.614 |
| "multiverse" (6 blocks + macro chain) | **3.39** | 0.598 |

**Naive finding:** Macro loops bring **no** memory advantage (3.39 vs.
3.62/3.59) — input enters only one universe and is damped across the bridges.

**Variant test (simulation_multiverse2.py, revised):** *Properly configured*
(timescale gradient per universe + input everywhere + modular isolation), the
tuned "multiverse" **mv_tuned (MC 5.66)** even beats the strict fair baseline
**flat_match (4.73)** — a flat reservoir with an *identical* leak vector and input,
just globally instead of modularly wired. Three levers stack: timescale (+0.58), input/
gradient (+0.53), **modular connectivity (+0.93)**. Mechanism: **timescale-
aligned modular organization** (each timescale in its own densely
wired sub-reservoir).

**Conclusion:** "Exponential" remains **refuted**, but the hierarchy approach contributes
*moderately* and genuinely (~+20% over the fair flat baseline, ~+56% over naive) — real,
mechanistically explainable, no order-of-magnitude jump; capacity still scales roughly
with the hardware. **Recursion/scaling (simulation_multiverse3.py):** neither more
universes (K scales → MC saturates, B-5) nor a 3rd level (universes-within-universes,
+0.00 vs. 2 levels) compound — the modular ~+0.8-MC bonus is constant; the
2-level design is the upper bound. **Nonlinear counter-check (NARMA-10/20/30,
simulation_multiverse4.py):** the modular advantage generalizes (mv2<flat NRMSE
everywhere), but does NOT grow with the memory horizon — it even shrinks (−0.080
→ −0.021), no long-range-specific bonus. **Quasar count & I/O (simulation_multiverse5/
io.py):** MC is **flat across the bridge bandwidth Q** (Q=1..100 all ~5.7±0.13) →
the advantage comes from the timescale-aligned *organization*, not from sparse
bridges; **ideal quasar count = minimal (Q≈1)**, hardware-driven. External
I/O bandwidth: fan-in saturates at ~4 ports; more independent channels raise the
overall throughput sublinearly (shared N budget, decreasing per channel). **Geometry of
the upper level (simulation_multiverse6.py):** macro topology (chain/ring/lattice/random/
*cosmic*/all-to-all) at equal size/leak/input — MC **identical 5.67±0.13 across
all** (spread 0.00), including the upper level wired by fractal proximity. Third
independent confirmation of the mechanism (alongside B-3 micro and the Q sweep): MC is invariant
to the macro coupling pattern → the upper level can be chosen by physics (shortest
wiring), not by geometry. Remains a
reservoir, not an LLM. Details:
`MULTIVERSE.md`.

---

### B-13 — Real capacity levers: reservoir topology + external memory
**Experiment (simulation_capacity.py):** After structure (B-12) was exhausted,
two levers that **actually** raise capacity — in contrast to scaling/
hierarchy/learning (all neutral):

| Linear MC (N=100, max=N) | MC | % of N |
|----------------------------|---:|--------:|
| random (used so far) | 42.9 | 43% |
| orthogonal | 96.1 | 96% |
| **cycle (ring)** | **99.1** | **99%** |

| NARMA (NRMSE ↓) | res(N=100) | res+taps(D=20) | res(N+D=120) |
|-----------------|-----------:|---------------:|-------------:|
| NARMA-10 | 0.547 | **0.326** | 0.540 |
| NARMA-20 | 0.549 | **0.326** | 0.543 |

**Finding:** (1) A **ring/orthogonal** reservoir nearly exhausts the linear-memory
bound MC≈N (~96–99%), random/fractal only ~43% — the topology choice
**~doubles** the linear memory at equal node count, for free. (2) **20
digital delay taps** lower the NARMA error 0.55→0.33; 20 additional reservoir
nodes bring almost nothing (~0.54) → **external memory ≫ more nodes** (confirms
ADR-3). *Caveat (memory-vs-nonlinearity tradeoff):* the ring maximizes *linear*
memory, possibly at the cost of nonlinear capacity — for nonlinear tasks the
**combination** (heterogeneous reservoir for nonlinearity + taps for memory).
Consequence: the random/fractal reservoirs used in the project were suboptimal for
*linear* memory. Details: `MULTIVERSE.md` §9.

---

## Status of the two project goals

| Goal | Question | Status |
|------|-------|--------|
| **Goal 1** (practicability, software) | Does fractal topology work as an NN? | ✅ **yes** (reservoir works) — ⚠️ but no *specific* advantage over random-sparse, and in-situ learning (ADR-8) is no computational lever (B-10) |
| **Goal 2** (usability, hardware) | Buildable on analog hybrid? | ✅ **supported, but narrow** (B-6): fits into 2 REDACs; real advantage = **routability/area** (11× less wire → compact/buildable), **not** energy (~1.1×) or speed (neutral). Locality is the main lever. **Substrate:** 1 REDAC = one 19″ rack; full network = 2 REDACs ≈ 2 racks (B-6), i.e. cabinet-scale/stationary. Mobile/wearable only via an integrated ASIC (TRL 2, unbuilt — the ~10 mm/49 nJ numbers from B-9 describe this ASIC, not the REDAC). |

---

## What was built (artifacts)

| File | Content | Status |
|-------|--------|--------|
| `CONCEPT.md` | 12 ADRs, 4 findings, stage plan, open questions | current |
| `fractal_llm.md` | Mathematics (PC/free energy, corrected signs, ι dynamics) | current |
| `cosmic_web_generator.py` | Topology generator (μ/ε/ι, τ-gradient, radial) | ADR-12, tested |
| `simulation_v5.py` | Reservoir test toy grammar | 81.8% (delay+ridge) |
| `simulation_stage5*.py` | Language-Markov test + Hebbian | revised (artifacts identified) |
| `simulation_ablation*.py` | Goal-1 Memory-Capacity ablation | B-3 |
| `simulation_hardware*.py` | Goal-2 layout ablation | B-4 |
| `simulation_variants.py` | Goal-2 ι deletion + multifractal | B-7 |
| `simulation_tile_balance.py` | Goal-2 REDAC tile balancing (FM + slack) | B-8 |
| `simulation_energy_spice.py` | Goal-2 energy/latency budget (SPICE proxy) | B-9 |
| `simulation_insitu.py` | Goal-1 in-situ learning (ADR-8) fairly tested | B-10 |
| `simulation_demo_edge.py` | Application demo (NARMA-10, vibration anomaly) | B-11 |
| `simulation_multiverse.py` / `MULTIVERSE.md` | "Multiverse" approach (hierarchical coupling) | B-12 |
| `simulation_capacity.py` | Capacity levers (ring/orthogonal topology + delay taps) | B-13 |
| `simulation_multiverse5.py` / `simulation_io.py` | Quasar-count sweep + external I/O bandwidth | B-12 |
| `simulation_multiverse6.py` | Geometry of the upper level (macro-topology sweep) | B-12 |
| `STAGE2_FRACTALITY.md` | d_H measurement methodology (as a tool, comparison superseded) | repurposed |
| `BERICHT.md` / `REPORT.md` | Short report DE/EN (B-1…B-10, figures, AI disclosure) | publication |
| `ABSTRACT.md` | Submission abstract DE+EN (≤250 W) | publication |
| `REFERENCES.md` / `references.bib` | Reference list + BibTeX | publication |
| `make_figures.py` → `fig0/0b/0_edges/0c–3.png` | Figures from live runs (model render, 3 viewpoints incl. filament network + μ/ε/ι schematic) | publication |
| `make_circuit.py` → `circuit_*.png` | Electrical circuit diagrams (node, crossbar, μ/ε/ι signal chain, hybrid loop) | publication |
| `build_pdf.py` → `*.pdf` | PDF generation (markdown→xhtml2pdf, DejaVu Unicode) | publication |

---

## Open / not proven

1. **True sequence learning** (beyond reservoir + delay buffer) — not
   demonstrated. Differentiated: the **Stage-5** numbers were artifacts (Euler saturation,
   zero-fixed-point decode). The **Stage-4** number (81.8%) was a *valid* measurement,
   but its interpretation as a reservoir effect (rather than a delay-buffer+ridge trick)
   is unconfirmed.
6. **Hurwitz/closed-loop stability** of the nonlinear PC dynamics for learned θ
   (G27) — unverified both analytically and empirically.
2. ~~**In-situ learning (ADR-8)**~~ — **TESTED (B-10):** fairly on repaired
   leaky-ESN numerics → **no** MC gain (fixed reservoir ~optimal). ADR-8 as a
   computational lever not supported. The only thing still open: other local rules (Oja/BCM/IP).
3. **Nonlinear/structured tasks** where hierarchy might after all help
   computationally — unverified (caution: motivated search).
4. ~~**Real energy budget** (G16)~~ — **ADDRESSED as a proxy (B-9):** RC model
   on the real topology → ~1.07× total advantage, converters + quiescent current dominate 83%,
   photonics uncritical (0.8%). Real pJ/token on silicon (SPICE/REDAC) still open.
5. ~~**Tile load balance** for REDAC~~ — **ADDRESSED (B-8):** Greedy+FM with
   capacity slack up to the HW limit (144/tile) reaches ~10% cut at
   maintained balance. The only thing still open is the real SPICE/energy budget (#4).

---

## Methodological lessons (for credibility)

- **Ensemble instead of single run:** B-3 would almost have ended up as a false positive (fractal superior);
  the 8-graph ensemble corrected it. Distrust single-draw results.
- **Fair controls:** random-geometric vs random-global separates "locality" from
  "hierarchy" — without this control, B-4 would have overclaimed.
- **Check for artifacts before conclusions:** numerics bugs (Euler instability,
  zero-fixed-point decode) produced apparent results that measured nothing.

---

## Recommended next step

The questions answerable with software alone are now closed:
- **Goal 2 (hardware):** tile balancing (B-8) + energy/latency budget (B-9) →
  **routability/area** is the real advantage, **not** energy (~1.1×) or speed.
- **Goal 1 (computation):** in-situ learning (B-10) is **no** computational lever; the fixed
  reservoir is ~optimal, fractal ≈/< random (B-3 consistently confirmed).

The remaining open points need either **real silicon** (SPICE/REDAC,
transistor-level energy — leaves the software scope) or are
**research side paths** (other local learning rules Oja/BCM/IP; structured
tasks where hierarchy might after all help — caution: motivated search).

Recommendation: **consolidate/publish.** The core questions are answered with evidence;
the overall finding (fractal = buildable, not magic) stands. Further
software experiments would have diminishing marginal returns.
