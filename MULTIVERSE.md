# "Multiverse" approach: recursive scaling of the fractal reservoir


> **⚠️ AI-assisted work.** The idea, code, experiment, evaluation and this text
> were produced with AI assistance (Anthropic Claude) under human direction.
> The results are exploratory software simulations — not peer-reviewed; reproduce
> them independently before citing.

---

## 1. The idea

The single fractal reservoir ("universe", a sphere with μ/ε/ι clusters) hits limits
when scaled: pressing more nodes in flatly makes the long links — and hence the
wiring — explode. The proposed way out applies self-similarity to the **next scale
level**:

- Each universe stays locally connected and routable on the inside.
- A few **surface ports ("quasars")** couple neighbouring universes.
- This creates a higher hierarchy level with **macro-filaments / macro-loops**.
- Physically attractive: quasars as micro-LEDs/lasers → optical coupling to the
  neighbouring sphere (photodiodes), a 3D-stacked, optically coupled cluster.

## 2. Classification (before the test)

**Plausible / feasible:**
- **Divide-and-conquer** numerically correct: many small, locally settling matrices
  in parallel instead of one large stiff one — well parallelisable.
- **Modularity** = the chiplet principle in 3D; intra-module local/routable (B-4/B-8).
- **Optical quasar coupling** = exactly the photonic interconnect niche (B-2/B-8).

**Overstated (against our own findings):**
- "Memory capacity grows exponentially" — **not supported**, contradicts B-5
  (saturation/plateau) and B-3 (no topology computation advantage). Expectation:
  capacity grows roughly **with the hardware**, not exponentially.
- "solves the routing problem" — it **structures** it; the inter-module links remain
  the expensive photonic load.
- Core verdict unchanged: a **reservoir**, not a learning language model.
- Macro-loops = feedback → **stability risk** (open Hurwitz question G27, now on two
  levels).

Classification: The "multiverse" is a known paradigm — **hierarchical / deep
reservoir computing** (reservoir-of-reservoirs) + **chiplets**. For this there are
*moderate, documented* multi-timescale gains — no exponential growth.

## 3. The test (fair baseline)

Instead of believing, measured — with a fair control (lesson from B-3). Three
variants at the **same node count N=600 (6×100), ~same edge count, same spectral
radius (edge of chaos), same leak rate**:

| Variant | Setup |
|----------|--------|
| **flat** | ONE connected reservoir, N nodes, input to V nodes |
| **ensemble** | K independent blocks (NO macro-links), input to ALL blocks |
| **multiverse** | K blocks + macro-chain (Quasar_k → Input_{k+1}), input only to block 0 |

Metric: memory capacity over short **and long** lags (the claimed advantage would be
*long* memory). Averaged over 3 seeds. Intra-topology random-sparse — since the
fractal topology is computationally neutral (B-3), this isolates the pure
macro-hierarchy effect at exactly the same size.

**Result (3 seeds):**

| Variant | MC total | long lags (≥16) |
|----------|----------:|-----------------:|
| flat | 3.62 ± 0.06 | 0.634 |
| ensemble | 3.59 ± 0.08 | 0.614 |
| multiverse | 3.39 ± 0.03 | 0.598 |

![Figure](figures/fig_multiverse.png)

*Fig. Memory per lag (averaged over 3 seeds). The three curves are almost
coincident; the "multiverse" lies minimally *below* flat and ensemble across all
lags.*

## 4. Finding (naive variant)

**In the naive form the macro-loops bring no memory advantage — on the contrary,
slightly less.** Neither in total (3.39 vs. 3.62/3.59) nor at long lags (0.60 vs.
0.63/0.61). Reason: the input enters only one universe and is damped across the
spectrally scaled bridges; the distant universes carry an attenuated signal → their
readout contribution is more like noise. "Exponential" is thereby clearly
**refuted**; consistent with B-3 and B-5.

**But:** that was a single, unoptimised design point. Configured correctly it looks
different — see **§6**.

## 5. Limits of this test / the tuned variants

What was tested (§3/§4) is **one** design point (chain topology, input to block 0,
fixed leak rate). Theoretically helpful would be:
- **Timescale gradient per universe** (slow core universes, fast edge universes),
- **Input to all universes + bridges** (instead of only block 0),
- a learned macro-readout / different macro-topology.

We tested these variants (§6) — with a **fair** caveat: a timescale gradient lifts
*any* reservoir; only beating a flat reservoir of **the same size, same timescale and
same input** demonstrates a genuine hierarchy contribution.

## 6. Variant test: how far can it be pushed?

`simulation_multiverse2.py` tests the tuned variants against fair baselines of the
same size. Decisive is the control **flat_match**: a flat reservoir with an
*identical* leak vector and *identical* input fan-in to the tuned "multiverse" — the
only difference is connectivity (modular + bridges vs. globally connected).

| Variant | MC total | long lags (≥16) |
|----------|----------:|-----------------:|
| flat (fixed leak) | 3.62 | 0.634 |
| flat_tau (timescale only, 6 inputs) | 4.20 | 0.619 |
| flat_match (= mv_tuned, but flatly connected) | 4.73 | 0.603 |
| ensemble_tau (blocks, no loops) | 4.50 | 0.619 |
| **mv_tuned (modular + timescale gradient)** | **5.66** | **0.731** |

![Variant test](figures/fig_multiverse2.png)

*Fig. Memory per lag. The tuned "multiverse" (green) holds substantially more memory
in the mid-lag range (3–16) than all baselines of the same size.*

**Three levers stack up:** timescale spread (+0.58), input fan-in + gradient (+0.53)
and — the multiverse-specific one — the **modular connectivity (+0.93)**.

**Revised finding:** Unlike the naive variant, the *tuned* "multiverse" brings a
**genuine, moderate advantage**: +0.94 MC (~+20 %) over the fair flat counterpart
`flat_match`, plus the best long memory of all variants; overall +56 % over the naive
flat reservoir.

**Probable mechanism (hypothesis):** The modular connectivity **isolates the slow
universes** (small leak = long memory) from the bombardment by fast dynamics. In the
globally mixed flat reservoir the slow nodes get "cluttered" by fast signals; the
bridges transmit just enough to chain the timescales. This matches the deep reservoir
computing literature. *(Made more precise in §10: the effect is robust against bridge
width — what is decisive is the timescale-aligned **organisation**, not the
**isolation** of thin bridges.)*

**Limits:**
- Moderate, **not exponential**: ~+20 % over the fair baseline, no order-of-magnitude
  jump. Capacity still scales roughly with the hardware.
- A toy (600 nodes), one task (linear MC), one regime — generality untested.
- What is tested is a **forward chain** (stable); a true feedback ring coupling would
  bring back the stability question (G27).
- Changes nothing about the core: a reservoir, not a learning language model.

**Conclusion:** The original intuition was not wrong. *Configured correctly*
(timescale gradient + modular isolation + distributed input) the "multiverse"
structure contributes **measurably** — an improvement of factor ~1.2 over the fair
flat reservoir, ~1.6 over the naive one. Real and mechanistically explainable, but
**moderate**, no exponential miracle.

## 7. Recursion (universes-in-universes) and scaling K

`simulation_multiverse3.py` checks whether the 2-level advantage can be pushed
further: by (a) **more universes** (scaling K) or (b) a **third level**
(universes-in-universes) — both against fair flat baselines of the same
size/edges/leak.

**Part 1 — scaling K (module size 100, 2 levels):**

| K | N | flat MC | mv2 MC | lead |
|---|----:|--------:|-------:|----------:|
| 3 | 300 | 4.83 | 5.66 | +0.83 |
| 6 | 600 | 4.81 | 5.67 | +0.87 |
| 9 | 900 | 4.90 | 5.72 | +0.82 |
| 12 | 1200 | 5.03 | 5.75 | +0.72 |

→ MC **saturates** (B-5): 4× nodes (300→1200) raise MC by only ~2–4 %. The modular
lead remains a ~**constant bonus (~+0.8)** and even erodes slightly at large K. More
universes ≠ proportionally more memory.

**Part 2 — depth (N=720, 12 modules, same edge count):**

| Variant | MC |
|----------|---:|
| flat (1 level) | 4.62 |
| mv2 (2 levels) | 5.48 |
| mv3 (3 levels, 3 groups of 4) | 5.48 |

→ The **third level brings nothing** (+0.00 vs. 2 levels). The 2-level structure
already exhausts the timescale-isolation effect; deeper nesting does not compound.

![Recursion & scaling](figures/fig_multiverse3.png)

*Fig. Left: MC vs. node count (saturation; constant modular lead). Right: 2 vs. 3
levels at the same size (on par).*

**Combined — answer to "can these be combined?":** *Mechanically yes* (mv3 **is** the
combination of recursion and several modules, K is scalable at each level). *But*
**neither lever compounds**: K-scaling saturates, the third level is redundant. The
tuned **2-level "multiverse" is practically the ceiling** of this mechanism — a
moderate, constant ~+0.8-MC bonus, **no** exponential and **no** scaling growth.
(Holds for the linear-MC task in this regime; other tasks/learning objectives
untested — beware of motivated searching.)

## 8. Counter-check: nonlinear task (NARMA-10/20/30)

`simulation_multiverse4.py` checks whether the hierarchy is more useful on a
**nonlinear** task with a growing memory horizon than on the linear MC — because
*there* a structural multi-timescale benefit would most likely be expected. NARMA-n
(standard benchmark), NRMSE (smaller = better), fair control (N=720, same
edges/leak/input).

| Order | flat | mv2 | mv3 |
|---------|-----:|----:|----:|
| NARMA-10 | 0.542 | 0.461 | 0.460 |
| NARMA-20 | 0.578 | 0.523 | 0.525 |
| NARMA-30 | 0.694 | 0.673 | 0.673 |

![NARMA counter-check](figures/fig_multiverse4.png)

*Fig. NRMSE per NARMA order. mv2/mv3 lie below flat everywhere (hierarchy helps), but
the gap shrinks with the order.*

**Two findings:**
1. The modular advantage **generalises** to the nonlinear task: mv2 has smaller NRMSE
   than flat at all orders.
2. But it **does NOT grow with the horizon** — on the contrary, it shrinks (−0.080,
   −0.054, −0.021 NRMSE). At a long horizon all of them saturate near the trivial
   predictor (~0.67–0.69); there the overall capacity binds (B-5), not the
   organisation. The third level stays redundant (mv3 ≈ mv2).

**Conclusion (refutes the hypothesis):** The expectation "hierarchy is *especially*
useful for long multi-timescale memory" is **not** confirmed — the advantage is
largest at short/medium horizon and fades at long horizon. The modular timescale
isolation is a **general, moderate** lever (also usable nonlinearly), but **no** key
to long memory, and it compounds neither with depth (§7) nor with horizon. The
overall picture remains: real, moderate, capped — no exponential miracle.

## 9. How do you actually raise the capacity?

Since structure (§6–§8) is exhausted, `simulation_capacity.py` checks the two levers
that **actually** raise the capacity — both independent of the "multiverse" idea.

**(1) Reservoir topology — linear memory capacity (N=100, max = N):**

| Reservoir | MC | % of N |
|-----------|---:|--------:|
| random (used so far) | 42.9 | 43 % |
| orthogonal | 96.1 | 96 % |
| **cycle (ring)** | **99.1** | **99 %** |

**(2) External digital memory (delay taps) on NARMA (NRMSE ↓):**

| Task | res(N=100) | res+taps(D=20) | res(N+D=120) |
|---------|-----------:|---------------:|-------------:|
| NARMA-10 | 0.547 | **0.326** | 0.540 |
| NARMA-20 | 0.549 | **0.326** | 0.543 |

![Capacity levers](figures/fig_capacity.png)

**Finding:** (1) A **ring** almost exhausts the theoretical maximum MC≈N (99 %),
random/fractal only ~43 % — the topology choice **~doubles** the linear memory for
free. (2) **20 delay taps** lower the NARMA error 0.55→0.33; 20 additional reservoir
nodes bring almost nothing — **external memory ≫ more nodes** (ADR-3).

**Caveat & synthesis:** The ring maximises *linear* memory (memory-vs-nonlinearity
tradeoff). The efficient overall solution is therefore: **heterogeneous reservoir**
(nonlinearity, + the modular timescale isolation from §6) **plus a digital delay
buffer** (long memory). More nodes or more hierarchy are **not** the lever; topology
+ external memory are. It remains a reservoir, not an LLM — but this is how you get
the maximum out of it.

## 10. Quasar count & external I/O bandwidth

Two detail questions about the interface (`simulation_multiverse5.py`,
`simulation_io.py`).

**(a) How many quasars between universes?** Sweep over Q (bridge ports per
connection; 2 levels, N=600, 3 seeds; flat_match reference 4.73):

| Q | 1 | 4 | 16 | 64 | 100 |
|----|---:|---:|---:|---:|---:|
| MC | 5.67 | 5.67 | 5.69 | 5.73 | 5.73 |

(Spread ±0.13.)

![Quasar sweep](figures/fig_multiverse5.png)

→ MC is **flat** over Q (range +0.06 ≪ spread ±0.13); even Q=100 (all surface nodes)
does not collapse towards flat. **Mechanism correction to §6:** the advantage comes
**not** from thin bridges/isolation, but from the **timescale-aligned modular
organisation** (each timescale in its own, densely connected sub-reservoir) — robust
against bridge width. **Ideal quasar count for the coupling = minimal (Q≈1)**, driven
by the photonics cost (B-2/B-9), not by capacity.

**(b) External I/O bandwidth:**

| Part A — 1 stream, P ports | P=1 | 4 | 16 | 64 |
|---------------------------|----:|---:|---:|---:|
| MC | 17.9 | **19.5** | 18.7 | 18.1 |

| Part B — C channels (8 ports each) | C=1 | 2 | 4 | 8 |
|--------------------------------|----:|---:|---:|---:|
| MC total | 19.1 | 33.9 | 59.3 | 86.1 |
| MC/channel | 19.1 | 17.0 | 14.8 | 10.8 |

![I/O bandwidth](figures/fig_io.png)

→ **Fan-in** (one stream): optimum at **few** ports (P≈4), then a slight drop
(over-distribution). **Multi-channel:** total capacity grows with C (19→86), but
**sublinearly**, and **per channel it falls** (19→11) — the fixed reservoir capacity
is shared (Dambre bound ≤ N). Notable: a single scalar stream barely uses the
reservoir (MC≈19 of ≤600) — **more independent channels unlock the latent capacity**,
at the cost of per-channel fidelity.

**Conclusion:** Keep quasars for the *coupling* minimal (MC-neutral, saves photonics);
**external** I/O throughput capacity scales over **multiple independent channels** (up
to the N ceiling), not over many ports for *one* stream.

## 11. Geometry of the upper level (macro-topology)

Up to here the *upper* level — the wiring **between** the universes — was a trivial
**chain** (k→k+1). The obvious question: does the capacity change if the upper level
itself gets a real geometry — for instance wired again *cosmically/fractally* by
spatial proximity? Controlled sweep (N=600, K=6 universes, 3 seeds), everything fixed
(graded leak, input everywhere, Q=4 ports per macro-edge each), varying only the
**macro-wiring pattern**:

| Macro-topology | macro-edges | MC |
|-----------------|:------------:|:----:|
| chain (so far) | 5 | **5.67 ± 0.13** |
| ring | 6 | 5.67 ± 0.12 |
| lattice (2×3 grid) | 7 | 5.67 ± 0.13 |
| random | 5 | 5.67 ± 0.13 |
| **cosmic** (3D proximity, fractal) | 12 | 5.67 ± 0.13 |
| alltoall | 15 | 5.67 ± 0.12 |
| *flat_match (reference)* | – | *4.73* |

![Geometry of the upper level](figures/fig_multiverse6.png)

**Range across all geometries: 0.00** — the bars are identical. Even the upper level
wired fractally by proximity lies exactly on chain and on all-to-all.

**Finding:** The "multiverse" advantage (+0.94 MC over flat_match) **remains** robust
but does **not** depend on the macro-coupling pattern. This is the **third independent
confirmation of the same mechanism**: cosmic topology brings no computation advantage
at the micro-level (B-3), MC is invariant against the bridge bandwidth (§10) — and now
invariant against the macro-geometry. What counts is solely the **timescale-aligned
modular organisation**; *how* the modules are coupled — count, bandwidth, geometry —
is irrelevant to the capacity.

**Consequence for the design:** The upper level can be chosen by purely **physical**
criteria (shortest wiring, minimal quasar bridges, routing effort) — a rich geometry
only costs complexity without capacity gain. A simple chain/ring suffices.

## Reproducibility

`simulation_multiverse.py` (naive test, `fig_multiverse.png`),
`simulation_multiverse2.py` (variant test, `fig_multiverse2.png`) and
`simulation_multiverse3.py` (recursion + K-scaling, `fig_multiverse3.png`) and
`simulation_multiverse4.py` (nonlinear NARMA counter-check, `fig_multiverse4.png`)
`simulation_capacity.py` (§9 capacity levers, `fig_capacity.png`),
`simulation_multiverse5.py` (§10 quasar sweep, `fig_multiverse5.png`),
`simulation_io.py` (§10 I/O bandwidth, `fig_io.png`) and
`simulation_multiverse6.py` (§11 macro-geometry, `fig_multiverse6.png`) —
deterministic (fixed
seeds), CPU, using the leaky-ESN mechanics from `simulation_insitu.py`. Main findings of the project: `RESULTS.md`, report:
`BERICHT.md` / `REPORT.md`.
