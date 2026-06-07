# Cosmos-Fractal Model — Master Concept

> **Status:** Foundation (v1)
> **Date:** 2026-06-04
> **Purpose:** This document records the complete vision. It is the
> top-level source of truth. Detailed derivations and code live in separate
> files ("the warehouse") and are only referenced here, not duplicated.

---

## 0. The Idea in One Sentence

A learning-capable model whose **topology** does not consist of flat layers, but
of a **three-dimensional fractal patterned after cosmic filaments** —
galaxy clusters as compute nodes, filaments as synaptic paths, voids as
skipped emptiness — which learns via **Predictive Coding** and runs on
**analog-digital hybrid hardware**.

The idea behind it: In nature, patterns recur **fractally**. Images
of neuron and synapse accumulations (nuclei) show a striking
morphological **similarity** to the clusters of galaxy clusters at the
nodes of cosmic filaments. This project takes that similarity seriously
and asks: Can it be not merely observed, but also made **computationally usable**?

---

## 1. Motivation — Why This Path?

Conventional LLMs (Transformers) have three structural problems that this
approach addresses:

| Problem of conventional LLMs | Answer of the Cosmos-Fractal approach |
|------------------------------|-------------------------------------|
| Dense matrices → billions of multiplications with zeros | Fractal sparsity: voids consume no compute time |
| Backpropagation is biologically implausible and global | Predictive Coding: local learning, only error flows |
| Digital hardware is inefficient at sparse operations | Analog hardware: Kirchhoff/Ohm compute "for free" in physics |
| Architecture is human-designed, arbitrary | Topology derived from a natural principle (self-similar, hierarchical) |

The guiding assumption: **The large-scale structure of the universe and the
structure of the brain are both hierarchical, self-similar, sparse networks.** If you
take this geometry as a template, you get efficiency and learning capability
"for free" instead of forcing them.

With this, the project pursues **two overarching goals**, against which every
milestone is measured:

- **Goal 1 — Practicability:** Is it even feasible to operate a neural network with
  fractal topology instead of a "conventional" LLM architecture —
  and does it deliver usable results? **Benchmark (from finding B-1):** *not*
  Transformer parity at long-range retrieval, but strength in the
  PC/streaming niche (predictive, low-energy). Compare also against RNN/SSM/
  Reservoir, not only against Transformers.
- **Goal 2 — Usability on hybrid hardware:** The observed similarity
  between neuronal and cosmic clusters/filaments is *probably only a
  subjective impression or coincidence* — and it doesn't even need to be "real".
  The question is: Can this fractal-inspired design principle be **used** to
  construct learning-capable LLMs on **analog-digital hybrid architectures**?
  Goal 2 asks **not about the truth of the analogy, but about its
  technical utility** (hardware realization, stages 6/7).

**Delineation of the two goals (so they don't blur together):**
- **Goal 1 = algorithmic practicability.** Does the fractal topology
  even work as a learning-capable NN? Tested *in software simulation*, measured against
  RNN/SSM/Reservoir.
- **Goal 2 = hardware-side usability.** Can the whole thing be built as a learning-capable
  model on *analog-digital hybrid hardware*? Tested on REDAC/memristor
  (stages 6/7). The "reality" of the cosmic-neuronal fractality is
  **irrelevant** for this — only the practical utility counts.

---

## 2. The Four Pillars

### Pillar 1 — Topology: The Cosmic Fractal

- The network structure is derived from the **cosmic web**: galaxy clusters
  cluster at the nodes of filaments; empty voids lie between them.
- This structure is **scale-invariant / fractal** — Hausdorff dimension `d_H ≈ 2`.
- Interpreted as a **heatmap** (3D density distribution):
  - **Dense nodes** = artificial neurons / attention heads (parameters concentrated)
  - **Filaments** = synaptic connections / logical paths
  - **Voids** = skipped in the model → sparse matrices
- Generation: Soneira–Peebles algorithm (recursive cluster decomposition) or
  DLA / 3D simplex noise.

### Pillar 2 — Hardware: Analog-Digital Hybrid Computer

- **Analog part:** The filament structure is rebuilt as a physical network of
  resistors / memristors. Computation happens instantaneously through the
  **Ohm's law** and the **Kirchhoff rules**.
  Voids = no current = no energy consumption.
- **Digital part:** Token management, text↔voltage translation, control of the
  slow components (nonlinearity, precision), symbol output.
- **Building blocks:**
  - Memristors / in-memory computing (weights = resistances, change locally)

> **DISCARDED — Photonic processors (finding B-2, 2026).** Photonics does not fit
> this architecture: (1) iterative settling (ADR-1) pays the
> electro-optical conversion overhead (~14 %) **per iteration** → latency
> multiplies; (2) **no optical memory** → conflict with ADR-3;
> (3) **no optical in-situ learning** as with the memristor → conflict with ADR-8;
> (4) nonlinearity requires an electronic detour. Industry signal: leading
> vendors (Lightmatter, Celestial AI) have pivoted from optical *computing* to optical
> *connecting*. **Only remaining open option:** photonics as an *interconnect*
> in later multi-chip scaling — not as a compute substrate.
>
> **Realistic hardware path (2026):** FPGA/SPICE proxy → analog
> flash compute (e.g. Mythic M1076, 25 TOPS) → research memristor crossbars.
> Related work: memristor-crossbar reservoir computing (iScience 2026) —
> a direct cousin (ADR-1).

### Pillar 3 — Learning Method: Predictive Coding

- **Why Predictive Coding (PC)?** It is the **most brain-like** model of
  information processing in the cortex.
- Principle: Higher levels **predict** the activity of lower levels.
  Only the **prediction error** (the residual) is propagated.
- Consequence for the hardware:
  - Expected text → error ≈ 0 → almost no current flow → extremely energy-saving
  - Unexpected word → error signal shoots through filaments → that is "attention"
- **Learning is local:** memristors change their resistance directly on site,
  based on the applied voltage. No global gradient computation.
- *Related, discarded alternatives:* Equilibrium Propagation, Forward-Forward,
  Contrastive Hebbian Learning — all local, but biologically less exact than PC.
  (Kept in mind as a fallback in case PC fails on hardware.)

### Pillar 4 — Mathematics: Free Energy Principle (Friston)

Three core equations (details in the warehouse, see section 5):

1. **Prediction error** `ε = μ − f(μ_higher, θ)` → physically: voltage difference
2. **Inference dynamics** `dμ/dt = −∂F/∂μ` → physically: Kirchhoff current balancing
3. **Learning rule** `dθ/dt = η · ε · ∂f/∂θ` → physically: memristor resistance change

Variable dictionary (abstract → physical):

| Symbol | Meaning | Hardware |
|--------|-----------|----------|
| `μ` | node activity | voltage `V` |
| `ε` | prediction error | voltage difference `ΔV` |
| `θ` | synaptic weight | memristor conductance `W` |
| `Π` | precision (variance control) | node conductance `G` |
| `F` | free energy | physical energy potential |

---

## 2.5 Fixed Architecture Decisions (ADR)

Decisions that bindingly fix the concept. Each with rationale and
discarded alternatives — so that later it remains traceable *why*.

### ADR-1 — Sequence representation: time-continuous recurrent system

**Problem:** A static resistor network at equilibrium is *timeless* — it
has no "before/after". How then is a word sequence mapped?

**Decision:** The network is **not an equilibrium solver, but a
time-continuous recurrent dynamical system with decaying memory**
(related to Reservoir Computing / Liquid State Machines / Continuous-Time RNNs).
Time arises from two mechanisms:

- **RC memory:** filaments are RC elements, not ideal resistors. The
  time constant `τ` gives the system inertia → a token leaves behind a
  decaying "echo" that influences the following token.
  `τ · dμ/dt = −μ + ε^(l) − ε^(l−1) · ∂f/∂μ`
- **Predictive token streaming:** tokens are streamed in time; the network predicts
  at each `t` the next input signal (`x̂(t+Δt) = f(μ^(1)(t))`).
  Expected token → ε ≈ 0; unexpected → large error impulse.

**Discarded:** *Traveling waves as positional encoding.* Rationale: A pure
RC network obeys the **diffusion equation** (signals smear out), not the
wave equation — true traveling waves would require **inductance (LC)** or *active*
elements (Hodgkin-Huxley-like), which passive memristor networks do not have. The option
remains deferred until LC / active components are deliberately part of the hardware.

**Consequence:** A **fourth time scale** is added — the token rate `Δt` —
alongside inference (µs), precision (tens of µs) and plasticity (ms). See open
questions 7 and 8.

**Cousin / de-risking:** Reservoir Computing and Continuous-Time RNNs are
established. Goal 1 should therefore **also** be measured against these, not only against
Transformers.

---

### ADR-2 — Time-constant gradient across the hierarchy (solves question 8)

**Problem (question 8):** PC wants `dμ/dt` for fast settling (inference),
the memory wants `dμ/dt` for carrying along across tokens. With *one*
time constant, incompatible.

**Decision:** **Fast inference** is prioritized (`τ_inference ≪ Δt`, the
inference settles fully between tokens). The memory arises **not** from
a separate system, but from a **gradient of the time constant across the
hierarchy levels**:

```
τ_l  grows with level l
```

| Level | Cluster | τ | Role |
|-------|---------|---|-------|
| deep (dense) | many | `τ ≪ Δt` | fast inference (per token) |
| high (sparse) | few | `τ ≫ Δt` | memory / context (across sequence) |

**Rationale:** Falls directly out of hierarchical PC theory — higher levels
encode more abstract, temporally more extended causes and naturally vary
more slowly. The cosmic hierarchy thus supplies both time scales "for free",
without a bolted-on additional memory.

**Consequence (sharpens question 7):** context length `≈ τ_top / Δt`. Long contexts
require large `τ_top` → large capacitances, sluggish top level. Hard
physical tradeoff: context costs inertia.

**Consolidated time-scale ordering:**
```
τ_inference  ≪  Δt  ≪  τ_memory(top)  ≪  τ_plasticity
   (µs)        (token)    (context)        (learning, ms+)
```

---

### ADR-3 — Memory partitioning: structure = memory (solves question 7)

**Problem (question 7):** If the analog fading memory is to hold long context,
`τ_top` must rise to seconds → huge capacitances, sluggish top level,
resolution loss. Hard physical tradeoff.

**Decision:** There are **three separate kinds of memory**, each in its place:

| Kind | Location | Time scale |
|-----|-----|-----------|
| **Long-term knowledge** | **Structure of the network** = learned weights θ (memristor conductances) + topology | plasticity (slow) |
| **Working context** | **digital part** (token buffer, streaming order) | sequence |
| **Short-term coherence** | analog RC echo (ADR-1/2) | a few tokens |

Guiding principle: **"The network *is* its memory."** Long-term memory does not flow
through the network — it is burned into the conductances. Long context is **not**
carried by the analog fading memory, but managed by the digital part.

**Consequence:** Question 7 is **defused** as a tradeoff — `τ_top` stays small and
cheap (only short-term coherence needed). The top level no longer has to buffer
long context.

**New guard (question 9):** The efficiency thesis requires that the digital
context management remains **lightweight** (buffer, no O(n²) attention).
Bookkeeping digital, associative heavy lifting analog in the structure.

---

### ADR-4 — Roles of the digital part: compiler/interface + state machine

**Problem:** ADR-3 left "digital manages context, lightweight" vague. What
exactly is the digital part?

**Decision:** Classical analog-hybrid division. The digital part has **two
clearly separated roles**:

1. **Compiler + interface (persistence).** The model = the analog network structure
   (weights θ + topology). The **canonical representation lies digital**; the
   compiler maps it onto the analog substrate (programs memristors) and
   reads learned states back. This is how "the memory = the model in the
   analog space" is stored, loaded, transferred.
2. **State machine (runtime orchestration).** Sequences the discrete steps:
   inject token → let analog settle → sample → optional learning/nudge phase
   → next token → periodically checkpoint. **Control flow, not computation.**

**Rationale:** Corresponds to the historical hybrid-computer architecture — a digital
host compiles the problem onto the analog fabric and clocks its execution.

**Consequences:**
- **Bidirectional interface:** writing (load/compile) *and* reading
  (checkpoint after in-situ learning). Memristor read-back has precision limits →
  connects with question 4.
- **Coarse-grained clocking (critical):** the FSM *sets up and samples*; between
  the transitions the analog physics runs **freely**. If the FSM micromanages every
  µs step, ADC/DAC becomes the bottleneck and the analog advantage is gone.
- **Answers question 9 largely:** the state-machine nature guarantees that
  digital structurally does control + I/O + persistence, *not* O(n²) attention.

---

### ADR-5 — Input: peripheral-radial propagation through the fractal

**Problem (question 1, part A):** How does a token signal enter the network and
distribute itself?

**Decision:** The signal enters at the **periphery** (outer, large
filaments) and propagates **from outside inward**: across the nodes onto the
individual clusters, their sub-elements, and branches at each node further onto
other filaments. Along the way:

- **Attenuation** along the path in a slow curve. **This curve *is*
  the conductance-distance law `W(r)`** (= Hausdorff scaling from
  `fractal_llm.md`, `W ~ r^{−(d_H−1)}`). Favorite: **power law** (scale-free,
  slow decay); alternative: exponential. Parameter `α`/`d_H` controls the
  **penetration depth**.
- **Transformation** at each cluster: weighted modification of the signal by
  the data vectors stored in the cluster = analog **matrix-vector multiplication**
  (memristor crossbar via Ohm/Kirchhoff).

**Consistency:** Corresponds to the **bottom-up error flow** in hierarchical PC
(periphery = dense leaf level / fast inference from ADR-2; core = sparse
top level / memory). The prediction flows the other way (inside → outside, top-down).

**Direction enforcement (passive):** "outside → inside" is realized as a **potential
gradient** — drive the periphery, read/sink the core → effective inward flow +
attenuation for free, purely passive. A strict one-way path would require active components
(deferred, like the traveling wave).

**Opens questions 10–12** (see below).

---

### ADR-6 — Output: top-down prediction at the periphery (PC-native)

**Problem (output readout):** How is the next-token prediction read out of the
network?

**Decision:** PC-native. The network **predicts its next input signal**;
for language that is the next token. The output is the **top-down prediction,
read off at the same peripheral nodes as the input** (input and output
**co-located**).

- Prediction arises in the **core** (top level, memory/context from ADR-2),
  flows outward, is concretized at each cluster by its data vectors,
  arrives at the periphery as a next-token pattern.
- **Generation:** settle → read periphery prediction (ADC) → digitally
  decode + sample → feed token back → next step (state machine
  ADR-4).
- **Training:** the real token clamps the periphery → error ε = real − predicted
  → flows bottom-up → local plasticity. *Output prediction and training error
  are the same physical quantity.*

**Discarded:** a dedicated separate output region — it breaks the PC symmetry
"predict your own input" without added value.

**Catch → question 13:** A voltage cannot be prediction (outward)
and error (inward) at the same time. Canonical PC microcircuit (Bastos 2012, Rao/Ballard
1999): each cluster needs **two node types** — representation units (μ)
and error units (ε). Roughly doubles the node count, affects generator + HW.

**Coupling:** Decoding = **inverse of the embedding map** (question 10). Input/output
are one mapping and its inverse.

---

### ADR-7 — Embedding: learned 3D place code in a peripheral shell

**Problem (question 10):** How is a token mapped onto a voltage pattern?

**Decision:** Learned **place code** (population code), three sub-decisions:

- **(A) distributed:** token = localized activation "bump", not a single
  node (noise-robust, biologically plausible).
- **(B) learned, digital:** `E` is a **digital position table** (token →
  position), co-trained. Lives digital (ADR-4).
- **(C) geometry = semantics:** token = **3D position `(r, θ, φ)` in a
  peripheral shell of finite thickness** (not just a 2D spherical surface). Similar
  tokens = nearby positions = overlapping propagation paths → **generalization
  for free from the geometry**. Decode = digital float distance from predicted bump
  → token positions → softmax.

**Hybrid division of labor (precision):** position table, encode target,
decode distance + softmax run **digital in float** (precision-critical); the
propagation/transformation runs **analog** (noisy, massively parallel). Digital
precision **refines** the capacity limit, **does not lift it** — the analog
spatial resolution of the bump remains the ceiling.

**Consequences:**
- **Learnable 3D basis (G28 — incorporated):** `θ_{j,n,l,m}` over the full basis
  `R_n(r) · Y_lm(θ,φ)` — θ itself selects relevant (n,l,m) combinations.
  A frozen single harmonic would produce zeros that θ can never
  overcome (universality breakage). A learnable basis fully restores B-1.
  Stage 4: restrict basis to l ≤ 2; stage 5: full basis.
- **Radius = processing depth (DECIDED: thick shell).** The embedding
  factorizes: angle `(θ,φ)` = semantic field (*about what*), radius `r` =
  information content / processing depth (*how much computation*). Injected deep
  (small `r`) = little processing → frequent/light words; surface (large
  `r`) = full processing → rare/complex words. Consistent with
  information theory + PC (computation ∝ surprise) and the energy motivation.
  Cousin: *Mixture-of-Depths / Adaptive Computation*. Radius is **learned**
  (position table), only **initialized** by corpus frequency (cold start).
- **Refines ADR-5:** input occurs **within a shell of thickness `Δr`**,
  not on an infinitesimal outer surface.

---

### ADR-8 — Learning loop: online-local plasticity, composed from existing parts

**Problem:** How is training done? Six sub-problems: (1) coordinate two learning sites,
(2) inference-before-learning, (3) credit assignment without backprop, (4)
temporal credit, (5) stability/drift, (6) learning schedule.

**Decision:** **Online-local plasticity** as the main path — composed from
already-decided building blocks, not newly invented:

| Sub-problem | Solution | Source |
|-------------|--------|--------|
| (2) inference-before-learning | τ-gradient: inference settles within `Δt`, plasticity (`τ_plast ≫ Δt`) averages over settled errors — no explicit phases | ADR-2 |
| (4) temporal credit | RC / memristor states = **eligibility traces** (e-prop-like, Bellec 2020); range limited by `τ_memory` | ADR-2/3 |
| (5) stability | **Joglekar window** `W(1−W)` (weights ∈ [0,1]) + **precision update** `dΠ/dt` as homeostatic gain control | `fractal_llm.md` |
| (1) two learning sites | digital FSM reads the periphery error **once**, drives both updates; the embedding learns more slowly / is briefly frozen during consolidation | ADR-4/7 |
| (6) schedule | continuous-online; periodic digital consolidation + checkpoint | ADR-4 |

**Fallback:** Equilibrium Propagation (two-phase: settle freely → clamp target
→ learn the difference) — cleanest gradient, in case the online variant is
unstable. Forward-Forward as a further fallback (both in `chat.md`).

**Residue (→ question 3, only simulation clarifies):**
1. Does the local rule really lower the *global* loss? (PC-backprop equivalence
   could be violated by sparse/noisy/streaming)
2. How long is the temporal horizon in reality? (capped by `τ_memory`)
3. Stability at scale (millions of nodes)?

The learning loop is the **hinge between the theory and the simulation stage**.

---

### ADR-9 — Soft geometry: elastically anchored weights (solves G25)

**Problem (G25, blocker):** On the inter-level edge (parent-μ↔child-ε) lie two
roles that seem to be mutually exclusive: `W ~ r^−(d_H−1)` (Hausdorff scaling,
*defines* the fractal) and `θ` (learned PC prediction weight). Fixed → nothing
learnable; plastic → learning eats the fractal scaling.

**Decision:** **Two protection layers** ("soft geometry", biologically: coarse
anatomy conserved, fine synapses plastic):

| Layer | Content | Preservation |
|-------|--------|-----------|
| **hard** | connectivity (which edges exist) | **absolutely fixed** — learning never creates/deletes edges → fractal graph (d_H, small-world, sparsity) is preserved as a *design principle* |
| **soft** | weight values θ | **elastically anchored** to `W(r)` |

Learning rule with spring term:
```
dθ_ij/dt = η · ε · ∂f/∂θ  −  κ_l · (θ_ij − W(r_ij))
```
- `W(r)` is the geometric **initialization + anchor** (not a fixed value).
- Equilibrium at learning pressure = spring force → deviation ∝ `η·ε / κ_l`; sustained
  errors permit **long-term drift**, the spring prevents **runaway**.

**Level-dependent stiffness (the core of the decision):**
```
κ_grob (Kern/Top)  ≫  κ_fein (Blatt/Peripherie)
```
→ coarse base structure stiff (preserved), fine details loose (freely learnable).
Uses the hierarchy from ADR-2.

**Side effects (positive):** the spring term is at once regularization (against
overfitting/drift, G9) and stabilization (G22). One mechanism, three effects.

**New parameter:** stiffness profile `κ_l` — a quantitative stage-4 lever.

**Extension: structural plasticity (controlled topology erosion)**

ADR-9 now also permits slow edge erosion and growth — analogous to
synaptic pruning and synaptogenesis in the brain. Three mechanisms, three time scales:

| Mechanism | Condition | Time scale |
|-------------|-----------|-----------|
| **Pruning** — remove edge | `θ_ij < θ_prune` after checkpoint (despite the spring) | hours |
| **Growth** — add edge | `corr(V_i, V_j) > θ_grow` over N tokens AND distance `r_ij < r_max` AND degree < budget | hours |
| **d_H monitoring** | box-counting periodically; if `d_H` drifts `> ±0.3` from target → lock pruning | days |

Growth prior (Hausdorff preservation): `P(new edge i↔j) ~ r_ij^{−(d_H−1)}`
— new edges follow the same geometric distribution as the original.

Safety condition: pruning only if the graph is still connected after removal.

**New parameters:** `θ_prune` (pruning threshold, e.g. 1% of W_init),
`degree_budget` (max. +X% degree per node), `d_H_tolerance` (±0.3).

**Revised core statement:** Connectivity is **slowly plastic within
geometric bounds** — no longer absolutely fixed. Time scales:
```
τ_weights   ≈  ms       (ADR-8, memristors)
τ_structure ≈  hours    (pruning/growth at checkpoints)
τ_d_H       ≈  days     (periodic fractality measurement, stage-2 methodology)
```

---

### ADR-10 — Symmetry consistency: Y symmetric by design, direction via topology (solves G24)

**Problem (G24, blocker):** Y is symmetric PSD (generator + fractal_llm.md);
PC dynamics is directed (μ→ε vs. ε→μ). Does the effective system matrix lose
the PSD property after including `I_pred(V,W)` → oscillation/divergence?

**Decision:** Y **deliberately stays symmetric** — role separation instead of conflict:

| Role | Carrier | Property |
|-------|--------|-------------|
| Passive substrate (convergence) | **Y** | symmetric PSD → λ_min > 0 → EP reciprocity |
| PC directedness (ε = μ − f) | **μ/ε topology + f/J digital** | asymmetric, but NOT in Y |

**Resolution via time scales:** Y is **frozen** during per-token settling
(`τ_plast ≫ Δt`), hence guaranteed symmetric-PSD *during* the inference. Slow
structural evolution happens only on `τ_plast` (ADR-9). Symmetry is never
broken *within* an inference step.

**Where does the PC directedness come from?** Not from Y, but from:
1. **μ/ε node separation (ADR-6):** prediction and error on physically separate nodes.
2. **Injection asymmetry:** ε-nodes receive `f(μ_higher)` injected digitally (DAC),
   μ-nodes do not → produces `ε ≈ μ − f` without an asymmetric Y.

### ADR-12 — Trisynaptic ganglion: third node type ι (interneuron)

**Problem:** The dual μ/ε microcircuit (ADR-6) is a minimal ganglion.
Biological ganglia have interneurons for local gain control. The precision G
is currently computed digitally (ADC→compute unit→DAC) — that violates the
"analog computes for free" principle (ADR-4).

**Decision:** A third node type ι (interneuron) per cluster.
The system becomes **trisynaptic**: ε → ι → μ (three synapses per level crossing).

**Role of ι — analog precision estimator:**
```
τ_ι · dV_ι/dt = −G_ι · V_ι + G_ει · V_ε²
```
ι accumulates the squared error signal (`V_ε²`). At equilibrium: `V_ι = ε²`.
From this the effective precision follows: `G_eff = G_scale / (V_ι + ε₀)`

**Three roles:**
1. Analog precision estimation → partially replaces the digital G update (G26)
2. Local gain control → strengthens normalization (G6)
3. Robustness against G mismatch (G15) through dynamic G_eff

**Node indexing:** `μ(c) = 3c`, `ε(c) = 3c+1`, `ι(c) = 3c+2`
**Node count:** N = 3 × n_clusters (previously 2×)

**Intra-cluster edges:**
- `μ ↔ ε` — prediction-error coupling (ADR-6, unchanged)
- `ε ↔ ι` — error signal drives the interneuron (`W_eps_iota`)
- `ι ↔ μ` — gain modulation (`W_iota_mu`)

**Time constant ι:** `τ_ι = sqrt(τ_leaf · τ_μ)` — geometric mean.
Positions ι exactly between fast ε and slow μ.
Symmetric logarithmic distances to both neighbors (explanation in the generator).

**Implemented in:** `cosmic_web_generator.py` (ADR-12 version, tested).
510 clusters → 1530 nodes (510 per type), 3025 filaments, 0.13% fill ratio.

---

**Remaining residue → question F(G27):** Analytic contraction bound of the
nonlinear closed loop `−Y + ∂I_pred/∂V` for concrete learned θ —
Hurwitz stability to be checked empirically in stage 4.

---

### ADR-11 — Token injection: population code via Gaussian bump (solves G48)

**Problem (G48, blocker):** ADR-7 defines token positions as continuous
float coordinates `(r, θ, φ)`. The generator produces discrete leaf nodes at fixed
positions. How is a bump injected whose center does not lie on a node?

**Decision: Option C — population code.**
Each peripheral node `i` receives a signal proportional to the Gaussian bell around
the learned token position `p_tok`:

```
V_in(i)  =  A · exp( −d(i, p_tok)² / (2σ²) )
```

- `d(i, p_tok)` — geodesic distance on the peripheral shell
- `σ` — bell width (controls semantic granularity; new design parameter)
- `A` — amplitude (normalized to 1 after the encoder's softmax)

**Staged plan for the injection:**

| Stage | Injection | Why |
|-------|-----------|-----|
| **4** (simulation) | Option A (nearest node) | small vocabulary, no collapse |
| **5** (language mini) | **Option C (Gaussian bump)** | generalization from geometry active |

**Why C fits:**
- Biologically: population coding (place cells, head-direction cells in the brain).
- Similar words → overlapping bumps → overlapping propagation paths
  → **generalization for free** (ADR-7 C fulfilled).
- Decode is the direct counterpart: output bump → distance to all
  token positions → softmax.

**New parameters:** `σ` (bell width) + `k_cutoff` (nodes below threshold
are cut off → sparsity preserved). To be tuned quantitatively in stage 5;
`σ` couples directly to question 14 (shell capacity).

---

## 3. Data Flow — How a Sentence Is Processed

```
   TEXT
    │  (digital: tokenizer)
    ▼
  Token IDs ──► Embedding ──► voltage vector  V_in
                                   │  (DAC: digital → analog)
                                   ▼
        ┌─────────────────────────────────────────┐
        │   ANALOG COSMIC FRACTAL NETWORK           │
        │                                           │
        │   • apply V_in to input nodes             │
        │   • Kirchhoff/Ohm let the voltages        │
        │     settle into the energy minimum        │
        │   • only error ε flows through filaments  │
        │   • memristors learn locally (slowly)     │
        └─────────────────────────────────────────┘
                                   │  (ADC: analog → digital)
                                   ▼
        voltage at output nodes  V_out
                                   │  (digital: σ-LUT, softmax)
                                   ▼
                           next token
```

Three time scales run **asynchronously**:
- Inference (analog): µs — voltages settle
- Precision (digital): tens of µs — variance control, DAC injection
- Plasticity (memristor): ms — slow weight learning

---

## 4. Open Questions & Risks

These points are **not yet solved** and must be answered over the course of the
project:

1. ~~**Embedding → geometry**~~ — **SOLVED:** propagation via ADR-5
   (peripheral-radial), encoding via ADR-7 (learned 3D place code). Remaining
   quantitatively → question 14.
2. **Sequence processing:** A Transformer has attention over the whole sequence.
   How does a static 3D network represent temporal order / context?
3. **Scaling of learning (collection point for the remaining ADR-8 items):** Does PC
   converge with hardware noise and at language scale? Concretely from ADR-8: (a) does the
   local rule really lower the *global* loss despite sparse/noisy/streaming?
   (b) how long is the temporal horizon in reality (capped by `τ_memory`)?
   (c) does learning stay stable at millions of nodes? *(empirical, simulation stage)*
4. **Hardware reality:** Memristors have limited precision, drift, finite
   write cycles. Is that enough for a learning-capable model?
5. ~~**Expressiveness**~~ — **NARROWED (finding B-1):** *Qualitatively universal*
   — sparsity not fatal (scale-free = small-world, path length ~log N;
   settling = unrolled depth ~log N; universality inherited from cousins CT-RNN /
   Reservoir / PC). *Practically a different inductive bias* than the Transformer:
   static inference weights instead of dynamic all-pairs attention; instead
   precision gating + geometric content addressing + adaptive depth. Stronger
   at predictive/streaming/low-energy, weaker at exact long-range retrieval.
   *Real limits quantitative* (horizon F7, capacity F14, noise F4), not
   qualitative. **Consequence: do not aim for Transformer parity, but for the
   PC/streaming niche — measure per Goal 1.**
6. ~~**Real fractality (core hypothesis)**~~ — **OBSOLETE due to the Goal-2 recasting.**
   Goal 2 no longer asks *whether* the cosmic-neuronal similarity is "real"
   (probably coincidence — and that's ok), but *whether it is usable*. The
   proof of the analogy via `d_H`/`P(k)`/correlation **drops out as a project goal**.
   The box-counting methodology survives only as a **tool** (ADR-9:
   d_H monitoring of the *own* topology during structural plasticity),
   not as a brain↔cosmos comparison.
7. ~~**Context length vs. inertia**~~ — **SOLVED via ADR-3** (memory partitioning:
   long context digital, long-term knowledge in the structure, analog only
   short-term coherence). `τ_top` stays small. Tradeoff defused.
8. ~~**Inference vs. memory dynamics**~~ — **SOLVED via ADR-2**
   (time-constant gradient across the hierarchy; fast inference below,
   memory above).
9. **Lightweight digital context management (from ADR-3, → Goal 1):** Through
   ADR-4 **structurally answered** — digital is state machine + compiler
   (control/I/O/persistence), no O(n²) attention. Remaining quantitative
   residue: **ADC/DAC bandwidth** and **memristor read-back precision** (→ question 4).
   *(quantitative, guard, for HW stage)*
10. ~~**Token encoding / embedding map**~~ — **SOLVED via ADR-7** (learned
    3D place code, digital position table). Remaining: table cold start +
    training (part of the still-pending learning-loop discussion).
11. **Shape of the attenuation curve (from ADR-5, → Goal 1):** power law vs.
    exponential; which `α`/`d_H` gives usable penetration depth? *(quantitative,
    can be tried out in simulation)*
12. ~~**Geometric-radial vs. hierarchical**~~ — **SOLVED in stage 3:** the generator
    places clusters radially (top in the core r=0.15 → leaf periphery r=1.0).
    *Consequence:* the closed-form `d_H` formula now holds only angularly as a reference;
    the true `d_H` is measured in stage 2 (box-counting).
13. ~~**Dual node type μ/ε**~~ — **SOLVED in stage 3:** the generator produces per
    cluster a μ/ε node pair (index `2c` / `2c+1`), intra-cluster coupling,
    vertically parent-μ ↔ child-ε. Doubles the node count as expected.
14. **Capacity of the peripheral shell (from ADR-7, → Goal 1):** radius semantics
    **decided** (thick shell, radius = processing depth). Remaining
    quantitatively: how many distinguishable token positions fit in the shell
    (limited by analog bump width/noise)? How thick must `Δr` be?
    *(quantitative, for the simulation stage)*

> This list is the counterweight to the enthusiasm. Each milestone should
> answer or narrow down at least one of these questions. **Assignment after
> the Goal-2 recasting:** questions 1–5, 7–14 serve **Goal 1** (algorithmic
> practicability, software). **Goal 2** (hardware-side usability) is no longer
> tested by a single "fractality question", but by the
> hardware stages 6/7 (REDAC, memristor) — there it is decided whether the
> fractal-inspired design is *usable* on hybrid hardware. Question 6 is
> obsolete.

36. **Polysemy in the place code (G51, → Goal 1, stage 5):**
    ADR-7 assigns each token a fixed position. Ambiguous words (bank, lock)
    have one position, but different meanings. Primary
    mitigation: the RC echo (ADR-1/2) modulates the analog state
    context-dependently → same token position, different analog state.
    Stage 5: measure empirically (generalization test D from question 23).
    If insufficient: BPE subword tokenization (natural disambiguation,
    e.g. "bank branch" vs. "park bench" as separate tokens).

35. **Throughput & pipelining (G43, → Goal 1, stage 4/6):**
    Stage 4: strictly serial (token → settling → next token). ADR-4
    serial FSM flow. Throughput = 1 / t_settle tokens/s, log it.
    Stage 6: pipelining possible via τ-gradients (new token into leaves,
    while the core is still settling for the previous token) → not contradicting ADR-2
    since the top echo is intentionally persisted. Quantitatively in stage 6.

34. **Toy-problem specification (G35, → Goal 1, stage 4):**
    Next-symbol prediction on a synthetic grammar: V={A,B,C,D} (4 characters),
    switching between patterns (ABCABC..., AABBAABB...), random transitions.
    Sequence length: 500–5,000 tokens. Network size: reduced (n_top=3, n_levels=3,
    ~100 nodes). Tests ADR-1 (sequential), RC memory, hierarchical
    structure AND adaptation speed after a pattern switch.
    Gate stage 4 (supplements F21): accuracy > 60% (chance=25%); PC correlation
    > 0.5; V ∈ [−1.5,+1.5]. Abort: accuracy < 30% after 2,000 tokens.

33. **Jacobian rate (G44, → Goal 1, stage 4/6):**
    Stage 4 (digital): recompute f and J at every solver step —
    scipy handles step-size control automatically.
    Stage 6/7 (hardware): once per token (frozen linearization); error
    acceptable at small ε (convergent learning). ADR-4: analog runs freely
    between FSM transitions. No additional architecture decision needed.

32. **ODE solver: stiff, implicit, sparse (G32, → Goal 1, stage 4):**
    The τ span 1..100 makes the system stiff (10,000 explicit steps vs.
    ~10–50 implicit per token). Recommendation: `scipy.solve_ivp(method='Radau',
    jac=J_rhs, jac_sparsity=Y_sparsity)`. Jacobian = −Y_eff + diag(G)·J_f,
    sparse O(N log N). Convergence criterion (G37): integrate until t = 5·τ_leaf
    (99.3% of the fast modes); top nodes deliberately do not settle (memory).

31. **f/I_pred concretely for stage 4 (G33, → Goal 1):**
    Y_lm/R_n are omitted. Simplified prediction: f_c = tanh(Σ W·V_μ
    of the parent μ-nodes). I_pred[μ_c] = G_prec · V̂_c (precision current);
    I_pred[ε_c] = G_prec · (V[μ_c] − V̂_c) (error current). Jacobian J_c =
    1 − V̂_c² (tanh derivative, digital). Full basis R_n·Y_lm from stage 5.

30. **τ→C mass matrix (G31, → Goal 1, stage 4):**
    C_i = τ_i · G_prec (choice A). Yields: C_core = 100·0.01 = 1.0,
    C_leaf = 1·0.01 = 0.01. Consistent with the fractal_llm.md normalization
    (C=1F in the core). Stiffness ratio 100:1 → implicit solver needed.
    Implementation: `C_diag = web.tau * cfg.G_min` (two lines of Python).

25. **Context feedback (G52, → Goal 1, stage 4/5):**
    Question 7 was marked "solved" too optimistically. State:
    Effective context length is limited primarily by the RC echo (τ_top/Δt tokens).
    Three options in stages: stage 4 = Option A (RC echo only, minimal);
    stage 5 = compare Option A vs. C; stage 6/7 = Option C
    (digital context anchor: k-dimensional compressed context vector h_t,
    computed digitally O(k), injected via DAC into the slow top nodes).
    Option B (re-streaming) breaks the streaming advantage → discarded for
    productive use. Question 7 remains quantitatively open until stage 5.

24. **Bipolar/unipolar — sign handling (G11, → Goal 1, confirmed):**
    No differential pair needed. Signed currents arise from
    bipolar voltages V ∈ [−1,+1] + positive conductances W ∈ [0,1]
    via Kirchhoff (I_ij = W_ij·(V_i−V_j)). Signed weights θ
    in the prediction function f live in the digital float64 part (ADR-4/G28).
    Lateral inhibition (−Y_lat·V) needs no negative Y_lat — the
    minus sign comes from the F functional §0. The simulation model confirms:
    W ∈ [0,1] (analog), θ ∈ ℝ (digital), V ∈ [−1,+1] (bipolar).

23. **Generalization measurement (G9, → Goal 1, stage 5):**
    Four standard tests + one architecture-specific:
    (A) Paraphrasing: invariance rate < 5% quality loss; addition: check
    whether bump overlap correlates with output similarity (ADR-7 C validation).
    (B) Contamination: sequence-pattern test (mask token at position N →
    sensible completion?); RC variant: different prior contexts.
    (C) OOD geometric: test tokens with geometrically near vs. far
    positions to the training set → generalization rate should fall monotonically with
    geometric distance → direct ADR-7 falsification.
    (D) Entropy: token entropy over 10 generations (T=0.7); addition:
    node-activation entropy → checks whether σ (Gaussian width, ADR-11) fits.
    (E) Geometric spectrum (new, architecture-specific): correlation
    d_geom(i,j) vs. d_output(i,j) over all token pairs → confirms or
    refutes "generalization from the geometry" (ADR-7 C) quantitatively.

22. **Reproducibility & checkpoint round-trip (G8, → Goal 1, stages 4–7):**
    **Software stages (4/5):** standard ML reproducibility — seeds for
    topology (seed_topo), init (seed_init) and data (seed_data); all
    hyperparameters logged. Fully deterministic.
    **Hardware stages (6/7) — master decision:** digital is the master,
    analog is the execution environment. New knowledge flows analog→digital
    (at checkpoints via ADC); deployment flows digital→analog (via DAC).
    **Checkpoint inventory:** θ (memristors, 3–6 bit), G/Π (float64),
    embedding table (float64), edge topology (exactly per ADR-9).
    RC states V are **not** stored (working memory, not model).
    **Tolerance budget:** PPL degradation after round-trip < 5% →
    determines the minimal memristor bit depth; to be measured quantitatively in stage 6.
    The ADR-9 spring keeps weights smooth → more robust against quantization noise.

21. **Success and abort criteria + evaluation framework (G7, → Goal 1):**

    **Evaluation framework:** Not against GPT/Transformer — against RNN/LSTM/SSM
    of the same size (B-1 niche). MMLU/HumanEval/LMSYS/GSM8K not applicable
    (wrong yardstick). Run NIAH and document the context limit.
    Two additional metrics beyond the standard framework:
    (a) **Continuous learning** (anti-forgetting rate < 20% PPL degradation
    after a new corpus); (b) **Fractality preservation** (d_H ≈ 2.0 after training).

    **Stage 4 — gate (architecture validation):**
    Passed: PC correlation > 0.5; stability V ∈ [−1.5,+1.5]; error < chance.
    Abort: correlation < 0 OR collapse < 1,000 steps.

    **Stage 5 — gate (language quality, weighted metrics):**
    Quality (70%): PPL ≤ 2× RNN (35%), anti-forgetting < 20% (20%),
    win rate > 50% (15%). Efficiency (30%): energy ≤ 0.5× RNN (20%),
    TTFT/throughput > 30 tok/s (10%).
    Abort: PPL > 5× baseline OR no convergence after 2 runs.

    **Stage 6/7 — gate (hardware validation):**
    Energy/token ≤ 0.3× GPU-RNN (40%); TTFT < 200ms (25%);
    PPL preservation after HW transfer (20%); throughput > 30 tok/s (15%).

    **Overall project:** Success = stage 5 passed. Scientific
    failure = stage 5 not passed after 2 iterations → publishable.

20. **Normalization of node activations (G6, → Goal 1, stage 4):**
    Drift of the absolute voltage V over long sequences is addressed by two already
    incorporated mechanisms: (1) precision update G = 1/ΔV²
    (G26 fix) regulates error amplification; (2) lateral inhibition −Y_lat·V
    (G30 fix) pulls the absolute V level back toward zero. Together they
    implement the analog of LayerNorm — distributed and local. Check empirically in
    stage 4: plot the V trajectory over the sequence. If needed: a periodic
    digital normalization step in the FSM (ADR-4) as a fallback.

19. **Weight initialization θ (G5, → Goal 1, stage 4):**
    `W_init = (ρ_target / ρ₀) · W(r)` with `ρ_target ≈ 0.9` (edge of chaos,
    reservoir criterion). `ρ₀` = spectral radius of the geometric init
    (once via `scipy.sparse.linalg.eigs`). The ADR-9 spring anchors to
    `W_init` → the edge-of-chaos property is preserved permanently.
    New parameter: `ρ_target` (stage 4: fixed 0.9; stage 5: hyperparameter).
    Simultaneously solves variance preservation across L levels (Xavier/He analog).

18. **Sampling strategy & RC feedback (G4, → Goal 1, stage 4/5):**
    Stage 4: **greedy** (deterministic, reproducible). Stage 5: **top-p**
    (p ≈ 0.9, standard, context-adaptive). Sampling is purely digital (ADR-4),
    no hardware intervention needed. RC feedback of the sampled token:
    uncritical, since (1) the fast leaf nodes settle anew per token (τ_inferenz ≪ Δt,
    ADR-2) and (2) the slow top nodes average over many tokens → sampling noise
    is naturally damped; produces the desired response diversity.

17. **Training data regime (G3, → Goal 1, stage 4/5):**
    Stage 4: synthetic single stream (1k–10k tokens), no replay needed.
    Stage 5: small natural-language corpus (1–10 million tokens) +
    simple replay buffer (~10%) as protection against catastrophic forgetting.
    The ADR-9 spring term (κ_l gradient) is the first anti-forgetting line: core-near
    weights (high κ) barely forget, leaf weights (low κ) are plastic.
    Quantify the strength of the protection in stage 4. Corpus size is naturally
    limited by shell capacity (question 14) and V (question 15) —
    the model is small-scale by nature.

16. **Training objective function (G2, → Goal 1, prerequisite for stage 4):**
    Full objective function: `L = F + λ·Cross-Entropy`. Analog minimizes F
    locally (Hebbian rule, pulls voltage toward the prediction); digital minimizes
    cross-entropy on the embedding table (ADR-7, pushes token positions
    apart). Both driven by the same periphery error (ADR-6).
    Bridge: Gaussian decode (ADR-11) makes F and cross-entropy compatible under
    `P(tok) = exp(−d²/2σ²)/Z` — F ≈ cross-entropy when tokens lie
    far apart. Stage 4 (V ≈ 50–100): F alone suffices.
    Stage 5 (BPE, semantically near tokens): λ > 0 needed. Solves the
    "global loss" left open in ADR-8/question 3.

15. **Tokenizer & vocabulary size V (from ADR-7/ADR-11, G1, → Goal 1):**
    In stages: stage 4 = **character level** (V ≈ 50–100, no training needed);
    stage 5 = **small BPE** (V ≈ 1,000–4,000, trained on the target corpus).
    Hard bound: `V_demand ≤ shell_capacity` (couples with question 14 —
    the F14 measurement in stage 4 gives the true upper limit). Radius cold start (ADR-7)
    is set by token frequency from the tokenizer corpus.

---

## 5. The Warehouse — What Already Exists (to Recycle)

| Artifact | File | When to recycle |
|----------|-------|---------------|
| All PC formulas + free energy | `fractal_llm.md` | When simulating inference/learning |
| Hybrid reformulation (voltages, 3 time scales) | `fractal_llm.md` | When designing the hardware/FPGA |
| Admittance-matrix derivation | `fractal_llm.md` | When building the system of equations |
| Topology generator (Soneira–Peebles) | `cosmic_web_generator.py` | At step 1 of the simulation |
| Box-counting / d_H measurement methodology | `STAGE2_FRACTALITY.md` | Now only as a **tool** for ADR-9 d_H monitoring (brain↔cosmos comparison dropped due to the Goal-2 recasting) |
| Original conversation log | `chat.md` | As an idea archive / rationale |

---

## 6. Staged Plan

Each stage: goal marker, what is built, which open questions it clarifies, what
is recycled from the warehouse (section 5).

### Stage 1 — Concept ✅ COMPLETED
Complete architecture core: **ADR-1 … ADR-12** + findings **B-1, B-2**. Everything
theoretically decidable is decided or narrowed down. No theoretical
hole remains open.
- ADR-1…8: base architecture (time, memory, learning, I/O, embedding)
- ADR-9: soft geometry (elastically anchored weights + structural plasticity)
- ADR-10: symmetry consistency (Y symmetric, direction via mu/eps topology)
- ADR-11: population-code embedding (Gaussian bump, stage 4 with nearest node)
- ADR-12: trisynaptic ganglion (third node type iota, gain control)

### Stage 2 — ~~Fractality proof~~ d_H measurement tool *(deprioritized)*
**Gutted by the Goal-2 recasting.** The original purpose — to prove that
neuronal and cosmic structures share the *same* fractality (compare `d_H`, `P(k)`,
`ξ(r)`) — is **no longer a project goal**, because Goal 2 deliberately leaves the
reality of the analogy open and asks only about usability.
- **What remains:** The box-counting / d_H measurement pipeline from `STAGE2_FRACTALITY.md`
  continues to be used as a **tool** — for the **d_H monitoring of the own
  topology** during structural plasticity (ADR-9), not for the
  brain↔cosmos comparison.
- **Optional/later:** The brain↔cosmos comparison can be done later as a *scientific
  add-on* (publication), but it is irrelevant for building the LLM.

### Stage 3 — Generator rebuild *(→ Goal 1)* ✅ COMPLETED
`cosmic_web_generator.py` raised to the ADR decisions and verified
(510 clusters, 1020 μ/ε nodes):
- **μ/ε node pairs** per cluster (ADR-6, question 13) ✓
- **hierarchical depth = geometric radius**, top in the core (ADR-5, question 12) ✓
- **thick peripheral shell** r ∈ [0.9, 1.1] (ADR-7, question 14) ✓
- **τ-gradient** core 100 → periphery 1 as a node attribute (ADR-2) ✓
- **Clarified:** questions 12, 13; shell prepared for 14.
- *Note:* radial rebuild → closed-form `d_H` formula now only angular;
  the true `d_H` is measured in stage 2.

### Stage 4 — Software simulation *(→ Goal 1, purely digital)* ✅ COMPLETED

**Result reservoir test (simulation_v5.py, 24 cores, 5.2s):**
- 52 clusters | 156 nodes | 8,000 tokens | Euler settling (dt=0.01, early-stop)
- Features: mu+eps+iota nodes × k=2 delayed states → 312 dimensions
- Readout: Ridge (alpha=0.1, class_weight=balanced)
- **Test accuracy: 81.8%** (chance: 33.3%, gate: >60%) → **gate greatly exceeded**
- Train=82.2% ≈ Test=81.8% → no overfitting, robust generalization
- Theoretical ceiling ~85-90% (grammar switch not predictable)
- Operation: saturated reservoir (clip ±2.0 as nonlinearity)
- **Finding:** Fractal hierarchy + tau gradient encodes sequence patterns with
  a 48 percentage-point lead over chance — purely through topology, without learning.
  Delayed states (k=2) are the strongest single lever (+19 PP vs. v4).

| Version | Method | Test accuracy | Runtime |
|---------|---------|-----------------|----------|
| v4 (24 cores) | mu only, k=1 | 62.3% | 5.2s |
| **v5 (24 cores)** | **mu+eps+iota, k=2, balanced** | **81.8%** | **5.2s** |

PC inference + learning loop (ADR-8) on the rebuilt topology. Proof on the
toy problem (compress/reconstruct patterns).
- **Clarifies:** question 3 (+ remaining ADR-8 items: global loss? horizon? stability?),
  question 11 (attenuation curve), question 14 (capacity, quantitative).
- **Recycled:** all PC formulas + admittance matrix from `fractal_llm.md`.

### Stage 5 — Language mini-model *(→ Goal 1)* ⚠️ RESULTS REVISED AFTER DIAGNOSIS

**IMPORTANT (diagnosis workflow pc-v2-diagnose, 5 hypotheses measured on the real network):**
The originally reported stage-5 numbers do **not** hold up under scrutiny.
Correction:

| original claim | diagnosis finding |
|---------------------------|-----------------|
| Reservoir 23-27.5% = "encodes Markov-3 text" | **Unigram lookup** — `within-token state std = 0.0000`, no sequence memory. The reservoir is deterministic in the current input. |
| RNN baseline 10.2% as a fair comparison | RNN **undertrained** (5000 steps), lies even *below* the majority class (11.97%). Comparison invalid. |
| PC learning 11.7% test (pc_v2) | **Majority-class artifact** — decode from a null fixed point (`I_free=0` → `||V||=0`), argmax pinned to cluster 0. No learning effect. |

**Underlying bugs (measured):**
- Euler instability: `dt·Cinv·Ydiag = 5.33 > 2` → leaf-mu saturates (70/378 nodes at the clip ±2), `||leaf-mu||=17.4`. Early-stop never fires.
- The core never settles: τ_top=40 needs ~4000 steps, the free run gives 300.
- The κ spring dominates the Hebbian signal ~230× → the weights barely move.

**Status:** So far it is **not demonstrated** that the fractal
topology contributes sequence memory beyond a trivial readout+delay buffer.
Stage 4 (81.8%) too could stem primarily from the delayed-state trick (k=2) +
Ridge, not from the reservoir dynamics. This must be clarified by **controlled
ablation** (fractal vs. random sparse vs. pure delay line),
before Goal-1 success is claimed.

**Stage 5b - ADR-8 validation (simulation_stage5b.py, 924s):**
Ridge-fair comparison W_init vs. W_hebb: no measurable Hebbian gain (+0.13 PP).
Consistent with the diagnosis: the learning rule is not the lever — but it was also
never fairly tested, because numerics (saturation) + decode (null fixed point) destroy the
experiment beforehand. ADR-8 thus remains **open**, not refuted.

Small vocabulary, real tokens, learned 3D place code (ADR-7),
measure prediction quality.
- **Benchmark (B-1):** compare against RNN/SSM/Reservoir, not only Transformer;
  focus on the PC/streaming niche.
- **Recycled:** embedding concept from ADR-7.

---

### Finding B-3 — Controlled ablation: "cosmic" gives NO memory advantage

**Experiment (simulation_ablation*.py, path B):** Memory-forcing task
(delayed copy, i.i.d. input, `target[t]=input[t−LAG]`) — no unigram lookup
possible. Standard leaky ESN (replaces the broken Euler), Ridge on the current state
(no delay buffer). Memory-capacity curve over LAG=1..30, **random_sparse over an
8-graph ensemble**, identical τ distribution/size/density/spectral radius.

| Regime | fractal | random (ensemble) | leaky_only |
|--------|---------|-------------------|------------|
| LAG ≤ 5 | 100% | 100% ± 0% | ≈ chance |
| LAG 8 | 57.9% | 98.9% ± 1.3% | ≈ chance |
| LAG ≥ 12 | falls faster | holds longer | ≈ chance |

**Findings:**
1. ✅ **Coupling carries:** fractal/random ≫ leaky_only — the reservoir principle
   works (a coupled sparse network has real memory, pure
   leaky integrators do not).
2. ❌ **"Cosmic" is measurably irrelevant:** fractal ≈ random short-range,
   **worse** long-range. A generic random graph of the same
   statistics performs equally well or better. The cosmic geometry gives **no
   demonstrable compute advantage** on this memory benchmark.
3. ⚠️ **Correction:** An earlier single run showed fractal seemingly superior
   (84% random) — that was an **outlier**, refuted by the ensemble. (Reminder:
   always check against the ensemble.)

**Consequence for the goals:**
- **Goal 1:** The fractal topology *works* as a reservoir (practicable),
  but its *specific* advantage over random-sparse is **not established**.
- Consistent with the **Goal-2 recasting**: the cosmic-neuronal similarity is
  probably coincidence — B-3 shows empirically that it also brings no
  benefit for memory. The possible value therefore lies (if anywhere) in **Goal 2**
  (hardware mappability/efficiency), not in a compute advantage.
- **Still open:** nonlinear/structured tasks where hierarchy might
  help after all. *(Hardware-layout question now answered by B-4.)*

---

### Finding B-4 — Hardware layout: "cosmic" is physically clearly superior

**Experiment (simulation_hardware.py, → Goal 2):** Layout ablation on identical
3D node positions. fractal edges (local: k-NN + parent-child) vs. random_sparse
(8-ensemble, same nodes/positions/edge count, **random** pairs). Metrics:
Euclidean wire length, locality, KMeans tile cut (REDAC crossbar mapping).
Network: 510 clusters, 1530 nodes, 3025 edges.

| Metric | fractal | random (ensemble) | advantage |
|--------|---------|-------------------|---------|
| Total wire length | 291 | 3330 ± 12 | **11.4× shorter** |
| Edges local (<0.2) | 79.5% | 1.7% ± 0.2% | ~47× more |
| Tile cut (k=6) | **4.7%** | 82.6% ± 0.6% | ~18× fewer |
| Tile cut (k=24) | 12.4% | 95.4% | — |

**Finding:** The fractal topology is **physically dramatically more hardware-friendly**
— ~11× shorter wiring, 95% of the edges stay within one of 6 tiles
(REDAC cluster structure). A random-sparse reservoir of equal *compute power*
(B-3) would be a wiring nightmare (83% inter-tile edges).

**The coherent overall statement of the project:**
> The cosmic-fractal structure **does not compute better** (B-3), but it is
> **buildable** where an equivalent reservoir is not (B-4). For an
> **analog-digital learning-capable hybrid model** — the core of Goal 2 — that is precisely the
> decisive advantage: local Kirchhoff computation, short wires, modular
> crossbar mappability. Pillar-2 motivation empirically confirmed.

**Calibration via random-geometric control (simulation_hardware2.py):**
The 11× advantage is **primarily locality**, not "fractal":

| Control | Wire length | Tile cut k=6 |
|-----------|-----------|------------------|
| fractal | 291 | 4.7% |
| random_geometric (local) | 379 (1.3×) | 8.2% |
| random_global | 3330 (11.4×) | 82.6% |

→ **Calibrated statement:** The dominant hardware lever is *locality*
(11× local vs. global). The *fractal hierarchy* gives on top of that a
**moderate** additional advantage over local-random (1.3× wire, ~1.7× fewer
tile cuts). The core point stands: *computing* does not need the structure (B-3),
*building* benefits from locality+hierarchy (B-4) — the fractal delivers both in
one construction (locality + τ-gradient + clean tiles at once).

**REDAC mapping detail:** 6 tiles → only **141 inter-tile connections** (out of 3025),
= the photonics-interconnect niche (B-2). Caveat: tile balance is unclean (117–312
nodes/tile, ideal 255) — load balancing for REDAC still to be solved.
Energy caveat (G16): the interconnect savings (11×) apply only to the
topology-dependent share; ADC/DAC/quiescent current (70–90% real) are independent.

### Stage 6 — REDAC prototype *(→ Goal 2: hardware usability)*

> **Core of Goal 2.** Here it is decided whether the fractal-inspired design is
> *usable* on analog-digital hybrid hardware — independent of whether the
> cosmic-neuronal analogy is "real".

**Hardware platform:** REDAC (Reconfigurable Analog Dynamic Computer).
Resources per unit: 432 multipliers, 864 integrators, 1,728 summation paths,
124,416 switch elements, 3,456 scaling elements. Calibration <1% deviation,
temperature compensation, Python interface.

**Division of labor (Kirchhoff principle):**
- **Analog (REDAC):** `Y·V` passively via Kirchhoff (crossbar + Ohm's law),
  `C·dV/dt` via integrators, `V_ε²` for ι via multipliers (~288 squarers).
- **Digital (Python/ADR-4):** compute `f = tanh(...)`, `J = ∂f/∂V`, inject via DAC
  as `I_pred`. The state machine orchestrates the token cycle.

**Sizing:**
- 1 REDAC: up to 288 clusters (864 integrators / 3 node types μ/ε/ι)
- 2 REDAC cascaded: 576 clusters → fits the full n_levels=4 configuration (510 clusters)

**Learning in stage 6:** digital (Hebbian rule in Python) → writes W into
the REDAC scaling elements. Proves **inference efficiency** on real analog hardware
(no memristor in-situ learning — that comes in stage 7).

**Clarifies:**
- Question 9 (ADC/DAC bandwidth, coarse-grained clocking)
- Question 16 (energy/token real vs. simulated)
- G20 (sneak paths: not present in the REDAC crossbar → cleaner model than the memristor)
- G14/G15 **directly solved** via REDAC calibration <1%
- G19 (integrator drift) **directly solved** via temperature stabilization

**Recycled:** All ADR-4 state-machine logic from stage 4/5 (Python stays).

### Stage 7 — REDAC + memristor extension *(→ Goal 2: hardware usability, in-situ learning)*
Replace the REDAC scaling elements with a memristor crossbar for selected
clusters (μ/ε/ι triplet). Proves in-situ plasticity (ADR-8).
Clarifies question 4 (memristor precision, drift, endurance), question 13 (endurance budget),
the bidirectional interface (ADR-4 read-back).

---

## 7. Glossary

- **Cluster / node:** compute unit (≈ neuron or attention head)
- **Filament:** connection between nodes (≈ synapse / weight)
- **Void:** empty space, no computation
- **Heatmap:** 3D density distribution that defines the topology
- **Predictive Coding (PC):** learning by minimizing prediction errors
- **Free energy:** Friston's energy measure that mathematically formalizes PC
- **Memristor:** component whose resistance changes depending on the current history
- **Hausdorff dimension:** measure of the fractal "space filling" (`d_H ≈ 2` for the cosmic web)
```
