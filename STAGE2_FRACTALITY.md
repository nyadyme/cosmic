# Stage 2 — Fractality Proof

> ⚠️ **SUPERSEDED by the Goal-2 revision (see CONCEPT.md Section 1).**
> Goal 2 no longer asks *whether* the cosmic-neuronal similarity is "real"
> (it is deliberately accepted as presumably coincidental), but only whether the
> fractal-inspired design is **usable** on hybrid hardware. The
> brain↔cosmos comparison described here (H₁/H₀) is therefore **no longer a project goal.**
>
> **What survives from this document:** the **measurement pipeline** (box-counting `d_H`,
> degree distribution `P(k)`, correlation function) — but as a **tool** for
> monitoring the *system's own* topology during structural plasticity (ADR-9),
> not as a comparison of two natural structures. The original comparison remains
> optional as a scientific addendum (publication), with no influence on the build.
>
> ---
>
> **Original purpose (historical, → old Goal 2 "Real Fractality"):**
> Quantitatively determine whether the similarity between neuronal clusters
> and the cosmic web is genuine cross-scale self-similarity or
> coincidence.

---

## 0. The Guiding Question as a Falsification Test

> **H₁ (Hypothesis):** Neuronal tissue and the cosmic web share the same
> statistical-geometric metrics over a common scale range.
>
> **H₀ (Null hypothesis):** The similarity is subjective; the metrics
> differ significantly.

We are **not** looking for "looks similar", but for agreement across
**at least three independent metrics**. If they do not agree, H₁ is
rejected and we know it early.

> **Important — state of the research:** This comparison has already been published.
> Vazza & Feletti (2020), *"The Quantitative Comparison Between the Neuronal
> Network and the Cosmic Web"*, Frontiers in Physics, found astonishing
> agreement in the power spectrum and in connectivity over a
> limited scale range. Our Stage 2 is therefore **reproduction + extension**,
> not new territory. The exact figures of that work must be verified before use.

---

## 1. The Three (+1) Metrics

| # | Metric | What it measures | Equality means |
|---|-----------|---------------|---------------------|
| 1 | **Fractal dimension `D`** | How space is filled with scale (box-counting) | Same "clumpiness" across scales |
| 2 | **Degree distribution `P(k)`** | How many connections the nodes have | Same connectivity statistics |
| 3 | **Two-point correlation `ξ(r)`** | Overdensity as a function of distance | Same clustering tendency |
| + | **Power spectrum `P(q)`** | Distribution of structure sizes (Fourier) | Same dominant scales (the Vazza & Feletti metric) |

Optional / deeper study: **multifractal spectrum** `f(α)` and **lacunarity** `Λ(r)`
— these distinguish a "true monofractal" from "multi-scale mixed behaviour".

---

## 2. Data Sources

### 2a. Cosmic Web

| Source | Type | Access |
|--------|-----|--------|
| **IllustrisTNG** | Simulation (density fields, halo catalogs) | public, API |
| **Millennium Simulation** | Simulation (halo catalogs) | public |
| **SDSS Galaxy Catalog** | Observation (3D galaxy positions) | public |

→ To start with: **halo/galaxy positions as a 3D point cloud** is enough for all
four metrics.

### 2b. Neuronal Tissue

| Source | Type | Access |
|--------|-----|--------|
| **MICrONS** | 3D electron-microscopy connectome (mouse cortex) | public |
| **FlyWire / Drosophila connectome** | complete insect connectome (graph) | public |
| **Allen Brain Atlas** | histological density images | public |
| **Neuron reconstructions (NeuroMorpho.org)** | individual cell morphologies | public |

→ To start with: **3D positions of neuronal somata (cell bodies) as a point cloud**
plus a **synapse graph** for `P(k)`.

> **Scale trap (critical):** The cosmic web spans ~megaparsecs,
> neuronal tissue spans ~micrometres — **27 orders of magnitude**. A fair comparison
> normalises both structures to **dimensionless units** (e.g. mean
> node spacing = 1). What is compared is the **shape** of the curves, not absolute values.

---

## 3. Methods per Metric

### 3.1 Fractal Dimension — Box-Counting

```
1. Normalise the point cloud into the unit cube [0,1]³
2. Subdivide the cube into N³ boxes of edge length ε = 1/N
3. Count the number of occupied boxes  N(ε)
4. Plot  log N(ε)  against  log(1/ε)
5. Slope of the line = box-counting dimension D
```

Comparison: `D_neuro` vs. `D_kosmos`. Bonus: over which ε range is the line
straight (= the scale range in which the fractality holds)?

### 3.2 Degree Distribution `P(k)`

```
1. Build the graph (synapses or filament connections as edges)
2. Count the degree k of each node
3. Form the histogram P(k), plot log-log
4. Test for a power law P(k) ~ k^(-γ), estimate the exponent γ
```

Comparison: `γ_neuro` vs. `γ_kosmos`. (Concept assumption so far: `γ ≈ 2–3`.)

### 3.3 Two-Point Correlation Function `ξ(r)`

```
1. Compute the pairwise distances of all nodes
2. Compare with a random distribution of equal density (Landy-Szalay estimator)
3. ξ(r) = (DD - 2DR + RR) / RR
4. Test for a power law ξ(r) ~ (r/r₀)^(-γ_ξ)
```

Comparison: slope `γ_ξ` and correlation length `r₀` of both structures.

### 3.4 Power Spectrum `P(q)` (Vazza–Feletti metric)

```
1. Point cloud → density field on a 3D grid (Cloud-in-Cell)
2. 3D FFT → Fourier amplitudes
3. Radially average → P(q) as a function of the wavenumber q
4. Plot log-log, compare the shapes
```

This is the metric in which the original work found the strongest agreement.

---

## 4. Decision Criterion

H₁ is considered **supported** if:

- the **scale range of shared fractality** spans at least ~1 decade, **and**
- at least **3 of the 4 metrics** agree within the error bars.

H₁ is considered **refuted** if:

- the curve shapes are qualitatively different (e.g. power law vs. exponential), **or**
- the fractal dimension differs by more than ~0.5.

> **Consequence for the project:** Even if H₁ (Goal 2) is refuted,
> **Goal 1 (practicability)** remains valid — a fractal NN can be technically useful
> even if the cosmic-neuronal analogy is merely poetic. Stage 2
> therefore decides on the **justification** of the topology, not on the whole project.

---

## 5. Recycling from the Stash

| Artefact | Use in Stage 2 |
|----------|------------------------|
| `cosmic_web_generator.py` | Provides a **synthetic** control point cloud with a *known* `d_H` → validates our measurement pipeline before we touch real data |
| `fractal_llm.md` (Hausdorff scaling) | Theoretical reference for the expected `D ≈ 2` |

> **Ordering trick:** First test the measurement routines (3.1–3.4) against the
> generator output, whose `d_H` we prescribe. If the pipeline reproduces the known value,
> it is trustworthy for real data.

---

## 6. Work Steps (concrete)

1. **Build the measurement pipeline** (Python): four functions `box_counting`,
   `degree_distribution`, `two_point_correlation`, `power_spectrum`.
2. **Validate the pipeline** against the `cosmic_web_generator.py` output (known `d_H`).
3. **Fetch real data**: one cosmic and one neuronal point cloud each (small).
4. **Measure both**, overlay the curves, tabulate the metrics.
5. **Document the decision** against the criteria from Section 4.

---

## 7. References (verify before use)

- Vazza, F. & Feletti, A. (2020). *The Quantitative Comparison Between the
  Neuronal Network and the Cosmic Web.* Frontiers in Physics, 8, 525731.
- Landy, S. D. & Szalay, A. S. (1993). *Bias and variance of angular correlation
  functions.* The Astrophysical Journal, 412, 64.
- Nelson et al. (2019). *The IllustrisTNG simulations: public data release.*
