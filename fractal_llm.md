# Fractal Predictive-Coding Network on Cosmic Topology for Analog-Digital Hybrid Computers

---

## Abstract

This work describes a novel architecture for learning neural networks that differs fundamentally from the static layer architectures of conventional language models. Instead of uniform, densely populated weight matrices, the network topology is derived from the self-similar structure of cosmic filaments: galaxy clusters at the nodes of the cosmic web form the computational units; the filaments connecting them form the synaptic paths. This topology follows a power-law degree distribution `P(k) ~ k^{-γ}` with `γ ≈ 2–3` and a Hausdorff dimension `d_H ≈ 2`, which leads to a naturally hierarchical, sparse connectivity.

The learning principle is based on the **Free Energy Principle** of Karl Friston. Each cluster on level `l` continuously minimizes the local prediction error `ΔV^(l) = V^(l) − V̂^(l)` against the prediction of the level above. The inference dynamics follow the differential equation `C^(l) · dV^(l)/dt = −G^(l) · ΔV^(l) + G^(l−1) · ΔV^(l−1) · J^(l) − Y_lat^(l) · V^(l)`, where precisions are represented as node conductances `G^(l)`. The synaptic weights are updated by a local Hebbian learning rule `dW^(l)_ij/dt = η · G^(l−1) · ΔV^(l−1)_i · V^(l)_j · f_w(W_ij)` — without global error backpropagation.

The architecture is formulated explicitly for **analog-digital hybrid computers** with memristors. **Four** time scales operate asynchronously: error settling ε `(τ_ε = τ_leaf)`, interneuron precision estimation ι `(τ_ι = √(τ_leaf · τ_μ))`, representation μ `(τ_μ ≈ 1–100, level-dependent)`, and physical memristor plasticity `(τ_plast ≈ 1–1000 ms)`. Each cluster contains three node types: μ (representation), ε (error), ι (interneuron/gain control) — a trisynaptic ganglion (ADR-12). State variables correspond directly to measurable voltages, weights correspond to memristor conductances, precisions correspond to node conductances. The prediction function `V̂^(l) = σ_LUT(Σ_j W^(l)_j · Y_lm(r_j) · V^(l+1)_j)` uses precomputed spherical harmonics of the fractal geometry and is injected by the digital part via DAC.

The complete network dynamics are described by the **nodal admittance matrix** `Y = L_W + diag(G^(l))`, where `L_W` is the weighted graph Laplacian of the cosmic filament topology. The analog network solves the associated linear system of equations `C · dV/dt = −Y · V + I_pred` physically through current balancing according to Kirchhoff. The fractal sparsity of `Y` — with fill ratio `O(N log N)/N²` — makes the architecture hardware-efficient and scalable. The minimal eigenvalue `λ_min(Y) ~ N^{-1/d_H}` determines the settling time and couples network depth directly to physical time constants.

The topology is generated algorithmically by a **Soneira–Peebles generator** (`cosmic_web_generator.py`). Each cluster contains three node types — μ (representation), ε (error), ι (interneuron/gain control) — a trisynaptic ganglion (ADR-12). The interneuron ι estimates the local error variance in analog form and partially replaces the digital precision update. Filament connections are set via k-nearest neighbors with Hausdorff-scaled conductance `W_ij ~ r_ij^{-(d_H − 1)}`. The resulting sparse admittance matrix `Y` over `3 × n_cluster` nodes is output as a `scipy.sparse.csr_matrix`.

---

## Mathematical Foundations

### 0. Free-Energy Functional F (explicit) — correction basis for §1–5

The functional is written out here for the first time. All three dynamics follow as
exact gradients from it — this guarantees sign and consistency.

**Extended hierarchical PC functional (incl. lateral precision, G30):**

```
F = Σ_l [ ½ · Π^(l) · ||ε^(l)||²  +  ½ · μ^(l)ᵀ · Ω^(l) · μ^(l) ]
```

- `ε^(l) = μ^(l) − f(μ^(l+1), θ^(l+1))` — raw prediction error (not scaled)
- `Π^(l) > 0` — scalar precision (inverse variance), not variance
- `Ω^(l)` — lateral precision matrix of level `l` (corresponds to the k-NN
  μ-μ filaments of the generator; implements within-level whitening, G30)

**Derivatives (verifiable):**

∂F/∂μ^(l) = `Π^(l) · ε^(l)` − `Π^(l−1) · ε^(l−1) · J^(l)` + `Ω^(l) · μ^(l)`

→ Inference dynamics (gradient descent):
```
dμ^(l)/dt = −∂F/∂μ^(l) = −Π^(l)·ε^(l) + Π^(l−1)·ε^(l−1)·J^(l) − Ω^(l)·μ^(l)
```

∂F/∂θ^(l) = `−Π^(l−1) · ε^(l−1) · ∂f(μ^(l), θ^(l))/∂θ^(l)`

→ Learning rule (gradient descent):
```
dθ^(l)/dt = −η · ∂F/∂θ^(l) = η · Π^(l−1) · ε^(l−1) · ∂f/∂θ^(l)
```

∂F/∂Π^(l) = `½ · (ε^(l))² − 1/(2Π^(l))` (incl. log-normalizer of the Gaussian)

→ Precision update (gradient descent):
```
dΠ^(l)/dt = −η_Π · ∂F/∂Π^(l) = −η_Π · [ Π^(l) · (ε^(l))² − 1 ] / (2 Π^(l))
```
**Fixed point:** `Π^(l) · (ε^(l))² = 1  →  Π^(l) = 1 / (ε^(l))²`
(Precision = **inverse** variance — Friston standard.)

---

### 1. The prediction error (filament voltage)

At each node (cluster) of level `l`, the actual state is compared with the prediction of the level above `l+1`:

```
ε^(l) = μ^(l) − f(μ^(l+1), θ^(l+1))
```

- `μ^(l)` — current state (activity) of the cluster on level `l`
- `f(...)` — nonlinear prediction function, generated from the higher level
- `θ^(l+1)` — synaptic weights (memristor conductances) of the filaments

### 2. Inference dynamics (state estimation)

```
dμ^(l)/dt = −∂F/∂μ^(l) = −Π^(l)·ε^(l) + Π^(l−1)·ε^(l−1)·J^(l) − Ω^(l)·μ^(l)
```

- **First term `−Π^(l)·ε^(l)`:** pulls μ toward the prediction f (stabilizing, not +ε)
- **Second term:** error of the level below, weighted by the Jacobian J^(l) = ∂f/∂μ^(l)
- **Third term `−Ω^(l)·μ^(l)`:** lateral inhibition (whitening within the level)
- `F` — explicit free-energy functional, see §0

**Sign check:** ε^(l) = μ^(l) − f > 0 → first term negative → μ decreases toward f ✓

### 3. Learning rule (synaptic plasticity)

```
dθ^(l)/dt = η · ε^(l−1) · ∂f(μ^(l), θ^(l))/∂θ^(l)
```

- `η` — learning rate
- Learning happens entirely locally, without a central computational unit

### 4. Prediction function for 3D fractals (learnable basis, G28)

**Complete form (universality guaranteed):**
```
f(μ^(l+1), θ^(l+1)) = σ( Σ_{j, n, l, m}  θ_{j,n,l,m}^(l+1) · R_n(r_j) · Y_lm(θ_j, φ_j) · μ_j^(l+1) )
```

- `R_n(r_j)` — radial basis functions (e.g., spherical Bessel, ADR-7)
- `Y_lm(θ_j, φ_j)` — spherical harmonics (angular component)
- `θ_{j,n,l,m}` — **fully learnable**: θ selects on its own which (n,l,m) combinations are relevant
- `σ` — nonlinear activation (e.g., tanh)

**Why a learnable basis (Option C instead of fixed Y_lm):**
A frozen single harmonic creates structural zeros — prediction directions that θ can never learn (universality breakdown, G28). With a learnable θ_{j,n,l,m} the basis is complete: θ can set any harmonic to zero (= frozen) or weight it freely. Finding B-1 (universality) thus holds without restriction.

**Implementation note (stage 4):** For the toy problem the basis can be restricted to low l (e.g., l ≤ 2) — θ then learns the relevant subset. For stage 5, the full basis.

### 5. Precision weighting (variance control)

`ε^(l) = μ^(l) − f(μ^(l+1), θ^(l+1))` remains the **raw** error (not scaled by Π).
Π appears as a weighting factor in the dynamics (§2) and in the functional (§0).

**Precision update (derived from §0, fixed point = inverse variance):**
```
dΠ^(l)/dt = −η_Π · [ Π^(l) · (ε^(l))² − 1 ] / (2 Π^(l))
```
Fixed point: `Π^(l) = 1 / (ε^(l))²`  (precision = **inverse** variance, not variance)

Numerically stable discretization with G_min clipping:
```
Π^(l)[t+1] = clip( Π^(l)[t] · (1 − η_Π · (ε^(l))²) + η_Π / Π^(l)[t],
              G_min, G_max )
```

**Lateral precision matrix Ω^(l) (G30 — new):**
The k-NN μ-μ filaments of the generator map onto Ω^(l) and implement
within-level whitening: neighboring μ nodes inhibit each other,
so that the representation of a level is decorrelated. Ω^(l) appears
in the functional F (§0) and in the inference term `−Ω^(l)·μ^(l)` (§2).

### 6. Interneuron dynamics (ι node, ADR-12)

Per cluster a third node type ι — an analog precision estimator.
It partially replaces the digital G update (§5) with a local analog circuit.

**ι dynamics (nonlinear, quadratic coupling):**
```
C_ι · dV_ι/dt = −G_ι · V_ι  +  G_ει · (V_ε)²
```

- `G_ι` — shunt conductance of the interneuron (leak term, keeps V_ι bounded)
- `G_ει` — coupling conductance ε→ι (quadratic: error magnitude, not sign)
- `(V_ε)²` — quadratic error signal (always positive → no sign problem)

**Equilibrium:** `V_ι* = (G_ει / G_ι) · ε²`  →  V_ι estimates the error variance.

**Effective precision:**
```
G_eff(c) = G_scale / (V_ι(c) + ε₀)
```
- Small V_ι (small error) → large G_eff → strong prediction restoring force
- Large V_ι (large error) → small G_eff → the network opens up for corrections

`G_eff` replaces the fixed `G_prec` in the μ dynamics:
```
C_μ · dV_μ/dt = −G_eff(ι) · ΔV_μ + G_eff(ι)^(l−1) · ΔV_ε^(l−1) · J^(l) − Y_lat · V_μ
```

**Time constant ι** (geometric mean, see generator explanation):
```
τ_ι = √(τ_leaf · τ_μ_l)
```
Positions ι symmetrically between ε (fast) and μ (slow) on a logarithmic scale.

**Intra-cluster edges in the generator:**
- `μ ↔ ε` — W_intra (prediction-error coupling)
- `ε ↔ ι` — W_eps_iota (drives the interneuron)
- `ι ↔ μ` — W_iota_mu (gain modulation, admittance in Y)

**What ι solves:**
- G26 (precision sign): ι estimates ε² locally → G_eff ∝ 1/ε² emerges in analog form
- G6 (normalization): ι dampens on large error → natural gain control
- G15 (G mismatch): dynamic G_eff is more robust than fixed G_prec

---

## Hybrid-Optimized Formulation

### Physical variable substitution

| Abstract    | Physical                            | Reason                             |
|-------------|-------------------------------------|------------------------------------|
| `μ^(l)`     | `V_μ` — voltage of representation node  | Kirchhoff-native                   |
| `ε^(l)`     | `V_ε = V_μ − V̂` — error-node voltage | Directly measurable as a difference |
| `ι^(l)`     | `V_ι` — interneuron voltage (ADR-12)   | analog precision estimator         |
| `θ^(l)`     | `W^(l)` — memristor conductance [S]    | Conductance instead of resistance: W = 1/R |
| `Π^(l)`     | `G_eff(ι)` — effective conductance [S] | dynamic via ι, not fixed           |

### Four time scales (asynchronous, ADR-12)

```
τ_ε      = τ_leaf           →  eps settles per token (fast, error)
τ_ι      = √(τ_leaf · τ_μ)  →  iota estimates precision (geometric mean)
τ_μ      ≈ 1–100            →  mu represents / remembers (level-dependent)
τ_plast  ≈ 1–1000 ms        →  memristor weight update (very slow)
```

Node-index convention: `μ(c) = 3c`,  `ε(c) = 3c+1`,  `ι(c) = 3c+2`

### Analog equations (Kirchhoff-ready)

Node dynamics (fully analog, capacitor as integrator):
```
C^(l) · dV^(l)/dt = −G^(l) · ΔV^(l)  +  G^(l−1) · ΔV^(l−1) · J^(l)  −  Y_lat^(l) · V^(l)
```

- `C^(l)` — node capacitance (τ_l = C^(l) / G^(l), ADR-2)
- **`−G^(l) · ΔV^(l)`** — drives V toward the prediction V̂ (stable leak, **negative sign**)
- `J^(l) = ∂f/∂V^(l)` — Jacobian, computed digitally and injected via DAC
- `Y_lat^(l)` — lateral μ-μ admittance matrix of level l (= Ω^(l) from §5)

**Equivalence to the full ODE:** `C · dV/dt = −Y_eff · V + I_pred`  
with `Y_eff = L_W + diag(G) + Y_lat` and `I_pred_i = G_i · V̂_i`

**Contractivity condition (G27):** The system is Hurwitz-stable when:
```
λ_max( diag(G) · J_f ) < λ_min(Y_eff)
```
with `||J_f||₂ ≤ ||W||₂` (tanh activation, ||σ'||∞ ≤ 1).
The ADR-9 spring term keeps `||W||₂ ≈ ||W(r)||₂` (geometrically bounded).
Sufficient condition: `G_max · ||W(r)||₂ < G_min`.
→ To be verified empirically in stage 4.

Memristor learning rule (Strukov model, local):
```
dW^(l)_ij/dt = η · G^(l−1) · ΔV^(l−1)_i · V^(l)_j · f_w(W^(l)_ij)
```

- `f_w(W) = W(1 − W)` — Joglekar window function for boundary effects

### Digital equations

Precision update (discrete, digitally clocked, derived from §0):
```
G^(l)[t+1] = clip(
    G^(l)[t] − α · ( G^(l)[t] · (ΔV^(l))² − 1 ),
    G_min, G_max
)
```
Fixed point: `G^(l) = 1 / (ΔV^(l))²` (inverse variance). G_min prevents τ→∞,
G_max prevents precision runaway at small error (G26).

Prediction function (digital, per time step):
```
V̂^(l) = σ_LUT( Σ_j W^(l)_j · Y_lm(r_j) · V^(l+1)_j )
```

### ADC/DAC interface protocol

```
Analog → ADC:  V^(l), ΔV^(l)           (read)
Digital:       compute f, J^(l), G^(l+1)
DAC → Analog:  V̂^(l), J^(l), G^(l)     (write)
```

### Normalization

```
V̄^(l) = V^(l) / V_ref  ∈ [−1, +1]
W̄^(l) = W^(l) / G_ref  ∈ [0, 1]     (memristor bounds)
```

---

## Admittance Matrix of the Fractal Topology

### Basic structure

For `N` nodes (clusters) with filament connections, Kirchhoff's law holds at each node `i`:

```
Σ_j W_ij · (V_i − V_j)  +  G_i · V_i  =  I_i
```

In matrix form:
```
Y · V = I_pred

Y = L_W + diag(G)
```

Matrix elements:
```
        ⎧  Σ_j W_ij + G_i   if i = j
Y_ij =  ⎨  −W_ij            if i ≠ j, (i,j) ∈ filaments
        ⎩   0               otherwise
```

Complete dynamics:
```
C · dV/dt = −Y · V + I_pred(V, W)
```

### Hierarchical block structure

For `L` hierarchy levels:
```
Y_total = Y_local + Y_inter
```

Local couplings (block diagonal):
```
         ⎡ Y^(1)    0       0   ⎤
Y_local =⎢  0      Y^(2)    0   ⎥
         ⎣  0       0      Y^(L)⎦
```

Inter-level couplings (filaments between levels):
```
           ⎡   0     −W^(12)    0    ⎤
Y_inter =  ⎢ −W^(21)    0    −W^(23) ⎥
           ⎣   0     −W^(32)    0    ⎦
```

### Hausdorff scaling of the conductances

```
W^(l)_ij  ~  r_ij^{−(d_H − 1)}    with  d_H ≈ 2
         →   r_ij^{-1}
```

Node count per level:
```
N_l ~ r_l^{d_H} = r_l²
```

Total number of filaments:
```
|E| ~ N · log N    (sparse regime)
```

### Eigenstructure and settling time

```
τ_settle  =  C / λ_min(Y)

λ_min(Y)  ~  N^{−1/d_H}  +  G_min
```

- `G_min` — minimal node conductance (prevents τ → ∞)
- Larger `N` → slower inference: physical speed-depth tradeoff

### Sparsity as a hardware advantage

| Property             | Dense matrix  | Cosmic fractal     |
|----------------------|---------------|--------------------|
| Matrix elements      | `N²`          | `O(N log N)`       |
| Memristors required  | `N²`          | `N · log N`        |
| Fill ratio at N=10⁴  | 100 %         | ≈ 0.13 %           |
| Fill ratio at N=10⁶  | 100 %         | ≈ 0.002 %          |

---

## Topology Generator (Soneira–Peebles)

### Algorithm

The generator in `cosmic_web_generator.py` (ADR-12 version) creates the
trisynaptic fractal topology in five steps (μ/ε/ι per cluster):

**Step 1 — Recursive point cloud:**
Starting from `n_top` top-level clusters, each node is decomposed into `η` sub-clusters
with radius `r_{l+1} = λ · r_l`:

```
d_H = log(η) / log(1/λ)
```

For `η = 4`, `λ = 0.5` this gives `d_H = log(4)/log(2) = 2.0` exactly.

**Step 2 — Filaments (intra-level):**
k-nearest neighbors within each level, conductance Hausdorff-scaled:
```
W_ij = max( r_ij^{-(d_H - 1)},  W_min )
```

**Step 3 — Filaments (inter-level):**
Connections between neighboring levels `l` and `l+1`,
same conductance scaling.

**Step 4 — Node expansion (ADR-12):**
One μ/ε/ι triplet per cluster. Node index: `μ(c)=3c`, `ε(c)=3c+1`, `ι(c)=3c+2`.
Intra-cluster edges: μ↔ε (W_intra), ε↔ι (W_eps_iota), ι↔μ (W_iota_mu).

**Step 5 — Build the admittance matrix:**
```python
Y = L_W + diag(G_prec)    # scipy.sparse.csr_matrix, 3*n_cluster nodes
```

### Configuration parameters (ADR-12 version)

| Parameter       | Default  | Meaning                                        |
|-----------------|----------|------------------------------------------------|
| `n_levels`      | 3-4      | Number of hierarchy levels L                   |
| `eta`           | 3-4      | Sub-clusters per parent cluster                |
| `lambda_scale`  | 0.5      | Angular self-similarity → controls d_H         |
| `n_top`         | 4-6      | Top-level clusters (core)                      |
| `tau_leaf`      | 1.0      | Leaf time constant (fast, inference)           |
| `tau_top`       | 20-100   | Core time constant (slow, memory)              |
| `k_neighbors`   | 3        | k-NN lateral filaments per level               |
| `G_min`         | 0.1      | Precision lower bound (simulation: 0.1)        |
| `G_iota`        | 0.1      | Shunt conductance of ι node                    |
| `W_eps_iota`    | 0.5      | ε→ι coupling conductance                       |
| `W_iota_mu`     | 0.3      | ι→μ gain-modulation conductance               |

### Output

```python
web = generate_cosmic_web(cfg)

web.G           # NetworkX graph (mu/eps/iota nodes + attributes)
web.Y           # scipy.sparse.csr_matrix  →  directly usable in ODE solver
web.positions   # np.ndarray [3N, 3]       node positions
web.levels      # np.ndarray [3N]          hierarchy level of each node
web.kinds       # np.ndarray [3N]          0=mu, 1=eps, 2=iota
web.tau         # np.ndarray [3N]          time constant per node
web.radius      # np.ndarray [3N]          distance from center
web.d_H_angular # float                    angular d_H reference value
web.N           # int                      total number of nodes (= 3*n_clusters)
web.n_clusters  # int                      number of clusters
web.n_filaments # int                      number of edges
web.fill_factor # float                    fill ratio of the Y matrix
```

### Example output (ADR-12, n_top=4, eta=3, n_levels=3)

```
Clusters             : 52
Nodes N (mu+eps+iota): 156
of which mu/eps/iota : 52 / 52 / 52
Filaments |E|        : 305
d_H (angular, ref.)  : 1.58
Fill ratio           : 0.1292 %
Level 1: 12 nodes | r=0.150 | tau_mu=100.00  tau_iota=10.00
Level 2: 36 nodes | r=0.433 | tau_mu=21.54   tau_iota=4.64
Level 3: 108 nodes | r=0.900-1.100 | tau_mu=1.00  tau_iota=1.00
```

---

## References

[1] Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

[2] Strukov, D. B., et al. (2008). The missing memristor found. *Nature*, 453(7191), 80–83.

[3] Bond, J. R., Kofman, L., & Pogosyan, D. (1996). How filaments are woven into the cosmic web. *Nature*, 380(6575), 603–606.

[4] Joglekar, Y. N., & Wolf, S. J. (2009). The elusive memristor. *European Journal of Physics*, 30(4), 661.

[5] Soneira, R. M., & Peebles, P. J. E. (1978). A computer model universe. *The Astrophysical Journal*, 211, 1–15.
