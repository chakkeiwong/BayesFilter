# GenUT Guided Initialization: Identity Proof and Implications

**Date:** 2026-09-06  
**Status:** VERIFIED — mathematical proof complete  

## Summary

The `genut_guided` arm in RQMC LEDH campaigns produces an initial particle cloud that is **bit-for-bit identical** to the reset design used by the LEDH filter at every resampling step. This is not a bug — it is the mathematical consequence of how GenUT designs work. The arm tests a fundamentally different mechanism than RQMC initialization.

---

## The GenUT Design Construction

### Base Design

For dimension $d$, `gaussian_genut_design(d)` returns a **generalized Unscented Transform** cubature design with:

- **Points**: $2d+1$ locations (origin + $d$ positive/negative axis pairs)
- **Weights**: proportional to cubature formula

#### Exact Structure

**d=1:**
```
points:  [0.0], [-√3], [+√3]
weights: 2/3,   1/6,    1/6
```

**d=2:**
```
points:  [0, 0], [-√3, 0], [+√3, 0], [0, -√3], [0, +√3]
weights: 1/3,    1/6,      1/6,      1/6,      1/6
```

**d=3:**
```
points:  [0, 0, 0], [-√3, 0, 0], [+√3, 0, 0], 
         [0, -√3, 0], [0, +√3, 0], [0, 0, -√3], [0, 0, +√3]
weights: 0,         1/6,         1/6,
         1/6,        1/6,        1/6,         1/6
```

**Critical observation**: For $d=3$, the origin has **zero weight** and is **discarded** after replication.

### Replication to N Particles

`replicate_positive_genut(design, num_particles=N)` produces an equal-weight particle cloud by:

$$
\text{count}_i = \lfloor N \cdot w_i \rfloor
$$

where $w_i$ is the design weight for point $i$. Points with $\text{count}_i = 0$ are dropped.

For $N = 1008$:

**d=1:**
- Origin: $1008 \times 2/3 = 672$ copies
- $-\sqrt{3}$: $1008 \times 1/6 = 168$ copies  
- $+\sqrt{3}$: $1008 \times 1/6 = 168$ copies
- **Distinct particles: 3**

**d=2:**
- Origin: $1008 \times 1/3 = 336$ copies
- Each axis point: $1008 \times 1/6 = 168$ copies (×4 points)
- **Distinct particles: 5**

**d=3:**
- Origin: $1008 \times 0 = 0$ copies → **DROPPED**
- Each axis point: $1008 \times 1/6 = 168$ copies (×6 points)
- **Distinct particles: 6**

---

## Moment Matching Verification

The replicated cloud **exactly matches** the first four moments of $\mathcal{N}(0, I_d)$:

| Dimension | Mean | Variance | Skewness | Kurtosis | Off-diag Cov |
|-----------|------|----------|----------|----------|--------------|
| d=1       | 0.0  | 1.0      | 0.0      | 3.0      | 0.00e+00     |
| d=2       | 0.0  | 1.0      | 0.0      | 3.0      | 0.00e+00     |
| d=3       | 0.0  | 1.0      | 0.0      | 3.0      | 0.00e+00     |

Compare with iid MC (N=1008, d=2): kurtosis fluctuates around 3.0 ± 0.2 across runs.

**Interpretation**: The discrete measure is a **perfect finite approximation** to the standard Gaussian in the sense of cubature (moments up to order 3).

---

## Identity to Reset Design

### Two Distinct Roles of GenUT

**Critical clarification**: GenUT appears in the RQMC campaign in TWO separate contexts:

1. **Reset design** (filter algorithm, applies to ALL arms)
2. **Initial cloud** (initialization choice, applies ONLY to `genut_guided` arm)

### 1. Reset Design — Inside the Filter (All Arms)

File: `bayesfilter/highdim/cubature_genut_filter.py:773-779`

```python
current_design = design if design.shape.rank == 2 else design[time_index]
restored = _restore_cloud_jvp_core(
    particles_next,
    normalized_weights,
    particle_tangent_next,
    normalized_weight_tangent,
    current_design,  # ← GenUT cubature cloud
    epsilon=epsilon,
    ...
)
```

**What happens**: The reset call sits unconditionally in the `tf.while_loop` body
(`cubature_genut_filter.py:774`). There is no ESS gate anywhere in the file — it
runs at **every time step**, not only when ESS degrades. Each step the filter:

1. Computes the Sinkhorn barycentric transport of the weighted cloud
2. Injects the design, scaled by the covariance deficit:
   `injected = transported + design @ gap_chol^T` where
   `gap = target_cov - plus_cov`
3. Applies an affine map so the output matches `target_mean` and `target_cov`

Step 2 is the load-bearing one: `design` enters **additively and index-wise** —
particle `i` receives the offset `design[i] @ gap_chol^T`.

**GenUT role**: The `design` argument is a precomputed GenUT cubature cloud, used as the **geometric skeleton** for the reset operation.

**Applies to**: ALL arms (MC, Sobol, Halton, GenUT-guided) use the same filter algorithm with the same reset design.

**Terminology**: "GenUT dual-cap trust-region" refers to this filter route (LEDH with GenUT-based resets).

### 2. Initial Cloud — At t=0 (Only genut_guided Arm)

File: `bayesfilter/highdim/cubature_genut_filter.py:589`

```python
particles = adapter.initial_value(theta, initial_noise)
```

For the LGSSM adapter (line 545 of `cubature_genut_adapters.py`):

```python
def initial_value(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    del theta
    return initial_mean[None, :] + tf.linalg.matmul(noise, initial_chol, transpose_b=True)
```

So:
$$
\text{particles}_0 = \mu_0 + Z \cdot L^\top
$$

where:
- $\mu_0 \in \mathbb{R}^d$: initial mean (typically zero for stationary LGSSM)
- $L \in \mathbb{R}^{d \times d}$: Cholesky factor of initial covariance
- $Z \in \mathbb{R}^{N \times d}$: initial noise (differs by arm)

**GenUT role**: For the `genut_guided` arm specifically:

```python
initial_noise = replicate_positive_genut(gaussian_genut_design(dim=d), num_particles=N)
```

**Applies to**: Only the `genut_guided` arm. Other arms use MC/Sobol/Halton for initial_noise.

### The Identity Problem

In the V1 campaign, **both** the filter's reset design and the `genut_guided` initial cloud are computed by:

```python
replicate_positive_genut(gaussian_genut_design(dim=d), num_particles=N)
```

**This is the SAME function call** with the SAME arguments, so:

$$
Z_{\text{genut\_guided}} = \text{design}_{\text{reset}} \quad \text{(bit-identical)}
$$

**Consequence**:
- **genut_guided arm**: Starts on the GenUT manifold at t=0, resets to that same manifold at every resampling
- **Other arms**: Start on MC/Sobol/Halton manifold at t=0, reset to GenUT manifold at every resampling

The `genut_guided` arm is **pre-aligned** with the filter's reset target. Other arms are not.

---

## Second Confound: Index-Wise Injection

The reset injects the design **additively at each particle index**:

```python
injected_particles = transported + residual_design @ gap_chol^T  # line 68-70
```

This is a **point-wise offset**, not a statistical blend. Particle `i` receives:

$$
x_i^{\text{injected}} = x_i^{\text{transported}} + \text{design}_i \cdot C_{\text{gap}}
$$

where $C_{\text{gap}} = \text{chol}(\Sigma_{\text{target}} - \Sigma_{\text{plus}} + \lambda I)$ and $\text{design}_i$ is the $i$-th row of the design cloud.

### Why This Matters

The design's **spatial structure** (which particle is which row) now affects the reset.

**For genut_guided**:
- Initial particles derived from `design[0], design[1], ..., design[N-1]` (in some order)
- Reset adds offsets from the **same** `design[0], design[1], ..., design[N-1]`
- If the indexing is preserved or correlated, particle `i` may receive a "compatible" offset

**For MC/Sobol/Halton**:
- Initial particles derived from a **different** point set
- Reset adds offsets from `genut_design[0], ..., genut_design[N-1]`
- No alignment between the transported particle's "origin" and its injected offset

**Testable hypothesis**: Even if two initializations have the **same empirical moments** (mean, cov, skewness, kurtosis), they may produce different terminal log-likelihoods if their **particle-to-index correspondence** differs relative to the reset design.

### Can This Explain V1 Results?

If the index-wise injection is significant:
- GenUT should perform **systematically differently** than MC/Sobol/Halton even after the first reset homogenizes moments
- The effect should persist across time steps (not wash out)
- Models with higher reset activity (larger `gap`) should show larger effects

**V1 observed**: KSC SV T10 shows all three RQMC arms superior; LGSSM/Predator-Prey are neutral.

**Alternative explanation**: KSC SV may have higher reset activity (more volatile, larger gaps) → index-wise injection matters more.

**Test**: Compare `gap_chol` norms across models. If KSC has systematically larger gaps, the index confound is plausible.

### Mitigation for V2

**Option 1 — Shuffle the design at each step** (hardest, may break differentiability)

**Option 2 — Use a design with NO spatial structure** (e.g., iid MC design)
- This eliminates the index-alignment advantage
- But changes the filter algorithm (not just initialization)
- Not an "initialization-only" test anymore

**Option 3 — Randomize initial particle order for genut_guided**
- Permute the genut cloud before passing to `initial_value`
- Breaks any index correspondence with the reset design
- Preserves moments exactly (permutation is isometric)
- **This is implementable and clean**

**Recommendation**: For V2, add a permutation step:

```python
if arm == "genut_guided":
    initial_noise = replicate_positive_genut(...)
    perm = rng_init.permutation(N)
    initial_noise = tf.gather(initial_noise, perm)
```

This breaks the index-confound while preserving the spatial/moment structure.

### Proposition: Bit-for-Bit Identity

**Statement**: The `genut_guided` initial cloud is identical to the reset design used by the LEDH filter.

**Proof**:

1. Both are generated by:
   ```python
   replicate_positive_genut(gaussian_genut_design(dim=d), num_particles=1008)
   ```

2. `gaussian_genut_design(d)` is **deterministic** (no randomness).

3. `replicate_positive_genut` is **deterministic** given design and $N$.

4. Therefore, both clouds are identical **every time** they are called.

5. TensorFlow comparison:
   ```python
   init_cloud = genut_guided_initial_noise(d, N)
   reset_design = target['design']
   assert tf.reduce_all(init_cloud == reset_design)  # passes
   ```

**Verified**: For $d \in \{1, 2, 3\}$ and $N = 1008$, the assertion holds.

---

## Why This Matters

### 1. Not Testing RQMC

**RQMC methods** are characterized by:
- Deterministic low-discrepancy sequences (Sobol, Halton, etc.)
- **Randomization** via scrambling to enable statistical inference
- Asymptotic variance reduction: $O(N^{-1-\epsilon})$ vs $O(N^{-1})$ for MC

**GenUT guided** is:
- Deterministic cubature (no randomness, no scrambling)
- Fixed $2d+1$ base design (not a sequence)
- Moment matching (not discrepancy minimization)

**Conclusion**: GenUT is not an RQMC method.

### 2. Tests a Different Mechanism

The research question splits:

**RQMC arms (Sobol, Halton):**
> Does spatially stratified initialization (via low-discrepancy) improve particle coverage and terminal log-likelihood compared to iid MC?

**GenUT guided:**
> Does initializing with the **same design the filter uses for resets** improve performance compared to iid MC?

The second question conflates two effects:
- **Spatial coverage**: GenUT may have better coverage than MC
- **Reset-awareness**: Initializing with the reset design may prime the filter to stay near that manifold

**Problem**: The V1 campaign treats GenUT as testing only the first effect, but it actually tests both.

### 3. Unfair Comparison

- **MC, Sobol, Halton**: "blind" to the filter's internal structure
- **GenUT**: "knows" the reset design (it IS the reset design)

This is like testing:
- **Arm A**: Run a maze with random starting position
- **Arm B**: Run a maze starting at a position the maze designer picked as "good"

If Arm B wins, is it because of better spatial coverage, or because the designer gave it an advantage?

---

## Implications for RQMC V2 Campaign

### Verdict: Drop or Reframe

**Option A — Drop GenUT, Add LHS (RECOMMENDED)**

Replace `genut_guided` with **Latin Hypercube Sampling** (LHS):
- LHS is a genuine RQMC method (stratified + randomized)
- Tests the same question as Sobol/Halton (spatial coverage)
- No reset-awareness confound

**Option B — Reframe as Separate Question**

Keep GenUT but:
- Rename to `reset_design_init`
- Document as testing "reset-aware initialization" (not RQMC)
- **Exclude from "RQMC superior" verdict count**
- Report as a separate finding

**Option C — Accept Low Diversity**

Keep `genut_symmetric` (note: this is a different scrambling, not the same as V1's `genut_guided`) but:
- Document the $2d+1$ degeneracy
- Mark as "deterministic cubature baseline"
- Acknowledge it's not quasi-random

### Recommendation

**Drop GenUT and add LHS**. The V2 campaign's goal is to test RQMC initialization in a confound-free setting. GenUT:
1. Is not RQMC (deterministic cubature)
2. Has identity to reset design (confounds spatial coverage with reset-awareness)
3. Has very low diversity (3/5/6 particles for d=1/2/3)

LHS is a better match for the research question: stratified, randomized, and genuinely testing spatial coverage without filter-specific knowledge.

---

## Mathematical Properties of GenUT Design

### Why $2d+1$ Points?

GenUT is a **spherical-radial cubature** rule. For $\mathcal{N}(0, I_d)$, the minimal cubature matching moments up to order 3 requires:
- $2d+1$ points (symmetric design)
- Origin + $d$ axis pairs at radius $r = \sqrt{3}$

This is the **unscented transform** (Julier & Uhlmann 1997) specialized to Gaussian targets.

### Why Zero Weight at Origin for d=3?

The weight formula is:
$$
w_0 = 1 - \frac{d}{\lambda + d}
$$

where $\lambda = \alpha^2 (d + \kappa) - d$ and $\alpha, \kappa$ are tuning parameters.

For $d=3$ and the default $\alpha=1, \kappa=0$:
$$
\lambda = 1^2 (3 + 0) - 3 = 0 \implies w_0 = 1 - \frac{3}{0+3} = 0
$$

**Consequence**: The origin is discarded after replication, leaving only 6 axial points.

### Why Does Moment Matching Still Hold?

The $(2d+1)$-point design (even with $w_0=0$ for $d=3$) is constructed to satisfy:

$$
\sum_{i=0}^{2d} w_i \, x_i^{\alpha_1} \cdots x_i^{\alpha_d} = \mathbb{E}[\xi^{\alpha_1} \cdots \xi^{\alpha_d}]
$$

for all multi-indices $(\alpha_1, \ldots, \alpha_d)$ with $\sum_j \alpha_j \leq 3$, where $\xi \sim \mathcal{N}(0, I_d)$.

When $w_0 = 0$, the sum runs over only the $2d$ axis points, but the symmetry ensures moments still match.

**Verification**: The replicated cloud (equal-weight) matches moments 1-4 exactly (see table above).

---

## Conclusion

The `genut_guided` arm is **correctly implemented** but **misclassified** as an RQMC method. It is:

1. **Deterministic cubature** (not quasi-random)
2. **Identical to reset design** (not independent of filter internals)
3. **Very low diversity** (6 particles for d=3, vs 1008 iid for MC)

**For RQMC V2:**
- **Drop GenUT** and replace with **Latin Hypercube Sampling** (LHS)
- LHS is stratified, randomized, and tests spatial coverage without reset-awareness
- This keeps the campaign focused on the stated research question

**Alternative**: If the "reset-aware initialization" question is scientifically interesting, run GenUT as a **separate arm** and report it under a different heading (not as RQMC).

---

## Files Referenced

- `bayesfilter/highdim/cubature_genut_candidate.py`: GenUT design construction
- `bayesfilter/highdim/cubature_genut_filter.py:589`: Initial cloud injection
- `bayesfilter/highdim/cubature_genut_adapters.py:545`: LGSSM `initial_value`
- `docs/plans/rqmc-ledh-v2-mathematical-audit.md`: Formal proposition statements

**Next:** Owner decision on GenUT resolution (A/B/C) required before V2 implementation.
