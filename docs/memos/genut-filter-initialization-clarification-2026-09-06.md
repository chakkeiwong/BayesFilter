# GenUT Particle Filter Initialization: Complete Clarification

**Date:** 2026-09-06  
**Status:** EXPLANATION  
**Related:** genut-guided-identity-proof-2026-09-06.md, rqmc-ledh-v2-mathematical-audit.md  

## Your Question

"The GenUT particle filter is still a particle filter, right? If it is, at time step 0, it has to construct a set of particles. How is that initial set of particles constructed? According to the latex document, it seems to me that this is randomly initialized according to a Gaussian distribution and then through the generalized unscented transform, is that right?"

## Short Answer

**You are correct about the conceptual description**, but the implementation has a subtle detail:

1. **At t=0**: The filter receives `initial_noise` as an **input** (not generated internally)
2. **For MC/Sobol/Halton arms**: `initial_noise` is randomly drawn (MC) or quasi-randomly drawn (RQMC)
3. **For genut_guided arm**: `initial_noise` is the **GenUT cubature design** (3-7 deterministic points, replicated to N=1008)
4. **The adapter transforms**: `particles = initial_noise * stationary_std` (line 31 of cubature_genut_adapters.py)
5. **At t=1,2,...,T**: Every time step applies the generalized unscented transform (GenUT reset) to restore moments

## Detailed Code Trace

### 1. Entry Point: `finite_value_score` (cubature_genut_filter.py:537-596)

```python
def finite_value_score(
    adapter,
    theta,
    observations,
    initial_noise,    # ← Received as INPUT, not generated
    process_noise,
    design,           # ← GenUT design for reset (used at t≥0)
    ...
):
    particles = adapter.initial_value(theta, initial_noise)  # Line 589
    # ... time-stepping loop ...
```

The filter **does not generate** `initial_noise` internally. It receives it as an argument.

### 2. Adapter Transformation (cubature_genut_adapters.py:28-31)

For the LGSSM model:

```python
def initial_value(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    gamma, _, _ = physical(theta)
    std = sigma_tensor / tf.sqrt(1.0 - tf.square(gamma))
    return noise * std  # Transform standard normal → stationary distribution
```

So: **particles[i] = initial_noise[i] * std**, where `std` is the stationary standard deviation.

### 3. Where `initial_noise` Comes From (run_rqmc_ledh_initialization.py:113-182)

The campaign runner generates `initial_noise` differently for each arm:

```python
def _generate_initial_noise(arm, state_dim, seed, rng):
    if arm == "mc":
        return rng.normal([N, state_dim])  # iid Gaussian
    
    elif arm == "sobol_owen":
        # scipy Sobol with Owen scrambling → inverse CDF
        sobol = qmc.Sobol(d=state_dim, scramble=True, seed=seed)
        uniforms = sobol.random(N)
        return tf.constant(ndtri(uniforms))  # Standard normal RQMC
    
    elif arm == "halton_owen":
        # TFP Halton with randomization → inverse CDF
        uniforms = tfp.mcmc.sample_halton_sequence(...)
        return tf.math.ndtri(uniforms)
    
    elif arm == "genut_guided":
        return _genut_design(state_dim, rng)  # ← GenUT cubature design
```

### 4. The GenUT Design (run_rqmc_ledh_initialization.py:101-110)

```python
def _genut_design(dim, rng):
    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )
    if dim >= 18:
        return cubature_design(dim=dim, num_particles=N)
    return replicate_positive_genut(gaussian_genut_design(dim=dim), num_particles=N)
```

For d=3, this produces **6 distinct particles at ±√3 along each axis**, replicated to fill N=1008.

### 5. Time-Stepping Reset (cubature_genut_filter.py:774-785)

At **every time step** (not just t=0), the filter calls:

```python
restored = _restore_cloud_jvp_core(
    particles_next,          # After transition
    normalized_weights,
    ...,
    current_design,          # ← Same GenUT design used for ALL arms
    ...
)
```

This reset operation injects the GenUT design **additively** (ledh_contract_e_reset_tf.py:68-70):

```python
injected_particles = transported_particles + _apply_rows(
    residual_design,  # ← This is the GenUT design
    gap_chol          # ← Cholesky of covariance gap
)
```

So at every time step, **all arms** (MC, Sobol, Halton, GenUT) receive an **additive injection** of the GenUT design scaled by the covariance gap.

## Two Distinct Uses of GenUT

1. **Initial cloud (t=0, only genut_guided arm)**:
   - `initial_noise = GenUT design`
   - Produces 3-7 distinct particles (deterministic)
   - Only the `genut_guided` arm starts this way

2. **Reset operation (t≥0, all arms)**:
   - `injected = transported + design @ gap_chol^T`
   - Restores target mean and covariance
   - **All arms** (MC, Sobol, Halton, GenUT) use this

## The Confound

Because the `genut_guided` arm uses the **same GenUT design** for:
- Initial cloud construction (t=0), AND
- Reset injection (t≥0, all arms)

...the `genut_guided` particles start **already aligned** with the structure that will be injected at every subsequent time step. This creates a **moment pre-alignment confound**:

- MC/Sobol/Halton: Start from random/quasi-random cloud → reset pulls them toward GenUT structure
- GenUT-guided: Start **on** GenUT structure → reset is consistent with initialization

This tests "reset-aware initialization" rather than "RQMC spatial coverage."

## LaTeX Document Interpretation

The LaTeX document's description "randomly initialized according to a Gaussian distribution and then through the generalized unscented transform" is correct **for the conceptual algorithm**, but:

1. The "random initialization" is **provided as input** (`initial_noise`), not generated by the filter
2. The "generalized unscented transform" happens at **every time step** (t≥0), not just at initialization
3. For the `genut_guided` arm specifically, the initialization is **deterministic** (GenUT design), not random

## Recommendation

The `genut_guided` arm conflates two research questions:
1. **RQMC initialization quality** (intended by campaign design)
2. **Reset-aware initialization** (actual mechanism being tested)

Drop `genut_guided` from the V2 campaign and use Latin Hypercube Sampling instead. This produces a clean RQMC test with no pre-alignment confound.

---

## Code References

- Filter entry: [cubature_genut_filter.py:589](../bayesfilter/highdim/cubature_genut_filter.py#L589)
- Adapter transform: [cubature_genut_adapters.py:28-31](../bayesfilter/highdim/cubature_genut_adapters.py#L28-L31)
- Arm generation: [run_rqmc_ledh_initialization.py:113-182](../docs/benchmarks/run_rqmc_ledh_initialization.py#L113-L182)
- Reset injection: [ledh_contract_e_reset_tf.py:68-70](../bayesfilter/highdim/ledh_contract_e_reset_tf.py#L68-L70)
