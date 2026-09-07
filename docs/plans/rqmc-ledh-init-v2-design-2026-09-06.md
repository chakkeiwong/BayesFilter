# RQMC LEDH Initialization: V2 Campaign Design

**Date:** 2026-09-06  
**Status:** DESIGN — not approved for execution  
**Supersedes:** rqmc-ledh-initialization-master-program-2026-09-02.md (invalidated by confound)  

## Motivation

The V1 campaign (45 runs, 3 models, 5 arms) is scientifically invalid due to:
1. **Process-noise confound**: MC and RQMC arms received different process-noise trajectories for the same seed
2. **GenUT degeneracy**: 3-6 distinct particles out of 1008
3. **Insufficient replication**: n=3 seeds too small for statistical power

See [rqmc-ledh-confound-analysis-2026-09-06.md](../memos/rqmc-ledh-confound-analysis-2026-09-06.md) for root-cause analysis.

This V2 design corrects all three defects.

---

## Research Question

**Does RQMC initialization improve LEDH particle filter performance compared to MC initialization, when all subsequent dynamics are identical?**

### Mechanism Under Test

Particle cloud initialization at t=0 only. The mechanism is **purely spatial**: how well does the initial cloud cover the latent state distribution at t=0?

### Not Under Test

- Process-noise stratification (would require RQMC trajectories, not just initialization)
- Resampling sequence (held constant across arms)
- Transport/likelihood computation (identical for all arms)
- Hardware/numerics (same device, dtype, XLA status)

### Success Criterion

**Primary**: terminal log-likelihood (higher is better). At least one RQMC arm has bootstrap 95% CI for (RQMC − MC) entirely above zero on at least one model.

### Promotion Veto

Any of:
- RQMC arm produces NaN, non-finite, or invalid result
- RQMC arm statistically inferior to MC (CI entirely below zero) on any model
- Process-noise trajectories differ across arms (confound check fails)

### Continuation Veto

Any of:
- Runner produces different process-noise for same seed across arms
- Particle cloud has <95% distinct particles (degeneracy check fails)
- More than 10% of planned cells fail infrastructure

---

## V2 Design: Six Key Changes

### 1. Independent RNG Seeding (Confound Fix)

**V1 defect**: One `tf.random.Generator` for both initial and process noise. MC advanced the stream; RQMC did not.

**V2 fix**: Use `numpy.random.SeedSequence` to derive independent seeds:

```python
def _run_evaluation(model, arm, seed):
    ss = np.random.SeedSequence(seed)
    initial_seed, process_seed = ss.spawn(2)
    
    initial_rng = replication_generator(initial_seed.generate_state(1)[0])
    process_rng = replication_generator(process_seed.generate_state(1)[0])
    
    initial_noise = _generate_initial_noise(arm, state_dim, seed, initial_rng)
    process_noise = _generate_process_noise(horizon, state_dim, seed, process_rng)
```

This guarantees process noise is **deterministic given seed** but **independent of initial-cloud generation method**.

**Verification**: Add runtime check that `hash(process_noise.numpy().tobytes())` is identical across all arms for the same seed.

### 2. GenUT Confounds (Three Issues)

The `genut_guided` arm has **three separate confounds** relative to the stated research question:

#### Confound 1: Moment Pre-Alignment (Unfixable)
- `genut_guided` initial cloud is **bit-identical** to the filter's reset design
- Starts on the GenUT manifold at t=0; other arms do not
- Tests "reset-aware initialization" not "RQMC spatial coverage"
- **Cannot be fixed** without changing the filter algorithm or dropping the arm

#### Confound 2: Index-Wise Injection (Fixable with Permutation)
- The reset injects design **additively at each particle index** (every time step, unconditionally):
  ```python
  injected[i] = transported[i] + design[i] @ gap_chol^T
  ```
- Particle `i` receives offset from `design[i]` specifically
- If `initial_cloud[i]` and `design[i]` are "related" by construction, the offset may be "compatible"
- MC/Sobol/Halton have no such index correspondence

**Mitigation** (if keeping GenUT):
```python
if arm == "genut_guided":
    initial_noise = replicate_positive_genut(...)
    perm = rng_init.permutation(N)
    initial_noise = tf.gather(initial_noise, perm)
```
This breaks index-correspondence while preserving moments and spatial structure.

#### Confound 3: Low Diversity (Architectural)
- V1 produced only 3-6 distinct particles out of N=1008 (dimension-dependent)
- Deterministic cubature, not quasi-random
- 168-336 copies per unique point

**V2 Option A** (mechanical fix): Full symmetric design
```python
design = gaussian_genut_design(dim=state_dim)  # [2d+1, d]
full = tf.tile(design, [N // (2*d+1), 1])
# shuffle to avoid systematic ordering
```
This gives `n_distinct = min(N, 2*d + 1)`. For d=1,2,3: 3, 5, 7 distinct particles.
**Still low diversity**, but every particle is a designed cubature point.

**V2 Option B** (Strongly Recommended): Drop GenUT, add Latin Hypercube Sampling
- LHS is genuine RQMC (stratified + randomized)
- No reset-awareness confound (#1)
- No index-injection confound (#2)
- Proper diversity: N distinct points (#3)
- Cleaner interpretation: tests RQMC coverage without filter-specific alignment

**Degeneracy gate** (if keeping GenUT): Before evaluating, check `n_distinct >= min(0.95*N, 2*d+1)` for genut. Fail the run if violated.

---

**Recommendation Strengthened**: Drop `genut_guided` entirely. Confound #1 is unfixable, and even with permutation fix for #2, the arm conflates two research questions. Option B (LHS) is genuine RQMC without any confound.

### 3. Increase Replication to n=10 Seeds

**V1**: n=3 seeds per cell (108 total draws: 36 cells × 3 seeds/cell).

**V2**: n=10 seeds per cell, but **stratified by model** to control cost:

| Model | state_dim | horizon | wall/run | Arms | Seeds | Total runs | Est. wall time |
|-------|-----------|---------|----------|------|-------|------------|----------------|
| KSC SV T10 | 1 | 10 | ~75s | 5 | 10 | 50 | 62 min |
| LGSSM T50 | 3 | 50 | ~85s | 5 | 10 | 50 | 71 min |
| Predator-Prey T20 | 2 | 20 | ~80s | 5 | 10 | 50 | 67 min |
| **Total** | | | | | | **150** | **200 min (3.3h)** |

Seed range: 98401-98410 (disjoint from V1's 98301-98303).

**Statistical power**: n=10 gives 80% power to detect |Δ| ≥ 0.7σ_diff at α=0.05 (two-sided). V1's n=3 had 80% power only for |Δ| ≥ 1.3σ_diff.

### 4. Revised Arm Set

Drop `sobol_matousek` (degenerates to `sobol_owen` at d=1). Keep 5 arms:

1. **mc**: `tf.random.normal` (baseline)
2. **sobol_owen**: Sobol' + Owen scrambling
3. **halton_owen**: Halton + Owen scrambling  
4. **halton_reverse**: Halton + reverse-radix scrambling (distinct from halton_owen)
5. **lhs**: Latin Hypercube Sampling (recommended replacement for genut_guided)

**Rationale**: 
- 4 genuine RQMC arms (sobol, 2 haltons, lhs) with known theoretical guarantees
- MC as the null hypothesis
- **GenUT dropped**: three confounds (moment pre-alignment, index-wise injection, low diversity) make interpretation impossible

### 5. Add Runtime Confound Checks

Every evaluation must verify:

**A. Process-noise determinism** (per-seed check):
```python
def _verify_process_noise_determinism(model, seed):
    """Check that all arms produce identical process noise for this seed."""
    hashes = {}
    for arm in ARMS:
        pnoise = _generate_process_noise_for_arm(model, arm, seed)
        hashes[arm] = hashlib.sha256(pnoise.numpy().tobytes()).hexdigest()
    
    if len(set(hashes.values())) != 1:
        raise ValueError(f"Process noise differs across arms for seed {seed}: {hashes}")
```

**B. Cloud distinctness** (per-cell check):
```python
def _verify_cloud_distinctness(cloud, arm, min_frac=0.95):
    """Check that cloud has enough distinct particles."""
    n = cloud.shape[0]
    unique = len(np.unique(np.round(cloud.numpy(), 10), axis=0))
    required = int(min_frac * n) if arm != "genut_symmetric" else min(int(min_frac * n), 2 * d + 1)
    
    if unique < required:
        raise ValueError(f"{arm} cloud has only {unique}/{n} distinct particles (required {required})")
```

Run both checks **before** the evaluator. Fail fast with actionable diagnostics.

### 6. Structured Artifact Schema V2

Every `result.json` must include:

```json
{
  "schema_version": "rqmc_ledh_v2",
  "model": "ksc_sv_T10",
  "arm": "sobol_owen",
  "seed": 98401,
  "status": "complete",
  "value": -19.9123,
  "cloud_distinctness": {
    "n_particles": 1008,
    "n_distinct": 1008,
    "fraction_distinct": 1.0,
    "passed_gate": true
  },
  "process_noise_hash": "a3f21c...",
  "process_noise_matches_baseline": true,
  "tuning_artifact": "docs/benchmarks/artifacts/ledh_trust_region_ksc_sv_t10_20260903/result.json",
  "wall_seconds": 74.2,
  "git_commit": "...",
  "device": "4080_SUPER",
  "runner_version": "v2"
}
```

Phase 3 analyzer must:
- Verify `process_noise_matches_baseline == true` for all cells
- Verify `cloud_distinctness.passed_gate == true` for all cells
- Reject the campaign if either check fails on >5% of cells

---

## Evidence Contract

### Primary Criterion
Bootstrap 95% CI for (RQMC − MC) on terminal log-likelihood. RQMC is **statistically superior** if `ci_lower > 0`.

### Promotion Veto
Any RQMC arm with `ci_upper < 0` on any model (MC superior).

### Continuation Veto
Any of:
- Process-noise confound detected (hash mismatch >0 cells)
- Degeneracy detected (distinctness gate failed >5% of cells)
- Infrastructure failure rate >10%

### Explanatory Diagnostics
- Per-seed value tables (like V1)
- Cloud visualization: 2D PCA projection of initial clouds
- Sensitivity: does the ranking change if we drop the highest/lowest seed?

### What Will Not Be Concluded
- RQMC superiority **in trajectory simulation** (would require RQMC process noise too)
- Hardware-specific claims (one GPU, one dtype)
- Generalization to N ≠ 1008 or other particle counts

---

## Phase Breakdown

### Phase 0: Runner Implementation and Validation

**Deliverable**: `run_rqmc_ledh_init_v2.py` with:
- Independent RNG seeding via SeedSequence
- Fixed GenUT or replacement arm
- Runtime confound checks (hash, distinctness)
- V2 artifact schema

**Validation**:
1. **Smoke test** (1 seed, all 5 arms, all 3 models): verify all checks pass
2. **Confound regression**: run mc + sobol_owen on same seed 5 times; verify process_noise_hash identical
3. **Distinctness regression**: verify genut passes gate on all 3 models
4. **Commit runner** with test evidence before proceeding

### Phase 1: Trust-Region Tuning (if needed)

V1 artifacts may be reused **only if**:
- Tuning was done with identical hardware, controls, and particle count
- Tuning scope matches V2 (model + route + horizon + N + dtype)

Otherwise, re-run Phase 1 for each model (3 campaigns, ~30 min each).

### Phase 2: Full Campaign Execution

Execute 150 cells (3 models × 5 arms × 10 seeds) in parallel where possible.

**Device**: 4080 SUPER (`CUDA_VISIBLE_DEVICES=1`)  
**Estimated wall time**: 3.3 hours (if sequential); 30-60 min (if 3-4 parallel workers)  
**Budget**: 4 hours (includes retries)

**Stop conditions**:
- Any continuation veto fires
- Infrastructure failure rate >10%
- Wall time exceeds 6 hours

### Phase 3: Statistical Analysis

Use corrected `analyze_rqmc_ledh_phase3.py` (already has correct sign convention from V1 fix).

**Required additions**:
1. Verify `process_noise_matches_baseline` column has all `True`
2. Verify `cloud_distinctness.passed_gate` column has all `True`
3. Report any violations as campaign invalidation

**Output**: `phase3_analysis_v2.json` with per-model verdicts and aggregate recommendation.

### Phase 4: Result Documentation

Write `rqmc-ledh-v2-result-2026-09-XX.md` with:
- Aggregate verdict (PROMOTE / NEUTRAL / REJECT / MIXED)
- Per-model CI tables
- Confound verification summary (all checks passed)
- Comparison to V1 (qualitative only — V1 is invalid)
- Next steps if PROMOTE (integration path) or if REJECT (alternative mechanisms)

---

## Cost and Risk

### Cost
- **Implementation**: 2-3 hours (runner + validation)
- **Execution**: 3-4 hours wall time
- **Analysis**: 30 min
- **Total**: 1 working day

### Risk
**Low**:
- Confound fix is straightforward (SeedSequence.spawn for independent seeding)
- LHS is well-established genuine RQMC with no confounds
- n=10 replication is affordable (~200 min wall time)

**Medium**:
- If hardware unavailable, campaign may take longer or need CPU fallback (slower but valid)

**Mitigation**:
- Smoke test Phase 0 validation catches confound bugs before full campaign
- Stop conditions prevent runaway costs
- V2 artifacts are self-verifying (hash + distinctness in every result.json)

---

## Decision Point

**This design is not approved for execution.** Owner review required on:

1. **GenUT resolution**: Confirm dropping GenUT and using LHS as 5th arm
2. **Seed budget**: n=10 acceptable, or increase to n=15-20 for more power?
3. **Arm set**: Confirm 5 arms (mc, sobol_owen, halton_owen, halton_reverse, lhs)
4. **Phase 0 validation depth**: Current smoke/regression sufficient, or add more?

Once approved, proceed to Phase 0 implementation.

---

## Appendix A: Why SeedSequence Fixes The Confound

`numpy.random.SeedSequence` generates **independent, reproducible** child seeds from one parent seed:

```python
ss = SeedSequence(98401)
child_a, child_b = ss.spawn(2)
# child_a and child_b are cryptographically independent
# but deterministic given 98401
```

This guarantees:
- **Determinism**: same seed → same process noise, always
- **Independence**: initial-cloud generation method does not affect process-noise stream
- **Reproducibility**: re-running seed 98401 gives identical results

The V1 runner failed independence: MC's initial draw advanced the stream, so process noise started at a different offset. V2 fixes this by deriving two independent child seeds.

---

## Appendix B: Statistical Power Calculation

For paired comparison (RQMC − MC), assuming:
- σ_diff ≈ 0.3 (conservative, based on V1 data after confound correction)
- α = 0.05 (two-sided)
- Desired power = 0.80

Minimum detectable effect (Cohen's d):
- n=3: d ≥ 2.3 → |Δ| ≥ 0.69 (V1 could only detect large effects)
- n=10: d ≥ 1.0 → |Δ| ≥ 0.30 (V2 has 2.3× better sensitivity)
- n=20: d ≥ 0.7 → |Δ| ≥ 0.21 (diminishing returns past n=15)

V2's n=10 balances power and cost. If owner wants to detect effects as small as |Δ| = 0.15, increase to n=20 (budget: 6 hours wall time).

---

## Appendix C: Why V1 Artifacts Cannot Be Salvaged

Even if we:
1. Fix the sign inversion in the analyzer
2. Collapse the Lloyd degeneracy
3. Measure the confound and attempt statistical adjustment

We **cannot** separate the confound from the mechanism post hoc because:
- The confound is seed-specific (different random trajectories)
- Process-noise effect is stochastic, not a fixed bias
- Statistical adjustment requires knowing the confound-free ground truth (which we don't have)

The only valid approach is to re-run with a corrected runner. V1 data can be used only for:
- Pilot estimates of effect size
- Smoke-testing the V2 runner (qualitative comparison)
- Documenting the confound as a methodological lesson

No scientific claim can rest on V1 data.
