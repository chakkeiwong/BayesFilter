# LEDH-PFPF-OT Dual-Cap Trust-Region Implementation Study

**Date:** 2026-09-01  
**Status:** AWAITING_OWNER_GO  
**Owner:** chakwong

## Research Question

What is the complete call-chain, parameter flow, and tuning-scope structure for the LEDH-PFPF-OT dual-cap trust-region covariance stabilization, and what work remains to make it production-ready according to repository policy?

## Background

The repository default execution target is GPU with TF32-enabled LEDH-PFPF-OT streaming transport. The Configuration-Status-First Reporting Rule (adopted 2026-08-26) requires that the production program be code-defined with wiring gates, and that omitting a required mechanism must surface as a labeled deviation rather than silently defining production downward through defaults.

The dual-cap trust-region mechanism was designed as covariance-explosion control for the Contract-E reset. Initial evidence exists that it provides stabilization, but the complete implementation audit and tuning-scope definition have not been completed.

## Objectives

1. **Call-chain audit**: Trace from every claim-bearing consumer endpoint (benchmark runners, leaderboard cells, production routes) to the dual-cap implementation to verify that the mechanism is actually reachable.

2. **Parameter provenance**: Document the complete parameter set for dual-cap trust-region, including:
   - Which parameters come from tuning artifacts vs warm-starts vs hardcoded defaults
   - Which parameters are mechanism-enabling vs shape-tuning
   - Which parameters affect comparability scope (trigger retuning requirements)

3. **Tuning-scope definition**: State the exact tuning-scope signature for dual-cap routes under the LEDH Per-Scope Tuning Rule, including:
   - Base scope fields (model, route family, horizon, particle count, dtype/backend, chunk policy)
   - Dual-cap-specific scope fields (diagonal steps/strength, pairwise steps/strength/cap, coordinate cap/power, trust-region damping/scale-floor/radius)
   - Whether diagonal vs pairwise vs trust-region variants create disjoint tuning scopes

4. **Production-readiness gaps**: Identify what remains before dual-cap can be declared production-ready:
   - Wiring gates to verify the mechanism is enabled
   - Tuning artifacts for each model under the canonical program
   - Parity tests between dual-cap and baseline routes
   - Safety evaluation (Class C numerics-altering protection per Safety Guardrail Reversed Burden)
   - Documentation of failure modes and earliest diagnostics

5. **Code-structure clarity**: Document the relationship between:
   - `bayesfilter/highdim/dual_cap_genut_primal_tf.py` (primal-only route)
   - `bayesfilter/highdim/higher_moment_shape_jvp_tf.py` (trust-region JVP route)
   - The `trust_region_enabled` flag in `_restore_cloud_primal`
   - How these paths interact with the production program definition

## Code Trace

### Entry Points

**Phase 1 LGSSM Guided Proposal** (file: `bayesfilter/highdim/genut_guided_proposal_tf.py`)

- Main function: `finite_value_standard_score_guided_proposal` (line 1136)
- Accepts dual-cap parameters but currently **all default to disabled**:
  - `dual_cap_enabled: bool = False`
  - No trust-region parameters exposed at this entry point

- Calls `guided_lgssm_step` → `_restore_cloud_primal` (line 1238)

**Initial-only RQMC** (file: `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`)

- Main function: `finite_value_standard_score_initial_rqmc` (line 316)
- Accepts complete dual-cap parameter set:
  ```python
  dual_cap_enabled: bool = False,
  dual_cap_diagonal_steps: int = 4,
  dual_cap_diagonal_strength: float = 0.2,
  dual_cap_pairwise_steps: int = 4,
  dual_cap_pairwise_strength: float = 0.02,
  dual_cap_pairwise_particle_rms_cap: float = 2.0,
  dual_cap_coordinate_cap: float = 0.98,
  dual_cap_coordinate_cap_power: int = 8,
  trust_region_enabled: bool = False,
  trust_region_lm_damping: float = 1.0e-2,
  trust_region_lm_scale_floor: float = 1.0e-4,
  trust_region_radius: float = 0.5,
  ```

- Passes these parameters through to reset calls

### Core Reset Function

**`_restore_cloud_primal`** (file: `bayesfilter/highdim/genut_guided_proposal_tf.py`, line 694)

```python
def _restore_cloud_primal(
    particles: Tensor,
    weights: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
    reset_policy: str = "contract_e",
    dual_cap_enabled: bool = False,
    dual_cap_diagonal_steps: int = 4,
    dual_cap_diagonal_strength: float = 0.2,
    dual_cap_pairwise_steps: int = 4,
    dual_cap_pairwise_strength: float = 0.02,
    dual_cap_pairwise_particle_rms_cap: float = 2.0,
    dual_cap_coordinate_cap: float = 0.98,
    dual_cap_coordinate_cap_power: int = 8,
    trust_region_enabled: bool = False,
    trust_region_lm_damping: float = 1.0e-2,
    trust_region_lm_scale_floor: float = 1.0e-4,
    trust_region_radius: float = 0.5,
    transport_plan_mode: str = "dense",
    transport_row_chunk_size: int | None = None,
    transport_col_chunk_size: int | None = None,
    marginal_tolerance: float = 1.0e-4,
) -> dict[str, Tensor]:
```

**Control flow:**

1. If `reset_policy == "none"`: return identity transform with zero diagnostics
2. Compute Sinkhorn barycentric (dense or streaming based on `transport_plan_mode`)
3. If `reset_policy == "ot_only"`: return barycentric without Contract-E
4. If `reset_policy == "contract_e"`:
   - Apply Contract-E affine correction (line 900)
   - **If `dual_cap_enabled`**:
     - **If `trust_region_enabled`**: call `higher_moment_shape_jvp` (line 946)
     - **Else**: call `dual_cap_genut_primal` (line 1032)
5. Return corrected particles with full diagnostic payload

**Key invariant (line 749-752):**
```python
if dual_cap_enabled and reset_policy != "contract_e":
    raise ValueError("dual cap requires the Contract-E reset")
if trust_region_enabled and not dual_cap_enabled:
    raise ValueError("trust-region dual cap requires the dual-cap reset")
```

### Dual-Cap Implementations

**Primal-only route** (`bayesfilter/highdim/dual_cap_genut_primal_tf.py`):
- Function: `dual_cap_genut_primal` (exported)
- Route ID: `"dual_cap_genut_primal_b098_p8_radial2_v1"`
- Applies diagonal and pairwise higher-moment corrections iteratively
- Coordinate-wise soft cap with power-law derivative
- Returns corrected particles + validity flags + diagnostics

**Trust-region JVP route** (`bayesfilter/highdim/higher_moment_shape_jvp_tf.py`):
- Function: `higher_moment_shape_jvp` (inferred from call site)
- Accepts zero tangents as inputs (line 939-945)
- Applies Levenberg-Marquardt-style damping and trust-region constraints
- Used when `trust_region_enabled=True`

### Diagnostic Payload

The `_restore_cloud_primal` function returns a dictionary with these dual-cap fields:

```python
"dual_cap_valid": Tensor,  # bool
"dual_cap_mean_residual": Tensor,  # max |output_mean - target_mean|
"dual_cap_covariance_residual": Tensor,  # max |output_cov - target_cov|
"maximum_pairwise_pre_cap_particle_rms": Tensor,
"maximum_pairwise_post_cap_particle_rms": Tensor,
"minimum_pairwise_particle_cap_scale": Tensor,
"maximum_coordinatewise_pre_cap_absolute": Tensor,
"maximum_coordinatewise_post_cap_absolute": Tensor,
"mean_coordinatewise_cap_displacement": Tensor,
"fraction_coordinatewise_cap_active": Tensor,
"minimum_coordinatewise_cap_derivative": Tensor,
"reset_route_id": Tensor,  # int32: 0=no_dual_cap, 1=trust_region
"maximum_diagonal_scaled_system_condition": Tensor,
"maximum_diagonal_pre_cap_particle_rms": Tensor,
"maximum_diagonal_post_cap_particle_rms": Tensor,
"trust_region_solver_id": Tensor,  # int32: 0=primal_only, 1=trust_region
```

These diagnostics are computed but **observability status unclear**: need to verify whether benchmark runners preserve and report these fields.

## Parameter Provenance Analysis

### Mechanism-Enabling Parameters

**`dual_cap_enabled: bool`**
- **Provenance:** Code default `False`
- **Status:** UNTUNED (binary mechanism switch)
- **Comparability impact:** Creates disjoint tuning scopes (dual-cap vs no-dual-cap)

**`trust_region_enabled: bool`**
- **Provenance:** Code default `False`
- **Status:** UNTUNED (binary mechanism switch)
- **Comparability impact:** Creates disjoint solver scopes within dual-cap family

### Shape-Tuning Parameters (Dual-Cap Family)

**Diagonal correction:**
- `dual_cap_diagonal_steps: int = 4`
- `dual_cap_diagonal_strength: float = 0.2`

**Pairwise correction:**
- `dual_cap_pairwise_steps: int = 4`
- `dual_cap_pairwise_strength: float = 0.02`
- `dual_cap_pairwise_particle_rms_cap: float = 2.0`

**Coordinate-wise cap:**
- `dual_cap_coordinate_cap: float = 0.98`
- `dual_cap_coordinate_cap_power: int = 8`

**Trust-region controls (when `trust_region_enabled=True`):**
- `trust_region_lm_damping: float = 1.0e-2`
- `trust_region_lm_scale_floor: float = 1.0e-4`
- `trust_region_radius: float = 0.5`

**Provenance:** All values are hardcoded defaults in function signatures. No tuning artifacts identified.

**Status:** WARM-START CANDIDATES ONLY. Per LEDH Per-Scope Tuning Rule, these must not be treated as universal or inherited defaults.

## Tuning-Scope Definition

### Base Scope (from LEDH Per-Scope Tuning Rule)

Every claim-bearing LEDH model run requires tuning for:
- Model/target
- Route/reset family
- Horizon/prepared-data regime
- Particle count
- Dimensions
- Dtype/backend
- Chunk policy
- Route-specific control family

### Dual-Cap Extension

**Route family identifier:**
- Base: `"ledh_pfpf_ot_contract_e"`
- With dual-cap: `"ledh_pfpf_ot_contract_e_dual_cap_primal"`
- With trust-region: `"ledh_pfpf_ot_contract_e_dual_cap_trust_region"`

**Additional scope fields:**
- `dual_cap_diagonal_steps`
- `dual_cap_diagonal_strength`
- `dual_cap_pairwise_steps`
- `dual_cap_pairwise_strength`
- `dual_cap_pairwise_particle_rms_cap`
- `dual_cap_coordinate_cap`
- `dual_cap_coordinate_cap_power`
- (Trust-region only) `trust_region_lm_damping`
- (Trust-region only) `trust_region_lm_scale_floor`
- (Trust-region only) `trust_region_radius`

**Scope-change rules:**
1. Enabling dual-cap (`dual_cap_enabled: False → True`) changes route family → new tuning scope
2. Changing any dual-cap shape parameter → new tuning scope
3. Enabling trust-region (`trust_region_enabled: False → True`) changes solver → new tuning scope
4. Changing any trust-region control parameter → new tuning scope

**Implication:** A model/horizon/particle-count combination with three reset variants (no-dual-cap, dual-cap-primal, dual-cap-trust-region) requires **three independent tuning campaigns** before any can make per-model claims.

## Production-Readiness Gap Analysis

### Gap 1: Wiring Gates (CRITICAL)

**Finding:** No wiring gate verifies that `dual_cap_enabled=True` when the production program requires it.

**Evidence:** Configuration-Status-First Reporting Rule amendment (2026-08-26) states:
> the production program is DEFINED IN CODE (`LEDH_PRODUCTION_PROGRAM_V1` in `ledh_alg1_contract.py`), "production" labels are validated against it (wiring gate), and omitting a required mechanism must surface as a labeled deviation in the configuration-status table.

**Required action:**
1. Locate or create `LEDH_PRODUCTION_PROGRAM_V1` definition
2. If dual-cap is a required mechanism, add a wiring gate that fails when `dual_cap_enabled=False` in a production-labeled run
3. If dual-cap is optional, document it as such and do not require wiring gate

**Blocking:** YES (for production claim)

### Gap 2: Per-Model Tuning Artifacts (CRITICAL)

**Finding:** No tuning artifacts identified for dual-cap routes on any model.

**Evidence:** All dual-cap parameters are code defaults. The LEDH Per-Scope Tuning Rule prohibits treating settings selected for another model/route as inherited defaults.

**Required action:**
For each active model (Austria SIR, SV variants, LGSSM, Predator-Prey):
1. Create offline tuning campaign under `docs/benchmarks/artifacts/ledh_dual_cap_tuning_<model>_<date>/`
2. Tune diagonal steps/strength, pairwise steps/strength/cap, coordinate cap/power
3. If trust-region is production target, separately tune trust-region controls
4. Record tuning artifact path in model-specific configuration
5. Gate claim-bearing runs on exact tuning-scope match

**Blocking:** YES (for per-model claims)

### Gap 3: Safety Evaluation (Class C Protection)

**Finding:** Dual-cap is a numerics-altering protection (changes computed particles). Per Safety Guardrail Reversed Burden, Class C protections require a mandatory dedicated evaluation.

**Acceptance criterion:** Non-harm (outputs identical where trajectory is healthy; bounded, flagged behavior where it is not), never primary-metric improvement.

**Required action:**
1. Define "healthy trajectory" diagnostics (e.g., gap eigenvalue > threshold, mean residual < threshold)
2. Run dual-cap vs no-dual-cap on held-out validation data
3. Verify: where baseline is healthy, dual-cap output ≈ baseline output (within numerical tolerance)
4. Verify: where baseline degrades (NaN, explosion, negative eigenvalue), dual-cap bounds degradation and sets `dual_cap_valid=False`
5. Document non-harm evidence and failure modes

**Blocking:** YES (for default promotion; optional features may skip)

### Gap 4: Observability Integration

**Finding:** Dual-cap returns 14 diagnostic fields. Integration status with benchmark runners, leaderboard cells, and artifact manifests is unclear.

**Required verification:**
1. Check whether `run_ledh_pfpf_genut_initial_rqmc_all_models.py` preserves dual-cap diagnostics in output JSON
2. Check whether leaderboard cells report dual-cap validity flags
3. Check whether Configuration-Status-First tables include dual-cap parameter values

**Blocking:** NO (for initial study), YES (for production observability)

### Gap 5: Parity Tests

**Finding:** No parity tests between dual-cap primal and trust-region routes identified.

**Required action:**
1. Create test fixture with known-stable baseline output
2. Enable `dual_cap_enabled=True, trust_region_enabled=False` → primal route
3. Enable `dual_cap_enabled=True, trust_region_enabled=True` → trust-region route
4. Verify both routes produce finite, valid output
5. Document output differences and their causes

**Blocking:** NO (routes may have different outputs by design), but clarifies implementation correctness

### Gap 6: Documentation

**Finding:** No user-facing documentation for dual-cap exists.

**Required action:**
1. Document mechanism purpose (covariance explosion control)
2. Document when to enable (models with observed explosion, large dimension, long horizon)
3. Document tuning protocol (offline grid search, non-harm criterion)
4. Document failure modes (too-strong cap → underdispersed cloud, too-weak cap → unbounded covariance)
5. Document earliest diagnostics (`dual_cap_valid` flag, mean/covariance residuals)

**Blocking:** NO (for research), YES (for external users)

## Open Questions

1. **Production program definition:** Where is `LEDH_PRODUCTION_PROGRAM_V1` defined? Does it currently include or exclude dual-cap?

2. **Trust-region tangent purpose:** Why does the trust-region route accept zero tangents (line 939-945 in `_restore_cloud_primal`)? Is this a JVP-at-zero pattern for stability, or placeholder for future gradient work?

3. **Coordinate cap functional form:** The coordinate cap uses a power-law soft clipping. What is the derivation or source for this choice? Is `power=8` empirically determined or theory-driven?

4. **Diagonal vs pairwise order:** Why is diagonal correction applied before pairwise? Does order matter for convergence or stability?

5. **Tuning budget:** How many seeds/replications are required for dual-cap tuning to produce defensible warm-starts for claim-bearing runs?

## Next Steps (Proposed)

### Phase 1: Code Audit (Non-Blocking, 1-2 hours)

1. Read `bayesfilter/highdim/dual_cap_genut_primal_tf.py` in full
2. Read `bayesfilter/highdim/higher_moment_shape_jvp_tf.py` in full
3. Trace call-chain from `run_ledh_pfpf_genut_initial_rqmc_all_models.py` to dual-cap implementation
4. Verify diagnostic observability in runner output JSON
5. Document findings in this plan

### Phase 2: Production Program Clarification (Blocking Decision Point)

1. Locate `LEDH_PRODUCTION_PROGRAM_V1` definition (or create if absent)
2. Owner decision: Is dual-cap a required mechanism or an optional feature?
3. If required: implement wiring gate
4. If optional: document as experimental and skip wiring gate

### Phase 3: Tuning Protocol Design (Blocking for Claims)

1. Design dual-cap tuning grid (parameter ranges, step sizes)
2. Define non-harm acceptance criterion
3. Select one pilot model (suggest Austria SIR, smallest dimension)
4. Run pilot tuning campaign (budget: 8-16 configurations × 4 seeds)
5. Validate tuning artifact format and comparability checks

### Phase 4: Full Model Tuning (Long Campaign)

1. Tune dual-cap for all active models under canonical program
2. Record artifacts with exact scope signatures
3. Update model-specific configurations to reference tuning artifacts
4. Gate claim-bearing runs on tuning-scope match

### Phase 5: Safety Evaluation (Mandatory Class C)

1. Design healthy-trajectory diagnostic
2. Run dual-cap vs no-dual-cap on validation partition
3. Verify non-harm criterion
4. Document failure modes and earliest diagnostics
5. Decide: promote to default, keep as optional, or defer

## Dependencies

- **Blocking this study:** None (pure code audit)
- **Blocked by this study:** Any claim that dual-cap is production-ready without tuning artifacts
- **Related work:** Surrogate-force HMC investigation (retracted 2026-08-30; reset routes are independent of HMC)

## References

- Configuration-Status-First Reporting Rule (CLAUDE.md, adopted 2026-08-25, amended 2026-08-26)
- LEDH Per-Scope Tuning Rule (CLAUDE.md)
- Safety Guardrail Reversed Burden (CLAUDE.md, adopted 2026-08-20)
- Implementation Audit Call-Chain Rule (CLAUDE.md, adopted 2026-08-20)

## Appendix: File Inventory

**Implementation files:**
- `bayesfilter/highdim/dual_cap_genut_primal_tf.py` (primal route)
- `bayesfilter/highdim/higher_moment_shape_jvp_tf.py` (trust-region route, inferred)
- `bayesfilter/highdim/genut_guided_proposal_tf.py` (`_restore_cloud_primal`)
- `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py` (initial-only entry point)

**Test files:**
- `tests/highdim/test_ledh_pfpf_genut_initial_rqmc_all_models.py` (rescued from genut worktree)
- Additional test coverage TBD

**Benchmark runners:**
- `docs/benchmarks/run_ledh_pfpf_genut_initial_rqmc_all_models.py` (RQMC campaign, CPU/XLA)
- Additional runners TBD

**Tuning artifacts:**
- None identified (critical gap)

---

**Study status:** Plan complete, awaiting owner approval to proceed with Phase 1 code audit.
