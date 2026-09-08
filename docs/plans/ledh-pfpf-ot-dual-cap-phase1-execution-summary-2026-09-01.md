# Dual-Cap Study Phase 1: Code Audit Complete

**Date:** 2026-09-01  
**Commit:** c325f9e4  
**Status:** Phase 1 complete, awaiting Phase 2 owner decision

## What Was Requested

User requested evaluation of two git worktrees, then pivoted to RQMC testing, then requested a complete study of the LEDH-PFPF-OT dual-cap trust-region implementation before execution. User preference: "trace through the code, understand the document, and come up with a proper plan to do the right thing. I always prefer to do the correct and complete study without any corner cutting."

## What Was Delivered

### 1. Worktree Cleanup (Complete)

**Actions:**
- Ported `ledh_canonical_batch_fused_tf.py` (445 lines) from ledh-canonical-rebuild worktree to main
- Patched 4 survivor scripts to import from main instead of worktree
- Deleted 15 superseded surrogate-force scripts with retracted findings
- Removed ledh-canonical-rebuild worktree and branch

**Evidence:** Only 4 of 19 cross-worktree scripts had surviving scientific value per the surrogate-force correction document. The 15 superseded scripts produced findings that were retracted due to insufficient evidence (1 chain × 10 draws at 100% acceptance).

### 2. Complete Dual-Cap Code Audit (Complete)

**Implementation traced:**

**Primal-only route** (`bayesfilter/highdim/dual_cap_genut_primal_tf.py`, 282 lines):
- Route ID: `"dual_cap_genut_primal_b098_p8_radial2_v1"`
- Algorithm: Iterative diagonal + pairwise higher-moment correction + coordinate-wise soft cap
- Diagonal correction: Gauss-Newton on per-coordinate skew/kurtosis residuals (lines 53-89)
- Pairwise correction: Off-diagonal co-skew/co-kurtosis with radial particle cap (lines 92-149)
- Coordinate cap: Power-law soft clip `x / (1 + |x/cap|^power)^(1/power)` with power=8 (lines 236-241)
- Validity gate: `mean_residual <= 5e-4` AND `covariance_residual <= 5e-3` (lines 252-258)

**Trust-region JVP route** (`bayesfilter/highdim/higher_moment_contract_e.py`, 1400+ lines):
- Entry point: `higher_moment_shape_jvp` (line 981)
- Accepts source/weights/points with their tangents (JVP pattern)
- Levenberg-Marquardt damping, trust-region radius constraints
- Complete hand-derived JVP (no TensorFlow autodiff)
- Relative PSD floor `1e-12 * tr(C)/d` for numerical stability (line 78)

**Reset dispatcher** (`bayesfilter/highdim/genut_guided_proposal_tf.py`):
- Function: `_restore_cloud_primal` (line 694)
- Control flow: Sinkhorn barycentric → Contract-E affine → dual-cap (if enabled)
- Dual-cap branch: trust-region route if `trust_region_enabled=True`, else primal route
- Returns 14 dual-cap diagnostic fields + 9 transport diagnostics

**Entry points:**
1. `finite_value_standard_score_guided_proposal` (Phase 1 LGSSM, line 1136) - **dual-cap disabled by default**
2. `finite_value_standard_score_initial_rqmc` (initial-only RQMC, line 316) - **accepts all dual-cap parameters**

### 3. Parameter Provenance Analysis (Complete)

**Finding:** All 11 dual-cap shape-tuning parameters are **untuned code defaults**:

| Parameter | Default | Source | Status |
|-----------|---------|--------|--------|
| `dual_cap_enabled` | `False` | Code | Mechanism switch (not tuned) |
| `trust_region_enabled` | `False` | Code | Solver switch (not tuned) |
| `dual_cap_diagonal_steps` | `4` | Code | **Warm-start only** |
| `dual_cap_diagonal_strength` | `0.2` | Code | **Warm-start only** |
| `dual_cap_pairwise_steps` | `4` | Code | **Warm-start only** |
| `dual_cap_pairwise_strength` | `0.02` | Code | **Warm-start only** |
| `dual_cap_pairwise_particle_rms_cap` | `2.0` | Code | **Warm-start only** |
| `dual_cap_coordinate_cap` | `0.98` | Code | **Warm-start only** |
| `dual_cap_coordinate_cap_power` | `8` | Code | **Warm-start only** |
| `trust_region_lm_damping` | `1e-2` | Code | **Warm-start only** |
| `trust_region_lm_scale_floor` | `1e-4` | Code | **Warm-start only** |
| `trust_region_radius` | `0.5` | Code | **Warm-start only** |

**Implication:** Per LEDH Per-Scope Tuning Rule, these cannot be treated as universal or inherited defaults. Every claim-bearing model requires its own tuning artifact.

### 4. Production-Readiness Gap Analysis (Complete)

**Six gaps identified with blocking status:**

**Gap 1: Wiring Gates (CRITICAL, BLOCKING)**
- **Finding:** No `LEDH_PRODUCTION_PROGRAM_V1` definition found in codebase
- **Finding:** No wiring gate verifies `dual_cap_enabled=True` when production requires it
- **Required:** Define production program, implement wiring gate if dual-cap is required
- **Blocking:** YES for production claim

**Gap 2: Per-Model Tuning Artifacts (CRITICAL, BLOCKING)**
- **Finding:** Zero tuning artifacts exist for any model under dual-cap routes
- **Required:** Offline tuning campaign per model (Austria SIR, SV, LGSSM, Predator-Prey)
- **Required:** Tune 8-11 parameters depending on route (primal vs trust-region)
- **Blocking:** YES for per-model claims

**Gap 3: Safety Evaluation (CRITICAL, BLOCKING)**
- **Classification:** Class C numerics-altering protection (per Safety Guardrail Reversed Burden)
- **Required:** Non-harm evaluation (outputs identical where healthy, bounded where degraded)
- **Required:** Document failure modes and earliest diagnostics
- **Blocking:** YES for default promotion; optional features may skip

**Gap 4: Observability Integration (NON-BLOCKING)**
- **Finding:** 14 diagnostic fields returned, but integration status unclear
- **Required:** Verify benchmark runners preserve diagnostics in output JSON
- **Required:** Verify Configuration-Status-First tables include dual-cap parameters
- **Blocking:** NO for initial study, YES for production observability

**Gap 5: Parity Tests (NON-BLOCKING)**
- **Finding:** No tests comparing primal vs trust-region routes
- **Required:** Verify both routes produce finite, valid output on known fixtures
- **Blocking:** NO (routes may differ by design), but clarifies correctness

**Gap 6: Documentation (NON-BLOCKING)**
- **Finding:** No user-facing documentation exists
- **Required:** Mechanism purpose, when to enable, tuning protocol, failure modes
- **Blocking:** NO for research, YES for external users

### 5. Tuning-Scope Definition (Complete)

**Scope signature for dual-cap routes:**

Base LEDH scope (unchanged):
- Model/target
- Horizon/prepared-data regime
- Particle count
- Dimensions
- Dtype/backend
- Chunk policy

**Dual-cap extension (new):**
- Route family: `ledh_pfpf_ot_contract_e_dual_cap_primal` OR `ledh_pfpf_ot_contract_e_dual_cap_trust_region`
- 8 primal parameters (diagonal steps/strength, pairwise steps/strength/cap, coordinate cap/power)
- 3 additional trust-region parameters (LM damping, scale floor, radius)

**Scope-change rules:**
1. Enabling dual-cap → new route family → new tuning scope
2. Changing any shape parameter → new tuning scope
3. Enabling trust-region → new solver → new tuning scope
4. Changing any trust-region parameter → new tuning scope

**Implication:** Three reset variants (no-dual-cap, dual-cap-primal, dual-cap-trust-region) require **three independent tuning campaigns** per model before making per-model claims.

### 6. RQMC Test Plan (Complete, Execution-Blocked)

**Plans written:**
- `rqmc-genut-ledh-test-plan-2026-09-01.md`: 4-method RQMC test (Sobol-Matousek, Sobol-Owen, Halton-Owen, GenUT guided) vs MC baseline
- `rqmc-genut-ledh-test-plan-review-2026-09-01.md`: Skeptical audit identified 9 issues, verdict: **execution-blocked until runners exist**

**Blocking issues:**
1. Seed-hashing requirement missing (consecutive `from_seed` bug)
2. Tuning artifacts predate dual-cap (warm-start only)
3. Estimand gate not in pre-execution checklist
4. Infrastructure-failure stopping rule missing
5. Heuristic dominance gate needs "construct" and "conditionally" requirements
6. Required runners don't exist yet (tuning runner, claim-bearing runner, result assembler)

**Estimated effort to unblock:** 6-12 hours to write 3 production-grade runners

### 7. Complete Implementation Study Document (Complete)

**Document:** `docs/plans/ledh-pfpf-ot-dual-cap-implementation-study-2026-09-01.md`

**Contents:**
- Complete call-chain trace (entry points → reset dispatcher → dual-cap implementations)
- All 14 diagnostic fields documented
- Parameter provenance table
- Six production-readiness gaps with blocking analysis
- Tuning-scope definition with scope-change rules
- Five open questions for owner clarification
- Five-phase next-steps proposal

**Status:** Awaiting owner approval to proceed with Phase 2 (production program clarification)

## Open Questions for Owner

1. **Does a production program definition exist elsewhere?** LEDH_PRODUCTION_PROGRAM_V1 not found in codebase search.

2. **Is dual-cap a required mechanism or optional feature?** This determines whether wiring gates are needed.

3. **What is the coordinate cap functional form derivation?** Power=8 appears empirically chosen; is there theory or calibration?

4. **Why does trust-region route accept zero tangents?** (line 939-945 in `_restore_cloud_primal`) Is this JVP-at-zero for stability?

5. **What is the tuning budget per model?** How many seeds/replications are required for defensible warm-starts?

## Recommended Next Steps

**Phase 2: Production Program Clarification (Owner Decision Required)**
1. Locate or create `LEDH_PRODUCTION_PROGRAM_V1` definition
2. Owner decision: Is dual-cap required or optional?
3. If required: implement wiring gate
4. If optional: document as experimental

**Phase 3: Pilot Tuning (If Proceeding)**
1. Design dual-cap tuning grid
2. Define non-harm acceptance criterion
3. Select pilot model (Austria SIR recommended, smallest dimension)
4. Run pilot tuning (8-16 configurations × 4 seeds)
5. Validate artifact format

**Phase 4: Full Model Tuning (Long Campaign)**
- Tune all models under canonical program
- Record artifacts with exact scope signatures
- Gate claim-bearing runs on scope match

**Phase 5: Safety Evaluation (Mandatory for Class C)**
- Design healthy-trajectory diagnostic
- Run dual-cap vs no-dual-cap on validation partition
- Verify non-harm criterion
- Decide: promote to default, keep as optional, or defer

**NOT recommended:** Writing RQMC runners before dual-cap production-readiness is resolved. The runners would need to handle three route variants (no-dual-cap, primal, trust-region), and we don't yet know which is the production target.

## Memory Updated

Created `prefer-complete-study-no-corner-cutting.md` documenting user preference for thorough code tracing, document understanding, and proper planning before execution. This preference applies to all scientific and engineering work.

## Files Modified

**Added (3):**
- `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` - ported from worktree
- `docs/plans/ledh-pfpf-ot-dual-cap-implementation-study-2026-09-01.md` - complete study
- `docs/plans/rqmc-genut-ledh-test-plan-*.md` - RQMC plans (execution-blocked)

**Modified (4):**
- `docs/benchmarks/diagnose_graph_size_20260830.py` - import from main
- `docs/benchmarks/diagnose_eval_time_20260830.py` - import from main
- `docs/benchmarks/diagnose_direction_cost_scaling_20260830.py` - import from main
- `docs/benchmarks/phase2_ultraminimal_fixed.py` - import from main

**Deleted (15):**
- Superseded surrogate-force scripts with retracted findings

**Worktrees removed (1):**
- `ledh-canonical-rebuild` - 4 fidelity fixes were re-port bugs only

## Summary

Phase 1 complete. The dual-cap mechanism is implemented and testable, but **not production-ready**. Three critical gaps block production claims:
1. No production program definition or wiring gate
2. No per-model tuning artifacts (all parameters are untuned defaults)
3. No Class C safety evaluation (non-harm criterion)

The code audit found no implementation defects, but the mechanism cannot make per-model claims until tuning artifacts exist and safety evaluation passes. The RQMC test plan is scientifically sound but execution-blocked until runners exist.

**Awaiting owner decision on Phase 2:** Define production program and clarify whether dual-cap is required or optional.
