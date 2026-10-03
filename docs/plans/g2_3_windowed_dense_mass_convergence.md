# G2.3 Windowed Dense Mass Convergence Plan

**Created**: 2026-08-26  
**Author**: Claude Code  
**Status**: Draft awaiting skeptical audit

## Research Question

Can windowed dense mass adaptation with covariance shrinkage achieve R̂ < 1.02 on all 9 theta parameters in the G2.3 337-dimensional flattened C1 joint target?

## Problem Statement

The G2.3 full C1 fixture recovery test (`test_g2_3_full_c1_fixture_recovery` in `tests/hardbound/test_phase2_joint_hmc.py`) currently fails convergence with dense mass matrix adaptation:

- **Observed**: max R̂ = 1.083 (θ₂), 4 of 9 theta coordinates exceed 1.02 threshold
- **R̂ array**: `[1.00256, 1.08299, 1.02224, 1.02218, 1.00552, 1.00149, 1.01024, 1.00236, 1.01693]`
- **Effective sample sizes**: `[214.6, 166.8, 235.9, 294.7, 305.6, 292.0, 625.7, 657.5, 170.1]`
- **Baseline (diagonal mass)**: max R̂ = 1.048, passes 1.02 threshold

### Root Cause Classification

Under the tuning guide taxonomy (`docs/chapters/ch21b_hmc_tuning_interfaces.tex`), this is a **tuning-candidate failure**, not a rejection of HMC or the target. The defect is in the mass adaptation implementation:

Current implementation (`bayesfilter/hardbound/dense_mass_matrix_adaptation.py`):
- Single-freeze TFP-style scheme: accumulate samples during first 80% of warmup (3200 of 4000 steps), freeze afterward
- No initial fast buffer → early off-posterior warmup draws contaminate covariance estimate
- No progressive doubling windows → covariance never refined after initial freeze
- No covariance shrinkage → n≪p regime (3200 samples for 56,953 unique entries in 337×337 covariance matrix) produces rank-deficient or ill-conditioned estimate
- Chain-runner-side adaptation with no artifact authority

The repository has reviewed windowed mass infrastructure (`bayesfilter/inference/hmc_tuning.py`, `bayesfilter/inference/hmc_warmup.py`) specifically designed for this regime with:
- Initial buffer to protect against early contamination
- Progressive doubling slow windows (e.g., 25, 50, 100, 200, ...) with covariance recomputation at boundaries
- Final buffer for locked-mass step-size tuning
- `welford_covariance_shrinkage` with default λ=0.1 (10% weight on stable identity/prior, 90% on empirical)

## Evidence Contract

### Primary Promotion Criterion
Max R̂ < 1.02 across all 9 theta coordinates after 4 chains × 3000 retained samples (matching current test configuration).

### Hard Veto Diagnostics
- Any non-finite state during sampling
- Any divergence during retained sampling
- Max R̂ on any theta coordinate > 1.05 (materially worse than diagonal baseline)
- Implementation fails to use windowed schedule with shrinkage

### Explanatory-Only Diagnostics
- Per-theta ESS (for understanding which coordinates benefit)
- Acceptance rate (for step-size tuning quality)
- Window-by-window covariance condition number (for shrinkage effectiveness)
- Wall time vs diagonal baseline

### Non-Claims
- **No claim** that dense mass is superior to diagonal in general for G2.3
- **No claim** that λ=0.1 is optimal shrinkage (it is repository default, not tuned)
- **No claim** about final posterior correctness (this is tuning-candidate evaluation, not posterior validation)
- **No claim** about transferability to other models/dimensions

## Route Decision

### Option A: Public Tuner Route (`tune_hmc_kernel`)

**Advantages**:
- Uses canonical artifact authority per route table (`docs/generated/hmc_tuning_route_table.md`)
- Automatic windowed mass adaptation with reviewed shrinkage
- Stricter verification (R̂ ≤ 1.01 by default vs test's 1.02)
- Produces replayable tuning artifact

**Disadvantages**:
- Requires building G2.3 adapter conforming to public interface:
  - `parameter_dim = 337`
  - `adapter_signature() -> str`
  - `value_score_capability() -> ValueScoreCapability`
  - `log_prob_and_grad(position) -> (log_prob, grad)`
- G2.3 target currently embedded in test fixture, not modular adapter
- Adapter extraction may be substantial work
- Public tuner produces tuning artifact (discarded draws), then would need separate retained-sample run

### Option B: Hardbound Port (Windowed Schedule + Shrinkage)

**Advantages**:
- Works with existing G2.3 test structure
- Direct path to test passing
- Can reuse windowed schedule builder and shrinkage from `bayesfilter/inference/hmc_tuning.py`
- Single test run produces retained samples for convergence check

**Disadvantages**:
- Chain-runner-side adaptation remains without canonical artifact authority
- Does not produce replayable tuning artifact
- Harder to transfer to other hardbound targets

**Decision**: **Option B (Hardbound Port)** for this phase.

**Justification**: The immediate goal is test passing with mathematically sound adaptation. Option A would require substantial adapter extraction work that is out of scope for this convergence fix. Option B directly addresses the root cause (missing shrinkage and windowed refinement) while preserving existing test structure. If G2.3 becomes a production tuning target, migration to public tuner route is future work.

## Implementation Plan

### Phase 1: Port Windowed Schedule and Shrinkage (Hardbound)

1. **Create `bayesfilter/hardbound/windowed_dense_mass_adaptation.py`** with:
   - Import `build_windowed_warmup_schedule`, `welford_covariance`, `_shrink_covariance` from `bayesfilter.inference.hmc_tuning`
   - Build windowed schedule with:
     - `warmup_steps = 4000` (from test config)
     - `initial_buffer = 75` (Stan default for dimensionality)
     - `final_buffer = 50` (lock mass for final step-size tuning)
     - `first_window_size = 25` (Stan default)
     - Progressive doubling: 25, 50, 100, 200, ... up to remaining slow region
   - Interleaved warmup kernel:
     - Run TFP SimpleStepSizeAdaptation for each window
     - At slow window boundaries where `update_mass=True`:
       - Compute Welford covariance from window draws
       - Shrink: `(1-λ)*welford_cov + λ*identity` with λ=0.1
       - Add jitter diagonal: `1e-9 * I`
       - Cholesky factor: `L = chol(shrunk_cov)`
       - Rebuild momentum distribution: `LinearOperatorLowerTriangular(L, is_non_singular=True)`
     - Continue to next window with updated mass
   - Return: final adapted kernel + window statistics

2. **Modify `bayesfilter/hardbound/hmc_runner.py`**:
   - Add `dense_mass_windowed: bool = False` flag to `run_nuts`
   - When `dense_mass_windowed=True`, call windowed adaptation instead of current single-freeze
   - Preserve existing `dense_mass_matrix` path as fallback for comparison

3. **Update test `tests/hardbound/test_phase2_joint_hmc.py`**:
   - Change `dense_mass_matrix=True` to `dense_mass_windowed=True` in G2.3 config
   - Keep all other parameters unchanged (4 chains, 4000 warmup, 3000 samples, seed 20260822)

### Phase 2: Focused Verification

Before full 4-chain run, verify implementation with smaller smoke tests:

1. **Shape and finite check**: 2 chains × 200 warmup × 50 samples, verify:
   - Windowed schedule builds correctly (initial buffer, doubling windows, final buffer)
   - Covariance shrinkage produces finite symmetric 337×337 matrix each window
   - Cholesky factorization succeeds
   - No NaN/Inf in any state

2. **Window mechanics check**: 1 chain × 1000 warmup × 10 samples, inspect:
   - Mass updates occur only at slow window boundaries
   - Shrinkage applied with λ=0.1
   - Condition number improves after shrinkage vs raw Welford covariance

3. **XLA compatibility**: Verify `LinearOperatorLowerTriangular` path remains XLA-compatible (no regression from prior fix)

### Phase 3: Full G2.3 Convergence Run

Execute `pytest tests/hardbound/test_phase2_joint_hmc.py::TestPhase2JointHMC::test_g2_3_full_c1_fixture_recovery -v -s`:

- 4 chains × 4000 warmup × 3000 samples
- 337 dimensions
- Target accept 0.95
- Windowed dense mass with shrinkage λ=0.1
- Expected wall time: ~4 hours (similar to prior dense run)

**Budget**: One full 4-chain run. If convergence fails, perform diagnostic analysis before any retry.

## Default and Assumption Audit

| Choice | Provenance | Justification | Failure Mode | Early Diagnostic |
|--------|-----------|---------------|--------------|------------------|
| λ=0.1 shrinkage | Repository default (`WindowedMassAdaptationConfig.mass_shrinkage`) | Reviewed for n≪p regime; 10% prior weight regularizes rank-deficient empirical covariance | Too much shrinkage → overly conservative (diagonal-like); too little → ill-conditioned | Condition number per window |
| Initial buffer = 75 | Stan default for moderate dimensions | Protects first slow window from off-posterior warmup contamination | Too short → contaminated; too long → fewer refinement windows | First-window covariance vs second-window |
| First window = 25 | Stan default | Small initial window for early covariance refinement | Too small → high variance; too large → slow doubling progression | N/A (convention) |
| ~~Identity as shrinkage target~~ → **diagonal of the empirical covariance** | Amended during Phase 1; see Amendment 1 below | Preserves every marginal variance exactly and is invariant to per-coordinate rescaling, which an identity target is not | Leaves marginal variances unregularized (they are estimated from all pooled draws, so this is the intended scope) | `test_shrinkage_preserves_marginal_variances` (closed-form) |
| Window doubling | Stan-style progressive refinement | Later windows have more on-posterior samples → better covariance estimates | N/A (reviewed algorithm) | N/A |
| 4000 warmup steps | Test configuration (unchanged) | Baseline diagonal succeeded with this budget | Insufficient for dense mass adaptation | Max R̂ outcome |

**Warm-start status**: Shrinkage λ=0.1 and windowed schedule are reviewed repository defaults for n≪p regime, not inherited from cross-model transfer. They are baseline choices for this plan, not promoted convenience settings.

## Phase 1 Amendments (recorded during execution, 2026-08-26)

Two material deviations from the plan as approved. Both were found by reading
the shared interfaces rather than by a failed run, and both are recorded here
because they change what the run tests.

### Amendment 1: shrinkage target is the empirical diagonal, not the identity

The plan specified `(1-λ)·empirical + λ·I`. That target is wrong for this
model. G2.3 is sampled in the raw chart, where per-coordinate posterior sd
spans about 1.9e4; an identity target is therefore negligible against the
large-variance coordinates and dominant over the small ones, so at λ=0.1 it
would replace the small-variance marginals with values four orders of magnitude
too large and destroy the preconditioner rather than regularize it.

The implementation shrinks toward `diag(empirical)` instead:
`(1-λ)·cov + λ·diag(cov)`. Only off-diagonal entries move, every marginal
variance is preserved exactly, and the operation is invariant to
per-coordinate rescaling. This also matches where the small-sample problem
actually is: each of the 337 marginal variances is estimated from all pooled
window draws, while the 56,953 covariance entries are not.

Consequence for the evidence contract: the λ=0.1 non-claim still stands and now
also covers the target choice. Neither λ nor the target has been calibrated for
G2.3; both are baseline choices. This amendment is not evidence that the
diagonal target is optimal, only that the identity target was unusable here.

### Amendment 2: the truncated tail window is merged

`build_windowed_warmup_schedule` doubles until the slow span is exhausted and
emits the remainder as a short final window. At the G2.3 budget the slow
lengths are `[25, 50, 100, 200, 400, 800, 1600, 700]`. The sampling phase
freezes the *last* slow window's metric, so as approved the plan would have
frozen a covariance estimated from 700 warmup steps immediately after a window
that had 1600 — the second-worst late metric, for the whole 3000-draw sampling
phase.

`_merge_truncated_tail_window` extends the last full window to the end of the
slow phase, giving `[25, 50, 100, 200, 400, 800, 2300]`. The shared builder is
left unmodified so other callers keep their schedules.

The 1000-step mechanics check observed condition numbers falling monotonically
with window length (1.7e5 → 5.0e3) and then rising to 5.1e4 on the truncated
tail window, which is the mechanism this amendment removes. That observation is
explanatory only: it is single-seed and descriptive, and it does not establish
that the merge is what makes the run converge.

No claim of equivalence to Stan's tail handling is made. Stan is cited in this
plan only for the buffer and doubling constants; its `windowed_adaptation`
source was not read. The justification is the argument above — never freeze a
metric built from fewer draws than an earlier window already achieved.

## Pre-Mortem

### How the run could pass while misleading us:
- Convergence due to increased warmup budget from window interleaving, not dense mass quality
- R̂ < 1.02 by chance (seed-dependent) rather than systematic improvement
- Shrinkage so aggressive that result is effectively diagonal (would see similar ESS profile)

**Diagnostic**: Compare per-theta ESS and acceptance rate to diagonal baseline. If ESS profile is identical and mass covariance eigenspectrum is near-identity, shrinkage may be too strong.

### How it could fail for implementation reasons:
- Shrinkage target (identity) has wrong scale for G2.3 theta coordinates
- Window boundaries misaligned with TFP kernel state updates
- Cholesky fails on shrunk covariance despite jitter (numerical issue)
- Memory/timeout on 337×337 covariance operations

**Diagnostic**: Phase 2 focused verification (shape/finite, window mechanics, condition numbers).

### How it could fail for tuning reasons:
- λ=0.1 inappropriate for G2.3's actual sample-to-parameter ratio
- Initial buffer too short → first window still contaminated
- Not enough slow windows for refinement (e.g., if final buffer too large)

**Diagnostic**: Window-by-window covariance condition number and per-window acceptance rate.

### Weakest part of evidence:
Using repository default λ=0.1 without G2.3-specific calibration. The n≪p ratio here (3200 samples / 56953 entries ≈ 0.056) is more extreme than typical reviewed cases.

**Mitigation**: If convergence fails, add λ ∈ {0.05, 0.2, 0.3} sweep as follow-up (not in this plan's budget).

## Artifact Location

- **Test output**: pytest stdout (includes R̂, ESS, acceptance rate)
- **Window statistics**: `window_diagnostics` in the runner's return dict —
  per window: kind, bounds, divergences, step size after, and for slow windows
  the eigenvalue extremes, condition number, marginal-variance extremes, and
  pooled draw count
- **Result note**: `docs/plans/g2_3_windowed_dense_mass_convergence_result.md` after execution

## Phase 2 Outcome (2026-08-26)

All six focused checks pass. Implementation is
[windowed_dense_mass_adaptation.py](bayesfilter/hardbound/windowed_dense_mass_adaptation.py),
checks are
[test_windowed_mass_smoke.py](tests/hardbound/test_windowed_mass_smoke.py).

Closed-form and structural checks (no sampling, ~5 s):

| Check | Role | Result |
|---|---|---|
| `test_flat_target_matches_part_target` | hard veto | Pass, exact to 0 atol/rtol — the full-joint flatten/split round trip computes the same density as the block target |
| `test_shrinkage_preserves_marginal_variances` | hard veto | Pass — diagonal-target shrinkage moves only off-diagonals, preserves positive definiteness under a 1e8 variance spread |
| `test_tail_window_merge_at_g2_3_budget` | hard veto | Pass — `[...,800,1600,700]` becomes `[...,800,2300]`, contiguous, reindexed |
| `test_tail_window_merge_is_a_noop_on_clean_schedules` | hard veto | Pass — untouched when the last slow window is already full |

Sampling mechanics checks:

| Check | Role | Result |
|---|---|---|
| `test_windowed_route_shapes_and_finiteness` (2 chains, 200 warmup, 50 draws) | hard veto | Pass — documented output structure, all draws finite, R̂/ESS on shape (9,), 0 sampling divergences, 10 warmup divergences |
| `test_windowed_route_doubling_schedule` (1 chain, 1000 warmup, 10 draws) | hard veto | Pass — slow lengths `[25,50,100,200,500]`, all finite |

Explanatory-only observation from the 1000-step check: slow-window condition
numbers fall monotonically with window length (1.5e5, 7.6e4, 2.7e4, 1.1e4,
4.7e3), consistent with more pooled draws giving a better-conditioned estimate.
Before Amendment 2 the same check ended at 5.1e4 on the truncated tail window.
Single seed, descriptive only; this does not establish that the merge causes
convergence.

What Phase 2 does **not** establish: nothing about the R̂ < 1.02 promotion
criterion. These budgets are one to two orders of magnitude below the G2.3 run
and the mechanics checks assert no convergence property at all.

## Commands

### Phase 2 Focused Verification:
```bash
# Shape and finite check (manual pytest invocation with modified config)
# Modify test temporarily: num_chains=2, num_warmup=200, num_samples=50
pytest tests/hardbound/test_phase2_joint_hmc.py::TestPhase2JointHMC::test_g2_3_full_c1_fixture_recovery -v -s

# Window mechanics check (manual)
# Modify test temporarily: num_chains=1, num_warmup=1000, num_samples=10, add window logging
pytest tests/hardbound/test_phase2_joint_hmc.py::TestPhase2JointHMC::test_g2_3_full_c1_fixture_recovery -v -s
```

### Phase 3 Full Run:
```bash
pytest tests/hardbound/test_phase2_joint_hmc.py::TestPhase2JointHMC::test_g2_3_full_c1_fixture_recovery -v -s
```

## Environment

- **Git commit**: (record at execution time)
- **TensorFlow/TFP version**: (query at execution time)
- **GPU**: CUDA visible devices, TF GPU list, memory growth enabled and verified
- **Backend**: TensorFlow (per repo backend rule)
- **Execution**: GPU default (per repo default execution target)

## Stop Conditions

- **Success**: Max R̂ < 1.02 on all 9 theta coordinates → test passes, plan complete
- **Hard veto**: Any non-finite state, divergence, or max R̂ > 1.05 → diagnose root cause before any retry
- **Tuning failure**: Max R̂ in [1.02, 1.05] → evaluate whether shrinkage/window schedule needs adjustment; consider λ sweep or public tuner route
- **Implementation failure**: Shape error, Cholesky failure, memory error → fix implementation, re-run Phase 2 focused verification

## Skeptical Audit Checklist

- [ ] Wrong baseline: Is diagonal mass the right comparator? **Yes** - it's the current passing test configuration and the alternative being evaluated against.
- [ ] Proxy metric as promotion criterion: Is R̂ < 1.02 the right criterion? **Yes** - it's the test's explicit assertion, not a proxy. ESS and acceptance are explanatory only.
- [ ] Missing stop condition: Are all veto and success cases covered? **Check above** - success, hard veto, tuning failure, implementation failure all defined.
- [ ] Unfair comparison: Are diagonal and windowed dense runs comparable? **Yes** - same seed, chains, warmup budget, samples, target, hardware.
- [ ] Hidden assumption: Identity as shrinkage target appropriate? **Flagged in pre-mortem** - this is the weakest assumption; failure mode is wrong scale.
- [ ] Stale context: Is G2.3 test structure still as described? **Verify** before execution.
- [ ] Environment mismatch: GPU memory growth required per repo rule? **Yes** - included in environment section, must verify before run.
- [ ] Command artifacts: Do the commands produce R̂ values for promotion criterion? **Yes** - pytest output includes rank-normalized split R̂.

## Next Action After Plan Approval

1. Perform final skeptical audit review (above checklist)
2. Implement Phase 1 (windowed dense mass adaptation for hardbound)
3. Execute Phase 2 focused verification
4. If Phase 2 passes, execute Phase 3 full convergence run
5. Write result note with decision table and inference-status table per CLAUDE.md scientific coding development policies
