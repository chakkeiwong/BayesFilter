> **SUPERSEDED 2026-09-06** by
> `docs/plans/ledh-surrogate-hmc-unified-program-2026-09-06.md` Phase 4.
> Retained as historical record. This plan would have executed against a lane
> that cannot run the production program, using a runner that silently
> disables the Contract-E reset. Do not execute from this file.

# Phase 3: LEDH Filter Application — Implementation Plan

**Date:** 2026-09-04  
**Phase:** Phase 3 (LEDH Filter Application)  
**Program:** `ledh-surrogate-force-hmc-master-program-2026-09-04.md`

## SCOPE

Run surrogate-force HMC with LEDH filtering on d=3 T=50 LGSSM. Test the core hypothesis: can we decouple correctness (exact value) from efficiency (damped score)?

## IMPLEMENTATION TASKS

### Task 1: LEDH Surrogate-Force Adapter ✅ IN PROGRESS

**File:** `bayesfilter/inference/ledh_surrogate_force_adapter.py`

**Requirements:**
- Wrap `canonical_batch_fused_value_score` from refactored kernel
- Two configurations:
  - Exact: λ=1e-5, δ=1e-5 (for value, acceptance)
  - Damped: λ=1e-3, δ=1e-3 (for force, leapfrog)
- Return `BatchValueScoreResult(value=exact, score=damped)`

**Interface:**
```python
class LEDHSurrogateForceAdapter:
    def __init__(
        self,
        model: PerPointScoreModel,
        exact_ridge: tuple[float, float],  # (λ, δ) for value
        damped_ridge: tuple[float, float],  # (λ, δ) for force
        ...
    ):
        ...
    
    def log_prob_and_grad(self, theta: tf.Tensor) -> BatchValueScoreResult:
        """Exact value at theta, damped force at theta."""
        ...
```

### Task 2: LGSSM Test Model

**Use existing:** `bayesfilter.ssm.lgssm` module  
**Configuration:** d=3 (state dim), T=50 (horizon), N=1000 particles

**Parameters to infer:** θ = process noise scale (1D for simplicity)

### Task 3: Runner Script

**File:** `docs/benchmarks/run_ledh_surrogate_hmc_phase3_20260904.py`

**3-Arm Comparison:**
1. **Exact:** λ=1e-5 for both value and force (baseline)
2. **Damped:** λ=1e-5 value, λ=1e-3 force (surrogate-force)
3. **Intermediate:** λ=1e-5 value, λ=1e-4 force (sanity check)

**HMC Configuration:**
- Chains: 4
- Warmup: 1000 (adaptive)
- Sampling: 1000 (post-warmup)
- Step size: 0.01 (initial, will adapt)
- Leapfrog steps: 10

**Diagnostics:**
- Acceptance rate per arm
- ESS per parameter
- Split R-hat
- Posterior mean/std comparison
- Trace plots

### Task 4: Analysis Script

**File:** `docs/benchmarks/analyze_ledh_surrogate_hmc_phase3_20260904.py`

**Analyses:**
- Acceptance comparison table
- ESS comparison table
- R-hat check (all < 1.01?)
- Posterior comparison (do all 3 arms agree?)
- Trace plot visualization
- Diagnostic payload extraction

### Task 5: Result Document

**File:** `docs/plans/ledh-surrogate-hmc-phase3-result-2026-09-04.md`

**Must include:**
- Decision table (promote/veto)
- Acceptance rates per arm
- ESS per arm
- R-hat per arm
- Posterior agreement check
- Full diagnostic payload summary
- Promotion verdict

## SUCCESS CRITERIA

From master program:

**Promotion Criteria:**
1. ✅ Acceptance ≥ 0.2 for all 3 arms
2. ✅ ESS > 100 per parameter per chain
3. ✅ R-hat < 1.01 for all parameters
4. ✅ Posterior agreement: all 3 arms within 2 posterior SD

**Promotion Vetoes:**
1. ❌ Acceptance < 0.1 for damped arm
2. ❌ R-hat > 1.05 for any arm
3. ❌ Posterior disagreement > 3 SD between arms

## BUDGET

**GPU Time:** 6-12 hours (3 arms × 2-4h each)  
**Attempts:** 3 attempts budgeted  
**Wall Time:** 1-2 days (including analysis)

## EXECUTION ORDER

1. ✅ Write adapter (Task 1)
2. ✅ Write runner (Task 3)
3. ▶️ Run experiment (requires GPU, escalated permissions)
4. ⏳ Analysis (Task 4)
5. ⏳ Result document (Task 5)

## CURRENT STATUS

**Starting Task 1: Adapter implementation...**
