# Audit: MP1 MLCOE particle adapter gate plan

## Date

2026-05-10

## Scope

Independent developer-style audit of:

`docs/plans/bayesfilter-student-dpf-baseline-mp1-mlcoe-particle-adapter-plan-2026-05-10.md`.

This audit is limited to the quarantined student DPF experimental-baseline lane.

## Verdict

The plan is executable and correctly scoped.  It addresses the largest current
asymmetry in the student-baseline harness: the advanced snapshot exposes
bootstrap-PF diagnostics, while the MLCOE adapter is still Kalman-only.

Proceed with MP1 under the plan's primary criteria and veto diagnostics.

## Required Tightenings

1. MLCOE BPF does not expose a direct resampling event log.  Any resampling
   count must be labeled as threshold-inferred from the BPF condition
   `ess < 0.1 * num_particles`, not as an explicit student-code event record.
2. MLCOE BPF stores `ess` before a possible resampling step while particle
   states and weights are post-step state.  Reports must label ESS as
   pre-resampling ESS for the step.
3. MLCOE BPF does not expose a filtering likelihood estimate.  Reports must
   keep `log_likelihood` and likelihood-surrogate fields null for MLCOE BPF
   unless a later, separately audited likelihood estimator is added.
4. Cross-student PF comparisons must use state/covariance/ESS/runtime metrics
   only.  Advanced PF log-likelihood can be compared to Kalman separately, but
   it must not be compared to an absent MLCOE BPF likelihood.
5. TensorFlow seeding must be explicit per run.  The implementation should call
   both `np.random.seed(seed)` and `tf.random.set_seed(seed)` before BPF
   initialization.
6. The adapter-local fixture bridge must not import or patch production
   `bayesfilter/` code.  It should be a small object built inside
   `mlcoe_adapter.py`.
7. Runtime and artifact size should be checked before committing.  If the full
   panel is too slow, reduce the panel and record the reduction in the result
   report rather than silently changing the plan.

## Missing Points Covered by the Plan

- Lane separation is explicit.
- Production code is out of scope.
- Vendored student code edits are prohibited.
- Missing metrics are treated as null or structured blockers.
- The phase has stop rules if MLCOE BPF cannot run through the adapter.
- The reset memo must be updated with phase results and next-phase
  justification.

## Audit Decision

Proceed to MP1.0 preflight and MP1.1 adapter probe.  Continue automatically
only if the adapter can produce structured `BaselineResult` records without
vendored-code or production-code edits.
