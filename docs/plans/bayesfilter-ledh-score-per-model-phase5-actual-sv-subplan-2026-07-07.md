# Phase 5 Subplan: Actual-SV Score

metadata_date: 2026-07-07
status: `DRAFT_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5

## Phase Objective

Build and admit, or explicitly block, the LEDH score for:

```text
zhao_cui_sv_actual_nongaussian_T1000
```

The score is the no-tape total derivative of the same realized finite-`N`
LEDH estimator admitted by the actual-SV value artifact:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

The target observation policy is:

```text
transformed_actual_sv_log_y_square
```

The score parameter order is:

```text
gamma_unconstrained, log_beta
```

in theta coordinate system:

```text
synthetic_unconstrained
```

## Entry Conditions Inherited From Previous Phase

- LGSSM score is blocked/not admitted.
- Fixed-SIR score is blocked/not admitted.
- Predator-prey score is blocked/not admitted.
- Phase 1 score artifact schema and no-tape guards exist.
- The actual-SV value artifact is admitted at `N=10000,T=1000` with seeds
  `[81120,81121,81122,81123,81124]`.
- No actual-SV LEDH score adapter is currently admitted.

## Required Artifacts

Source value artifact:

- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`

Relevant value code and tests:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`
- `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py`
- `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py`

Expected score code and tests:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`

Expected score artifacts:

- tiny score diagnostic:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.json`
- if admitted, full score artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-score-artifact-2026-07-07.json`
- Phase 5 result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-result-2026-07-07.md`
- Phase 6 generalized-SV subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase6-generalized-sv-subplan-2026-07-07.md`
- review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase4-calibration-result-phase5-subplan-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

Preflight checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Implementation checks after adding the score adapter:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Tiny score smoke before any full score attempt:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  --source-value-artifact docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json \
  --output docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.json \
  --markdown-output docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.md \
  --batch-seeds 81120 \
  --time-steps 2 \
  --num-particles 64 \
  --sinkhorn-iterations 2 \
  --row-chunk-size 64 \
  --col-chunk-size 64 \
  --particle-chunk-size 64 \
  --dtype float64 \
  --tf32-mode disabled \
  --fd-step 1.0e-4
```

Full-row score admission may be attempted only after:

- exact target bridge tests pass;
- all-parameter tiny same-scalar FD passes;
- no-tape static and runtime sentinels pass;
- a reviewed full-row score/memory command is written with explicit evidence
  contract and stop conditions.

Review:

- bounded read-only review of this subplan before implementation;
- bounded read-only review of the Phase 5 result and Phase 6 subplan before
  Phase 6 execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can actual-SV produce a no-tape total derivative of the same finite-`N` exact transformed LEDH `log_likelihood` scalar admitted by the value artifact? |
| Baseline/comparator | Admitted actual-SV value artifact, exact transformed log-square target density, same-scalar coordinate FD with fixed randomness, and Phase 1 score artifact validator. |
| Primary criterion | Full score admission requires an artifact validating with `validate_ledh_score_artifact(..., require_admitted=True)`, all two parameters checked, no-tape provenance, same value/score route identity, and `N=10000,T=1000` memory pass. |
| Veto diagnostics | KSC mixture used as actual-SV target, raw Gaussian observation likelihood used as target, augmented-noise Gaussian closure, wrong theta coordinate/order, wrong scalar, wrong output field, target-density correction missing, tape/autodiff, stopped partial, nonfinite score, tiny diagnostic promoted as full, failed FD/exact reference, or memory failure. |
| Explanatory diagnostics | Runtime, compile time, memory, FD step sensitivity, per-coordinate error, per-seed dispersion, dtype/TF32 mode, and device placement. |
| Not concluded | HMC readiness, posterior correctness, exact native raw-observation likelihood, KSC score admission, generalized-SV score admission, scientific superiority, runtime ranking, or all-algorithm comparison. |
| Artifact | Score adapter, tests, tiny diagnostic, optional full score artifact, Phase 5 result, Phase 6 subplan, and review bundle. |

## Step-By-Step Plan

1. Replay the admitted actual-SV value artifact and Phase 1 score schema tests.
2. Inventory actual-SV value runner internals and identify the exact forward
   scalar components needed by score:
   - transition log density;
   - exact transformed observation log density;
   - pre-flow Gaussian proposal density;
   - flow log determinant;
   - streaming transport branch and randomness.
3. Implement a dedicated score adapter that reuses the same actual-SV value
   target and emits Phase 1 score artifacts.
4. Implement manual/no-tape parameter VJPs for:
   - `gamma_unconstrained -> gamma`;
   - `log_beta -> beta`;
   - stationary initial density;
   - SV transition density;
   - exact transformed log-chi-square observation correction;
   - proposal-flow observation surface where it affects the same finite-`N`
     realized estimator.
5. Add tests that reject:
   - KSC mixture target substitution;
   - raw Gaussian target substitution;
   - augmented-noise Gaussian closure;
   - wrong theta coordinate/order;
   - any `GradientTape`, `ForwardAccumulator`, or stopped-partial route in
     admitted score provenance;
   - tiny diagnostic artifacts promoted as full.
6. Run tiny all-coordinate same-scalar FD in FP64 CPU-hidden mode.
7. If tiny FP64 passes, draft a reviewed full-row score/memory subplan before
   any `N=10000,T=1000` score run.
8. If any target, derivative, or memory boundary is ambiguous, write a blocker
   result and stop.
9. Write the Phase 5 result and draft or refresh the Phase 6 generalized-SV
   subplan.
10. Review the Phase 5 result and Phase 6 subplan.

## Forbidden Claims/Actions

- Do not claim score admission from value admission.
- Do not claim score admission from tiny diagnostics.
- Do not use `GradientTape`, `ForwardAccumulator`, hidden autodiff, or stopped
  partial derivatives for admitted score evidence.
- Do not substitute KSC finite-mixture likelihood for actual-SV target
  evidence.
- Do not substitute raw Gaussian observation likelihood or augmented-noise
  Gaussian closure for the exact transformed actual-SV target.
- Do not change row id, target scalar, output field, target observation policy,
  theta coordinate system, parameter order, seeds, `N`, or `T`.
- Do not run a full `N=10000,T=1000` score admission command until a reviewed
  full-row score/memory subplan exists.

## Exact Next-Phase Handoff Conditions

Phase 6 generalized-SV may start only if:

- Phase 5 writes an admitted score result or explicit blocker result;
- Phase 6 subplan exists and preserves generalized-SV target boundaries;
- required local checks pass;
- read-only review agrees the Phase 5 decision and Phase 6 handoff are
  boundary-safe.

## Stop Conditions

Stop and write a blocker result if:

- actual-SV value artifact fails replay validation;
- no-tape provenance becomes ambiguous;
- the score route uses KSC, raw Gaussian, or augmented-noise target evidence;
- tiny all-coordinate FD fails without a reviewed fixable implementation
  explanation;
- full-row memory/correctness evidence cannot be produced within bounded
  execution;
- score artifact cannot validate against the admitted value artifact;
- review finds a material issue that does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | The admitted actual-SV value artifact is the only score source anchor. |
| Proxy promotion | Value admission, runtime, memory, and tiny FD cannot by themselves admit full score. |
| Missing stop condition | Stop on target substitution, no-tape ambiguity, FD failure, validation failure, or memory failure. |
| Hidden assumption | Parameter order and theta coordinate are copied from the value artifact and must be tested. |
| Stale context | Phase starts with score-route inventory rather than assuming a score adapter exists. |
| Environment mismatch | CPU-hidden tiny checks are diagnostic; GPU full-row checks require trusted execution and a reviewed subplan. |
| Useless artifact | Any admitted score must validate with `validate_ledh_score_artifact(..., require_admitted=True)`. |
