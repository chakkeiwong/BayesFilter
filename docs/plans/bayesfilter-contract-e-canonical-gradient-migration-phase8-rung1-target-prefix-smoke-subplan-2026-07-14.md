# Phase 8 Subplan: Target-Prefix Canonical Harness Smoke

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `CLOSED_WIRING_SMOKE_PASSED_DESCRIPTIVE_ONLY`

## Phase Objective

Verify the new preparation, target-observation, canonical-callable, Kalman, and
telemetry wiring on the smallest target prefix without making a scientific
equivalence or numerical-default decision.

## Entry Conditions

- Rung 0A dtype repair, Rung 0B oracle harness, and preparation/telemetry
  infrastructure passed their narrow gates.
- Formal Phase 1 FD, target ridge adequacy, and owner primary statistical design
  remain blocked.
- No `T=50,N=10000` result may be observed.

## Frozen Smoke Configuration

Execute CPU-hidden float64 only at dataset seed `81100`, estimator seed `81120`,
`T=1,N=4`. Use the new PHILOX preparation builder. The fixed reset mask is
active-all. Transfer the Phase 5 fixture values exactly: ridge `4`, epsilon
`1/2`, scaling `3/4`, two Sinkhorn steps, row/column chunks `2`.

The evaluated physical parameter vector is exactly
`[0.72,0.55,0.35,0.35,0.45]`. The first-observation float64 serialized-tensor
SHA-256 is frozen before execution as
`ded8c5326f970868dccebe2719af8302bbf9c2124bb5daf909c1956b24e6373f`;
the CLI must derive it from the repository dataset generator and fail before
canonical execution if the hash, shape `[1,3]`, or parameter vector differs.

These values are a `fixture_transfer_harness_smoke_only` arm. Ridge `4` is
expected to have material raw-covariance bias and is not a target hypothesis,
candidate, comparator, or default. No setting may be changed after inspecting
the result in this subplan.

## Executable Hard-Check Contract

The canonical callable is `jit_compile=True` and is invoked exactly twice at
the identical center. Repeatability means exact equality of
`tf.io.serialize_tensor` bytes for every canonical output key on those two
calls; no tolerance is used. Chart validity means `reduce_all(valid_chart)` is
true and every element of `flow_valid_history`, `geometry_valid_history`,
`quotient_valid_history`, and `reset_valid_history` is true. Scalar identity
means the repeated objective and score serialized bytes are equal; this smoke
does not compare them to Phase 5 fixture values because the observations and
prepared randomness differ.

Telemetry schema `contract_e_phase8_target_prefix_telemetry_v1` requires these
exact canonical keys:

```text
quotient_mass_history, quotient_row_residual_history,
target_mean_history, target_covariance_history,
output_mean_history, output_covariance_history,
injected_covariance_history, reset_affine_history,
ridged_identity_residual_history, ridged_identity_scale_history,
ridged_identity_residual_fro_history,
raw_covariance_residual_history,
predicted_raw_covariance_residual_history,
raw_covariance_prediction_error_history,
raw_covariance_residual_fro_history,
raw_covariance_prediction_error_fro_history,
mean_residual_history, mean_residual_infinity_history,
residual_design_sum_history, residual_design_absolute_scale_history,
gap_chol_diagonal_history, target_chol_diagonal_history,
injected_chol_diagonal_history, gap_condition_proxy_history,
target_condition_proxy_history, injected_condition_proxy_history,
realized_ridge_history, active_reset_history
```

Every required tensor must have the statically expected `B=1,T=1,N=4,d=3`
shape, be finite except boolean masks, and be serialized in full with dtype,
shape, value, and serialized-tensor SHA-256. The CLI uses a fixed schema
validator; a missing, extra, wrong-shape, or nonfinite required field fails the
smoke. The preparation identity and all of its tensor hashes are also required.

## Execution And Attempt Budget

The first and only planned result path is
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/rung1-target-prefix-smoke-attempt1/result.json`.
The exact runtime prefix is
`CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 MPLCONFIGDIR=/tmp`, set before
TensorFlow import. The command must record Python/TensorFlow versions, Git
revision, logical devices, intentional CPU-hidden role, `jit_compile=true`,
command, wall time, dataset/preparation/source hashes, and output path.

The CLI refuses an existing output path. The attempt timeout is `300` seconds.
At most one retry is allowed, only for a localized wiring/serialization defect,
with unchanged scientific configuration and a fresh `attempt2/result.json`;
both attempts remain preserved and the repair is recorded. A nonfinite value or
invalid chart is not retryable under this subplan. No attempt may cross the
campaign end at approximately `2026-07-14T09:32:19+08:00`.

## Required Artifacts

- a dedicated target-prefix smoke CLI importing only the target data generator,
  owned preparation module, canonical factory, and Kalman oracle;
- one structured JSON with source/preparation hashes, values/scores, physical
  and HMC-coordinate differences, all telemetry, and nonclaims;
- exact environment/run provenance and a no-overwrite attempt record;
- focused tests proving no historical raw route import and exact shape/config;
- a result or blocker record;
- the next numerical-design subplan, without selecting a setting from smoke
  output.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is target-prefix wiring executable, finite, repeatable, branch-valid, and telemetry-complete? |
| Comparator | Exact float64 Kalman value/autodiff gradient for the same first observation and transition-first likelihood |
| Pass criterion | CLI/hashes complete; canonical and oracle finite; two canonical calls exactly equal; all four chart-history predicates true; fixed telemetry schema complete and finite |
| Hard vetoes | wrong frozen observation hash/shape/timing/theta; raw route import; telemetry schema mismatch; nonfinite output; invalid chart/history; serialized repeat drift; existing output path; wrong CPU-hidden/XLA provenance |
| Explanatory only | every Contract E/Kalman difference and every telemetry magnitude |
| Not concluded | same-program formal FD, ridge/transport/reset adequacy, Kalman equivalence, larger-prefix feasibility, GPU/HMC/default/leaderboard readiness |

## Skeptical Plan Audit

Decision: `PASS_FOR_HARNESS_SMOKE_ONLY`.

- The transferred values are deliberately poor scientific defaults and are
  labeled accordingly; success cannot promote them.
- `N=4,T=1` answers only whether the new target-prefix wiring works.
- A failed chart is evidence that the transferred fixture configuration is not
  a valid smoke arm, not evidence against Contract E or the target.
- No repair may tune numerical settings from the observed smoke result. Only a
  wiring/serialization defect may be repaired and rerun unchanged.
- The exact-equality and schema predicates are engineering checks, not numerical
  adequacy thresholds. They were frozen before target-prefix output.

## Required Checks And Reviews

1. Implement and test the bounded CLI and exact schema validator.
2. Run the exact CPU-hidden/XLA command with the frozen attempt-1 path and
   `300`-second timeout.
3. Verify target data/hash/timing, hard checks, telemetry completeness, and
   nonclaims.
4. Write the result and a separate next numerical-design subplan.
5. Review the next plan before any new numerical arm is executed.

## Forbidden Claims And Actions

- Do not call ridge `4` or any transferred setting reasonable, adequate, or a
  candidate for the target.
- Do not tune or select from the smoke output.
- Do not run `T=10`, another `N`, float32/GPU, primary shape, HMC, nonlinear,
  leaderboard, release, or integrity work.
- Do not apply the `0.1%` value or unfrozen gradient criterion to this smoke.

## Exact Handoff Conditions

The next numerical-design plan must be drafted after either a passing smoke or
a blocker result. If the frozen transferred arm has a nonfinite value or invalid
chart, the result is `TRANSFERRED_FIXTURE_ARM_INVALID_FOR_TARGET_PREFIX`; the
next plan may define independent, pre-result numerical hypotheses from
mathematics and telemetry definitions, but it must not select or tune settings
from the observed smoke magnitudes. A localized harness defect may be repaired
once under the unchanged arm. No scientific or default handoff is possible
from this smoke.

## Stop Conditions

Stop for wrong frozen target hash/shape/timing/theta, raw route reachability,
telemetry schema failure, nonfinite/invalid chart under the frozen arm, exact
serialized repeat drift, wrong CPU-hidden/XLA provenance, attempt timeout,
campaign-clock exhaustion, or an unrepairable harness defect. Preserve a failed
artifact and do not replace its configuration inside this subplan.
