# Phase 2W Result: LGSSM Full-Shard Score/FD Split Repair

Date: 2026-07-09

Status: `BLOCKED_FIXABLE_COMPACT_SCORE_PASS_FULL_RUNTIME_NO_ARTIFACT`

## Phase Objective

Split full-shard score computation from finite-difference correctness so the
runbook can determine whether Phase 2V's single-shard no-artifact blocker was
caused by the compact score pass, the FD correctness pass, or their unsplit
combination.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the full-shard no-artifact blocker caused by compact score execution, FD correctness execution, or the unsplit combination? |
| Baseline/comparator | Phase 2V single-shard full command with no artifact after bounded window. |
| Primary criterion | A trusted GPU score-only `N=10000,T=50` shard either emits a durable raw score-only artifact or writes a precise score-pass blocker. |
| Secondary criterion | If score-only emits, FD-only emits a matching correctness artifact or writes a precise FD blocker. |
| Admission criterion | No full admission until combined shards aggregate and `validate_ledh_score_artifact(..., require_admitted=True)` passes. |
| Veto observed | Full score-only shard did not emit a raw artifact within the bounded window. |

## Implementation Completed

- Added `--score-diagnostic-stage {score-and-fd,score-only,fd-only}` to
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`.
- Added `--score-reference-json` for FD-only diagnostics.
- Split `_manual_score_diagnostic` into score-only and FD-only helper paths.
- Score-only diagnostics return compact no-tape score but mark FD as
  `not_run_score_only`.
- FD-only diagnostics read a score-only reference and run the value-only scalar
  finite-difference route without calling the compact score/JVP route.
- Main runner blocks split-stage admission with
  `blocked_score_diagnostic_stage_not_admitted`.

## Checks Passed

- Focused CPU-hidden tests:

```text
48 passed, 2 warnings
```

- Trusted GPU score-only smoke at `N=256,T=3` passed and remained non-admitted:
  - `score_diagnostic_stage = score-only`;
  - `same_scalar_fd.status = not_run_score_only`;
  - `score_runtime_gate_applicable = true`.
- Trusted GPU FD-only smoke at `N=256,T=3` passed using the score-only smoke
  artifact:
  - `score_diagnostic_stage = fd-only`;
  - `same_scalar_fd.status = pass`;
  - `same_scalar_fd.uses_value_only_scalar_route = true`;
  - split-stage result remained non-admitted.

## Full Score-Only Attempt

Launched:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 10000 \
  --time-steps 50 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode compact-sensitivity \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-n10000-score-admission-repair-phase2w-lgssm-score-fd-split/lgssm-seed-81120-score-only.json
```

Log:

`docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2w-score-only-seed-81120.log`

Observed:

- TensorFlow initialized trusted `/GPU:0` and compiled XLA.
- GPU memory stayed near `15712 MiB / 16376 MiB`.
- No score-only shard JSON was emitted after about 6.75 minutes.
- The process was stopped with `kill 262046`.
- Post-stop checks showed no remaining compute app and GPU memory near
  `1862 MiB / 16376 MiB`.

## Gate Assessment

Phase 2W local and smoke gates passed, but the full score-only shard did not
emit an artifact. This narrows the blocker: the full-scale compact
forward-sensitivity score pass itself is the current runtime/artifact blocker,
not only the coordinate finite-difference correctness arm or the earlier
five-seed batching.

## Root Cause Hypothesis

The compact score pass carries full particle tangents through each filtering
step. In particular, the transport JVP path materializes and stacks tangent
blocks in:

- `_filterflow_streaming_softmin_jvp`;
- `_filterflow_streaming_column_log_normalizer_jvp`;
- `_filterflow_streaming_transport_from_potentials_jvp`.

At full `N=10000,T=50`, the score-only path still holds high GPU memory and
does not emit within the bounded window. The next repair should target the
compact score kernel/tensor-lifetime design, not additional artifact plumbing.

## Required Next Subplan

Draft Phase 2X to implement or prototype a reduce-only compact score path for
LGSSM that avoids carrying/storing full transported particle tangents when only
the mean score over seeds is needed, or otherwise repairs tensor lifetimes in
the transport JVP. The subplan must keep the same finite-`N` LEDH scalar and
must not fall back to `GradientTape`, `ForwardAccumulator`, stopped partials,
or historical `manual_total_vjp*` admission evidence.

## Nonclaims

This result does not admit LGSSM, does not reject the compact score derivation,
does not establish HMC readiness, and does not authorize moving to fixed-SIR
before a reviewed handoff condition is satisfied.
