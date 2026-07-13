# Phase 2U Result: LGSSM Full N10000 Score Admission Run

Date: 2026-07-09

Status: `BLOCKED_FIXABLE_FULL_RUNTIME_NO_ARTIFACT`

## Phase Objective

Run the full LGSSM `N=10000,T=50` compact score admission attempt using the
Phase 2S score-procedure repair and Phase 2T disclosed no-TF32 correctness arm.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can LGSSM emit a validator-admitted compact `N=10000,T=50` score artifact under the repaired score procedure and reviewed correctness policy? |
| Baseline/comparator | Phase 2S/2T smoke gates and admitted LGSSM value artifact. |
| Primary criterion | Output nested `score_artifact` validates with `validate_ledh_score_artifact(..., require_admitted=True)` and reports score-specific `n10000_memory_pass = true`. |
| Veto diagnostics | No artifact; validator failure; FD correctness fail; score-memory peak above 14 GiB; raw top-level status admitted while nested artifact is not; hidden no-TF32 substitution; target/row/parameter mismatch; historical route. |
| Not concluded | Non-LGSSM admission, HMC readiness, posterior correctness, exact Kalman score equality, runtime ranking, or scientific superiority. |

## Command Launched

The reviewed trusted GPU full-row command was launched with:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 10000 \
  --time-steps 50 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode compact-sensitivity \
  --score-fd-tf32-mode disabled \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.json \
  --markdown-output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.md
```

Full output was redirected to:

`docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-full-run-2026-07-09.log`

## Observed Result

- TensorFlow initialized trusted `/GPU:0` and compiled an XLA cluster.
- During the bounded monitoring window the process occupied about
  `15744-15754 MiB / 16376 MiB`.
- No full score artifact was written at
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.json`.
- The launched process was interrupted and then killed:
  - shell PID `229799`;
  - Python PID `229800`.
- Trusted post-kill checks showed no remaining compute app and GPU memory back
  to about `2057 MiB / 16376 MiB`.

## Gate Assessment

The Phase 2U primary criterion did not pass because the required artifact was
not emitted and therefore could not be validator-admitted.

This is a fixable procedural/runtime blocker rather than evidence that the
compact score mathematics is wrong. The prior Phase 2S/2T smoke gates still
stand, but they are not full LGSSM score admission.

## Root Cause Hypothesis

The monolithic five-seed full run combines:

- value execution;
- full compact forward-sensitivity score over a `batch x N x state_dim x param`
  tangent state;
- score-memory measurement;
- same-scalar finite-difference checks;
- JSON/Markdown artifact emission only after all preceding work completes.

At `batch=5,N=10000,T=50`, this makes the run fragile: a late timeout or memory
pressure loses all completed partial evidence. The code also has no durable
checkpoint between seed-level score completion and final artifact admission.

## Required Next Subplan

Draft Phase 2V to repair the procedure before any new full attempt:

- emit durable per-seed raw score shards for the exact fixed full-row seeds;
- aggregate the score and finite-difference correctness by the exact batch mean
  contract;
- preserve the same target scalar, same source value artifact, same parameter
  order, and compact no-tape provenance;
- disclose segmented execution and use the maximum shard score-memory peak as
  the memory diagnostic;
- add tests that per-seed aggregation matches direct batch aggregation on a
  small deterministic case and that partial shards cannot be admitted.

## Nonclaims

This result does not admit the LGSSM score, does not reject the compact score
derivation, does not prove exact Kalman score equality, and does not authorize
moving to later model phases before Phase 2V is reviewed and either admits
LGSSM or writes a precise blocker.
