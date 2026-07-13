# Phase 2V Result: LGSSM Seed-Sharded N10000 Score Admission Repair

Date: 2026-07-09

Status: `BLOCKED_FIXABLE_SINGLE_SHARD_FULL_RUNTIME_NO_ARTIFACT`

## Phase Objective

Repair the Phase 2U monolithic no-artifact blocker by implementing exact
fixed-seed sharded aggregation, then attempting a full `N=10000,T=50` shard.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the full LGSSM score be admitted by exact fixed-seed sharding and aggregation without changing the target scalar? |
| Baseline/comparator | Phase 2U monolithic no-artifact blocker, Phase 2S/2T smoke gates, and admitted LGSSM value artifact. |
| Primary criterion | Aggregate nested score artifact validates with `validate_ledh_score_artifact(..., require_admitted=True)` against the admitted LGSSM value artifact. |
| Comparator check | On a small deterministic case, direct multi-seed batch score equals seed-sharded aggregate within stated float32 tolerance. |
| Veto observed | The first full `N=10000,T=50` shard did not emit a raw shard artifact within the bounded window. |
| Not concluded | LGSSM score admission, monolithic five-seed memory pass, runtime ranking, HMC readiness, posterior correctness, exact Kalman score equality, or non-LGSSM admission. |

## Implementation Completed

- Added seed-sharded aggregation helpers and CLI aggregation mode to
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`.
- Aggregation requires the admitted LGSSM source value artifact's exact fixed
  seed set: `[81120, 81121, 81122, 81123, 81124]`.
- Per-seed raw shards are explicitly diagnostic and are rejected if marked as
  admitted.
- The aggregate score and aggregate finite-difference score are arithmetic
  means over the fixed full-row seeds.
- Aggregate artifacts disclose segmented execution and state that they do not
  claim monolithic five-seed memory or runtime.

## Checks Passed

- Focused CPU-hidden tests:

```text
46 passed, 2 warnings
```

- Python compile check for the LGSSM runner passed.
- Synthetic CLI aggregation smoke passed and emitted a validator-valid
  synthetic aggregate with segmented execution disclosure.
- Trusted GPU one-seed prefix smoke at `N=256,T=3` passed:
  - `same_scalar_fd.status = pass`;
  - `score_runtime_gate_applicable = true`;
  - shard remained non-admitted because it was not a full row.

## Full Shard Attempt

Launched the first reviewed full shard:

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
  --score-fd-tf32-mode disabled \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-n10000-score-admission-repair-phase2v-lgssm-shards/lgssm-seed-81120-raw-score.json
```

Log:

`docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2v-lgssm-seed-81120.log`

Observed:

- TensorFlow initialized trusted `/GPU:0` and compiled XLA.
- GPU memory stayed near `15738 MiB / 16376 MiB`.
- No shard JSON was emitted after about six minutes.
- The process was stopped with `kill 251240`.
- Post-stop checks showed no remaining compute app and GPU memory near
  `1879 MiB / 16376 MiB`.

## Gate Assessment

Phase 2V improved the procedure and made aggregation auditable, but it did not
admit LGSSM. Sharding alone does not remove the full-score blocker because a
single `N=10000,T=50` score shard still did not emit an artifact under the
bounded window.

## Root Cause Hypothesis

The remaining blocker is inside the full single-shard score execution, not only
inside five-seed batching or artifact aggregation. The current shard command
does all of the following before writing a raw shard artifact:

- value execution;
- compact forward-sensitivity score execution;
- same-scalar coordinate finite differences over all parameters;
- JSON emission only after both score and FD complete.

The next repair must split score-only and FD-only work so the runbook can tell
whether the bottleneck is the compact score pass, the FD correctness arm, or
their combination.

## Required Next Subplan

Draft Phase 2W to add an execution mode that can:

- emit raw score-only full-shard artifacts immediately after compact score
  completion;
- optionally skip or defer coordinate FD without allowing admission;
- run FD-only/value-only correctness as a separate bounded stage against a
  completed score-only shard;
- keep full admission blocked until score and correctness are both present and
  aggregated through the existing validator.

## Nonclaims

This result does not admit the LGSSM score, does not reject the compact score
math, does not prove exact Kalman score equality, and does not authorize moving
to fixed-SIR before Phase 2W is reviewed or a human explicitly changes the
handoff condition.
