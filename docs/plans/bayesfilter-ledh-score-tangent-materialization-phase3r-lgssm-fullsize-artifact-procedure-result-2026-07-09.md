# Phase 3R Result: LGSSM Full-Size Artifact Procedure Repair

Date: 2026-07-09

Status: `PARTIAL_PASS_BLOCK_INCOMPLETE_PROGRESS_ARTIFACT`

## Objective

Make the full-size LGSSM score diagnostic emit observable JSON evidence rather
than disappearing without an artifact.

## Implementation

Updated:

- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `tests/test_ledh_lgssm_manual_score_phase4.py`

The runner now writes JSON atomically through `_write_json_atomic` and emits
progress or terminal records at key boundaries:

- `started`
- `initialized`
- `value_completed`
- `score_started`
- `score_completed`
- `failed_exception`
- `completed`

Progress records are nonterminal and include PID, timestamps, elapsed seconds,
shape, transport settings, current TensorFlow GPU memory info, and
score-started/score-completed flags. Terminal success artifacts remain
backward-compatible with the previous final result schema and now include
`artifact_status=completed` and `terminal_artifact=true`.

## Reviews

Subplan review:

- Fresh Codex review initially returned `VERDICT: REVISE` because the subplan
  did not classify progress-only artifacts.
- Patched the subplan to add terminal artifact criteria, process metadata, and
  `BLOCK_INCOMPLETE_PROGRESS_ARTIFACT`.
- Focused re-review returned `VERDICT: AGREE`.

Implementation review:

- Fresh Codex review initially returned `VERDICT: REVISE` because a
  post-score exception could misreport `score_started`.
- Patched the runner to set `last_completed_stage="score_started"` before score
  execution, preserve `score_finite`, and avoid clobbering derived flags in the
  exception artifact.
- Focused re-review returned `VERDICT: AGREE`.

Claude was not retried for this program because the approval reviewer rejected
the bounded Claude artifact disclosure path.

## Local Checks

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
57 passed, 2 warnings
```

The focused gate passed before and after the implementation-review patch.

## Trusted GPU Tiny Smoke

Command:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 256 \
  --time-steps 3 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 2 \
  --sinkhorn-epsilon 0.5 \
  --annealed-scaling 0.9 \
  --annealed-convergence-threshold 0.001 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode manual-reverse \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-score-tangent-materialization-phase3r-lgssm-score-only-n256-t3-s2-smoke-2026-07-09.json
```

Result artifact:

```text
docs/plans/artifacts/ledh-score-tangent-materialization-phase3r-lgssm-score-only-n256-t3-s2-smoke-2026-07-09.json
```

Key result:

| Field | Value |
| --- | --- |
| `artifact_status` | `completed` |
| `terminal_artifact` | `true` |
| `last_completed_stage` | `completed` |
| `score_status` | `blocked_score_only_diagnostic_not_admitted` |
| `score_peak_mib` | `66.6923828125` |

Interpretation: the repaired artifact procedure works on the tiny GPU
score-only path.

## Trusted GPU Full-Size Single-Seed Rerun

Command:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 10000 \
  --time-steps 50 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --annealed-scaling 0.9 \
  --annealed-convergence-threshold 0.001 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode manual-reverse \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-score-tangent-materialization-phase3r-lgssm-score-only-n10000-t50-seed81120-2026-07-09.json
```

Observed status before stop:

| Field | Value |
| --- | --- |
| Process PID | `858503` |
| Elapsed wall time at stop decision | approximately `12:55` |
| CPU use | approximately `101%` |
| Artifact status | `initialized` |
| `terminal_artifact` | `false` |
| `last_completed_stage` | `null` |
| `score_started` | `false` |
| `score_completed` | `false` |
| GPU memory at stop check | approximately `15744 MiB` reported by `nvidia-smi` |

The run was stopped with `kill 858503` after the Phase 3R stop condition was
met: it remained a nonterminal progress-only artifact stuck before
`value_completed`. The command exited with code `143`.

Progress artifact:

```text
docs/plans/artifacts/ledh-score-tangent-materialization-phase3r-lgssm-score-only-n10000-t50-seed81120-2026-07-09.json
```

Interpretation: this is not a score-memory failure and not a mathematical score
failure. The score path never started. The current blocker is full-size LGSSM
value execution/procedure at `N=10000,T=50,Sinkhorn=10` under the monolithic
score runner.

## Evidence Contract Status

| Requirement | Status |
| --- | --- |
| Atomic progress/failure/success artifacts added | Passed |
| Progress-only artifacts treated as blockers, not passes | Passed |
| Tiny GPU smoke leaves terminal success artifact | Passed |
| Full-size single-seed run leaves terminal success/failure artifact | Failed: nonterminal initialized artifact only |
| Full-size score path began | No |
| Full-size score-memory evidence | Not produced |
| Score admission | Not attempted |

## Nonclaims

This phase does not claim score admission, full leaderboard completion, HMC
readiness, posterior correctness, scientific superiority, or evidence about
full-size score correctness. It also does not claim that the Phase 3 same-points
score-memory repair failed; the full-size rerun did not reach score execution.

## Handoff

The next smallest repair is a full-size LGSSM value-stage runtime/procedure
diagnostic. It should determine why the `N=10000,T=50,Sinkhorn=10` value core
does not reach `value_completed` in the monolithic runner, and whether the
runner needs value-core checkpointing, smaller chunk policy, host-side timeout
supervision, or separate value/score shard orchestration.

Next subplan:

```text
docs/plans/bayesfilter-ledh-score-tangent-materialization-phase3s-lgssm-fullsize-value-runtime-subplan-2026-07-09.md
```
