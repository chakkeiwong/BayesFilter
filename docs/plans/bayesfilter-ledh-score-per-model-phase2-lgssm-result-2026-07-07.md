# Phase 2 Result: LGSSM Score

metadata_date: 2026-07-07
status: `CLOSED_BLOCKED_FULL_SCORE_NOT_ADMITTED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 2

## Phase Objective

Admit, or explicitly block, the LEDH score for:

```text
benchmark_lgssm_exact_oracle_m3_T50
```

The target score remains the no-tape total derivative of the realized
finite-`N` LEDH estimator:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

## Decision Table

| Field | Result |
| --- | --- |
| Decision | LGSSM score is blocked, not admitted, for the current runbook. |
| Primary criterion status | Not met: no T=50,N=10000 score artifact passed `validate_ledh_score_artifact(..., require_admitted=True)`. |
| Veto diagnostic status | No target/schema/no-tape veto found in preflight; execution-plan blocker found because the full raw runner performs expensive per-coordinate full finite differences and produced no artifact in a bounded visible window. |
| Main uncertainty | Need a bounded all-parameter correctness route for T=50,N=10000 LGSSM score: coordinate-wise FD, exact/reference all-parameter score, or proof-backed tests. |
| Next justified action | Proceed to Phase 3 fixed-SIR with LGSSM recorded as not admitted, unless a later dedicated LGSSM all-parameter correctness phase is authorized. |
| What is not concluded | No LGSSM score admission; no evidence against the mathematics of the compact score; no exact Kalman score equality; no HMC/posterior/scientific/runtime claim. |

## What Passed

Preflight/repair checks passed:

- active LGSSM full-row identity now uses `N=10000`;
- stale `N=1000` raw results are rejected;
- old `T=2` score-memory evidence is rejected for Phase 2 full admission;
- raw full-score fixtures can normalize into the Phase 1 schema;
- CPU/runtime misses remain tiny diagnostics, not admitted scores;
- full-mode manual streaming dispatch reaches total-VJP code despite the
  legacy CLI constant name;
- local no-tape tiny score tests still pass.

Local check result:

```text
32 passed, 2 warnings
```

## What Failed Or Blocked

The trusted full command was launched after preflight:

```text
python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --device-scope visible \
  --expect-device-kind gpu \
  --device /GPU:0 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --num-particles 10000 \
  --time-steps 50 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --transport-ad-mode full \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float64 \
  --tf32-mode disabled
```

It compiled and entered the compact total-JVP route, but produced no raw JSON
artifact in the bounded visible window. It was interrupted to preserve the
runbook's visible execution control.

Log:

```text
docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-n10000-run-2026-07-07.log
```

## Review Record

Claude reviewed the repair proposal.

First review: `VERDICT=REVISE`

- A single directional finite-difference check is too weak to admit a 5D score.
- Old T=2 evidence needed explicit rejection.

Repair:

- Phase 2 repair subplan now states directional FD is diagnostic only.
- Full admission still requires coordinate-wise same-scalar FD for all five
  parameters, exact/reference all-parameter score, or proof-backed reviewed
  tests.
- Added tests rejecting stale N=1000 and old T=2 evidence.

Second focused review: `VERDICT=AGREE`

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can LGSSM produce an admitted no-tape total derivative of the same finite-`N` LEDH scalar as the admitted value artifact? |
| Answer | Not within this phase. Preflight and route guards pass, but all-parameter T=50,N=10000 correctness evidence was not produced. |
| Baseline/comparator | Admitted LGSSM value artifact, compact no-autodiff route, tiny/local tests, and attempted full GPU run. |
| Primary criterion | Failed/not met because no admitted score artifact exists. |
| Veto diagnostics | No tape/target/schema veto; execution-plan blocker and insufficient correctness evidence veto admission. |
| Explanatory diagnostics | Old T=2 score-memory evidence remains historical diagnostic only. |
| Not concluded | No HMC readiness, posterior correctness, exact Kalman score equality, scientific superiority, runtime ranking, or nonlinear-row validity. |

## Phase 3 Handoff

Phase 3 fixed-SIR may begin with this inherited condition:

```text
LGSSM score is blocked/not admitted in the current score integration ledger.
```

Phase 3 must not use the LGSSM old T=2 artifact as full-row evidence, and must
not weaken the all-parameter correctness requirement merely because fixed SIR
has an older directional FD memory artifact.

## Nonclaims

- LGSSM score is not admitted.
- The compact score mathematics is not rejected.
- Old T=2 memory evidence is not full T=50 score evidence.
- No HMC readiness, posterior correctness, exact Kalman score equality,
  scientific superiority, runtime ranking, or all-algorithm comparison is
  claimed.
