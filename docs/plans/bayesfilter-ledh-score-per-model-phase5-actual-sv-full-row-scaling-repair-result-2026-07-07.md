# Phase 5 Repair Result: Actual-SV Full-Row Score Scaling

metadata_date: 2026-07-07
status: `CLOSED_BLOCKED_TRANSPORT_VJP_SCALING`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-repair-full-row-score-scaling

## Phase Objective

Determine whether the actual-SV full-row score blocker was caused mainly by
coordinate finite-difference replay overhead, or by the single no-tape score
reverse pass itself.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Full actual-SV score remains blocked. |
| Primary criterion status | Not met: no validator-compatible full-row correctness/admission path exists. |
| Veto diagnostic status | Single value/score-only reverse pass at `T=20,N=1024` also showed near-budget GPU memory and impractical runtime. |
| Main uncertainty | Need a transport VJP memory/runtime redesign, checkpoint/recompute strategy, or strict exact-reference route that avoids the current total-pullback scaling. |
| Next justified action | Draft and review a dedicated transport VJP scaling repair subplan before any larger score ladder. |
| What is not concluded | No rejection of the repaired tiny mathematics; no full score admission; no HMC readiness, posterior correctness, runtime ranking, or scientific superiority claim. |

## Review

The scaling repair subplan was reviewed by direct bounded Claude read-only
review. Claude returned:

```text
VERDICT: AGREE
```

## Local Checks

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
29 passed, 2 warnings
```

## Implementation

Added a non-admission CLI mode:

```text
--diagnostic-mode value-score-only
```

This mode runs the repaired no-tape value/score route once, skips coordinate
finite differences, and writes an artifact with:

```text
score_correctness.status = not_run
score_admission_status = blocked_score_not_run
```

The artifact is intentionally not admissible.

## Diagnostic Runs

### CPU-Hidden Tiny Smoke

Command: CPU-hidden value-score-only smoke at `T=2,N=64`.

Artifact:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-value-score-only-tiny-smoke-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-value-score-only-tiny-smoke-2026-07-07.md`

Result:

```text
elapsed_seconds = 5.361246170010418
score = [-0.13676240070260542, 0.38478843496586546]
score_correctness.status = not_run
score_admission_status = blocked_score_not_run
```

### Trusted GPU No-FD Rung: T=20,N=1024

Command: trusted GPU value-score-only diagnostic at `T=20,N=1024`.

Result: manually interrupted after about 9 minutes. No JSON artifact was
written.

Runtime/GPU status during the run:

```text
PID 3625519 elapsed about 04:54 at status check, CPU active
nvidia-smi showed about 15763 MiB / 16376 MiB in use
```

The traceback showed execution inside:

```text
annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_pullback
annealed_transport_tf._filterflow_streaming_transport_from_potentials_vjp
annealed_transport_tf._filterflow_streaming_column_log_normalizer
```

Interpretation:

- Coordinate finite-difference replay is not the only blocker.
- The single no-tape score reverse pass through streaming transport VJP is
  itself too slow and memory-heavy at `T=20,N=1024`.
- Larger ladders and full `N=10000,T=1000` runs are not justified with the
  current total-pullback implementation.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can we produce a validator-compatible full-row actual-SV score evidence path without repeated full-row all-coordinate FD blowup? |
| Answer | Not yet. Removing FD replay still leaves the streaming transport VJP total-pullback as a runtime/memory blocker. |
| Primary criterion | Failed/not met. |
| Veto diagnostics | Value-score-only `T=20,N=1024` near-budget GPU memory and impractical runtime. |
| Explanatory diagnostics | Tiny no-FD smoke works; moderate no-FD rung blocks in transport VJP internals. |
| Artifact | This result and the next transport VJP scaling subplan. |

## Required Next Repair

The next phase must target the transport VJP implementation, not just the CLI
ladder:

1. Audit `_filterflow_manual_streaming_finite_transport_total_pullback`.
2. Identify whether full `transport_ad_mode=full` is mandatory for admission or
   whether a reviewed same-forward-scalar, no-stopped-partial route can use a
   more memory-stable exact pullback.
3. Prototype bounded per-block/checkpointed/recomputed transport VJP diagnostics
   at small shapes.
4. Preserve no-tape and same-target constraints.
5. Do not run larger ladders until the transport VJP scaling blocker is closed.

## Nonclaims

- Full actual-SV score is not admitted.
- Tiny repaired score correctness remains tiny diagnostic evidence only.
- Runtime/memory blocker is not evidence against the mathematics.
- No KSC, generalized-SV, raw Gaussian, or augmented-noise score claim is made.
- No HMC readiness, posterior correctness, runtime ranking, scientific
  superiority, or all-algorithm comparison is claimed.
