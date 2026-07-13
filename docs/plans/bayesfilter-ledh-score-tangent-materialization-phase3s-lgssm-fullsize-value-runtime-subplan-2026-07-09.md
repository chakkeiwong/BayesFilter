# Phase 3S Subplan: LGSSM Full-Size Value Runtime Diagnostic

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Diagnose and repair the full-size LGSSM value-stage runtime/procedure blocker
that prevents the monolithic `N=10000,T=50,Sinkhorn=10` runner from reaching
`value_completed` before score execution starts.

This phase must not change the LEDH target scalar, parameter order, seeds,
transport policy, Sinkhorn settings, score route, score admission criteria, or
the score memory budget.

## Entry Conditions Inherited From Previous Phase

- Phase 3 same-points finite-Sinkhorn VJP repair passed focused CPU tests and
  fresh Codex review.
- Trusted GPU `N=1000,T=10,Sinkhorn=10` score-only emitted under budget.
- Phase 3R artifact procedure repair passed focused CPU tests, fresh Codex
  review, and tiny trusted GPU smoke.
- Full-size single-seed `N=10000,T=50,Sinkhorn=10` rerun left a nonterminal
  `initialized` progress artifact for PID `858503`; it did not reach
  `value_completed`, and `score_started=false`.

## Required Artifacts

- Trace/result note:
  `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase3s-lgssm-fullsize-value-runtime-result-2026-07-09.md`
- Any implementation diff, if needed, expected in:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
  or lower-level value-core progress instrumentation.
- Diagnostic artifacts under:
  `docs/plans/artifacts/`

## Required Checks, Tests, And Reviews

Pre-implementation trace:

1. Identify which part of value execution is slow or stuck:
   - first XLA compile;
   - first value call;
   - warm repeat;
   - TensorFlow host synchronization;
   - artifact write after value;
   - GPU allocator/device cleanup.
2. Add stage timing if current progress records are too coarse.
3. Verify whether a value-only full-size run without score mode behaves the
   same way.
4. Test a ladder that isolates scale:
   - `N=1000,T=50,Sinkhorn=10`;
   - `N=5000,T=50,Sinkhorn=10`;
   - `N=10000,T=10,Sinkhorn=10`;
   - `N=10000,T=50,Sinkhorn=2`;
   subject to stopping on first nonterminal artifact or over-budget memory.

Review:

- Use fresh Codex read-only review before any implementation or long GPU rung.
- Claude is not retried unless a new human-approved and approval-reviewer-safe
  path is available.

CPU-hidden checks if code changes:

```bash
python -m py_compile docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Trusted GPU checks:

- Start with value-only diagnostics (`--score-mode none`) so the blocker is not
  confused with score execution.
- Use Phase 3R artifact procedure fields to classify each rung.
- Do not run score-mode full-size again until a value-only full-size rung reaches
  `value_completed` or `completed` with a terminal artifact.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Where does full-size LGSSM `N=10000,T=50,Sinkhorn=10` value execution stall, and what is the smallest procedural or implementation repair? |
| Baseline/comparator | Phase 3R full-size score-only run stuck at `initialized`, `score_started=false`, no `value_completed`. |
| Primary criterion | A reviewed diagnostic identifies the active stage and either produces a terminal value-only full-size artifact or a blocker with exact stage/timing/memory evidence. |
| Correctness criterion | No target, parameter, transport, score-route, or admission criteria drift; focused CPU tests pass after any code change. |
| Veto diagnostics | Exact Kalman substitution; score admission claim; score execution before value-only blocker is resolved; no artifact; progress-only artifact after the chosen stop threshold; GPU memory over reviewed budget. |
| Explanatory diagnostics | Per-stage timing, XLA compile timing, value-call timing, TensorFlow memory info, `nvidia-smi` memory/utilization, chunk sizes, scale ladder outcome. |
| Not concluded | Score admission, full leaderboard completion, score correctness, HMC readiness, posterior correctness, or scientific superiority. |

## Forbidden Claims And Actions

- Do not claim score readiness from value-only diagnostics.
- Do not rerun full-size score-mode until full-size value-only behavior is
  observable and terminal.
- Do not change full-row settings after seeing results without writing a new
  subplan.
- Do not replace LEDH finite estimator value with exact Kalman value except as
  an already-labeled comparator.
- Do not classify a pre-score value-stage blocker as a score-memory failure.

## Exact Next-Phase Handoff Conditions

If value-only full-size reaches a terminal completed artifact, hand off back to
score-only full-size with the artifact procedure. If a smaller value ladder rung
fails, hand off to the smallest identified implementation repair. If all value
rungs pass but score still stalls later, hand off to a score-stage-specific
subplan with the new artifact evidence.

## Stop Conditions

Stop if review finds an unpatched material flaw, focused tests fail after code
changes, any trusted GPU rung emits no artifact, any trusted GPU rung remains a
nonterminal progress artifact past the reviewed threshold, GPU memory exceeds
budget, or continuing would require package installation, network/data fetches,
credentials, destructive git actions, or changing criteria after seeing
results.

## Skeptical Audit Before Execution

- Wrong baseline checked: this phase compares against the Phase 3R
  `initialized` progress-only blocker, not exact Kalman likelihood.
- Proxy metric checked: value-only completion is required only to unblock score
  diagnostics; it is not score admission.
- Hidden assumption checked: the full-size blocker currently occurs before
  score starts.
- Environment checked: GPU rungs require trusted/escalated execution.
- Artifact sufficiency checked: every rung must leave terminal or classified
  progress evidence.

Audit status: `READY_FOR_REVIEW_BEFORE_EXECUTION`.
