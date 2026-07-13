# Phase 2 Repair Subplan: LGSSM Bounded Full Score Artifact

metadata_date: 2026-07-07
status: `DRAFT_REPAIR_SUBPLAN`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 2

## Phase Objective

Repair the LGSSM score execution path so it can either produce a T=50,N=10000
score artifact that validates against the Phase 1 score contract, or write a
reviewed blocker explaining why full admission remains out of reach.

## Entry Conditions Inherited From Previous Phase

- Phase 1 schema passed local checks and read-only review.
- Phase 2 LGSSM preflight passed:
  - active full-row identity is N=10000;
  - stale N=1000 raw evidence is rejected;
  - raw score fixtures normalize into the Phase 1 schema;
  - full-mode dispatch reaches total-VJP code.
- The first full raw runner attempt was interrupted after a long visible window
  with no artifact. The blocker is execution cost, not score target mismatch.

## Required Artifacts

- Repair subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-repair-subplan-2026-07-07.md`
- Blocker result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-full-run-blocker-result-2026-07-07.md`
- Repair implementation:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- Repair tests:
  `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
- Bounded full score artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-score-artifact-2026-07-07.json`
- Bounded full score log:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-bounded-score-2026-07-07.log`
- Phase 2 result or blocker result.

## Required Checks/Tests/Reviews

CPU-hidden repair checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Trusted GPU repair run:

- Run a bounded score-only T=50,N=10000 compact score computation once.
- A predeclared same-scalar directional FD check may be used as a bounded
  diagnostic only. It is not sufficient by itself to admit a 5D LGSSM score.
- Full score admission still requires either:
  - coordinate-wise same-scalar finite differences for all five parameters at
    T=50,N=10000; or
  - an exact/reference score check for all five parameters; or
  - a separately reviewed mathematical proof plus tests that make the compact
    score route exact for the stated finite estimator.
- Capture stdout/stderr to a log.
- Validate the resulting score artifact with:

```text
validate_ledh_score_artifact(..., require_admitted=True)
```

Review:

- Use Claude read-only review if available; otherwise record fresh Codex review
  fallback. Review must focus on whether the repair is still same target,
  no-tape, T=50,N=10000, and schema-admitted.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can LGSSM produce a bounded T=50,N=10000 compact no-tape score artifact for the same value scalar without weakening the correctness gate? |
| Baseline/comparator | Admitted T=50,N=10000 value artifact, Phase 2 preflight tests, coordinate-wise same-scalar finite differences, exact/reference score if available, and directional FD as diagnostic only. |
| Primary criterion | Score artifact validates with `require_admitted=True`; score route is `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`; T=50,N=10000 identity matches value artifact; score is finite; memory gate passes; and coordinate-wise or exact/reference all-parameter correctness passes. |
| Veto diagnostics | Directional FD used as sole full-admission diagnostic; old T=2 evidence admitted; stale N=1000 evidence admitted; wrong scalar; wrong parameter order; tape/ForwardAccumulator/stopped partial; nonfinite score; memory failure; no artifact produced. |
| Explanatory diagnostics | Runtime, compile time, GPU memory, directional FD absolute/relative error, device placement. |
| Not concluded | Exact Kalman score equality, HMC readiness, posterior correctness, scientific superiority, runtime ranking, or nonlinear-row validity. |
| Artifact | Score artifact JSON/Markdown, run log, Phase 2 result or blocker, review bundle. |

## Step-By-Step Plan

1. Add a bounded LGSSM score artifact writer that computes the compact score
   once for T=50,N=10000 and records full-row metadata.
2. Make correctness mode explicit:
   - `directional_fd` is diagnostic only;
   - full admission still requires coordinate-wise finite differences,
     exact/reference all-parameter score, or reviewed proof-backed tests.
3. Do not extend the Phase 1 schema to accept directional FD as an admitted
   correctness kind in this repair. If only directional FD is feasible, write a
   blocker result rather than an admitted score artifact.
4. Add tests that:
   - old T=2 artifact cannot validate as admitted Phase 2 score;
   - directional FD correctness artifacts are rejected for full admission;
   - full score artifacts still require T=50,N=10000 and memory pass.
5. Run CPU-hidden tests.
6. Run the trusted bounded GPU command.
7. Validate any candidate artifact with Phase 1 score contract.
8. If only diagnostic directional FD evidence exists, write a new blocker
   result and keep LGSSM score not admitted.
9. Write Phase 2 result or a new blocker result.
10. Review with Claude if available.

## Forbidden Claims/Actions

- Do not admit the old T=2 `ledh-phase5-lgssm-score-memory-n10000` artifact.
- Do not admit stale N=1000 evidence.
- Do not admit a score from a single directional FD check.
- Do not rerun the unbounded per-coordinate FD full command unchanged.
- Do not use tape, ForwardAccumulator, hidden autodiff, or stopped partials.
- Do not claim exact Kalman score equality, HMC readiness, posterior
  correctness, scientific superiority, runtime ranking, or nonlinear-row
  validity.

## Exact Next-Phase Handoff Conditions

Phase 3 fixed-SIR may start only if Phase 2 writes either:

- an admitted LGSSM T=50,N=10000 score result reviewed as boundary-safe; or
- a blocker result explicitly preserving LGSSM score as not admitted.

## Stop Conditions

Stop if:

- bounded score-only T=50,N=10000 still cannot produce an artifact;
- directional FD at T=50,N=10000 is too expensive or fails;
- schema extension is needed but cannot be reviewed safely;
- no-tape provenance becomes ambiguous;
- memory or device gates fail;
- review returns a material blocker that does not converge after five rounds.
