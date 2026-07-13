# Phase 1 Subplan: Shared Score Contract And Precision Gate

Date: 2026-07-10

## Phase Objective

Strengthen shared score admission and default-policy tests so every model phase
has one common gate: compact no-time-history score provenance is required for
full admission, historical memory-style routes are diagnostic-only, and
production LEDH score defaults are `float32` with TF32 enabled.

## Entry Conditions Inherited From Phase 0

- Master program and visible runbook exist.
- Phase 0 inventory identifies model-specific wiring gaps.
- No code repair has been claimed yet.
- Claude or Codex substitute launch review has agreed that the program
  structure is feasible.

## Required Artifacts

- Updated shared contract tests:
  `tests/highdim/test_ledh_score_contract_phase1.py`
- Updated artifact emitter tests if needed:
  `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`
- Phase 1 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase1-shared-contract-result-2026-07-10.md`
- Phase 2 subplan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-subplan-2026-07-10.md`
- Claude review bundle for Phase 1 result and Phase 2 subplan.

## Required Checks, Tests, Reviews

- `python -m py_compile bayesfilter/highdim/ledh_score_contract.py bayesfilter/highdim/ledh_score_artifact.py`
- `pytest -q tests/highdim/test_ledh_score_contract_phase1.py tests/highdim/test_ledh_score_artifact_emitter_phase1.py`
- Source search verifying historical routes remain listed as historical only:
  `rg -n "manual_total_vjp|memory_style|compact_forward_sensitivity" bayesfilter/highdim tests/highdim`
- Claude read-only review of the Phase 1 result and Phase 2 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the shared score contract prevent historical route full admission and provide reusable gates for compact route and precision defaults? |
| Baseline/comparator | Phase 0 inventory and existing `ledh_score_contract.py` behavior. |
| Primary criterion | Tests prove historical `manual_total_vjp*` and `memory_style*` routes cannot full-admit, compact route constants are the only full-admissible no-tape provenance, and default precision expectations are testable per model. |
| Veto diagnostics | Any historical route can full-admit; precision defaults are not tested; contract silently accepts stopped/partial/autodiff route tokens; tests rely on only one model. |
| Explanatory diagnostics | Route constant search and existing per-model contract tests. |
| Not concluded | No model default path repair, no GPU score memory pass, no leaderboard admission. |

## Forbidden Claims And Actions

- Do not claim any model runner is repaired in Phase 1.
- Do not run full GPU score-memory tests.
- Do not change per-model numerical algorithms except if a shared contract
  import requires a trivial naming update.
- Do not admit exact-reference correctness as full score admission unless a
  reviewed same-realized-scalar contract exists.

## Exact Next-Phase Handoff Conditions

Advance to Phase 2 only if:

- shared tests pass;
- Phase 1 result records all changes and nonclaims;
- Phase 2 LGSSM subplan exists;
- review agrees or a substitute review is documented.

## Stop Conditions

- Shared contract cannot express compact-vs-historical distinction without
  changing default policy.
- Tests reveal existing admitted artifacts depend on historical full admission
  and no reviewed migration path is available.
- Claude/Codex review does not converge after five rounds.

## Skeptical Plan Audit

Pending at Phase 1 precheck.
