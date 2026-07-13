# Phase 2 Subplan: LGSSM Compact Default Cleanup

Date: 2026-07-10

## Phase Objective

Clean the LGSSM score adapter so compact score is the unambiguous default,
historical reverse/manual paths are explicitly diagnostic-only, score timing is
measured separately from value timing, and full score artifacts can satisfy the
Phase 1 `score_precision` gate.

## Entry Conditions Inherited From Phase 1

- Shared score contract rejects historical full admission.
- Full score admission requires production `float32` plus TF32 metadata.
- LGSSM `N=10000,T=50` score-only diagnostic already completed through the
  compact route with `uses_full_history_reverse_route=false`.

## Required Artifacts

- Updated LGSSM runner:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- Updated LGSSM tests:
  `tests/test_ledh_lgssm_manual_score_phase4.py`
  `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
- Phase 2 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-result-2026-07-10.md`
- Phase 3 subplan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-subplan-2026-07-10.md`

## Required Checks, Tests, Reviews

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `pytest -q tests/test_ledh_lgssm_manual_score_phase4.py tests/highdim/test_ledh_lgssm_score_phase2_contract.py tests/highdim/test_ledh_score_contract_phase1.py`
- Source search proving default `compact-sensitivity` dispatches to
  `_compact_value_and_score_from_components` and not
  `_manual_value_and_score_from_components`.
- Review Phase 2 result and Phase 3 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the LGSSM default score path unambiguously compact, with historical reverse route demoted and score timing/precision metadata ready for downstream admission? |
| Baseline/comparator | Phase 1 shared contract and prior LGSSM compact `N=10000,T=50` score-only artifact. |
| Primary criterion | Tests prove `compact-sensitivity` uses compact score; manual reverse is explicit historical diagnostic; score timing fields exist; full score artifact builder includes score precision metadata when applicable. |
| Veto diagnostics | Default path calls `_manual_value_and_score_from_components`; historical route can full-admit; missing score precision in admitted artifact; timing labels still imply score elapsed is compile time. |
| Explanatory diagnostics | Existing `N=10000,T=50` score-only memory/time artifact and value-vs-Kalman comparisons. |
| Not concluded | No exact Kalman score claim, no HMC readiness, no non-LGSSM repair, no new full admission unless score+FD is explicitly run and admitted. |

## Forbidden Claims And Actions

- Do not alter the LGSSM target scalar.
- Do not use `manual-reverse` as default.
- Do not remove historical diagnostic code unless tests and docs are updated.
- Do not claim score correctness from score-only memory diagnostics.
- Do not run long GPU jobs before local tests pass.

## Exact Next-Phase Handoff Conditions

Advance to Phase 3 only if:

- LGSSM tests pass;
- Phase 2 result records changes and nonclaims;
- Phase 3 fixed-SIR subplan exists and is reviewed.

## Stop Conditions

- LGSSM compact default cannot be preserved while satisfying shared precision
  contract.
- Tests reveal that prior score artifacts or validators require old
  `memory_style` status names for full admission.
- Review does not converge after five rounds.

## Skeptical Plan Audit

Recorded at Phase 2 precheck after Phase 1 substitute-review repair:

- Wrong baseline risk: use Phase 1 repaired contract, not the earlier
  compact-only contract that accepted wrong-row compact provenance.
- Proxy metric risk: score-only memory remains explanatory; LGSSM full score
  admission still requires same-scalar FD, trusted score memory, GPU runtime,
  and production `score_precision`.
- Hidden stale-status risk: existing raw status
  `admitted_same_target_memory_style_score` is wrong relative to the compact
  default and must be replaced or demoted before Phase 2 can pass.
- Environment mismatch risk: local Phase 2 tests intentionally run CPU-hidden
  import/smoke checks; no GPU memory or leaderboard claim may be made from
  them.
- Artifact sufficiency: the required checks answer LGSSM wiring and artifact
  schema only, not all-model score readiness.

Audit result: execution is allowed only for the scoped LGSSM cleanup and tests.
