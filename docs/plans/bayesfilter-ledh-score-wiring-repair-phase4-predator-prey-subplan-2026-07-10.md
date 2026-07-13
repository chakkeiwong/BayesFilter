# Phase 4 Subplan: Predator-Prey Compact Default And Precision Gate

Date: 2026-07-10

## Phase Objective

Repair the predator-prey score adapter so the default/current score computation
and full-admission artifact path use the compact forward-sensitivity
same-scalar route, the historical reverse/manual memory-style route is
diagnostic-only, and production full admission requires `float32` TensorFlow
with TF32 enabled.

## Entry Conditions Inherited From Phase 3

- Shared score contract requires row-matched compact provenance and explicit
  production precision for full admission.
- LGSSM and fixed-SIR now reject nested historical/manual relabeling into
  compact full admission.
- Historical `memory_style` or `manual_total_vjp` evidence is classified as
  diagnostic-only unless it is physically replaced by the compact route.

## Required Artifacts

- Updated predator-prey runner:
  `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- Updated predator-prey tests:
  `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
- Phase 4 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-result-2026-07-10.md`
- Phase 5 subplan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-subplan-2026-07-10.md`

## Required Checks, Tests, Reviews

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `pytest -q tests/highdim/test_ledh_predator_prey_score_phase4_contract.py tests/highdim/test_ledh_score_contract_phase1.py`
- Source search proving:
  - `_compact_value_and_score_from_components` is the current/admissible score
    path;
  - `_manual_value_and_score_across_seeds` or equivalent reverse/manual path is
    explicitly historical/diagnostic-only;
  - admitted score artifacts include `score_precision`;
  - outer compact/admitted metadata cannot relabel nested historical/manual
    provenance.
- Review Phase 4 result and Phase 5 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is predator-prey wired so compact forward-sensitivity is the only full-admissible score path, with reverse/manual route demoted and precision enforced? |
| Baseline/comparator | Existing predator-prey tests that still identify reverse VJP as default, plus compact helper already present in the module. |
| Primary criterion | Tests prove compact score runs no-autodiff, value/score scalar identity is preserved on tiny diagnostics, full admission requires production precision, and historical reverse/manual provenance cannot full-admit or be relabeled as compact. |
| Veto diagnostics | Any full-admission path uses `PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID` or `PREDATOR_PREY_MEMORY_STYLE_SCORE_ROUTE_ID`; any full artifact lacks production precision; CLI/default production score remains `float64` or TF32 disabled; default-source tests assert reverse VJP as the default. |
| Explanatory diagnostics | Tiny CPU-hidden compact-vs-value/FD checks and source inspections. |
| Not concluded | No new predator-prey N=10000 GPU score run, no exact nonlinear likelihood claim, no all-model leaderboard readiness. |

## Forbidden Claims And Actions

- Do not relabel historical reverse/manual evidence as compact.
- Do not claim full score admission from tiny diagnostics.
- Do not default predator-prey LEDH score to `float64` or TF32 disabled for
  production.
- Do not alter the predator-prey target scalar, row id, or value artifact.
- Do not run long GPU jobs before focused local checks pass.

## Exact Next-Phase Handoff Conditions

Advance to Phase 5 only if:

- predator-prey py-compile and focused tests pass;
- Phase 4 result records compact default, historical demotion, precision gate,
  and nonclaims;
- Phase 5 actual-SV subplan exists and is reviewed.

## Stop Conditions

- Compact predator-prey score cannot satisfy same-scalar tiny checks without
  changing the target scalar.
- Existing tests reveal the artifact adapter can still mask nested
  historical/manual provenance.
- Review does not converge after five rounds.

## Skeptical Plan Audit

- Wrong baseline risk: prior tests may encode reverse VJP as default; that is
  now historical/wrong relative to the owner directive.
- Proxy metric risk: tiny compact checks validate wiring only, not full
  N=10000 memory or scientific correctness.
- Hidden assumption risk: matching value route on tiny examples does not admit
  full score without production precision and memory evidence.
- Environment mismatch risk: local tests are CPU-hidden diagnostics; GPU claims
  require trusted runs in later phases.
- Artifact sufficiency: the phase answers predator-prey wiring only, not
  actual-SV/generalized-SV/KSC-SV readiness.

Audit result: execution is allowed for scoped predator-prey wiring repair only
after review of Phase 3 result and this subplan.
