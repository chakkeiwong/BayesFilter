# Phase 5 Subplan: Actual-SV Compact Default And Precision Gate

Date: 2026-07-10

## Phase Objective

Repair the actual-SV score adapter so the default/current score computation and
full-admission artifact path use the compact forward-sensitivity same-scalar
route for the transformed actual-SV finite-`N` LEDH log-likelihood scalar. The
historical reverse/manual route remains diagnostic-only, and full admission
requires `float32` TensorFlow with TF32 enabled.

## Entry Conditions Inherited From Phase 4

- Shared score contract requires row-matched compact provenance and explicit
  production precision for full admission.
- LGSSM, fixed-SIR, and predator-prey now reject nested historical/manual
  relabeling into compact full admission.
- Predator-prey demonstrated the desired repair pattern: compact score base,
  value-only finite-difference comparator, explicit score precision, and
  historical route demotion.

## Required Artifacts

- Updated actual-SV runner:
  `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- Updated actual-SV tests:
  `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- Phase 5 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-result-2026-07-10.md`
- Phase 6 subplan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md`

## Required Checks, Tests, Reviews

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `pytest -q tests/highdim/test_ledh_actual_sv_score_phase5_contract.py tests/highdim/test_ledh_score_contract_phase1.py`
- Source search proving:
  - `_coordinate_fd_score_diagnostic` uses `_compact_value_and_score_from_components` for the score base;
  - finite differences use a value-only same-scalar objective, not the score route;
  - `_manual_value_and_score_across_seeds` remains historical/diagnostic-only;
  - admitted score artifacts include `score_precision`;
  - outer compact/admitted metadata cannot relabel nested historical/manual provenance;
  - actual-SV target policy remains `transformed_actual_sv_log_y_square` and does not become KSC/native exact likelihood.
- Review Phase 5 result and Phase 6 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is actual-SV wired so compact forward-sensitivity is the only full-admissible score path for the transformed actual-SV same scalar? |
| Baseline/comparator | Current actual-SV tests and code still route coordinate FD through memory-style reverse/manual score; Phase 4 predator-prey and Phase 3 fixed-SIR repair patterns. |
| Primary criterion | Tests prove compact score no-autodiff execution, same-scalar tiny FD, transformed actual-SV target preservation, production precision requirement for full admission, and rejection of nested historical/manual relabeling. |
| Veto diagnostics | Any full-admission path uses `ACTUAL_SV_MANUAL_SCORE_ROUTE_ID` or `ACTUAL_SV_MEMORY_STYLE_SCORE_ROUTE_ID`; default/CLI production score remains `float64` or TF32 disabled; artifact target shifts to KSC or exact native actual-SV likelihood; FD comparator calls the score route instead of value-only scalar. |
| Explanatory diagnostics | Tiny CPU-hidden compact-vs-value/FD checks and source inspections. |
| Not concluded | No new actual-SV `N=10000,T=1000` GPU score-memory run, no exact native likelihood claim, no leaderboard completion, no HMC/posterior/scientific claim. |

## Forbidden Claims And Actions

- Do not relabel historical reverse/manual actual-SV evidence as compact.
- Do not change the actual-SV target scalar or target observation policy.
- Do not claim exact native actual-SV likelihood correctness from transformed
  actual-SV LEDH evidence.
- Do not default actual-SV LEDH score to `float64` or TF32 disabled for
  production.
- Do not launch a full `N=10000,T=1000` GPU run before focused local checks and
  review pass.

## Exact Next-Phase Handoff Conditions

Advance to Phase 6 only if:

- actual-SV py-compile and focused tests pass;
- Phase 5 result records compact default, historical demotion, transformed
  target preservation, precision gate, and nonclaims;
- Phase 6 generalized-SV subplan exists and is reviewed.

## Stop Conditions

- Compact actual-SV score cannot satisfy same-scalar tiny checks without
  changing the transformed actual-SV target scalar.
- Existing tests reveal the artifact adapter can still mask nested
  historical/manual provenance.
- Review does not converge after five rounds.

## Skeptical Plan Audit

- Wrong baseline risk: prior actual-SV tests assert reverse/manual default; that
  is now historical/wrong relative to the owner directive.
- Proxy metric risk: tiny compact checks validate wiring only, not full
  `N=10000,T=1000` memory or scientific correctness.
- Hidden assumption risk: transformed actual-SV and KSC/native likelihoods must
  remain separate; target substitution is a veto.
- Environment mismatch risk: local tests are CPU-hidden diagnostics; GPU claims
  require trusted runs in later phases.
- Artifact sufficiency: the phase answers actual-SV wiring only, not
  generalized-SV, KSC-SV, cross-model smoke, GPU memory, or leaderboard
  readiness.

Audit result: execution is allowed for scoped actual-SV wiring repair only
after review of Phase 4 result and this subplan.
