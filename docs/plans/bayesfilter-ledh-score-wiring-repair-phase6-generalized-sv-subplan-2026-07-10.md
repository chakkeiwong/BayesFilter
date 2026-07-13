# Phase 6 Subplan: Generalized-SV Compact Precision Gate

Date: 2026-07-10

## Phase Objective

Repair the generalized-SV score adapter so its existing compact
forward-sensitivity same-scalar route carries explicit production precision
metadata and full-admission artifact hardening. The source-route prior-mean
generalized-SV target must be preserved.

## Entry Conditions Inherited From Phase 5

- Shared score contract requires row-matched compact provenance and explicit
  production precision for full admission.
- LGSSM, fixed-SIR, predator-prey, and actual-SV now reject nested
  historical/manual relabeling into compact full admission.
- Actual-SV demonstrated the transformed-target preservation pattern and
  exact-native-likelihood nonclaim boundary.

## Required Artifacts

- Updated generalized-SV runner:
  `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py`
- Updated generalized-SV tests:
  `tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py`
- Phase 6 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-result-2026-07-10.md`
- Phase 7 subplan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-subplan-2026-07-10.md`

## Required Checks, Tests, Reviews

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py`
- `pytest -q tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py tests/highdim/test_ledh_score_contract_phase1.py`
- Source search proving:
  - compact route remains the score base;
  - finite differences use value-only same-scalar objective;
  - production defaults are `float32` and TF32 enabled;
  - admitted score artifacts include `score_precision`;
  - full admission rejects tiny-shape or wrong-seed diagnostics;
  - generalized-SV target policy remains `source_route_prior_mean_generalized_sv`.
- Review Phase 6 result and Phase 7 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is generalized-SV wired so its existing compact score route is compatible with the shared production precision and full-admission gates? |
| Baseline/comparator | Current generalized-SV compact route already passes tiny same-scalar checks, but module/CLI defaults are `float64`/TF32-disabled and artifacts lack the Phase 1 precision gate. |
| Primary criterion | Tests prove compact score no-autodiff execution, same-scalar tiny FD, source-route generalized-SV target preservation, production precision requirement for full admission, and rejection of tiny-shape promotion. |
| Veto diagnostics | Default/CLI production score remains `float64` or TF32 disabled; artifact target shifts to KSC/log-chi-square surrogate; full artifact lacks production `score_precision`; full admission accepts wrong shape/seeds. |
| Explanatory diagnostics | Tiny CPU-hidden compact-vs-value/FD checks and source inspections. |
| Not concluded | No new generalized-SV `N=10000` GPU score-memory run, no leaderboard completion, no HMC/posterior/scientific claim. |

## Forbidden Claims And Actions

- Do not change the generalized-SV target scalar or target observation policy.
- Do not claim exact native actual-SV or KSC likelihood correctness from
  generalized-SV evidence.
- Do not default generalized-SV LEDH score to `float64` or TF32 disabled for
  production.
- Do not launch a full GPU run before focused local checks and review pass.

## Exact Next-Phase Handoff Conditions

Advance to Phase 7 only if:

- generalized-SV py-compile and focused tests pass;
- Phase 6 result records compact route preservation, production precision gate,
  target preservation, and nonclaims;
- Phase 7 KSC-SV subplan exists and is reviewed.

## Stop Conditions

- Compact generalized-SV score cannot satisfy same-scalar tiny checks without
  changing the source-route target scalar.
- Existing tests reveal the artifact adapter can still full-admit missing or
  non-production precision metadata.
- Review does not converge after five rounds.

## Skeptical Plan Audit

- Wrong baseline risk: generalized-SV is not a reverse-route repair phase; it
  is primarily precision/full-admission hardening.
- Proxy metric risk: tiny checks validate wiring only, not full `N=10000`
  memory or scientific correctness.
- Hidden assumption risk: source-route prior-mean generalized-SV target must
  not be replaced by KSC or actual-SV target semantics.
- Environment mismatch risk: local tests are CPU-hidden diagnostics; GPU claims
  require trusted runs in later phases.
- Artifact sufficiency: the phase answers generalized-SV wiring/precision only,
  not KSC-SV, cross-model smoke, GPU memory, or leaderboard readiness.

Audit result: execution is allowed for scoped generalized-SV precision and
artifact-boundary repair only after review of Phase 5 result and this subplan.
