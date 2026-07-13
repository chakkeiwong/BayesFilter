# Phase 3 Subplan: Fixed-SIR Compact Default And Legacy Normalizer Demotion

Date: 2026-07-10

## Phase Objective

Repair the fixed-SIR score adapter so the default/current score computation is
the compact forward-sensitivity same-scalar route, any historical memory/manual
result normalizer is diagnostic-only and cannot full-admit, and full score
artifacts carry production `float32`/TF32 `score_precision` metadata.

## Entry Conditions Inherited From Phase 2

- Shared score contract requires row-matched compact provenance and explicit
  production precision for full admission.
- LGSSM compact default cleanup passed focused tests.
- Stale `memory_style` admission labels are classified as historical/wrong
  relative to current compact score admission.

## Required Artifacts

- Updated fixed-SIR runner:
  `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- Updated fixed-SIR tests:
  `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
- Phase 3 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-result-2026-07-10.md`
- Phase 4 subplan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-subplan-2026-07-10.md`

## Required Checks, Tests, Reviews

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- `pytest -q tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py tests/highdim/test_ledh_score_contract_phase1.py`
- Source search proving:
  - `_fixed_sir_compact_score_artifact_from_diagnostic` is the full-admission
    path;
  - `_fixed_sir_score_artifact_from_memory_result` is historical/diagnostic and
    cannot produce full admission;
  - admitted artifacts include `score_precision`.
- Adversarial mismatch fixture proving an outer compact/admitted status cannot
  relabel nested historical/manual diagnostic provenance as compact full
  admission.
- Review Phase 3 result and Phase 4 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is fixed-SIR wired so compact forward-sensitivity is the only full-admissible score path, with old memory/manual result normalization demoted to diagnostic-only? |
| Baseline/comparator | Existing fixed-SIR compact helper/tests and the historical fixed-SIR score memory artifact. |
| Primary criterion | Tests prove compact score runs no-autodiff, passes same-scalar FD on tiny diagnostics, full admission requires N=10000 shape/memory plus production precision, historical memory-result normalization cannot full-admit, and nested historical/manual provenance cannot be relabeled as compact. |
| Veto diagnostics | Any full-admission path uses `FIXED_SIR_MANUAL_SCORE_ROUTE_ID` or `FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID`; any full artifact lacks production precision; any legacy memory artifact validates as full; default helper uses reverse records/transport VJP. |
| Explanatory diagnostics | Tiny CPU-hidden compact-vs-FD tests and historical memory artifact normalization as tiny diagnostic evidence. |
| Not concluded | No new fixed-SIR N=10000 GPU score run, no Zhao-Cui source-faithfulness claim, no nonlinear exact likelihood claim, no leaderboard rebuild. |

## Forbidden Claims And Actions

- Do not relabel historical memory/manual VJP evidence as the current compact
  score computation.
- Do not claim fixed-SIR full score admission from the old score-memory JSON.
- Do not default fixed-SIR score to `float64`; production full admission must
  use `float32` with TF32 enabled.
- Do not alter the fixed-SIR target scalar or row id.
- Do not run long GPU jobs before focused local checks pass.

## Exact Next-Phase Handoff Conditions

Advance to Phase 4 only if:

- fixed-SIR py-compile and focused tests pass;
- Phase 3 result records the legacy normalizer demotion and nonclaims;
- Phase 4 predator-prey subplan exists and is reviewed.

## Stop Conditions

- Fixed-SIR compact diagnostic cannot satisfy the repaired shared score
  contract without changing the target scalar.
- Existing tests require full admission from old memory/manual artifacts.
- Review does not converge after five rounds.

## Skeptical Plan Audit

- Wrong baseline risk: the historical score-memory artifact is not a promotion
  baseline; it is diagnostic evidence only.
- Proxy metric risk: tiny FD checks can validate wiring, but cannot admit a
  full N=10000 score.
- Hidden assumption risk: the old manual route and compact route may agree on
  tiny cases, but agreement does not make the old route the default.
- Environment mismatch risk: local tests are CPU-hidden diagnostics; GPU
  production claims require trusted runs in later phases.
- Artifact sufficiency: the phase answers fixed-SIR wiring only, not
  predator-prey/actual-SV/default leaderboard readiness.

Audit result: execution is allowed for scoped fixed-SIR wiring repair only
after review of Phase 2 result and this subplan.
