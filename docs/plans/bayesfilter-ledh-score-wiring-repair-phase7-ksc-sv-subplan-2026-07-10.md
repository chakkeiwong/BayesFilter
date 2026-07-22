# Phase 7 Subplan: KSC-SV Compact Precision Gate

Date: 2026-07-10

## Phase Objective

Repair the KSC-SV score adapter so its existing compact forward-sensitivity
same-scalar route carries explicit production precision metadata and hardened
full-admission provenance, shape, and seed checks. Preserve the admitted KSC
Gaussian-mixture surrogate target and its exact-native-actual-SV nonclaim.

## Entry Conditions Inherited From Phase 6

- Shared score admission requires row-matched compact provenance, admitted
  source-value semantics, trusted memory diagnostics, and production
  `float32`/TF32 precision.
- Generalized-SV now calls an explicitly named compact sequential-seed wrapper,
  uses a value-only FD objective, and rejects forged nested route or full-row
  metadata.
- KSC-SV and generalized-SV are distinct admitted rows and must not exchange
  target policies or parameter contracts.

## Required Artifacts

- Updated KSC-SV runner:
  `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py`
- Updated KSC-SV tests:
  `tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py`
- Phase 7 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-result-2026-07-10.md`
- Phase 8 subplan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase8-cross-model-tests-subplan-2026-07-10.md`

## Required Checks, Tests, Reviews

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py`
- `pytest -q tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py tests/highdim/test_ledh_score_contract_phase1.py`
- Source and adversarial checks proving:
  - the default diagnostic score base calls the compact across-seed wrapper,
    which invokes the component helper sequentially per seed;
  - finite differences use a value-only same-scalar objective;
  - production defaults are `float32` and TF32 enabled;
  - score artifacts include explicit `score_precision`;
  - full admission rejects nested non-compact provenance and mismatched
    particles, time steps, or seeds;
  - target policy remains
    `ksc_log_chi_square_gaussian_mixture_surrogate`;
  - `claims_exact_native_actual_sv_likelihood` remains false and an overclaim
    is rejected.
- Review the Phase 7 result and Phase 8 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is KSC-SV wired so its existing compact score route satisfies production precision and full-admission boundaries without being relabeled as native actual-SV likelihood evidence? |
| Baseline/comparator | Current KSC-SV compact implementation and admitted KSC forward-scalar artifact. The score module/CLI defaults are `float64`/TF32-disabled and its artifact constructor lacks Phase 1 precision and nested full-row checks. |
| Primary criterion | Focused tests prove compact sequential-seed score use, value-only same-scalar FD, no-autodiff execution, KSC target preservation, production precision enforcement, and rejection of forged full-row metadata. |
| Veto diagnostics | Default remains `float64` or TF32 disabled; score base reaches a compatibility/manual wrapper; FD perturbations reach a score route; artifact target shifts to transformed/native actual-SV or generalized-SV; exact-native likelihood claim becomes true; full admission accepts non-production precision or wrong nested route/shape/seeds. |
| Explanatory diagnostics | Tiny CPU-hidden compact/value and finite-difference checks. |
| Not concluded | No new KSC-SV `N=10000,T=1000` GPU score-memory run, no native actual-SV correctness, no leaderboard completion, no HMC/posterior/scientific claim. |

## Forbidden Claims And Actions

- Do not change the admitted KSC-SV target scalar, target observation policy,
  theta coordinate system, or parameter order.
- Do not represent KSC Gaussian-mixture surrogate evidence as exact native
  actual-SV likelihood evidence.
- Do not use generalized-SV or transformed actual-SV target semantics as a
  replacement for the KSC row.
- Do not default production KSC-SV score execution to `float64` or TF32
  disabled.
- Do not launch a full GPU run before focused local checks and review pass.

## Exact Next-Phase Handoff Conditions

Advance to Phase 8 only if:

- KSC-SV py-compile and focused tests pass;
- the Phase 7 result records compact route use, production precision,
  KSC-target preservation, exact-native nonclaim, and full-admission guards;
- the Phase 8 cross-model test subplan exists and is reviewed.

## Stop Conditions

- Compact KSC-SV score cannot satisfy tiny same-scalar checks without changing
  the admitted KSC surrogate target.
- Full artifact construction can still accept missing/non-production
  precision, non-compact nested provenance, or wrong full-row metadata.
- Exact-native actual-SV overclaim cannot be rejected.
- Review does not converge after five rounds.

## Skeptical Plan Audit

- Wrong baseline risk: KSC-SV already has compact computation; this phase is
  explicit compact-route clarity plus precision/admission hardening, not a new
  score derivation or a change from sequential to batched seed execution.
- Proxy risk: tiny CPU-hidden checks establish wiring only, not full-row GPU
  memory, posterior correctness, or native likelihood validity.
- Hidden assumption risk: KSC is an explicitly admitted Gaussian-mixture
  surrogate row, not the transformed actual-SV or generalized-SV row.
- Environment mismatch risk: GPU devices will be intentionally hidden for
  focused local tests; no GPU conclusion may follow.
- Artifact sufficiency: the phase answers KSC-SV wiring and artifact boundaries
  only; cross-model consistency, full GPU memory, and leaderboard readiness
  remain later gates.

Audit result: execution is allowed only after review of the Phase 6 result and
this subplan.
