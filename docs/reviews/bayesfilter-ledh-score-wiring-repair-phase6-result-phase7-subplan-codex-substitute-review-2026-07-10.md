# Codex Substitute Review: LEDH Score Wiring Repair Phase 6 Result And Phase 7 Subplan

Date: 2026-07-10

Supersession status: `SUPERSEDED_BY_BATCHING_REPAIR_REVIEW_PENDING`

After this verdict, a focused cross-model audit found that calling the compact
component once with all seeds changed the prior sequential seed schedule and
could inflate full-row memory. The implementation was repaired to use an
explicit compact sequential-seed wrapper. This original verdict is retained as
audit history but does not authorize further advancement until focused tests
and a substitute re-review pass.

## Scope

This is a fresh local Codex substitute read-only review for the Phase 6
generalized-SV result and Phase 7 KSC-SV subplan. It is not a Claude review.
Claude calls remain policy-blocked as external repository data disclosure, so
no retry or workaround was attempted.

## Paths Reviewed

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase6-result-phase7-subplan-review-bundle-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-subplan-2026-07-10.md`
- `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py`
- `tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py`
- `bayesfilter/highdim/ledh_score_contract.py`
- `bayesfilter/linear/kalman_qr_tf.py`

## Findings

| Check | Finding |
| --- | --- |
| Direct score route | At review time, the generalized-SV coordinate diagnostic called `_compact_value_and_score_from_components` once with all seeds. Its finite-difference perturbations called the value-only objective. This was later superseded because the all-seed component call changed the sequential seed schedule. |
| Precision boundary | Module and CLI defaults are `float32`/TF32-enabled. Diagnostic artifacts carry explicit score precision, and the shared full-admission validator rejects non-production precision. |
| Provenance and shape | The constructor inspects nested compact route, no-autodiff and same-route declarations, parameter order, particles, time steps, and seeds before full admission. |
| Target boundary | Artifacts inherit `source_route_prior_mean_generalized_sv` and the theta coordinate system from the admitted value artifact. Tests reject target-policy substitution. |
| Evidence scope | The result confines `68 passed` to CPU-hidden wiring evidence and explicitly withholds full-row GPU memory, score admission, HMC, posterior, leaderboard, runtime-ranking, and scientific claims. |
| Import-chain repair | The prerequisite change removes one duplicated `maximum_iterations` keyword from a committed `tf.while_loop` call. `py_compile`, import, and `tests/test_linear_kalman_qr_tf.py` pass. The result correctly separates this engineering repair from generalized-SV evidence. |
| Phase 7 target | The subplan preserves the admitted KSC Gaussian-mixture surrogate row, parameter contract, and `claims_exact_native_actual_sv_likelihood = false`; it forbids substitution with actual-SV or generalized-SV semantics. |
| Phase 7 completeness | Required artifacts, primary criterion, vetoes, explanatory-only diagnostics, forbidden actions, handoff conditions, and stop conditions are present. |

No blocking issue was found.

## Limitation

This is a local substitute review and is weaker than independent Claude
convergence. It authorizes only the scoped Phase 7 KSC-SV wiring and artifact
boundary work in the reviewed subplan.

VERDICT: AGREE
