# Codex Substitute Review: LEDH Score Wiring Repair Phase 5 Result And Phase 6 Subplan

Date: 2026-07-10

## Scope

This is a fresh local Codex substitute read-only review for the Phase 5
actual-SV result and Phase 6 generalized-SV subplan. It is not a Claude review.
The earlier bounded Claude review attempt was rejected by execution policy as
external repository data disclosure; no workaround was attempted.

## Paths Reviewed

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase5-result-phase6-subplan-review-bundle-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `bayesfilter/highdim/ledh_score_contract.py`

## Findings

| Check | Finding |
| --- | --- |
| Phase 5 route wiring | `_coordinate_fd_score_diagnostic` takes its score from `_compact_value_and_score_from_components`; its finite-difference comparator calls the value-only same-scalar objective. |
| Nested provenance | The actual-SV artifact constructor inspects the nested base route, no-autodiff declaration, same-route status, shape, and seeds before full admission. Historical manual and memory-style identifiers are rejected. |
| Shared admission boundary | Full admission delegates row-matched compact provenance, production `float32`/TF32 metadata, admitted source value, trusted memory source, and memory-budget checks to the shared score contract. |
| Target boundary | The artifact inherits `transformed_actual_sv_log_y_square` and the theta coordinate system from the admitted value artifact. It retains `claims_exact_native_actual_sv_likelihood = false`. |
| Evidence scope | The result describes CPU-hidden checks as wiring evidence only and does not claim a trusted full actual-SV score run, leaderboard completion, HMC readiness, posterior correctness, or scientific superiority. |
| Phase 6 baseline | The subplan correctly treats generalized-SV as an existing compact route needing precision and admission hardening, not as a reverse-route replacement. |
| Phase 6 controls | The subplan includes required artifacts, focused checks, explicit vetoes, target-preservation requirements, handoff conditions, and stop conditions. |

No blocking issue was found.

## Limitation

This is a local substitute review and is weaker than an independent Claude
read-only review. It authorizes only the scoped Phase 6 generalized-SV wiring
and artifact-boundary work already described by the reviewed subplan.

VERDICT: AGREE
