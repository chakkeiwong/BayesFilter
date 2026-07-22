# Codex Substitute Re-Review: Phase 6 Repair, Phase 7 Result, Phase 8 Subplan

Date: 2026-07-10

## Scope

Fresh local substitute review after the first Phase 6 verdict was superseded by
the sequential-seed batching repair. This is not independent Claude review.

## Findings

| Check | Finding |
| --- | --- |
| Phase 6 batching repair | The default generalized-SV diagnostic calls `_compact_value_and_score_across_seeds`, which clones one seed at a time and invokes the compact component once per seed. The compatibility alias is not called. |
| Phase 7 batching boundary | KSC-SV uses the same explicit sequential-seed compact wrapper. Its mixture/target math is unchanged. |
| Behavioral tests | Two-seed monkeypatch tests require exactly one compact component call per seed, forbid the compatibility alias, and require only the value objective for FD perturbations. |
| Focused checks | Phase 6 passed `68` tests after repair; Phase 7 passed `68` tests after repair. Both were CPU-hidden wiring checks. |
| Admission boundary | Both constructors require row-matched compact nested provenance, production precision, exact full-row particles/time/seeds, and shared trusted-memory validation. |
| Target semantics | Generalized-SV remains `source_route_prior_mean_generalized_sv`; KSC-SV remains `ksc_log_chi_square_gaussian_mixture_surrogate` with exact-native actual-SV claim false. |
| Phase 8 scope | The plan tests current cross-model invariants and does not duplicate existing validator-based leaderboard admission logic or launch GPU/full-row work. |
| Phase 8 runtime control | Verification is split into four bounded shards with a stop/split rule for any shard expected to exceed five minutes. |

No blocking issue remains.

## Limitation

This local substitute review is weaker than independent Claude convergence. It
authorizes only the CPU-hidden Phase 8 cross-model regression gate described by
the reviewed subplan.

VERDICT: AGREE
