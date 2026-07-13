# LEDH Score Wiring Repair Master Program

Date: 2026-07-10

## Status

`DRAFT_REVIEW_REQUIRED`

## Objective

Repair LEDH score wiring across all current same-target model score adapters so
the default score computation physically uses the compact forward-sensitivity
same-scalar route, with historical full-history or memory-style routes demoted
to explicit diagnostic-only paths. Then test the wiring and rebuild the LEDH
leaderboard without admitting historical score routes.

## Binding Invariants

- The target scalar is the finite-`N` observed-data LEDH log likelihood
  estimator recorded by each admitted forward-scalar artifact.
- Default LEDH score means compact forward-sensitivity/no-time-history score,
  not a relabeled historical reverse route.
- Historical `manual_total_vjp*` and `memory_style*` routes may remain only as
  explicit diagnostic/historical paths and must not be full admitted.
- LEDH production score execution uses `float32` TensorFlow tensors with TF32
  enabled.
- Same-scalar finite-difference checks must compare against the value-only
  scalar route, not recompute the score route.
- Claude is read-only reviewer only. Codex is supervisor and executor.

## Model Scope

| Model | File | Current classification | Required repair |
| --- | --- | --- | --- |
| LGSSM | `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py` | Compact default repaired; naming/timing residue remains. | Preserve compact default, clean misleading route/status labels, add score timing instrumentation. |
| fixed-SIR | `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py` | Compact helper exists; historical memory-result normalizer remains. | Ensure default/current score path is compact; mark legacy normalizer import-only diagnostic. |
| predator-prey | `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py` | Default diagnostic still uses memory-style reverse route. | Switch default score/FD path to compact score; keep reverse route explicit diagnostic-only. |
| actual-SV | `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py` | Default diagnostic still uses memory-style reverse route. | Switch default score/FD path to compact score; keep reverse route explicit diagnostic-only. |
| generalized-SV | `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py` | Compact route wired; precision defaults wrong. | Preserve compact route and set default `float32` plus TF32 enabled. |
| KSC-SV | `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py` | Compact route wired; precision defaults wrong. | Preserve compact route and set default `float32` plus TF32 enabled. |

## Phase Index

| Phase | Name | Subplan | Required result |
| --- | --- | --- | --- |
| 0 | Launch inventory and governance freeze | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase0-launch-inventory-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase0-launch-inventory-result-2026-07-10.md` |
| 1 | Shared score contract and precision gate | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase1-shared-contract-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase1-shared-contract-result-2026-07-10.md` |
| 2 | LGSSM compact default cleanup | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-result-2026-07-10.md` |
| 3 | fixed-SIR compact default | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-result-2026-07-10.md` |
| 4 | Predator-prey compact default | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-result-2026-07-10.md` |
| 5 | Actual-SV compact default | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-result-2026-07-10.md` |
| 6 | Generalized-SV compact precision gate | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-result-2026-07-10.md` |
| 7 | KSC-SV compact precision gate | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-result-2026-07-10.md` |
| 8 | Cross-model wiring and smoke tests | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase8-cross-model-tests-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase8-cross-model-tests-result-2026-07-10.md` |
| 9 | Trusted GPU score-memory ladder | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-2026-07-10.md` |
| 10 | Leaderboard rebuild and closeout | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase10-leaderboard-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase10-leaderboard-result-2026-07-10.md` |

## Program Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Are all LEDH same-target model score adapters wired so their default score computation uses the compact no-time-history score route, and can the leaderboard be rebuilt without admitted historical score routes? |
| Baseline/comparator | Current code inventory on 2026-07-10, the repaired LGSSM `N=10000,T=50` score-only artifact, and existing per-model score contract tests. |
| Primary pass criterion | All model default score paths have tests proving compact route use; historical routes are explicit diagnostic-only and blocked from full admission; leaderboard rebuild emits no admitted historical route. |
| Veto diagnostics | Any default path calls memory-style reverse VJP; any full admission uses historical route; any production score defaults to float64 or TF32 disabled; any FD correctness path calls the score route instead of value-only scalar; GPU claims from non-trusted runs. |
| Explanatory diagnostics | Runtime, score memory, compile time, value-vs-exact comparisons, and short smoke artifacts. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, exact nonlinear likelihood correctness, full actual-SV `N=10000,T=1000` score admission unless explicitly run and admitted. |
| Artifacts | This master program, phase subplans/results, visible runbook, visible ledger, review bundles, JSON/Markdown run artifacts, and leaderboard outputs. |

## Review Loop

Each material subplan and result is reviewed by Claude when available via
`~/python/claudecodex/scripts/claude_review_gate.sh`. If Claude is unavailable
or blocked, Codex writes a substitute read-only review artifact. Claude review
is advisory only and cannot authorize boundary crossings.

For each phase:

1. Read subplan and restate evidence contract.
2. Run local checks required by the subplan.
3. Implement only scoped changes.
4. Write result/close record.
5. Draft or refresh next subplan.
6. Review next subplan for consistency, correctness, feasibility, artifact
   coverage, and boundary safety.

Stop after five review rounds for the same blocker.

## Anticipated Trusted Commands

Previously approved trusted execution is sufficient for:

- bounded Claude read-only review commands;
- trusted GPU/CUDA/TensorFlow/XLA checks;
- long benchmark/test commands that touch GPU/XLA.

Codex will request explicit approval if execution would require package
installation, external data fetch, destructive git operations, or modifying
files outside `/home/chakwong/BayesFilter` and `/tmp`.
