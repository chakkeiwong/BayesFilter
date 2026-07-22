# Complete High-Dimensional Leaderboard Phase 1 Local Review

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Reviewer role: current-session Codex supervisor, read-only review pass over the
material implementation/result after executing local checks. The local-only
runbook forbids Claude, child Codex, network, and external API processes.

## Scope

- `docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py`
- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_compact_score_adapter.py`
- `docs/benchmarks/build_complete_highdim_ledh_phase2_phase3_command_manifest.py`
- `tests/highdim/test_ledh_compact_score_gpu_xla_harness.py`
- `docs/plans/complete-highdim-leaderboard-ledh-phase2-phase3-exact-commands-2026-07-11.json`
- `docs/plans/artifacts/complete-highdim-leaderboard/phase1-run-manifest-2026-07-12.json`
- `docs/plans/bayesfilter-complete-highdim-leaderboard-phase1-ledh-harness-result-2026-07-11.md`

## Review Question

Does Phase 1 safely close the six-row target/score/FD/aggregation and future
command-freeze gaps without admitting a cell or authorizing GPU execution?

## Review Rounds

| Round | Finding | Repair and evidence |
| --- | --- | --- |
| 1 | `REVISE`: validator tests did not explicitly cover all required target, theta, endpoint-vector, identity, seed-set, output-collision, aggregate-masking, and total-vs-average attacks | Added bounded adversarial cases and six-row aggregate fixtures; focused and full harness suites passed |
| 2 | `REVISE`: shard command hashes were self-descriptions and the old exact-command manifest was explicitly superseded | Added a current six-row deterministic builder, exact argv matching, unique outputs/hashes, timeout/environment/target/config/route/endpoint identities, and separate Phase 2/3 authority receipts |
| 3 | `REVISE`: current-source identity was present in the generated commands but not compared during runtime command matching; declared logs lacked an executable redirection form | Added current target/source/config/route/endpoint comparisons, `shell_command` log redirection, and a complete required-directory set; regenerated and checked P1-D |
| 4 | `AGREE`: no remaining material Phase 1 engineering or boundary finding after focused reruns | Dedicated harness `131 passed`; six row/cross-model suite `146 passed`; focused P1-D/validator suite `70 passed`; independent target and deterministic command checks passed |

## Final Assessment

- Baseline is correct: Phase 0/P1-A canonical targets and current-source
  compact routes, not historical July score claims.
- FD remains validation-only and uses the owner-directed individual
  `5% * sqrt(p)` rule; it is not an admitted score or confidence interval.
- Aggregate admission logic cannot hide an individual failed seed/direction.
- Total and average likelihood semantics are separated.
- LGSSM is target-preserving through a wrapper rather than a change to the
  target-defining value module.
- P1-D is deterministic and complete for its declared 96 commands but remains
  unusable without a phase-specific authority receipt.
- CPU-hidden checks do not establish GPU feasibility.
- No Zhao-Cui source-faithfulness claim is made; all exact adapters remain
  owner-approved `extension_or_invention`.

## Residual Risks

- Actual GPU/XLA compilation, placement, memory, full-time numerical behavior,
  and FD outcomes are not checked until Phase 2.
- A one-seed Phase 2 pass cannot admit a cell; Phase 3 five-seed evidence is
  still required.
- The dirty worktree is large. Runtime artifacts bind current computation
  hashes and must reject later computation-relevant drift.

VERDICT: AGREE
