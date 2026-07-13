# Phase 0 Subplan: Contract, Baseline, And Source Fingerprint

Date: 2026-07-11
Status: `PHASE_CLOSED_HANDOFF_PASSED`

## Phase Objective

Freeze the current engineering question, historical-invalid artifact boundary,
source fingerprints, environment facts, and phase gates before any source edit.

## Entry Conditions

- The 2026-07-10 reset memo is loaded.
- No Kalman QR worker is active.
- The dirty worktree is preserved; no reset/revert is authorized.
- The master program exists and has passed local document checks.

## Required Artifacts

- This subplan.
- `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase0-contract-baseline-result-2026-07-11.md`.
- `docs/benchmarks/kalman_qr_batched_xla_repair_phase0_inventory_2026-07-11.json`.
- `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-master-review-bundle-2026-07-11.md`.
- `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase0-review-bundle-2026-07-11.md`.
- Initial substitute review:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase0-codex-substitute-review-round1-2026-07-11.md`.
- If repair requires later rounds, sequential exact paths ending in
  `round2-2026-07-11.md` through `round5-2026-07-11.md`; only actually reached
  rounds are required, and the final reached record must contain the verdict.
- Claude policy-block status is preserved in
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-review-boundary-blocker-result-2026-07-11.md`.
- Refreshed Phase 1 subplan:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase1-harness-integrity-subplan-2026-07-11.md`.
- Phase 1 next-subplan review record:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase1-subplan-codex-substitute-review-round1-2026-07-11.md`,
  followed by sequential `round2` through `round5` paths only if reached.

## Required Checks, Tests, And Reviews

- Record `git rev-parse HEAD` and `git status --short` for this explicit relevant
  path set: `scripts/benchmark_kalman_qr_parameter_count_scaling.py`,
  `bayesfilter/linear/kalman_qr_derivatives_tf.py`,
  `bayesfilter/linear/kalman_qr_tf.py`, `bayesfilter/linear/qr_factor_tf.py`,
  `tests/test_linear_qr_batched_analytical_score_tf.py`,
  `tests/test_linear_kalman_qr_derivatives_tf.py`, and
  `docs/benchmarks/run_kalman_qr_core_batch_grid_overnight_2026_07_09.py`.
- Record SHA-256 for every path in that exact set before inventory and repeat
  HEAD/status/hashes immediately before handoff; require exact stability.
- Record non-mutating environment facts without importing TensorFlow or probing
  GPU/CUDA: interpreter path/version, `CONDA_DEFAULT_ENV`, `CONDA_PREFIX`,
  `CUDA_VISIBLE_DEVICES`, `XLA_FLAGS`, package metadata versions for TensorFlow,
  TensorFlow Probability, NumPy, and Python implementation/platform. GPU identity
  is `not_probed_phase0`; cite only previously recorded GPU configuration as
  historical evidence, never as a current probe.
- Run `ps -eo pid,ppid,stat,etime,cmd` filtered to the exact Kalman benchmark
  patterns from the reset memo before inventory and again immediately before
  handoff. Both checks must show no active worker other than the inspection.
- AST/text inventory of Python loops, `tf.function`, `jit_compile`, timed materialization, method orchestration, and status/resume logic.
- Inventory these exact historical anchors:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-reset-memo-2026-07-10.md`,
  `docs/benchmarks/kalman_qr_core_batch_grid_overnight_status_2026-07-09.json`,
  `docs/benchmarks/run_kalman_qr_core_batch_grid_overnight_2026_07_09.py`, and
  `docs/benchmarks/kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_overnight_2026-07-09.json`.
  The inventory JSON must also contain an explicit `historical_cpu_row_artifacts`
  array enumerating every discovered path matching
  `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads*_batch*_xla_2026-07-09.json`
  and a `historical_gpu_preflight_artifacts` array enumerating every discovered
  path matching
  `docs/benchmarks/kalman_qr_core_batch_grid_preflight_gpu*_2026-07-09.json`.
  For each record store path, SHA-256, status, source fingerprint when known or
  `unknown`, invalidity reason, and disposition
  `historical_debug_only_nonresumable_under_repaired_schema`.
- Phase 0 verifies only the documented non-resumable contract. Behavioral
  resume enforcement and tests belong to Phase 1.
- Review protocol: the requested Claude Opus/max gate remains policy-blocked
  before probe despite user approval. If a future external attempt returns no
  response, first send the tiny `Reply exactly: OK` probe; probe success means
  redesign/shrink the bounded prompt, while only confirmed transport
  unavailability permits fresh Codex fallback. Current Phase 0 uses fresh bounded
  Codex substitute review, recording reviewer type, allowed path, findings,
  repairs, round, and verdict. Maximum five rounds for one material review blocker.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is there a complete, current, source-fingerprinted contract sufficient to begin surgical harness repair? |
| Baseline | Current dirty source plus reset memo and failed overnight artifacts. |
| Primary criterion | Inventory is reproducible, historical artifacts are explicitly non-promoting/non-resumable, and review finds no material planning gap. |
| Vetoes | Missing/invalid inventory, missing source/environment identity, changed closing fingerprint, active worker, unresolved historical disposition, wrong comparator, unbounded execution, failed required check, or unresolved material review finding. |
| Explanatory only | Historical timings and graph sizes. |
| Not concluded | No implementation correctness, compile viability, runtime ranking, or GPU readiness. |

## Forbidden Claims And Actions

- Do not edit algorithmic or benchmark source in Phase 0.
- Do not rerun XLA benchmarks or GPU probes.
- Do not call historical rows valid runtime evidence.
- Do not reset, clean, commit, or overwrite unrelated dirty work.

## Exact Next-Phase Handoff Conditions

All conditions are conjunctive:

- Every required local check completes successfully, including both no-worker checks.
- The inventory JSON exists, parses as strict JSON, contains every required
  source/environment/historical field, and all referenced paths/hashes validate.
- Closing Git HEAD, relevant status, and per-file SHA-256 exactly match the
  opening fingerprint.
- Phase 0 result records environment, source hashes, confirmed defects,
  hypotheses, historical disposition, checks, review trail, and nonclaims.
- Every evidence-contract veto is resolved.
- Phase 0 substitute review reaches exact `VERDICT: AGREE`; it is labeled weaker
  than Claude review and does not claim Claude transport failure.
- Phase 1 subplan is refreshed against Phase 0 findings and receives its own
  exact `VERDICT: AGREE` review for consistency, correctness, feasibility,
  artifact coverage, and boundary safety.

## Stop Conditions

- Source changed unexpectedly during inventory.
- An active benchmark worker is found.
- Inventory creation/schema/path/hash validation fails.
- Environment identity conflicts with the declared interpreter or historical context.
- Historical artifact disposition remains ambiguous.
- Any required local check or evidence-contract veto remains unresolved after in-scope repair.
- Review fails to converge after five rounds.
- The current source cannot be distinguished from historical artifacts.

## Mandatory Phase-End Sequence

1. Run all required local checks.
2. Write the Phase 0 result/close record.
3. Draft or refresh the Phase 1 subplan.
4. Review Phase 1 and repair/recheck it before advancing.
