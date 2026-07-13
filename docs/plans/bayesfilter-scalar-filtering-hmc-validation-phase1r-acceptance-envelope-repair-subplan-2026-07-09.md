# Phase 1R Subplan: Acceptance-Envelope Repair Screen

Date: 2026-07-09
Status: `REVISED_AFTER_CODEX_SUBSTITUTE_REVIEW_ROUND_1`

## Phase Objective

Repair the Phase 1 acceptance-boundary veto without changing the scientific
claim boundary.  The repair keeps the same scalar fixed kernel, trajectory
length `1.57`, seeds, and burn-in length, reruns the same three Phase 1 seeds
with more retained draws, and asks whether the all-accepted seed was a
short-chain granularity artifact or a persistent conservative-envelope signal.

## Entry Conditions

- Phase 1 result exists and records `FAILED_ACCEPTANCE_SCREEN_REPAIR_TRIGGER`.
- Phase 1 artifact records finite samples, finite target log probabilities,
  finite log-accept ratios, and native divergence unavailable for all seeds.
- Phase 1 failed only because seed `(20260709, 6103)` had acceptance `1.0`.
- Native divergence unavailability remains not-zero-divergence evidence.
- No Phase 2 reference-agreement runtime has started.

## Required Artifacts

- Phase 1R harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py`
- Phase 1R JSON and Markdown:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md`
- Quiet runtime log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase1r_longer_same_kernel.log`
- Phase 1R result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-result-2026-07-09.md`
- Refreshed Phase 2 subplan only if Phase 1R passes.
- Serious-run manifest fields preserved in JSON/Markdown/result: git commit
  and dirty status, exact command, conda/Python/TensorFlow environment,
  CPU-hidden/GPU visibility, JIT/TF32 status, seeds, wall time, output paths,
  plan/subplan/result paths, and native-divergence telemetry status.

## Required Checks, Tests, And Reviews

- Run `git diff --check` before and after edits.
- Run focused tests for the Phase 1R harness and telemetry regression:
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1.py`
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1r.py`
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_common_inference_runtime_contracts.py::test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence tests/test_common_inference_runtime_contracts.py::test_hmc_screen_keeps_unavailable_diagnostics_from_passing_as_zero`
- Review this subplan with Claude or documented Codex fallback before runtime.
- Create the quiet log directory before redirected runtime:

```bash
mkdir -p docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09
```

- Run the CPU-hidden repair command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 720 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase1r_longer_same_kernel.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the same scalar fixed-kernel route avoid the Phase 1 all-accepted seed when the acceptance screen has more retained draws? |
| Baseline/comparator | Phase 1 result and its failing seed `(20260709, 6103)`. |
| Primary criterion | All three seeds produce retained finite samples, finite target log probabilities, finite log-accept ratios, no runtime errors, acceptance strictly between `0.05` and `0.99`, and no positive native divergence if native divergence telemetry is available. |
| Repair settings | Same `num_leapfrog_steps=4`, `step_size=0.3925`, trajectory length `1.57`, seeds `(20260709,6101)`, `(20260709,6102)`, `(20260709,6103)`, and `num_burnin_steps=4`, with only `num_results` increased from `16` to `64`. |
| Veto diagnostics | Runtime error, nonfinite required arrays, invalid artifact, missing seed, acceptance-screen failure, positive native divergence when available, treating unavailable native divergence as zero, telemetry semantics mismatch, or unsupported claim. |
| Explanatory diagnostics | Acceptance, log-accept tails, target-log-prob ranges, sample summaries, runtime, and native divergence availability status. |
| Not concluded | HMC convergence, posterior correctness, zero divergences if unavailable, sampler superiority, GPU/XLA readiness, default readiness, or source faithfulness. |

## Forbidden Claims And Actions

- Do not relax the acceptance threshold after seeing Phase 1.
- Do not tune step size, mass matrix, trajectory length, or seed selection in
  this repair.
- Do not call a Phase 1R pass HMC readiness.
- Do not run GPU/XLA in this repair.
- Do not use log-accept threshold exceedances as native divergence telemetry.
- Do not claim zero divergences unless native divergence telemetry is available
  and all native divergence indicators are false.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the Phase 1 failing acceptance-boundary artifact, not HMC readiness. |
| Proxy metrics promoted | Acceptance is a repair screen only; it does not establish posterior correctness or sampler quality. |
| Missing stop conditions | Any primary-criterion failure or veto diagnostic stops Phase 1R before Phase 2. |
| Unfair comparison | No method ranking occurs; this is a same-kernel repeat with more retained draws. |
| Hidden assumptions | The repair assumes the all-accepted seed may be a 16-draw granularity issue; this is tested directly with 64 retained draws while keeping burn-in fixed. |
| Stale context | Phase 1 result and Phase 0 telemetry repair are explicit entry conditions. |
| Environment mismatch | CPU-hidden repair evidence cannot support GPU/XLA or default-readiness claims. |
| Artifact mismatch | Phase 1R artifacts must carry manifest fields, telemetry status, metric roles, and nonclaims. |

Audit status: `PASSED_FOR_CODEX_SUBSTITUTE_REVIEW_BEFORE_RUNTIME`.

## Exact Next-Phase Handoff Conditions

Advance to Phase 2 only if Phase 1R passes its predeclared screen and the
Phase 2 reference-agreement subplan is refreshed and reviewed.  If native
divergence remains unavailable, the Phase 2 handoff must repeat that Phase 1R
made no zero-divergence claim.

## Stop Conditions

Stop for any primary-criterion failure or veto diagnostic: runtime error,
nonfinite required arrays, invalid or missing artifacts, missing seed,
acceptance-screen failure, positive native divergence when available,
unavailable native divergence treated as zero, telemetry semantics mismatch,
unsupported claim, missing telemetry policy, or review nonconvergence.  If
Phase 1R fails only by persistent boundary acceptance, write a blocker/next
repair result that recommends a separately reviewed fixed-trajectory
integration-envelope ladder rather than changing settings in place.
