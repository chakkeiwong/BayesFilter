# Phase 1 Subplan: CPU-Hidden Short-Chain Validation Screen

Date: 2026-07-09
Status: `REVISED_AFTER_CODEX_SUBSTITUTE_REVIEW_ROUND_1`

## Phase Objective

Run a modest CPU-hidden scalar fixed-kernel HMC validation screen after Phase 0
locks the telemetry policy.  The screen reruns the scalar filtering route with
new 2026-07-09 artifacts and an explicit repaired native-divergence contract.
This phase may support a limited finite/acceptance short-chain screen pass, but
not convergence, posterior correctness, or zero divergence when native
divergence is unavailable.

## Entry Conditions

- Phase 0 result exists and passes.
- Phase 1 subplan has been refreshed after Phase 0 with actual telemetry
  status: prior scalar artifacts report native divergence as
  `not_exposed_by_kernel`, not as zero.
- If native divergence is unavailable or `not_exposed_by_kernel`, Phase 1 must
  not use `zero_divergences` as a pass criterion and must not claim zero
  divergences.
- Scalar geometry, mass handoff, and Phase 5 replicated artifacts from
  2026-07-08 remain valid preconditions.  Phase 1 supersedes only the runtime
  validation artifact, not the geometry or mass handoff.
- Phase 1 execution uses `CUDA_VISIBLE_DEVICES=-1` and must be labeled a
  CPU-hidden debug/reference exception.
- Phase 1 uses the fixed reviewed scalar kernel settings
  `num_leapfrog_steps=4`, `step_size=0.3925`, trajectory length `1.57`,
  `num_results=16`, `num_burnin_steps=4`, and seeds
  `(20260709, 6101)`, `(20260709, 6102)`, `(20260709, 6103)`.

## Required Artifacts

- Phase 1 harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py`
- Phase 1 JSON and Markdown artifacts under `docs/benchmarks/`:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md`
- Quiet runtime log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase1_cpu_short_chain.log`
- Serious-run manifest fields preserved in JSON/Markdown/result: git commit
  and dirty status, exact command, conda/Python/TensorFlow environment,
  CPU-hidden/GPU visibility, JIT/TF32 status, seeds, wall time, output paths,
  plan/subplan/result paths, and native-divergence telemetry status.
- Phase 1 result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-result-2026-07-09.md`
- Refreshed Phase 2 reference-agreement subplan.

## Required Checks, Tests, And Reviews

- Run `git diff --check` before and after edits.
- Run focused tests for the new benchmark harness and the repaired telemetry
  policy:
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1.py`
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_common_inference_runtime_contracts.py::test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence tests/test_common_inference_runtime_contracts.py::test_hmc_screen_keeps_unavailable_diagnostics_from_passing_as_zero`
- Create the quiet log directory before redirecting benchmark output:

```bash
mkdir -p docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09
```

- Run CPU-hidden benchmark command with full stdout/stderr redirected to the
  quiet log path:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase1_cpu_short_chain.log 2>&1
```

- Review Phase 1 result and Phase 2 subplan with Claude or documented fallback.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the scalar fixed-kernel route pass a modest CPU-hidden finite/acceptance short-chain validation screen under the Phase 0 telemetry policy? |
| Baseline/comparator | 2026-07-08 three-seed finite-telemetry diagnostic. |
| Primary criterion | All three seeds produce retained samples, finite samples, finite target log probabilities, finite log-accept ratios, no runtime errors, per-seed acceptance strictly between `0.05` and `0.99`, and no positive native divergence if native divergence telemetry is available.  If native divergence is unavailable, it is recorded as unavailable and is not part of the pass criterion. |
| Veto diagnostics | Runtime error, nonfinite required arrays, invalid artifact, missing seed, acceptance screen failure, positive native divergence when available, treating unavailable native divergence as zero, telemetry semantics mismatch, or unsupported claim. |
| Explanatory diagnostics | Acceptance, log-accept tails, target-log-prob ranges, sample summaries, runtime, and native divergence availability status. |
| Not concluded | HMC convergence, posterior correctness, zero divergences if unavailable, sampler superiority, GPU/XLA readiness, default readiness, or source faithfulness. |

Implementation boundary: the Phase 1 harness must derive a Phase 1
finite/acceptance gate separately from raw `HMCScreenResult.passed`.  Under the
Phase 0 repair, raw `screen_hmc_diagnostics(...).passed` must be false whenever
native divergence telemetry is unavailable because `zero_divergences` is
unavailable/false.  That raw screen result must not be used as the Phase 1 pass
flag when native divergence is unavailable.

## Forbidden Claims And Actions

- Do not rank sampler quality from descriptive short-chain metrics.
- Do not tune thresholds after seeing results.
- Do not call this HMC readiness.
- Do not run GPU/XLA in this phase.
- Do not use log-accept threshold exceedances as native divergence telemetry.
- Do not claim zero divergences unless native divergence telemetry is available
  and all native divergence indicators are false.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the 2026-07-08 three-seed scalar finite-telemetry artifact, not HMC readiness or convergence. |
| Proxy metrics promoted | Acceptance and log-accept tails are finite/acceptance screen inputs or explanatory diagnostics only; they do not establish posterior correctness or native zero divergence. |
| Missing stop conditions | Nonfinite outputs, acceptance-screen failure, invalid artifacts, positive native divergence when available, telemetry mismatch, or unsupported claims stop the phase. |
| Unfair comparison | No method comparison or ranking occurs. |
| Hidden assumptions | The fixed kernel remains the reviewed 2026-07-08 scalar kernel with trajectory length `1.57`; no tuning is introduced. |
| Stale context | Phase 0 repair is an entry condition, and the new artifact must record the current dirty commit/status. |
| Environment mismatch | CPU-hidden results cannot support GPU/XLA or default-readiness claims. |
| Artifact mismatch | The new 2026-07-09 JSON/Markdown artifacts must carry plan/result paths, metric roles, nonclaims, and native-divergence semantics. |

Audit status: `PASSED_FOR_CODEX_SUBSTITUTE_REVIEW_BEFORE_RUNTIME`.

## Exact Next-Phase Handoff Conditions

Advance to Phase 2 only if Phase 1 artifacts pass the predeclared screen and a
reviewed Phase 2 subplan defines the scalar reference and agreement criterion.
If native divergence remains unavailable, the Phase 2 handoff must repeat that
Phase 1 made no zero-divergence claim.

## Stop Conditions

Stop for any primary-criterion failure or veto diagnostic: runtime error,
nonfinite required arrays, invalid or missing artifacts, missing seed,
acceptance-screen failure, positive native divergence when available,
unavailable native divergence treated as zero, telemetry semantics mismatch,
unsupported claim, missing telemetry policy, or review nonconvergence.  A
fixable implementation/documentation issue may enter the visible repair loop;
otherwise write a Phase 1 blocker result before Phase 2.
