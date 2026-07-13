# Phase 1 Result: CPU-Hidden Short-Chain Validation Screen

Date: 2026-07-09
Status: `FAILED_ACCEPTANCE_SCREEN_REPAIR_TRIGGER`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 1 does not pass the predeclared finite/acceptance screen | Failed: seed `20260709,6103` had acceptance `1.0`, outside the strict `(0.05, 0.99)` screen | Runtime produced finite samples, finite target log probabilities, and finite log-accept ratios; native positive-divergence veto was unassessable because native divergence telemetry was unavailable; acceptance boundary veto fired | With only 16 retained draws, an all-accepted seed may be a short-chain granularity/tuning-envelope issue rather than target invalidity | Draft and review a Phase 1 repair subplan that keeps the fixed kernel and trajectory but increases retained draws before any Phase 2 reference-agreement handoff | No HMC readiness, convergence, posterior correctness, zero-divergence claim, GPU/XLA readiness, default readiness, sampler superiority, or Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered: the 2026-07-09 Phase 1 finite/acceptance screen did not pass. |
| Baseline/comparator | 2026-07-08 three-seed finite-telemetry diagnostic. |
| Primary criterion | Failed because one seed had acceptance `1.0`; all other required finite telemetry checks passed. |
| Veto diagnostics | `seed_2_acceptance_outside_phase1_screen`. |
| Explanatory diagnostics | Acceptance rates `[0.9375, 0.75, 1.0]`; log-accept max abs `[2.1187000672423397, 48.620365994974954, 0.580877228359129]`; max abs `u` `[3.9364129599327216, 6.603472829746424, 10.427120225910855]`. |
| Not concluded | No HMC convergence, posterior correctness, zero divergences, sampler superiority, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase1_cpu_short_chain.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py`

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `f297b103303c64019302ed5d9b9aaf2c8f919b64` |
| Git dirty status | Dirty; artifact recorded current planned edits and untracked runbook files |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | CPU-hidden debug/reference exception with `CUDA_VISIBLE_DEVICES=-1`; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seeds | `(20260709, 6101)`, `(20260709, 6102)`, `(20260709, 6103)` |
| Wall time | `179.8436475569615` seconds |
| Plan/result paths | Master, Phase 1 subplan, and this result file recorded in JSON |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Failed by acceptance boundary screen. |
| Statistically supported ranking | None; no method comparison and no uncertainty interval. |
| Descriptive-only differences | Acceptance, log-accept tails, target-log-prob ranges, sample ranges, and runtime are descriptive after the hard screen. |
| Default readiness | Not assessed. |
| Next evidence needed | A reviewed Phase 1 repair run or blocker before Phase 2. |

## Native-Divergence Telemetry

All three seeds reported native divergence as `not_exposed_by_kernel`.
Unavailable native divergence was not treated as zero divergences.  The artifact
records:

- `zero_divergence_claim_made: false`
- `unavailable_native_divergence_is_zero_divergence: false`
- `log_accept_threshold_used_as_native_divergence: false`

## Checks

| Check | Status |
| --- | --- |
| `git diff --check` before runtime | Passed |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1.py` | Passed: `5 passed` |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_common_inference_runtime_contracts.py::test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence tests/test_common_inference_runtime_contracts.py::test_hmc_screen_keeps_unavailable_diagnostics_from_passing_as_zero` | Passed: `2 passed, 2 warnings` |
| Phase 1 runtime command | Exited `0`; artifact decision failed by predeclared acceptance screen |

## Repair Trigger

The failure is classified as a Phase 1 repair trigger, not a research-direction
rejection:

- What failed: one short chain had all 16 proposals accepted.
- What did not fail: finite samples, finite target log probabilities, finite
  log-accept ratios, scalar geometry/mass preconditions, and telemetry
  boundary semantics.
- What repair is justified: a reviewed Phase 1 repair that keeps the same
  fixed kernel and trajectory length `1.57` but increases retained draws before
  deciding whether the acceptance boundary is persistent.

## Final Nonclaims

- No HMC readiness.
- No HMC convergence.
- No posterior correctness.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
