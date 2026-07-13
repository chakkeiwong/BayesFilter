# Phase 2U Subplan: Retuned MAP-Local HMC Screen

Date: 2026-07-09
Status: `DRAFT_PENDING_CODEX_SUBSTITUTE_REVIEW`

## Phase Objective

Run a CPU-hidden finite/acceptance screen for the scalar filtering target in
the new Phase 2S MAP-local `u_new` coordinate.  The screen tests a fixed grid
of equal trajectory-length HMC kernels and selects the first candidate that
passes hard vetoes plus the acceptance envelope for a later reviewed longer
screen.

This phase is not a posterior agreement phase, not a convergence phase, and not
a GPU/XLA/default-readiness phase.

## Entry Conditions

- Phase 2S passed and produced a MAP-local center, scale, `factor_z`,
  `precision_z`, `covariance_z`, `precision_theta`, and `covariance_theta`.
- Phase 2T passed and validated the MAP-local handoff, including:
  - `precision_z @ covariance_z` identity max error
    `1.1254733403169899e-15`;
  - `factor_z @ factor_z.T` reconstruction max error
    `8.881784197001252e-16`;
  - precision theta scale-transform max error
    `1.0000285044498014e-09`;
  - covariance theta scale-transform max error
    `4.887283910903761e-10`.
- Old Phase 1R HMC summaries remain old-geometry diagnostics only and must not
  be used as MAP-local HMC evidence.
- Native divergence remains unavailable in earlier artifacts and unavailable
  telemetry must not be treated as zero divergences.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2U harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py`
- Phase 2U tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.py`
- Phase 2U JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2u_retuned_map_local_hmc_screen.log`
- Phase 2U result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-result-2026-07-09.md`
- Refreshed next subplan only if Phase 2U passes:
  a longer selected-kernel MAP-local screen subplan before any GPU/XLA phase.

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.  Claude is unavailable for repo-context
  review in this recovered run, so use a fresh Codex substitute reviewer and
  record that it is weaker than a full Claude material review.
- Run focused tests before runtime:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py
```

- Run `git diff --check`.
- Create the quiet log directory before redirected runtime:

```bash
mkdir -p docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09
```

- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 720 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2u_retuned_map_local_hmc_screen.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does a fixed equal-trajectory-length HMC grid in the MAP-local `u_new` coordinate produce at least one candidate with finite samples/telemetry and acceptance inside the predeclared envelope? |
| Baseline/comparator | Phase 2S/2T MAP-local affine handoff.  Phase 1R old-geometry HMC summaries are excluded from pass/fail and may appear only as explanatory stale-context metadata. |
| Primary pass criterion | At least one candidate in the exact grid passes hard vetoes and has acceptance strictly between `0.05` and `0.99`; select the first passing candidate in the listed order. |
| Veto diagnostics | Runtime error, invalid Phase 2S or 2T artifact, invalid MAP-local affine adapter, nonfinite initial target value/score, nonfinite retained samples, nonfinite target log-prob trace, nonfinite log-accept ratio, positive native divergence when native divergence telemetry is available, missing candidate rows, no candidate satisfying the acceptance envelope, invalid output artifact, or unsupported claim. |
| Explanatory diagnostics | Per-candidate acceptance, log-accept summaries, target-log-prob summaries, sample ranges, initial target values/scores, runtime, and native-divergence availability status. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences when native telemetry is unavailable, sampler superiority, statistically supported ranking, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2U JSON, Markdown, quiet log, result file, and refreshed handoff/next subplan if passed. |

## Fixed Candidate Grid

| Candidate | Leapfrog steps | Step size | `L * step_size` |
| --- | --- | --- | --- |
| 0 | `2` | `0.785` | `1.57` |
| 1 | `4` | `0.3925` | `1.57` |
| 2 | `8` | `0.19625` | `1.57` |
| 3 | `16` | `0.098125` | `1.57` |

The selection policy is the first candidate in the table that passes all hard
vetoes and the acceptance envelope `(0.05, 0.99)`.  This policy is a handoff
rule for the next reviewed longer screen, not evidence that earlier candidates
are superior.

## Implementation Design

- Load Phase 2S and Phase 2T JSON artifacts.
- Validate:
  - Phase 2S `decision.phase2s_geometry_centering_repair_passed is True`;
  - Phase 2T `decision.phase2t_map_local_reference_handoff_passed is True`;
  - `factor_z`, `scale`, and `center_free_parameter_values` are finite with
    expected shapes;
  - `factor_z @ factor_z.T` reconstructs `covariance_z`;
  - `precision_z @ covariance_z` is identity;
  - theta/z scale transforms match the recorded `precision_theta` and
    `covariance_theta`.
- Rebuild the scalar filtering target using the existing benchmark target
  builder and verify target scale/free-parameter names match Phase 2S.
- Construct a base free-parameter adapter and wrap it in
  `LatentAffineBatchValueScoreAdapter` using:
  - HMC coordinate: `u_new`;
  - transform factor: `diag(scale) @ factor_z`;
  - adapter convention: `free = center + u_new @ (diag(scale) @ factor_z).T`,
    using the repository row-vector convention
    `theta = center + z @ factor.T`.
- Start every candidate at `u_new = 0`, the MAP-local center.
- For each candidate, run the existing non-XLA fixed-kernel HMC runner with:
  - `num_results=64`;
  - `num_burnin_steps=4`;
  - seed `(20260709, 6301 + candidate_index)`;
  - `trace_policy="standard"`;
  - `adaptation_policy="fixed_kernel_no_adaptation"`;
  - `chain_execution_mode="eager"`.
- Evaluate each candidate with the hard vetoes and acceptance envelope.
- Write a blocker/repair result if no candidate passes; do not continue to GPU.

## Forbidden Claims And Actions

- Do not run GPU/XLA in Phase 2U.
- Do not change defaults or public API behavior.
- Do not retune after seeing results or add unplanned candidate values.
- Do not treat old Phase 1R samples as MAP-local samples.
- Do not treat unavailable native divergence telemetry as zero divergences.
- Do not use log-accept thresholds as native-divergence telemetry.
- Do not claim posterior correctness, HMC readiness/convergence, sampler
  superiority, statistically supported ranking, default readiness, or
  Zhao-Cui source faithfulness.

## Exact Next-Phase Handoff Conditions

If Phase 2U passes:

- write the Phase 2U result with a decision table, inference-status table, run
  manifest, selected candidate, nonclaims, and post-run red-team note;
- draft a dedicated longer selected-kernel MAP-local screen subplan before any
  longer runtime;
- review that next subplan before execution;
- keep Phase 3 GPU/XLA blocked until the longer MAP-local repair branch has a
  valid handoff.

If Phase 2U fails:

- write a blocker/repair result;
- preserve all candidate rows and hard vetoes;
- do not proceed to Phase 3;
- draft only a narrower repair subplan if the failure is implementation or
  tuning-localization evidence rather than target invalidity.

## Stop Conditions

Stop for invalid Phase 2S/2T artifacts, invalid MAP-local affine adapter,
runtime exceptions that prevent artifact creation, nonfinite target or samples,
positive native divergence when available, no candidate passing the acceptance
envelope, review nonconvergence, or any need to cross GPU, default-policy,
model-file, source-faithfulness, or scientific-claim boundaries.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the Phase 2S/2T MAP-local handoff, not old Phase 1R HMC success. |
| Proxy metrics promoted | Finite telemetry and acceptance only select a candidate for a longer screen; they do not prove posterior correctness or readiness. |
| Missing stop conditions | Artifact, adapter, telemetry, acceptance, review, and claim-boundary stop conditions are explicit. |
| Unfair comparison | No method ranking occurs; candidate order is a predeclared handoff rule. |
| Hidden assumptions | The MAP-local covariance is a diagnostic local geometry; it is not assumed to be exact posterior covariance. |
| Stale context | Phase 2U reloads current Phase 2S and Phase 2T JSON artifacts before runtime. |
| Environment mismatch | CPU-hidden non-XLA evidence cannot support GPU/XLA/default-readiness claims. |
| Artifact mismatch | JSON/Markdown/result/log paths and the next handoff are predeclared. |

Audit status: `PASSED_FOR_REVIEW_ONLY`.  Runtime may begin only after review
converges.
