# Phase 2W Subplan: MAP-Local Importance Reference Agreement

Date: 2026-07-09
Status: `DRAFT_PENDING_REVIEW`

## Phase Objective

Build an independent CPU-hidden scalar reference diagnostic for the MAP-local
`u_new` coordinate using a fixed standard-normal proposal and self-normalized
importance weights, then compare Phase 2V selected-kernel HMC moment summaries
against that reference only if the reference validity gates pass.

This phase is a reference/agreement diagnostic.  It is not posterior
certification, not HMC readiness, and not a GPU/XLA/default-readiness phase.

## Entry Conditions

- Phase 2V passed and wrote:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-result-2026-07-09.md`.
- Phase 2V selected kernel:
  - `num_leapfrog_steps=2`;
  - `step_size=0.785`;
  - trajectory length `1.57`;
  - retained draws `128`;
  - acceptance `0.40625`.
- Phase 2V native divergence telemetry was unavailable and made no
  zero-divergence claim.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2W harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py`
- Phase 2W tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py`
- Phase 2W JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2w_importance_reference_agreement.log`
- Phase 2W result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.  If Claude remains unavailable for
  repo-context review, use a fresh Codex substitute reviewer and record that it
  is weaker than full Claude material review.
- Run focused tests before runtime:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py
```

- Run `git diff --check`.
- Create the quiet log directory before redirected runtime.
- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2w_importance_reference_agreement.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does a fixed MAP-local standard-normal proposal produce a usable self-normalized importance reference, and if so do Phase 2V HMC moment summaries agree with it under predeclared moment screens? |
| Baseline/comparator | Phase 2V selected-kernel HMC summaries in `u_new`; independent fixed-seed standard-normal proposal in the same coordinate. |
| Primary pass criterion | Reference validity passes, then all four HMC mean components are within `max(0.75, 4 * reference_mean_mcse)` of the importance reference mean and all four HMC standard-deviation ratios are within `[0.5, 2.0]`. |
| Reference validity vetoes | Invalid Phase 2S/2U/2V artifacts, invalid Phase 2U selected-kernel handoff, invalid MAP-local adapter, nonfinite proposal values, nonfinite target values, nonfinite log weights, normalized weight degeneracy, reference ESS below `128`, reference ESS ratio below `0.125`, or invalid artifact. |
| HMC agreement vetoes | Missing Phase 2V HMC mean/std summaries, nonfinite HMC summaries, mean component outside threshold, std ratio outside `[0.5, 2.0]`, or unsupported claim. |
| Explanatory diagnostics | Reference ESS, weight range, reference means/stds, approximate MCSEs, HMC-reference deltas, HMC std ratios, proposal sample extrema, runtime, and Phase 2V acceptance/telemetry status. |
| Not concluded | No exact posterior correctness, no HMC readiness/convergence, no zero divergences when native telemetry is unavailable, no sampler superiority, no statistically supported broad ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2W JSON, Markdown, quiet log, result file, and refreshed handoff. |

## Fixed Reference Design

- Proposal coordinate: Phase 2S/2V MAP-local `u_new`.
- Proposal distribution: standard normal `N(0, I_4)` in `u_new`.
- Proposal sample count: `1024`.
- Sampling design: fixed NumPy PCG seed `(20260709, 6501)` with antithetic
  pairs; `512` base draws and their negatives.
- Target evaluation: Phase 2U MAP-local adapter target log probability in
  `u_new`, with constant affine Jacobian omitted consistently for normalized
  weights.
- Proposal log density: standard normal log density in `u_new`, including the
  normalizing constant; constants are harmless but recorded.
- Reference mean/std: self-normalized importance estimates.
- Reference mean MCSE proxy: square root of the weighted second-moment
  variance divided by ESS; explanatory only except where the predeclared mean
  threshold uses it.

## Forbidden Claims And Actions

- Do not run GPU/XLA in Phase 2W.
- Do not change defaults or public API behavior.
- Do not retune the HMC kernel, proposal sample count, ESS threshold, or
  agreement thresholds after seeing results.
- Do not treat the MAP-local Gaussian proposal as exact posterior truth.
- Do not treat unavailable native divergence telemetry as zero divergences.
- Do not use log-accept thresholds as native-divergence telemetry.
- Do not claim posterior correctness, HMC readiness/convergence, sampler
  superiority, statistically supported ranking, default readiness, or
  Zhao-Cui source faithfulness.

## Exact Next-Phase Handoff Conditions

If Phase 2W passes, write the Phase 2W result and draft a reviewed strengthened
reference replication subplan or a narrowly scoped GPU/XLA reproduction subplan
only if the result explicitly states why the reference evidence is enough for
that next plan.  Runtime for GPU/XLA remains blocked until that next subplan is
reviewed and any required human/trusted-runtime approval is satisfied.

If Phase 2W fails because the importance reference is invalid, write a blocker
or reference-repair result and do not interpret HMC-vs-reference agreement.  If
the reference is valid but HMC agreement fails, write a candidate-rejection or
tuning/localization repair result without rejecting the whole research
direction.

## Stop Conditions

Stop for invalid Phase 2S/2U/2V artifacts, invalid Phase 2U selected-kernel
handoff, invalid MAP-local adapter, nonfinite proposal/target/log weights,
reference ESS below threshold, missing HMC summaries, failed agreement screen,
timeout, review nonconvergence, or any need to cross GPU, default-policy,
model-file, source-faithfulness, or scientific-claim boundaries.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Comparator is a new fixed-seed importance reference in the same `u_new` coordinate, not the old Phase 1R geometry or MAP Gaussian itself. |
| Proxy metrics promoted | Importance agreement is a diagnostic screen and cannot certify exact posterior correctness or readiness. |
| Missing stop conditions | Reference validity, HMC agreement, artifact, review, and claim-boundary stops are explicit. |
| Unfair comparison | No method ranking occurs; the HMC chain is compared to a predeclared independent reference diagnostic. |
| Hidden assumptions | The proposal is only a proposal. ESS and finite-weight gates must pass before agreement is interpreted. |
| Stale context | Phase 2W reloads current Phase 2S and Phase 2V artifacts before runtime. |
| Environment mismatch | CPU-hidden non-XLA evidence cannot support GPU/XLA/default-readiness claims. |
| Artifact mismatch | JSON/Markdown/result/log paths and fixed reference design are predeclared. |

Audit status: `PASSED_FOR_REVIEW_ONLY`.  Runtime may begin only after review
converges.
