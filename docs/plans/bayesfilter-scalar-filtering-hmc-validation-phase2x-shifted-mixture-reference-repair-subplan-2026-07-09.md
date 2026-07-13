# Phase 2X Subplan: Shifted-Mixture Importance Reference Repair

Date: 2026-07-09
Status: `DRAFT_PENDING_REVIEW`

## Phase Objective

Repair the Phase 2W reference-proposal failure by building a defensive
shifted-mixture importance proposal in the same Phase 2S/2U MAP-local `u_new`
coordinate.  Use only Phase 2W pilot reference diagnostics to set the shifted
proposal center/scale, then compare Phase 2V HMC moment summaries only if the
new reference validity gates pass.

This phase is a reference-proposal repair diagnostic.  It is not posterior
certification, not HMC readiness, and not a GPU/XLA/default-readiness phase.

## Entry Conditions

- Phase 2W ran and wrote:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-result-2026-07-09.md`.
- Phase 2W failed only because the fixed standard-normal proposal had
  insufficient ESS:
  - `reference_ess=22.894679726459746`;
  - `reference_ess_ratio=0.022358085670370845`;
  - vetoes `reference_ess_below_threshold` and
    `reference_ess_ratio_below_threshold`.
- Phase 2W had finite proposal values, finite target values, finite log
  weights, and did not interpret HMC-vs-reference agreement.
- Phase 2V selected-kernel HMC summaries remain the comparator:
  - `L=2`;
  - `step_size=0.785`;
  - retained draws `128`;
  - acceptance `0.40625`.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2X harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py`
- Phase 2X tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.py`
- Phase 2X JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2x_shifted_mixture_reference_repair.log`
- Phase 2X result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.  Claude remains unavailable for
  repo-context material review per prior handoff, so use a fresh Codex
  substitute reviewer and record that it is weaker than full Claude material
  review.
- Run focused tests before runtime:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py
```

- Run `git diff --check`.
- Create the quiet log directory before redirected runtime.
- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2x_shifted_mixture_reference_repair.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does a defensive shifted-mixture proposal, tuned only from Phase 2W pilot reference diagnostics, produce a usable self-normalized importance reference; and if so do Phase 2V HMC moment summaries agree with it under the predeclared moment screens? |
| Baseline/comparator | Phase 2W failed standard-normal proposal as the reference-proposal baseline; Phase 2V selected-kernel HMC summaries as the agreement comparator. |
| Primary pass criterion | Phase 2X reference validity passes with ESS at least `256` and ESS ratio at least `0.125`, then all four HMC mean components are within `max(0.75, 4 * reference_mean_mcse)` of the Phase 2X reference mean and all four HMC standard-deviation ratios are within `[0.5, 2.0]`. |
| Reference validity vetoes | Invalid Phase 2S/2U/2V/2W artifacts, Phase 2W failure for reasons other than ESS thresholds, invalid Phase 2U selected-kernel handoff, invalid MAP-local adapter, proposal parameters using Phase 2V HMC moments, nonfinite proposal values, nonfinite target values, nonfinite mixture log densities, nonfinite log weights, normalized weight degeneracy, reference ESS below `256`, reference ESS ratio below `0.125`, or invalid artifact. |
| HMC agreement vetoes | Missing Phase 2V HMC mean/std summaries, nonfinite HMC summaries, mean component outside threshold, std ratio outside `[0.5, 2.0]`, or unsupported claim. |
| Explanatory diagnostics | Phase 2W pilot center/scale, mixture component counts, mixture log-density range, reference ESS, weight range, reference means/stds, approximate MCSEs, HMC-reference deltas, HMC std ratios, proposal sample extrema, runtime, and Phase 2V acceptance/telemetry status. |
| Not concluded | No exact posterior correctness, no HMC readiness/convergence, no zero divergences when native telemetry is unavailable, no sampler superiority, no statistically supported broad ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2X JSON, Markdown, quiet log, result file, and refreshed handoff. |

## Fixed Repair Design

- Proposal coordinate: Phase 2S/2U MAP-local `u_new`.
- Pilot source: Phase 2W weighted mean and weighted standard deviation in
  `u_new`.  Do not use Phase 2V HMC moments to set proposal parameters.
- Pilot center:
  `[0.16900152112527375, 0.34590014590251295, 0.47216707577215133, -0.3362900480743778]`.
- Pilot scale before clipping:
  `1.25 * [1.1289232726542155, 1.3947178163994365, 1.7877962561383989, 1.7764811837333756]`.
- Shifted diagonal scale:
  `clip(1.25 * phase2w_reference_std_u_new, lower=0.75, upper=3.0)`.
- Proposal distribution:
  `0.25 * N(0, I_4) + 0.75 * N(pilot_center, diag(shifted_scale^2))`.
- Proposal sample count: `2048`.
- Sampling design: fixed NumPy PCG seed `(20260709, 6601)` with antithetic
  pairs inside each component:
  - `512` standard-normal component samples;
  - `1536` shifted diagonal component samples.
- Proposal log density: exact mixture density evaluated for every proposal
  sample using a stable log-sum-exp formula, with normalizing constants.
- Target evaluation: Phase 2U MAP-local adapter target log probability in
  `u_new`, with constant affine Jacobian omitted consistently for normalized
  weights.
- Reference mean/std: self-normalized importance estimates.
- Reference mean MCSE proxy: square root of the weighted second-moment variance
  divided by ESS; explanatory only except where the predeclared mean threshold
  uses it.

## Forbidden Claims And Actions

- Do not run GPU/XLA in Phase 2X.
- Do not change defaults or public API behavior.
- Do not retune proposal weights, sample count, scale multiplier, clipping
  bounds, ESS thresholds, or agreement thresholds after seeing Phase 2X results.
- Do not use Phase 2V HMC moments to set the Phase 2X proposal center or scale.
- Do not treat the shifted-mixture proposal as exact posterior truth.
- Do not treat unavailable native divergence telemetry as zero divergences.
- Do not use log-accept thresholds as native-divergence telemetry.
- Do not claim posterior correctness, HMC readiness/convergence, sampler
  superiority, statistically supported ranking, default readiness, or
  Zhao-Cui source faithfulness.

## Exact Next-Phase Handoff Conditions

If Phase 2X passes, write the Phase 2X result and draft a reviewed independent
shifted-mixture reference replication subplan with frozen proposal parameters
and a fresh seed.  GPU/XLA remains blocked until that later reference
replication result explicitly authorizes a GPU/XLA reproduction subplan.

If Phase 2X fails because the shifted-mixture reference is invalid, write a
reference-repair blocker result and do not interpret HMC-vs-reference
agreement.  If the reference is valid but HMC agreement fails, write a
candidate-rejection or tuning/localization repair result without rejecting the
whole research direction.

## Stop Conditions

Stop for invalid Phase 2S/2U/2V/2W artifacts, Phase 2W failure for reasons
other than ESS thresholds, use of HMC moments to set proposal parameters,
invalid Phase 2U selected-kernel handoff, invalid MAP-local adapter, nonfinite
proposal/target/log weights, reference ESS below threshold, missing HMC
summaries, failed agreement screen, timeout, review nonconvergence, or any need
to cross GPU, default-policy, model-file, source-faithfulness, or
scientific-claim boundaries.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | The baseline is the failed Phase 2W standard-normal proposal; the HMC comparator remains the unchanged Phase 2V selected chain. |
| Proxy metrics promoted | ESS validity and moment agreement are diagnostic screens only and cannot certify posterior correctness or readiness. |
| Missing stop conditions | Pilot validity, reference validity, HMC agreement, artifact, review, and claim-boundary stops are explicit. |
| Unfair comparison | The proposal is tuned only from Phase 2W target-weight diagnostics, not from HMC moments; no method ranking occurs. |
| Hidden assumptions | Phase 2W pilot moments are noisy because Phase 2W ESS was low; the mixture includes a defensive standard-normal component and must pass fresh ESS gates before agreement is interpreted. |
| Stale context | Phase 2X reloads current Phase 2S, Phase 2U, Phase 2V, and Phase 2W artifacts before runtime. |
| Environment mismatch | CPU-hidden non-XLA evidence cannot support GPU/XLA/default-readiness claims. |
| Artifact mismatch | JSON/Markdown/result/log paths and fixed repair design are predeclared. |

Audit status: `PASSED_FOR_REVIEW_ONLY`.  Runtime may begin only after review
converges.
