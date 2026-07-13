# Phase 2Y Subplan: Target Geometry Localization

Date: 2026-07-09
Status: `REVIEWED_READY_FOR_RUNTIME_USER_CLEARED_DIAGNOSTIC_ONLY`

## Phase Objective

Localize why Phase 2W and Phase 2X importance proposals produced severe
weight concentration despite finite target and proposal evaluations.  The
phase should diagnose target/proposal mismatch in the Phase 2S/2U MAP-local
`u_new` coordinate before any new reference-agreement attempt.

This phase is a diagnostic localization phase.  It is not an HMC run, not an
importance-reference validity claim, not posterior certification, not HMC
readiness, and not a GPU/XLA/default-readiness phase.

## Entry Conditions

- Phase 2W result exists and failed only at reference ESS/ESS-ratio gates:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-result-2026-07-09.md`.
- Phase 2X result exists and failed only at reference ESS/ESS-ratio gates:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-result-2026-07-09.md`.
- Phase 2W and Phase 2X artifacts recorded finite target values, finite
  proposal log densities, and finite log weights.
- HMC-vs-reference agreement was not interpreted in Phase 2W or Phase 2X.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2Y harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py`
- Phase 2Y tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization.py`
- Phase 2Y JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2y_target_geometry_localization.log`
- Phase 2Y result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.  Claude review was attempted through the
  local review gate and blocked by the approval layer because it would transmit
  private repository planning and diagnostic context to an external Claude
  service.  Use a fresh Codex substitute review and record that it is weaker
  than full Claude material review.
- Run focused tests before runtime:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py
```

- Run `git diff --check`.
- Create the quiet log directory before redirected runtime.
- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 300 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2y_target_geometry_localization.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Do the Phase 2W/2X failures look like proposal-family mismatch, target tail/ridge/multimodality, local quadratic misspecification, or an artifact/log-density bug? |
| Baseline/comparator | Phase 2S MAP-local center, Phase 2W top-weight proposal points, Phase 2X top-weight proposal points, and fixed ray profiles through those points in `u_new`. |
| Primary pass criterion | The diagnostic writes a valid artifact with finite target values and scores for all predeclared anchor and ray points, records top-weight localization summaries, and preserves all nonclaims. |
| Veto diagnostics | Invalid Phase 2S/2U/2V/2W/2X artifacts, Phase 2W/2X failures not limited to ESS thresholds, missing top-weight data, nonfinite target value or score at predeclared diagnostic points, invalid ray construction, missing artifact, or unsupported claim. |
| Explanatory diagnostics | Top-weight coordinates/components, target values, score norms, radial target profiles from MAP center to top-weight points, symmetry/asymmetry diagnostics along antithetic rays, quadratic MAP approximation residuals along rays, and proposal log-density ranks. |
| Not concluded | No new valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistically supported ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2Y JSON, Markdown, quiet log, result file, and refreshed handoff. |

## Predeclared Hypotheses

The implementation audit before Phase 2Y found the shared affine adapter code
internally consistent with the row-vector contract:
`free = center + u_new @ factor.T` and
`grad_u_new = grad_free @ factor`.  Phase 2S/2U also verify
`factor_z @ factor_z.T = covariance_z` and compose the adapter factor as
`diag(scale) @ factor_z`.

Phase 2S nevertheless records a handoff display string
`free = center_free_parameter_values + scale * (factor_z @ u_new)`.  That
string is column-vector notation and is ambiguous in a row-vector codebase.
Phase 2Y therefore treats transform orientation as an explicit diagnostic
hypothesis instead of assuming the display string is harmless.

| ID | Hypothesis | Diagnostic Test | Evidence Role |
| --- | --- | --- | --- |
| H1 | The Phase 2S quadratic geometry is locally adequate only inside the trust radius `0.6`; Phase 2W/2X high-weight anchors are far outside that radius. | Compare high-weight anchor norms to the trust radius and replay target-minus-quadratic residuals along rays from the center. | Explanatory diagnostic; can motivate proposal redesign, not posterior correctness. |
| H2 | The target has heavy, ridge-like, or curved high-density regions in `u_new` that Gaussian proposals under-cover. | Inspect ray profiles, score directional components, antithetic asymmetry, top-weight coordinate summaries, and proposal log-density ranks. | Explanatory diagnostic; can nominate richer proposal families. |
| H3 | A subtle orientation or scaling mismatch exists despite current matrix checks. | Replay both row-vector and wrong-column orientation formulas at top anchors and compare target values, score norms, quadratic residuals, and recorded adapter output. | Bug-localization diagnostic; a confirmed mismatch triggers a repair subplan before any new reference attempt. |
| H4 | Proposal log densities are correct, but the proposal family assigns too little density to target-relevant regions. | Recompute standard-normal and shifted-mixture log densities at shared anchors, compare against saved log densities, and report target-log-probability versus log-proposal tension. | Bug-localization plus explanatory diagnostic. |
| H5 | The Phase 2S center is a local locator, not a certified global MAP; other modes or ridges may have comparable target value. | Report target values at high-weight anchors and along rays relative to the center value. | Explanatory only; does not certify modes or MAP quality. |
| H6 | The quadratic fit has acceptable local value holdout error but extrapolates poorly in score/curvature-relevant directions. | Compare MAP-local standard Gaussian quadratic predictions to target values along positive and antithetic rays. | Explanatory diagnostic; can justify local-reference abandonment or redesign. |

## Fixed Diagnostic Design

- Coordinate: Phase 2S/2U MAP-local `u_new`.
- Anchor set:
  - MAP-local center `u_new = [0, 0, 0, 0]`;
  - top `8` Phase 2W normalized-weight proposal points;
  - top `8` Phase 2X normalized-weight proposal points;
  - unique antithetic partners for those top points when present in the
    source proposal artifact.
- Ray profiles:
  - For each top point `v`, evaluate target value and score at
    `alpha * v` for `alpha in {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}`.
  - For each top point `v`, evaluate the antithetic ray at
    `-alpha * v` for the same alpha grid.
- Quadratic comparator:
  - Use the MAP-local standard Gaussian quadratic value
    `target(center) - 0.5 * ||u_new||^2` only as an explanatory local
    comparator.
  - Do not use quadratic residuals as a posterior correctness or reference
    validity criterion.
- Proposal-log-density check:
  - Replay Phase 2W standard-normal log density and Phase 2X shifted-mixture
    log density at anchor points to rank where high target values were
    under-covered.
- Do not use Phase 2V HMC moments to construct anchors, rays, or proposal
  repairs.

## Forbidden Claims And Actions

- Do not run HMC or GPU/XLA in Phase 2Y.
- Do not change defaults or public API behavior.
- Do not create or claim a new valid importance reference.
- Do not interpret HMC-vs-reference agreement.
- Do not use Phase 2V HMC moments to construct diagnostics or proposal repair
  parameters.
- Do not treat unavailable native divergence telemetry as zero divergences.
- Do not use log-accept thresholds as native-divergence telemetry.
- Do not claim posterior correctness, HMC readiness/convergence, sampler
  superiority, statistically supported ranking, default readiness, or
  Zhao-Cui source faithfulness.

## Exact Next-Phase Handoff Conditions

If Phase 2Y passes, write the Phase 2Y result and draft a reviewed proposal
redesign subplan.  That next subplan may choose among a non-diagonal Gaussian
mixture, mode/ridge-local mixture, transport proposal, or stopping the
importance-reference branch, but it must be justified by Phase 2Y diagnostics
and must preserve the same claim boundaries.

If Phase 2Y fails because target values or scores are nonfinite at diagnostic
points, write a target-validity blocker and stop for human direction.  If
Phase 2Y finds evidence of an artifact/log-density bug, write a bug-repair
subplan before any further reference attempt.

## Stop Conditions

Stop for invalid Phase 2S/2U/2V/2W/2X artifacts, Phase 2W/2X failures not
limited to ESS thresholds, missing proposal top-weight data, nonfinite target
values or scores, invalid ray construction, timeout, review nonconvergence, or
any need to cross HMC-runtime, GPU, default-policy, model-file,
source-faithfulness, or scientific-claim boundaries.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baselines are the failed Phase 2W/2X reference proposals and the MAP-local center, not HMC success. |
| Proxy metrics promoted | Target/ray profiles are explanatory diagnostics only; they cannot certify posterior correctness, HMC readiness, or valid reference agreement. |
| Missing stop conditions | Artifact validity, finite target/score, ray construction, review, and claim-boundary stops are explicit. |
| Unfair comparison | No method ranking occurs and HMC moments are not used to build diagnostics. |
| Hidden assumptions | Top-weight points come from low-ESS proposals and are diagnostic anchors only, not posterior samples. |
| Stale context | Phase 2Y reloads current Phase 2S, Phase 2U, Phase 2V, Phase 2W, and Phase 2X artifacts before runtime. |
| Environment mismatch | CPU-hidden non-XLA diagnostics cannot support GPU/XLA/default-readiness claims. |
| Artifact mismatch | JSON/Markdown/result/log paths and fixed diagnostic design are predeclared. |

Audit status: `PASSED_FOR_DIAGNOSTIC_RUNTIME`.  The Phase 2X blocker boundary
is cleared only for Phase 2Y CPU-hidden target-geometry localization by the
user's 2026-07-09 request to trace, audit, plan, review if possible, and execute
the plan.  This does not clear Phase 3 GPU/XLA, HMC-readiness, default-policy,
or scientific-claim boundaries.
