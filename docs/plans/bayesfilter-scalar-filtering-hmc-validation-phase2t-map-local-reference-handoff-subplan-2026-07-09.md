# Phase 2T Subplan: MAP-Local Reference Handoff Diagnostic

Date: 2026-07-09
Status: `DRAFT_PENDING_CODEX_SUBSTITUTE_REVIEW`

## Phase Objective

Convert the Phase 2S MAP-local quadratic geometry into an auditable local
reference/handoff diagnostic and decide whether the next runtime should be a
retuned fixed-kernel HMC screen.  This phase does not run HMC.  It checks that
the MAP-local handoff is internally consistent, that the old Phase 1R summaries
are not being reused as if they came from the new geometry, and that any next
HMC subplan has an exact evidence contract.

## Entry Conditions

- Phase 2R localized the Phase 2 mismatch to
  `outside_geometry_trust_region`.
- Phase 2S passed and produced:
  - MAP-local center free parameters;
  - `precision_theta` and `covariance_theta`;
  - `precision_z`, `covariance_z`, and `factor_z`;
  - explicit nonclaims.
- Native divergence remains unavailable in the prior HMC artifacts and must
  not be treated as zero.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2T harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py`
- Phase 2T tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py`
- Phase 2T JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2t_map_local_reference_handoff.log`
- Phase 2T result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-result-2026-07-09.md`
- Refreshed next subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-subplan-2026-07-09.md`
  if Phase 2T passes.

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.
- Run focused tests before runtime:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair.py
```

- Run `git diff --check`.
- Create the quiet log directory before redirected runtime:

```bash
mkdir -p docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09
```

- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 180 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2t_map_local_reference_handoff.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the Phase 2S MAP-local handoff internally consistent and specific enough to justify a reviewed retuned fixed-kernel HMC screen? |
| Baseline/comparator | Phase 2S MAP-local geometry artifact and Phase 1R old-geometry HMC summaries. |
| Primary pass criterion | The Phase 2S artifact passes schema/decision checks; MAP-local precision/covariance/factor are finite SPD and mutually consistent in both `z` and free-parameter `theta` coordinates; the map candidate has finite target replay; any old Phase 1R projected summaries are excluded from every pass/fail field; and the next subplan is a retuned MAP-local HMC screen with exact candidate list, selection policy, and veto policy, not GPU/XLA or posterior promotion. |
| Veto diagnostics | Invalid Phase 2S artifact, non-SPD or inconsistent matrices, theta/z transform mismatch, missing map-local center/factor, target replay missing or nonfinite at the map candidate, treating old Phase 1R summaries as new-geometry samples or pass/fail evidence, missing native-divergence boundary, unsupported HMC/posterior/GPU/default/source-faithfulness claim, or incomplete Phase 2U candidate/selection/veto policy. |
| Explanatory diagnostics | New local Gaussian reference mean is zero in `u_new`, covariance identity in `u_new`, old Phase 1R pooled mean transformed approximately into the new coordinate if possible, target replay values from Phase 2S, eigen summaries, and proposed HMC retuning search bounds. |
| Not concluded | No posterior correctness, no HMC readiness/convergence, no zero divergences, no sampler superiority, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2T JSON, Markdown, quiet log, result file, and next subplan draft. |

## Implementation Design

- Load Phase 2S JSON.
- Validate:
  - `decision.phase2s_geometry_centering_repair_passed is True`;
  - `initializer.accepted is True`;
  - `locator_diagnostics.accepted_optimizer_position is True`;
  - `locator_diagnostics.uses_optimizer_inverse_hessian is False`;
  - map-local handoff has finite center, scale, `precision_z`,
    `covariance_z`, and `factor_z`.
- Check matrix identities:
  - `precision_z @ covariance_z` is identity within `1e-8`;
  - `factor_z @ factor_z.T` reconstructs `covariance_z` within `1e-8`;
  - `precision_theta = diag(1 / scale) @ precision_z @ diag(1 / scale)`
    within `1e-8`;
  - `covariance_theta = diag(scale) @ covariance_z @ diag(scale)` within
    `1e-8`;
  - eigen summaries are finite, SPD, and condition number is at most `1e5`.
- Define the new reference only as:
  - `u_new` center mean: zero;
  - `u_new` covariance: identity;
  - free-parameter center:
    `center_free_parameter_values`;
  - coordinate formula:
    `free = center_free_parameter_values + scale * (factor_z @ u_new)`.
- If old Phase 1R pooled summary is transformed into `u_new`, label it
  `old_geometry_summary_projected_for_diagnostics_only`; it cannot be a
  promotion criterion and must be excluded from all Phase 2T pass/fail fields.
- Draft Phase 2U retuned MAP-local fixed-kernel HMC screen subplan only if
  Phase 2T passes.  The draft must predeclare:
  - exact candidate list:
    `(L=2, step_size=0.785)`, `(L=4, step_size=0.3925)`,
    `(L=8, step_size=0.19625)`, and `(L=16, step_size=0.098125)`;
  - each candidate has trajectory length `L * step_size = 1.57`;
  - finite sample/log-prob/log-accept gates are hard vetoes;
  - positive native divergence is a hard veto only when native divergence is
    available;
  - unavailable native divergence is not zero-divergence evidence;
  - acceptance envelope is `(0.05, 0.99)`;
  - selection policy for a follow-up longer screen is the first candidate in
    the listed order that passes all hard vetoes and the acceptance envelope;
  - if no candidate passes, Phase 2U writes a blocker/repair result;
  - no posterior-readiness claim.

## Forbidden Claims And Actions

- Do not run HMC in Phase 2T.
- Do not run GPU/XLA in Phase 2T.
- Do not treat old Phase 1R samples as samples from the MAP-local geometry.
- Do not claim posterior correctness, HMC readiness/convergence, zero
  divergences, sampler superiority, default readiness, or Zhao-Cui source
  faithfulness.
- Do not change defaults or public API behavior.

## Exact Next-Phase Handoff Conditions

If Phase 2T passes, draft and review Phase 2U retuned MAP-local fixed-kernel
HMC screen before any runtime.  Phase 2U must specify:

- MAP-local affine adapter construction;
- exact fixed grid of `(L, step_size)` values and trajectory lengths;
- predeclared candidate selection/tie-breaking policy;
- finite sample/log-prob/log-accept gates;
- acceptance envelope;
- native-divergence availability semantics;
- artifact paths, quiet log, and stop conditions.

If Phase 2T fails, write a blocker result and stop or draft a narrower handoff
repair.  Phase 3 GPU/XLA remains blocked either way.

## Stop Conditions

Stop for invalid Phase 2S artifact, matrix inconsistency, missing target
replay, unsupported claims, review nonconvergence, or any need to cross GPU,
default-policy, model-file, or source-faithfulness boundaries.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the Phase 2S diagnostic handoff, not HMC success. |
| Proxy metrics promoted | Matrix consistency and target replay only justify a next HMC screen; they do not prove posterior correctness or readiness. |
| Missing stop conditions | Matrix, artifact, review, and claim-boundary stop conditions are explicit. |
| Unfair comparison | No method ranking occurs; old Phase 1R summaries are labeled old-geometry only. |
| Hidden assumptions | The `u_new` reference is explicitly the local Gaussian coordinate induced by Phase 2S, not an exact posterior. |
| Stale context | Phase 2T loads the current Phase 2S JSON before runtime. |
| Environment mismatch | CPU-hidden artifact analysis cannot support GPU/XLA/default-readiness claims. |
| Artifact mismatch | JSON/Markdown/result/log and next-subplan paths are predeclared. |

Audit status: `PASSED_FOR_REVIEW_ONLY`.  Runtime may begin only after review
converges.
