# Phase 2Y Result: Target Geometry Localization

Date: 2026-07-09
Status: `PASSED_DIAGNOSTIC_PROPOSAL_FAMILY_MISMATCH_INDICATED`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2Y target-geometry localization passed as a diagnostic and found no affine/proposal replay bug | Passed: finite target values and scores were recorded for all predeclared anchors and rays; artifact bug indicated `False`; proposal-family mismatch indicated `True` | Final vetoes: `[]` | Diagnostics are anchored to Phase 2W/2X failed proposals and fixed rays, not exhaustive posterior exploration | Draft and review a proposal-strategy pilot or local-reference-abandonment subplan before any new agreement attempt | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered diagnostically: failures look like proposal-family/global-tail mismatch rather than an affine orientation or proposal-log-density replay bug. |
| Baseline/comparator | Phase 2S MAP-local center, Phase 2W/2X top-weight anchors, antithetic partners, and fixed rays in `u_new`. |
| Primary criterion | Passed: JSON/Markdown artifacts were written with finite target/score diagnostics and nonclaims preserved. |
| Veto diagnostics | No invalid artifacts, nonfinite target/score, invalid ray construction, orientation mismatch, or proposal replay mismatch. |
| Explanatory diagnostics | Anchor norms, target deltas, score norms, ray profiles, target-minus-quadratic residuals, and proposal log-density replay were recorded. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2y_target_geometry_localization.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization.py`
- Review bundle:
  `docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase2y-hypothesis-plan-review-bundle-2026-07-09.md`

## Hypothesis Outcomes

| Hypothesis | Phase 2Y status | Interpretation boundary |
| --- | --- | --- |
| H1: local quadratic trust region exceeded | `supported_descriptively` | Top-weight anchor norms were all well outside the Phase 2S trust radius `0.6`; this motivates proposal redesign but does not certify posterior shape. |
| H2: tail/ridge undercoverage | `supported_descriptively` | Ray and score diagnostics show high proposal tension; this is explanatory only. |
| H3: orientation/scaling mismatch | `not_supported` | Adapter replay matched the row-vector formula to `4.440892098500626e-16`; the Phase 2S display string remains ambiguous but did not indicate a runtime transform bug. |
| H4: proposal log density correct, family poor | `proposal_density_replay_passed_family_mismatch_plausible` | Saved Phase 2W/2X proposal log densities replayed with max absolute delta `0.0`; proposal family mismatch remains plausible. |
| H5: local, not global MAP locator | `plausible_not_certified` | Top anchors have target values below the center but far above the local Gaussian quadratic expectation. |
| H6: quadratic extrapolation failure | `supported_descriptively` | Target-minus-quadratic residuals along rays reached max absolute `104.96420467633178`. |

## Key Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Anchor count | `33`: center plus Phase 2W/2X top-weight anchors and partners | Fixed diagnostic design |
| Top-weight anchor norm range | `[3.1676400712527686, 8.91162729981821]` | Explanatory support for H1 |
| Phase 2S trust radius | `0.6` | Prior geometry-local fit scope |
| Top-anchor target delta from center | min `-6.422949248307816`, max `-0.3769510875147901`, mean `-1.8449613168467351` | Explanatory; no MAP certification |
| Anchor score norm range | `[2.4416858704074592e-11, 63.79285939037887]` | Hard finiteness plus explanatory geometry |
| Ray target-minus-quadratic abs max | `104.96420467633178` | Explanatory support for H6 |
| Orientation replay max abs error | `4.440892098500626e-16` | Bug-localization diagnostic |
| Proposal log-density saved replay max abs delta | `0.0` | Bug-localization diagnostic |
| Artifact bug indicated | `False` | Phase 2Y decision field |
| Proposal-family mismatch indicated | `True` | Phase 2Y decision field |

## Review And Boundary Record

Claude review was attempted through the local review gate but was blocked by
the approval layer as external transfer of private repository planning and
diagnostic context.  A fresh local Codex substitute review was used instead and
is weaker than full Claude material review.

The user's 2026-07-09 request cleared only the Phase 2Y diagnostic-localization
boundary.  It did not clear Phase 3 GPU/XLA, HMC-readiness, default-policy, or
scientific-claim boundaries.

## Checks

| Check | Status |
| --- | --- |
| Phase 2Y subplan skeptical audit | Passed for CPU-hidden diagnostic runtime |
| Claude review gate | Blocked by approval layer; no workaround attempted |
| Codex substitute review | `AGREE_FOR_DIAGNOSTIC_RUNTIME`; weaker than Claude |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py` | Passed before runtime: `20 passed` |
| `git diff --check` | Passed before runtime |
| Phase 2Y runtime command | Exited `0`; artifact decision passed diagnostic gate |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `52ee244498988e046a6356f926003b581103083b` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 300 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seeds | `N/A`; deterministic replay of saved proposal artifacts |
| Wall time | `54.949073967000004` seconds |
| Plan/result paths | Master, Phase 2Y subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for diagnostic artifact validity. |
| Reference validity | Not assessed; Phase 2Y does not build a reference. |
| HMC-reference agreement | Not assessed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Anchor norms, target values, scores, proposal log densities, ray profiles, and quadratic residuals. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed proposal-strategy pilot, transport/local mixture path, or local-reference-abandonment decision. |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | The finite Phase 2Y rays may miss other important target regions; they localize the observed Phase 2W/2X failures but do not exhaust the posterior geometry. |
| What would overturn | A later artifact showing affine/proposal replay mismatch, or an independent proposal/reference replication that passes fresh ESS gates without reusing Phase 2Y anchors for validation. |
| Weakest evidence | The diagnostic is based on top weights from failed proposals and fixed rays, not an exact reference or long-chain validation. |

## Final Nonclaims

- No new valid importance reference.
- No HMC-vs-reference agreement.
- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
