# Adaptive particle-count iAPF: execution result and next phase

The selected iAPF procedure now executes through calibration, validation, repository-issued selection and independent final sampling, with explicit offline and final costs. This closes the adaptive-count implementation prerequisite in master Phase 0E. It establishes no score-accuracy ranking or default promotion.

Source `1c12eefa2ed55b81c4336fca4c006b9877c2df5f` remains frozen in `.localresources/worktrees/younis-score-iapf-adaptive-scope-20260916`. Seven implementation/test files were integrated into main after verifying every preceding byte against `cbcfea1e`. Backups and checksums are in `main-before-iapf-adaptive-scope-01/` and `iapf-adaptive-scope-main-integration.json`. Unrelated changes were preserved.

## Checked evidence

The corrected GPU run completed ten rows, fourteen iAPF recursive fits and two log-quadratic comparator fits. All executed filter and fit kernels report one trace and verified GPU memory growth, FP32/TF32/XLA filtering with explicit FP64 iAPF fitting. One validation history was `[16,16,16,32]`, so an actual count change was exercised. The selected procedure finished at N=16. Its two final replicates used the same fitted coefficients and disjoint final sampling streams. Observations may enter the prescribed offline fit; oracle scores, final sampling noise and heldout errors may not.

The computed derivative holds the fitted coefficients, realized N and discrete labels fixed. It is the corresponding finite-program derivative, not a derivative through adaptive fitting and not an unbiased marginal model score. Starting N is never presented as the realized count or as equal total cost.

Evidence: `iapf-adaptive-scope-gpu-02/run-manifest.json`, `consumer-evidence.json`, `selection.json`, ten saved row results, and `iapf-adaptive-scope-verification.json`. The latter verifies all saved result digests, the driver checksum and source revision. Exact commands, seeds, selected controls, hardware, timings and study paths are preserved in the manifest and the [phase plan](../../../younis-score-iapf-adaptive-scope-2026-09-16.md).

Attempt 01 completed six iAPF rows and fourteen recursive fits, then failed comparator validation: Kalman inherited the particle estimator label. This was a driver metadata failure before comparator numerical execution. The original driver and all results remain in `iapf-adaptive-scope-gpu-01/`. Correcting that label passed the four comparator validations; attempt 02 used unchanged scientific settings and frozen kernels.

CPU checks: 23 focused tests, eight after the final validation change, three mandatory commit oracle checks, and eleven main consumer tests passed. Logs and timing files use the `iapf-adaptive-scope-*` prefix. A sandbox commit failed before numerical execution and the trusted retry succeeded. Total measured CPU process time is 219.98 seconds. The allocation closes at 2/3 GPU launches, 46/60 charged row/recursive-fit attempts and 52.191/1800 GPU wall seconds; no further adaptive-scope launch is needed.

## Decision and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Integrate the adaptive-count consumer | Actual selection and final execution pass | No unresolved implementation veto; driver failure repaired | Tiny scalar Gaussian fixture | Extend the shared kernels to the existing scalar nonlinear fixture | Better model-score estimation |
| Preserve the selected settings as mechanics settings | k=1, permissive tau=100 gives a bounded demonstration | No default promotion attempted | Stopping and optimizer controls are not scientifically calibrated | Include target-specific tuning in later comparisons | Universal settings or equal-cost fairness |

| Inference status | Finding |
|---|---|
| Hard veto screen | No unresolved failure in the checked adaptive-count consumer |
| Statistically supported ranking | None |
| Descriptive-only differences | Final iAPF squared score errors 0.1188 and 0.3207; bootstrap 0.3118, fixed twist 0.2795, log-quadratic fit 0.1357, Kalman 0 on the same fixture |
| Default-readiness | Not established |
| Next evidence needed | Nonlinear implementation, full control calibration, independent replicated conditional comparisons and measured total-cost matching |

Kalman is exact here, so these particle procedures cannot clear a claim of useful improvement over all applicable cheap heuristics on this fixture. That is a scientific promotion veto, not a rejection of the nonlinear research direction. The strongest alternative explanation for descriptive differences is Monte Carlo noise. Two final samples cannot establish a ranking; a replicated oracle comparison could overturn their apparent ordering.

## Between-phase repair and refresh

The next prerequisite is nonlinear iAPF through the same Gaussian-plus-floor normalizer, twisted sampling and density fitting kernels. A nonlinear transition mean does not invalidate Gaussian convolution when transition noise remains additive Gaussian. The nonlinear observation enters the fitting target and importance weights directly. Bind model coefficients and the actual cast observation data to the consumer evidence; preserve the FP64 physical-data identity separately. Check zero-curvature parity, nonlinear density identities, frozen-fit derivatives, selection scope and the real GPU consumer before using this lane in a scientific comparison.

Comprehensive LEDH control calibration, broader horizons and particle counts, normalization and consistency studies, combinations, matched-cost replication and terminal review remain in the master. The whole master is incomplete.
