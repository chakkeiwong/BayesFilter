# Bounded density-fit iAPF implementation and repair

The implemented iAPF consumer now completes the eight-row GPU/XLA mechanics study. Both final iAPF scores pass all six same-input finite-difference directions. Source `cf82324143e364a485581b45c20e975c93d2593a` is frozen in `.localresources/worktrees/younis-score-iapf-precision-20260916`; the tested implementation has also been integrated into main after exact predecessor checks. This closes the scalar Gaussian mechanics prerequisite, not the iAPF scientific comparison.

The computed score is the analytical derivative of the final finite likelihood program, holding the offline fit and realized particle count fixed and treating the sampled labels as locally fixed. It is not asserted to be an unbiased model score or the derivative of the expected likelihood. The density-scale least-squares objective follows Guarniero, Johansen and Lee (2017), equation (15), with a diagonal Gaussian plus positive floor and the Algorithm 4 iteration logic. Explicit local adaptations are finite parameter bounds, projected local optimization, sample CV, finite iteration/particle caps and particle approximation of the initial distribution. The older log-quadratic fitted twist remains a separate method. No unrestricted global-minimizer or author-code-faithfulness claim is made.

## Failures and repairs

| Evidence | Finding | Repair and interpretation |
|---|---|---|
| `iapf-implementation-01`, source `7517473e` | Six baselines passed; both iAPF rows failed their first fit. | The arbitrary optimizer step cap stalled on a flat density objective. A bound derived from parameter-box extent and tolerance allowed descent without changing the objective or convergence test. |
| `iapf-implementation-02`, source `748aa461` | Eight rows completed. One fitted shape residual was 0.771 despite tiny absolute loss. | Convergence does not imply a useful fit. Bounds, relative shape error and density scale remain visible. |
| `iapf-fp32-diagnostic-01.json` | Squared-density underflow produced a zero/zero shape residual and infinite log scale; validity check rejected it. | Algebraically equivalent log-scaled arithmetic preserves the density objective and analytical gradient while making scale and shape diagnostics finite. Underflow is explicitly flagged. |
| `iapf-implementation-03`, source `c6dcb2d8` | Eight CPU rows completed; final FP64 scores differ from the previous implementation by at most 2e-15. | Healthy outputs are preserved within FP64 roundoff. |
| `iapf-integration-gpu-01`, same source | Six baselines completed; both iAPF rows failed at iteration one, projected gradient 1.788e-7 versus 1e-7 tolerance after 2,000 steps. | The failure artifact retains fit parameters, diagnostics, configuration and every seed. Derivative checks were skipped. |
| `iapf-gpu-fit-precision-diagnostic-01.json` | Exact GPU likelihood replay. FP32 reproduces the failure; FP64 on identical cast inputs converges in 17 steps with gradient 8.531e-8. | Add an explicit offline `fit_dtype` option, preserving the declared tolerance. No automatic precision fallback. |
| `iapf-integration-gpu-02`, source `cf823241` | Eight of eight rows complete in 27.817 seconds including derivative checks. | GPU FP64 offline fitting, coefficient cast to FP32, final GPU FP32/TF32/XLA filter. The controller reaches N=32 from N0=16 and stops after four iterations. |

The final two rows share the same frozen fit and use independent final streams. Exact FP32 replay errors are zero. Maximum FP32-versus-FP64 score differences are 1.4634e-7 and 1.3174e-7. All 12 central differences with h=1e-6 pass; the largest FP64 derivative discrepancy is 4.5542e-10. These checks concern the same fixed finite scalar. The comparison does not equalize adaptive fitting cost or particle counts against the N=16 baselines.

The new option records both precisions, the single coefficient-cast policy, absolute cast errors and cast validity. Unsupported or reduced fitting precision is rejected. An intentional one-step optimizer failure in `iapf-failure-observability-01` verifies that failure diagnostics survive the coordinator. The original fixed-precision failures remain preserved.

## Verification and provenance

Fifteen objective, analytical-gradient, controller and consumer checks passed in the frozen repair checkout (`iapf-precision-tests-01.log`). Both affected consumer tests passed in main (`iapf-precision-main-consumer-tests-01.log`). The snapshot commit's three required oracle-contract checks passed. Earlier 26-test coordinator/fitting and 14-test arithmetic regressions remain in their corresponding logs. No NumPy runtime or autodiff score was introduced.

The experiment plan is [the iAPF prerequisite](../../../younis-score-iapf-implementation-2026-09-15.md). Exact argv, revision, interpreter, seeds, per-row timings and runtime metadata are in each `state.json` and result file. The final command uses `run-iapf-gpu-integration.py`, `iapf-integration-gpu-precision-study.json` and output `iapf-integration-gpu-02`, from the frozen checkout with the `tftwogpu` interpreter. `CUDA_VISIBLE_DEVICES` selected RTX 4080 SUPER UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`; trusted access, TF32, XLA and verified GPU memory growth are recorded. CPU reference checks intentionally hid GPUs. Full command logs use the corresponding run names under this directory.

The initial 12-launch engineering tranche is exhausted. The explicitly allocated Phase 0E follow-through used two of three additional launches and 36.094 of 1,800 GPU seconds. The combined iAPF coordinator executions used 41 rows; 12 final derivative directions and six fixed-input fitting executions bring the diagnostic accounting to 59 of 80, with tests separately logged. The 90 CPU process-minute slice and overall eight GPU device-hour envelope are not exhausted. No further iAPF launch is needed for this mechanics result.

The manuscript now derives the profiled scale, a nonattainment counterexample for the unrestricted objective, the analytical gradients and equivalent log-scaled evaluation. The 54-page PDF built without unresolved references; rendered pages 21–23 were inspected. MathDev's saved algebra checks establish the listed local identities only; they are not a whole-filter certificate.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Complete scalar Gaussian iAPF mechanics prerequisite | Actual consumer, independent streams and derivative checks pass | Failed old precision routes preserved; explicit repaired route passes | Fit quality, adaptive-N tuning/reporting, nonlinear and dimensional coverage | Continue numerical-control calibration and remaining iAPF comparison implementation | Better model score, unbiasedness, global fit, default or HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Old FP32-only fitting is rejected on this fixture; repaired mixed-precision mechanics passes. |
| Statistically supported ranking | None. No quality-ranking experiment was run. |
| Descriptive-only differences | Fit residuals, timings, two final scores and realized particle counts. |
| Default-readiness | Not established. The precision option is explicit and the consumer remains mechanics-only. |
| Next evidence needed | Scope-aware adaptive-N selection, fresh calibration, matched-cost model-score comparisons and uncertainty intervals. |

The strongest alternative explanation for an apparently successful fit is the density objective's amplitude degeneracy: a small absolute loss can accompany poor shape. This already occurs in the retained CPU fixture. Numerical convergence and derivative agreement do not rescue that scientific weakness. The next independent prerequisite is dedicated safety and accuracy calibration of the existing LEDH controls, starting by retaining the shared correction diagnostics that the current consumer drops. Earlier nonlinear evaluation datasets 400–423 remain opened holdout evidence and must not be reused for tuning.
