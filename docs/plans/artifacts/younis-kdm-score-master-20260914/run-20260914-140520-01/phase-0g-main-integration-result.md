# Shared main-checkout consumers: execution result and refresh

The main checkout now executes the completed score-study consumers through one shared analytical LEDH executor. The integration preserves the concurrent native flow-substep loop and its initial-tangent checks, while retaining callback evidence, diagnostic traces, persistent covariance providers and nonlinear observations. The isolated composite is frozen at `6a32740c`; file hashes, the original main executor and overwritten score files are preserved in this directory. Other agents' edits were not reverted. Main HEAD advanced independently to `509871fd` during this work, so the numerical manifests and source hashes identify each execution.

Validation: 22 isolated consumer tests, six protected/native-loop parity cases at one, two and four substeps, and 29 main-checkout consumer/coordinator/report tests passed. Parity covered affine and nonlinear observations and all four value/tangent outputs. Each diagnostic kernel traced once and retained a native While operation. These are engineering checks, including six-parameter finite-direction checks, not evidence that a finite-program derivative equals the model marginal score.

The first GPU launch completed seven of nine rows in 40.246 seconds. Both KDM consumers failed with `InaccessibleTensorError`: the integration copy had omitted their already-repaired callback modules, leaving Python list side effects across the new time loop. This was an integration error. After verifying the two main files against the original baseline, the protected callback implementations were restored. Both main-checkout KDM finite-direction tests passed in 20.97 seconds. The focused GPU retry completed both rows in 19.704 seconds. The failed attempt and all seven successful rows are retained; no report has been regenerated under a changed source fingerprint.

The combined GPU coverage comprises Kalman, bootstrap, locally adapted PF, canonical LEDH, SGQF covariance, persistent mixture covariance, fitted Gaussian-plus-floor twist, integrated KDM and resampled KDM. The two versioned runs use the RTX 4080 SUPER, FP32, TF32 and XLA with verified memory growth, N=16 and T=3. Controls are explicitly unpromoted mechanics settings. Ten of the parent limit of twelve GPU launches have been consumed. This integration slice used eleven attempts, nine successful numerical rows, about one GPU minute and less than five CPU process-minutes for the recorded integration checks.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Main integration passes its bounded engineering contract | Shared-loop parity and actual consumer checks pass | Original KDM callback failure repaired and independently retested | Larger dimensions, horizons and settings remain scope dependent | Continue the nonlinear tuning repair with fresh partitions | No score superiority, default readiness or whole-program completion |
| Retain the nonlinear candidate promotion veto | All previously tested LEDH settings lost descriptively to EKF in all three regimes | Numerical reference checks passed; underperformance concerns the candidates | Two datasets and untuned settings cannot settle method quality | Calibrate separate proposal scopes and evaluate untouched data | No rejection of the LEDH research direction |

| Inference status | Finding |
|---|---|
| Hard veto screen | All nine consumers now have successful GPU mechanics evidence; the earlier failed launch remains incomplete |
| Statistically supported ranking | None |
| Descriptive differences | Prior nonlinear score errors and normalization associations only |
| Default readiness | Not established |
| Next evidence needed | Scope-specific calibration, untouched conditional heuristic comparisons, paired uncertainty, larger-budget replication |

Strongest alternative explanation for the nonlinear losses is tuning or finite-N error, rather than an inherent limitation of the covariance provider. A fresh scoped comparison can distinguish that explanation; another successful integration smoke cannot. The weakest part of the present evidence is its tiny statistical and model coverage. The master remains incomplete: published-objective iAPF, broader models and horizons, calibrated consistency-based combinations, matched-cost replication and terminal mathematical/scientific review remain open.

Evidence: `main-integration-gpu-01/`, `main-kdm-repair-gpu-01/`, `phase-0g-integration-file-audit.json`, `phase-0g-integration-callback-repair.json`, `phase-0g-flow-loop-parity.json`, and the corresponding integration and repair logs. The companion result is `phase-0g-capacity-normalization-result.md`.
