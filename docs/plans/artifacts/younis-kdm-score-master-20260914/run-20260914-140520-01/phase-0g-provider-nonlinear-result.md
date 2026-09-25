# Provider and nonlinear consumer execution, 15 September 2026

The current LEDH, SGQF-covariance LEDH, and mixture-covariance LEDH settings have larger observed score error than EKF in every nonlinear regime tested. This vetoes their promotion from this pilot. All 96 requested nonlinear rows completed and the numerical reference checks passed, so the result calls for scope-specific tuning and further comparisons, rather than an infrastructure repair or rejection of the research direction.

The executed implementation is frozen at a99a1c55 in `.localresources/worktrees/younis-score-nonlinear-20260915`. Its scalar model has transition `a*x+c*sin(x)` and observation mean `H*x+b*x^2`. It differentiates all six model parameters, including the initial law. A deterministic density-grid recursion supplies a numerical reference whose likelihood, score and filtering moments are checked under mesh refinement and domain expansion; omitted mass and posterior boundary mass are also checked. These checks bound observed discretization sensitivity, not the unknown exact error by a theorem.

## Evidence and decisions

Each regime uses N=32, T=4, two independently simulated datasets and two particle streams per dataset, on CPU FP64/XLA with GPU intentionally hidden. The eight methods include the numerical reference, EKF, UKF, bootstrap, locally linear importance-corrected PF, and the three shared-executor LEDH consumers. The fixed controls are mechanics hypotheses, not tuned defaults. Mean squared error below is summed over the six score components and then averaged over the four rows.

| Method | Weak curvature | Strong curvature | Concentrated observation |
|---|---:|---:|---:|
| EKF | 0.002108 | 0.573383 | 0.153645 |
| UKF | 0.001539 | 1.795163 | 0.188179 |
| Bootstrap | 0.598888 | 2.507586 | 3.251525 |
| Locally linear corrected PF | 0.140895 | 1.438627 | 0.316357 |
| LEDH | 0.188071 | 1.695467 | 0.806618 |
| SGQF covariance LEDH | 0.181740 | 1.518048 | 0.786536 |
| Mixture covariance LEDH | 0.180770 | 1.777064 | 0.792030 |

These numbers are descriptive. The independent unit for across-data inference is the dataset; repeated particle streams do not increase its count. Conditional heuristic tables and paired dataset MCSEs are preserved in each run's comparison.json. Covariance-provider changes cannot yet be ranked statistically.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Admit these implementations to further diagnostics | Actual consumers execute, all six tangents checked | No failed rows; grid checks pass | Tiny scalar scope and grid approximation | Capacity and normalization checks, then fresh tuning | Broad validity or high-dimensional readiness |
| Do not promote tested LEDH settings | Score error to declared reference | Observed loss to EKF in all three regimes | Two datasets; untuned controls | Scope-specific calibration and untouched evaluation | Rejection of LEDH, SGQF or mixture filtering |

| Inference status | Finding |
|---|---|
| Hard veto screen | No execution/numerical failures; observed heuristic losses veto promotion |
| Statistically supported ranking | None |
| Descriptive-only differences | All reported MSE and timing differences |
| Default-readiness | Not established |
| Next evidence needed | Target-specific tuning, fresh independent datasets, predeclared paired uncertainty and heuristic checks |

## Reproducibility and repairs

The exact study, command, commit, environment, seeds, data digests and runtime are in each state.json and initial-manifest.json. Row results preserve the TensorFlow/device settings and reference diagnostics. Runs `nonlinear-weak-cpu-01`, `nonlinear-curved-cpu-01`, and `nonlinear-concentrated-cpu-01` each contain 32 complete rows, taking 29.170, 26.327, and 25.854 process seconds respectively. Their run logs have the same basename. Governing plan: `docs/plans/younis-score-nonlinear-continuation-2026-09-15.md`.

Kernel tests cover the affine Kalman limit, nonlinear reference refinement and finite differences, EKF/UKF, corrected particle laws, and all three shared LEDH consumers. Actual consumer/coordinator/report tests: 17 passed. An incorrect UKF derivative keyword found in the extended tests was repaired; both moment-filter tests then passed. Two MathDev symbolic identities for the scalar covariance and its total derivative passed; `nonlinear-mathdev-checks.json` preserves their scope. This is not a certificate for the entire manuscript or algorithm.

Earlier provider work is frozen at 677e38a8 in `.localresources/worktrees/younis-score-models-20260915`: 12 affine CPU rows (20.218 s), six GPU FP32/TF32/XLA rows (30.769 s), eight-row mixture selection and two-row untouched consumption, and eight-row fitted-twist selection and two-row untouched consumption completed. An initial mixture claim dropped the candidate-family scope field and correctly failed before numerical work; the two failed attempts are retained. Restoring that field repaired consumption without changing the scientific method. GPU growth was verified on the RTX 4080 SUPER. The fitted twist is a local log-quadratic Gaussian-plus-floor adaptation, not a reproduction of the published iAPF training procedure. The mixture covariance filter is likewise a local persistent assumed-density construction.

Post-run red team: the strongest alternative explanation for the LEDH losses is untuned flow/reset controls on a tiny particle budget, while EKF is well matched to these short scalar examples. A fresh tuned comparison could overturn the candidate conclusion. The weakest evidence is the two-dataset sample and absence of target-specific tuning. Main-checkout integration remains separate because its concurrently modified executor has a different loop/callback interface; snapshot tests do not certify that revision.

Refresh: continue the 0F/0G prerequisites with bounded capacity and normalization/consistency mechanics. The remaining 56 numerical rows in this slice are assigned by phase-0g-capacity-normalization-plan.md. Scientific tuning and replication stay in the master, with their own subsequent bounded allocation inside the parent budget.
