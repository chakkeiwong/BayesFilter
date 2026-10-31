# SIR reference at the 1,482.48% discrepancy point

The old OT score is materially inaccurate relative to the model score at this point. The five-step configuration is closer, but its kappa and observation-noise components still fail the declared materiality screen. This diagnoses accuracy error; it does not establish a coding bug or posterior error.

Exact scope: fresh dataset 26101401, T=50, 18 states, nine observations, theta=(-0.0175,-0.0175,-0.0175) in the three log-scale coordinates. Saved observations and model-source hashes match. Prior gradients are excluded. Both LEDH arms use N=4032, 16 flow steps, FP64 recursion with FP32 pair computations and TF32 off. Both epsilon and reset iteration count change; their individual effects are not isolated.

| Quantity | Old epsilon=0.5, 40 steps | New epsilon=1, 5 steps | Reference | MCSE |
|---|---:|---:|---:|---:|
| Log likelihood | -1676.280681 | -1676.484031 | -1676.085221 | 0.002864 |
| Score: log kappa scale | -46.244229 | 4.421015 | 13.016542 | 1.704516 |
| Score: log nu scale | -1.325783 | -20.980306 | -24.255869 | 1.297832 |
| Score: log observation-noise scale | -13.302526 | -6.497191 | -5.990261 | 0.017494 |

## Independent references

The final reference uses 16 independent replications of 131,072 Gaussian-smoothing trajectory proposals (2,097,152 paths). Exact transition-density importance weights correct the proposal, and the analytical complete-data Fisher score targets the model score. The Gaussian approximation changes the proposal only. Finite Monte Carlo error, ratio-estimator bias and unseen importance-weight tails remain; this is not an exact oracle.

| Method | N | Replications | Log likelihood (MCSE) | Scores | Score MCSE |
|---|---:|---:|---|---|---|
| bootstrap | 131072 | 8 | -1676.049201 (0.049706) | -4.835, -11.005, -6.142 | 15.748, 4.913, 0.384 |
| bootstrap | 524288 | 16 | -1676.088148 (0.017578) | 6.584, -22.210, -5.856 | 5.474, 1.928, 0.107 |
| trajectory | 8192 | 8 | -1676.096919 (0.010380) | 12.211, -23.853, -5.945 | 4.745, 3.466, 0.083 |
| trajectory | 32768 | 16 | -1676.079994 (0.003247) | 10.850, -22.313, -5.937 | 3.647, 2.384, 0.030 |
| trajectory | 131072 | 16 | -1676.085221 (0.002864) | 13.017, -24.256, -5.990 | 1.705, 1.298, 0.017 |

Final trajectory minus complete bootstrap differences, in combined MCSE units (value, kappa, nu, noise): 0.16, 1.12, 0.88, 1.23. These references are compatible at this resolution. Both share model callbacks. Seven focused tests passed: data/coordinate checks, SIR/predator-prey fixed-state score finite differences, an exact Kalman fixture, and linear/nonlinear trajectory quadrature fixtures. The nonlinear fixture checks both likelihood and score. This does not independently verify the epidemiological specification.

Final-rung minimum trajectory ESS: 4772.5/131072; maximum normalized weight: 0.008537; replication likelihood ESS: 15.9980/16. All returned values and covariance factors were finite. Observed ESS does not bound unseen tails.

The million-particle bootstrap attempt was deliberately interrupted for variance/cost repair after two saved replications. This is not a numerical failure. Both rows and the original files are preserved; the incomplete rung is excluded from complete-rung certification.

## Materiality and uncertainty

Predeclared resolved-error screen: |arm-reference| > 0.01 max(1,|reference|) + 3 MCSE + |last same-method ladder shift|. This is a diagnostic screen, not a simultaneous confidence guarantee or posterior-impact bound.

| Score | Old absolute error | New absolute error | Threshold | New verdict |
|---|---:|---:|---:|---|
| Score: log kappa scale | 59.2608 | 8.5955 | 7.4100 | Resolved discrepancy |
| Score: log nu scale | 22.9301 | 3.2756 | 6.0790 | Unresolved; not certified within 1% |
| Score: log observation-noise scale | 7.3123 | 0.5069 | 0.1654 | Resolved discrepancy |

The 1,482.48% change divides by the old nu score near zero. That old value was itself far from the model score. This percentage therefore does not demonstrate catastrophic degradation from using five iterations. New nominal relative score errors are 66.0%, 13.5%, 8.5%; these remain uncertain, particularly nu.

Old/new log-likelihood errors: -0.195461 and -0.398811; likelihood ratios to the reference: 0.822456 and 0.671118. Both gaps exceed Monte Carlo uncertainty, but pass the literal 1%-of-absolute-log-likelihood convention (about 16.76). This convention hides differences in likelihood ratios. No criterion has been silently changed. A theta-constant additive log-likelihood error cancels from the posterior; posterior impact requires multiple parameter points.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Old score inaccurate here | Three resolved errors | Accuracy blocks promotion | Finite reference error | Preserve failed comparison | Algorithm-wide failure |
| Five-step setting closer but inaccurate | Kappa/noise resolved; nu unresolved | Accuracy blocks promotion | MCSE and unseen tails | Existing master: same-point value/score consistency, then scoped tuning | HMC or posterior validity |
| Reference subtask complete | Independent methods compatible | No input/numerical veto | No exact nonlinear oracle | Report; no additional reference launches | Global superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | No numerical/reference validity failure; six-launch diagnostic cap reached. |
| Statistically supported ranking | New score components are conditionally closer at this exact point; no cross-dataset ranking. |
| Descriptive-only differences | Nominal percentage errors, runtime, ESS and likelihood ratios; posterior impact unmeasured. |
| Default readiness | Neither configuration receives accuracy or HMC promotion. |
| Next evidence needed | Same-program value/score parity, disjoint scoped tuning, independent checks over the posterior region. |

The next repair belongs to the existing computation/HMC master. This reference subtask is complete; the larger master is not complete. Production defaults are unchanged.

Post-run red team: shared model and derivative callbacks are the strongest common-error explanation. Finite differences and independent fixtures reduce that risk but do not prove source-model faithfulness. An independent reference outside the reported uncertainty, or a same-program derivative mismatch, would change the interpretation. Kappa/nu precision and unseen weight tails are the weakest evidence.

## Execution provenance

Diagnostic worker wall time: 3124.812 seconds (0.868 hours), including interruption. Trajectory rungs: 222.860 seconds. Campaign remaining: 14.353 GPU hours.

RTX 4080 SUPER; TensorFlow 2.20.0-dev0+selfbuilt; FP64/XLA, TF32 off, verified memory growth. Final-rung allocator peak: 1,314,576,128 bytes. Commands, seeds, hashes and timings are in per-run manifests and state.json. FP64 is a reference exception; FP32/N4032 remains the owner testing direction.

Plan: docs/plans/sir-fresh-point-reference-2026-10-14.md. Executors: scripts/run_sir_fresh_reference.py and scripts/run_sir_trajectory_reference.py. Machine report: result.json. Earlier interpretation: bootstrap-only-result.json.
