# Batch-correlation veto: excessive rejection of independent draws

The absolute batch-lag rule rejected **454 of 512 independent Gaussian
replications (88.7%; Wilson 95% interval 85.6–91.1%)** at one or both planned
looks. This exceeds the prospective 5% false-veto tolerance. An independent
standard-library formula check passed: this is a calibration defect in the
rule, not an observed formula/implementation mismatch. It does not establish
an HMC error or an error in the batch-means formula.

The [plan](surrogate-hmc-mcse-diagnostic-2026-09-20.md) fixed 512 IID and 128
stationary AR(1) replications, fresh seeds, the actual fifteen C01 functions and
batch-statistic graph. Each replication has four chains and nested 1,000/2,000
looks, both with ten largest batches per chain. No HMC samples were generated
or reinterpreted.

| Generator | Draws per chain | Any absolute-lag veto | Any negative-lag veto | Any positive-lag veto | Any sensitivity veto |
|---|---:|---:|---:|---:|---:|
| Independent Gaussian | 1,000 | 68.95% | 67.77% | 2.54% | 0% |
| Independent Gaussian | 2,000 | 65.43% | 63.87% | 3.13% | 0% |
| Stationary AR(1), rho=.99 | 1,000 | 96.88% | 0.78% | 96.88% | 0% |
| Stationary AR(1), rho=.99 | 2,000 | 66.41% | 10.16% | 60.94% | 0% |

Conditional rates are descriptive; the IID any-look primary rate supplies the
predeclared interval-based decision. Positive and negative events can overlap.
For IID means, conservative MCSE averaged about 1.043–1.047 times exact truth;
the naive independent-draw estimate was close to truth. For correlated means,
conservative MCSE averaged only about 0.58 of truth at 1,000 draws and 0.74 at
2,000; the naive estimate was about 0.072 of truth. A dependence guard remains
necessary. Deleting the guard or retaining only its positive half is not a
calibrated replacement.

Across both looks 126/128 correlated controls triggered the veto (98.44%;
Wilson95 94.48–99.57%, explanatory only). Full R-hat/ESS qualification,
equivalence interval coverage and a replacement rule were not evaluated.
Excessive false veto means the current diagnostic often fails to answer its
question; it does not mean the procedure falsely certifies agreement.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Flag diagnostic calibration defect | IID interval wholly above 5% | Formula parity and finite checks passed | This function family and schedule only | Prospectively calibrate a replacement with dependence protection | Closed inconclusive outcomes become passes |
| Continue model/reference preparation | Independent of this rule | No target or implementation failure found | Remaining consumer/reference gaps | Prepare actual SV under its own contract | All-model validation or surrogate-HMC failure |

| Inference status | Evidence |
|---|---|
| Hard veto screen | The lag rule rejects independent exact draws excessively |
| Statistically supported ranking | No sampler ranking; the primary false-veto finding has binomial uncertainty |
| Descriptive-only differences | Per-look rates, dependence-control sensitivity and mean MCSE ratios |
| Default-readiness | No replacement diagnostic or sampler default promoted |
| Next evidence needed | Prospective coverage/dependence calibration across relevant regimes |

C01's alternative explanation now has direct support: the uncertainty rule
often rejects exact independent data. C01's separate interval-width limitation
remains. Scope is the weakest evidence: Gaussian controls do not calibrate
every posterior or rescue nonstationary chains. Completed attempts, thresholds,
targets and margins stay fixed.

Execution completed in 6.521 wall seconds, charging 10.811 CPU-worker seconds.
GPUs were intentionally hidden. The diagnostic used TensorFlow/TFP, FP64 and
CPU/XLA, plus the independent standard-library parity calculation. The manifest
preserves exact command, git revision, source snapshots, seed ranges and
per-replication records. Cumulative charges are 8,276.791 GPU seconds and
8,688.024 CPU-worker seconds under unchanged 63/15-hour caps.

Evidence: [report](artifacts/surrogate-hmc-all-models-20260920-mcse-diagnostic-01/report.json),
[manifest](artifacts/surrogate-hmc-all-models-20260920-mcse-diagnostic-01/run-manifest.json),
[accounting](artifacts/surrogate-hmc-all-models-20260920-mcse-diagnostic-01/accounting.json).
