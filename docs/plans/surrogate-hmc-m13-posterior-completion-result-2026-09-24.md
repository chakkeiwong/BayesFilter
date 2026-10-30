# M13 posterior-reference completion result — 2026-09-24

The M13 finite-box posterior diagnostic is now complete on the predeclared
129×129 nested Simpson grid. The run filled the 386 cells left by the CPU
budget stop in refinement-04 and reused the 12,030 preserved cells. All
16,641 grid nodes are finite. This supplies a resolved mean comparison at the
declared box and does not qualify an unbounded posterior or an HMC endpoint.

The scientific target was the finite-box posterior induced by the M13
calibration likelihood, the independent `N(0, log(2)^2)` prior on
`(lambda_Q, lambda_R)`, and the declared box
`[-4 log(2), 4 log(2)]^2`. The worker computed the same finite-box integral by
positive composite Simpson quadrature, using the independent wrapped-density
importance reference at every node. This computes an estimate of the
declared finite-box diagnostic, with importance and discretization error. It is not the full `R^2` posterior because no
omitted-tail bound has been established.

The fine-minus-coarse changes for the eight reported metrics were
`[-2.9740e-6, 2.0759e-6, 2.4662e-6, -3.4748e-6, -1.3872e-7,
-2.8076e-6, 1.6765e-6, -1.2600e-6]`. The corresponding paired grid
MCSEs were at most `1.91e-8`, and the changes in posterior means measured in
fine-grid posterior standard deviations were `[-8.56e-6, 1.30e-5]`. The grid
mean changes are 0.017 and 0.052 times the marginal importance MCSEs.
They exceed their much smaller paired difference MCSEs, so the grid difference
is resolved but small. The outer-region probability changes by 1.26e-6
(about 7% of its small value); grid resolution is not uniform across metrics.
The new-cell minimum importance ESS was 9.8836 and the largest normalized
weight was 0.3152. Those tail diagnostics are too weak to certify the
importance reference as an unbounded posterior oracle, even though every node
is finite.

The fine-grid diagnostic reports means `(0.1894866, 0.1430383)`, standard
deviations `(0.3472615, 0.1591393)`, covariance `-0.0207766`, mean noise
standard-deviation multipliers `(1.2802389, 1.1683287)`, outer-box probability
`1.7893e-5`, and finite-box log evidence `27.27435` with importance MCSE
`0.00475`. These are descriptive values for the finite-box diagnostic. No
surrogate-HMC chain, endpoint comparison, acceptance decision, or posterior
ranking was run from them.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Grid completion | 16,641/16,641 finite nodes; complete nested 129 grid | No node validity veto | Omitted tails and low worst-cell importance ESS | Localize the independent M13 correction failure before endpoint work | No posterior-oracle or HMC qualification |
| Grid refinement | Mean changes are small relative to marginal MCSE; paired differences are resolved | None at the grid-resolution level | Importance-sampling MCSE and tail coverage | Preserve this grid as the finite-box reference | No claim that the box equals `R^2` support |
| Endpoint admission | Not run | Endpoint evidence absent | No chain/reference comparison | Repair or explain M13 FP32 correction failures, then run declared endpoint | No acceptance, R-hat, ESS, or surrogate-HMC correctness claim |

| Inference status | Result |
|---|---|
| Hard veto screen | Passed for finite-node validity; endpoint screen not run |
| Statistically supported ranking | None; no candidates were compared |
| Descriptive-only differences | Fine/coarse quadrature differences and finite-box summaries |
| Default-readiness | Not assessed; FP32 remains the owner-approved default with saved-run discrepancy as a warning |
| Next evidence needed | Correction-failure localization, an explicit omitted-tail bound or an accepted finite-box target, and then endpoint posterior comparison with declared HMC diagnostics |

The strongest alternative explanation is that the finite-box result is biased
by omitted posterior mass or by low-ESS importance cells rather than by grid
resolution. A tail-bound calculation, independent reference seed, and a
particle-count replication would overturn that interpretation. The weakest
part of the present evidence is the tail importance estimate, not the nested
grid comparison.

## Reproducibility

Executed command:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-posterior-completion --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-05
```

The run manifests preserve commit, environment, seeds, source hashes and
verified GPU memory growth. The reference used 4,096 particles and 8 replicates
with seed `[2026092402, 9101]`.

- Plan: `docs/plans/surrogate-hmc-m13-posterior-completion-2026-09-24.md`
- Run report: `docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-05/m13-posterior-completion/report.json`
- Summary: `docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-05/m13-posterior-completion/summary.json`
- Campaign report: `docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-05/report.json`
- Status: `M13_POSTERIOR_REFERENCE_COMPLETION_COMPLETED`
- New evaluations: 386; preserved evaluations: 12,030; valid nodes: 16,641
- Numerical route: FP64 independent reference, GPU/XLA, `full_support=True`
- Endpoint qualification: `false`; posterior-reference qualification: `false`
- Accounting charge: 50.909737 CPU seconds and 27.032487 GPU seconds

