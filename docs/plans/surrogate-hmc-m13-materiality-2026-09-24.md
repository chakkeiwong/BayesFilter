# M13 saved-run materiality analysis — 2026-09-24

Owner decision after this analysis: accept the saved-run discrepancy as a
warning and retain FP32 as the default. The
[implemented policy and validation](surrogate-hmc-fp32-warning-policy-2026-09-24.md)
preserve these diagnostic results and remove the repeatability batch as a
required next step.

The owner's question is whether the observed FP32 replay changes matter for
testing surrogate HMC. This is post-run analysis of existing evidence, within
the master program. It launches no filtering, tuning, or HMC run.

## Evidence contract and skeptical audit

Compare the saved N256 FP32 consumer, the repaired replay, and the independent
whole-trajectory importance reference, on the same nine T20 noise settings.
Calculate uncertainty from the reference's 16 independent likelihood estimates
at N65536. Report both pointwise changes and changes in log-likelihood
contrasts, which remove a constant offset.

These are explanatory diagnostics, not a new promotion or continuation veto.
A materiality threshold must concern the intended posterior or proposal use;
neither the old 3e-5 limit nor reference MCSE alone supplies that threshold.
Preserve the old regression result. Do not retrofit a passing tolerance.
Invalid input provenance or mismatched points invalidates this analysis.

The MCSE of the likelihood mean is s_L/sqrt(16). First-order propagation
through log gives MCSE(log(mean(L))) approximately s_L/(4*mean(L)).
Use stabilized likelihood weights to avoid overflow. This matches the stored
coefficient of variation divided by four. It is not the standard error of the
mean of the replicate log estimates, and it is not uncertainty in the fixed
DCTR approximation. One fixed scrambled-Halton design cannot estimate
variation across independent designs or systematic approximation error.

All comparisons are descriptive and post hoc. The two largest reference
likelihoods among nine grid points are an illustration, not a validated
posterior region, fitted MLE, or a new selection criterion. The prior and
full-support posterior reference remain undeclared. A large force-surrogate
potential error does not itself demonstrate failure of an HMC kernel using a
separately correct endpoint target.

Default audit: R=16 and N=65536 are inherited from the qualified reference,
not newly tuned. Delta-method uncertainty is justified by the small measured
coefficient of variation; it remains an approximate sampling-error estimate
and does not measure bias. Different parameter points use distinct saved
reference seeds. Repeating a frozen DCTR evaluation would test computational
repeatability, not supply independent integration estimates.

No new heuristic contest or method ranking is performed. Previously checked
reference-law and validity checks remain the prerequisites. A possible
misleading result is a tiny shift at two chosen points hiding larger shifts
elsewhere; therefore preserve all nine rows and the largest pairwise change.

Budget: at most 5 CPU seconds for standard-library postprocessing; zero GPU.
Artifact: artifacts/surrogate-hmc-m13-materiality-20260924-01.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/plans/artifacts/surrogate-hmc-m13-materiality-20260924-01/analyze.py


## Results

The recomputed reference means and coefficients of variation match the saved
values. Input hashes and all nine parameter points match. The reference
log-likelihood MCSE ranges from 0.00080248 to 0.00114449. At nominal noise it
is 0.00099393, corresponding to a 0.099393% relative standard error in the
likelihood mean. The nominal likelihood estimate is 1.81035983e13 and its
MCSE is 1.79937004e10. These uncertainty estimates describe the independent
integration reference, not variation across fixed-design DCTR runs.

| Process / observation SD multipliers | Reference log likelihood | Log MCSE | Replay minus saved | Shift / reference MCSE | Replay minus reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 / 0.5 | -20.48895689 | 0.00080248 | -0.04585266 | 57.14 | -192.64080385 |
| 0.5 / 1 | 27.95144087 | 0.00081688 | +0.12027359 | 147.24 | -51.00298887 |
| 0.5 / 2 | 26.92550575 | 0.00114449 | -0.02050018 | 17.91 | -12.68124497 |
| 1 / 0.5 | 1.54157049 | 0.00081947 | +0.06980896 | 85.19 | -35.88930304 |
| 1 / 1 | 30.52713184 | 0.00099393 | +0.00975800 | 9.82 | -7.62711505 |
| 1 / 2 | 26.34322724 | 0.00082798 | -0.00144386 | 1.74 | -1.10815415 |
| 2 / 0.5 | 19.22682663 | 0.00088814 | +0.00170803 | 1.92 | -5.30003926 |
| 2 / 1 | 30.58225264 | 0.00093428 | +0.00730324 | 7.82 | -1.62906661 |
| 2 / 2 | 24.14571173 | 0.00084107 | -0.00098991 | 1.18 | -0.11075957 |

The two highest-reference-likelihood grid points have process/observation
standard-deviation multipliers (2,1) and (1,1). Their log-likelihood contrast
changes by -0.00245476 between saved and replayed DCTR evaluations, a likelihood
ratio factor of 0.99754825, or a 0.2452% decrease. The independent reference
contrast MCSE is approximately 0.00136410. Across all nine points the largest
absolute log-ratio change is 0.16612625, corresponding to a factor of
1.18072216 (18.0722%). The worst pair is different from the illustrative
highest-likelihood pair. No posterior support region has been established.

A separate, much larger approximation issue exists before and after this
replay. For the same high-reference-likelihood pair, the reference log ratio
is 0.05512081 but the replayed DCTR log ratio is 6.05316925: their ratios
differ by a factor of 402.64225. This is a checked shape discrepancy at two
points, not a constant offset and not an MCSE-scale effect. It is not new
evidence caused by the observability edit. It does not by itself invalidate
surrogate-force HMC with a separately correct endpoint target; it would be
material if the finite DCTR likelihood were asserted to equal that target.

## Decision and interpretation

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Treat the old saved-run tolerance as an engineering diagnostic | No posterior criterion evaluated | Provenance and numerical reference checks pass; old regression failure retained | Practical effect over the posterior and on proposals is not measured | Within the existing master, evaluate materiality using log-ratio/force effects and the declared downstream posterior criteria | Neither blanket acceptance nor rejection of surrogate HMC |
| Do not use reference MCSE alone as the materiality threshold | Precision is measured, utility tolerance is not | No new scientific veto | MCSE can shrink with more reference effort while practical impact is unchanged | Keep a separate scientific tolerance and include MC uncertainty when testing it | Detectable difference does not automatically mean material harm |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No new validity failure in postprocessing; old saved-number regression remains failed |
| Statistically supported ranking | None; deterministic discrepancies are scaled by independently estimated reference precision, not subjected to a two-sampler hypothesis test |
| Descriptive-only differences | All nine shifts, MCSE ratios, and the selected pair/full-grid contrast changes |
| Default readiness | No new force, endpoint, tuning, posterior, or default qualification |
| Next evidence needed | Posterior-relevant likelihood/force effects and actual surrogate-HMC diagnostics under the existing master; multiple independent designs if DCTR randomization variability is wanted |

Post-run red-team: selecting two favorable points would hide the 18% contrast
change elsewhere; both are reported. Treating the reference MCSE as DCTR
error would confuse integration uncertainty with approximation error. The
remaining weakness is absent posterior support and trajectory evidence.
A downstream test could show the FP32 change immaterial even when detectable,
or show a force/acceptance effect missed by pointwise value comparisons.

Execution used only standard-library CPU postprocessing of saved results:
0.018487 CPU seconds, 0.018855 wall seconds, zero GPU. No new random draws or filter/HMC execution.

Reproduction code, full numeric table, and provenance:
- artifacts/surrogate-hmc-m13-materiality-20260924-01/analyze.py
- artifacts/surrogate-hmc-m13-materiality-20260924-01/report.json
- artifacts/surrogate-hmc-m13-materiality-20260924-01/run-manifest.json
