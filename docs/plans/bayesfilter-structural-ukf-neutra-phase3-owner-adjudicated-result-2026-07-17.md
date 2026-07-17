# Structural UKF NeuTra Phase 3 Owner-Adjudicated Result

Date: 2026-07-17

Decision: `QUALIFIED_NONCENTRAL_ONE_SEED_TRUTH_TAIL_PASS_OWNER_ADJUDICATED`

## Outcome

The final repair admitted a fixed HMC kernel with step size `0.05`, 12
leapfrog steps, trajectory length `0.6`, zero energy-screen failures, and
modern tuning R-hat `1.03240 <= 1.05`. Adaptive warm-up retained 4,000 draws
per chain and passed on its final 1,000-draw window with max modern R-hat
`1.04742 <= 1.05`. All warm-up tensors are preserved and excluded from
posterior summaries.

The retained archive contains 4,000 draws per chain across four chains, or
16,000 physical draws. Its diagnostics are:

| Diagnostic | Result | Status |
| --- | ---: | --- |
| Max rank-normalized split R-hat | `1.00668` | pass at `<=1.01` |
| Max folded rank-normalized split R-hat | `1.00156` | pass at `<=1.01` |
| Max modern R-hat | `1.00668` | pass; maximum of rank and folded components |
| Minimum bulk ESS | `971.06` (`rho`) | below original convenience gate `1000`; above owner-revised sufficiency gate `900` |
| Minimum tail ESS | `2354.53` | pass at `>=400` |
| Target/status/finite vetoes | none in the admitted verification and completed chunks | pass |

The original `bulk ESS >=1000` threshold was a conservative campaign
convenience rule, not a mathematical discontinuity. After inspecting the
4,000-draw result, the owner declared `971.06` scientifically sufficient for
this complex one-seed test and authorized pass classification. To preserve the
post-result nature of that decision, this record does not claim the original
automated `>=1000` gate passed. Under the explicit owner-revised threshold
`bulk ESS >=900`, all five parameters pass the joint R-hat, bulk-ESS, and
tail-ESS screen.

## Truth-tail result

The posterior-tail diagnostic uses
`F=(n_less+0.5*n_equal+0.5)/(N+1)` and
`p_truth=2*min(F,1-F)`. It is not a frequentist p-value.

| Parameter | Truth | Posterior mean | Empirical 95% interval | `p_truth` | Status |
| --- | ---: | ---: | --- | ---: | --- |
| `rho` | 0.8 | 0.76681 | [0.54404, 0.92807] | 0.80351 | pass |
| `sigma` | 0.5 | 0.50931 | [0.31073, 0.75452] | 0.99863 | pass |
| `phi` | 0.7 | 0.46558 | [0.07612, 0.81323] | 0.28442 | pass |
| `gamma` | 0.4 | 0.50803 | [0.13871, 0.94130] | 0.71602 | pass |
| `R` | 0.25 | 0.19625 | [0.03573, 0.40915] | 0.55053 | pass |

All five generating values lie within their empirical 95% intervals. The
minimum truth-tail value is `0.28442`, well above the predeclared `0.05`
threshold, so no second data seed is triggered.

## Evidence binding

- target signature:
  `e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`;
- transport semantic hash:
  `32a19a8d8d02f6a94851ed489d3b64f961f3bc92ba7ac36506f8078a6649c1e0`;
- tuning progress SHA-256:
  `b33bd630d4221615a9cb33165c272ab78171fea107b0d70d644146040b4bc0bd`;
- retained chunk 0 physical SHA-256:
  `7e3fac17f20c1c454b4c5b597b433cdc8168f6730bdf39f919a2332ec5043ff4`;
- retained chunk 1 physical SHA-256:
  `056a9ac2f57bc15bcdcb6d9acd89c591f9abd8f58aa7e79dea5a0b3e092e758f`;
- retained metadata SHA-256 values:
  `733200477f5e9431b48fdd8e5c9bcd78b77c4252f0e00775136f0717bc67750d`
  and
  `1ec5435c00790b47e4d8288b49e015a4b0fa4d02dd59dea20c80b108039bc991`.

## Decision table

| Decision item | Status |
| --- | --- |
| Primary criterion | qualified pass under explicit owner-revised bulk-ESS sufficiency threshold |
| R-hat veto | clear; rank, folded, and modern maxima below `1.01` |
| Tail-ESS veto | clear |
| Original bulk-ESS gate | missed by 2.9%; not represented as an automated pass |
| Truth-tail veto | clear for all five parameters |
| Main uncertainty | one noncentral synthetic fixture and post-result ESS-threshold adjudication |
| Next justified action | document the qualified pass and close the bounded structural campaign |
| Not concluded | calibration theorem, exact-filter correctness, universal reliability, superiority, or default readiness |

## Inference status

| Inference item | Status |
| --- | --- |
| Hard veto screen | no target, finite-value, R-hat, tail-ESS, or truth-tail veto in completed evidence |
| Statistically supported ranking | none; no method comparison was attempted |
| Descriptive-only differences | acceptance, runtime, posterior moments, interval widths, and individual tail magnitudes |
| Default readiness | not claimed |
| Next evidence needed | more data seeds only for broader calibration or reliability claims |

## Post-run red team

The strongest alternative explanation is fixture specificity: the learned
transport and HMC kernel may work on this noncentral synthetic data set without
being reliable across repeated structural data. A repeated-seed extreme truth
tail, energy failure, or nonconvergence would weaken this conclusion. The
weakest evidence is the post-result relaxation of a conservative bulk-ESS gate;
that is why the result is labeled qualified and owner-adjudicated rather than
an unqualified automated pass.
