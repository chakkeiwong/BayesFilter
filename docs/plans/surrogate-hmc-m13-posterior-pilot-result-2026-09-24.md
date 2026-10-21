# M13 posterior-reference pilot result — 2026-09-24

77 focused tests passed. All1089 nodes in the33x33 grid were valid.
The calibration posterior on this finite box has provisional log-noise means
lambda_Q=.18469626 and lambda_R=.14526493, with SDs.34684522 and.15148346.
Importance MCSE of the means is.00017104 and.00003688. The fine-minus-coarse
differences are.01272255 and-.04432008; the coarser grid is visibly inadequate.
No posterior estimate is promoted from this localization result.

The outer-region mass is2.0437e-5; it is not an omitted-tail bound. Minimum
per-node importance ESS10.26 and maximum normalized weight.3093 occur over
the whole broad grid; interpret their location before judging posterior-wide
accuracy. Log-evidence27.29264769 has importance MCSE.00474078 conditional on
this quadrature. Eight replications give descriptive uncertainty only.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue nested refinement | All likelihood points valid; grid unresolved | No missing/invalid cells | Quadrature error and omitted tails |65/129 grids on same data and seeds | Continuous model-posterior or HMC correctness |

| Inference status | Finding |
|---|---|
| Hard veto screen | No integrator validity failure at these nodes |
| Statistically supported ranking | None |
| Descriptive differences | Grid sensitivity and importance MCSE above |
| Default readiness | Not assessed; FP32 DCTR default unchanged |
| Next evidence needed | Resolved grid, independent replication/sample size, tail control, endpoint and HMC |

Post-run red team: common seeds can make resolution comparisons smooth while
leaving Monte Carlo bias or missed modes unresolved. The changed grid mean is
far larger than its importance MCSE, so more particles alone would not fix
this observed problem. The corrected reference remains independent of DCTR.

Actual master command and manifests are in
artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-03.
Numerical stage36.041498213GPU seconds; total60.743162CPU-worker seconds.
Cumulative43594.301904292CPU/34336.315550209GPU. Remaining global
10405.698095708CPU/192463.684449791GPU; reference pool494.187540CPU/545.148222130GPU.
Next action is the existing master's reference refinement04, not a change in
research direction or a stop awaiting approval.
