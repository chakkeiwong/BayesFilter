# Final factor precision symmetry result

The shared final fitted precision now explicitly computes `(P + P.T) / 2`
after its Cholesky solve. The inner L-BFGS objective/gradient and covariance
parameterization are unchanged. Exact-input GPU replay04613 now returns
error0 and usable `consensus_diagonal_consensus` geometry; before04607 returned
stability error2. Exact-input CPU04614 remains usable with the same selected
family. This repairs the diagnosed final representation error; whole-consumer
and cross-backend record equivalence remain separate.

CPU04611 and GPU04612 diagnostic trials pass the saved CDF and analytic/scaled
matrix checks, independent inverse/residual checks, analytical derivative
comparison and indefinite-input rejection. GPU04615 and CPU04616 each pass
52 runtime regressions. Readback/policy04617 passes161 checks. All12 saved
fits preserve covariance, statuses, flags, anchors, domain counters, optimizer
counters and loss at the predeclared bounds. All returned factor precisions
are exactly symmetric. The largest GPU entry correction is3.81446e-12;
CPU precision and the complete public CPU record are unchanged.

The before GPU result has no usable public record, so its repaired status is
not an equivalence claim. The repaired GPU and CPU complete records still have
4539 differing leaves under the strict controlled-fitter comparison. Those
differences are preserved in04617/factor-precision-symmetry-qualification.json;
this repair closes the localized final inverse representation defect only.

| Backend | Before/after runs | Fit seconds before/after | Host peak MiB before/after | Device peak bytes before/after |
|---|---|---:|---:|---:|
| CPU | 04609 / 04614 | 43.142 / 45.768 | 2884.578 / 2907.262 | N/A |
| GPU | 04607 / 04613 | 62.614 / 62.855 | 2783.090 / 2789.211 | 2,306,304 / 2,306,304 |

These are single fresh-worker compilation/fit observations, not repeated warm
timings. The rejected GPU before result cannot establish a performance ranking.
The observed repair adds no device allocator peak and only a modest host-peak
difference in this fixture; native retention and actual-consumer memory still
need their declared lifecycle checks.

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| Commit minimal final-precision repair | CPU/GPU trials,52 regressions each and161 readback/policy checks pass | No changed optimizer, covariance or condition cap;4539 cross-backend leaves still differ | Renew actual-consumer source and lifecycle evidence | Cross-backend equivalence or optimizer convergence |
| Preserve earlier numerical failures | Before GPU rejected at unchanged symmetry guard | No tolerance/condition-cap relaxation | Retain raw matrices and exact-input hashes | Old-record equivalence |
| Keep main unmerged | Broader master remains unfinished | Source-renewed consumer/lifetime and terminal audit pending | Continue the master queue | Whole-program completion |

Post-run review: exact saved operand hashes remove the earlier regenerated-cloud
confound. The inner optimizer is untouched in source and the saved optimizer
fields pass comparison. Projection could conceal inaccurate input matrices in
general; here the independent inverse/derivative trials and existing invalid
input regressions remain passing. No arbitrary caller input is symmetrized
before validation. The weakest evidence is complete cross-backend/consumer
equivalence, which remains open. No independent reviewer was used in this unit.
