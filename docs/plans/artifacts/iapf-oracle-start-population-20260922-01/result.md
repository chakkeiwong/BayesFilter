# Correct initialization does not repair the fitting criterion

Starting from the known KL-optimal diagonal guide does not prevent the existing
optimizer from producing poor global guide geometry. All 64 oracle-start
problems preserve the previous inputs, box, objective units, tolerance and
optimizer controls. Initial objective/gradient identities pass. Fifty-seven
meet the original convergence gate; the other seven remain explicit failures.
This is an independent R diagnostic, not a replacement filter or author code.

The clearest exact-target counterexample is terminal time3 in the d5 case:
density fitting starts at KL0.19794, lowers its objective by a factor about
182,400, and ends at KL7.17662. Its normalized sampled shape residual worsens
from0.10471 to0.68321. The optimizer converges. There is no future target at
terminal time, and the initial diagonal covariance was already KL-optimal.
Thus neither contaminated future targets nor poor initialization explains this
particular failure. Density-energy escape remains after an oracle initialization.

Normalization alone is also insufficient. At d5 time1 with the exact target,
the relative-shape optimizer converges, reduces its residual from0.0012504 to
1.4897e-5, but moves from KL0.18222 to1.96742. This computes its declared
sampled criterion correctly. That criterion and global Gaussian KL are different
quantities. The seven nonconverged cases are not used as convergence evidence.

| Target | Objective | Original convergence criterion | Largest observed final KL |
|---|---|---:|---:|
| actual | density_l2 | 16/16 | 7.17662 |
| actual | relative_shape | 13/16 | 25.6867 |
| exact | density_l2 | 16/16 | 7.17662 |
| exact | relative_shape | 12/16 | 14.4897 |

## Population calculation and finite Gaussian controls

Let X~N(0,r I_d), b(x)=exp(-||x||²/2), and let p be the same Gaussian shape
shifted by delta in coordinate one. Completing the square gives
E[b(X)^k]=(1+k r)^(-d/2). For squared-target weights w=b², the large-N limit
of ESS/N is (Ew)²/Ew²=(sqrt(1+4r)/(1+2r))^d. This is not a finite-N ESS
prediction: a realized ESS is at least one, and convergence to this limit can
require vastly more observations than1000. The finite Gaussian draws preserve
that distinction rather than failing an erroneous asymptotic-equality test.

Even when the sampling law exactly matches the target Gaussian (r=1), the
large-N fraction at d80 is6.1532e-11. In32 independent N1000 clouds the mean
realized effective count is1.590 and the mean largest weight is0.809. This
proves a concentration mechanism can exist under a correctly specified Gaussian
sampling law. It does not assert that the real iAPF clouds follow that law.
The identical-guide controls have zero shape error despite concentration;
concentration alone is not a proof that a filter must fail.

The population shape error can be evaluated independently. In the b²-weighted
Gaussian, covariance is r I/(1+2r). Since p/b=exp(delta x1-delta²/2), Gaussian
exponential moments give
S_q=1-(E_q[p b])²/(E_q[p²]E_q[b²])=1-exp[-delta² r/(1+2r)].
Meanwhile KL[N(0,I)||N(delta e1,I)]=delta²/2. For delta3 and r1e-6,
S_q=8.99994e-6 while KL remains4.5. Letting r approach zero makes S_q approach
zero at this fixed KL. Therefore no distribution-free guarantee of small global
KL follows from small normalized fitting error, even with exact population
integration. This is a counterexample, not a recommended narrow sampling law.

The matched-law shifted-guide population error is0.95021 in every dimension.
At d80, the finite-N mean sampled value is only0.57805. It is a nonlinear ratio
estimate with severe coverage error; increasing particle count might help that
specific approximation, but cannot remove the preceding population counterexample.

| d | Large-N ESS/N fraction | Mean finite ESS | Approx.99% lower | Approx.99% upper | Mean largest weight | Mean sampled shape |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 0.230048 | 232.681 | 226.983 | 238.379 | 0.0128653 | 0.945835 |
| 10 | 0.0529221 | 58.6698 | 52.8193 | 64.5203 | 0.0618909 | 0.942029 |
| 20 | 0.00280075 | 10.5351 | 7.05775 | 14.0124 | 0.29841 | 0.901252 |
| 40 | 7.84422e-06 | 2.98598 | 1.89543 | 4.07653 | 0.610806 | 0.768992 |
| 80 | 6.15318e-11 | 1.58989 | 1.24369 | 1.93609 | 0.80882 | 0.578047 |

The intervals describe Monte Carlo means from32 seeds and use a t approximation;
heavy ratio distributions limit their precision. They do not establish a filter
ranking. All r0.5 and r1/d controls, all seeds, both shifts, specialized/general
Gaussian identities and analytical narrow-law limits are preserved. No parameter
was selected from these results and no real-cloud distribution was inferred.

## Decision and inference status

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Reject initialization alone as a sufficient repair | Converged exact-target counterexamples | Seven solver failures remain explicit | Selected small cases | Preserve energy/concentration diagnostics in actual consumer | All initializations fail or original author code fails |
| Keep density normalization diagnostic only | Population counterexample and converged finite fit | No arithmetic veto | Sampling law controls practical usefulness | Evaluate repairs with downstream likelihood and oracle checks | Relative-shape minimization is a correctness certificate |
| Close Gaussian mechanism controls |36 formula identities,1152 sample records and18 limits pass | No numerical/input veto | Large-N limit can be far from N1000 | Use coverage-aware diagnostic reports before new fitting trials | Actual paper clouds are isotropic Gaussian |

| Inference status | Finding |
|---|---|
| Hard veto screen | Input/arithmetic/finite/formula checks pass; seven nonconvergences retained |
| Statistically supported ranking | None |
| Descriptive-only differences | Fixed-case KL/loss changes and Monte Carlo control means |
| Default readiness | No production change; previous filtering and TF32 vetoes remain |
| Next evidence needed | Instrument the actual fit records, then test a concrete repair against fresh downstream controls |

Terminal skeptical review PASS. The strongest alternative is diagonal-family
misspecification: a density minimizer need not be the KL minimizer. The result
states precisely that mismatch and relies on large, converged deteriorations
for its repair conclusion; it does not label a correctly minimized objective
an arithmetic bug. Unknown author initialization, local optimization and floor
choices remain. The population controls prove possible behavior and do not
estimate failure probabilities for the original paper algorithm. A changed
objective requires explicit adaptation labeling and fresh downstream evidence.

One base-R4.1.2 CPU launch, GPU intentionally hidden, one BLAS/OMP thread.
The manifest preserves exact commands, source/input snapshots, seeds, versions,
wall time, budget and all outputs. No new packages, production algorithm edits,
original-author identity or paper-replication promotion occurred.
