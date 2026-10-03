# Positive-floor diagnosis: results

The local 1%-of-peak floor can nearly disable the Gaussian proposal, and this
occurs at some times even in the saved low-dimensional filters. Lowering the
floor restores Gaussian proposals but does not repair filter quality in these
diagnostics. The evidence identifies a real floor mechanism alongside a
remaining guide-quality problem; it does not identify the original authors'
floor choice or establish a superior replacement.

## Mathematical explanation

Write the transition density as f(x)=N(x;m,Q) and the frozen guide as
psi(x)=N(x;c,V)+epsilon. Their integral is

Fpsi(m)=N(c;m,Q+V)+epsilon.

The twisted transition f(x)psi(x)/Fpsi(m) is a mixture of the Gaussian
conditioned by the guide and the original transition f. Its Gaussian component
has probability

p_G = 1 / [1 + rho * sqrt(det(Q+V)/det(V)) * exp(r_S^2/2)],

where epsilon=rho*N(c;c,V) and r_S^2=(c-m)'(Q+V)^(-1)(c-m).
The determinant factor penalizes dimension and a narrow guide; the exponential
factor penalizes displacement of the predictive mean from the guide center.
At Q=qV and m=c, p_G=1/[1+rho*(1+q)^(d/2)]. With rho=.01 and q=1:

| dimension | Gaussian proposal probability |
|---|---:|
| 2 | .980392 |
| 5 | .946460 |
| 10 | .757576 |
| 20 | .088968 |
| 40 | .0000953583 |
| 80 | .0000000000909495 |

Thus the fixed peak fraction has a dimension-dependent effect even for a
perfectly centered Gaussian guide. A fixed rho=1e-6 delays but does not remove
the effect: p_G is about .488 at d40 and 9.09e-7 at d80 in this example.
This is a local derivation. GJL Section 5.1, equation 16, specifies a positive
floor to protect importance-weight tails but does not give this numerical rule.
The independent R tail rule remains an explicitly reconstructed choice.

## Checked execution and observed filter behavior

All 162 closed-form normalizer/tangent cases pass on both CPU and GPU FP64/XLA.
Maximum log-probability error is 5.69e-14; normalizer error 2.85e-14; tangent
error 2.58e-14. The negligible-floor non-harm check has zero output difference.
Independent base-R tail quantiles pass 24 inversion checks. Each graph traces
once. CPU deliberately hides GPU; GPU memory growth is verified before use.

All 19 completed previous consumers were replayed with their saved actual
draws, guide coefficients and counts, for fitted and heldout observations and
three floors: 114 evaluations. Baseline value/score replay and reconstruction
of the actual ancestor-dependent Gaussian probabilities both have zero error.
Finite outputs, CDF validity and stable traces pass. The five previous rejected
adaptive candidates remain excluded, so this is conditional on completed fits.

The table reports averages over saved particles/times/cases within each
situation; these are descriptive, not population estimates. The full conditional
rows and time-specific probabilities are in conditional-summary.json and each
case's trace. The R-tail column is the reconstructed dimension/count rule.

| d | observations | baseline mean p_G | rho=1e-6 mean p_G | R-tail mean p_G | baseline fraction p_G<.01 |
|---|---|---:|---:|---:|---:|
| 2 | fitted | .6456 | .9302 | .7466 | .3091 |
| 2 | heldout | .6405 | .9242 | .7519 | .2896 |
| 5 | fitted | .5986 | .9908 | .9972 | .2108 |
| 5 | heldout | .6127 | .9219 | .9590 | .2349 |
| 10 | fitted | .5117 | .9613 | .9998 | .3408 |
| 10 | heldout | .5108 | .9066 | .9995 | .3560 |

Holding baseline ancestors fixed, lower floors also raise mixture probabilities;
the result is not merely an effect of different later trajectories. Yet cheap
heuristic losses remain: baseline 35/38, rho=1e-6 36/38, R-tail 34/38 single-draw
evaluations. All three floors lose in every d10 fitted/heldout comparison. The
d10 heldout median absolute log error is descriptively 5.56 at baseline and
25.03 with R-tail. Those heldout guides were intentionally fitted to different
observations; this is a transfer diagnostic, not a properly refitted iAPF run.
Restored twisting can expose a poor or misplaced guide instead of fixing it.

## Decision and inference status

| decision | primary criterion | veto status | uncertainty | next action | limits |
|---|---|---|---|---|---|
| Accept the floor-suppression mechanism | Closed-form and actual-consumer probability checks pass | No arithmetic/replay/source failure | Its contribution to adaptive-fit failure has not been isolated | Retain explicit floor diagnostics in further studies | Does not identify author floor or justify changing defaults |
| Reject floor reduction as a sufficient filter-quality repair | More Gaussian proposals do not clear the conditional heuristic screen | Every arm still has heuristic losses | Two seeds, reused completed fits, guide transfer, no uncertainty-aware ranking | Compare saved guides with exact Gaussian backward messages and a diagonal-family oracle | No rejection of iAPF; no ranking of viable floors |

| inference status | finding |
|---|---|
| Hard veto screen | Formula, source, replay, finite values and CDF checks pass; previous five adaptive failures remain |
| Statistically supported ranking | None |
| Descriptive-only differences | Mixture shares, ESS, label changes, likelihood errors and heuristic loss counts |
| Default readiness | No change; lower floors are explicit diagnostic candidates only |
| Next evidence needed | Guide-geometry/oracle diagnosis, then fresh calibrated fitting and uncertainty-aware validation |

Terminal skeptical review: the mechanism is algebraic, but its importance in
paper-scale filtering is not measured here. Gaussian proposal share is not a
quality metric. Guide displacement and covariance error may explain suppression
even at small d; lowering the floor may aggravate tails. The next experiment
must use exact backward messages to separate fit error, diagonal-family error,
guide transfer and the finite initial integration. Comparing only these three
floors again would not answer that question. The prior TF32 veto, paper-model
differences and unknown original-author implementation choices remain open.

Commands, immutable launch snapshots, saved inputs, device settings, timings
and remaining budget are recorded in the launch records and terminal manifest.
