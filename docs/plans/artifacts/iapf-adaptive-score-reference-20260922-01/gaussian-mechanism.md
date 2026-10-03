# What adaptation can change in this Gaussian reference

This is a local derivation for the diagnostic score fitter, not the density
least-squares fit in GJL(2017), Eq15. Here a score means the gradient with respect
to the latent state used to fit a guide. It is not a likelihood derivative with
respect to model parameters.

Write the backward target, without its positive constant floor, as
`log b(x) = constant + h' x - x' J x / 2`, where `J` is positive definite.
Its state gradient is `h - J x`. Let a fitting cloud have empirical mean `m`
and positive-definite covariance `C = L L'`. The whitened states are
`z_i = L^{-1}(x_i-m)`, so their empirical covariance is exactly the identity.
The gradients in these coordinates are

`g_i = L' (h-Jm) - L' J L z_i = a - P z_i`.

Consequently the empirical affine-score regression recovers
`a = L'(h-Jm)` and `P = L' J L` exactly in real arithmetic, for every such
cloud. Transforming back gives

`V = L P^{-1} L' = J^{-1}`,
`c = m + L P^{-1} a = J^{-1} h`.

The finite-cloud requirement is full rank, which can hold with `N >= d+1`.
It does not require the points to sample the target Gaussian. The actual R
implementation additionally checks relative cloud and precision margins and
rejects an invalid fit. The independent preflight verifies this algebra against
the prior TensorFlow recursion and checks the state gradients by finite
differences. Symmetrizing the fitted precision changes only roundoff for this
exactly affine target; the positive-floor target below is generally nonlinear.

The diagonal extension replaces `V` with `diag(diag(V))`. In this linear model,
its negligible-floor recursion therefore has a deterministic analytic form:

`J_t = H' R^{-1} H + A' (Q+D_{t+1})^{-1} A`,
`h_t = H' R^{-1} y_t + A' (Q+D_{t+1})^{-1} c_{t+1}`,
`c_t = J_t^{-1} h_t`, `D_t = diag(diag(J_t^{-1}))`.

At the terminal time omit the future terms. The full-covariance recursion is
the exact backward Gaussian message; diagonalizing at each step changes that
recursion when correlations are present. With no appreciable floor effect,
repeating the score fit cannot improve this diagonal approximation by moving
the fitting cloud. A direct analytic diagonal recursion would then produce the
same guides without Monte Carlo learning. It is a useful next mathematical
control, not a newly inserted arm in the frozen phase15 comparison.

A positive constant floor changes the target gradient. Put
`s(x) = N(c_{t+1}; Ax, Q+D_{t+1})`, and let the next guide's constant be `f>0`.
The future contribution becomes

`r(x) A' (Q+D_{t+1})^{-1} (c_{t+1}-Ax)`,
`r(x) = s(x)/(s(x)+f)`.

This is nonlinear in `x`. On remote bootstrap points, `s(x)` can be tiny and
the future gradient can nearly disappear even when `f` is a tiny fraction of
the guide's peak. Fitting clouds moved toward the guide may raise `r(x)` toward
one and remove this distortion. Phase15 measures that mechanism directly using
the same-cloud negligible-floor recursion and the observed min/mean of `r(x)`
at every fitting time. Tiny floor probabilities in the final proposal alone
cannot establish this earlier fitting property.

The derivation predicts that any later benefit of adaptation here must be
attributable to floor effects, changes in particle count or numerical error;
it cannot establish that iterative fitting is needed for an exactly quadratic
log target. Stochastic final likelihood accuracy remains a separate empirical
question, assessed against Kalman and the predeclared simple filters.

The early-doubling convention also interacts with fitting speed. For an already
fixed guide, independent continuous likelihood estimates have probability
`1/q!` of being strictly increasing over `q` runs. A six-run window containing
one much worse bootstrap estimate followed by five effectively fixed-guide
estimates is therefore nonmonotone with probability close to `119/120`.
Allowing doubling at iteration five will almost always increase N in that
setting, even if variability is already small enough for the next stopping
test. This calculation assumes an effectively fixed guide and independent
runs; it is not a statement about every adaptive fitter.

GJL(2017), §5.2, reports average final counts of1000 at d5–20,1033 at d40 and
1142 at d80 (local paper text lines1070–1076). That is a useful source-side
check on a reconstructed complete method. A mismatch cannot identify the
controller alone: slower improvement by the original fitter could make its
early likelihood history more monotone. These extensions and the authors'
unknown Eq15 optimizer need not have the same early history. Thus systematic
doubling here establishes a local cost mechanism, not which convention the
authors actually implemented.
