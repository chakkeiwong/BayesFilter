# What these numerical guesses change

The paper defines an auxiliary guide at each time as a diagonal Gaussian
density plus a positive constant. The constant and the fitting procedure affect
the proposal distribution and its importance-weight variance. They do not
replace the observation likelihood in the importance correction. With the
checked fresh final run, changing either choice therefore preserves the
ideal-arithmetic likelihood target, conditional on the learned guides and
selected particle count. See the preceding d40 confirmation likelihood audit
for the derivation; this experiment does not re-prove finite-precision accuracy.

The particle controller is another substantive choice. The printed stopping
condition is l>k, while the first nonnegative-index doubling window exists at
l=k. The old implementation can double before it is allowed to stop. The new
alternative waits until l>k for both decisions, and checks stopping first.
This is an explicit reconstruction of unspecified early behavior. It does not
change k, tau, the final independence condition or the APF weight equations.

Equation (15) fits density values, whereas the successful optional R path fits
log densities. These are different optimization problems. For density vector
p and backward target vector y, profiling the scale gives

    lambda = (p' y)/(y' y)
    L = p' p - (p' y)^2/(y' y).

Making a Gaussian very diffuse or moving its center far from the finite cloud
can drive every entry of p toward zero and hence L toward zero. This is the
checked obstruction to interpreting the equation as an unrestricted global
fitting prescription. It does not show that every local optimizer fails.

This experiment therefore also tried a compact local neighborhood of the QR
initialization: each mean coordinate may move by one initial standard deviation,
and each variance may change by a factor of two. On calibration d10, the
optimizer returned convergence code zero at backward time 99, but an imposed
bound was active. The absolute mean loss fell from 3.700644e-5 to 3.529118e-9;
the scale-free residual also fell, from .033519 to .011114. It would be wrong
to describe this as optimizer nonconvergence or to infer that its proposals
necessarily perform poorly. It failed the narrower, predeclared requirement
that this local reconstruction not depend on an arbitrary active constraint.
Its full fit and cloud are preserved for a possible future bounded-fit study.

The log-density quadratic fit instead retains an intercept, so changing the
Gaussian amplitude cannot by itself eliminate a shape discrepancy in that
objective. This is why the QR reference can be useful even though it is not
the paper's equation-(15) implementation. Better-looking empirical variability
would establish neither author-code identity nor statistical superiority.

All performance comparisons here use new simulated observations. The paper's
tables condition on its own five observation sequences, which are unavailable.
Even 1000 new replicates cannot identify a missing solver or uniquely recover
the published table. Agreement is evidence for practical reproduction of a
behavior, not proof of the unknown author's numerical choices.

Two reporting cautions matter. First, Table 2 calls its entries average
resampling counts without explicitly documenting how the iterative fitting
passes are counted. This comparison uses the final APF count, interpreting the
table in the context of the final likelihood estimates. Count differences are
therefore explanatory, not proof of an incorrect implementation. Second, a
failed importance sampler can report nearly zero likelihood ratios in every
small-sample repetition, giving a misleadingly tiny observed variance. Its
variance alone cannot outrank an accurate estimator. The heuristic screen uses
mean squared error relative to Kalman, with ordinary and large-innovation
observations assessed separately. Terminal variance comparisons require the
corresponding mean-accuracy and numerical checks before a ranking is defensible.
