# SQMC repair: mathematical corrections

Date: 2026-09-25. The tested quantity is the local derivative of the finite
particle program. Its approximation to the statistical likelihood score is a
separate question.

## Gaussian model and derivative

For each component of a diagonal Gaussian let e=x-m(theta), v(theta)>0, and
let d denote a directional derivative. Its log density and derivative are

$$
\ell=-\frac12\{\log(2\pi)+\log v+e^2/v\}, \qquad
d\ell=-\frac{e\,de}{v}+
       \frac12\left(\frac{e^2}{v^2}-\frac1v\right)dv .
$$

This follows by differentiating log(v) and e squared divided by v; sum over
components for a multivariate diagonal density. Both the state and mean
contribute to de=dx-dm. The repaired transition and observation callbacks
include this full residual derivative and the variance derivative. The sampled
initial cloud carries derivatives of its initial mean and covariance.

The P44 specification uses Q and R entries as variances, whereas the diagonal
AR specification squares its q/r standard deviations. The frozen 3D target
retains its original transition and nonidentity observation matrix; dimension
alone does not identify a model. All three start from the raw initial law and
transition before the first observation, matching the executor and Kalman
reference. Direct parameter checks and comparison with the existing P44 value
fixture cover the target identity.

Every parameter direction goes through the same shared analytical executor.
The wrapper checks that all directions return the same scalar value. Fixed
random-input central differences test the corresponding local finite-program
derivative. These comparisons do not prove accuracy relative to the exact
Kalman likelihood score, differentiability at ancestry changes, or unbiasedness.

## Annealed stages and normalization

Index stages by j=0,...,K-1. Write x_j for a particle before stage j and x_{j+1}
for its image under that stage's flow. At the start of the stage the tempered
factor is transition(x_j|a) times observation(y|x_j) to the power j/K. The
destination factor uses power (j+1)/K. Consequently, with normalized incoming
weight w_j and flow Jacobian J_j, the unnormalized log weight is

$$
b_{j,i}=\log w_{j,i}
 +\log p_{\rm trans}(x_{j+1,i}|a_i)
 +\frac{j+1}{K}\log p_{\rm obs}(y|x_{j+1,i})
 +\log|\det J_{j,i}|
 -\log p_{\rm trans}(x_{j,i}|a_i)
 -\frac{j}{K}\log p_{\rm obs}(y|x_{j,i}).
$$

The increment is c_j=log(sum_i exp(b_{j,i})). The derivative follows from the
chain rule:

$$
dc_j=\sum_i \exp(b_{j,i}-c_j)\,db_{j,i}.
$$

The code accumulates c_j and dc_j. The first stage retains the incoming
normalized weights and their tangents. Subsequent stages start with uniform
weights after systematic resampling. The flow uses P/K and KR with their
matching derivatives; this is the restored finite proposal construction,
not an assertion that this choice is optimally tuned.

Stateless resampling fixes the realized indices for the local derivative.
States, transition anchors, predicted means and covariances, and all their
tangents follow the same gathered indices. The existing tests compare the
analytical result with a diagnostic autodiff oracle, including multi-stage
annealing followed by Contract-E reset and the correction. They do not prove a
likelihood-estimator theorem for the entire modified algorithm.

## Withdrawn Fisher and HMC interpretations

Fisher information is the expected score outer product,
I(theta)=E_y[s(theta;y)s(theta;y)^T], under its specified data law. A realized
component's absolute score is not an estimate of its information merely by
definition. Dividing score error by the square root of that absolute score was
therefore wrong as a Fisher interpretation. The repaired diagnostics report
absolute error, relative error, cosine, and norm discrepancy under their
ordinary definitions; ratios with zero denominators are unavailable.

Under Neal's leapfrog convention, for identical initial position and momentum,
score error e, step size epsilon, and mass M, the first half-step momentum
difference and ensuing first position difference are

$$
\Delta p_{1/2}=\frac{\epsilon}{2}e,\qquad
\Delta q_1=\frac{\epsilon^2}{2}M^{-1}e.
$$

These follow by substituting the perturbed score into the momentum half-step
and then the position step. Further trajectory differences depend on force
variation, integration time, and mass. The historical epsilon times relative
score error is not a position-error formula or a trajectory bound.
Source: Neal, equations (2.28)--(2.30),
[cached text](../plans/artifacts/sqmc-independent-audit-20260925/sources/neal-hmc-1206.1901v1.txt),
lines 397--406; the local perturbation equations above are deductions from
those updates, not a theorem quoted from the paper.

A cosine of 0.9995 corresponds to arccos(0.9995), about 1.812 degrees, not
0.05 percent accuracy. Similar vector norms do not establish vector agreement.
An interval for a paired difference that contains zero does not establish
equivalence: that requires a declared equivalence margin and appropriate
inference. A larger realized log likelihood is also not an accuracy criterion
without a target value.

## SQMC theorem boundary

Gerber--Chopin Algorithm 3 sorts the joint point rows by ancestor coordinate,
Hilbert-sorts the states, then uses the weighted inverse CDF while preserving
each row's innovation coordinates. The author-maintained
[particles implementation](../plans/artifacts/sqmc-independent-audit-20260925/sources/particles-core.py),
lines 339--355, follows that sequence. Equal weights do not turn arbitrary
Halton first coordinates into a permutation: two coordinates can land in the
same inverse-CDF bin. Deterministic rank pairing is a different operation.

The [paper](../plans/artifacts/sqmc-independent-audit-20260925/sources/gerber-chopin-sqmc-1402.4039v5.pdf),
Algorithm 3, Theorems 5--7, and Lemma 8, does not by itself prove results for
the repository's rank pairing, Contract-E reset, and higher-moment correction.
Nor does an unbiased normalizing constant imply an unbiased log likelihood or
score. No such extension theorem, global smoothness, or HMC validity has been
established by this repair.
