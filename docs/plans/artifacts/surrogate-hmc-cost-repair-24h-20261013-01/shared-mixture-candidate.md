# Shared-covariance score contraction candidate

Question: eliminate pair-by-parameter materialization while preserving the total
derivative of the same finite Gaussian mixture. This is an algebraic rearrangement,
not a change to mixture components or covariance terms.

Let z=x-origin, m=mean-origin, P=C^-1 and H_a=P (d_a C) P.
Use matching tangent origins, so dz-dm=dx-dmean. For a Gaussian component,
d_a log phi = -(P(z-m))' (dz-dm) + (z-m)'H_a(z-m)/2 - tr(P d_a C)/2.
The derivative of logsumexp is its softmax-weighted mean of these scores.
Taking that mean and distributing products gives
 -dz'Pz + dz'P E[m] + (Pz)'E[dm] - E[(Pm)'dm]
 + z'H_a z/2 - z'H_a E[m] + E[m'H_a m]/2 - tr(P d_a C)/2.
Thus one multiplication of weights by concatenated m, dm, (Pm)'dm and m'Hm
computes every parameter direction. Covariance is shared only when the existing
call-site already declares it shared. Nonshared covariance keeps the old formula.

Audit: includes direct location, propagated particles, means, covariance and
log-determinant derivatives. Centering cancels in both primal and tangent.
No derivative of weights is added: the logsumexp differentiation identity has
already included it. Tile means are merged with their primal log masses exactly
as in the old finite mixture. Test against an independently written dense
Gaussian derivative in FP64/FP32, shifted coordinates and d=1/3/18, then full
five-model recurrence and finite-difference checks. Changed summation order
requires the owner's 1% arithmetic screen before runtime promotion.
