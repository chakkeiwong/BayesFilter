# M21 fixed-count estimator diagnosis

The completed controller confirmation has 400 observations per case. In the
rho=0.995 dispersed-start case, fixed-count lugsail intervals cover the true
mean in 293/400 replications, while exact Gaussian intervals cover their known
transient expectation in 373/400. The pooled transient mean is zero by symmetry.
This triggers the estimator diagnosis already required by M21; it does not
authorize changing a stopping threshold or declaring a new default.

Use all preserved fixed retained tensors from this case, and the rho=0.98
stationary comparator, to compare the existing lugsail batch lengths 100, 250
and 500 at n=10000 per chain. The first is the existing sqrt(n) baseline;
250 is an explicit intermediate diagnostic hypothesis; 500 is the largest
batch length retaining the existing minimum of 20 complete batches. Keep
r=3,c=0.5 and the same pooled mean. Also compute the existing autocorrelation
MCSE comparator. No new trajectories or thresholds are selected from outcomes.

The question is whether the discrepancy is consistent with finite-batch bias
or remains when a longer bandwidth is used. Check the exact expected batch
estimate from the Gaussian covariance before interpreting the sample estimates.
For a stationary AR(1), let
V(b)=b+2 sum(k=1..b-1)(b-k)rho^k. With a=n/b complete batches, the expectation
of the usual per-chain batch-means long-run variance estimate is
[V(b)-V(n)/a^2]/[b(1-1/a)]. For deterministic starts, the covariance additionally
subtracts rho^s rho^t; its contribution to the demeaned batch quadratic form
and the nonconstant transient mean must both be included. Derive those terms
from the expansion in the existing fixed_mean_law fixture; do not apply the
stationary expectation silently to a deterministic-start case.

Primary engineering checks are independent dense-covariance agreement on tiny
arrays and complete accounting of all 400 fixed arms. Coverage, empirical
MCSE/exact fixed-count SE ratios, and expectation/true-variance ratios are
explanatory diagnostics of the frozen baseline. Report exact pointwise binomial
intervals with their denominators. Comparing several bandwidths on this evidence
is development only: a favorable result nominates a fresh confirmation arm and
cannot establish a ranking, coverage guarantee or default readiness.

Reserve 900 CPU worker-seconds including focused checks, within M21's existing
64800-second allocation. Use hidden GPUs, one TensorFlow intra/inter-op thread,
no overlap beyond two numerical workers, and a new estimator-diagnosis-r1 root.
Record code hashes, exact tensor-input hashes, commands, environment and wall
time. Preserve every original controller result and source snapshot.

Skeptical audit: a correct lugsail formula need not be adequately calibrated at
a short bandwidth. More batches can reduce variance while retaining bias; fewer
batches can trade bias for high estimator variance. The exact finite-count
variance differs from asymptotic integrated autocorrelation time. This diagnosis
checks those distinctions and uses no model truth to alter an HMC candidate.
An unsupported alternative remains a hypothesis even if its coverage rises.
