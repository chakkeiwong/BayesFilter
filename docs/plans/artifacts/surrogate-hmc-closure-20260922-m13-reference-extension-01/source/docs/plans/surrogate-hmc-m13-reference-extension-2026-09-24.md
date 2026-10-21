# M13 posterior-reference extension — 2026-09-24

This is part of the existing surrogate-HMC master. M13's independent likelihood
reference currently certifies nine points with bearing sigma <= .08. Its
two log-scale Gaussian priors have support R^2. An alias-domain restriction
cannot silently become a prior truncation. Extend the independent reference
before using it for posterior comparisons. The canonical FP32 force is unchanged.

## Evidence contract and derivation

Question: can the independent FP64 trajectory importance reference evaluate the
same observation law outside the old angular-scale limit? The baseline on the
old domain is the existing trajectory reference and its diagonal wrapped
Gaussian density. The numerical extension must agree there, normalize as a
wrapped density, match an independently long spatial sum over small and large
scales, and produce consistent angular derivatives. These are implementation
criteria; fixed-theta checks do not qualify a posterior or HMC endpoint.

For delta in [-pi,pi], wrapping a N(0,sigma^2) variable gives
f(delta)=sum_k exp(-(delta+2*pi*k)^2/(2*sigma^2))/(sqrt(2*pi)*sigma).
Its nth Fourier coefficient is the characteristic function of that Gaussian,
E exp(-i*n*epsilon)=exp(-n^2*sigma^2/2). Thus the same density equals
[1+2*sum_(n>=1) exp(-n^2*sigma^2/2)*cos(n*delta)]/(2*pi).
Differentiating these absolutely convergent sums gives the angular score.
No inference method or state transition is changed.

For sigma<=1, retaining k=-4,...,4 leaves its closest omitted alias at distance
at least 9*pi. Relative to a retained alias at distance at most pi, its
exponent gap is at least 40*pi^2/sigma^2; the two tails are geometric-bounded.
The relative truncation error is below 1e-170. For sigma>1, retain n=1,...,32.
The absolute numerator tail is at most
2*exp(-33^2*sigma^2/2)/(1-exp(-67*sigma^2/2)).
The exact density is bounded below by its spatial aliases, including at delta=pi;
at sigma=1 the Fourier numerator exceeds .036. The truncation is negligible
relative to FP64 rounding throughout the overlap. For sigma>=40 all nonconstant
coefficients are below exp(-800), below FP64 subnormal range; computation uses
the floating-point uniform limit. This is a numerical representation of the
wrapped law, not truncation of the parameter prior.

The branch point 1 and term counts are justified by these tail bounds, not fitted
to likelihood results. Cancellation near the branch and derivative consistency
are the early diagnostics. Bad scales/nonfinite quantities retain invalidity.
FP64 is the explicit independent-reference exception; FP32 remains the force
default. The existing defensive trajectory mixture and Gauss-Newton proposal
remain hypotheses requiring ESS/MCSE checks at newly evaluated scales.

## Bounded diagnostic before posterior integration

After focused CPU tests with GPUs deliberately hidden, run the existing GPU
trajectory reference with this extension at theta=(0,r), for r=0,log(2),1.5,3,4.5.
These cross the old .08 limit and the force alias cap without redefining support.
Use claim seed3701 only when conducting a declared untouched posterior claim;
this diagnostic uses existing calibration seed1701. Use 4096 trajectories,
8 independent replicates and fixed declared seeds. Compare old/new values at
the two shared-domain points with common seeds. Report full diagnostics,
likelihood-estimator MCSE, ESS and maximum weight for every point.

Numerical nonfiniteness, broken normalization/derivative checks, wrong device or
source drift veto this reference implementation. Failed proposal fitting or poor
importance efficiency triggers a proposal repair; it does not reject the model.
Old/new likelihood disagreement must be inspected against Monte Carlo error and
the common-random-number comparison. No posterior, marginal-likelihood exactness,
sampler validity, broad default-readiness, or stochastic ranking is concluded.

Reserve at most 600 CPU-worker / 600 GPU seconds within the unchanged campaign.
Use a separate master m13-reference-extension stage and a fresh versioned root
surrogate-hmc-closure-20260922-m13-reference-extension-01. The launch command will
be frozen in this note when the stage wiring and focused tests are complete.
A posterior-wide estimator and a declared endpoint remain later master work.

Skeptical audit: extending a density formula does not solve proposal efficiency
or posterior coverage. Validation uses independent identities, derivatives and
the existing narrow-domain comparator; it does not call a finite alias sum
globally exact. The largest risk is falsely interpreting a successful fixed-grid
extension as a posterior reference, prevented by explicit nonqualification.

## Exact execution and implementation audit

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-reference-extension --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-01

Master review cap180 CPU seconds; numerical worker cap420 CPU /480 GPU seconds;
total pool600 CPU /600 GPU. Fresh retry directories consume the same pool.
The trajectory factory keeps its historical default and exposes an explicit
full_support=True reference option. The option reaches the fitting objective,
angular proposal score and importance-weight likelihood. The Gaussian proposal
Hessian is still a proposal approximation; every importance weight retains the
normalized target and complete mixture denominator.

The negative observation-position score is derived as follows: delta=y_angle-h,
so d log f/dh = -d log f/ddelta. The likelihood position gradient is J' times
[(y_range-h_range)/sigma_range^2, -angular_score]. The negative-log-target
gradient subtracts that from the exact Gaussian-prior gradient. No partial
score is represented as a total likelihood derivative.

Review includes density normalization, an independently long spatial sum,
angular autodiff parity, old-domain parity, and an executable call-chain check
that forbids reaching the old restricted density. Master budget classification
uses the longest matching stage prefix so extension charges cannot be silently
deducted from the older reference pool. Data seed1701 and design-independent
stateless replicate seeds[2026092401,8301+4*point] are recorded.

Old/new common-seed maximum log-estimate difference <=1e-7 is a numerical
agreement diagnostic, not a posterior materiality criterion. Failure triggers
inspection, not a claim against the method. MCSE uses independent likelihood
replicates; log MCSE is the delta-method relative standard error, not an MCSE
for a deterministic saved likelihood. Student-t intervals with7 degrees of
freedom are approximate small-replicate diagnostics.

Skeptical audit passes for this bounded implementation question. Neither the
finite point grid nor healthy importance weights certify full posterior
coverage. Broad-noise fitting and weight concentration are expected repair
triggers and are reported even when estimates are finite.
