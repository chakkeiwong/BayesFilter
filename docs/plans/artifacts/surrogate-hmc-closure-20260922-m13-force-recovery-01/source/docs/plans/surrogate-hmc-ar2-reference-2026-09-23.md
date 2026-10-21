# AR(2) reference stage of the existing surrogate-HMC master

This is existing-roadmap work after structural canonical integration. The
question is whether a precise independent model posterior is available for
testing endpoint-corrected surrogate HMC. It does not test a different method
or revise LG1's accepted judgment.

Use the declared structural fixture: F=[[.45,-.15],[1,0]], B=[1,0]',
Q=.09 exp(2 lambda_Q) B B', R=.04 exp(2 lambda_R), H=[1,0], x0~N(0,I).
The two lambda parameters have independent N(0,log(2)^2) priors, as already
specified in the structural scope audit. The first concrete scope is T=10,
transition then observation. Three independently seeded simulated datasets
(calibration, validation, claim) are frozen once at lambda=(0,0). Observations
are rounded to FP32 once and promoted exactly to FP64 for every comparator.
T=10 is an initial scope hypothesis, not evidence for longer horizons.

The Kalman endpoint integrates the declared singular-innovation model exactly
up to floating-point arithmetic. Check it against an independently assembled
joint Gaussian covariance of all ten observations, over interior and two
opposed noise-scale cases. Check the endpoint's total automatic derivative by
finite differences. This is a reference-gradient exception, never the DCTR
analytical score. The reference has no process-covariance Cholesky or ridge.

Compute the two-dimensional posterior with nested trapezoid grids (64,128,256
intervals on [-4s,4s]^2; 384 on [-6s,6s]^2, s=log 2). If needed, extend the
same protocol to 512 intervals on [-4s,4s], 768 on [-6s,6s], and 1024 on
[-8s,8s] within budget, preserving
every result. The mesh widths in the domain comparison match. Tail expansion
is an empirical check, not a rigorous posterior-tail bound. Report that limit.
Record means, standard deviations, all standardized second/cross moments,
10/50/90% marginal CDF cuts, and next-observation probabilities above 0 and .2.

Primary reference criterion: roundoff-level Kalman/joint Gaussian agreement,
relative endpoint gradient error <=2e-6, finite normalized densities, and
empirical quadrature/domain allowances below one tenth of the existing practical
margins (means .1 SD, moments .1, CDF/predictive .02). Final mesh change must
contract, allowing a 1e-5 floating/reference floor. Nonfinite values or a failed
independent likelihood check veto the reference. Insufficient mesh or domain
precision triggers the declared refinement, not a rejection of surrogate HMC.

Skeptical audit: compare the same initial law and timing; the old mechanics
fixture's smaller cloud covariance is not the inferential prior. Normal priors
remain unbounded despite finite reference integration. The reference-only stage
cannot qualify DCTR, tune HMC, establish convergence, or rank samplers. Kalman
and direct joint Gaussian are constructed exact comparators; bootstrap/no-flow
and ordinary Kalman-gradient HMC are retained for subsequent conditional method
comparisons. No weak heuristic is used to certify this reference.

Execution: closure master `ar2-reference` stage, fresh structural-pass output
directory. Reserve up to 300 CPU-worker / 600 GPU-wall seconds per numerical
attempt within the existing structural pass's 2,400 CPU / 3,600 GPU total;
three attempts including prescribed refinements. CPU tests use a 180-second
allowance. Use the existing tftwogpu environment, GPU/XLA batch-native Kalman
calculations, verified memory growth and trusted access. Save each dataset,
grid tensors, score checks, reference summary, source hashes, environment,
seeds and cost in the stage directory. There is no new campaign allocation.

Next, already in the master: freeze scope-specific DCTR controls using only
calibration/validation evidence, qualify the finite endpoint, and use the public
position-only-force HMC tuner before retained B and C chains. Continue if the
prerequisites and remaining budget permit; report the precise reason otherwise.
