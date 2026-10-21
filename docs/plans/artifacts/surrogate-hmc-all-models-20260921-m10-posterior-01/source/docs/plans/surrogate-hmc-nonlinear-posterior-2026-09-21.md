# M10 independent posterior reference

The scientific goal is testing endpoint-corrected HMC with an imperfect full
canonical DCTR force. This stage prepares the independent comparator R for the
first nominal M10 scope. Its input is the frozen claim-nominal T2 dataset from
the original M10 reference run. Reading claim observations for independent
integration is allowed; they must not select DCTR controls or HMC settings.
Stress inference remains a separate scope. The sine/quadratic law and independent
Normal(0,log(2)^2) log-noise priors remain unchanged.

Integrate the repaired two-state likelihood times the actual prior over the
parameter square [-9s,9s]^2, s=log(2). Use tensor-product Gauss-Legendre rules
with24,40,64 nodes per axis. The initial state integration uses20 Legendre
nodes per composite interval and128 normal nodes. Reevaluate all64^2 parameter
points with32/128 and32/64 state/convolution nodes. These are numerical
hypotheses tested by separate parameter, state and convolution refinements.
Zero likelihood from FP64 underflow is allowed only in the value-only integral;
it contributes zero mass and is counted. It cannot issue a gradient or authorize
an HMC endpoint. All densities and unnormalized predictive integrals must be
finite, nonnegative, and ordered between zero and likelihood.

Freeze posterior means, standard deviations and marginal .1/.5/.9 quantiles
from the64-by64,32/128 result. Compare13 functions: two standardized log-noise
means, two centered second moments, the cross moment, six marginal CDFs, and
two posterior predictive probabilities Pr(y2_rep<=.5) and Pr(y2_rep<=2).
Here y2_rep is an independent observation conditional on the same x2 and noise
scale as the second observed datum; this is a posterior replicate, not y3.
Integrate its normal CDF with the same latent posterior weights. The thresholds
are prospective fixed scales, not chosen from claim observations.

Use Legendre interpolation of each marginal at its quadrature nodes to integrate
partial CDF cells. For coefficient c_k of P_k(x), integrate P_0 as x+1 and
P_k as (P_(k+1)-P_(k-1))/(2k+1) for k>=1. A numerical test against known
uniform and normal CDFs, a monotonicity/positivity screen and parameter
refinement must pass before those interpolants define reference cuts.

For a full-prior tail bound, Normal(y;h(x),R)<=1/(sqrt(2*pi)*sigma_R)
gives likelihood times prior bounded by
C*exp(2*s^2)*Normal(lambda_Q;0,s^2)*Normal(lambda_R;-2*s^2,s^2),
where C=1/(2*pi*exp(-1.2)). Compute the mass and centered second-moment
integrals outside the square analytically. A union bound for the two coordinate
tails supplies nonnegative upper bounds; Cauchy-Schwarz supplies absolute
first-moment bounds, and abs(ab)<=(a^2+b^2)/2 bounds the cross moment.
Bound the omitted state probability by6*Phi(-10), times the same likelihood
envelope. State truncation is assessed separately from quadrature refinement.

The denominator for these bounds needs a strictly positive lower bound, not
a posterior estimate silently called exact. Construct one by restricting both
noise scales to a finite positive-prior rectangle and x0,x1,x2 to finite
intervals. Integrate x0 prior mass exactly; lower-bound both transition and
observation normal densities using their largest possible residual and the
smallest/largest scale in that rectangle. Multiply by the x1,x2 interval
lengths and noise-prior rectangle mass. Use the largest of a finite declared
set of such valid lower bounds: noise intervals [j*s/2,(j+1)*s/2] for
j=-2,...,5, latent half-widths .05,.1,.2,.4, and x0 centered at its prior
mean; x1,x2 centers are the positive observation roots, or the quadratic
vertex for observations below its minimum. Maximizing valid lower bounds
does not select a sampler or change its target.

Primary numerical qualification: the final parameter, state and convolution
normalization changes are each<=1e-6 relatively; the last parameter function change is <=half the previous
change+1e-5. For each function, twice the sum of the maximum of the last two
parameter changes, the state refinement and convolution refinement, plus
analytical tail/state allowances, must be <=one tenth of its eventual margin
(.1 for five moments, .02 for six CDFs, .03 for two predictive probabilities).
Require finite positive normalization and scales, valid inverse residuals,
probability bounds and interpolation checks. Empirical quadrature allowances
are not rigorous error bounds; the omitted-tail bounds are conditional only
on the explicitly constructed analytical evidence lower bound.

Failure is a reference repair trigger and blocks B/R or C/R using this
reference. It does not reject the surrogate-HMC direction or the separately
defined C/D sampling question. Runtime and posterior summaries are explanatory;
there is no sampler ranking or default promotion. The later heuristic set is
direct quadrature sampling, prior independence Metropolis, a Gaussian
independence proposal fitted to this frozen reference, and zero-force HMC,
evaluated centrally and in marginal tails. All heuristic results remain NOT_RUN.

CPU-only independent-reference exception, FP64 TensorFlow/XLA, GPU hidden,
threads2/1. No NumPy, pfor, learned component or runtime DCTR change. New
value-only output omits gradient fields and reuses the same quadrature algebra.
Budget:40 CPU seconds for focused tests and700 CPU seconds/600 wall seconds
for preparation including at most one localized infrastructure retry; all
charge the existing M10 allocation. Time representative values at both state
resolutions and forecast1.5 times all evaluations plus60 CPU seconds overhead
before the full ladder. Defer if the forecast exceeds the remaining allowance.
Archive all grids, source/data hashes, lower-bound inputs, exact command,
environment, seed, timings and accounting under fresh
surrogate-hmc-all-models-20260921-m10-posterior-* roots.

Skeptical audit: probe accuracy alone was insufficient, so this stage compares
integrated posterior functions over a larger domain and bounds omitted mass.
State and parameter refinements cannot substitute for one another. The proper
prior must be included once; a truncated integral must retain its omission
bound. Posterior replicates must not be mislabeled as future observations.
The claim data is used solely for reference integration. If the positive
evidence lower bound is too weak, record the qualification failure rather than
replacing it with an unjustified empirical denominator. The audit passes the
bounded stage subject to the numerical checks above.

Pre-run implementation verification: four focused tests pass, including the
previous score/local-grid tests, value-only likelihood parity, known uniform
and normal partial CDFs/quantiles, prior normalization and moments, and a finite
positive analytical evidence lower bound. Interpolation screening on8193
points permits only1e-8 absolute CDF rounding excursions or decreases and
negative density no larger than1e-5 of its maximum. These are empirical
diagnostics, not positivity proofs. Finest parameter interpolation must pass at
each of the three state/convolution configurations. Scores are absent from the
value-only return; zero-mass values cannot be mistaken for HMC score evidence.
