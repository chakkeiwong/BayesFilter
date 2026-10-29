# Mode discovery and author-based concurrent flow training

Benchmark-first update: the user has deferred direct q20 testing. Follow
`bayesfilter-neutra-warm-start-canonical-benchmarks-2026-09-29.md` for the
initial analytic and author-benchmark ladder. References below to the q20
target describe the later application, not the next experiment.

Updated after the user's instruction to borrow Gabrié's implementation.
The first implementation should adapt the authors' `flonaco` concurrent
sampling/forward-training procedure to the existing canonical IAF. Source
has been acquired and checked; the port and its equivalence tests have not
been implemented. No mode search, sampler or training experiment has been
launched by this note. A converged posterior archive is not a prerequisite
for training initialization.

The earlier fixed importance-cloud proposal is preserved below as a separate
optional mechanism. It is no longer the first implementation proposal and
must not be presented as Gabrié's procedure.

## Question and scope audit

Can mode-representative initialization followed by the authors' concurrent
sampling and forward training produce a useful canonical IAF while allowing
the walkers to correct their initial basin proportions?

The target remains the current four-free-parameter UKF posterior. Resolve
the current evaluator's identity and numerical-domain behavior before a
fresh run; do not silently substitute the generative-model SMC² target or
an old target closure. The architecture remains the configured author-based
IAF. A later pure or mixed reverse-KL refinement remains a separate local
hypothesis; it is not a compulsory stage of the selected author procedure.

Skeptical audit: optimizer endpoints are neither posterior draws nor region
mass estimates; optimizer attraction frequencies are not posterior weights;
height cutoffs can discard broad important regions; stationary points can
be saddles; sampling a mixture at random does not ensure that every component
is represented; conditional redraws alter the proposal law; and a good
forward warm start can lose coverage during subsequent reverse KL. The
construction below addresses these issues explicitly. No requirement for a
precise posterior archive is introduced.

| Role | Requirement |
|---|---|
| Initial-stage pass | Valid local candidates, represented candidate neighborhoods, finite target values/scores, evaluable flow density and checked selected sampler operations |
| Promotion criterion | After training, independently assessed geometry and useful downstream estimation under the existing NeuTra rules |
| Promotion veto | Missing important tested region, invalid density/gradient, wrong target or failed downstream numerical/estimation checks |
| Repair trigger | New basin discovered, ineffective global proposals, poor within-region fit, or loss of a represented region during training/refinement |
| Continuation veto | Invalid target/harness, unavailable required check, corrupted evidence or exhausted run-specific allowance |
| Explanatory only | Optimizer counts, peak heights, Hessian/Laplace approximations, raw ESS and training loss |
| Not concluded | Every mode found, posterior masses known precisely, canonical map trained successfully, or final HMC unnecessary/necessary |

## First implementation: reuse flonaco's selected author route

Use the MIT-licensed author source at
<https://github.com/marylou-gabrie/flonaco>, pinned to
`6b9286b4e58194aa65373200d7bfacde06a2d180`. Exact source anchors, the licence,
verification record and adaptations are documented in
`bayesfilter-gabrie-flonaco-source-adoption-2026-09-29.md`.

Initialize persistent walkers across the located basins, apply flow-based
global Metropolis moves and the author's corrected local MALA steps, then
train on detached, refreshed configurations. Keep terminal walkers for the
next update. The selected source options are `fwd` and `mhmalangevin`;
the common example uses the distinct unadjusted `mhlangevin` option.
The finite-stage loss fits the current walker distribution. Its basin
proportions need not be correct initially, and finite adaptive training is
not automatically posterior-converged.

Borrow this sampling/training procedure through an explicit TensorFlow
adaptation while retaining the shared canonical IAF. Upstream RealNVP and
example hyperparameters are not new local defaults. Source-equivalence
checks must establish update order, detached gradients, acceptance formulas
and persistent state before target-specific calibration. The authors'
implementation supplies neither a general mode finder nor a guarantee that
unrepresented modes will be discovered.

## Discovery and local proposal construction

The concrete published starting point is the mode-search part of Pompe,
Holmes and Latuszynski (2020), JAMS Algorithm 3, Sections 2.2.1–2.2.3 and
Supplementary Section 11. It uses uniform or prior starts, parallel BFGS,
first/second-order optimality checks and curvature-scaled merging. This is
separate from Gabrié's sampler/trainer. We have inspected the paper; a JAMS
author-code implementation has not been acquired or validated. Reusing a
TensorFlow Probability optimizer implements the numerical solver, not the
complete JAMS algorithm or its sampling guarantees.

For the first analytic benchmark tests, supply basin representatives to
isolate training. Then test the discovery procedure with benchmark mode
locations withheld from its starts. Low-dimensional reference grids plus
independent refinement can assess missed basins; exact mixture component
centers are useful representatives but are not generally the exact modes
of an overlapping mixture.

Use dispersed independent prior starts plus a separate space-filling design
in prior-standardized coordinates. Previously located points may nominate
additional starts only after fresh evaluation against the current target.
Independent starts can run concurrently; their counts, seeds, worker limits
and target-query allowance belong to the priced executable plan.

Use the target in the declared training coordinates. Affine standardization
only contributes a constant density Jacobian; a nonlinear unconstraining
transformation changes density modes, so include its Jacobian and label the
coordinate-specific target explicitly. On synthetic targets without a prior,
declare an independent broad start law or a search box, and examine wider
boxes/start scales separately. The low-dimensional grid is an evaluation
reference, not hidden information supplied only to one training arm.

Optimize the same log posterior used by training. Keep non-finite failures
and incomplete searches in the record. Reevaluate endpoint scores and local
curvature, distinguish saddles and unresolved ridges from local maxima, and
deduplicate candidates with respect to local scales and numerical uncertainty.
Do not assume there are exactly two modes or reject a candidate solely by
its log-density gap. For a nonsingular isolated mode, inverse negative
curvature can nominate a local proposal shape; it is not a posterior
covariance estimate. Ridge/width choices require declared calibration.

With `U=-log_gamma`, an interior strict candidate satisfies a numerically
small `grad U(m)` and a positive-definite actual Hessian `H=Hessian U(m)`.
A BFGS approximation maintained positive definite is not a test of actual
curvature. Near-zero eigenvalues leave the classification unresolved; retain
such ridge/degenerate candidates for investigation rather than inventing
positive curvature. Constrained boundary cases need the appropriate KKT
conditions. Gradient and curvature tolerances must reflect coordinate scales
and evaluator accuracy measured on the reference fixtures.

The JAMS duplicate heuristic is

    d_ij^2 = (m_i-m_j)^T (H_i+H_j)/2 (m_i-m_j).

Its merging threshold is a numerical hypothesis. Calibrate it from repeated
convergence to the same reference mode and check that genuinely distinct
reference modes remain distinct. Tighten/refine uncertain endpoints and
retain ambiguous pairs instead of treating a loose geometric threshold as
a proof of basin identity. Keep one accurately evaluated representative
per identified basin and its start provenance, score, curvature and cost.

Run independent batches of starts and track the number of newly discovered
regions as a function of work. For a deterministic optimizer and iid starts
from nu, let b be the probability under nu of its attraction basin for a
particular mode. Then

    P(miss that basin after M starts) = (1-b)^M,
    M >= ceil(log(delta)/log(1-b_min))

gives miss probability at most delta if `0<b_min<1` and the assumed lower
bound b>=b_min is valid. For at most K basins with that bound, replace delta by delta/K
for a union bound. This reasoning does not apply directly to dependent or
space-filling starts, and posterior mass is not b. Validate start-law
choices on the known benchmarks; on an unknown target, no-new-mode rounds
and wider starts are stopping diagnostics, not a proof that all modes have
been found. Record the unresolved coverage risk and search budget.

For Gabrié initialization, spread walkers around every retained basin using
declared local initial shapes or short corrected local chains. The inverse
Hessian is only a local shape hypothesis; check validity and actual region
representation. Equal allocation may deliberately give wrong initial basin
weights, which the combined sampler must subsequently correct. Do not
assign posterior weights from optimization frequencies, peak heights or
inverse Hessians. For SMC, instead sample an evaluable normalized proposal
and retain its appropriate importance correction as described below.

For the optional importance-cloud mechanism, construct normalized local
proposal densities q_j near the discovered
candidates and retain a prior component q_0=r for support. Gaussian or
Student components are useful broad choices. Drawing from such a component
guarantees its allocation, not membership in a particular basin. If the
design requires a fixed number inside specified neighborhoods, use directly
samplable normalized local proposals supported there, such as uniform
ellipsoids. The ellipsoid density is 1/(V_d |det A_j|) on
theta=c_j+A_j u, ||u||<=1, with V_d=pi^(d/2)/Gamma(d/2+1). Its scale and
overlap must be checked against the actual local geometry. An arbitrary
rejection/redraw rule is not allowed without its conditional density.

## Optional mechanism: fixed allocations and correct weights

This is the earlier static importance-cloud proposal, distinct from the
selected concurrent author procedure. Its mathematical correction is
preserved for possible later use; it is not required to initialize flonaco
walkers or a replacement for its sampling/training loop.

After freezing the pilot proposal, allocate n_j>0 draws to each component,
including the prior, and let N=sum_j n_j and alpha_j=n_j/N. Define

    m(theta) = sum_j alpha_j q_j(theta).

For independent theta_ji~q_j, evaluate the full mixture denominator and use

    w_ji = gamma(theta_ji)/m(theta_ji),  gamma=r L.

Fixed component allocations do not invalidate this correction. For an
integrable f,

    E[(1/N) sum_j sum_i w_ji f(theta_ji)]
      = sum_j alpha_j integral q_j(theta) gamma(theta) f(theta)/m(theta) dtheta
      = integral gamma(theta) f(theta) dtheta.

The normalized ratio remains a finite-sample approximation. Neither alpha_j
nor an optimizer hit count is a posterior probability. For disjoint regions
A_k, an initial mass estimate is sum_ji W_ji 1(theta_ji in A_k), where
W_ji=w_ji/sum w. The same weights define forward pretraining through
-sum_ji W_ji log q_psi(theta_ji).

Start by checking whether these direct importance weights are useful. Full
SMC is optional. If they are too concentrated, the same proposal can seed
posterior annealing with initial weights r/m at beta=0, followed by L^DeltaBeta
increments and correctly targeted mutation. Direct insertion of optimized
centers with unit weights is not that algorithm.

Preserve the weighted cloud rather than unnecessarily resampling away a
small represented region. Stratified training minibatches may oversample a
region with the corresponding selection-probability correction. If a
deliberately balanced, uncorrected initializer is studied, name its changed
training distribution explicitly; it is not forward KL for p.

## Optional refinement after initialization

Use the weighted cloud to initialize the canonical IAF, stopping before
expensive precision fitting or overfitting to a finite dependent cloud.
Pure reverse refinement uses
E_phi[log phi(z)-log|det DT_psi(z)|-log gamma(T_psi(z))], up to the constant
log Z; phi is the standard Gaussian base. The rough sample no longer defines
that objective. Preserve this as one testable refinement hypothesis.

The subsequent literature audit found a directly relevant comparison in
Noe et al. (2019), Supplementary Figure S2: pure energy/reverse refinement
could collapse after ML pretraining, while combined example and energy
training avoided that failure on the paper's double well. Include mixed
refinement as the source-grounded alternative rather than treating a zero
forward coefficient as established. No coefficient is adopted here. A
retained forward term based on biased, fixed examples changes the training
objective; it is not automatically KL(p||q). Refreshed target-corrected
examples and a fixed biased regularizer must be distinguished. See
`bayesfilter-neutra-mode-initialization-literature-2026-09-29.md` for the
paper and archived author-code anchors.

Save the forward checkpoint and evaluate coverage throughout refinement
without forcing final masses to equal noisy initial estimates.

Region probabilities can be represented coarsely at this stage. Unknown-mode
coverage and rough probabilities are different uncertainties. For iid
optimizer starts from nu, a particular attraction basin of probability b is
missed with probability (1-b)^M; b is not its posterior mass and is normally
unknown. Fresh search rounds and wider starts provide diagnostic evidence,
not a certificate of exhaustive discovery. A mode search also does not
characterize every extended ridge or posterior tail.

The 1000-point base score probe remains a geometry diagnostic. Add checks
from independently generated points in the discovered physical regions so
that loss of a region cannot hide behind the map's own sampling distribution.
Final posterior convergence/precision is assessed during actual estimation,
not imposed as a prerequisite for this approximate initialization.

## Local implementation findings and next execution boundary

Read-only inspection found
`docs/benchmarks/run_ssl_lstm_q20_gap_closure_mode_discovery_2026_08_18.py`
and its focused tests. That historical harness uses a fixed old target
signature and geometry file, prior/corner starts, local L-BFGS endpoints,
two known-region comparisons, and an inherited peak-height cutoff. It is not
a current implementation of the corrected initialization above and was not
rerun. The old result itself records that requiring a converged physical
archive before NeuTra training was a circular planning choice.

`inference/sequential_map_covariance.py` provides other local geometry
machinery but imports NumPy in its runtime path. Its presence is not an
eligible new TensorFlow implementation or a checked consumer call chain.

Before execution, the smallest useful engineering checks are target identity
and valid scores, known-mode fixtures that distinguish saddles, controlled
author/TF sampler and forward-gradient equivalence, and correct canonical
IAF handoff. Test mixture-weight identities only if the optional static cloud
is implemented. Cost pricing then supplies start/walker counts, optimizer
and query caps, transitions per update, CPU/GPU allocation, a unique output
directory and the total attempt/time allowance. No historical numerical
constant or earlier campaign allowance is promoted by this design. No extra
review or approval token is required by the local research policy.

Companion sources: the derivation and references in
`docs/chapters/ch26d_multimodal_recovery.tex`, and the component/test design
in `docs/plans/bayesfilter-smc-library-integration-and-tests-2026-09-29.md`.
This note changes no executable source, training default or existing result.
