# HMC methods for BayesFilter, MacroFinance and dsge_hmc

Literature and source review, 7 October 2026. This report recommends an order
of investigation; it does not change the tuning protocol or report a new
sampler experiment. The official exposition remains [the book](../main.tex)
and the operational entry points remain in the
[HMC interface reference](../reference/hmc-tuning-interface.md).

The most useful near-term combination is **validated whitening, randomized
trajectory HMC with SNAPER adaptation, and cheaper evaluations of the existing
state-space target**. MAMS deserves a subsequent, serious comparison. LAPS is
a promising option when many chains can run efficiently together, but its
headline parallel gains and its warm-up diagnostics need more qualification
than a reading of the abstract suggests. None of these methods removes the
need to establish the target, its gradient, or posterior accuracy.

This is a targeted survey of developments within approximately 2016–2026,
covering 18 papers and relevant author implementations. It is not an exhaustive
systematic review. Sources were inspected beyond their abstracts: numerical
dynamics, corrections, adaptation, experimental designs, and the assumptions
in relevant theoretical and appendix sections. The bibliography below records
the examined versions and limits of that inspection. All recommendations are
engineering hypotheses for these repositories, not measured local performance
rankings.

**First distinguish what is expensive and what is difficult.**

An HMC run can be slow because each target/gradient evaluation is expensive,
because trajectories require many evaluations, because the retained chain
mixes poorly, or because compilation, adaptation and orchestration dominate.
These causes require different repairs. A useful accounting is

\[
T_{\rm total}=T_{\rm compile}+T_{\rm initialization/map}
 +T_{\rm tuning/verification}+T_{\rm retained}.
\]

Within each phase, measure target/score time, integration time, diagnostics and
host overhead. For a batch of chains, measure the actual batch latency; do not
multiply a scalar time by the number of chains or assume perfect parallelism.
NUTS has irregular trajectory lengths and tree construction, which can be
awkward for synchronized accelerator execution [Hardware, ChEES]. That is a
reason to test a simpler trajectory schedule, not evidence that TensorFlow's
NUTS implementation is the dominant cost on these particular models. This
review did not profile the consumers.

The sampled dimension also matters. Sampling structural parameters after
integrating out a long latent path is a different problem from sampling the
whole path. A representative MacroFinance MIDAS adapter expects seven
parameters, although other MacroFinance models are larger. A stochastic
volatility benchmark with thousands of sampled latent states does not establish
a speedup for that seven-parameter likelihood.

The inspected MIDAS path is
`daily_asset_midas_gap_closure_r1.py::_batched_value_score_fn` and
`daily_asset_midas_gap_closure_r1_posterior.py::likelihood_log_prob_and_grad`.
It differentiates the batched, masked, correlated-noise Kalman likelihood.
The inspected BGS description in dsge_hmc's
`docs/experiments/bgs/2026-09-28-longer-evidence/README.md`, lines 20–21,
instead declares an endpoint target `T_source_ukf97` and a deterministic
position-only proposal field **not asserted to be its exact gradient**.
These are representative source constraints, not an executable audit of every
current consumer or a claim about their present runtime.

An exact sampler for a UKF-based posterior samples that declared approximate
posterior. It does not make the UKF likelihood equal to the nonlinear model's
exact likelihood. Keep filter approximation error separate from sampler error.

| Method or improvement | Problem it addresses | Applicability and adoption effort | Recommendation |
| --- | --- | --- | --- |
| Randomized trajectory length, then SNAPER | Resonance, expensive or poorly chosen trajectory lengths | Installed TensorFlow Probability source; still needs integration with frozen-kernel evidence and target/score adapters | First sampler candidate |
| Whitening/noncentering and frozen NeuTra | Correlation and position-dependent scale | Existing project direction; map quality, Jacobian and inverse must be checked | Continue alongside sampler work |
| Parallel Kalman filtering / target-score optimization | Expensive state-space evaluations | Installed TFP parallel LGSSM implementation; correlated noise and component masks need an equivalent formulation | First performance investigation for applicable models |
| MAMS | Gradient efficiency and microcanonical exploration | New TF kernel and a non-volume-preserving MH correction | Next major sampler candidate |
| LAPS | Fast movement from initial states using large ensembles, then adjusted sampling | New ensemble adaptation plus MAMS; measured batch scaling essential | Conditional later candidate |
| Adaptive MALT | Resonance and inefficient persistent dynamics | Author JAX/FunMC implementation; TF port needed | Alternative if residual problems justify it |
| MEADS | Short generalized-HMC transitions and ensemble adaptation | Specialized momentum/slice state and folded adaptation | Lower initial priority |
| Pathfinder | Initialization and approximate geometry | Stan author implementation; new local adapter/implementation | Compare as an initializer if that phase is costly |
| Delayed-rejection GHMC | Residual variation in appropriate step size | Correct reverse proposals and potentially large extra gradient cost | Specialist option after whitening |
| Particle-MALA / Particle-mGRAD | Sampling long latent state paths | Different conditional target; author code is JAX | Separate investigation when latent paths are sampled |

**What MAMS contributes.**

Robnik, Cohn-Gordon and Seljak's *Metropolis Adjusted Microcanonical
Hamiltonian Monte Carlo* [MAMS] changes the dynamics, not just the selection of
epsilon and the number of leapfrog steps. Write the target as
\(\pi(x)\propto e^{-U(x)}\), with dimension \(d>1\). Its microcanonical
position and unit-velocity dynamics are

\[
\dot x=u,\qquad
\dot u=-\frac{(I-uu^\top)\nabla U(x)}{d-1},\qquad \|u\|=1.
\]

The force turns the velocity without changing its norm. The numerical
correction must account for the change in volume on position–sphere space.
For the paper's force substep, define
\(e=-\nabla U/\|\nabla U\|\) and
\(\delta=\epsilon\|\nabla U\|/(d-1)\). Equation (8) gives the contribution

\[
\Delta K=(d-1)\log\{\cosh\delta+(e^\top u)\sinh\delta\}.
\]

The corresponding log-Jacobian is \(-\Delta K\). The acceptance ratio
therefore includes the sum of these contributions, as well as the target
density change. Calling ordinary HMC's endpoint kinetic-energy calculation
would be wrong for this method. The zero-gradient limit and stable evaluation
of the hyperbolic expressions need explicit numerical treatment.

The inspected BlackJAX implementation accumulates the integrator's work and
uses `end.logdensity - start.logdensity - accumulated_work` in its acceptance
calculation (`blackjax/mcmc/adjusted_mclmc.py`, approximately lines 234–274 in
the archived revision). The Langevin variant adds partial velocity
refreshments; the paper's trajectory-space correction and noise treatment must
be preserved. The adjusted kernel can have the desired invariant distribution
under the stated conditions. This does not make a finite run from arbitrary
initial states unbiased or converged.

MAMS still needs geometry, epsilon, trajectory length and, for its stochastic
variant, a refreshment scale. Section 6 uses dual averaging; its practical
acceptance target is 0.90, with a special higher setting for the funnel. Its
diagonal preconditioner is estimated using a preliminary run: the authors
explicitly say that, in practice, they use NUTS. Their length adaptation uses
autocorrelation information. The paper's symbol \(L\) is a physical length;
BayesFilter's \(L\) is an integer step count. A local implementation should
call physical length \(\tau\) to avoid conflating them. Neither the paper's
0.90 nor its asymptotic optimal-acceptance calculation should silently replace
BayesFilter's numerical policy.

The gradient-efficiency evidence is promising, but more specific than a
universal several-fold speedup. The following are selected entries in its
Table 1, comparing BlackJAX NUTS with the basic MAMS column:

| Author benchmark | NUTS gradient calls | MAMS gradient calls | Descriptive ratio, NUTS/MAMS |
| --- | ---: | ---: | ---: |
| Gaussian | 19,652 | 3,249 | 6.05 |
| Banana | 95,519 | 14,078 | 6.78 |
| Bimodal | 210,758 | 139,418 | 1.51 |
| Brownian | 29,816 | 13,528 | 2.20 |
| Stochastic volatility | 843,768 | 430,088 | 1.96 |

The criterion is a normalized squared error in the worst second moment below
0.01. Section 7 takes the median error across at least 128 chains at each step.
Tuning is excluded.
These are the authors' results, not local measurements or uncertainty-qualified
rankings for MacroFinance/DSGE. In particular, the state-space example supports
investigating a roughly twofold gradient reduction; it does not establish a
sevenfold end-to-end gain. The paper also finds similar condition-number
scaling to NUTS on its Gaussian examples. Its excellent funnel example does
not make geometry repair unnecessary.

For BayesFilter, MAMS is worth implementing only after the cheaper SNAPER
comparison, or sooner if profiling establishes that reducing total gradients
is the dominant opportunity. Use the same validated frozen map in both arms.
Retain a genuine MAMS kernel identity, including integrator, sphere dynamics,
preconditioning, trajectory law and refreshment settings.

**What LAPS contributes, and what its diagnostics cannot establish.**

Robnik and Seljak's *Faster parallel MCMC: Metropolis adjustment is best served
warm* [LAPS] combines an unadjusted microcanonical transient with adjusted
MAMS. The idea is to tolerate integration bias while the ensemble is still far
from the target, adapt using statistics across chains, and apply MH correction
for the final stage. This is a distinct initialization/adaptation strategy,
not evidence that MH should be removed from final economic inference.

Its strongest appeal is latency when many chains genuinely run together. The
experiments use an NVIDIA A100 with 40 GB; Table 1 uses 4,096 chains, and Table 2
uses 256. Table 2 explicitly distinguishes total gradient work from work per
chain. For stochastic volatility it reports 29,900 calls for sequential NUTS,
1,100 per LAPS chain, and 281,600 total LAPS calls. Thus an approximately
27-fold reduction in per-chain work comes with approximately 9.4 times the
total gradient work. Whether that reduces wall time for batched Kalman, UKF or
DSGE solution operations is an empirical hardware question. It cannot be
answered from the chain count alone.

The second issue is its estimate of distance from the target. Equations (4)–(6)
use

\[
V_{ij}(\pi,\rho)
=\mathbb E_\rho[-(x_i-\mathbb E_\rho x_i)\partial_j\log\pi(x)],
\qquad
\widetilde D=\frac1d\|I-V\|_F^2.
\]

Integration by parts gives \(V(\pi,\pi)=I\) when the required moments and
boundary terms are well behaved. The converse is false for general
distributions, as the paper acknowledges. Appendix B establishes separation
on a restricted zero-mean Gaussian family and adopts a diagonal approximation
for routine use. Two simple deductions show why this is not a general
warm-up certificate:

* If \(\pi=N(0,I)\) and \(\rho=N(m,I)\), then
  \(\partial_j\log\pi=-x_j\) and
  \(V_{ij}=\mathbb E_\rho[(x_i-m_i)x_j]=\operatorname{Cov}_\rho(x_i,x_j)
  =\delta_{ij}\). The paper's centered diagnostic is zero for an arbitrarily
  wrong mean.
* If \(\rho\) is zero-mean Gaussian with unit marginal variances and incorrect
  correlations, its diagonal diagnostic is zero. The full matrix detects this
  error; the default diagonal reduction does not.

The first example applies to the paper's centered definition. It is not a
claim about every code variant: the current code described below uses an
uncentered statistic. Neither version uniquely determines an arbitrary
distribution. A stable ensemble trapped in a subset of modes is a further
failure that moment diagnostics alone cannot rule out.

The energy-error-to-bias relation used in the adaptation has rigorous support
in restricted Gaussian settings; extrapolation to general targets is empirical.
The related unadjusted-HMC paper [Bias], in the June 2026 revision inspected
here, reports a Brownian-motion case where the proposed bound underpredicts
the observed error by about a factor of 1.5. Its Gaussian theorem should not
be described as a universal non-Gaussian error guarantee. The contraction and
bias assumptions in LAPS's theory have not been established for these
MacroFinance or DSGE targets.

There are also material differences between the LAPS paper and the current
pinned BlackJAX source:

| Operation | Paper | Inspected BlackJAX revision |
| --- | --- | --- |
| Desired energy-error variance | Apply \(F(C\widetilde D)\), where \(F(z)=4z^{3/2}/(1+\sqrt z)^2\) | `C * bias**(3/8)`; the paper expression appears immediately below as commented code |
| Equipartition diagonal | Center \(x\) by its ensemble expectation | `-x * grad_log_density`, without that centering |
| Fluctuation statistic | Relative standard deviation | `contract_history` uses variance divided by squared mean; the numerical cutoff therefore has a different meaning |
| End of adjusted adaptation | Must eventually freeze | An acceptance-tolerance termination flag controls freezing; the stored `num_adaptation_samples` is not the operative stopping counter |

At the first tolerance-satisfying update, the bisection routine still changes
epsilon using the previous termination flag, then freezes on the subsequent
call. A port must test the epsilon actually frozen, not merely the acceptance
measured just before it. These observations identify decisions to resolve
before a faithful implementation. They do **not** establish that the published
benchmarks are wrong: the inspected revision has not been identified as the
revision used in those experiments.

For an optional local LAPS path, archive and discard all adaptive/unadjusted
draws, freeze all shared parameters, and use separately initialized validation
chains. Shared adaptation couples chains; freezing does not instantly remove
their finite-time dependence or initialization bias. Keep independent
posterior checks and use the LAPS statistics as adaptation signals and
explanatory diagnostics. A final ensemble of a few hundred draws is not
automatically sufficient for tail probabilities or precise economic summaries.

**Trajectory adaptation is the most accessible sampler improvement.**

ChEES-HMC [ChEES] replaces a dynamic NUTS tree with trajectories whose length
distribution is adapted across a batch of chains. It optimizes movement in a
centered squared-radius statistic. Randomizing trajectory length also protects
against resonance. For a one-dimensional Gaussian with standard deviation
\(\sigma\), unit mass, and exact Hamiltonian integration, refreshed-momentum
HMC has lag-one position correlation \(\cos(\tau/\sigma)\). At
\(\tau=2\pi\sigma\), it returns to its initial position despite acceptance
one. At half a period the position changes sign, but its square is unchanged.
Acceptance or movement in the mean therefore cannot certify useful exploration.

SNAPER [SNAPER] addresses two limitations of ChEES: many moderate directions
can conceal one slow direction, and an objective unadjusted for integration
cost can favor overlong trajectories. It uses a principal direction \(w\),
with an objective of the form

\[
f(x)=((x-\mu)^\top w)^2,\qquad
C=\frac{\mathbb E[(f(x')-f(x))^2]}{\tau}.
\]

This is an adaptation proxy for efficient exploration, not a test of posterior
correctness. The paper's endpoint derivative construction avoids
differentiating through every integration step; a port should not accidentally
introduce filter Hessians. It also learns diagonal scaling. After adaptation,
the trajectory *distribution* can remain randomized while its parameters are
fixed.

There is a concrete adoption advantage: the installed TFP 0.25.0 source contains
`tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo` and `sample_snaper_hmc`.
The former explicitly does not adapt epsilon; the latter supplies dual
averaging. The kernel requires at least two chains, estimates a principal
direction and diagonal scaling, and freezes their updates after the configured
adaptation period. This is source availability, not proof that the consumer
call chains are compatible or that their complete GPU/XLA execution works.

The paper and helper favor very concentrated initialization for adaptation.
That choice must not silently replace BayesFilter's independent dispersed
verification starts. A sensible comparison distinguishes a concentrated pilot
used to propose settings from validation of the frozen kernel at the declared
start bank. The helper's chain count, burn-in length and learning-rate defaults
are implementation choices to investigate, not automatically justified
defaults for state-space targets.

The quickest smaller change is a declared, state-independent random trajectory
length in otherwise familiar HMC, followed by an actual SNAPER comparison.
Using SNAPER only to nominate several fixed \((\epsilon,L)\) pairs could fit
the current authority model sooner, but it would be a different final sampler.
It must not inherit the published performance claims for SNAPER sampling.

**Persistent dynamics offer alternatives, with additional state and correction.**

MALT [MALT] inserts partial Gaussian momentum refreshments within a Langevin
trajectory and applies one MH correction. The acceptance calculation sums the
energy errors of the deterministic integration substeps; it is not the total
endpoint energy change including stochastic refreshments. Its full refresh at
trajectory boundaries and internal damping address resonance and problems
associated with persistent momentum rejection.

Adaptive MALT [AMALT] adds online geometry, principal-direction, damping,
epsilon and trajectory adaptation. Its comparison with SNAPER is instructive:
gains per iteration need not be gains per gradient, and the paper reports
cases with slightly lower gradient efficiency than uniformly randomized
SNAPER. It is consequently a useful alternative for a diagnosed residual
problem, rather than a reason to implement several new samplers at once.
The author implementation under TensorFlow Probability's
`discussion/adaptive_malt` uses **JAX/FunMC**. Its location in the TFP repository
does not make it a drop-in TensorFlow kernel.

MEADS [MEADS] uses generalized HMC, persistent slice variables and ensemble
estimates of scales. Its “tuning-free” description means a particular automatic
adaptation construction, not an absence of assumptions. The paper's ongoing
ensemble adaptation uses folds and a held-out fold to preserve the required
conditional independence. Simultaneously adapting each chain from an ensemble
that includes itself is not automatically the same valid construction.
Poor starts or inaccurate gradients can also spoil its scale estimates. Its
state, fold scheduling and many-chain regime make it a larger first project
than SNAPER for the current consumers.

Delayed-rejection generalized HMC [DRGHMC] responds to a rejected proposal by
trying smaller integration steps. This could help with residual multiscale
geometry after whitening. Correct delayed rejection includes the probabilities
of rejecting earlier proposals on both the forward and reverse paths. The
author code recursively constructs reverse “ghost” proposals; bounded retry
depth is essential because the work can grow exponentially with depth. Simply
halving epsilon and retrying ordinary MH after rejection is wrong. Its
benchmarks include chains initialized from reference draws and CPU parallelism;
they do not establish cold-start or GPU performance for expensive filters.

For genuine jumps, such as some regime-boundary formulations, discontinuous
HMC [DHMC] offers Laplace-momentum reflection/refraction constructions. Smooth
MAMS or LAPS does not solve that mathematical problem. DHMC is a specialist
option requiring an analysis of the actual boundary, not an easy general
replacement. Its author code was not audited in this review, so no port-ready
recommendation is made.

**Geometry and initialization remain complementary work.**

For a frozen invertible map \(x=T(z)\), the density to sample is

\[
\log\pi_z(z)=\log\pi_x(T(z))+\log|\det DT(z)|.
\]

NeuTra [NeuTra] applies ordinary corrected MCMC to this transformed target.
SNAPER or MAMS can operate in the same coordinates. The map addresses a
different difficulty from trajectory adaptation. For funnel tests, retain the
user-directed exact or partially whitened map; an unsuccessful centered-funnel
run alone is not evidence of a broken tuner.

A learned map can be underdispersed or miss modes, and exact correction does
not guarantee that finite chains will repair that exploration failure. Include
map training, inverse evaluation for matched model-coordinate starts,
forward/Jacobian computation and transformed gradients in the cost account.
Follow the repository's current architecture policy: the October 6
forward/reverse study selects configured Huang DSF/NAF, while the repaired
author IAF remains the policy outside that study. This survey does not
reclassify historical maps as current valid evidence.

Pathfinder [Pathfinder] offers a cheaper initialization hypothesis. It forms
Gaussian approximations along L-BFGS paths, selects them using an estimated
ELBO, and combines multiple paths with importance resampling. The inspected
Stan source checks curvature information and retains the approximation with
the favorable ELBO. It can propose starting states and geometry; it does not
replace exact posterior sampling. Check failed curvature updates, dispersed
paths, importance-weight concentration and mode coverage. It is worth testing
if initialization is a measured bottleneck, especially before deciding that a
new neural map is always necessary.

The broader unadjusted MCLMC work [MCLMC, Bias] is valuable for fast transient
exploration and explicit bias–cost tradeoffs. Its continuous-time invariance
and exact subflow calculations do not remove bias from a finite-step splitting
scheme. For reported macroeconomic posterior quantities, keep the adjusted
final phase unless a separate study establishes the relevant discretization
error to the required accuracy. High ESS or stable running moments alone
cannot establish that bound.

**State-space structure may provide the larger gain.**

Särkkä and García-Fernández [Parallel] express filtering and smoothing as
associative message composition. On suitable parallel hardware this reduces
the sequential depth of a linear Gaussian calculation from \(O(T)\) to
\(O(\log T)\), while retaining \(O(T)\) total work with different constants
and memory requirements. It is a latency opportunity, not a logarithmic total
cost algorithm.

TFP already exposes `LinearGaussianStateSpaceModel(experimental_parallelize=True)`
and `experimental.parallel_filter`. The inspected parallel-filter source
computes filtered moments and per-time log likelihoods and supports a
time/batch-level observation mask. That is not automatically equivalent to
MacroFinance's component-specific missingness, time-varying matrices and
correlated process/measurement noise. Establish an equivalent Gaussian
formulation and check likelihood **and gradient** parity before comparing
runtime. Inspect memory growth with horizon, backpropagation cost, precision
and XLA compilation as well as steady-state speed. A nonlinear UKF cannot be
replaced with a linear scan while claiming the same target.

Even without temporal parallelization, profiling may identify reusable
parameter-independent work, an expensive derivative, retracing, scalar calls
inside a purported batch, or duplicate value/score computation. Those are
specific possibilities to inspect, not new defects diagnosed here. A batched
filter gradient can dominate enough that accelerating it benefits fixed HMC,
NUTS and every proposed alternative.

When the sampled unknown is the latent trajectory, Particle-MALA and
Particle-mGRAD [Particle] are more directly tailored alternatives. They combine
gradient-informed proposals with conditional SMC and backward sampling.
Their invariant target is the smoothing distribution for a fixed parameter
setting; unknown parameters need an additional valid update. Particle-mGRAD
uses conditionally Gaussian transitions, with the parent-independent
transition covariance in equation (13) for the particular marginalized
construction. Those restrictions matter before transferring its favorable
high-dimensional examples. Author code is JAX, and its CPU experiments do not
establish TF/GPU performance.

For models whose states are already integrated out exactly by Kalman
filtering, adding thousands of latent unknowns is not an automatic improvement.
For nonlinear stochastic volatility with a sampled path, a conditional
particle method may be substantially more relevant than another generic
structural-parameter HMC tuner. Compare those as separate inference designs.

**The DSGE force distinction needs an explicit decision.**

A position-only force different from \(\nabla\log\pi\) does not, by itself,
make every Metropolized proposal invalid. To see this, write a kick as
\((q,p)\mapsto(q,p+\epsilon F(q))\). Its Jacobian is triangular with unit
determinant, and its inverse is the negative-step kick. The usual drift has
the same properties. A symmetric kick–drift–kick composition with momentum
reversal can therefore be used in an MH proposal against the actual endpoint
Hamiltonian, under the required regularity and deterministic evaluation
conditions. This is a local derivation, not a certification of the BGS code.

However, gradient-based adaptation, equipartition identities and the
microcanonical derivations must be checked for that actual force. Replacing
the score in LAPS's identity with an arbitrary force destroys the identity's
stated justification. Ordinary leapfrog volume preservation also does not
prove the correction for a different, compressible microcanonical proposal.

For BGS, first establish either an eligible exact value/total-score route or a
separately derived force-based transition and adaptation. The present
capability registry classifies the position-field branch as conditional
mechanics, without exact-score retained-member authority. Do not send it
through the fixed-transport tuner by labeling it an exact gradient. Custom
QZ/eigendecomposition/solution operations also make a wholesale JAX migration
a separate engineering project, not a sampler option flag.

**Warm-up and precision should use complementary evidence.**

The distinctions already made in BayesFilter remain useful: acceptance
compatibility and numerical movement qualify a candidate; warm-up readiness
and posterior precision answer later questions. R-hat should not return as an
acceptance-tuning requirement. It remains useful in posterior assessment.

Nested R-hat [Nested] is relevant if an ensemble method produces many short
chains. Its construction uses dispersed superchains, with subchains sharing
an initial state and independent randomness conditional on that state. Its
threshold depends on that design; a familiar long-chain cutoff cannot simply
be copied. Shared cross-chain adaptation complicates those independence
assumptions. An independently fitted, frozen kernel followed by a correctly
constructed diagnostic run is the simpler first integration. Neither ordinary
nor nested R-hat proves that undiscovered modes are absent.

Lugsail long-run variance estimation [Lugsail] addresses finite-sample bias in
MCSE estimates. In its usual notation it combines batch estimates as

\[
\widehat\Sigma_{\rm lug}
=\frac{\widehat\Sigma_b-c\widehat\Sigma_{b/r}}{1-c}.
\]

Its justification needs appropriate mixing, moments and batch-size conditions;
it does not diagnose all initialization or integration bias. Finite-sample
covariance estimates also need numerical validity checks. Source inspection
shows that BayesFilter **already has** `method="lugsail"` in
`bayesfilter/inference/hmc_precision.py`, and posterior assessment imports this
precision machinery. This is not a fresh executable qualification of every
consumer. The separate experimental lugsail acceptance-uncertainty policy has
not passed its strong-dependence promotion study; do not confuse it with
posterior MCSE estimation.

Define precision for the quantities actually reported: parameters, second
moments, predictive spreads, impulse responses, quantiles and event
probabilities as applicable. Use estimators appropriate to each quantity;
mean MCSE is not a quantile MCSE. Heavy tails can invalidate moment-based
criteria, and zero observed rare events can produce a misleadingly small
empirical variance. More chains and better ESS are useful only relative to
these inferential requirements.

**A discriminating evaluation, before an implementation campaign.**

The next implementation plan should ask: at a fixed target and stated accuracy
in economic quantities, which change reduces total elapsed time without
introducing a correctness failure? It should contain two comparisons:

1. **Matched geometry:** compare frozen kernels on the same target, map, data,
   numerical precision and model-coordinate starts. Allow method-specific
   tuning under a declared comparable resource budget. Count all integrator
   force evaluations, including multi-stage methods and rejected retries.
2. **End to end:** include compilation, initialization, transport fitting,
   tuning, fresh verification and retained sampling. Vary initialization
   separately so a sampler is not credited for an initializer only it received.

The baseline ladder should include the actual broad-grid HMC, simple
randomized-length HMC on the same geometry, and a correctly tuned TFP NUTS
comparison before SNAPER and MAMS. Where mathematically available, construct
cheap checks: independent Gaussian draws for the Gaussian control, exact
noncentering for the funnel, analytic/quadrature parameter references for small
state-space models, and sequential Kalman calculations for the parallel
filter. Evaluate by regime—interior/boundary, short/long horizon,
weak/strong identification—rather than hiding failures in an aggregate score.

Promotion would require target/gradient and transition validity, agreement
with independent references where available, and achievement of predeclared
MCSE or error goals. Divergences, wrong moments/mode weights, invalid target
evaluations or missing required diagnostics can veto a candidate. Acceptance,
trajectory objectives, equipartition and descriptive ESS/gradient cannot
replace those conditions. Method ranking needs repeated runs and uncertainty
for the declared comparison; passing a screen alone leaves a candidate viable.

| Test model or constructed case | Failure that a favorable headline can conceal | Required discriminating check |
| --- | --- | --- |
| Gaussian at a full or half Hamiltonian period | Acceptance near one with no movement in positions or squares | Means, variances, squared-observable autocorrelation and randomized-length control |
| Rotated anisotropic Gaussian with one slow direction | ChEES or average diagnostics hide the difficult direction; diagonal scaling is insufficient | Known covariance, projections onto eigenvectors, slowest declared estimand |
| Shifted Gaussian ensemble with correct covariance | Paper's centered LAPS diagnostic reports zero bias | Known mean error as well as equipartition; test centered and code variants explicitly |
| Zero-mean correlated Gaussian ensemble with correct marginal variances | Diagonal equipartition misses dependence | Off-diagonal covariance and full-matrix diagnostic |
| Exactly and partially whitened funnels | Good latent behavior hides model-coordinate tail error or a map/Jacobian bug | Forward/inverse and total-gradient checks; known model-coordinate moments and tail quantities |
| Heavy-tailed scale model | Variance-based adaptation or MCSE assumes nonexistent or poorly estimated moments | Declare existing moments; use bounded functions and quantile diagnostics where necessary |
| Separated mixture with deliberately unbalanced starts | Local diagnostics and all-chain agreement miss a mode | Known mode probabilities, transitions and independent dispersed starts |
| Constrained parameter near a boundary | Missing transform Jacobian, saturation or wrong score | Analytical/quadrature reference and directional gradient checks near the boundary |
| LGSSM with missing components and correlated noises | A faster filter computes a different likelihood | Sequential/parallel value and gradient parity before posterior comparison |
| Nonlinear stochastic volatility or other nonlinear SSM | Sampler agreement conceals shared filter approximation error | Small independent reference and declared approximation; separate latent-path comparison if applicable |
| Representative MIDAS and BGS targets | Synthetic success does not transfer to filtering, QZ operations or practical starts | Scope-bound target/score checks, batch timing, economic estimands and frozen-kernel validation |

MAMS mechanics tests should cover the unsupported one-dimensional formula,
zero force, aligned/antialigned velocity, sphere norm, forward/reverse
consistency, and the accumulated Jacobian/work. Stochastic-refreshment tests
must test its actual correction, not apply a deterministic inverse test to
unrelated random draws. MALT requires its deterministic-work correction;
delayed rejection requires reverse-path probabilities.

All adaptive methods need tests for freeze and resume boundaries, changed
batch sizes, one stuck or invalid chain, repeated random streams, interrupted
checkpoints, and exclusion of every adaptive draw. LAPS additionally needs
paper-versus-code formula fixtures and verification of the final frozen
epsilon. GPU tests must record incremental allocation, stable TensorFlow
signatures, XLA behavior, actual batch memory and contention. Short mechanics
tests establish implementation properties; replicated posterior tests answer
statistical questions. They are different kinds of evidence.

An old \((\epsilon,L)\) pair cannot describe all of these kernels. Extend the
typed kernel description before granting new artifact authority: transition
family, integrator, mass/map, length distribution, refreshment, adaptation
freeze, exact-score or force contract, and random-stream/checkpoint semantics
must be represented. Preserve the shared controller, independent verification
and retention of **all** verified candidates. Avoid adding another competing
single-winner tuner. These are proposed implementation requirements, not
changes completed by this review.

The suggested sequence is therefore: profile and establish consumer target
contracts; qualify randomized HMC/SNAPER and equivalent filter optimizations;
compare MAMS on the same maps; then decide whether measured ensemble scaling
justifies LAPS. Advance MALT, Pathfinder or latent-state methods when their
specific mechanism matches the observed bottleneck. A failed candidate is a
repair trigger, while a broken target, invalid correction or exhausted budget
stops the corresponding experiment. A future campaign must specify its own
compute budget and accuracy thresholds; the paper's constants are not a
substitute for that plan.

**Sources, revisions and limits of the review.**

PDFs, extracted text and author-code snapshots are retained under
`.localresources/hmc-methods-review-20261007/`. The
[source manifest](../../.localresources/hmc-methods-review-20261007/source-manifest.json)
records file hashes, examined revisions, source repositories and inspection
scope. NeuTra uses the existing local September 23 paper/source archive, also
hashed in that manifest. ResearchAssistant ingestion was attempted; unreliable
parsed metadata was replaced by direct PDF and source inspection. No downloaded
author implementation was executed.

| Label | Primary source and examined version | Main technical evidence used |
| --- | --- | --- |
| MAMS | [Robnik, Cohn-Gordon and Seljak, arXiv:2503.01707v2, May 2025](https://arxiv.org/abs/2503.01707v2) | Dynamics/correction §§3–5 and Appendix A; tuning §6; benchmarks §7/Table 1; integrator appendices |
| LAPS | [Robnik and Seljak, arXiv:2601.16696v1, January 2026](https://arxiv.org/abs/2601.16696v1) | Ensemble adaptation, equations (4)–(8), Tables 1–2; assumptions and implementation in Appendices A–F |
| ChEES | [Hoffman et al., 2021, PMLR 130](https://proceedings.mlr.press/v130/hoffman21a.html) | Squared-radius objective, randomized trajectories, adaptation and comparison design |
| SNAPER | [Sountsov and Hoffman, arXiv:2110.11576v3, May 2022](https://arxiv.org/abs/2110.11576v3) | Principal-direction objective, cost normalization, experiments and Appendix C derivative |
| MEADS | [Hoffman and Sountsov, 2022, PMLR 151](https://proceedings.mlr.press/v151/hoffman22a.html) | Generalized dynamics, persistent slice and folded ensemble adaptation |
| MALT | [Riou-Durand and Vogrinc, arXiv:2202.13230v3, December 2023](https://arxiv.org/abs/2202.13230v3) | Langevin splitting, accumulated energy error and validity conditions |
| AMALT | [Riou-Durand et al., 2023, PMLR 206](https://proceedings.mlr.press/v206/riou-durand23a.html) | Automatic damping/length adaptation, SNAPER comparison and implementation appendix |
| Pathfinder | [Zhang et al., arXiv:2108.03782v4, May 2022](https://arxiv.org/abs/2108.03782v4) | L-BFGS Gaussian construction, ELBO selection, multipath resampling, failures |
| NeuTra | [Hoffman et al., 2019, arXiv:1903.03704](https://arxiv.org/abs/1903.03704) | Transformed HMC §2.3, evaluations and limitations; author `MakeNeuTra` |
| Hardware | [Sountsov, Carroll and Hoffman, arXiv:2411.04260v1, November 2024](https://arxiv.org/abs/2411.04260v1) | Parallel workload design and worked MCMC example; contextual review |
| MCLMC | [Robnik et al., arXiv:2303.18221v3, May 2025](https://arxiv.org/abs/2303.18221v3) | Continuous dynamics, subflow/integration §§4–6, assumptions and experiments; original work dates from 2023 |
| Bias | [Robnik, Cohn-Gordon and Seljak, arXiv:2412.08876v3, June 2026](https://arxiv.org/abs/2412.08876v3) | Gaussian bias relation, non-Gaussian checks and §5 counterexample; original preprint dates from 2024 |
| Nested | [Margossian et al., arXiv:2110.13017v6, May 2024](https://arxiv.org/abs/2110.13017v6) | Superchain construction, conditional independence, thresholds and appendix arguments |
| Lugsail | [Vats and Flegal, arXiv:1809.04541v3, July 2021](https://arxiv.org/abs/1809.04541v3) | Long-run covariance correction and consistency/moment assumptions |
| Parallel | [Särkkä and García-Fernández, arXiv:1905.13002v2, February 2020](https://arxiv.org/abs/1905.13002v2) | Associative Gaussian filtering/smoothing and complexity; original preprint 2019 |
| Particle | [Corenflos and Finke, arXiv:2401.14868v1, January 2024](https://arxiv.org/abs/2401.14868v1) | Latent-path target, Algorithms 3–7, validity propositions, covariance restriction, SV experiments; complete appendix proof not independently rederived |
| DRGHMC | [Turok, Modi and Carpenter, arXiv:2406.02741v2, February 2026](https://arxiv.org/abs/2406.02741v2) | Delayed-rejection correction, §5 experimental design and ghost-proposal cost; original preprint 2024 |
| DHMC | [Nishimura, Dunson and Lu, arXiv:1705.08510v5, August 2019](https://arxiv.org/abs/1705.08510v5) | Reflection/refraction and mixed-momentum integrator; specialist orientation, author code not audited |

The principal author-code anchors are:

* [BlackJAX, revision 6f0edc2](https://github.com/blackjax-devs/blackjax/tree/6f0edc214bf3329989153d5a2aba00a2f86edf6c):
  adjusted MCLMC, integrators, LAPS burn-in/adaptation, step-size termination,
  and MEADS/ChEES adaptation. LAPS differences above refer specifically to this
  revision; `laps_burn_in.py` lines 198 and 329 and `step_size.py` lines
  265–305 are central anchors.
* [Author benchmark repository, revision 78ab785](https://github.com/reubenharry/sampler-benchmarks/tree/78ab7852f0fd59e0b9a096d27d369a8e4f6776ef):
  LAPS wrapper, experiment setup and squared-coordinate observables. Current
  source is not established to be the exact paper-run revision.
* [TFP, revision d6b503d](https://github.com/tensorflow/probability/tree/d6b503dd2492522af7c260002539514b959afbe6):
  `discussion/adaptive_malt` and FunMC's OBABO/energy-error implementation.
  Separately archived installed TensorFlow files identify the available
  SNAPER, trajectory-adaptation and parallel-filter source by SHA-256;
  installed-package source is not equated to that upstream revision.
* [Stan, revision babdd0e](https://github.com/stan-dev/stan/tree/babdd0e67933e92355207a7cfe4c1dd07459e8c6):
  `src/stan/services/pathfinder/single.hpp`, particularly curvature updates,
  approximation construction and ELBO selection. Multipath theory was read
  in the paper; this is not a complete audit of Stan's multipath call chain.
* [Particle-MALA, revision 83f62f6](https://github.com/AdrienCorenflos/particle_mala/tree/83f62f6ede504b36cc0d76932dcd670d9e16a5aa),
  [DRGHMC, revision 7a4b0da](https://github.com/gil2rok/drghmc/tree/7a4b0daa12dbbc733bdd53496ccee2a941a10fb0),
  and [nested R-hat, revision 74bccf7](https://github.com/charlesm93/nested-rhat/tree/74bccf73df3bb0c90155970ed797a17bfeaa917d):
  inspected conditional-particle interfaces, reverse-rejection recursion and
  superchain diagnostic code respectively. These snapshots are audit sources,
  not approved runtime dependencies.

The strongest alternative explanation for the recommendation is that target
evaluation or geometry already dominates so completely that a sampler change
saves little. Conversely, excellent batch scaling could make LAPS attractive
earlier. The proposed profiling and matched-target comparisons distinguish
those possibilities. The weakest evidence is transfer from published benchmark
targets to the actual consumers: no local sampler ranking, default promotion
or release-readiness conclusion is supported by this literature review alone.
