# Literature remedies for the observed NeuTra fitting failures

This source audit answers how the literature treats poor forward fits, loss of
modes during energy training, and inaccurate low-probability geometry. It runs
no experiments and changes no training defaults. The comparison is with the
recorded October 4 exact-teacher IAF fits and the earlier October 2 controlled
campaign. The skeptical check is to distinguish a published remedy from a
demonstrated local solution, and density fitting from corrected sampling and
Gaussianization. A better approximate teacher alone cannot explain away failure
when the student already receives exact target samples.

## Preserve example-based training during energy refinement

Noé et al. (2019), *Boltzmann Generators: Sampling Equilibrium States of
Many-Body Systems with Deep Learning*, explicitly report that energy-only
training tends to concentrate in the most stable metastable state. Methods
equation (9) combines maximum likelihood, energy and optional reaction-coordinate
losses. Gabrié et al., Section III.B and Appendix A.2, also describe combining
forward and reverse objectives after the map has become sufficiently accurate.

With correctly weighted target examples, the combined population objective can
be written in our notation as

\[
J_\lambda(q)=D_{\rm KL}(q\|p)+\lambda D_{\rm KL}(p\|q),\qquad\lambda>0.
\]

The original examples may be imperfect or unequilibrated; replacing the target
expectation by those examples is a distinct approximation. For a partition into
regions A_k, write w_k=p(A_k), a_k=q(A_k), and p_k,q_k for the corresponding
normalized conditional densities. Directly splitting the KL integrals gives

\[
D(p\|q)=\sum_k w_k\log(w_k/a_k)+\sum_k w_kD(p_k\|q_k),
\quad
D(q\|p)=\sum_k a_k\log(a_k/w_k)+\sum_k a_kD(q_k\|p_k).
\]

Thus a positive forward term imposes a divergent cost as a_k tends to zero
for fixed w_k>0. This explains its resistance to complete mode loss. It does
not force accurate weights or conditional shapes at a finite optimum, guarantee
successful optimization, or prescribe a universal lambda.

The original archived DoubleWell notebook, extracted cells 35 and 39, actually
uses maximum-likelihood pretraining followed by a combined ML/KL objective.
Gabrié's original `flonaco/training.py:239` also has a combined-loss branch.
These are source implementations, not merely an interpretation of their prose.

Local limitation: the October 2 controlled campaign already ran forward,
reverse, joint and joint-continuation arms for eight target/teacher/seed cases;
none reached fresh posterior confirmation under that bounded protocol. This
is not a newly untried cure. The October 4 r3 controller instead switches to
pure RKL unconditionally at `neutra_scientific_campaign.py:352`. Removing that
regression alone does not repair the forward maps' pre-existing shape errors.

Sources: `.localresources/neutra-mode-initialization-literature-20260929/`
`noe-boltzmann-generators.txt:359,1323`, `noe-doublewell-training-cells.txt`;
`.localresources/fab-coverage-followup-20260928/gabrie-adaptive-flows.txt:272,1062`;
`docs/plans/bayesfilter-neutra-controlled-repair-results-2026-10-02.md:19`.

## Use a representation that can resolve the required deformation

Huang et al. (2018), *Neural Autoregressive Flows*, Section 3 and Figure 5,
replace each conditional affine scalar transformation by a monotone nonlinear
one. An affine transformation of a scalar Gaussian remains Gaussian at fixed
conditioning variables; a nonlinear derivative can create several density
peaks. Stacking and permuting affine layers can still represent much richer
multivariate distributions, so this is not an impossibility result for our IAF.

Durkan et al. (2019), *Neural Spline Flows*, Section 3.1, equations (4)-(8),
use monotone rational-quadratic splines with analytic derivatives and inverses.
The Jacobian remains triangular in an autoregressive construction:

\[
\log|\det J_T|=\sum_j\log|\partial T_j/\partial z_j|.
\]

Thus improved scalar flexibility need not require a dense Jacobian determinant.
The original spline implementation computes the inverse by a quadratic root
and the log determinant from a closed-form derivative. Autoregressive inverse
evaluation can still be sequential across coordinates. Standard splines are
C1; transformed-force regularity at knots and HMC behavior need separate checks.
This is a comparison candidate, not authority to replace the canonical IAF.

Gabrié Section IV.E explicitly proposes informed base distributions, appropriate
coordinate scaling and mixtures of separately trained maps for different modes.
Its mixture experiment uses six pairs of RealNVP coupling layers with width-100
conditioners, rather than our three IAF stages. A mixture of maps can serve as
an evaluable sampling proposal but is not automatically a single bijection from
a standard Gaussian for the existing NeuTra HMC consumer.

Sources: `.localresources/q20-flow-training-literature-20260923/papers/`
`huang-2018-naf.txt:175`, `durkan-2019-splines.txt:200`; original
`code/nsf-rational_quadratic.py:94`; Gabrié text `:406,1641`.

## Refresh sampling information and reduce the size of each transport task

Gabrié Algorithm 1 learns from newly advanced chains, alternating local moves
with flow-based global proposals. Local moves explore details that the flow
has missed; global moves can equilibrate mode weights. Section IV.C assumes
initial representatives in the relevant modes and explicitly disclaims reliable
discovery of previously unrepresented basins. Its experimental ULA paths have
discretization error; the availability of corrected MALA does not make every
reported experiment an exact finite-time sampler.

AFT (Arbel et al., 2021), Section 3.3 and Appendix F, separates training,
validation and test particles and uses validation to stop stage fitting.
CRAFT (Matthews et al., 2022), Section 2.3, diagnoses AFT's sample replenishment
problem: repeatedly optimizing on one finite population can overfit or stop
learning useful gradients. CRAFT repeats complete annealed passes with fresh
initial particles, retaining importance, resampling and Markov corrections.
Its gradients are not unconditionally unbiased gradients of the exact-target
objective: Section 3.2 gives an unbiased-gradient interpretation for a modified
objective involving the expected finite-particle measure.

Annealing breaks one difficult transport into smaller adjacent tasks. It does
not prove all modes are found, and fresh data does not by itself repair an
underfitting architecture. Our existing exact-teacher failure keeps learner
calibration logically separate from these teacher improvements.

Sources: Gabrié text `:249,312,336`; `.localresources/smc-modern-improvements-20260929/`
`aft-linear.txt:532`, `craft-linear.txt:582,607,712`; original
`aft-author.py:113` and `craft-author.py:125`.

## Make missed mass expensive, subject to the objective's applicability

FAB (Midgley et al., 2023), Section 3.1, minimizes an alpha-two objective
proportional to integral p(x)^2/q(x) dx, using AIS toward that integrand's
normalized density. It concentrates training effort where p is appreciable
but q is small. This is a stronger missed-mass penalty than ordinary RKL.
It requires a finite integral and effective transitions between the AIS
distributions. Section 3.1's favorable dimensional scaling analysis explicitly
assumes factorized targets and perfect intermediate transitions.

The previous q20 applicability and AIS-calibration failures must be resolved
before treating FAB as a remedy there. Neither the paper nor our finite parity
tests supply a guarantee for an arbitrary initial proposal or nonlinear target.
Source: `.localresources/neutra-highdim-literature-20260925/fab-2023.txt:160`.

## Collect rare-region information deliberately and correct its weights

Boltzmann Generators add an optional reaction-coordinate loss to encourage
transition-region sampling. That deliberately changes the training objective;
importance correction is used for equilibrium estimates. Umbrella sampling
instead simulates biased overlapping windows. EMUS (Thiede et al., 2016),
Sections II-III, estimates their relative normalizations through an overlap
matrix and combines the weighted information. Overlap and within-window
sampling are essential assumptions; a poor reaction coordinate can miss
barriers in other directions.

For exact disjoint strata A_k, the simpler identity is

\[
\mathbb E_p[f(X)]=\sum_k p(A_k)\mathbb E[f(X)\mid A_k].
\]

This permits oversampling rare strata without assigning them false posterior
mass. Estimating unknown stratum weights remains part of the problem. Even a
perfect weighted estimator leaves a rare region's contribution to ordinary
KL small. If the goal is accurate derivatives there, that requires its own
criterion; it is not guaranteed by ordinary likelihood fitting.

Sources: Noé text `:383,1323`; `.localresources/neutra-rare-events-20261002/`
`thiede-emus-2016.layout.txt`, Sections II-III, equations (9)-(17).

## Corrected samples and Gaussian whitening remain different outcomes

For a fixed positive proposal q, independence Metropolis uses
alpha(x,y)=min(1,p(y)q(x)/(p(x)q(y))). The identity
p(x)q(y)alpha(x,y)=min(p(x)q(y),p(y)q(x)) proves detailed balance, independently
of whether q equals p. This is invariance, not finite-time convergence or
an unconditional adaptive-MCMC theorem. Importance weights likewise correct
integrals under their support and integrability assumptions, with finite-sample
bias for self-normalized estimates.

Small density loss does not control scores. As a local counterexample, let p
be a standard Gaussian and q(x)=p(x)[1+epsilon sin(kx)], 0<epsilon<1. Symmetry
makes q normalized, and D(p||q)=O(epsilon^2) uniformly in k. Yet

\[
\partial_x\log q-\partial_x\log p
=\frac{\epsilon k\cos(kx)}{1+\epsilon\sin(kx)},
\]

which can grow arbitrarily large as k increases. This derivation explains why
the literature's successful sampling or low KL cannot establish our stronger
Gaussian-score requirement. No claim is made that this oscillatory example
describes our saved maps.

The justified next comparison is the author's complete documented fit on an
exact-teacher control, its matched port, and then the canonical IAF with
separately calibrated architecture and objective choices. The earlier joint
negative results must be retained. Assess mode weights, conditional shape,
rare-region geometry and actual corrected sampling separately; a successful
check at one level cannot silently establish the others. No new local success
or stochastic ranking is established by this literature audit.
