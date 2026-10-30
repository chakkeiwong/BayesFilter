# Why the NeuTra warm-start campaign failed on some targets

The completed campaign combines distinct failures. The unwarped mixture has a
real optimization and residual-shape problem, compounded by a training limit
that discards improving trajectories. The funnel SMC teacher has inadequate
proposal overlap and almost immobile mutation. The funnel HMC test has a
demonstrably unsuitable start bank and an ineffective step-size repair range.
Several other HMC rejections are caused by a short-block variability rule,
without numerical or movement failure. The aggregate 12/30 result is therefore
not a measure of how often the IAF successfully learned a useful transport.

This audit follows the executed consumers into the shared numerical code and
checks the saved results. Three maps also received a bounded GPU diagnostic.
All evidence below is under
`artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/causal-audit-20260930-r1/`.
The audit plan is `bayesfilter-neutra-failure-causal-audit-2026-09-30.md`.

## 1. The unwarped mixture: Gaussian stationarity and incomplete shape learning

The target in `neutra_warm_start_targets_tf.py::_log_prob` is

\[
 p(x_1,x_2)=\left[\tfrac13\phi(x_1+5)+\tfrac23\phi(x_1-5)\right]\phi(x_2).
\]

Its coordinates are independent. Its first-coordinate mean is 5/3 and variance
is \(1+100(1/3)(2/3)=209/9\). The best moment-matched product Gaussian therefore
has approximately

\[
 D_{\rm KL}(p\|g)
 \simeq \tfrac12\log(209/9)-H(1/3,2/3)=0.93604.
\]

The approximation drops the tiny conditional component-label entropy from
overlapping components. The campaign's finite development sample gives about
0.94739. Many recorded fits remain near this value. That agreement is supported
by a separate saved-map check: the plateau map's deviation from its fitted
affine transformation has RMS 0.0112 in coordinate 1, against output standard
deviation 4.868. The map is nearly affine on those Gaussian base draws.

There is a mathematical reason this can be a difficult starting point for
forward-KL training with affine autoregressive layers. Standardize the target
by the moment-matched diagonal affine map. Its coordinates \(U_j\) remain
independent, with \(E_pU_j=0\) and \(E_pU_j^2=1\), although one marginal is
non-Gaussian. At a standard-Gaussian model, an infinitesimal conditional shift
\(u_j\mapsto u_j+\eta h(u_{<j})\) changes the log density by

\[
 \left.\partial_\eta\log q_\eta(u)\right|_0=u_jh(u_{<j}).
\]

An infinitesimal conditional scale
\(u_j\mapsto e^{\eta h(u_{<j})}u_j\) changes it by

\[
 \left.\partial_\eta\log q_\eta(u)\right|_0=(u_j^2-1)h(u_{<j}).
\]

The shift formula follows by differentiating the inverse substitution
\(u_j-\eta h\) in the Gaussian density. For scale, differentiate both
\(-e^{-2\eta h}u_j^2/2\) and the log-Jacobian term \(-\eta h\).
Independence and the matched moments give

\[
 E_p[U_jh(U_{<j})]=0,
 \qquad E_p[(U_j^2-1)h(U_{<j})]=0.
\]

Consequently the forward-KL derivative is zero in these directions. The
diagonal-affine subfamily of a stack whose individual layers are diagonal
affine, separated by coordinate permutations, retains this property. Escaping
can require coordinated nonlinear changes across layers. This derives a
stationary configuration; it does not prove a local minimum, impossibility of
escape, or that every observed plateau is exactly stationary. The measured
plateau-map gradient is not exactly zero. Finite samples, nonzero neural
initialization, and small non-affine components all matter.

The source uses the canonical three-stage IAF with two ELU hidden layers,
coordinate reversal and the free scale bias outside the conditional cap.
`make_transport` changes the configured widths; it does not insert a new
transport family. The author-style small variance initializer places the
conditioners near weak nonlinear transformations. The current forward training
objective is a separate training choice: author architecture alone does not
guarantee that this objective avoids the stationary configuration above.
The architecture and original RKL recipe are described in Hoffman et al.,
Section 4.1.1, and the locally preserved author `MakeIAFBijectorFn` and
`DenseAR` code; the stationarity argument here is a local derivation.

The warped benchmark changes the conditional mean to
\(E[X_2\mid X_1]=0.1(X_1^2-26)\). A shift conditioner can therefore receive a
first-order signal \(E[X_2h(X_1)]\ne0\). This explains why geometrically adding
a nonlinear shear need not make this optimization problem harder. It is a
mechanism consistent with the results, not a statistically established ranking
between benchmark difficulty levels.

The full history is more informative than the six terminal failures:

| Generator | Fitted configurations over both attempts | Stopped on plateau | Nonlinear screen passed and reached 8,192 updates |
|---|---:|---:|---:|
| SMC | 24 | 15 | 9 |
| Gabrié | 24 | 22 | 2 |

All eleven escaped configurations had lower recorded validation loss at 8,192
updates than at 4,096. These are descriptive loss trajectories, not independent
statistical comparisons. The lowest observed loss, SMC seed 37/width 8/LR .003,
fell from 0.25975 to 0.13028 to 0.06934 at 2,048, 4,096 and 8,192 updates.
Its only failed physical observable was the valley probability:

\[
 p(|X_1|<2)=\Phi(7)-\Phi(3)=0.00134990,
 \qquad \widehat q(|X_1|<2)=0.0301514.
\]

The SMC teachers for all three seeds passed their screens and had valley
probabilities 0.00109–0.00140. Thus the teacher is not the observed source of
this fit's excess valley mass. A fresh saved-map sample reproduced about
3.14% valley mass. Its affine residual is large, so this fit has learned
nonlinear structure, unlike the plateau map.

Forward KL need not heavily penalize a few percent of unwanted mass where
the target assigns almost none. Coarsening into the valley and its complement
gives the exact lower bound

\[
 D_{\rm KL}(p\|q)\ge
 p_A\log(p_A/q_A)+(1-p_A)\log[(1-p_A)/(1-q_A)].
\]

For \(p_A=0.00134990\), \(q_A=0.03\), the right side is only 0.02488. Low
forward loss and unacceptable valley mass can therefore coexist. The valley
check detected a real discrepancy; it is not evidence that nothing was learned.

The controller then makes the wrong repair choice for the improving fits.
`fit` has a literal terminal rung of 8,192. On failure the master launches a
fresh initialization on the same grid, rather than restoring the saved
parameters and Adam state of an improving fit. The allocation still had more
than 26 GPU hours. More funding does not affect this stopping rule. No ordinary
mixture candidate in this repair campaign reached RKL or HMC, because the warm
fit gate stopped it first. Their 0/6 is not an observed RKL or HMC failure.

## 2. Funnel teachers: a density mode is not a typical region

The implemented ten-dimensional funnel is

\[
 V\sim N(0,1),\qquad X_{1:9}\mid V=v\sim N(0,e^{2v}I).
\]

At zero children, log density is constant \(-v^2/2-9v\), so the density mode
is at \(v=-9\). Its local child variance is \(e^{-18}=1.523\times10^{-8}\).
Nevertheless the marginal of V is centered at zero. This is the volume effect
of nine conditional coordinates; the density maximum does not locate typical
posterior mass.

The proposal puts 90% of its mass in the mode's local Laplace Gaussian and 10%
in a broad Student distribution. The local proposal's V marginal is near
\(N(-9,1)\), so most particles begin in an atypical narrow neck. The broad
component has support everywhere, but its scale of 4 in nine children does
not provide many particles in the small conditional scales at negative V.
Full support is not useful finite-sample overlap.

The mutation time step was calibrated against this extremely narrow proposal
as well as the later bridges. The chosen \(dt=1.523\times10^{-9}\) is reused
at all temperatures. In seed 11's 8,192-particle populations, resampling and
mutation occurred only three times, four MALA steps each. Ignoring drift, their
cumulative noise scale is

\[
 \sqrt{2(12)dt}=1.91\times10^{-4}.
\]

That is negligible relative to the unit scale of V. At the old mode even the
joint target score in V vanishes when all children are zero, so gradient drift
does not solve the missing-volume problem. The tiny accepted steps cannot be
interpreted as exploration of the typical funnel mass.

The actual teacher errors are substantial:

| Seed | Estimated E[V] | Estimated P(V<0) |
|---|---:|---:|
| 11 | 1.1861 | 0.0203 |
| 23 | 0.9901 | 0.0171 |
| 37 | 0.8066 | 0.0488 |
| Exact target | 0 | 0.5 |

For seed 11, final particle ESS values of 5,363–6,310 coexisted with only
145–175 distinct original ancestors out of 8,192. Weight ESS after resampling
does not measure independent exploration; duplicated particles with nearly
equal weights can have high ESS. The logged normalizer estimates also differed
substantially from the known log normalizer zero. The reference disagreement,
genealogy and mutation scale support poor overlap and inadequate rejuvenation,
not an incorrect SMC importance-weight formula. A factor increase in particle
count alone has not repaired this proposal/kernel combination.

## 3. Funnel HMC: the start bank creates the immediate failure

The HMC consumer loads the selected map but recreates physical starts with
`initial_walkers`, using discovered modes plus the same isotropic 0.2 noise in
every coordinate. It does not use the saved, assessed walkers or draws from the
learned map. At \(v=-9\), the child's conditional standard deviation is
\(e^{-9}=1.234\times10^{-4}\). The chosen jitter is about 1,621 conditional
standard deviations. Its expected child contribution to negative log density
is

\[
 E\left[\tfrac12e^{18}\sum_{j=1}^9X_j^2\right]
 =\tfrac12 e^{18}(9)(0.2)^2\simeq1.18\times10^7.
\]

For the exact recorded seed-11 start bank, target log densities ranged from
-7.95 million to -28.80 million. Exact noncentered coordinates had norms
3,987–7,590; even under the learned map their latent norms were 20–30.
These are unsuitable starts for claiming a test of useful trained-map geometry
near posterior mass.

A bounded diagnostic held the saved map, target, epsilon .0625, L=3 and random
momentum seed fixed. It evaluated 64 independent proposals per start bank:

| Start bank | Mean acceptance probability | Nonfinite log-acceptance values |
|---|---:|---:|
| Recorded four starts, replicated | 0 | 0 |
| Draws from the learned map | 0.999388 | 0 |
| Exact target draws, transformed through the same map | 0.999641 | 0 |

This is direct diagnostic evidence that changing initialization removes the
immediate rejection at this epsilon. It does not establish convergence,
efficient trajectories or complete tail whitening. The learned-map bank is
available without the oracle; the oracle bank only checks the interpretation.

The selected funnel maps also retain a separate training limitation. For all
three Gabrié seeds, refinement failed the shape screen after 256 RKL updates
and preserved the preceding warm map. The selected maps' saved 1,000-point
diagnostics give median score-residual norms of 0.561, 0.482 and 0.543, and
95th percentiles of 2.291, 2.238 and 2.306 for seeds 11, 23 and 37 respectively.
These are standard-normal base draws, not posterior draws. They show remaining
departure from Gaussianization; they do not explain away the demonstrated
initialization defect or establish downstream convergence. The evidence is in
each `attempts/closure-refine-funnel-gabrie-s{seed}-r1/` directory's
`phase.json`, `history.json` and `post-training-1000.json`.

At four recorded starts, the largest local transformed-potential Hessian
eigenvalue was 2.59e7–9.71e7, versus 2.81–7.04 at four learned-map draws. For a
harmonic direction, leapfrog stability requires \(\epsilon\sqrt\lambda<2\).
That local diagnostic gives epsilon scales about 0.00020–0.00039 at the bad
starts, hundreds of times smaller than .0625. A nonlinear Hessian at a point
is not a global stability guarantee, but the measured proposal failures are
consistent with this extreme disparity.

The frozen Gabrié sampler can escape such starts through global independence
proposals. Its Metropolis ratio is \(p(y)q(x)/(p(x)q(y))\), so a proposal near
typical mass can immediately replace an extremely poor starting point. HMC
uses local gradients and does not inherit that escape mechanism. Passing the
Gabrié check does not justify discarding its resulting states and rebuilding
the same pathological starts for HMC.

The purported wider HMC repair also fails to widen the required range.
`qualify` fixes `fixed_grid_max_attempts=3`. The worker repair increases work
units and refinement rounds but leaves this value unchanged. The public tuner
maps it into a three-repair family cap and epsilon domain. All three funnel
seeds therefore tested only .5, .25, .125 and .0625 for L=3,9,18; every pilot
had acceptance zero. Each used 12 of 144 work units, leaving 132 unused. This
is a configured search-range limitation, not exhausted compute or proof that
the supplied map cannot support HMC.

## 4. Other HMC failures: a noisy-block rule becomes an admission barrier

In `_acceptance_decision_from_summary`, a chain with one short block below
0.65 and another above 0.75 is classified as `inconclusive_evidence` before
the pooled uncertainty interval is considered. This uses raw extrema rather
than testing whether the block difference exceeds Monte Carlo variation.

One recorded wiggle verification at epsilon 0.1869186, L=9 had pooled mean
acceptance 0.69058, a chain-mean interval [0.66122,0.71994], no numerical or
movement veto, and 256 decisions per chain. One chain's block means were
0.68145, 0.79138, 0.63507 and 0.76826. Those raw crossings alone forced the
inconclusive classification. Across the five failed non-funnel qualifications,
94 observation-level decisions satisfied the other recorded compatibility
conditions but were held inconclusive by this block rule. These are repeated
observations, not 94 independent candidates.

The rule can frequently trigger even under an independent stationary example.
For iid Bernoulli(0.7) observations, let \(l\) and \(h\) be the probabilities
that one block average is below .65 and above .75. Among four blocks, the
probability of both types is

\[
 s=1-(1-l)^4-(1-h)^4+(1-l-h)^4.
\]

Among four independent chains, the probability any chain triggers is
\(1-(1-s)^4\). Exact binomial sums give 94.5% with blocks of 16 and 66.3% with
blocks of 64. This is an illustrative stationary bounded-variable reference,
not an estimated rejection rate for the actual correlated HMC acceptance
probabilities. It demonstrates why the crossing rule is not itself evidence of
temporal instability. The appropriate repair is to calibrate a temporal
heterogeneity diagnostic with uncertainty, preserving genuine numerical and
movement vetoes. Silently admitting every current inconclusive candidate would
also be unsupported; fresh verification remains necessary.

## 5. What the inspected numerical code computes

`TrainingBlock` samples SMC replay rows according to normalized particle weights
and then uses an equal-weight minibatch. Conditional on the saved cloud, this
has the correct expectation for its weighted forward objective. The weighted
trainer differentiates \(-\sum_i\bar w_i\log q_\phi(x_i)\) with respect to all
trainable transport parameters. The shared map computes
\(\log q_\phi(x)=\log\phi(T_\phi^{-1}x)-\log|\det J_{T_\phi}|\), including the
inverse-map dependence. This is the intended density objective, not an omitted
Jacobian or an accidentally detached inverse.

Saved-map directional finite differences agreed with autodiff to absolute
errors 4.7e-10, 2.7e-8 and 7.2e-10 for the plateau, learning-mixture and funnel
maps. Their round trips agreed to at most 5.6e-15. These checks reject a large
gradient/sign/roundtrip error in the inspected directions; they are not a
proof of every derivative throughout the parameter space.

The selected mixture fits had clip thresholds about 227, whereas the diagnostic
full-reference gradient norms were 0.114 and 3.65. Calibration's clipped and
unclipped Adam steps agreed at the tested initial batches. Thus these artifacts
do not implicate the earlier tiny clipping threshold. However, `TrainingBlock`
computes accumulated norms and clipped-update counts that `fit` does not save
in its histories. Full training-time clipping frequency was not checked and
cannot be inferred from the initial pilot or one final full-data gradient.
That observability gap should be repaired.

## Recommended repair order and decision

1. Repair HMC initialization to use a checked bank from the frozen map and/or
   assessed frozen-sampler states, with deliberately dispersed states constructed
   in meaningful coordinates. Preserve warm-up, independent verification and
   posterior checks. The oracle is a diagnostic control only.
2. Make epsilon initialization and the maximum number/domain of downward repairs
   reflect actual geometry. Increasing a work budget without changing the
   binding repair cap does not extend the search.
3. Calibrate the temporal-block rule against stationary references and replace
   raw crossings with uncertainty-aware evidence. Run fresh verification of
   affected frozen maps before claiming any new HMC admission.
4. Continue the improving unwarped-mixture checkpoints with their Adam state;
   use actual progress and shape errors to determine the next rung. Treat
   Gaussian plateaus separately, with a controlled test of initialization and
   objective schedule within the canonical IAF. Do not replace the architecture
   based on these results.
5. Repair funnel teacher overlap and mutation mobility jointly at the mechanism
   level, then assess them separately. Mode discovery alone cannot provide
   typical-set coverage in this model. Stage-specific calibrated mutation and
   a proposal representing posterior volume need explicit validation; another
   particle-count increase with the same nearly immobile kernel is not supported.

The present valley discrepancy genuinely fails the declared warm-map screen.
Mathematically, exact Gaussianization is not required for corrected HMC:
\(\pi_z(z)=p(T(z))|\det J_T(z)|\) remains the target for an invertible T.
Therefore a map failing an approximation-quality screen has not thereby failed
HMC. Whether to permit an explicitly diagnostic downstream test of such a map
must be stated in the next evidence contract; this audit changes no threshold.

| Finding | Evidence status | Decision | Uncertainty / next check |
|---|---|---|---|
| Funnel start bank creates immediate HMC rejection | Reproduced with same map/kernel and changed starts | Repair initialization before interpreting HMC failure as training failure | Full warm-up and posterior performance remain untested under repaired starts |
| Wider retry retains a narrow downward epsilon range | Direct call-chain and artifact finding | Repair the effective cap/domain | A larger search alone need not be efficient |
| Raw temporal-block crossings cause avoidable inconclusive decisions | Source trace, recorded counterexample and stationary-reference calculation | Calibrate the diagnostic | Actual HMC dependence and fresh verification still matter |
| Eleven mixture fits stopped while loss was decreasing at the last rung | Saved histories; residual valley error reproduced | Continue preserved checkpoints | Additional updates may still plateau before the shape criterion |
| Forward KL has a diagonal-affine stationary configuration for this independent target | Derived exactly under the stated conditions; saved plateau map is nearly affine | Test a controlled escape intervention | Not proof this is the sole cause of all 37 plateaus |
| Funnel SMC misses substantial negative-V mass | Reference disagreement and ancestry/mobility evidence | Repair proposal and mutation | Quantifying each component's causal contribution requires a matched follow-up |

| Inference status | Finding |
|---|---|
| Hard veto screen | Existing failed candidates stay unpromoted; fixed-start proposal diagnostics produced finite values |
| Statistically supported method ranking | None |
| Descriptive differences | Loss trajectories, map residuals, short-proposal acceptance and per-case success counts |
| Default-readiness | Not established |
| Next evidence needed | Targeted repairs with preserved comparators, fresh verification and downstream checks |

The strongest alternative explanation for some plateaus is finite-capacity or
optimizer tuning rather than the derived stationary mechanism. The strongest
limitation of the HMC intervention is its short length: eliminating immediate
rejection does not show adequate exploration. Both limits are explicit, while
the initialization and search-cap defects are directly demonstrated.

The GPU diagnostic used 35.08 process-seconds and 59.11 CPU core-seconds and was
charged once to the existing campaign ledger. It used host GPU 1, FP64,
TensorFlow/XLA and verified memory growth. Its script, input hashes, source
hashes, seeds, resource limits and results are preserved alongside
`saved-map-diagnostic-manifest.json`. No training default, candidate status or
production numerical code was changed in this audit.
