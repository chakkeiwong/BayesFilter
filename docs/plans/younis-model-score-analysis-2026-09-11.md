# Using Younis's mixture gradients to estimate the model score

Date: 2026-09-11; applicability corrected 2026-09-12. This is a conceptual analysis and a proposed
research direction, not a new experiment or an implementation promotion.

For models with regular transition densities, a candidate can use a continuous mixture to put particles where the
model posterior has mass, retain the original model density in the importance
correction, and estimate the model score through a complete posterior-score
recursion. Younis's importance-weighted mixture gradients are useful tools for
this construction. They are not, by themselves, a theorem that smoothing a
particle filter removes its score bias. A reference construction for those
models is **a model-corrected marginal particle filter, informed
by the LEDH proposal geometry, with a Rao–Blackwellized Fisher-score estimate**.
A hybrid pathwise/importance-weight gradient is a separate candidate for
reducing gradient variance when differentiating mixture expectations is needed.

This is a smoothing-based score construction. It is not applicable as written
to the degenerate DSGE target: transitions from fixed ancestors can lie on
distinct lower-dimensional supports, and compatible backward weights may
collapse to the original ancestry. The earlier general recommendation was
wrong in scope. A support-aware DSGE score construction remains unresolved;
full-rank Gaussian tests cannot establish its validity. See the
[manuscript correction](younis-degenerate-transition-correction-2026-09-12.md).
The subsequent whole-problem reassessment is recorded in
[degenerate-model-score-reassessment](artifacts/degenerate-model-score-reassessment-20260912-01/degenerate-model-score-reassessment.md);
that document supersedes the regular-transition option table below for the
degenerate DSGE score question.

This recommendation is based on the mathematical target and inspected source
operations. It is not an empirical ranking: this combination has not been
implemented or tested here. The previous restriction to the derivative of the
executed LEDH finite program was too narrow for the user's latest question,
which explicitly asks how to estimate the underlying model score better.
Canonical LEDH identity and its analytical-derivative requirements still
govern anything presented as that existing implementation.

## What has actually been established

The newest relevant research is on `kdm-total-score-continuation-20260909`,
commit `804616e3`, rather than the current checkout. The recovered
[Phase 4B result](artifacts/younis-score-recovery-20260911/recovered/bayesfilter-ledh-younis-kdm-phase4b-campaign-result-20260910.md)
reports the following matrix-LGSSM score comparisons with an independent
Kalman score:

| Scope | Canonical score MSE | Raw-IWSG KDM score MSE | Paired 95% interval for KDM minus canonical MSE |
|---|---:|---:|---|
| N=32, T=5 | 3.94210 | 4.45250 | [0.07106, 0.94986] |
| N=64, T=20 | 3.86260 | 10.13463 | [5.10083, 7.55881] |

These are reported results from the saved branch, not reruns in this recovery.
They support rejecting those candidates for promotion in those scopes. Their
small-bandwidth calibration also produced very large score errors while
remaining finite. Neither fact establishes failure of every KDM construction.
Both N and T changed between scopes, so their difference cannot identify a
horizon effect alone. The canonical numerical controls were not independently
tuned for each scope. The comparison therefore does not identify the strongest
possible canonical baseline. Its bootstrap comparator used a fixed-ancestry
finite derivative, which is not an adequate stand-in for every conventional
particle-score estimator.

The subsequent [five-adapter result](artifacts/younis-score-recovery-20260911/recovered/bayesfilter-ledh-younis-kdm-repository-models-result-20260910.md)
reports 39 passing correctness tests after repairs. It explicitly leaves
parameter-dependent initialization and the Austria target identity unresolved.
The analytical executor at that commit initializes state and covariance
tangents to zero (`ledh_canonical_score_tf.py:199–202`); this is correct for a
fixed supplied cloud, but omits derivatives required by a parameter-dependent
initialization. The SIR adapter's continuous recurrence also differs from the
clipped simulator. A kernel cannot repair either mismatch. Further DSGE
implementation was deferred by the user; the next validation should use
existing repository models.

## Define the score before choosing its estimator

For fixed observations y and model parameter theta, let

\[
 Z(\theta)=p_\theta(y_{1:T}),\qquad
 S(\theta)=\nabla_\theta\log Z(\theta).
\]

The canonical finite filter instead produces a scalar
\(L_N(\theta;\xi)\), with random design \(\xi\), and an analytical derivative
\(G_N=\nabla_\theta L_N\). A derivative check establishes that the code computes
this derivative. Accuracy as a model-score estimate concerns
\(\mathbb E_\xi[G_N\mid y]-S\), its variance, and its resulting mean squared
error. These are different questions.

Four mechanisms need separate diagnoses:

| Mechanism | What would establish it | Suitable repair |
|---|---|---|
| Missing derivative or wrong model | Endpoint derivative tests or explicit model-identity mismatch | Repair the analytical recursion, initialization, or target |
| Missing distributional derivative | A discrete sampling law changes while its probability derivative is omitted | Derive a valid expectation/Fisher estimator; a fixed-stream finite difference alone is insufficient |
| Changed filtering approximation | Reset, kernel, clipping, or omitted correction changes the measure being propagated | Restore the required model correction or declare and study the approximation |
| Finite Monte Carlo normalization | Correct numerator and denominator estimates still form a random ratio | Reduce integration error, use posterior-score estimators, increase/couple effort, or investigate debiasing |

Even a positive unbiased likelihood estimate does not generally give an
unbiased log-likelihood score. Suppose \(\widehat Z=\theta+\epsilon\), with
\(\epsilon=\pm a\) equally likely and \(\theta>a>0\). Then
\(\mathbb E\widehat Z=\theta\) and \(\partial_\theta\widehat Z=1\), but

\[
 \mathbb E\,\partial_\theta\log\widehat Z
 =\tfrac12\left(\frac1{\theta+a}+\frac1{\theta-a}\right)
 =\frac{\theta}{\theta^2-a^2}\ne\frac1\theta.
\]

This elementary counterexample establishes the distinction without an
asymptotic argument. Conversely, finite-sample score bias is not inevitable in
every special case; it must be measured or derived for the estimator at hand.
Error averaged over different observation paths also does not establish the
conditional bias for any one fixed dataset. The next experiment needs repeated
filter randomizations within each fixed dataset to separate those quantities.

## What we should take from Younis

[Younis and Sudderth (2023)](https://arxiv.org/abs/2404.08789), Section 4,
equations (14)–(15), fix a proposal \(q_0(z)=m_{\theta_0}(z)\) and vary the
numerator in \(m_\theta(z)/q_0(z)\). Under common support and differentiation
under the integral, for a differentiable test function phi,

\[
 \nabla_\theta\int\phi_\theta(z)m_\theta(z)\,dz
 =\mathbb E_{q_0}\left[
 \frac{m_\theta(z)}{q_0(z)}\nabla_\theta\phi_\theta(z)
 +\frac{\phi_\theta(z)}{q_0(z)}\nabla_\theta m_\theta(z)
 \right].
\]

Samples remain fixed for this estimator; changes in all overlapping mixture
components influence their weights. This provides a way to handle mixture
expectations without transporting samples between separated modes. It does
not identify the mixture with our model posterior, remove a random logarithm
or normalization, or establish unbiasedness for an arbitrary nonlinear
downstream function of an empirical cloud.

The paper's learned MDPF is a discriminative state-prediction model. Its
training objective and evidence are not a proof about the score of our
generative state-space model. The author code confirms the mixture log-density
gradient injection and its multiplication into the subsequent weights before
normalization (`kde_particle_filter.py:738–748,927–934`, commit `b0e2fd54`).
That operation is different from first normalizing away its raw weight mass.
Appendix B.1 also states quadratic particle cost for the gradient computation;
the linear inference cost does not describe this score calculation.

[Younis and Sudderth (2024)](https://doi.org/10.52202/079017-0228), Section 4,
offers another useful idea: propose from both forward and backward mixtures
to cover plausible states. Equation (19) includes a generative importance
correction and an auxiliary density; equation (20) replaces the numerator
with a learned discriminative weight. For model-score inference we need the
former kind of correction. Multiplying two posterior approximations without
the required reference-density adjustment can count prior information twice.
Their single-state smoothing marginals alone also do not supply the adjacent
state-pair expectations needed for transition-parameter scores.

## A model-corrected mixture route

Given normalized filtering weights \(W_{t-1}^j\) and states \(x_{t-1}^j\), the
actual model transition defines the predictive mixture

\[
 a_t(x)=\sum_j W_{t-1}^j f_\theta(x\mid x_{t-1}^j).
\]

When f has a tractable continuous density, this is already a continuous
mixture. We can study Younis-style mixture calculus on it without adding a
kernel bandwidth to the model. Alternatively, use a KDM or LEDH-informed
mixture \(q_t\) as the proposal. For \(x_t^i\sim q_t\), retain

\[
 v_t^i=\frac{g_\theta(y_t\mid x_t^i)\,
                 \sum_j W_{t-1}^j f_\theta(x_t^i\mid x_{t-1}^j)}
                {q_t(x_t^i)},\qquad
 \widehat z_t=\frac1N\sum_i v_t^i,
 \quad W_t^i=\frac{v_t^i}{\sum_k v_t^k}.
\]

Conditionally on the incoming cloud, integrating \(v_t\) under \(q_t\) gives
\(\int g_\theta a_t\): the proposal affects sampling efficiency while the
numerator specifies the intended update. The denominator must be the actual
proposal density, including mixture probabilities, transformations, and any
Jacobian, and must be positive wherever the numerator has mass. Numerator
weights and states must belong to the intended incoming
filtering approximation. Substituting reset centers with uniform weights into
the numerator silently changes that approximation.

A possible support safeguard is a proposal
\(q_t=(1-\eta)q_{\mathrm{guided}}+\eta a_t\), with fixed \(0<\eta<1\).
Then \(q_t\ge\eta a_t\), so \(v_t\le g_\theta(y_t\mid x_t)/\eta\).
When the observation likelihood is bounded, this also bounds incremental
weights. It does not bound every score derivative. The mixing fraction is a
scope-specific proposal choice requiring calibration, not a new default.

[Lai, Domke and Sheldon (2022)](https://proceedings.mlr.press/v151/lai22a.html),
Algorithm 2 and Theorem 2, equations (9)–(12), provide the relevant generative
construction: marginalize the ancestor index and evaluate the full transition
and proposal mixtures. Their theorem supports unbiased normalizer estimation
for the specified recursion. It does not automatically apply to a recursion
with an extra uncorrected deterministic OT/GenUT reset. Their Section 4 also
explicitly limits the Rao–Blackwell variance guarantee to the local operation;
it does not prove lower variance for the complete filter in every case.

The [author implementation](artifacts/younis-score-recovery-20260911/sources/vmpf-author.py),
lines 53–61 at commit `7c5970d1`, computes the transition mixture with
`reduce_logsumexp` and subtracts the proposal log density. This supports the
specific weight formula, not compatibility of that external implementation
with BayesFilter's analytical, GPU/XLA, initialization, and tuning requirements.
No external implementation was executed in this analysis.

LEDH flow, UKF covariance information, GenUT, and OT can help construct
proposal centers, scales, and couplings. The physical transition and
observation densities still determine the numerator. If a transformed
proposal cannot be evaluated, it cannot support this correction. A Gaussian
mixture built from those centers with explicit component covariances is a
possible evaluable proposal; treating its covariance as extra model noise is
a different operation.

There is an especially clean initial experiment. In a linear-Gaussian model,
each term \(f_\theta(x\mid x_{t-1}^j)g_\theta(y_t\mid x)\) has an analytically
integrable Gaussian product. Its integral supplies the component evidence,
and the normalized product supplies a conditional proposal. Selecting
components in proportion to incoming weight times component evidence gives
the fully adapted mixture; all incremental importance weights then equal
\(\int g_\theta a_t\) conditionally on the cloud. This removes current-step
proposal variance, while earlier particle approximation remains. The
transition covariance in this calculation is the model's Q, not an
arbitrarily added kernel covariance or a per-particle bookkeeping mark.

For singular transitions or constrained states, ordinary ambient-space density
ratios may not exist. Use a declared ancestor/innovation representation with
the correct base measure and retain the parameter dependence of the state map.
A full-dimensional Gaussian kernel is not a generic repair for singular or
clipped support. This is a boundary to respect in later existing-model tests,
not a reason to resume the deferred DSGE implementation.

## Estimate the model score directly

For a differentiable state-space density on a common support, differentiating
the joint integral and dividing by Z gives Fisher's identity:

\[
 S(\theta)=\mathbb E_\theta\left[
 \nabla_\theta\log\mu_\theta(X_0)
 +\sum_{t=1}^T\{\nabla_\theta\log f_\theta(X_t\mid X_{t-1})
               +\nabla_\theta\log g_\theta(y_t\mid X_t)\}
 \mid y_{1:T}\right].
\]

Here the derivatives inside the expectation hold latent states fixed. They
are complete-data scores, not derivatives along a sampled trajectory. The
initial-law term is required. Moving support needs an appropriate
reparameterization and derivation; this displayed density formula cannot be
used without those conditions.

The following recursion is therefore restricted to models for which its
transition densities and backward conditionals are valid. In the manuscript's
degenerate AR(2) example, a child's deterministic lag coordinate identifies
its generating ancestor when parent coordinates are distinct. All other
ancestor probabilities vanish. PaRIS cannot restore those forbidden
transitions, and an ambient Gaussian proposal cannot exactly correct to the
finite singular predictive mixture. An innovation-coordinate construction
would require a new derivation including the state map's parameter dependence.

A particle implementation can average histories rather than commit to a
single sampled ancestry. Define

\[
 b_t^{ji}=\frac{W_{t-1}^j f_\theta(x_t^i\mid x_{t-1}^j)}
                   {\sum_k W_{t-1}^k f_\theta(x_t^i\mid x_{t-1}^k)},
\]

initialize \(A_0^j=\nabla_\theta\log\mu_\theta(x_0^j)\), and propagate

\[
 A_t^i=\nabla_\theta\log g_\theta(y_t\mid x_t^i)
       +\sum_j b_t^{ji}\left[A_{t-1}^j+
          \nabla_\theta\log f_\theta(x_t^i\mid x_{t-1}^j)\right],
 \qquad \widehat S=\sum_i W_T^i A_T^i.
\]

This is the conditional-expectation recursion for the additive joint score
on the particle approximation. It exposes how all plausible ancestors
contribute and where the complete model derivatives enter. The inspected
Ścibior–Wood paper, Section 4.2, equations (24)–(25), describes the
related marginal score recursion and its connection to Poyiadjis et al. The original [Poyiadjis–Doucet–Singh paper](https://www.stats.ox.ac.uk/~doucet/poyiadjis_doucet_singh_particlescoreparameterestimation.pdf) was subsequently recovered and checked during the manuscript revision: Section 2.2, Algorithm 2, equations (20) and (22), gives the direct primary-source construction.
Finite particle normalization and filtering error remain: this estimator is
not generally unbiased at finite N.

This route is a serious alternative because its scientific target is directly
S. It need not equal the derivative of the old OT-reset finite scalar. An
implementation must be labeled as a separate model-score estimator; it must
not acquire canonical LEDH-score identity by relabeling. Its decisive check is
replicated agreement with an independent model-score reference, alongside
local complete-data derivative checks. Requiring it to match a fixed-stream
finite difference of a different finite program would test the wrong claim.

The transition responsibilities can share all-pairs work with a marginal
filter. The straightforward cost is quadratic in N at each step. Streaming
reduces memory, not the pair count. Any BayesFilter transport implementation
must continue to use the exact-divisor chunk policy; approximating mixture
sums or changing chunk semantics would require its own error analysis.

## Use importance-weight gradients selectively

An all-IWSG derivative can have large variance even for a single narrow
Gaussian. For \(Z=\mu+h\epsilon\), \(\epsilon\sim N(0,1)\), and
\(\phi(Z)=aZ+b\), the location-score estimator is
\(\phi(Z)\epsilon/h\). Its mean is a, but its variance is
\((a\mu+b)^2/h^2+2a^2\). The pathwise derivative is the constant a.
Subtracting the detached baseline \(\phi(\mu)\) from the score-function term
removes the diverging part in this example. This is a local derivation; it
does not assert the same scaling for every full filtering recursion.

For a mixture \(m_\theta=\sum_i w_i k_{i,\theta}\), sample a component I and
reparameterize its continuous variable \(Z=T_{I,\theta}(\epsilon)\). Then

\[
 \nabla_\theta\mathbb E_m[\phi_\theta(Z)]
 =\mathbb E\left[
       \frac{d}{d\theta}\phi_\theta(T_{I,\theta}(\epsilon))
       +(\phi_\theta(Z)-c)\nabla_\theta\log w_I
   \right].
\]

The first term holds I fixed and differentiates within the component; the
second accounts for changing component probabilities. The baseline c must
be independent of the sampled component and noise for this simple formula.
If a component-dependent baseline is used, its compensating weight term is
required. Enumerating components or suitable stratification can reduce the
categorical noise further, at additional cost. Fitted baselines should use
independent data or a valid leave-one-out construction; an arbitrary
same-sample fit can destroy the zero-mean property.

The author's `ImportanceImplicitHybrid` branch (`kde_particle_filter.py:750–790`)
separates continuous-parameter and weight derivatives, but uses an implicit
mixture reparameterization for the former. The componentwise formula above is
a distinct local proposal. Its useful limiting behavior in the Gaussian
example does not prove favorable variance for multimodal filtering, nor does
this branch's existence establish published validation of it.

A hybrid must preserve the declared expectation derivative throughout the
recursion. Substituting it into the fixed-anchor Phase 4B program while
claiming the same finite derivative would be wrong. Similarly, the small-h
limit of a kernel expectation, the limit of its derivative estimator, and the
limit of that estimator's variance are separate questions.

## Where control variates and other options fit

The original control-variate idea remains useful, with a more precise role.
If \(\mathbb E C=c\) and beta is fixed or independently fitted, then
\(G_{cv}=G-\beta(C-c)\) has the same expectation as G. It reduces variance
when suitably correlated, and therefore may reduce MSE; it cannot remove
bias already present in G. Calling a biased KDM score a centered control
variate does not supply the missing known expectation.

A stronger construction uses a tractable approximate joint density
\(\widetilde p_\theta(x,y)\), with known integral \(\widetilde Z\), and an
exact residual:

\[
 Z=\widetilde Z+\mathbb E_q\left[
       \frac{p_\theta(X,y)-\widetilde p_\theta(X,y)}{q(X)}\right].
\]

With fixed q and valid differentiation, the corresponding identity for
\(\nabla Z\) uses the difference of the joint-density derivatives. This can
reduce integration error in numerator and denominator when the approximation
is close. Taking their estimated ratio still introduces finite-sample bias,
and a signed residual estimate can make the likelihood estimate nonpositive.
It is not automatically a usable pseudo-marginal likelihood. Computing both
full scores and writing \(S_0=S_h+(S_0-S_h)\) supplies an identity, but no
variance or cost improvement without an additional integration argument.

| Option | Legitimate benefit | Remaining issue and priority |
|---|---|---|
| Corrected physical/KDM mixture proposal | Better particle coverage while retaining the model update | Incoming-filter error and ratio bias remain; first new construction to investigate |
| Marginal/Fisher score recursion | Average ancestor uncertainty and estimate the model score directly | Finite-N bias, support conditions, quadratic reference cost; pair with the first construction |
| Hybrid pathwise/IWSG with valid baselines | Reduce avoidable mixture-gradient variance | Separate estimator identity and multimodal evaluation needed; focused parallel hypothesis, not a validated default |
| Analytic integration or Rao–Blackwellization of tractable state blocks | Remove sampled dimensions and current-step noise | Only exact for justified conditional submodels; use before adding another approximation |
| Positive-bandwidth KDM as the filtering approximation | Potential bias–variance tradeoff and better mode coverage | Declare changed finite approximation; consistency requires an appropriate N/bandwidth limit; tested Phase 4 variants have no promotion case |
| Younis-style forward/backward proposals | Cover states favored by past or future data | Preserve generative correction, adjacent-state information, and auxiliary density; proposal design to explore after the reference recursion |
| Exact-mean control variates or tractable surrogate plus residual | Reduce variance of integrals or scores | No automatic score debiasing; centering, support, positivity, and cost must be checked |
| More particles, coupled replications, randomized QMC | Reduce Monte Carlo integration error | Does not repair wrong derivatives or a persistent approximation; include as an equal-compute comparator |
| N/bandwidth extrapolation or randomized multilevel debiasing | Potentially cancel or remove limiting bias | Needs a demonstrated expansion or convergent coupled telescope, plus variance/work bounds; deferred research rather than a ready repair |
| KDM as a Metropolis-corrected proposal force | Potentially improve proposals for a declared endpoint density | Corrects sampling for that endpoint, not its model approximation; separate downstream use |

For randomized telescoping, the formal expectation identity is
\(\mathbb E[S_0+\sum_{l=1}^{L}(S_l-S_{l-1})/\Pr(L\ge l)]
=\lim_l\mathbb E S_l\), when independent truncation and absolute
integrability justify interchange. It only gives the model score if that
limit is the model score. Finite expected work and variance are further
requirements, not consequences of writing the telescope.

For the last row, a deterministic position-only force can be used with a
reversible, volume-preserving proposal and acceptance against the declared
endpoint Hamiltonian. Acceptance preserves that endpoint target under the
usual Metropolis conditions; it cannot convert an approximate likelihood into
the exact model likelihood. Exact pseudo-marginal inference additionally
requires a valid nonnegative unbiased likelihood estimate and its auxiliary
randomness in the target. The local HMC documentation and capability registry
were inspected: any future raw-coordinate force use must go through the typed
binding and `tune_hmc_kernel`, not the fixed-transport tuner. No HMC consumer
was changed and no HMC run is proposed here.

## The next discriminating study

For regular-transition reference models, a study could answer whether
model-corrected mixture proposals and ancestor averaging reduce conditional
model-score error at equal compute. This does not resolve the degenerate DSGE
question or constitute its implementation plan. It should begin after the
transition-support, initial-law, and target-identity checks for its chosen
model. The initial comparisons should separate proposal construction from
score construction: ordinary SMC versus marginal mixture weights; genealogy
versus ancestor-averaged Fisher scores; and, only where an expectation
derivative is needed, all-IWSG versus componentwise pathwise/weight gradients.
Changing all three at once would hide the reason for success or failure.

Use a nondegenerate Kalman case first, an exactly enumerable short
Gaussian-mixture model with evaluable transition densities second, and a
scalar nonlinear model with regular transitions and a checked quadrature score third.
Kalman validates correctness and conditional integration; the mixture case
tests the separated modes that motivate Younis's method. A unimodal Kalman
failure is valuable falsification evidence, but passing it would not establish
the multimodal benefit. Later existing-model expansion must retain the actual
initial law and support, and each model needs its own tuning scope.

Construct the cheap adversaries before interpretation:

| Comparator | Construction and reason |
|---|---|
| Bootstrap PF with a proper Fisher-score estimator | Propose from the actual transition and weight by the observation; tests whether added mixture machinery earns its cost. |
| Locally adapted proposal with exact importance weights | Use available local transition/observation information to guide each ancestor's proposal; tests whether simple observation adaptation explains an apparent gain. |
| EKF approximate-likelihood score, where defined | Linearize the actual dynamics and observation map and differentiate the resulting Gaussian likelihood; a cheap adversary in nearly linear regimes. |
| UKF approximate-likelihood score, where defined | Use sigma-point Gaussian moment propagation with complete derivatives; tests whether low-order nonlinear moment information suffices. |
| Tuned canonical LEDH | Use the complete current algorithm with its own calibration scope; tests the proposed extension against the actual project baseline. |

The first four are the simple adversaries; the fifth is the required project
comparator. Exact Kalman, mixture enumeration, or converged quadrature remains
the reference. Report conditional comparisons for weak/strong
observations, overlapping/separated modes, low/high persistence, short/long
horizons, and near-zero/large score components where relevant. These cases
must be chosen from model structure before inspecting claim data. Loss to a
cheap adversary in a salient case vetoes promotion under repository policy;
do not optimize against the adversary table after seeing the result.

For each fixed dataset, replicate the particle randomness and estimate
componentwise bias, variance, and MSE with uncertainty. Pair methods within
datasets and use uncertainty calculations that respect the nesting of
randomizations within datasets. Freeze coordinate scales to avoid unstable
relative errors near zero true score. Change N and T independently. Compare
both equal N and equal measured compute, because all-pairs work can erase an
apparent fixed-N advantage. A pilot determines replication needs; an
underpowered cap yields an inconclusive result, not a forced selection.

The study should separately report likelihood unbiasedness evidence, finite
program derivative correctness where claimed, posterior-score accuracy, and
numerical validity. Nonfinite values, invalid support/density ratios, missing
initial derivatives, a wrong reference, or omitted sampling-law terms are
validity vetoes. ESS, overlap, bandwidth, cap activity, tangent norms, and
memory explain behavior; they cannot substitute for model-score accuracy.
A valid candidate's loss triggers the next prespecified repair, not rejection
of the whole research direction. Any serious execution needs its own concise
per-scope plan, compute budget, commands, and untouched validation data; this
analysis does not authorize a new campaign or change the production route.

## Decision and remaining uncertainty

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not promote the tested Phase 4B arms | Their declared score improvement failed | Reported paired losses and design power limits | Generalization beyond those scopes | Preserve the negative results | Every KDM design fails |
| Restrict corrected mixtures plus Fisher recursion to regular-transition reference models | Algebra and sources support this restricted construction | Degenerate DSGE support obstructs the ordinary recursion; initial-law and density checks remain required | Finite-N error and equal-compute value; DSGE score construction unresolved | Verify support before any bounded reference study; keep DSGE implementation deferred | This solves degenerate DSGE scores, is empirically superior, or is implemented |
| Retain hybrid gradients and valid control variates as focused candidates | Local expectation identities established | Must not replace a different claimed derivative silently | Full-recursion variance and cost | Independent small checks before integration | A control variate removes arbitrary baseline bias |

| Inference status | Current finding |
|---|---|
| Hard veto screen | Existing initialization/target gaps prohibit claims that rely on those missing pieces; no new numerical run was performed |
| Statistically supported ranking | Saved Phase 4B intervals support losses in the two stated scopes; no ranking exists for the proposed new constructions |
| Descriptive-only differences | Saved bias/variance decompositions and bandwidth curves; no separate N/T scaling inference |
| Default readiness | No new default or canonical-score admission established |
| Next evidence needed | Correct target/initialization, independent score oracle, scope-specific calibration, nested replication, and equal-compute comparisons |

The strongest alternative explanation for a future improvement is a better
proposal or more effective computation, rather than the chosen gradient
identity. The ablations above distinguish them. A corrected-mixture/Fisher
candidate that loses to a tuned conventional particle score at equal compute
would overturn its practical priority. Source fidelity is strongest for the
local mixture-weight identities and the MPF recursion; finite-N performance
of their combination with LEDH geometry remains untested.

The [source and omission record](artifacts/younis-score-recovery-20260911/literature-ledger.md)
records the inspected technical sections, author code, metadata limits, and
deferred alternatives. The [recovery checkpoint](younis-score-analysis-recovery-2026-09-11.md)
preserves the session diagnosis and a small entry point for future work.

The full mathematical development, historical Section 3.6 results, current implementation distinction, and proposed comparisons are now integrated in [the revised LaTeX manuscript](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex) and its [PDF](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf). The later main-checkout batch wrapper forwards reset controls but remains row-mapped, and its shared correction dependency differs from the tested research snapshot; current runtime parity was not rerun for this document update.
