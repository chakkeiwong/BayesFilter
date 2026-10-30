# What the surrogate-HMC program tests, and what currently prevents an answer

This audit answers the owner's request to understand the algorithm, models,
code and possible repairs before seeking another Claude review. No reviewer or
new posterior experiment was launched. The checkout is
`ledh-graceful-failure-execution-20260917`, at commit
`8a4cffa2fd085f43ab5710c24d6e7b48770aea0d` with existing working-tree changes.
The source hashes for the focused checks are in the linked run record.

The research question remains: **can the full canonical analytical DCTR score,
used as an imperfect deterministic proposal field, produce reliable parameter
inference through endpoint-corrected HMC, and at what computational cost across
the repository's models?** A filtering-accuracy comparison or a score-parity
test cannot answer that question. Neither can successful compilation or a
high acceptance rate.

The checked proposal/acceptance mechanism is mathematically coherent. The
current evidence has three distinct limitations: the uncertainty procedure
has a demonstrated calibration defect and insufficient planned precision;
the SV20 trajectory was restricted to one leapfrog step; and the finite DCTR
likelihood can define a posterior different from the model posterior. Some
remaining model adapters also fail to represent their model laws. These are
different problems and require different repairs.

## The algorithm and the two approximations

For observed data `y`, the parameters are `theta` and the hidden states are
`x[0:T]`. The desired posterior is proportional to
`prior(theta) * integral p_theta(x[0:T], y) dx[0:T]`. The current HMC runs sample
the parameters; the likelihood calculation integrates or approximates the
hidden-state distribution. They do not run HMC jointly over every hidden state.

Let `theta=b(z)` be the fixed chart from unconstrained HMC coordinates. The
declared endpoint density is

\[
\pi_E(z)\propto p(b(z))L_E(b(z);y)|\det Db(z)|,\qquad
U_E(z)=-\log\pi_E(z).
\]

There are two possible endpoints: a reference likelihood `L_R`, or the fixed
finite DCTR likelihood `L_D`. The proposal field is independently supplied by
DCTR. In potential-gradient sign it is

\[
g(z)=-Db(z)^T\{\widetilde s_D(b(z))+\nabla_\theta\log p(b(z))\}
      -\nabla_z\log|\det Db(z)|.
\]

Here `s_D` is the analytical DCTR likelihood score. It is allowed to disagree
with the gradient of the endpoint. The chart derivative, prior and Jacobian
must still be accounted for with the correct sign. The consumer wiring tests
check those operations. Despite the filename `neural_force_hmc.py`, these
experiments use the DCTR analytical particle-filter score, not a neural network.

At each parameter value DCTR propagates a particle cloud and per-particle
covariances, flows the cloud towards the observations, assembles likelihood
weights, updates the covariance lifecycle, and performs its Contract E–Chol
transport reset and configured higher-moment/dual-cap corrections. The same
recursion propagates analytical parameter tangents. The finite random bank and
numerical controls are frozen for sampling. This supplies a deterministic
value/score program; freezing it does not make its likelihood the exact model
likelihood or an unbiased likelihood estimator.

With fixed mass matrix `M`, a proposal refreshes `p ~ N(0,M)` and repeats

\[
p_{1/2}=p-\tfrac h2 g(z),\quad
z'=z+hM^{-1}p_{1/2},\quad
p'=p_{1/2}-\tfrac h2g(z').
\]

After the fixed number of steps it flips momentum. Acceptance uses the
**declared endpoint** and the kinetic energy:

\[
\alpha=\min\{1,\exp[U_E(z)+K(p)-U_E(z')-K(p')]\},\qquad
K(p)=\tfrac12p^TM^{-1}p.
\]

The implementation caches the current potential, evaluates one new endpoint
per transition, and accepts or rejects separately for each chain. Omitting
kinetic energy would be wrong; it is present in the checked code.

Why can an imperfect field work? A kick is a translation of momentum at fixed
position, and a drift is a translation of position at fixed momentum. Each
preserves volume. Their symmetric composition is reversed by momentum flip,
so the flipped proposal is an involution. The Metropolis correction therefore
preserves the declared endpoint density under these assumptions. This argument
only requires a fixed, deterministic position-only field; it does not require
that field to be conservative. For a nonconservative field the proposal should
not be described as exact Hamiltonian flow or automatically symplectic. This
is a local derivation of invariance, not a proof of numerical reversibility,
irreducibility, mixing, finite-run precision or DCTR likelihood accuracy.

[Neal, Section 5.5](https://arxiv.org/abs/1206.1901) explains approximate
Hamiltonian trajectories with correction against the true endpoint. The
nonconservative-field extension above follows from the translation/involution
argument, rather than being attributed to that section.

| Comparison | Proposal field and endpoint | Question answered |
|---|---|---|
| A/R | Reference-gradient HMC, reference endpoint | Does the ordinary HMC baseline recover the reference? |
| B/R | DCTR field, reference endpoint | Does the corrected imperfect-field sampler recover the reference? |
| C/D | DCTR field, finite DCTR endpoint | Does the sampler recover its declared finite posterior? |
| D/R | Independent integration of finite DCTR posterior versus model reference | How much posterior error comes from the finite likelihood? |
| C/R | Finite-endpoint samples versus model reference | What is the total practical inference error? |

`D` is independently integrated, not another HMC chain. `R` is analytic where
available or independently refined numerical integration. Numerical refinement
allowances are empirical; they are not rigorous integration bounds. B/R and
C/D need not have the same answer as D/R. A correct sampler for `D` cannot
remove a discrepancy between `D` and `R`.

Here “surrogate” identifies the replacement of the acceptance target's gradient
by the DCTR proposal field. B/R is the distinctive surrogate-force experiment.
If C's field is the exact derivative of its own finite DCTR potential, including
the prior and coordinate transformation, C is ordinary HMC for an approximate
posterior. That equality must be checked: sharing a DCTR implementation name
does not establish it across different precision, rounding or derivative paths.
The current experiments do not train a neural network. Numerical integration
error in ordinary leapfrog is separate from this replacement of the force.

## The concrete models

The easiest anchor is LG1, a scalar linear Gaussian state-space model:

\[
x_t=\phi x_{t-1}+\epsilon_t,\quad \epsilon_t\sim N(0,1),\qquad
y_t=x_t+\eta_t,\quad \eta_t\sim N(0,0.5).
\]

The second Normal argument here is a variance. The initial state has variance
`1/(1-phi^2)`. There are ten observations, `phi` has a uniform prior on
`(-.95,.95)`, and the generating value is `.8`. The force uses 1,024 frozen
particles. A Kalman filter gives the exact likelihood, and one-dimensional
quadrature gives an independent parameter-posterior reference. This permits
separate tests of the proposal and the finite likelihood with minimal model
ambiguity. It is an accuracy control, not a setting where a particle score
should be presumed cheaper than Kalman derivatives.

The most recent retained-chain run was SV20, the actual stochastic-volatility
model with twenty observations:

\[
x_t=\gamma x_{t-1}+\epsilon_t,\quad\epsilon_t\sim N(0,1),\qquad
y_t=\beta e^{x_t/2}\eta_t,\quad\eta_t\sim N(0,1).
\]

The stationary initial variance is `1/(1-gamma^2)`. Both inferred parameters
have independent Uniform(.1,.9) priors; generating values are `(.6,.4)`.
The chart is `theta=.1+.8*Phi(z)`, so the uniform prior plus Jacobian becomes
a standard-normal density in `z`. The DCTR field runs in FP32/TF32, while
the integrator and independent quadrature endpoint use FP64. The force uses
log-squared returns to construct flow geometry, but its weights use the raw
return density. Inspection of `actual_sv_model::density` confirms that the
likelihood was not silently changed to a Gaussian log-return approximation.
SV20 B/R completed four chains of 2,000 retained draws; it is INCONCLUSIVE.
All eight chunks were numerically healthy and the original-force audit found
zero fallbacks. The finite-endpoint comparisons have not run.

LG2 is the next linear two-state qualification, with twenty observations:
`x_t=a*A*x[t-1]+epsilon`, covariance `exp(2*b)*Q0`, and
`y_t=x_t+eta`, covariance `.5*I`. Here
`A=[[.7,.2],[.1,.6]]`, `Q0=[[1,.3],[.3,.8]]`,
`a ~ Uniform(.5,1.2)`, `b ~ Uniform(-1,1)`. Its parameter-dependent stationary
covariance is included. Kalman likelihoods remain available. All sixteen
endpoint qualification checks passed; its posterior comparisons have not run.

The catalogue contains seventeen entries, including larger linear systems,
nonlinear scalar models, stochastic volatility and tracking models. The KSC T2
mixture likelihood is a different model from raw-return SV20. M10 is a scalar
nonlinear transition/observation model inferring two noise scales. Its B/R
result is also INCONCLUSIVE. Catalogue membership does not prove a qualified
adapter, reference and completed experiment for every entry.

## What the code and results actually show

The main call chain was inspected through the consumer, shared DCTR executor,
public tuning authority and retained transition, rather than inferred from
the presence of a suitably named function:

| Stage | Inspected source anchor and finding |
|---|---|
| Master entry | `scripts/run_surrogate_hmc_program.py:1` imports `surrogate_hmc_program.main`; `bayesfilter/inference/surrogate_hmc_program.py:1310` dispatches bounded stages. |
| SV20 worker | `scripts/run_surrogate_hmc_sv.py:43` constructs the consumer, binds the proposal/endpoint, invokes `tune_hmc_kernel`, loads verified tuning, and archives retained chunks. |
| Endpoint separation | `bayesfilter/testing/surrogate_hmc_sv_inference_consumer.py:37` supplies the checked independent quadrature endpoint; `surrogate_hmc_sv_consumer.py:196` supplies the signed DCTR field. |
| Actual full DCTR path | `ledh_program_fixtures_tf.py:82` reaches `ledh_canonical_batch_fused_tf.py:310`; it calls `canonical_value_and_analytical_score` including parameter-dependent initial-distribution tangents. |
| Reset and corrections | `ledh_canonical_score_tf.py:553` calls the Contract E reset with particle, covariance and weight tangents; line 590 reaches the shared `batched_higher_moment_shape_jvp` with configured controls. |
| Proposal and acceptance | `neural_force_hmc.py:468` executes the symmetric kicks/drifts; line 604 implements the full endpoint/kinetic acceptance calculation. |
| Adapted coordinates | `neural_force_hmc.py:1084` applies the fixed affine map and its transpose to the field consistently; the public typed binding at line 1191 exposes this to the tuner. |
| Original-force health | `surrogate_hmc_force_health.py:12` rejects retained analysis if the original DCTR field used fallback; the SV worker invokes this before posterior functionals. |
| Uncertainty decision | `surrogate_hmc_equivalence.py:59` computes functional intervals and qualification, with the batch-lag rule at line 94. |

Paths in the last seven rows are under `bayesfilter/inference`,
`bayesfilter/testing`, or `bayesfilter/highdim` as indicated by their role.
Consumer tests mock the expensive canonical evaluator to verify endpoint-to-
engine wiring and sign/chart/status contracts. They do not revalidate every
numerical operation of DCTR. Archived model-specific GPU checks remain the
numerical evidence for those scopes.

**The uncertainty screen has a demonstrated defect.** It requires absolute
lag-one correlation of the longest batch means to be at most `.3` for every
functional. Only ten such batches per chain are used. On exact independent
Gaussian draws, the existing calibration rejected 454 of 512 replications
at one or both looks: 88.7%, with a 95% Wilson interval of 85.6–91.1%.
Most rejections came from negative sample batch correlations. The independent
formula check passed, so this is a poorly calibrated decision rule, not an
identified arithmetic bug. See the
[calibration result](surrogate-hmc-mcse-diagnostic-result-2026-09-20.md).

SV20's final parameter R-hat was about 1.00169, with minimum bulk/tail ESS of
1909/1678, but four functionals failed this batch-correlation screen. These
facts leave the frozen verdict INCONCLUSIVE. They establish neither sampler
bias nor equivalence. Removing the screen without replacement is also unsafe:
on strongly positively correlated controls the MCSE estimate understated the
true uncertainty. A diagnostic randomization/binary-function repair exists
and passed its Gaussian calibration; it has not been qualified for the actual
thirteen-function, four-comparison experiments or adopted by the runtime.

**The retained schedule also lacks adequate prospective precision.** For a
CDF at the true median, four chains of 2,000 independent draws have standard
error `sqrt(.25/8000)=.00559`. With the existing simultaneous critical value
`3.8391`, the nominal interval radius is `.02146`, already wider than the `.02`
margin, before reference error. This is an idealized precision calculation,
not a proof that any realized interval must fail. It provides no justification
for another nominally identical run. See the
[precision result](surrogate-hmc-inference-precision-result-2026-09-22.md).

**SV20's one-step trajectory was imposed.**
`scripts/run_surrogate_hmc_sv.py:28` sets `max_leapfrog_steps=1`.
The tuner did not compare longer trajectories. LG1, KSC and M10 permit at most
two. For one step, integrating out momentum gives

\[
z'=z-\tfrac{h^2}{2}M^{-1}g(z)+hM^{-1/2}\xi,
\qquad \xi\sim N(0,I).
\]

This is a Gaussian proposal with a position-dependent drift. With the exact
potential gradient it is MALA; with the DCTR field it is generalized drift
Metropolis-Hastings. Setting `g=0` yields Gaussian random-walk Metropolis.
[Neal, Section 5.2](https://arxiv.org/abs/1206.1901) derives the gradient case.
The focused check added here independently calculates the Gaussian forward/
reverse density ratio for a nonconservative field and unequal diagonal masses,
and confirms equality with the code's Hamiltonian acceptance ratio. The run
tests a legitimate special case, but does not establish useful long-trajectory
behavior or any computational advantage of HMC.

**The finite endpoint has a separate observed accuracy problem.** For LG1,
independent integration gives finite-DCTR posterior CDF about `.478883` at the
exact posterior median. The difference, `-.0211172`, exceeds the `.02` margin
with empirical quadrature allowance about `.00000803`. Thus D/R is DISCREPANCY
conditional on that numerical-reference qualification. Five other functionals
passed their margins. C/D itself remains INCONCLUSIVE because its median-CDF
interval is too wide. This is one frozen dataset/bank/scope, not a proof that
all DCTR likelihoods are inaccurate. See the
[finite-endpoint result](surrogate-hmc-lg1-finite-result-2026-09-20.md).

**Several remaining models have real representation gaps.** M08's AR2 state
contains a deterministic lag coordinate and singular process covariance
`diag(.09,0)`; the current canonical consumer fails a factorization check.
Adding arbitrary process noise changes the model. M09's nonadditive, shared
innovation law is not represented by the current additive full-rank Gaussian
consumer. M13's simulator wraps a Gaussian angle, whereas the callback uses
only the nearest Gaussian alias. A wrapped-normal likelihood requires summing
aliases. The narrow default noise makes the discrepancy negligible in the
checked case, but that does not qualify the callback for arbitrary inferred
noise scales. These need model-faithful adapter/reference repairs, not more
retained draws. The
[structural](surrogate-hmc-structural-scope-result-2026-09-22.md) and
[range/bearing](surrogate-hmc-range-bearing-scope-result-2026-09-22.md) notes
preserve their tests.

### AR(2): structural deterministic filtering is the intended repair

The owner's 2026-09-22 direction selects the formulation in
[Chapter 18b](../chapters/ch18b_structural_deterministic_dynamics.tex).
For the lag-stack state \(x_t=(m_t,\ell_t)\), the transition is

\[
m_t=\phi_1m_{t-1}+\phi_2\ell_{t-1}+\sigma\epsilon_t,
\qquad \ell_t=m_{t-1},\qquad \epsilon_t\sim N(0,1).
\]

Filtering uses this latent-state law to evaluate the likelihood at a proposed
parameter vector. HMC operates outside that calculation, proposing new parameter
vectors and using the likelihood in their posterior density.

There is one fresh innovation. The lag coordinate copies the previous first
coordinate and inherits its uncertainty; it receives no independent new noise.
Thus \(Q=\operatorname{diag}(\sigma^2,0)\) is correct. A singular Q does not
imply a singular predictive covariance: the latter also includes uncertainty
propagated from the previous state. Chapter 18b explicitly permits an exact
Kalman representation of this affine Gaussian law. Adding noise to the lag
coordinate would change the model.

The repository already contains the relevant structural machinery.
`bayesfilter/structural_tf.py::TFStructuralStateSpace` separates the innovation
space from the completed state. The structural sigma-point implementation in
`bayesfilter/nonlinear/sigma_points_tf.py` forms points for the augmented
previous-state/innovation variable, splits those points, and calls
`model.transition(previous_points, innovation_points)` (line 382).
`tests/test_structural_svd_sigma_point_tf.py` exercises affine AR(2) cubature
and UKF likelihood agreement with a linear Gaussian reference, plus pointwise
deterministic-completion support. These are reference implementations and tests;
they do not establish that the canonical DCTR consumer uses this route.
The current structural sigma-point wrapper explicitly reports `value_only` and
a blocked derivative target (lines 610 and 621); it is not an admitted DCTR
analytical-score implementation.

The current DCTR consumer instead reaches
`bayesfilter/highdim/ledh_canonical_score_tf.py:199`, which attempts Cholesky
factorization of the full process covariance. Its propagation and Gaussian
transition-density calculations also use that full-state factor. Consequently,
the required change is to carry the structural transition, integration measure,
flow/importance-weight calculation and analytical derivatives through the DCTR
call chain. Merely replacing Cholesky with SVD does not supply a full-dimensional
transition density for this degenerate conditional law. The structural UKF is a
useful reference and Gaussian component; substituting its score for the full
DCTR score would test a different proposal mechanism.

This repair addresses AR(2)'s model representation. It does not resolve the
separate uncertainty-screen defect, finite-particle likelihood error or sampler
mixing questions. Structural DCTR force with a Kalman acceptance endpoint remains
a surrogate-force experiment. Kalman value and its own gradient throughout give
the ordinary-HMC reference.

Verification contract for this follow-up: run only the three existing TensorFlow
AR(2) reference tests named above, with GPU deliberately hidden and no XLA or
production-speed claim. Comparator: the existing linear Gaussian likelihood and
the deterministic completion identity. Pass criterion: their existing assertions;
failure vetoes a claim that this reference implementation passed these fixtures,
not the structural mathematical formulation. No HMC, convergence, DCTR wiring
or all-parameter-domain conclusion follows. One attempt is bounded by 150 wall
seconds and 120 CPU seconds (110 child plus 10 parent). Evidence is saved under
`artifacts/surrogate-hmc-all-models-20260922-ar2-structural-reference-01/`.
The skeptical pre-run audit finds these tests suitable for this limited question:
the affine Gaussian comparator is exact, the fixture scope is explicit, and the
checks do not replace the DCTR consumer's known failing singular-Q test. Numerical
floors, test parameters and tolerances remain inherited fixture choices, not
promoted AR(2) training or sampler defaults.

Result: all three existing tests passed (two dependency deprecation warnings),
with selected source hashes unchanged. The explicitly CPU-only check used
5.230534 CPU seconds and 5.527331 wall seconds, with zero GPU time. See the
[run manifest](artifacts/surrogate-hmc-all-models-20260922-ar2-structural-reference-01/checks.json)
and [complete log](artifacts/surrogate-hmc-all-models-20260922-ar2-structural-reference-01/checks.log).
This supplies executable evidence for the reference structural call chain on
these affine fixtures. It does not execute the missing structural DCTR route or
qualify gradients. The strongest remaining implementation risk is treating a
working structural likelihood reference as a completed DCTR adapter. No runtime
algorithm, closed posterior verdict or default was changed in this follow-up.

## Repairs justified by the evidence and literature

First, prospectively calibrate a functional uncertainty procedure and design
its sample size. R-hat and ESS checks remain useful, but ESS 400 is not a
guarantee of the desired CDF precision. [Vehtari et al., Sections 3–4 and the
practical recommendations](https://arxiv.org/abs/1903.08008) explicitly distinguish diagnostic
thresholds from precision for the quantity being estimated. Calibration should
include continuous moments, correlated CDF indicators and predictive
probabilities, all intended looks/comparisons, positive and negative dependence,
between-chain disagreement, drift, and joint interval coverage. It must
quantify false vetoes and false declarations of equivalence.

Spectral variance or batch-means estimates, including lugsail variants, are
literature-grounded candidates for Monte Carlo uncertainty. [Vats and Flegal,
Sections 2 and 4, Theorems 3–4 and Appendix B](https://arxiv.org/abs/1809.04541)
explain how lugsail estimators counter downward covariance-estimation bias
under stated assumptions. Their consistency arguments require growing batch
length and number of batches. The current largest-batch rule keeps ten batches
per chain, so longer chains alone do not invoke that consistency result.
No particular lugsail tuning or diagnostic cutoff is established here. The
existing randomization repair is another candidate, subject to the same actual-
functional calibration. Frozen past outcomes remain frozen.

Second, test the intended trajectory mechanism with a small prospective
length ladder and scope-specific mass/step tuning, frozen before untouched
retained runs. Keep the reference endpoint fixed while comparing the DCTR
field with reference-gradient HMC where available. Construct cheap checks
against Gaussian random-walk Metropolis, a separately fitted fixed Gaussian
independence proposal, and the one-step drift proposal. Examine central and
tail functionals separately. Judge inference using valid uncertainty first,
then compare total tuning plus sampling cost at comparable precision with
replication. High acceptance alone does not establish good exploration.

Third, treat the finite likelihood as its own numerical approximation. Where
a checked reference endpoint is affordable, B/R directly isolates the proposed
DCTR-force mechanism. For C/D and D/R, use fresh scope-specific calibration of
particle count and value-program controls, then independent datasets/banks
and untouched posterior comparisons. More particles are a hypothesis to test,
not a guarantee that finite reset/transport approximations become exact.

Pseudo-marginal MCMC is a possible alternative when a model likelihood cannot
be evaluated accurately enough. [Andrieu and Roberts, Section 2.1, equations
(2.1)–(2.4)](https://arxiv.org/abs/0903.5480) construct an extended target whose
parameter marginal is correct because the auxiliary nonnegative likelihood
weight is unbiased. The random estimator state must be carried consistently
through acceptance/rejection. The existing frozen DCTR bank is not that
construction, and DCTR likelihood unbiasedness has not been established.
Refreshing the bank casually or averaging conditional posterior draws would
not implement the required method. A proved unbiased particle-filter endpoint
would be an explicitly different future design.

Kernel-learned surrogate gradients offer a separate cost-reduction research
direction. [Strathmann et al., Section 3, equation (3), and Algorithm 1](https://arxiv.org/abs/1506.02564)
use a learned field for trajectories and
correct with the target endpoint. This supports the proposal/endpoint
separation, but replacing the full DCTR field would answer an additional
question and requires its own training, freezing, tail and downstream tests.
There is currently no evidence that learning a replacement is the first repair
needed here. Author code is linked by the paper but was not audited; no
implementation-faithfulness claim is made for that optional extension.

For AR2, use Chapter 18b's structural transition and the existing TensorFlow
structural interface; rank-aware factorization alone is insufficient. The
canonical DCTR adapter remains to be implemented and qualified. The other model
repairs are a generative shared-noise adapter for M09 and a normalized
wrapped-normal endpoint with controlled alias truncation for M13. Their
derivatives, support and numerical behavior must be checked before admission.

## Evidence, decisions and continuation

The skeptical pre-check for this turn distinguished the intended endpoint
from its reference, excluded new posterior looks, and chose small deterministic
mechanics/wiring checks that directly address the explanation. The new
one-step check is CPU-only and eager as an explicit tiny reference exception.
No implicit-pfor Jacobian test was run. All thirteen selected checks passed;
full logs, command, source hashes and resource measurements are in
[the run record](artifacts/surrogate-hmc-all-models-20260922-explanation-audit-01/checks.json)
and [log](artifacts/surrogate-hmc-all-models-20260922-explanation-audit-01/checks.log).
The check run took 8.968027 wall seconds and charged 8.219429 CPU seconds,
with GPUs intentionally hidden. No new GPU or posterior work was charged.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Retain the corrected-proposal mechanism for investigation | Derivation and focused mechanics/wiring checks pass | No new kernel defect found in inspected paths | Floating-point and model-wide numerical behavior; actual posterior precision | Calibrated fresh B/R design | All-model correctness, convergence or speedup |
| Repair the inference test before another long run | IID false-veto defect established; nominal precision inadequate | Existing closed functional checks still fail | Actual-function joint coverage and power | Prospective diagnostic calibration and precision/cost plan | Past INCONCLUSIVE results become passes |
| Investigate finite-likelihood accuracy separately | LG1 D/R exceeds one declared margin | That frozen finite target fails its accuracy screen | Other datasets/banks and approximation controls | Fresh value calibration and untouched D/R | Rejection of the DCTR-force research direction |
| Keep unsupported model scopes unqualified | Model representation gaps identified | Singular/nonadditive/angle-law issues remain | General adapter correctness | Small law-preserving reference and consumer checks | Adding noise or changing a likelihood is a harmless repair |

| Inference status | Finding |
|---|---|
| Hard veto screen | Known uncertainty qualification failures; LG1 finite-target discrepancy; remaining model support failures. No new posterior look. |
| Statistically supported ranking | None between samplers. The IID false-veto rate has reported binomial uncertainty. |
| Descriptive-only differences | Existing acceptance, timing, R-hat and ESS; no speed or quality ranking. |
| Default-readiness | Neither a new diagnostic nor an all-model sampler has been established as ready. |
| Next evidence needed | Calibrated actual-function uncertainty, a funded prospective precision design, meaningful trajectory/baseline comparisons, and model-specific reference/adapter completion. |

The strongest alternative explanation for the inconclusive chains is the
measurement procedure and its short schedule. That does not prove the sampler
works: a calibrated, sufficiently precise B/R experiment could still find a
discrepancy. The weakest parts of this audit are generalization beyond the
inspected scopes and the absence of an exhaustive literature search. Primary
papers were fetched/read locally; indexed web search failed with HTTP 502/503.
The bounded [literature ledger](artifacts/surrogate-hmc-all-models-20260922-explanation-audit-01/literature-ledger.json)
records inspected sections, local copies, citation checks and omissions.
No exhaustive forward-citation, correction or retraction screen is claimed.

The master program has been updated in preceding work and dispatches the
current scopes, but it remains incomplete as an all-model validation program.
This turn changed no runtime algorithm or closed inference verdict. It added
the explanatory audit and focused reference check, updated the active plan's
next action, and clarified that SV20's
one-step trajectory was an imposed configuration restriction. Claude review
remains deferred at the owner's request. The next research action is the
prospective uncertainty/precision plan, rather than another underpowered chain.
