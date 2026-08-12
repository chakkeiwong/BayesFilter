# GenUT Observation-Guided Initialization: Fable Review Plan

Date: 2026-08-02  
Status: `FABLE_REVIEW_UNAVAILABLE_WORKSPACE_TRUST_NOT_AUTHORIZED_FOR_EXECUTION`

Review role: Claude Fable read-only mathematical, statistical, and implementation
design review. Review is advisory. No code or research run is authorized by this
document.

## Requested Fable Verdict

Review this plan for the following single decision:

> Is a defensive, observation-guided Gaussian proposal whose moments are generated
> by an exact conditional update where available and otherwise by the repository's
> square-root SVD-UKF machinery, with exact model-density importance correction and
> total proposal derivatives, a mathematically valid and well-staged way to reduce
> the current GenUT route's initialization and first-weighting failures?

End with `VERDICT: AGREE` or `VERDICT: REVISE`. A `REVISE` verdict must identify
the exact mathematical, statistical, source, or implementation blocker and the
smallest repair.

## Executive Decision

Yes, an SVD-UKF may be used for this purpose, but only as a **proposal-moment
generator**. Its approximate filtered likelihood must not replace the GenUT finite
likelihood, and UKF-guided particles must not receive equal weights or ordinary
bootstrap likelihood weights. The corrected weight must contain the exact initial
or transition density, the exact observation density, and the complete proposal
density. The score must differentiate those same terms, including the dependence of
the UKF proposal mean, covariance, stabilizing ridge, and defensive mixture on the
parameters.

The proposed default candidate is not a pure UKF Gaussian. It is a defensive
mixture

\[
q_{t,\theta}^{(i)}(x)
=\rho f_\theta(x\mid x_{t-1}^{(i)})
 +(1-\rho)\,\mathcal N
       \!\left(x;m_{t,\theta}^{(i)},P_{t,\theta}^{(i)}\right),
\qquad 0<\rho<1,
\]

where the Gaussian moments are exact when a conjugate conditional update is
available and otherwise come from a fixed-branch square-root UKF update conditioned
on the current observation. The transition component preserves support whenever
the transition density itself has the required support and bounds the proposal
denominator away from omission of a target region. It does not guarantee low
variance; that remains an empirical gate.

For a directly observed initial state, replace the transition component in this
display by the initial density `p_theta(x_0)`. In the rest of the plan, `b_theta`
denotes the appropriate defensive base density: `p_theta(x_0)` at initialization
and `f_theta(x_t|x_(t-1))` at a transition.

Before this candidate, the project should implement two cheaper changes: exact
antithetic initial clouds and scrambled randomized quasi-Monte Carlo (RQMC) initial
clouds. A generic sample-whitened or exactly moment-balanced cloud is not in the
first ladder because its rows are dependent and its joint proposal law is not the
product of standard Gaussian row densities. Using it inside an ordinary particle
importance formula without a derivation would be wrong.

## 1. The Actual Initialization Problem

The current shared route is
`bayesfilter/highdim/cubature_genut_filter.py::finite_value_score`. It receives a
fixed `initial_noise` tensor and constructs

\[
x_0^{(i)}=T_{0,\theta}\!\left(z_0^{(i)}\right),
\qquad z_0^{(i)}\sim\mathcal N(0,I),
\]

through a model adapter. It immediately assigns `1/N` to every row. There is no
initial proposal-to-prior importance correction in the executed recurrence. The
optional adapter fields `initial_log_density` and
`initial_log_density_tangent` are not read by `finite_value_score` as of this plan.

The name "GenUT initialization" therefore hides two different objects:

1. the random initial/process innovation cloud, which supplies the particle paths;
2. the deterministic GenUT or cubature residual design used only after weighting,
   inside every Contract-E equal-weight reset.

Changing the first object is the scope of this plan. It does not repair a defective
post-weighting residual design, Sinkhorn tuning, higher-moment correction, or
recursive tangent amplification.

### Failure modes

| Failure | Effect on value | Effect on score | Classification |
|---|---|---|---|
| Unlucky but correctly distributed initial cloud | finite-`N` predictive and tail error | pathwise tangent variance may be amplified over time | variance/accuracy problem |
| Cloud drawn from the wrong initial law with uniform weights | targets the wrong first predictive integral | derivative of the wrong finite scalar | mathematical target error |
| Parameter-dependent initial law with a missing/wrong initial tangent | value may remain plausible | score is wrong relative to the claimed total derivative | derivative error |
| Narrow observation far from bootstrap particles | first weights collapse; log-likelihood may be severely noisy | normalized-weight derivatives become extreme | proposal mismatch |
| Bad cloud causes invalid OT/covariance gap | current route returns `NaN` through `program_valid=false` | same | hard numerical veto |
| Early cloud error survives repeated equal-weight resets | likelihood increments drift over the horizon | recursive score variation can grow strongly | finite-route propagation problem |

Existing implementation evidence supports concern about the last row: the
repository's LGSSM review attributes long-horizon variability to initial/process
innovation paths plus finite-cloud evolution, and the Austria score audit identifies
recursive Jacobian amplification. It does **not** isolate initial noise as the sole
cause, so this plan forbids that conclusion.

### Event-time distinction

The proposal must be attached to the first **weighted state**, not blindly to a
variable named `x0`:

- when `transition_before_first_observation=False`, propose the initialized state
  against `p_theta(x_0) g_theta(y_0|x_0)`;
- when `transition_before_first_observation=True`, initialize from the prior and
  propose the transitioned state against
  `f_theta(x_1|x_0) g_theta(y_1|x_1)`; or derive a joint proposal explicitly.

This distinction is binding for the current mixture of LGSSM, SV,
predator-prey, and Austria-SIR event conventions.

## 2. Correct Observation-Guided Formulation

### 2.1 Directly observed initial state

For a model whose first observation is attached directly to the initialized
state, choose a tractable proposal

\[
x_0^{(i)}=m_{0,\theta}(y_0)
           +L_{0,\theta}(y_0)\,z_i,
\qquad z_i\sim\mathcal N(0,I),
\]

with evaluable density `q_{0,theta}(x|y_0)`. The initial unnormalized log weight is

\[
\ell_0^{(i)}
=\log p_\theta(x_0^{(i)})
 +\log g_\theta(y_0\mid x_0^{(i)})
 -\log q_{0,\theta}(x_0^{(i)}\mid y_0).
\]

The likelihood increment is

\[
\widehat Z_0={1\over N}\sum_{i=1}^N\exp\ell_0^{(i)},
\qquad
\Delta\widehat{\mathcal L}_0=\log\widehat Z_0.
\]

This is a different finite estimator from the current bootstrap-initialized
program. Agreement with the current program is not required seed by seed; agreement
with an eligible target/reference within uncertainty is the relevant value gate.

### 2.2 Transition before the first observation and later guided steps

Given a parent `x_(t-1)^i`, sample

\[
x_t^{(i)}\sim q_{t,\theta}
  (\cdot\mid x_{t-1}^{(i)},y_t)
\]

and update

\[
\ell_t^{(i)}
=\log w_{t-1}^{(i)}
 +\log f_\theta(x_t^{(i)}\mid x_{t-1}^{(i)})
 +\log g_\theta(y_t\mid x_t^{(i)})
 -\log q_{t,\theta}
   (x_t^{(i)}\mid x_{t-1}^{(i)},y_t).
\]

For the current GenUT/Contract-E recurrence, weights are uniform immediately
after a valid reset, but the general `log w_(t-1)` term must remain part of the
interface and derivation. Hard resampling is outside the first implementation.

### 2.3 Complete pathwise derivative

For a reparameterized proposal sample

\[
x=m_\theta+L_\theta z,
\qquad
\dot x=\dot m_\theta+\dot L_\theta z,
\]

the score contribution is the **total** derivative at fixed base randomness:

\[
\dot\ell
=D_\theta^{\mathrm{tot}}\log f_\theta(x\mid x_-)
 +D_\theta^{\mathrm{tot}}\log g_\theta(y\mid x)
 -D_\theta^{\mathrm{tot}}\log q_\theta(x\mid x_-,y).
\]

Every displayed total derivative includes direct parameter dependence and the
state paths `dot x_-` and `dot x`. For a Gaussian proposal it also includes
`dot m`, `dot P`, and the factor derivative. Stopping the gradient through UKF
moments, covariance stabilization, the proposal sample, or `log q` computes a
partial derivative and is wrong relative to the total-score claim.

For a defensive mixture, evaluate

\[
\log q=\operatorname{logsumexp}
 \left(\log\rho+\log b_\theta,\;
       \log(1-\rho)+\log q_{\mathrm{UKF}}\right),
\]

and differentiate the log-sum-exp mixture density. The component label is
auxiliary sampling randomness; the denominator is the full mixture density, not
only the density of the sampled component.

## 3. Recommended Candidate And Baseline Ladder

### Arm A: current IID bootstrap baseline

Keep the current stateless IID normal innovation path and current GenUT reset.
This is the exact baseline, not a weak surrogate comparator.

### Arm B: antithetic bootstrap cloud

For even `N`, generate `N/2` IID standard-normal rows and append their negatives.
Apply the same model initial map. This gives exact sample mean zero and preserves
the marginal Gaussian law of every row, but rows are dependent. It changes the
finite estimator and must be compared at equal particle count and wall-clock cost.

Repository evidence already shows partial, coordinate-specific nomination rather
than universal improvement: antithetic averaging reduced variance decisively only
for the SV value in the earlier equal-cost LGSSM/SV campaign. It is an optional
baseline, not a proven default.

### Arm C: scrambled RQMC bootstrap cloud

Generate a randomized scrambled Sobol point set `u_i in (0,1)^d`, apply a guarded
normal inverse CDF, and transform through the same initial law. Each replicate must
use an independent scramble; uncertainty is estimated across scrambles, not by
treating points inside one net as IID. This arm retains the bootstrap target because
it changes the integration design rather than the proposal density.

The inverse-CDF endpoint policy, scramble implementation, power-of-two handling,
base-randomness derivative convention, and XLA/GPU placement must be fixed in the
tuning artifact. Until a source and implementation audit closes these details, RQMC
is `SOURCE_GAP_BLOCKER`, not an executable claim arm.

### Arm D: exact conditional guided proposal

Use the exact locally optimal conditional proposal wherever the model permits it.
For a linear-Gaussian transition and observation, compute the conditional Gaussian
analytically. For a finite Gaussian-mixture observation such as the KSC route, use
the exact normalized component-mixture conditional if all component probabilities,
conditional moments, and the mixture density are available in TensorFlow.

This arm is preferable to SVD-UKF when available: it removes an unnecessary
approximation and supplies a high-value correctness oracle for Arm E.

### Arm E: defensive square-root-UKF guided proposal

For genuinely nonlinear/nonconjugate cases, compute a fixed-branch local Gaussian
approximation of

\[
p_\theta(x_t\mid x_{t-1}^{(i)},y_t)
\]

for each particle. The candidate uses the square-root UKF machinery, preferably the
factor-native route in `bayesfilter/nonlinear/srukf_factor_tf.py`, because it already
exposes filtered mean/factor and their analytical derivatives. SVD is allowed for
the proposal factor only under a fixed, tested degeneracy convention; an
uncontrolled eigenvector-sign or repeated-eigenvalue branch must fail closed.

Use the defensive mixture `rho*f + (1-rho)*q_UKF`, not `q_UKF` alone. Freeze `rho`
within an evaluation. A minimal tuning grid is `rho in {0.1, 0.25, 0.5}`; these are
hypotheses, not defaults. The UKF covariance requires a repository-owned positive
floor/ridge with its realized value and derivative recorded. If the factor or its
derivative is invalid, the candidate fails closed for that tuning row; it must not
silently switch algorithms during a claim run.

`rho` is a frozen, offline-tuned algorithmic control, not a model parameter in the
reported model score. Its derivative is zero in the claim score. Phase 1 nevertheless
tests differentiation with respect to `rho` as an independent mechanics check of the
mixture formula; that test is not part of the scientific parameter score.

### Deferred arm: auxiliary guidance

An auxiliary particle filter may additionally preselect ancestors using an
approximation to `p(y_t|x_(t-1))`, but it changes the ancestor proposal and adds a
second correction. It is deferred until the simpler guided kernel passes. It does
not solve time-zero initialization by itself.

### Rejected/deferred arm: exact empirical moment balancing

Centering and whitening an IID matrix makes the rows jointly dependent and imposes
deterministic constraints. The resulting cloud law is not `prod_i N(z_i;0,I)` and
may be singular on the ambient row space. It may be useful as randomized cubature,
but it cannot enter the ordinary particle proposal ladder until its induced measure
and estimator identity are derived. Random orthogonal frames have the same warning.

## 4. Why SVD-UKF Is Helpful But Not Sufficient

SVD-UKF can cheaply identify an observation-compatible center and covariance.
This directly attacks the narrow-likelihood/remote-observation failure of the
bootstrap proposal. It is not "properly biased filtering" in the corrected particle
route: after exact density-ratio correction, UKF approximation affects variance and
finite-sample behavior, not the intended importance target, subject to support and
integrability.

However, SVD-UKF alone can be a poor proposal when the conditional target is
multimodal, bounded, skewed, heavy-tailed, or close to singular. A single Gaussian
may cover one mode and miss another, or have lighter tails than the target. The
defensive transition mixture is the minimum safeguard. For models whose observation
depends only on a low-dimensional subspace, a still better later design is to guide
only the informed subspace and leave the conditional prior unchanged in its
orthogonal complement. That reduces UKF factor cost and avoids artificially
contracting unobserved directions, but it requires a model-specific conditional
factorization and is not assumed in Phase 1.

The existing `bayesfilter/highdim/ukf_initializer.py` is not reusable as this
proposal. It initializes TT density-training cores and is explicitly classified
`extension_or_invention` and `scout_not_truth`; it does not expose the particle
sample/density/total-JVP contract required here.

## 5. Proposed Interfaces And Implementation Boundaries

Do not add optional booleans to the current adapter and let callers self-attest
correctness. Introduce repository-owned protocols with explicit identities.

```text
InitialProposalAdapter
  sample_and_tangent(theta, observation, base_noise)
      -> particles, particle_tangent
  target_initial_log_density_and_tangent(theta, particles, particle_tangent)
      -> log_p0, total_tangent_log_p0
  proposal_log_density_and_tangent(theta, observation,
                                   particles, particle_tangent)
      -> log_q0, total_tangent_log_q0

TransitionProposalAdapter
  sample_and_tangent(theta, parents, parent_tangent,
                     observation, base_noise, time)
      -> children, child_tangent
  transition_log_density_and_tangent(theta, parents, parent_tangent,
                                     children, child_tangent, time)
      -> log_f, total_tangent_log_f
  proposal_log_density_and_tangent(theta, parents, parent_tangent,
                                   observation, children, child_tangent, time)
      -> log_q, total_tangent_log_q
```

The observation adapter must already return total observation tangents at the
proposed children. Route identity binds model/target/event timing, proposal family,
mixture weight, UKF rule, covariance policy, dtype, TF32, XLA, particle count,
base-design family, and source dependency closure.

The first implementation should be a separate
`finite_value_score_guided_proposal` function. Do not mutate the admitted current
finite scalar in place. Reuse the existing weighting, Contract-E reset, and recursive
tangent code only after a proposal-corrected pre-reset tuple
`(particles, particle_tangent, normalized_weights, weight_tangent, increment,
increment_tangent)` has passed its own validity checks.

TensorFlow/TFP, GPU, XLA-on, FP32/TF32, and memory growth remain the default
candidate execution path. Float64 CPU is permitted only for small exact/reference
checks with `CUDA_VISIBLE_DEVICES=-1`. No NumPy may enter the candidate proposal,
density, selection, or artifact path.

## 6. Correctness Gates Before Any Performance Run

### Phase 0: source and target closure

1. Add a compact source ledger for the sigma-point particle proposal, guided SMC,
   defensive mixtures/support, and RQMC construction.
2. Verify each model's first-observation timing and exact `p0`, `f`, and `g` density
   conventions.
3. Inventory which current adapters can evaluate total `log p0` and `log f`.
   Missing density/tangent functions are implementation blockers, not zero terms.
4. Classify exact conditional, KSC-mixture conditional, UKF approximation, and
   observation-subspace variants separately.

### Phase 1: scalar and LGSSM mechanics

Implement a one-dimensional linear-Gaussian case and the existing three-state
LGSSM first.

Required gates:

- analytic conditional proposal mean/covariance equals the Kalman conditional;
- proposal samples have the declared mean/covariance across independent replicates;
- analytic `log q`, `log f`, `log g`, mixture `log q`, and corrected `log w`
  match independent TensorFlow/TFP evaluations;
- the exact conditional proposal yields the known parent-dependent predictive
  weight in the linear-Gaussian case;
- the defensive mixture at `rho=1` exactly recovers the bootstrap value program;
- fixed-base-randomness total score matches autodiff in float64 and central finite
  differences of the **same guided finite scalar**;
- the proposal score test includes perturbations of mean, covariance, model
  parameters, parent particles, and mixture weight separately (the mixture-weight
  perturbation is a mechanics test only; claim scores hold it fixed);
- SVD/factor branch, ridge activation, event timing, and support failures are
  deliberately triggered and fail as declared.

No nonlinear run is allowed until these gates pass.

### Phase 2: cheap initialization-design comparison

On LGSSM `T in {1, 10, 50}` compare Arms A--D with frozen current reset controls.
The scientific question is whether initialization/proposal design reduces value and
score error at equal total compute, not whether one seed looks favorable.

Use independent outer observation datasets and independent inner randomizations:

- IID seeds for Arm A;
- pair seeds for Arm B;
- scramble seeds for Arm C;
- proposal base-noise seeds for Arms D/E.

Record time-local ESS, maximum normalized weight, log-weight variance, value and
score increments, reset validity, and wall time. Exact Kalman value/score is the
reference. Phase 2 may nominate one cheap arm and validates Arm E against exact Arm
D; it cannot establish nonlinear superiority.

### Phase 3: one nonlinear discriminator

Use exact transformed SV first, not Austria SIR. It has a low-dimensional accurate
reference and makes non-Gaussian observation guidance meaningful at bounded cost.
Compare current bootstrap, the selected cheap arm, exact/numerical conditional
guidance if available, pure UKF Gaussian as a diagnostic, and defensive UKF mixture.

Pure UKF is retained only to determine whether the defensive component trades some
weight variance for support robustness. It is ineligible for promotion by itself.

### Phase 4: high-dimensional transfer only after repair

Austria SIR is deferred until:

- its initial and transition densities and total tangents are explicit;
- the observation-informed subspace is defined without changing the target;
- the current GenUT residual-design and score-instability issues are separated from
  proposal effects;
- a fresh per-scope tuning artifact exists under the LEDH tuning policy.

At `d=18`, a full per-particle UKF using `2d+1` or augmented sigma points may add
substantial work but still be dominated by the current `O(N^2)` transport. Runtime
alone cannot decide promotion; value and score validity remain veto-first.

## 7. Evidence Contract

| Item | Frozen contract |
|---|---|
| Main question | Can corrected initial/observation-guided proposals reduce the current GenUT route's finite-`N` likelihood and total-score error/variance without changing the intended model target? |
| Exact baseline | Current IID bootstrap-initialized `finite_value_score`, with its existing GenUT/Contract-E reset and scope-specific frozen controls |
| Comparator ladder | IID bootstrap; antithetic bootstrap; scrambled RQMC bootstrap after source closure; exact conditional guidance; pure UKF diagnostic; defensive UKF mixture candidate |
| Primary correctness criterion | On LGSSM, the guided finite value and score agree with the exact Kalman reference within predeclared paired uncertainty, and same-scalar derivative parity passes |
| Primary nonlinear nomination criterion | Defensive mixture lowers paired value-and-score MSE or uncertainty width at equal compute on untouched outer datasets, with no coordinate worsening beyond a predeclared noninferiority margin |
| Hard vetoes | Wrong event target; missing `p0/f/g/q` term; support failure; nonfinite or invalid factor; same-scalar FD/autodiff failure; reset invalidity; diagnostic leakage; stale tuning artifact; GPU/XLA/memory-policy failure in claim runs |
| Explanatory only | ESS, maximum weight, log-weight variance, UKF likelihood, moment agreement, runtime, allocator peak, and descriptive per-seed tails |
| What will not be concluded | No exact nonlinear posterior, unbiased log likelihood, unbiased score, universal UKF benefit, HMC readiness, default promotion, Austria repair, or scientific superiority from mechanics/small nonlinear results |
| Artifacts | Fresh versioned roots under `docs/benchmarks/artifacts/genut_guided_initialization_<date>/attemptNN/`, plus plan, run manifest, raw per-replicate rows, and result note |

The score target is the total derivative of each arm's own finite value program at
fixed base randomness. Different proposal arms define different finite estimators.
Same-seed equality between arms is neither expected nor a correctness gate.

## 8. Diagnostics By Role

| Diagnostic | Role |
|---|---|
| Missing density or tangent term | continuation veto |
| Same-guided-scalar autodiff/finite-difference failure | continuation veto |
| Support or factor invalidity | candidate hard veto; repeated scope-wide invalidity is a continuation veto |
| Exact LGSSM value/score disagreement outside declared uncertainty | promotion veto and repair trigger |
| ESS/max weight/log-weight variance | explanatory and possible predeclared promotion veto; never sufficient for promotion |
| UKF approximate likelihood | explanatory only |
| Lower observed SD with few seeds | descriptive only |
| Runtime or allocator peak | engineering veto only at declared budget; never scientific promotion evidence |
| Nonlinear reference disagreement | promotion veto, not automatic rejection of the research direction |

## 9. Statistical And Tuning Design

Every distinct model, horizon, particle count, dtype/backend, proposal family,
mixture-weight family, event timing, and reset-control family is a separate tuning
scope. Prior settings are warm starts only.

For a serious comparison:

1. Use disjoint calibration, validation, and untouched claim datasets.
2. Freeze proposal and reset controls before the claim run.
3. Use common outer datasets and coupled base randomness where mathematically
   meaningful; do not pretend antithetic or RQMC rows are IID.
4. Estimate uncertainty across independent outer datasets and independent proposal
   randomizations/scrambles.
5. Compare equal total compute. Report fixed-`N` results separately because they
   answer a different engineering question.
6. Report paired confidence intervals for value error, each physical-score error,
   MSE, and cost-adjusted error. Do not rank viable arms from means or extreme
   quantiles alone.
7. Apply multiplicity control across score coordinates for any arm-level promotion.
8. Require a noninferiority margin for every score coordinate; a proposal that
   improves value while destabilizing one score coordinate is not score-ready.

Exact sample counts and margins must be set after Phase 1 timing and pilot-variance
data. Choosing them now would be an unsupported default. The pilot may size the
claim campaign but may not contribute claim observations.

## 10. Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|---|
| Current IID bootstrap | executed shared route | exact baseline | weight collapse | `T=1` ESS and exact LGSSM error | baseline |
| Antithetic cloud | prior GenUT campaign | exact marginal symmetry at low complexity | nonlinear sign reflection may not cancel; coordinate worsening | equal-cost LGSSM/SV comparison | optional hypothesis |
| Scrambled Sobol RQMC | proposed by user/agent; source audit incomplete | potentially lower integration error | endpoint, dimension, scrambling, and uncertainty mistakes | independent-scramble Gaussian moment/integral tests | source-blocked hypothesis |
| Exact conditional proposal | guided-SMC identity and conjugate models | removes avoidable UKF approximation | only model-locally available | analytic LGSSM equality | preferred when available |
| Square-root UKF moments | van der Merwe sigma-point particle construction; repository factor derivatives | tractable observation-aware Gaussian approximation | multimodality, light tails, singular covariance, branch instability | exact LGSSM conditional parity and nonlinear weight tails | candidate hypothesis |
| Defensive mixture | support-risk repair derived here | retains transition component and evaluable density | large `rho` gives little guidance; small `rho` may still yield heavy ratios | mixture normalization/support/weight-tail ladder | candidate hypothesis |
| `rho in {0.1,0.25,0.5}` | convenience grid | small bounded tuning family | misses useful scope or overfits validation | fresh per-scope tuning | convenience, not default |
| No hard resampling initially | current deterministic reset and differentiability boundary | isolates proposal effect | may miss a useful APF mechanism | ESS/time-local error trace | scoped design |
| Full-state UKF before informed-subspace variant | implementation simplicity | strongest mechanics oracle | expensive and contracts unobserved directions | cost and covariance-direction audit | Phase 1 only |

## 11. Skeptical Plan Audit

| Risk | Finding and repair |
|---|---|
| Wrong baseline | Keep the current IID bootstrap route as the exact baseline; SVD-UKF standalone filtering is not the comparator target. |
| Proxy promoted | ESS and UKF likelihood are explanatory only; value/score reference error with uncertainty is primary. |
| Missing stop conditions | Density/tangent, support, branch, same-scalar derivative, reference, reset, environment, and budget vetoes are explicit. |
| Unfair comparison | Report equal-compute and fixed-`N` views separately; use common outer data and frozen reset controls within scope. |
| Hidden assumption that UKF removes bias | Rejected. Exact importance correction preserves the intended target; UKF only shapes the proposal. Finite self-normalized and log estimates remain biased. |
| Hidden assumption of Gaussian conditional | Pure UKF is diagnostic; defensive mixture and tail/support diagnostics are mandatory. |
| Stale initializer reuse | Existing TT UKF initializer is explicitly excluded from the particle-proposal interface. |
| Event mismatch | Proposal is defined at the first weighted state, conditional on each route's transition-before-observation flag. |
| Partial score | Total derivatives of sample, target densities, full mixture `log q`, factor, and stabilization are required. |
| Invalid moment-balanced Gaussian claim | Exact empirical whitening is deferred until its dependent cloud measure is derived. |
| RQMC uncertainty error | Independent scrambles are the replication unit; within-net rows are not IID. |
| Artifact cannot answer question | Preserve raw increments, corrected terms `log p0/log f/log g/log q`, total tangents, ESS, branches, seeds/scrambles, controls, time, and reference rows. |
| Wrong high-dimensional first target | Exact transformed SV precedes Austria SIR; LGSSM precedes all nonlinear work. |

Audit verdict: `PASS_FOR_INDEPENDENT_REVIEW_NOT_FOR_EXECUTION`. The proposal
identity and derivative obligations are explicit. Execution remains blocked on
Fable review, source closure for RQMC, adapter density/tangent inventory, Phase 1
sample-size design, and a post-review implementation subplan.

## 12. Pre-Mortem

The campaign could pass while misleading us if it improves ESS but changes the
finite scalar or omits a proposal derivative. Same-scalar derivative parity and exact
LGSSM target comparisons address this.

It could fail for engineering rather than scientific reasons if per-particle UKF
work, factorization, or XLA tensor shapes are poorly implemented. A scalar and batched
LGSSM parity ladder distinguishes that from proposal failure.

It could fail because the Gaussian proposal is wrong rather than because guidance is
wrong. Pure versus defensive proposal comparison, weight-tail diagnostics, and exact
conditional comparators distinguish those explanations.

It could look favorable because the claim datasets were used to select `rho`, ridge,
or reset controls. Disjoint data, repository-issued tuning artifacts, and raw seed
ledgers prevent this.

## 13. Compute And Attempt Budget For The Eventual Campaign

This review plan authorizes no run. If approved and converted into an execution
plan, use the following ceiling unless timing pilots justify a smaller one:

- Phase 1: CPU-only float64 scalar/LGSSM reference tests, each below five minutes;
- Phase 2 pilot: one GPU/XLA timing and variance pilot, at most 20 GPU-minutes;
- Phase 2 claim: at most 90 GPU-minutes and two localized repair attempts;
- Phase 3 nonlinear claim: at most 120 GPU-minutes and two localized repair
  attempts;
- no Austria-SIR campaign under this budget;
- stop if allocator peak exceeds 14 GiB, a density/derivative continuation veto
  fires, or the cumulative GPU budget reaches 210 minutes.

Every GPU command must use trusted/escalated GPU access, verified memory growth,
FP32/TF32, XLA JIT, and a fresh versioned output directory. A serious run manifest
must record commit, dirty-worktree source hashes, command, environment, device,
memory policy, seeds/scrambles, wall time, proposal identity, tuning artifact, plan,
and result paths.

## 14. Source And Claim-Support Ledger

Metadata date: 2026-08-02. Network metadata lookup was not needed for this design
pass. Citation counts and venue rankings were not queried and are recorded as
`not_available`; they are not correctness evidence.

| Source | Class | Local artifact and inspected anchor | Supports | Does not support |
|---|---|---|---|---|
| Chopin and Papaspiliopoulos, *An Introduction to Sequential Monte Carlo* (2020) | `FOUNDATIONAL` | `docs/An Introduction to Sequential Monte Carlo Chopin(20).pdf`; Chapter 10, especially guided PF, Theorem 10.1 local optimality, Examples 10.3 and 10.5, APF section | corrected guided-kernel identity; locally optimal conditional proposal; Gaussian approximations may guide nonlinear models; heavy-tail warning | BayesFilter implementation, score JVP, defensive mixture superiority, GenUT readiness |
| van der Merwe, *Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models* (bibliography says 2004; local PDF metadata/text checked) | `DIRECT_METHOD` | `docs/Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models Merwe(03).pdf`; Sections 5.1--5.2, Eq. 31 and Eqs. 33--35, generic PF Algorithm 1, SPPF update | per-particle sigma-point Gaussian proposal conditioned on current observation; ordinary importance recursion remains required | correctness of the current vendor UPF, defensive mixture, BayesFilter total score, universal improvement |
| BayesFilter particle-filter chapter | `PROJECT_DERIVATION` | `docs/chapters/ch19_particle_filters.tex`, Eqs. `bf-pf-trajectory-proposal` through `bf-pf-sis-recursion` | local target/proposal factorization and support condition | empirical performance |
| BayesFilter high-dimensional filtering chapter | `PROJECT_DERIVATION` | `docs/chapters/ch35_highdim_particle_transport_tensor_filters.tex`, Eqs. `bf-hd-pf-weight` and `bf-hd-corrected-proposal` | local corrected-weight identity and support requirement | proposal optimality or implementation readiness |
| Current GenUT route | `IMPLEMENTATION_EVIDENCE` | `bayesfilter/highdim/cubature_genut_filter.py`, initialization, weighting, reset, score accumulation, fail-closed return | what the current finite program executes | exact likelihood/posterior correctness |
| Repository square-root UKF factor step | `IMPLEMENTATION_EVIDENCE` | `bayesfilter/nonlinear/srukf_factor_tf.py`, filtered mean/covariance/factor and derivatives | reusable proposal-moment/factor derivative machinery candidate | particle proposal correction or nonlinear target correctness by itself |
| Existing GenUT antithetic campaign | `IMPLEMENTATION_EVIDENCE` | `docs/plans/bayesfilter-genut-antithetic-lgssm-sv-result-2026-07-22.md` | antithetic effects were coordinate-specific; no universal default | guided-proposal performance |
| Vendored MLCOE UPF | `IMPLEMENTATION_OR_SOFTWARE`, quarantined as correctness support | `experiments/student_dpf_baselines/vendor/2026MLCOE/src/filters/particle.py`, `UPF.step` | local example of per-particle UKF sampling | **forbidden as correctness support**: executed weights use observation likelihood only and omit transition/proposal ratio |

### Backward snowball and omission risks

The inspected van der Merwe source points to the original UPF/SPPF reports and
particle-filter references. The inspected SMC book cites Pitt--Shephard APF and
Johansen--Doucet APF analysis. These are relevant to a later APF phase, but APF is
not required to decide the present guided-kernel plan.

| Omission/source gap | Reviewer risk | Required action |
|---|---|---|
| Original unscented particle filter technical report / peer-reviewed successor not locally inspected | medium: direct method lineage | acquire and inspect before publication-grade SPPF claims; van der Merwe Section 5 is sufficient for this internal design review only |
| RQMC/SQMC primary source and TensorFlow implementation not inspected | high for Arm C | block Arm C execution until primary algorithm, scrambling, inverse-normal mapping, and replicate inference are sourced and tested |
| Defensive-mixture importance-sampling primary source not inspected | medium | either add a checked primary source or retain the support claim as the elementary project derivation `q >= rho*f` plus direct tests |
| Heavy-tailed or mixture proposal alternatives | medium for nonlinear multimodality | evaluate only if pure/defensive Gaussian weight tails fail; not needed in Phase 1 |
| Differentiable resampling literature | low for current scope | hard resampling is explicitly excluded; inspect before adding APF/resampling gradients |

Forward citation snowballing was not performed because no external metadata query was
needed or authorized for this compact internal review. This is a publication-grade
coverage gap, not a blocker for Fable's bounded plan review.

## 15. Questions Fable Must Answer

1. Are the initial and transition guided weights written for the correct target and
   event timing?
2. Does the fixed-base-randomness score require any term not listed here, especially
   for the mixture density, parent-state path, covariance stabilization, or factor
   derivative?
3. Is the defensive mixture `rho*f + (1-rho)q_UKF` the right first candidate, or is
   there a simpler proposal with equal support protection and an evaluable total
   derivative?
4. Should the first nonlinear target be exact transformed SV, or is another current
   adapter a materially better discriminator with an eligible value/score reference?
5. Is exact KSC component-mixture guidance feasible under the current adapter, and
   should it precede a UKF approximation?
6. Is the warning about exact empirical moment balancing correct, or can a valid
   randomized construction enter the particle likelihood with a simpler derivation?
7. Are the promotion criteria too weak, too strong, or aimed at the wrong estimator?
8. What primary source is the largest omission risk before implementation?

## 16. Required Post-Review Actions

If Fable agrees:

1. incorporate any non-material clarifications without changing the scientific
   target;
2. close the adapter density/tangent inventory;
3. acquire and audit RQMC/SQMC sources or remove Arm C from the first campaign;
4. write a bounded Phase 1 implementation subplan with exact test names and artifact
   schema;
5. implement only scalar/LGSSM mechanics;
6. obtain timing/variance pilot evidence before fixing serious sample counts;
7. write a separate serious Phase 2/3 experiment plan before GPU claim runs.

If Fable revises, do not implement until the identified mathematical target,
derivative, support, comparator, or evidence flaw is repaired.

## Fable Review Attempt

On 2026-08-02 Codex launched the bounded prompt below with
`scripts/claude_worker.sh`, model `claude-fable-5`, high effort, trusted/escalated
execution, and this file as the only review path. Claude Code reported that the
workspace trust dialog had not been accepted and that local permission entries were
therefore ignored. It then produced no review text or verdict and was terminated
after a bounded wait.

This is an external reviewer-availability limitation, not `AGREE`, `REVISE`, or
scientific evidence. No user-level trust setting was changed. Rerun the identical
one-path prompt after the workspace trust prerequisite is resolved; do not weaken
the plan merely to obtain agreement.

## Bounded Review Prompt

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter-genut-observation-guided-initialization-fable-review-plan-2026-08-02.md.
Do not edit, run commands, launch agents, or review the whole repo. Question: Is
the proposed defensive observation-guided GenUT particle initialization/proposal
plan mathematically target-correct, total-derivative complete, statistically fair,
and staged safely enough to proceed to a scalar/LGSSM implementation subplan?
Answer the eight questions in Section 15. End with VERDICT: AGREE or VERDICT:
REVISE.
```
