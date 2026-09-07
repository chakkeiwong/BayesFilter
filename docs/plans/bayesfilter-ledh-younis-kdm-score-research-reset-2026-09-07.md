# Younis KDM for the LEDH-OT-GenUT Score

Date: 2026-09-07

Status: PHASE4A_ENGINEERING_COMPLETE_TF32_VETO; PHASE4B_MATH_BLOCKED; NO RESEARCH CAMPAIGN YET

This plan corrects the research question after the earlier
Fisher/FFBSm/PaRIS rewrite proposal. The earlier proposal is preserved as
historical material and is explicitly superseded. The spelling LEDH is used
below because it is the name used by the repository and by the Li--Coates
particle-flow route; if LEDG denotes a different algorithm, the route name
must be changed before implementation.

## Research question

Can the kernel-density-mixture (KDM) construction of Younis and Sudderth be
used to compute a lower-error score for the *executed* LEDH-OT GenUT
dual-cap trust-region filter, while preserving a precisely declared target,
the complete derivative through that target, and the structural support of a
degenerate DSGE model?

The question is not whether a generic particle smoother can be attached to
the project. Nemeth, Scibior--Wood, PaRIS, and Del Moral are not the active
method family for this investigation. Corenflos is part of the existing OT
context. They may be cited for boundaries, but no new implementation is
authorized by this plan.

## Research intent ledger

| Field | Frozen statement |
|---|---|
| Main question | Does Younis-style KDM reduce score error or score variance for the actual LEDH-OT-GenUT dual-cap program? |
| Candidate mechanism | Phase 4A tests exact Gaussian kernelization of the observation factor with full reset feedback; Phase 4B may later test Younis's fixed-proposal importance-weighted mixture gradient after its complete sequential proposal law is derived. |
| Expected failure mode | Positive bandwidth changes the finite state measure; a full-dimensional kernel leaves a degenerate DSGE support; mixture-density gradients can hide an all-pairs interaction; stopping a cloud, weight, or bandwidth tangent yields a partial derivative. |
| Primary promotion criterion | A candidate must pass zero-bandwidth parity, finite-difference derivative parity for its declared positive-bandwidth program, and a predeclared score-error comparison at equal compute on an oracle case. |
| Promotion veto | Non-finite or support-invalid states, a target change while the route is advertised as unchanged, omitted derivative terms, hidden quadratic work, stale scope settings, or failed endpoint mechanics. |
| Continuation veto | Corrupted observations or artifacts, an invalid oracle, missing required diagnostics, or exhaustion of the bounded campaign budget. A failed candidate alone is a repair trigger. |
| Repair trigger | Re-derive the insertion point, constrain the kernel to the innovation subspace, expose the missing tangent, or move the all-pairs operation to a labelled reference lane. |
| Explanatory diagnostics | Bandwidth, mixture overlap, effective sample size, mode coverage, cap activity, covariance spectrum, tangent norms, memory, and wall time. These do not certify correctness. |
| Nonclaim | No positive-bandwidth result is called the score of the zero-bandwidth program or the exact model score without a separate proof and oracle evidence. |

## Objects that must not be conflated

For a fixed observation sequence and fixed random stream xi, write the
canonical finite program as

~~~
x^-_{t,i} = Phi_{t,theta}(x^+_{t-1,i}, xi_t)
x^out_{t,i}= F^LEDH_{t,theta}(x^-_{t,i}; xi_t)
a_{t,i}  = log w^-_{t,i}
            + log p_theta(x^out_{t,i} | x^+_{t-1,i})
            + log g_theta(y_t | x^out_{t,i})
            + log |J_{t,i}|
            - log q_{t,theta}(x^-_{t,i} | x^+_{t-1,i}, y_t)
Z_t      = sum_i exp(a_{t,i})
w^+_{t,i}= exp(a_{t,i}) / Z_t
x^+_t    = R_theta(x^out_t, w^+_t; xi_t)
L_0^N(theta; xi) = sum_t log Z_t
S_0^N(theta; xi) = d L_0^N(theta; xi) / d theta
~~~

Here Phi includes the declared transition and UKF covariance lifecycle, and
F^LEDH is the flow map. The displayed logits make the PF--PF correction
explicit: x^out is the post-flow state, J is the forward flow Jacobian, and q
is the pre-flow proposal density. In a bootstrap/no-flow special case the
transition and proposal terms cancel and a reduces to log w^- plus the
observation log density. R includes the actual OT/Sinkhorn, Contract-E,
GenUT moment, and dual-cap trust-region operations. If a concrete
implementation contains an additional value term, it must be listed
explicitly; it may not be silently dropped from the value or score.

The exact state-space score, when it is the scientific target, is a separate
quantity, grad_theta log p_theta(y_1:T). The present plan does not assume that
it equals S_0^N. The first implementation question is narrower and
checkable: can KDM improve estimation of S_0^N, or must it define a new
finite target?

For the actual implementation, the one-step logits are factored before the
observation term.  With `b_i = log w_i^- + log f_i + log|J_i| - log q_i`, the
code forms `B = logsumexp(b)`, `bar_w_i = exp(b_i-B)`, then `a_i=b_i+log g_i`,
`log Z=logsumexp(a)`, and posterior weights `w_i^+=exp(a_i-log Z)`.  The
trace exposes both `bar_w_i` (`prior_observation_weights`) and `w_i^+`
(`posterior_weights`).  A KDM observation functional must use `bar_w_i`;
using `w_i^+` and multiplying by the observation factor again is the
double-weighted estimand documented in the historical Section 3.6 memo.

For a weighted cloud, Younis's regularized representation is

~~~
mu_0^N(dx) = sum_i w_i delta_{x_i}(dx)
mu_h^N(dz) = sum_i w_i K_{B_i}(z - x_i) dz
m_theta(z)  = sum_i w_i(theta) k_{B_i(theta)}(z - x_i(theta)).
~~~

The zero-bandwidth object is the atom measure. Any h or B that is positive
on a nontrivial direction defines a different measure unless an exact
correction is supplied.

The experiment uses three explicit target labels. `ATOM-FINITE` is the
canonical scalar `L_0^N` produced by the existing finite LEDH-OT-GenUT
program. `KDM-FINITE` is the scalar obtained after a declared positive-
bandwidth operation; its derivative is `d L_h^N / d theta`. `MODEL-IS` is an
importance-sampling estimator whose proposal may be a KDM but whose density
ratio targets the underlying one-step state-space factor. A `MODEL-IS` ratio
can preserve the model normalizer in expectation when its support and density
are correct, but it does not make the finite Contract-E reset equal to the
atom program. Every result must carry one of these labels; the word
"target-preserving" is reserved for preservation of the stated model or
finite-program target, not for a generic smoothing operation.

## What the Younis papers establish

The local full texts are

- docs/papers/differentiable/Differentiable and stable long-range tracking of multiple posterior modes Younis(23).pdf
- docs/papers/differentiable/Learning to be smooth An end-to-end differentiable particle smoother Younis(24).pdf

The checked technical anchors are:

| Source anchor | Result supported | Boundary here |
|---|---|---|
| Younis--Sudderth 2023, Sec. 2.2, Eq. (4) | A weighted Dirac cloud is represented by a continuous kernel mixture and can be resampled from that mixture. | This changes the finite state measure used after resampling. |
| Younis--Sudderth 2023, Sec. 4, Eqs. (14)--(15) | IWSG fixes a proposal at the current parameter value and differentiates mixture importance weights rather than moving sampled locations. | The derivative is for a mixture expectation under the paper's objective, not automatically for an atom-based LEDH value. |
| Younis--Sudderth 2023, App. B.1 | Mixture-gradient training evaluates mixture interactions and is quadratic in particle count. | The linear inference cost cannot be transferred to a score path that differentiates the mixture. |
| Younis--Sudderth 2024, Secs. 2.3 and 4, Eqs. (6)--(7), (17)--(23) | MDPF/MDPS use Gaussian mixtures, stratified mixture resampling, and an importance-weighted two-filter construction. | MDPS is a future-data smoother and its training/gradient path has all-pairs mixture work; it is not the online LEDH score. |

The papers study learned discriminative tracking and smoothing. They do not
prove that KDM removes finite-particle bias in a generative parameter score,
and they do not analyze OT, LEDH flow, GenUT moments, or dual-cap constraints.
Any adaptation to this repository is therefore classified as a new
extension, even when the mixture and IWSG algebra is source-faithful.

## The mathematical boundary

For any test function phi,

~~~
int phi(z) mu_h^N(dz)
  = sum_i w_i int phi(x_i + u) K_{B_i}(du)
  != sum_i w_i phi(x_i)
~~~

in general. This is the reason a positive-bandwidth KDM cannot be described
as an exact estimator of the atom program merely because it is smooth. The
logarithm does not remove this shift. For a KDM observation increment,

~~~
bar Z_t = sum_i w^-_{t,i} int g_theta(y_t | x^-_{t,i} + u) K_{B_{t,i}}(du),
bar L^N = sum_t log bar Z_t,
~~~

bar L^N is a new finite scalar. In the linear-Gaussian observation case,
g_theta(y | x + u) with u distributed as N(0,B) integrates exactly to

~~~
bar g_theta(y | x,B)
  = Normal(y; C_theta x, R_theta + C_theta B C_theta^T).
~~~

This identity is valuable for a controlled experiment, but it is not an
identity with the original observation factor unless B is zero.

### IWSG differential

At a reference parameter theta0, draw fixed locations z_j from
q_0(z)=m_{theta0}(z), and define

~~~
rho_j(theta) = m_theta(z_j) / q_0(z_j).
I(theta)     = int phi_theta(z) m_theta(z) dz
I_hat(theta) = (1/M) sum_j rho_j(theta) phi_theta(z_j).
~~~

At theta0, rho_j equals one, while

~~~
d I_hat = (1/M) sum_j [d rho_j phi_theta(z_j)
                       + rho_j d phi_theta(z_j)].
~~~

The sample locations are held fixed in this construction. That is the
Younis IWSG convention; it is not the total JVP of a deterministic
parameter-dependent inverse-CDF or mixture-resampling map.

For a Gaussian component k_i with displacement u_i=z-x_i, the complete
component differential is

~~~
d log k_i =
    u_i^T B_i^{-1} d x_i
  + 1/2 [u_i^T B_i^{-1} (d B_i) B_i^{-1} u_i
          - tr(B_i^{-1} d B_i)].
d m_theta(z) = sum_i [(d w_i) k_i + w_i (d k_i)].
~~~

Consequently, an IWSG score path must include every declared dependence of
the cloud, weights, kernel scale, LEDH parameters, OT map, and GenUT
correction. A route that differentiates only the observation factor or only
the transported locations is a partial derivative relative to its own
declared scalar.

For a finite importance-weighted observation normalizer,

~~~
Z_hat_h(theta) = (1/M) sum_j rho_j(theta) g_theta(y_t | z_j),
d log Z_hat_h =
  [sum_j rho_j g_j (d log rho_j + d log g_j)]
  / [sum_j rho_j g_j].
~~~

This is a legitimate derivative of the KDM importance-sampling program.
It is not S_0^N unless a separate equality or an exact residual correction
is proved.

### Degenerate DSGE support

In a degenerate transition, a full-dimensional kernel can put mass outside
the model support. The first admissible constrained construction is

~~~
u = G_theta L_beta epsilon,
B_theta = G_theta H_beta G_theta^T,
epsilon ~ Normal(0,I),
~~~

with zero variance in deterministic directions. If a constraint map D
describes a linear support, require D G_theta=0. For a nonlinear manifold,
use an explicit chart or retraction and include its Jacobian in the mixture
density. A rank-deficient ambient Gaussian has no ordinary Lebesgue density;
the implementation must evaluate a density in innovation or chart
coordinates, not pretend that a full SPD ambient covariance exists.

Constraining the kernel in this way is an extension of the Younis construction,
not a claim made by the source papers. It is required by the DSGE target and
must be tested as such.

## Candidate uses, with honest labels

### 1. Auxiliary KDM estimator or control variate

Keep the canonical LEDH-OT-GenUT value and reset unchanged. Build a KDM from
the actual cloud and evaluate a separately named mixture score. It may be
used as a control variate or exploratory force. To preserve the target
S_0^N, any control-variate correction must retain the original score and use a
known or independently estimated expectation; otherwise it is only a changed
estimand. A force used with the canonical energy is a proposal field, not the
canonical score.

This is the lowest-risk first test because it answers whether KDM contains
useful score information without silently changing the filter.

### 2. KDM proposal with an exact model-target ratio

There is one direct KDM use that does not replace the model factors.  At a
continuous, density-bearing stage choose an ancestor `J` with probability
`r_t(j)` and draw a pre-flow state from a conditional mixture
`q_t^KDM(x | j, y_t)`.  Apply the existing LEDH map to obtain `X_1`.  The
joint proposal density is

~~~
q_t(j, x_1) = r_t(j) q_t^KDM(x_0 | j, y_t) / |det D F_t(x_0)|,
~~~

so the corrected extension weight is

~~~
omega_t = w^-_{t-1,J} f_theta(X_1 | x^+_{t-1,J})
           g_theta(y_t | X_1) |det D F_t(X_0)|
           / [r_t(J) q_t^KDM(X_0 | J, y_t)].
~~~

The complete mixture density, not the sampled component density, is required
in the denominator.  An APF lookahead may choose `r_t`, but its approximate
predictive likelihood must not replace the exact `f_theta` or `g_theta`
factors.  This is a `MODEL-IS` correction for the one-step proposal.  It can
be useful for estimating the model normalizer and its derivative in
expectation, and it can feed the unchanged LEDH flow before the existing
reset.  For `M` proposal draws, `hat Z` and `d hat Z` are unbiased under the
support, invertibility, and differentiation-under-the-integral conditions.
The ratio `d hat Z / hat Z` is generally a biased finite-`M` estimator of the
score `d log Z`; consistency and any MSE improvement must be tested rather
than asserted.  The one-step identity does
not establish equality with `ATOM-FINITE`: the deterministic OT/Contract-E
reset remains a separate finite transformation, and its score must still be
differentiated in full.

Following the fixed-denominator principle of Younis IWSG, the BayesFilter
`MODEL-IS` adaptation fixes `r_t` and `q_t^KDM` at an anchor `theta_0` while
differentiating the model numerator and LEDH map.  Younis does not derive this
conditional LEDH extension.  If an implementation instead recomputes a
proposal as `theta` changes, it needs a separate derivation that couples the
sampling map, proposal density, ancestor selection, and state map.  Merely
subtracting proposal or selection log-density tangents from a fixed-sample
weight is wrong for the anchored model-normalizer derivative.  This route
therefore gets a distinct estimator and route ID, and is compared with both
the canonical `ATOM-FINITE` score and the exact Kalman score.

At the atom reset itself, ordinary importance ratios do not repair the
change: a finite atom measure and an absolutely continuous KDM are mutually
singular.  The KDM proposal must be inserted before a continuous transition or
be accompanied by an explicitly defined new measure and correction.

### 3. Integrated KDM-LEDH finite program

Replace a declared atom operation by a KDM operation and propagate the
result through the same LEDH flow, PF--PF weight correction, OT/Sinkhorn,
Contract-E, GenUT moments, and dual-cap trust region. Two insertion points
must be kept separate:

1. convolve the observation factor before the weight update; or
2. represent the post-reset cloud by a constrained mixture before the next
   transition.

Each defines a new scalar L_h^N and a new route identifier. The score is
d L_h^N/d theta, computed with the full IWSG or analytical mixture tangent.
Neither route may inherit the canonical route ID or be called the score of
L_0^N.

### 4. Exact-preserving residual/control-variate construction

If a smoothed functional A_h is useful, retain the exact residual:

~~~
L_0^N = A_h + (L_0^N - A_h),
d L_0^N = d A_h + d(L_0^N - A_h).
~~~

Dropping the residual changes the target. Keeping it exactly may leave no
variance reduction, so the only useful version must demonstrate a genuine
control-variate covariance reduction and must freeze its coefficient on
calibration data. This is a hypothesis to test, not a presumed repair.

### 5. Full Younis MDPF/MDPS

The source-faithful all-components mixture gradient is a reference
implementation only. It is expected to be O(N^2) per mixture-gradient
evaluation and O(TN^2) for a sequential gradient. Sparse neighborhoods,
random features, or low-rank approximations would be new algorithms and must
not be presented as Younis's exact method. No quadratic route is eligible for
the production score lane under the current budget.

## Execution phases

### Phase 0: document and target freeze

Complete this note and the accompanying LaTeX document. Define L_0^N, S_0^N,
L_h^N, and the IWSG anchor in one notation. Add the source boundary,
degenerate-support construction, route labels, and nonclaims. Mark the
Fisher/PaRIS proposal superseded. No implementation change occurs in this
phase.

### Phase 0A: canonical endpoint wiring repair

Before adding KDM code, resolve the existing registry mismatch under the
implementation-audit rule.  Inspect every consumer of the registered
`canonical_value_score_and_diagnostics` name, choose a repository-owned
adapter only if its signature and diagnostics contract require one, and make
the endpoint resolve to `canonical_value_and_analytical_score` (or to an
explicit adapter that calls it).  Add a wiring test that follows the call
chain, passes `reset_policy=contract_e`, and fails if it resolves to
`finite_value_score`, an autodiff oracle, or the historical sidecar.  This is a
bounded correctness repair, not a KDM experiment; no KDM result is admissible
until it passes.

Completion status: passed on 2026-09-07. The registry now names
`canonical_value_and_analytical_score`, and an executable discovery test
resolves every registered entry point to a callable. The same audit exposed a
batch API drift: `canonical_batch_value_score` accepted `substeps` but its
tests and the authority use `flow_substeps`; the wrapper now accepts both,
rejects conflicting values, and forwards the resolved value as
`flow_substeps`.

### Phase 1: Younis-aligned KDM algebra and proposal diagnostic

Implement only diagnostic TensorFlow kernels, in two separately labelled
pieces.  First implement the Younis-aligned mixture algebra and its analytical
tangent:

- normalized mixture density and log density;
- cloud, weight, and bandwidth tangents;
- fixed-proposal IWSG contribution;
- explicit complexity and rank checks.

Then implement the proposal diagnostic as a distinct extension:

- conditional KDM proposal density, support check, and PF-PF correction for a
  `MODEL-IS` arm;
- explicit forward-map invertibility and log-determinant checks;
- a fixed-anchor derivative that includes the exact target factors but does
  not silently differentiate a moving proposal.

Use a small all-pairs reference for parity, and keep it out of any admitted
runtime path. Do not use NumPy in the algorithmic path.

### Phase 2: zero-bandwidth and finite-difference gates

For a fixed stream and identical cloud:

- B=0 or an exact atom limit must reproduce the canonical value and score;
- B=0 is an explicit atom branch; never attempt a Cholesky or inverse of a
  zero covariance to manufacture that limit;
- positive B must report L_h^N-L_0^N and may not claim value identity;
- finite differences of L_h^N must agree with the complete declared tangent;
- perturbing x, w, B, and model parameters separately must expose each term;
- no claim of a lower score error is made from parity alone.

### Phase 3: auxiliary route on the actual call chain

Wire the diagnostic evaluator from the actual canonical endpoint, not from a
standalone helper. The call chain must be:

~~~
claim-bearing caller
  -> canonical LEDH value/reset producer
  -> captured pre/post cloud with declared weight semantics
  -> KDM evaluator
  -> explicitly labelled auxiliary score or control-variate result
~~~

Run first with the canonical value untouched. Verify that the sidecar cannot
silently feed its weights or states back into Contract-E or the dual-cap
reset. If a control variate is attempted, freeze its coefficient on
calibration paths and estimate uncertainty on disjoint paths.

### Phase 4: integrated candidates

The skeptical audit split this phase.  The governing plan is
`docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md`.

Phase 4A replaces the point observation factor by the exact
linear-Gaussian convolution
`N(y; C x, R + C B C^T)`, then sends the changed weights and their total
tangent through the existing Contract-E, GenUT, and both dual-cap correction
mechanisms.  It shares the canonical recurrence rather than copying it.  This
is `KDM-FINITE` kernelized observation weighting.  It is not a full Younis
mixture-density particle filter because it does not propagate each mixture
component's conditional posterior mean and covariance.

Phase 4B is the complete fixed-anchor `MODEL-IS` mixture-resampling reference.
It is mathematically blocked until the component cloud, covariance/support,
sampling law, ancestry semantics, complete proposal, matching numerator,
LEDH map/Jacobian, sequential anchor, and posterior/reset payload have all
been derived.  The conditional density and anchored weight helpers do not by
themselves define that algorithm.  No Phase 4B code may substitute a sampled
component density, omit all-pairs terms, or invent an unstated proposal.

### Phase 5: oracle and DSGE validation

Use the following ladder:

1. linear-Gaussian model with exact Kalman value/score;
2. the canonical zero-bandwidth LEDH-OT-GenUT finite program;
3. the auxiliary KDM arm with unchanged canonical value;
4. the integrated KDM-LEDH arm with its new target;
5. degenerate DSGE fixtures with explicit support constraints.

Primary comparisons use paired observation paths and independent fixed
streams, with score bias and variance reported separately. For a new target,
also report the value shift, bandwidth ladder, and N ladder. A nonlinear DSGE
comparison without an exact oracle is descriptive only.

The constructed heuristic adversary set is:

- the current canonical analytical JVP with no KDM;
- a constrained innovation-jitter proposal with no KDM score replacement;
- a plain bootstrap particle score on the nondegenerate LGSSM reference;
- the KDM auxiliary route with the canonical score retained.

Evaluate these conditionally in short versus long horizons, degenerate versus
nondegenerate support, separated versus overlapping modes, and small versus
large N. The Kalman solution is an oracle, not a heuristic. Losing to a
simple admissible comparator blocks promotion in that situation.

### Phase 6: mechanics and promotion decision

If an auxiliary force is used for HMC mechanics, retain the exact declared
finite energy and test reversibility, replay, Jacobian, and endpoint energy
accounting. If L_h^N is used as the energy, call it a new regularized target
and test its own gradient and posterior consequences. No canonical LEDH
promotion, default change, or HMC scientific claim follows from a finite
mixture parity test.

## Code and document crosswalk

The single-cloud analytical score authority is:

- bayesfilter/highdim/ledh_canonical_score_tf.py,
  canonical_value_and_analytical_score.

The batch wrapper is
bayesfilter/highdim/ledh_canonical_batch_tf.py,
canonical_batch_value_score. It accepts the historical `substeps` keyword and
the authority-aligned `flow_substeps` alias, with an explicit conflict check.
The source-route initialization and score marks
are in bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py, and the
documented LEDH/PF--PF boundary is in
docs/chapters/ch19c_dpf_implementation_literature.tex.

The separate bayesfilter/highdim/cubature_genut_filter.py,
finite_value_score, is a lower-level GenUT candidate/diagnostic. It does not
establish that the LEDH PF--PF flow and Contract-E reset were called.

The call-chain audit found and repaired a governance discrepancy:
bayesfilter/highdim/ledh_alg1_contract.py registered the nonexistent
canonical_value_score_and_diagnostics name while the analytical implementation
exported canonical_value_and_analytical_score. The registry now names the
exported analytical endpoint, and
tests/highdim/test_ledh_canonical_governance.py imports every registered module
and verifies that every endpoint resolves to a callable. This closes registry
discovery only; it does not implement KDM or prove numerical score validity.

The analytical function defaults to reset_policy=none, which is a diagnostic
slice. Every canonical baseline invocation for this plan must explicitly use
reset_policy=contract_e and the owner-registered dual-cap/trust-region
settings; otherwise it is not the LEDH-OT-GenUT program named here.

The Phase 1 diagnostic implementation is
`bayesfilter/highdim/ledh_younis_kdm_tf.py`.  Its route ID is
`ledh_younis_kdm_algebra_diagnostic_v1` and its classification is
`source_aligned_extension_diagnostic_only`.  The global and conditional
mixture evaluators expose log density, responsibilities, complete cloud/weight/
covariance tangents, rank/normalization validity, and an explicit all-pairs
counter.  The source-aligned IWSG evaluator has no sample-location tangent and
keeps the proposal log density fixed at the supplied anchor.  The conditional
KDM kernel is labelled as a proposal-density evaluator, not a model-score
estimator.  The anchored `MODEL-IS` helper is
`make_anchored_pfpf_kdm_weight_kernel`; its value contains the transition,
observation, forward log-determinant, ancestor-selection, and complete
conditional-mixture denominator.  Its tangent contains only the moving model
numerator and LEDH-map terms: proposal and selection tangents are absent by
construction.  It requires normalized ancestor/selection laws and a checked
per-particle `forward_map_valid` certificate.  These kernels have static
TensorFlow signatures and XLA-on defaults, but remain outside the canonical
registry and claim-bearing call chain.

The controlled positive-bandwidth scalar from the LaTeX convolution identity
is `make_linear_gaussian_kdm_normalizer_kernel` in the same module.  Its
`bandwidth_is_zero=True` branch is explicitly `ATOM-FINITE`; the positive
branch is `KDM-FINITE`.  It differentiates state, normalized weights,
bandwidth, observation matrix, observation covariance, and observation, and
rejects a rank-deficient positive ambient covariance.  This is a one-step
linear-Gaussian gate, not an integrated LEDH target.

The fixed-chart rank-aware evaluator is `make_subspace_gaussian_kdm_kernel`.
It uses `fixed_orthonormal_support_chart_v1`, computes the mixture in the
support coordinates, and rejects off-support points, off-support tangents,
non-orthonormal charts, and invalid coordinate covariances.  A
parameter-dependent DSGE chart and its Jacobian are intentionally not hidden
inside this diagnostic.

Phase 3 is implemented by `canonical_linear_gaussian_kdm_auxiliary`.  It calls
the analytical endpoint once with `return_trace=True`, requires the canonical
Contract-E and dual-cap mechanisms, checks the supplied linear observation
callbacks, reconstructs the `ATOM-FINITE` value and score from the
pre-observation factorization, and evaluates a separate `KDM-FINITE`
observation functional.  The KDM result is explicitly marked no-feedback and
cannot alter the canonical reset trajectory.

Phase 4A is implemented by
`bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py`.  Its eager reference
and fixed-shape TensorFlow factory call the same internal executor as
`canonical_value_and_analytical_score`, replacing only the observation factor
with `N(y; C x, R + C B C^T)`.  The endpoint requires Contract-E, the
diagonal and pairwise corrections, and a positive coordinate cap.  Changed
weights and their total tangent feed the actual reset and all later steps.
It is labelled `kdm_finite_full_feedback_diagnostic_only`, records that it is
not a complete mixture posterior, and is not registered as canonical.

The Phase 4A factor differentiates `x`, `C`, `R`, and `B`.  Rank-deficient
positive-semidefinite `B` is permitted because only the effective observation
covariance is factored.  At every step the wrapper checks the supplied linear
map and its tangent against the model callbacks, then checks the explicit
zero-bandwidth Gaussian value and tangent against the model's executed atom
density.  A mismatch fails closed.

The shared analytical executor was repaired during this audit.  Generic
Gaussian transition and observation fallbacks now include the direct `dQ`
and `dR` log-density terms.  `NonlinearScoreModel` can supply a total
observation-Jacobian tangent, whose contributions now enter every LEDH flow
coefficient.  The flow log determinant uses QR plus a triangular solve, which
is algebraically equivalent to the prior determinant/inverse formula and
compiles under CPU XLA.

The analytical endpoint now has an opt-in `return_trace=True` diagnostic
surface.  It returns the actual pre-flow/post-flow clouds, analytical
tangents, normalized weights, observation, and post-reset state from the same
finite program.  The default two-tuple API and arithmetic are unchanged; the
trace is not fed back into the score path.

The Phase 2B support evaluator is `make_subspace_gaussian_kdm_kernel` in the
same module.  It uses a shared frozen orthonormal chart, evaluates the mixture
in coordinate dimension `r`, checks ambient and tangent support residuals, and
fails closed on a bad chart or rank-deficient coordinate covariance.  Its
chart policy is `fixed_orthonormal_support_chart_v1`; it does not yet include
the Jacobian/tangent of a parameter-dependent chart.

The existing Section 3.6 implementation and
docs/memos/section-3-6-prior-weight-sidecar-reset-2026-09-03.md are diagnostic
prior-observation cloud sidecar work from an isolated worktree; they are not
present as an integrated canonical route in this checkout. They do not
implement Younis IWSG through the actual integrated LEDH-OT-GenUT reset and
must not be silently relabelled.

Before any port, an implementation audit must verify the full call chain from
the claim-bearing endpoint to the KDM routine, including tensor rank, batch
shape, dtype, device, static-shape/XLA contract, and whether the routine sees
pre-observation or post-observation weights. A function that exists but cannot
be called by the endpoint does not satisfy the claim.

## Evidence contract

| Item | Requirement |
|---|---|
| Question | Does KDM improve the declared score target for the actual LEDH-OT-GenUT route? |
| Target labels | Every row is marked `ATOM-FINITE`, `KDM-FINITE`, or `MODEL-IS`; no cross-label equality is assumed. |
| Comparator | Canonical analytical total JVP at the same streams, route, dtype, N, horizon, and compute budget. |
| Primary criterion | Zero-bandwidth parity plus a predeclared paired uncertainty comparison of score error against an oracle; for an unchanged-value sidecar, exact value identity is mandatory. |
| Veto diagnostics | Non-finite output, support violation, omitted tangent, hidden N-by-N work, invalid proposal correction, target-label mismatch, stale tuning scope, or failed endpoint mechanics. |
| Explanatory diagnostics | ESS, overlap, mode recall, cap activity, tangent growth, memory, runtime, and bandwidth sensitivity. |
| What is not concluded | A lower variance does not prove lower bias, exact model-score correctness, posterior correctness, HMC readiness, or production readiness. |
| Preserved artifact | A versioned plan/result directory containing source/route IDs, git commit, command, environment, seeds, hardware, bandwidth, kernel rank, diagnostics, and the rendered LaTeX note. |

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|
| LEDH-OT-GenUT with Contract-E and dual caps | Repository owner policy and canonical rebuild plan | A partial or stale lane may be mistaken for the canonical call chain | Endpoint wiring/parity test | Frozen baseline |
| Gaussian KDM | Younis 2023/2024 Eq. (4)/(6) construction | Positive bandwidth shifts moments and likelihood | B=0 and value-shift test | Hypothesis |
| Innovation-subspace kernel | DSGE degeneracy requirement, not Younis source | Incorrect rank or missing chart Jacobian | Constraint residual and rank test | Extension/invention |
| Fixed proposal q at theta0 | Younis IWSG | Re-anchoring changes the estimator/force | Anchor identity and replay test | Source-faithful adaptation |
| All-pairs mixture reference | Direct evaluation of Younis mixture | O(N^2) memory/time | Complexity counter and peak-memory log | Diagnostic only |
| Positive-bandwidth score as a possible improvement | Research hypothesis | It may improve a changed target only | Kalman/oracle score MSE with uncertainty | Not promoted |

## Scholarly audit

The local PDFs above were read through their method, theory, computational
appendix, and limitation sections. The project bibliography supplies the
Corenflos and Li--Coates entries; a dedicated bibliography for this note adds
the two Younis papers and records the exact local paths.

The source-support conclusion is narrow: Younis supports the kernel-mixture
representation and fixed-proposal IWSG for its mixture objective. It does not
support an exact LEDH-OT-GenUT score claim. Live forward-citation metadata was
attempted but the configured web lookup returned an upstream 502 response, so
no citation counts, venue rankings, or completeness claim is made. A future
publication survey should close that metadata and forward-snowball gap.

## Skeptical plan audit

The plan was checked before implementation for:

- a wrong baseline: the comparator is the actual canonical endpoint, not the
  old post-observation sidecar or a generic smoother;
- a proxy promotion error: ESS, NLL, mode coverage, and runtime are
  explanatory unless the evidence contract promotes them;
- a missing stop rule: support, finiteness, derivative parity, complexity,
  and budget vetoes are explicit;
- an unfair comparison: streams, observations, route, dtype, N, horizon, and
  compute are paired;
- a hidden target change: every positive-bandwidth integrated route gets a new
  scalar and route ID;
- an environment mismatch: TensorFlow/XLA/device and memory policy belong in
  the run manifest;
- an unexamined default: bandwidth rank, proposal anchor, kernel geometry,
  and all-pairs reference status are recorded above.

The audit passes for documentation, bounded diagnostics, and Phase 4A
engineering on the reference/no-TF32 arms. The trusted-GPU calibration
records a TF32 identity/parity veto, so the plan does not authorize a
production port or establish score-error improvement. The documentation review, bounded Phase 0A registry repair,
Phase 1 algebra diagnostic, Phase 2A controlled normalizer/trace gate,
Phase 2B fixed-chart support gate, Phase 3 no-feedback auxiliary, and Phase 4A
CPU/XLA mechanics are complete. Phase 4B remains mathematically blocked. The
immediate next action is a code-level TF32 repair evaluation or an explicit
no-TF32/float64 route decision; the score-error pilot remains held until that
gate passes.

## Phase 0 execution record

The documentation phase was executed after the skeptical audit.  The LaTeX
note was compiled with the following bounded local sequence from
`docs/papers/ledh_younis_kdm_score`:

~~~
pdflatex -interaction=nonstopmode -halt-on-error ledh_younis_kdm_score.tex
bibtex ledh_younis_kdm_score
pdflatex -interaction=nonstopmode -halt-on-error ledh_younis_kdm_score.tex
pdflatex -interaction=nonstopmode -halt-on-error ledh_younis_kdm_score.tex
~~~

The final build produced an eight-page PDF with no unresolved-reference,
overfull-box, or fatal-error diagnostics.  The rendered pages were inspected
for clipping, equation overlap, and broken section transitions; none was
found.  The plan and note now contain the same target labels, proposal-ratio
equations, zero-bandwidth branch rule, and Phase~1 boundary.  This was a
documentation result; no KDM runtime or canonical mathematics was changed.

Phase 0A then changed only the stale registry name, added the executable
resolution guard, and repaired the batch keyword/forwarding drift.  The
governance command
`CUDA_VISIBLE_DEVICES=-1 python -m pytest -q
tests/highdim/test_ledh_canonical_governance.py` passed all five tests, the
analytical score suite passed `7/7`, and the combined batch/fused parity suite
passed `12/12`.  The
first attempt used a different `pytest` interpreter without TensorFlow and
failed during collection; it is classified as an environment-launch failure,
not a code or mathematical failure.

## Phase 1 execution record

The bounded Phase 1 diagnostic was implemented after the documentation and
endpoint audit.  The evaluator uses the all-pairs Gaussian-mixture identity
and the complete differential
`d log k = r^T(d x-d z) + (r^T dB r-tr(B^{-1}dB))/2`, with
`r=B^{-1}(z-x)`, plus the weight differential.  A first draft incorrectly
used `L^{-1}(z-x)` in place of `B^{-1}(z-x)`; the non-identity-covariance
finite-difference test exposed and repaired that error before any result was
interpreted.  The PF--PF mask also exposed a rank-broadcasting defect, which
was repaired, and singular-flow rejection is now explicit through
`forward_map_valid`.

The test command was:

~~~
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-mpl-cache \
python -m pytest -q tests/highdim/test_ledh_younis_kdm_tf.py
~~~

At the Phase 1 checkpoint the kernel-specific subset passed `6/6` tests; after
the Phase 2 additions the full module suite is reported below.  The suite is
an intentional CPU-only reference
diagnostic; one float32 XLA smoke is included, while the derivative checks use
non-JIT graph execution to keep their finite-difference comparison readable.
No GPU evidence, canonical score result, target comparison, or production
claim was made.  The KDM module is not imported by the canonical registry.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Keep Phase 1 kernel as diagnostic foundation | Complete tangent and proposal algebra pass bounded parity | No Phase 1 veto fired | No evidence yet about score error, DSGE support, or endpoint usefulness | Execute Phase 2 zero-bandwidth and separate-term gates | Does not show KDM improves, preserves, or computes the canonical score |

## Phase 2A execution record

The controlled linear-Gaussian convolution gate was implemented and connected
to the actual analytical endpoint through its opt-in trace.  The positive
branch evaluates the new one-step scalar
`log sum_i w_i N(y; C x_i, R+C B_i C^T)`; the atom branch uses `B=0` without
factoring or inverting the bandwidth.  Finite differences cover each of the
six dependency families separately and jointly, and the endpoint test uses
the owner Contract-E reset and dual-cap settings while keeping the canonical
value untouched.

The command was:

~~~
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-mpl-cache \
python -m pytest -q tests/highdim/test_ledh_younis_kdm_tf.py
~~~

It passed `10/10` tests.  This is CPU-only diagnostic evidence; no GPU,
score-error comparison, DSGE support claim, or route promotion was made.
The gate establishes only that the declared controlled scalar and its
derivative are implemented consistently and that the KDM sidecar can read a
real canonical trace.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Retain Phase 2A as a diagnostic gate | Atom/positive branches and complete dependency tangents pass FD; endpoint trace parity passes | No Phase 2A veto fired | Ambient SPD kernels still do not represent degenerate DSGE support | Implement and test innovation-space/rank-aware KDM before an integrated route | The controlled normalizer is not the full LEDH value or exact model score |

## Phase 2B execution record

The fixed-chart innovation-space evaluator was implemented for a shared
orthonormal support basis.  It computes the Gaussian mixture and its complete
cloud/weight/coordinate-bandwidth tangent in the support coordinates, then
checks both the ambient support residual and its tangent.  The same singular
covariance is rejected by the ambient evaluator and accepted by this
rank-aware evaluator when all points lie on the support.

The command was:

~~~
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-mpl-cache \
python -m pytest -q tests/highdim/test_ledh_younis_kdm_tf.py
~~~

It passed `13/13` tests.  This remains CPU-only diagnostic evidence.  The
fixture is an abstract rank-deficient support, not a calibrated DSGE model;
the chart is frozen and no chart Jacobian has been claimed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Retain fixed-chart support kernel as a Phase 2 diagnostic | On-support complete tangent and off-support/bad-chart vetoes pass | No Phase 2B veto fired | Parameter-dependent DSGE charts and integrated proposal wiring remain unevaluated | Wire an auxiliary evaluator from the canonical trace, with the canonical value unchanged | Does not establish DSGE support validity or a production score |

## Phase 3 execution record

The auxiliary endpoint now runs from
`canonical_value_and_analytical_score(..., return_trace=True)` under
Contract-E and the owner dual-cap settings.  The trace exposes the PF--PF
prior-observation factorization, rather than reusing posterior weights.  This
was a material correction: reusing posterior weights with the current
observation factor would double-count the observation and recreate the bug in
the September 3 sidecar memo.

At zero bandwidth, the auxiliary reconstructs the canonical atom value and
score, including the transition, proposal, forward-Jacobian, observation, and
recursive reset tangents.  The positive-bandwidth branch is returned under a
separate `KDM-FINITE` label, with
`kdm_feedback_into_canonical=false`.  During this gate, exact value
reconstruction also exposed a dtype normalization error: casting a Python
`2*pi` literal before taking its logarithm introduced a common
`2.78e-8` shift.  Replacing it with a dtype-specific `log(2*pi)` constant
restored exact component normalization.

The focused command was:

~~~
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-mpl-cache \
python -m pytest -q tests/highdim/test_ledh_younis_kdm_tf.py \
  tests/highdim/test_ledh_canonical_governance.py \
  tests/highdim/test_ledh_canonical_score_recursion.py \
  tests/highdim/test_ledh_canonical_score_full.py \
  tests/highdim/test_ledh_canonical_batch.py \
  tests/highdim/test_ledh_canonical_batch_fused.py
~~~

It passed `38/38` tests.  The KDM-specific suite is `14/14`, including a
two-step Contract-E/dual-cap auxiliary finite-difference check.  This remains
CPU-only reference evidence and does not compare score error or variance.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Retain Phase 4A as a full-feedback diagnostic | Shared-executor identity, zero-bandwidth trajectory parity, total finite differences, PSD support, CPU XLA pass, and trusted-GPU validity | TF32 identity/parity veto; Phase 4B remains math-blocked | Whether kernelized observation weighting reduces model-score MSE at equal compute is untested | Repair or explicitly exclude TF32 for this route, then run the bounded timing/power gate before freezing a serious LGSSM campaign | No evidence yet that KDM helps the score; Phase 4A is not a full Younis MDPF |

## Decision

The research direction is now Younis KDM applied to the real LEDH-OT-GenUT
dual-cap trust-region program. The current canonical total derivative remains
the baseline. A KDM sidecar may help only if it is either an explicitly
changed finite program or a demonstrably target-preserving auxiliary estimator.
Phase 1 through Phase 4A have supplied the tested algebraic, endpoint,
fixed-chart, auxiliary, and full-feedback kernelized-observation foundation.
The first trusted-GPU calibration then found a TF32 identity/parity veto, so
the next step is a code-level TF32 repair evaluation or an explicit
no-TF32/float64 route decision, followed only then by the timing and power
gate. This is not a port of Nemeth, Scibior--Wood, PaRIS, or Del Moral. The
exact all-pairs Phase 4B route must wait for its complete proposal derivation
rather than being approximated silently.
