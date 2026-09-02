# Younis MDPF/MDPS and the GenUT Dual-Cap Score

Date: 2026-08-31
Status: **CONDITIONALLY PROMISING AS A SIDECAR; NOT A CANONICAL SCORE REPAIR**

This note is an addendum to `memo_to_codex_review_response_2026-08-30.md`.
It answers whether the regularized/mixture-density particle-filter idea in
the two Younis papers can be combined with the current GenUT dual-cap
trust-region route. I read the method, theory, computational appendix, and
limitations sections of both local PDFs. No new experiment was run.

## Executive decision

Yes, a **small, smooth Gaussian-kernel sidecar** is worth testing. It can
reduce Monte Carlo variance, preserve modes, or provide a lower-variance
force. It does **not** by itself remove the finite-particle bias in the
GenUT/OT program, and it does not turn the current score into the exact model
score.

The recommended combination is:

1. Leave the accepted GenUT dual-cap/Contract-E value and reset unchanged.
2. Use a diagonal or block-diagonal Gaussian kernel in standardized state
   coordinates to form an analytic likelihood-convolution or score sidecar.
3. Keep that sidecar separate from the claim-bearing finite-value derivative;
   use it first as a diagnostic/control variate and, only after a mechanics
   check, as a deterministic surrogate force.
4. Use fixed, named random streams only when sampling is unavoidable. Average
   a calibrated small number of independent fixed streams if one realization
   is too noisy.

Do not import the full Younis mixture-resampling/IWSG or two-filter MDPS path
into the runtime. Their gradient/training paths require all mixture-component
interactions and are quadratic in the particle count. That remains outside
the stated budget, just as PaRIS is.

## What the papers actually establish

The relevant technical anchors are:

| Paper | Checked result | Boundary for this project |
|---|---|---|
| Younis and Sudderth (2023), Sec. 2.2 Eq. (4), Sec. 4 Eqs. (14)-(15), App. B.1 | Replaces a weighted Dirac cloud by a continuous kernel mixture; samples from that mixture; uses an importance-weighted sample-gradient (IWSG) with a proposal fixed at the evaluation parameters. | The unbiasedness statement is for the appropriate mixture expectation gradient under the paper's setup, not for the derivative of our finite OT/GenUT value. Their experiments are learned, discriminative 3-D tracking filters. |
| Younis and Sudderth (2024), Sec. 2.1, Sec. 4 Eqs. (17)-(23), App. D.2.3 | Uses Gaussian mixture resampling, stratification, and an importance-weighted two-filter smoother. Direct forward/backward mixture multiplication has `N^2` components; the proposed smoother avoids that at inference by sampling a smaller proposal. | It is an offline smoother using future observations, not an online filtering score. Its training/gradient computation is `O(TN^2)`, and its state dimension is still small. |

The first paper explicitly reports `O(N^2)` training complexity for MDPF
and A-MDPF because the mixture gradient evaluates the mixture at every
sample (App. B.1). The second reports `O(TN^2)` at training time for MDPS,
with `O(TN)` inference only because gradients and importance weights are
not needed at inference. A parameter-score evaluation is a gradient-bearing
operation, so the linear inference plot cannot be transferred to our score
path.

Neither paper proves lower bias for a model likelihood score. Their primary
criteria are learned state-density NLL, tracking accuracy, and mode recall.
Applying their construction to GenUT is therefore an `extension_or_invention`,
not a source-faithful implementation of either paper.

Strictly, the papers smooth the empirical posterior representation and its
resampling operation. They do not replace the underlying state transition or
observation model by a kernel, and they do not claim that every Dirac-based
particle score becomes unbiased after convolution.

## Where it meets the current call chain

The current same-scalar route starts at
[`finite_value_score`](../../bayesfilter/highdim/cubature_genut_filter.py:537).
It propagates fixed innovations, forms likelihood weights, applies the
Sinkhorn/Contract-E reset and higher-moment dual-cap/trust-region map, and
returns a manual total JVP. The mathematical note already distinguishes

```text
Object A: d L_hat_N(theta; fixed streams) / d theta
Object B: grad log p_theta(y_1:T)
```

The two are not equal at finite `N`. The current proposal score endpoint
[`finite_value_standard_score_guided_proposal`](../../bayesfilter/highdim/genut_guided_proposal_tf.py:1136)
also states that its score is the target-model backward-kernel filtering
score and deliberately does not differentiate proposal, transport, or reset
operations. The initial-RQMC route recomputes score marks after the primal
reset (see [`standard_pairwise_backward_marks`](../../bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py:163)).

Consequently, "kernelize the particles" has three different meanings. They
must not be mixed:

1. Change the finite state measure used by the reset and claim its JVP.
2. Change the likelihood/score functional evaluated on the existing cloud.
3. Change only the proposal or force used to explore the finite energy.

Only (2) as a separately named finite-program arm, and (3) as a sidecar, are
reasonable first tests. (1) changes the moment problem at the heart of the
dual-cap route and is the riskiest option.

## Why a direct KDE replacement is not a bias cure

For a weighted empirical measure

```text
mu_N(dx) = sum_i w_i delta_{x_i}(dx),
```

the Younis representation is

```text
mu_N^h(dx) = sum_i w_i K_h(x - x_i) dx.
```

This removes point masses, but it also changes the finite measure. For a
centered Gaussian kernel with covariance `B`, writing `Z = X + E` with
`E ~ N(0, B)` gives

```text
E[Z]       = E[X]
Cov(Z)     = Cov(X) + B
E[Z_j^4]   = E[X_j^4] + 6 Cov(X)_jj B_jj + 3 B_jj^2.
```

Thus applying the kernel before the GenUT correction makes the dual-cap
target covariance and fourth moments include an artificial kernel component.
The correction can then spend its limited trust-region steps undoing the
kernel's own shape, or it can preserve the wrong covariance if `B` is not
accounted for. Subtracting (B) is not generally safe near a singular or
indefinite empirical covariance. A large bandwidth can also merge distinct
posterior modes.

More generally, KDE has the usual bias-variance tradeoff: a smooth test
functional often gets an `O(h^2)` smoothing bias while sampling variance can
fall with `h`, but the logarithm in a likelihood increment adds another
finite-sample bias term. No result in either Younis paper identifies the
optimal `h` for a parameter score in a nonlinear state-space model.

## The viable combinations

### A. Analytic likelihood-convolution sidecar (recommended)

Keep the GenUT cloud (x_i) and weights (w_i) exactly as they are, and
evaluate a separate smoothed observation factor

```text
g_bar(theta, y | x_i, B_i)
  = integral g_theta(y | x_i + u) K_{B_i}(u) du.
```

The corresponding finite increment is

```text
ell_bar_t = log sum_i w_i g_bar(theta, y_t | x_i, B_i).
```

For a linear-Gaussian observation (y=Cx+v), a Gaussian state kernel gives
the exact identity

```text
g_bar(y | x_i) = Normal(y; C x_i, R + C B_i C^T),
```

so a diagonal/block `B_i` costs `O(Nd)` or `O(Nd^2)`, with no all-pairs
mixture evaluation. This is an especially clean LGSSM diagnostic. For a
nonlinear observation, a fixed local quadrature or linearization is an
additional approximation and must be named as such.

There are two legitimate uses:

* **Same-scalar arm:** define `ell_bar_t` as a new finite target and include
  derivatives through (x_i), (w_i), (B_i), and every model parameter.
  Stopping gradients through any of these terms computes a partial derivative,
  not the total derivative of the declared scalar.
* **Sidecar/force:** use the smoothed model score as a diagnostic, control
  variate, or deterministic force while the original finite value remains the
  HMC energy. The force-field MH result in the existing mathematical note
  makes this formally valid for the finite energy, but mixing and force
  quality still require measurement.

This option tests whether smoothing the *likelihood evaluation* reduces score
noise without contaminating the dual-cap moment targets.

This does not replace the earlier O(NT) Nemeth/Liu-West score-sidecar
recommendation. If the scientific target is the model Fisher score rather than
the finite GenUT derivative, the additive-score recursion remains the more
direct candidate. A Younis-style convolution can be used there as a separate
control variate or as an explicitly smoothed mark, but the combined object is
new and must be validated against the small all-pairs positive control.

### B. Local fixed-jitter proposal (conditional)

After the existing reset, draw a fixed, independent perturbation for each
particle, `x_i^jit = x_i + L_i eta_i`, and optionally apply the existing cap
checks. This is `O(Nd)` for diagonal `L_i` and can preserve
mode support when the cloud is sparse. It is a proposal regularization, not a
new posterior representation, unless an exact importance correction is
derived.

Use this only with a declared proposal density and correction. A component-
conditioned density (K_{B_i}) is not the full mixture density
(\sum_j w_jK_{B_j}); replacing one by the other is a changed estimand.
This route is therefore suitable first as an exploratory proposal/force arm,
not as a claim-bearing likelihood.

### C. Kernelized state measure inside GenUT (defer)

This would require redefining the source moments, the Contract-E target, the
dual-cap residuals, the trust-region Jacobian, and the score. It must carry the
kernel covariance and higher-moment contributions explicitly. It is a new
finite program, not a plug-in MDPF. Do not make it the default or combine it
with the current route under the old route identity.

### D. Full Younis IWSG/MDPF or MDPS (reject for runtime)

At a sample `z_i`, IWSG needs `m(z_i | phi)` and
`grad_phi m(z_i | phi)`, each a sum over mixture components. Doing
this for `N` samples is `O(N^2)`. The MDPS two-filter product has the same
all-pairs issue, and its training complexity is explicitly `O(TN^2)`.
Sparse neighbors or random-feature approximations would be new methods with
new derivative and bias analyses; they are not a free interpretation of the
papers.

## Fixed random numbers

Fixed randomness is useful, but for a narrower reason than "it removes bias."

* Reusing the same innovations and uniforms makes a parameter-to-output map
  repeatable and makes paired comparisons much less noisy.
* Stratified uniforms can reduce resampling variance, as reported in the 2024
  paper. They do not make a fixed realization unbiased.
* A categorical component selected by an inverse CDF is piecewise constant in
  the weights. A fixed uniform does not make its derivative smooth; it only
  fixes which side of a discontinuity is used.
* Younis IWSG avoids moving sampled locations by differentiating mixture
  weights against a proposal frozen at the current evaluation point. That is
  a different estimator from the total JVP of our fixed finite program.
* In a long HMC trajectory, re-anchoring the proposal (q=m_{\phi_0}) at
  every new parameter value would change the force construction. A single
  frozen anchor or a fixed stream ensemble is needed for a deterministic
  force, and the resulting target/force identity must be documented.

Use independent named streams for calibration, validation, and the claim run.
If one stream is too noisy, average `K` independent fixed-stream sidecar
forces at `O(KNTd)` cost. This reduces realization variance; it does not
establish finite-`N` unbiasedness. One seed, even if fixed forever, is not an
uncertainty analysis.

## Recommended bounded validation

**Question.** Does a Gaussian-kernel sidecar lower score RMSE or MCSE without
changing the canonical GenUT value, adding an all-pairs interaction, or
violating reset validity?

**Baseline ladder.** Compare the current dual-cap/trust-region analytical JVP,
an analytic Gaussian-likelihood sidecar, and (separately) a local fixed-jitter
proposal. Keep a direct kernelized-reset arm out of promotion until the first
two arms are understood.

**Controls.** Use the same observations, route identity, dtype, fixed design,
and named streams. Tune the dimensionless bandwidth in Contract-E-standardized
coordinates on calibration data only; freeze it for validation/claim data.
Use diagonal or predeclared blocks. Do not tune a bandwidth on the claim
observations.

**Hard vetoes.** Reject an arm for non-finite values, invalid Cholesky or
dual-cap factors, changed value when it is advertised as a sidecar, hidden
(N\times N) kernel construction, stale tuning scope, or failed deterministic
reverse-map/endpoint checks for a force arm.

**Primary evidence.** On the LGSSM, compare score error to the exact Kalman
score over multiple observation paths and independent fixed streams, with a
predeclared paired uncertainty interval. For a sidecar, also require value
identity to the canonical route. For a new kernelized value, require an
explicit new target ID and an `h -> 0`, `N`-ladder study before making any
bias statement.

**Explanatory diagnostics.** Record ESS, maximum weight, kernel overlap,
bandwidth, mode coverage, cap activity, covariance eigenvalues, directional
growth (D1), memory, and wall time. These explain a result; they do not by
themselves promote it. On nonlinear models without an exact score oracle,
report the comparison as descriptive.

## Decision table

| Candidate | Estimand | Incremental cost | Decision |
|---|---|---:|---|
| Gaussian likelihood-convolution sidecar | Separate smoothed score/force | `O(Nd)` for diagonal linear-Gaussian case | **Test first** |
| Local fixed Gaussian jitter | Proposal-regularized finite route | `O(Nd)` per jitter stream | **Test only with explicit correction/sidecar label** |
| Kernelized GenUT reset | New finite value and total JVP | At least `O(Nd)`, with altered moments | **Defer; new route ID** |
| Younis IWSG mixture gradient | Mixture expectation gradient | `O(N^2d)` | **Reject under budget** |
| Younis MDPS/two-filter smoother | Smoothed posterior using future data | `O(TN^2)` training/gradient path | **Reject for online score** |
| Fixed random stream alone | One deterministic realization | No extra asymptotic cost | **Adopt for reproducibility, not bias correction** |

## Scholarly audit record

**decision:** Conditionally pursue an analytic Gaussian-kernel sidecar or
fixed-jitter proposal; keep the canonical GenUT value/reset unchanged. Do not
use full MDPF/MDPS/IWSG in the no-`N^2` score path.

**metadata_date:** 2026-08-31. Venue/year and equation references were read
from the local PDFs. Citation counts and venue rankings were not used.

**seed_papers:**

- `Differentiable and stable long-range tracking of multiple posterior modes`
  (Younis and Sudderth, NeurIPS 2023), local full PDF.
- `Learning to be smooth: An end-to-end differentiable particle smoother`
  (Younis and Sudderth, NeurIPS 2024), local full PDF.

**source_support_summary:** Full-text technical sections, equations, appendices,
computational requirements, experiments, and limitations were inspected. The
papers support kernel-mixture representation, IWSG's fixed-proposal mixture
gradient construction, stratified mixture resampling, and the stated
complexity claims. They do not support an exact or lower-bias GenUT parameter
score claim.

**citation_venue_summary:** NeurIPS 2023 and NeurIPS 2024 metadata is present
in the PDFs. No live citation count or ranking was available/needed for this
method decision.

**backward_snowball_summary:** The papers cite regularized PF/KDE bandwidth
selection, differentiable resampling, entropy-regularized OT, score-gradient
estimators, and fixed-lag/two-filter smoothing. Those lines are represented in
the existing memo response and GenUT score-estimator note. They are relevant
background or competitor methods, not evidence that a kernel fixes the
current finite OT/GenUT score.

**forward_snowball_summary:** A live metadata/forward-citation lookup was
attempted but the service returned an upstream error. Forward coverage is
therefore not claimed; no citation count or follow-up result is fabricated.

**quarantined_sources:** None of the two local PDFs showed a retraction or
withdrawal notice. Author code was not available in the PDFs; the 2024
checklist says code would be released after acceptance, so implementation
details beyond the paper are source-blocked.

**top_omission_risks:** High-dimensional KDE bandwidth theory; exact score
estimators compatible with deterministic OT/moment resets; sparse or
random-feature mixture approximations; and an author-code audit. These are
open literature/implementation tasks, not grounds to promote the hybrid.

**claim_support_gaps:** No source establishes finite-`N` score unbiasedness,
lower bias, or HMC posterior correctness for a kernelized GenUT route. Any
such statement requires a project derivation and matched validation.

**next_required_actions:** (1) implement/read-only benchmark for analytic
LGSSM convolution; (2) run a predeclared multi-path, multi-stream variance
comparison; (3) only if useful, test local jitter as a proposal sidecar; (4)
keep full IWSG/MDPS and PaRIS out of runtime; (5) update the route ledger if a
new finite target is introduced.

**what_is_not_concluded:** This note does not establish that kernels reduce
the current score bias, that the Younis method is source-faithful for GenUT,
that a fixed seed is unbiased, that a sidecar improves HMC mixing, or that any
candidate is ready for canonical, default, leaderboard, or scientific use.
