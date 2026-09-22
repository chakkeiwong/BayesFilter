# Target-Preserving Score-Estimation Rewrite Proposal

> **SUPERSEDED (2026-09-07).** This proposal pursued a Fisher-identity,
> FFBSm, and PaRIS-first direction. That is not the active research question.
> It is retained as historical reasoning only. The active direction is the
> Younis kernel-density-mixture construction applied to the executed
> LEDH-OT-GenUT dual-cap trust-region program. See
> docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md
> and the accompanying LaTeX note. Nothing in this historical proposal
> authorizes a Fisher/PaRIS port or changes the canonical route.

Date: 2026-09-06  
Status: `PROPOSAL_ONLY; REWRITE BEFORE IMPLEMENTATION`  
Scope: Section 3.6 score-estimation direction; no code or canonical LEDH change is authorized by this document.

## Decision to freeze before further implementation

The current Section 3.6 note should not be extended as an estimator of the
original observed-data score until its mathematical target is rewritten.  The
replacement document must distinguish three objects:

1. the exact state-space score
   `s(theta) = grad_theta log p_theta(y_1:T)`;
2. the total derivative of the finite GenUT value
   `s_G^N(theta; xi) = grad_theta L_G^N(theta; xi)`; and
3. any alternative score estimator or deterministic proposal field.

The current total JVP is correct for object 2.  The positive-bandwidth
observation convolution is not an identity for object 1.  A revised paper may
study it only as a changed model, a proposal/lookahead device, or a candidate
estimator whose error against object 1 is measured rather than assumed away.

The recommended research direction is a target-preserving Fisher-identity
score estimator, first on a standard particle-filter reference lane and only
then, if a correction is derived, on the GenUT/OT lane.  The Gaussian kernel
may be used as an auxiliary proposal/lookahead statistic, with the original
observation factor retained in the importance correction.

## Research question

For fixed observations and parameter value, estimate

```text
s(theta) = grad_theta log p_theta(y_1:T)
```

with lower mean-squared error than the current finite-GenUT total derivative,
without silently changing the state-space model or the HMC endpoint target.

For an estimator `S_hat`, the primary diagnostic is the conditional error
decomposition

```text
MSE(S_hat | y) = || E_xi[S_hat | y] - s(theta) ||^2
                  + tr Var_xi(S_hat | y).
```

The first term is target/estimator bias and the second is Monte Carlo variance.
A variance reduction that changes the target is not a score improvement unless
the target shift is separately bounded and accepted.

## Mathematical target hierarchy

### Exact model

The state-space model is

```text
p_theta(x_0:T, y_1:T)
  = mu_theta(x_0) prod_{t=1}^T f_theta(x_t | x_{t-1}) g_theta(y_t | x_t),

ell(theta) = log p_theta(y_1:T),
s(theta) = grad_theta ell(theta).
```

This is the only quantity that may be called the observed-data score without
an estimator qualifier.

### Current finite GenUT program

For a fixed stream `xi`, the current route computes

```text
L_G^N(theta; xi)
  = sum_t log sum_i w^-_{t,i}(theta; xi)
                         g_theta(y_t | x^-_{t,i}(theta; xi)),

s_G^N(theta; xi) = grad_theta L_G^N(theta; xi).
```

The analytic recursive tangent is the total derivative of this finite scalar
on its fixed differentiable branch.  It is not thereby equal to `s(theta)`.
The OT/moment reset is a changed finite particle program, not an
unbiasedness-preserving categorical resampling step.

### Positive-bandwidth convolution

The current kernel factor is

```text
g_bar_B,theta(y | x) = integral g_theta(y | x + u) K_B(du).
```

Replacing `g` by `g_bar_B` changes the observation model.  Program A evaluates
that factor on the ordinary GenUT cloud; Program B also feeds it into the
recurrence.  Their total derivatives are valid derivatives of their own
finite programs, but neither is `s_G^N` for positive bandwidth and neither is
automatically an estimator of `s(theta)`.

## The target-preserving score construction

### Fisher identity

Under a dominated-differentiation condition,

```text
s(theta)
 = E_theta[ grad_theta log p_theta(X_0:T, y_1:T) | y_1:T ]
 = E_theta[ H_theta(X_0:T) | y_1:T ],
```

where

```text
H_theta(x_0:T)
  = grad_theta log mu_theta(x_0)
    + sum_{t=1}^T [ grad_theta log f_theta(x_t | x_{t-1})
                   + grad_theta log g_theta(y_t | x_t) ].
```

The reset, Sinkhorn map, Contract-E map, and dual-cap map do not appear in
`H_theta`; they belong to an approximation algorithm, not to the model joint
density.  This is the central mathematical route for estimating object 1.

### Exact backward recursion for the additive score

Let `eta_{t-1}` be the exact filtering law and define

```text
B_theta,t(x_t, dx_{t-1})
 = eta_{t-1}(dx_{t-1}) f_theta(x_t | x_{t-1})
   / integral eta_{t-1}(dz) f_theta(x_t | z).
```

With `T_0(x_0) = grad log mu_theta(x_0)`, define

```text
T_t(x_t)
 = integral B_theta,t(x_t, dx_{t-1})
   [ T_{t-1}(x_{t-1}) + h_theta,t(x_{t-1}, x_t) ],
```

where `h_theta,t = grad log f_theta + grad log g_theta`.  Then

```text
s(theta) = integral T_T(x_T) eta_T(dx_T).
```

This is a smoothing calculation.  It is not the derivative of the finite
GenUT value program and should not be implemented by propagating the OT/reset
Jacobian.

### Particle versions

For a standard particle filter with particles `z_{t-1,j}`, weights `beta_{t-1,j}`,
and current particles `x^-_{t,i}`, the forward-filtering backward-smoothing
plug-in is

```text
W_t(i,j)
 = beta_{t-1,j} f_theta(x^-_{t,i} | z_{t-1,j})
   / sum_k beta_{t-1,k} f_theta(x^-_{t,i} | z_{t-1,k}),

T_hat_{t,i}
 = sum_j W_t(i,j)
   [ T_hat_{t-1,j} + h_theta,t(z_{t-1,j}, x^-_{t,i}) ],

S_hat_FFBSm = sum_i alpha_{T,i} T_hat_{T,i}.
```

This is the first reference estimator to implement because it exposes the
target directly and costs `O(N^2)` transition-density evaluations per time.
PaRIS replaces the dense backward sum with a fixed number `N_tilde >= 2` of
backward draws and can be linear in `N` under a bounded transition density.
The path-space estimator is cheaper but suffers genealogical degeneracy and
is a variance reference, not the preferred route.

The standard-PF assumptions must be checked before these estimators are
attached to the GenUT/OT cloud.  The current deterministic OT/moment reset
does not supply the categorical ancestry or a proven approximation of the
original filtering law.  Using the displayed recursion on that cloud without
such a proof is an implementation of a diagnostic heuristic, not a claim of
the exact model score.

## How the Gaussian kernel can help without changing the target

The convolution can be used as an auxiliary lookahead/selection function in
an auxiliary particle filter.  Let `m_{t-1,i} > 0` be a cheap lookahead,
possibly `m_{t-1,i} = g_bar_B(y_t | x_{t-1,i})`.  Select an ancestor with

```text
r_t(i) = beta_{t-1,i} m_{t-1,i} / M_t,
M_t = sum_j beta_{t-1,j} m_{t-1,j},
```

then propagate from a proposal `q_theta(x_t | x_{t-1}^{a_t}, y_t)`.  The
corrected importance factor is

```text
omega_t
 = beta_{t-1,a_t} f_theta(x_t | x_{t-1}^{a_t}) g_theta(y_t | x_t)
   / [ r_t(a_t) q_theta(x_t | x_{t-1}^{a_t}, y_t) ]
 = M_t f_theta(x_t | x_{t-1}^{a_t}) g_theta(y_t | x_t)
   / [ m_{t-1,a_t} q_theta(x_t | x_{t-1}^{a_t}, y_t) ].
```

The original `g_theta` remains in the correction.  Therefore the forward
particle approximation still targets the original state-space model; the
kernel only reallocates computation.  The score statistic remains
`H_theta`, so derivatives of `m` or `q` are not inserted into the model score.
This is the mathematically legitimate way for the sidecar to help score
estimation while preserving the original target.

For the current deterministic GenUT reset, this correction is not available
merely by naming the OT map a proposal.  A target-preserving use would require
an evaluable proposal density and, for a deterministic transformation, its
appropriate change-of-variables/Jacobian correction.  Until that is derived,
the GenUT/OT route remains a finite surrogate or proposal heuristic.

## Method comparison and recommendation

| Method | Target | Cost | Main benefit | Main limitation | Proposed status |
|---|---|---:|---|---|---|
| GenUT total JVP | finite `L_G^N` | current route | exact derivative of executed finite scalar | not exact model score; tangent variance | retain as baseline |
| Path-space Fisher estimator | exact model score asymptotically under standard PF | `O(N)` | simplest reference | path degeneracy, poor long-horizon variance | diagnostic lower rung |
| FFBSm / marginal backward recursion | exact model score asymptotically under standard PF | `O(N^2)` | avoids path collapse; transparent target | quadratic memory/work unless chunked | first reference authority |
| PaRIS | exact model score asymptotically under standard PF | expected `O(N)` with bounded transition | forward-only, stable additive smoothing for `N_tilde >= 2` | needs transition density and standard-PF compatibility | preferred scalable candidate |
| Nemeth shrinkage/Rao--Blackwellization | unbiased estimating equation at the true parameter under stated assumptions | `O(N)` | lower variance and linear cost | shrinkage changes the finite estimating equation; not pointwise exact score | optional parameter-estimation comparator |
| Kernel as auxiliary lookahead | original model after importance correction | proposal-dependent | uses smoothing without target shift | requires proposal/correction implementation | preferred use of current kernel idea |
| Program B convolution | altered convolved model | current recurrence | well-defined alternate model | not original target | separate model only |
| Surrogate force with canonical energy | finite `L_G^N` target | current plus force | approximate force can improve proposals without changing finite-target invariance | not exact posterior; efficiency empirical | HMC mechanics option |

The primary recommendation is therefore:

1. retain `s_G^N` as the finite-GenUT baseline;
2. implement and validate FFBSm on a plain standard particle filter against the
   Kalman score in the linear-Gaussian case;
3. use that reference to validate a PaRIS implementation;
4. only then study a target-preserving auxiliary proposal using the Gaussian
   lookahead; and
5. treat any use of a Fisher/PaRIS estimate as a surrogate force for the finite
   GenUT energy unless an exact target energy is separately available.

## Replacement LaTeX document structure

The existing sidecar note should be rewritten, not incrementally patched, with
this narrative:

1. **Question and target.** Define exact likelihood, exact score, finite GenUT
   value, finite GenUT gradient, and the conditional MSE objective.
2. **What the current derivative proves.** Give the fixed-stream finite-program
   theorem and explicitly stop short of an exact-score claim.
3. **Why observation convolution is not a score correction.** Prove the target
   shift, including the Program-A/Program-B distinction and the `B=0` limit.
4. **Fisher identity.** Derive the exact score as a smoothed additive functional
   and explain why reset derivatives do not occur.
5. **Smoothing estimators.** Derive path-space, FFBSm, and PaRIS boundaries;
   distinguish consistency, finite-sample bias, variance, and computational cost.
6. **Target-preserving auxiliary proposals.** Derive the lookahead selection
   and original-likelihood correction above; state the density/Jacobian gap for
   the current deterministic OT reset.
7. **HMC boundary.** Separate exact same-scalar gradient, surrogate force with
   canonical finite energy, and exact-model score. Do not call a proposal field
   a score.
8. **Evidence protocol.** State exact oracle pairings, MSE decomposition,
   refinement ladders, and the conditions required before code integration.

The rewritten conclusion must say: the present implementation computes a
finite-program gradient; Fisher/PaRIS is the candidate route for the exact
model score; the Gaussian sidecar is target-preserving only when used as a
proposal/lookahead with the original likelihood correction; and no estimator is
promoted until the standard-PF oracle and GenUT compatibility gaps are closed.

## Literature audit

### Source-support ledger

| Source | Class | Local source | Inspected technical anchors | Allowed claim | Boundary |
|---|---|---|---|---|---|
| Nemeth, Fearnhead, Mihaylova, *Particle Approximations of the Score and Observed Information Matrix for Parameter Estimation in State Space Models With Linear Computational Cost* | `DIRECT_METHOD` | `.localresources/papers/nemeth2013-linear-cost-score.txt` and PDF | Sec. 3.2 Eqs. (8)--(12); Sec. 4.1 Eqs. (13)--(16); Sec. 5.1 Eqs. (17)--(18); Theorem 5.2 | Fisher-identity score decomposition; kernel/Rao--Blackwell recursion; variance/bias tradeoff; unbiased estimating equation at the true parameter under assumptions | Does not prove pointwise finite-sample equality to the exact score; does not analyze GenUT/OT reset |
| Del Moral, Doucet, Singh, *Uniform Stability of a Particle Approximation of the Optimal Filter Derivative* | `DIRECT_METHOD` | `.localresources/papers/delmoral-doucet-singh-filter-derivative.txt` and PDF | Sec. 2 Eqs. (2.1)--(2.6), backward kernel (1.8)--(1.11), Algorithm 1 | Backward conditional representation of filter derivatives and pairwise recursion; stability distinction from path-space method | Does not validate the current OT/moment reset |
| Olsson and Westerborn, *The PaRIS algorithm* | `DIRECT_METHOD` | `.localresources/papers/olsson-westerborn-paris-1412.7550.txt` and PDF | Sec. 1.1--1.2; Sec. 2.3; Algorithm 2; Theorems 1, 3, 8, 9; Sec. 2.3.4 accept-reject | Forward-only additive smoothing, `N_tilde >= 2` stability result, linear-cost accept-reject under bounded transition | Does not establish compatibility with deterministic GenUT/OT reset |
| Corenflos et al., *Differentiable Particle Filtering via Entropy-Regularized Optimal Transport* | `DIRECT_METHOD; COMPETITOR` | `.localresources/papers/corenflos21a-differentiable-particle-filtering.txt` and PDF | Sec. 1.2 Algorithm 1; Sec. 2.2; Sec. 4.1; Prop. 4.3 | Standard PF likelihood unbiasedness versus differentiable OT-filter objective; warning that differentiating an altered/partial resampling program can have non-vanishing bias | Their consistency assumptions and objective are not a proof for this GenUT route |
| Ścibior and Wood, *Differentiable Particle Filtering without Modifying the Forward Pass* | `IMPLEMENTATION_OR_SOFTWARE; DIRECT_METHOD` | `.localresources/papers/scibior-wood-dpf-forward-pass.txt` | Sec. 12.2 Eqs. (137)--(139), stop-gradient calculus, marginal PF discussion | AD construction of Fisher/Poyiadjis estimators for standard PF without changing its forward values | Does not authorize an AD shortcut for the current deterministic OT reset |
| Current GenUT score-variance note | `PROJECT_DERIVATION` | `docs/bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex` | Fisher identity, backward recursion, surrogate-force proposition | Project derivation of the target boundary and candidate equations | Not an empirical certificate for exact-score accuracy |

### Backward snowball

The inspected seed papers point to the standard lineage: Fisher/Louis
identities, Poyiadjis score estimators, Del Moral backward Feynman--Kac
representations, FFBSm/FFBSi, fixed-lag smoothing, PaRIS, and auxiliary
particle filters.  These are classified as foundational or direct methods;
fixed-lag and Nemeth shrinkage remain comparators because they trade bias for
variance.  The new document should not cite them as interchangeable algorithms.

### Forward snowball and metadata

The local technical copies were inspected.  Live citation/venue metadata and a
complete forward-citation search were not completed because the configured web
lookup returned upstream `502` errors on the attempted primary-source queries.
No citation count or venue ranking is used as evidence.  A publication-grade
survey must close this metadata/forward-snowball gap before claiming
literature completeness.

### Omission risks

| Omission risk | Why it matters | Action |
|---|---|---|
| Poyiadjis et al. full technical text is unavailable locally; cached files are HTML error pages | It is the primary source for path/marginal score variance claims | Use Del Moral/Nemeth/PaRIS anchors now; obtain and inspect the Biometrika paper before any categorical rate claim |
| Auxiliary particle filter source (Pitt--Shephard) is only present through citations in local copies | The target-preserving lookahead derivation relies on its proposal correction | Add a checked primary copy before implementation |
| Standard PF theory under deterministic OT reset | Current cloud is not automatically a standard filter approximation | Treat GenUT/OT Fisher route as blocked until a consistency/correction proof is supplied |
| Recent differentiable-PF follow-ups | They may contain alternative target-preserving gradient constructions | Forward-snowball after network access is restored; do not infer novelty now |

## Pre-implementation gates

No Section 3.6 implementation should be changed until the rewritten document
has:

1. one exact score symbol and one finite-GenUT gradient symbol, used
   consistently throughout;
2. a proved Fisher identity and backward recursion in the project's notation;
3. a proved target-preserving auxiliary proposal correction, or an explicit
   decision not to use the Gaussian kernel for the original target;
4. a compatibility statement for standard PF versus deterministic GenUT/OT;
5. matched oracle pairings for original and convolved models;
6. a method table that labels variance reduction, bias, consistency, and HMC
   target status separately; and
7. a revised evidence contract whose primary metric is score MSE against an
   exact oracle, with bias and variance reported separately.

## What this proposal does not conclude

It does not prove that PaRIS, FFBSm, Nemeth shrinkage, or the Gaussian
lookahead improves the current GenUT score.  It does not prove that a
deterministic OT/moment reset can be inserted into a Fisher-identity particle
estimator without correction.  It does not authorize a code port, HMC use,
canonical LEDH promotion, or a positive-bandwidth scientific claim.
