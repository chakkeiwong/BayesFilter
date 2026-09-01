# Implementation Plan: PaRIS Forward-Only Fisher-Identity Score

Date: 2026-08-29
Status: Implementation recipe, awaiting go decision
Source: Olsson & Westerborn (2017), `.localresources/papers/olsson-westerborn-paris-1412.7550.pdf`,
Algorithm 2 (p. 12), §2.3.4 accept-reject, §3.2 variance theory
Companion: `docs/papers/paper_by_paper_derivation_findings_2026-08-29.md`

## Why this candidate

Of everything surveyed, PaRIS is the only method that satisfies all four owner constraints at once
and attacks the bias rather than only the variance:

- forward-only recursion, no backward pass over the data (DSGE-admissible);
- targets the Fisher-identity score, which contains **no derivative of any transport or reset map**,
  so the entropic-OT conditioning gap that Corenflos Prop 3.1 identifies does not enter;
- `O(N)` per step with the accept-reject backward sampler, constant memory;
- introduces **no new bias dial** — unlike fixed-lag (lag `L`) or Nemeth shrinkage (`λ`), both of
  which buy variance with a mixing-dependent or shrinkage bias.

The one genuinely new requirement is that the transition density must be *evaluable and bounded*,
not merely samplable. That holds for LGSSM and for the DSGE transitions in scope. It would not hold
for a model where only forward simulation is available.

## What the score becomes

Under the Fisher identity the score is a smoothed **additive** functional. In the paper's notation
(eq. 1.2), with `h̃` the per-step increment,

```
h_t(x_0:t) = sum_{l=0}^{t-1} h̃_l(x_l, x_{l+1})
```

and for the score we instantiate

```
h̃_t(x_t, x_{t+1}) = grad_theta log f_theta(x_{t+1} | x_t) + grad_theta log g_theta(y_{t+1} | x_{t+1})
```

so that `grad_theta log p_theta(y_{1:T}) = E[ h_T(X_{0:T}) | y_{1:T} ]`. Each `h̃` is a derivative of a
**model density only**. Nothing in it depends on Sinkhorn, Contract-E, the dual cap, the trust region,
the ridge `λ`, or the damping `δ`. That is the entire point: the per-step gain factors
`O(λ^{-1/2}) ≈ 316` and `O(δ^{-1}) = 1e5` from our variance note Props 2.5-2.6 disappear from the
score path, because there is no reset Jacobian left to propagate.

## The recursion (Algorithm 2, verbatim structure)

State carried at time `t`: the particle cloud `{(xi_t, wi_t)}_{i=1..N}` and one auxiliary statistic
per particle, `tau_i_t`, each a vector in R^P (P = parameter dimension).

```
initialise  tau_i_0 = 0                                   for i = 1..N

for t = 0, 1, ..., T-1:
    1.  propagate the cloud one step:  {(xi_{t+1}, wi_{t+1})} <- PF({(xi_t, wi_t)})
    2.  for each i = 1..N:
    3.      for j = 1..Ntilde:
    4.          draw J(i,j) ~ Pr({ w^l_t * q(x^l_t, x^i_{t+1}) }_{l=1..N})
    5.      end
    6.      tau_i_{t+1} = (1/Ntilde) * sum_j [ tau_{J(i,j)}_t + h̃_t(x_{J(i,j)}_t, x_i_{t+1}) ]
    7.  end

score estimate at any t:   sum_i w_i_t * tau_i_t / Omega_t,     Omega_t = sum_i w_i_t
```

Two structural facts worth stating plainly, because they are what make it work:

**The backward kernel is sampled, never materialised.** Step 4 draws from the discrete distribution
proportional to `w^l_t q(x^l_t, x^i_{t+1})`. Computing that normaliser explicitly for every `i` is the
`O(N^2)` forward-only FFBSm cost. The accept-reject sampler of §2.3.4 avoids it: propose
`J* ~ Pr({w^j_t})`, accept with probability `q(x_{J*}_t, x^i_{t+1}) / epsbar` where
`q(x,x') <= epsbar` for all `(x,x')`, repeat until acceptance. Under a two-sided density bound this
is `O(N)` overall (their Prop 2 citation to Douc et al.).

**`Ntilde >= 2` is a hard threshold, not a tuning preference.** The paper's §3.1 support argument:
with `Ntilde = 1` the active-support counts `#A_{s,t}` are non-decreasing in `t` and collapse to a
single ancestor, reproducing poor-man's-smoother degeneracy and `O(t^2)` variance. With
`Ntilde >= 2` the sequence is no longer monotone, so local depletion stays local, and the asymptotic
variance is `O(t)` for joint and `O(1)` for marginal smoothing. Their Figures 4-5: the variance ratio
between `Ntilde = 1` and `2` grows linearly in `t`, while `2` vs `3` is nearly flat and `10` vs `30` is
"close to indistinguishable". So `Ntilde = 2` captures essentially the whole benefit; `Ntilde = 3` is a
cheap safety margin. Do not spend compute above ~4.

Consistency holds for any fixed `Ntilde >= 1` as `N -> inf`; the threshold is about **numerical
stability in `t`**, which at `T = 50` and DSGE horizons is exactly what we care about.

## Cost and memory for our fixtures

Per time step, on top of the existing filter:

| Quantity | Cost |
|---|---|
| backward index draws | `N * Ntilde` accept-reject trials, each `O(1)` expected |
| density evaluations | `N * Ntilde * (expected trials)` calls to `q(x, x')` |
| statistic update | `N * Ntilde * P` adds |
| memory | `N * P` for `{tau_i}` plus the current cloud only |

At `d = 3`, `T = 50`, `N = 3000`, `P = 5`, `Ntilde = 2`: `tau` storage is `3000 * 5 = 15000` floats,
negligible. The added work is `~6000` density evaluations and accept-reject draws per step. Compare
against the current reset, which is a dense `3000 x 3000` Sinkhorn with 8 iterations plus a
Contract-E Cholesky — the PaRIS overhead is small relative to what we already pay.

`epsbar` for a Gaussian transition with covariance `Q` is available in closed form:
`q(x,x') <= (2*pi)^{-d/2} det(Q)^{-1/2}`, attained at `x' = mean(x)`. This is exactly the bound
Assumption 1(ii) needs and it is tight, so acceptance rates should be reasonable rather than
pathological. Their §4 notes a trials threshold is used in practice, falling back to the explicit
`O(N^2)` draw when exceeded; we should implement the same guard.

## Relationship to what we already have

PaRIS replaces the **score path**, not the filter. Concretely:

- The forward filter, including LEDH flow, Sinkhorn OT reset, Contract-E, dual cap, and trust region,
  stays exactly as is. It supplies `{(xi_t, wi_t)}` and the value `log Zhat`.
- The analytical tangent recursion (`ledh_canonical_score_tf`, and the vendored reset-JVP in
  `ledh_canonical_reset_score_tf`) is **not used** for the score. It can remain for diagnostics and
  for the existing FD-parity tests.
- The value lane and the score lane stop being two differentiations of one program and become two
  different estimators of two different objects: `log Zhat` from the filter, and the Fisher-identity
  score from the PaRIS recursion. That is a genuine change in what the score *is*, and it must be
  declared as such — it is not a repair of the current tangent, it is a different estimand with a
  different justification.

An important consequence for the reset debate: because `h̃` contains no reset derivative, the
question "does Contract-E / dual-cap / trust-region make the score better" becomes moot for the
score. Those mechanisms continue to matter for the **filter quality** feeding `w_i_t` and `x_i_t`,
and hence indirectly for the score's variance, but they no longer sit inside the differentiated path.

## HMC compatibility

This satisfies constraint 4 without a tape. The adapter contract is unchanged: `log_prob_and_grad`
returns `(value, score)`, where `value = log Zhat` from the forward filter and `score` is the PaRIS
estimate at `t = T`. It plugs into `reviewed_value_score_target_fn`
(`bayesfilter/inference/batched_value_score.py:173-224`) exactly as the current analytical score does,
because that wrapper never checks that the returned score is the derivative of the returned value.

Two cautions specific to HMC, both of which we must handle rather than assume away:

1. **Determinism.** Every random draw in the PaRIS recursion — the backward indices `J(i,j)` and the
   accept-reject uniforms — must be a frozen function of `theta` for the leapfrog involution to hold
   (variance note Remark 5.3; Maskell's CRN Jacobian-cancellation argument, §7). Seed them from the
   same frozen stream as the filter noise, keyed per chain position, not per call.
2. **The score is not the gradient of the value.** With PaRIS the returned score is a Fisher-identity
   estimate, while the value is the filter's `log Zhat`. These are consistent for the same limit but
   are not exactly gradient-of-value at finite `N`. That is precisely the surrogate-force situation,
   so the correctness argument to invoke is Corollary `cor:gsv-surrogate-force`, **not** "the gradient
   matches the value". The two plans compose: PaRIS supplies a better force, surrogate-force supplies
   the licence to use a force that is not exactly `-grad U`.

## Verification protocol

Same fixture as everything else so the numbers are comparable: `d = 3`, `T = 50`, diagonal LGSSM,
`theta = [0.72, 0.55, 0.35, 0.35, 0.45]`, observation-path seed `81100`, exact Kalman reference
value `-145.4343` and `phi_1` score `-6.2167`, 16 paired seeds, CPU float64 with GPU hidden.

Rungs, in order:

1. **`Ntilde` threshold, `N = 1008`.** `Ntilde in {1, 2, 3, 4}`. Predeclared expectation from §3.1:
   `Ntilde = 1` shows materially larger score SD than `Ntilde >= 2`, and the gap should widen with
   `T`. If `Ntilde = 1` and `2` are indistinguishable at `T = 50`, either the implementation is wrong
   or `T` is too short to expose the degeneracy — check by rerunning the pair at `T = 200`.
2. **Bias check at fixed `Ntilde = 2`,** `N in {1008, 2016, 3000}`. Primary quantity: score error mean
   and its SE, with the `z = mean/SE` screen. The question is whether the separated negative bias we
   measured (`z = -3.3` to `-7.2`, `-4%` to `-8.6%` relative) collapses toward zero.
3. **`T` ladder at `Ntilde = 2`, `N = 1008`,** `T in {3, 10, 25, 50}`, nested prefixes. Reads the
   variance growth law directly: PaRIS predicts `O(t)` variance for the joint additive functional,
   against the `O(t^2)` of a degenerate path estimator.

Roles, declared before running:

- promotion criterion: score error mean not separated from zero at 2 SE across rung 2, **and** score
  SD at least as good as the current route at matched `N`;
- promotion veto: any non-finite `tau`, accept-reject fallback firing on more than a small fraction of
  draws, or score SD worse than the current analytical route;
- continuation veto: `Ntilde = 1` and `Ntilde >= 2` behaving identically at both `T = 50` and
  `T = 200` (would indicate the backward sampling is not doing what Algorithm 2 specifies);
- explanatory only: wall time, acceptance-trial counts, ESS of the forward filter.

What will not be concluded: no HMC-readiness, no posterior-correctness, and no default change from
this diagnostic alone. A passing rung 2 licenses an HMC arm, not a new default.

## Pre-mortem

- **It could pass while misleading us** if the exact-Kalman reference and the PaRIS estimator are both
  evaluated with the same frozen noise in a way that correlates their errors. Mitigation: the Kalman
  reference is deterministic and noise-free, so this is unlikely, but keep the reference computation
  entirely separate from the particle code path.
- **It could fail for implementation rather than scientific reasons** in three specific places: the
  backward-index distribution omitting the `w^l_t` factor (giving a uniform rather than
  weight-tilted kernel), the `h̃` increment being attributed to the wrong time index (off-by-one
  between `x_t` and `x_{t+1}`), or `epsbar` set too tight so accept-reject silently never accepts.
  Cheapest discriminating check: on a `T = 2`, `N = 16` LGSSM, compare the PaRIS score against an
  explicit `O(N^2)` forward-only FFBSm computation of the same statistic. They must agree up to
  Monte Carlo error in `Ntilde`, and at `Ntilde` large they must agree closely.
- **The `d = 3` LGSSM may be too easy** to discriminate the methods. It is nonetheless the right first
  fixture because it is the only one with an exact reference. A nonlinear fixture follows only after
  the linear one passes.

## Implementation checklist

- [ ] `h̃` increment: `grad_theta log f` and `grad_theta log g` per particle pair, batched, TF float64.
      The model callbacks already expose the tangents needed
      (`bayesfilter/highdim/ledh_diagonal_lgssm_any_dim.py` supplies transition/observation
      log-density tangent functions).
- [ ] `epsbar` provider per model, closed-form for Gaussian transitions, with an assertion that
      `q <= epsbar` on the realised pairs.
- [ ] Accept-reject backward sampler with a trials cap and an explicit `O(N^2)` fallback, counting
      how often the fallback fires.
- [ ] `tau` update as a single batched op over `(N, Ntilde, P)`; no Python loop over particles.
- [ ] `tf.function` with a stable `input_signature` per repository TF policy; no `pfor`,
      no `vectorized_map`.
- [ ] Frozen-seed plumbing for backward indices and accept-reject uniforms.
- [ ] `T = 2`, `N = 16` exact-FFBSm parity test as the first gate.
- [ ] Rungs 1-3 above, artifact per run with the standard manifest fields.

## Honest status of this plan

Everything above about Algorithm 2, the `Ntilde >= 2` threshold, the `O(N)` accept-reject reduction,
and the `O(t)` / `O(1)` variance orders is read directly from the paper. What is **not** established:
that PaRIS removes *our* measured bias. Our bias arises in a filter that also contains an entropic-OT
reset, Contract-E restoration, and dual caps, none of which appear in the PaRIS analysis. The
argument for expecting improvement is structural — the Fisher-identity score contains no reset
derivative, and Corenflos Prop 3.1 locates the DPF gradient bias precisely in the
filtering-versus-smoothing conditioning gap that PaRIS closes — not empirical. Rung 2 is the test.
