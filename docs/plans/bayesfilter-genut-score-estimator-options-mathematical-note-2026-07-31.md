# Score Estimator Options For The GenUT Route: Mathematical Note

Date: 2026-07-31
Status: `METHODS_NOTE_AUDITED_AND_FOCUSED_DIAGNOSTICS_EXECUTED`
Companions:
- audit: `docs/plans/bayesfilter-genut-score-computation-audit-result-2026-07-30.md`
- handoff: `docs/plans/bayesfilter-genut-score-audit-followup-handoff-2026-07-30.md`
- empirical state: `bayesfilter-austria-sir-pairwise-moment-genut-score-trial-result-2026-07-30.md`,
  `bayesfilter-pairwise-moment-genut-lgssm-ksc-predator-prey-trial-result-2026-07-30.md`,
  `bayesfilter-zhao-cui-moment-teacher-result-2026-07-30.md`

Purpose: given that the recursive forward-sensitivity score is mechanically
correct (audited 2026-07-30) but empirically high-variance on Austria SIR
(diagonal-only SDs up to `3435.6`; pairwise repair cuts SD `14.6–94.1x` but
shifts the value by `7.9` baseline SEs), derive (i) why the variance arises,
(ii) what can repair it while keeping the current estimator, (iii) what could
replace the estimator, and (iv) whether the Zhao–Cui squared-TT moment-teacher
idea is sound and useful for this problem. Every proposed run below still
requires its own experiment plan; this note authorizes nothing.

## 0. Notation and the executed finite program

Carried equal-weight cloud `X_t^+ ∈ R^{N×d}`; fixed innovations `e_{t,i}`;
observations `y_t`. One step of the executed route:

```text
x_{t,i}^- = f_theta(x_{t-1,i}^+, e_{t,i})                       (transition)
l_{t,i}   = log g_theta(y_t | x_{t,i}^-)                        (log-likelihoods)
lhat_t    = log( N^{-1} sum_i exp(l_{t,i}) )                    (increment)
alpha_t,i = softmax_i(l_{t,i})                                  (weights)
X_t^+     = R_theta(X_t^-, alpha_t)                             (OT + Contract E
                                                                 + higher-moment reset)
```

Finite scalar and its score (current route):

```text
Lhat^N(theta) = sum_t lhat_t(theta),
score = d Lhat^N / d theta        (total derivative, manual JVP recursion)
```

All designs, controls, innovations, and branches are fixed; `Lhat^N` is a
smooth deterministic function of `theta` on the fixed branch (audit, Prop.
`bf-eot-hm-score`). Two distinct mathematical objects must never be conflated:

- **Object A (same-scalar score):** `d Lhat^N / d theta` — exactly what the
  code computes. Correct for `Lhat^N`; says nothing about the exact model.
- **Object B (exact score):** `∇_theta log p_theta(y_{1:T})` — the scientific
  target. `Lhat^N` is a biased approximation of `log p_theta(y_{1:T})` at
  finite `N` (the OT/moment reset is not unbiased; chapter states this), so
  Object A is a biased estimator of Object B even with zero Monte Carlo noise.

The observed problem is the **particle-seed variance of Object A** (and,
separately, its unknown bias against Object B).

## 1. The HMC contract, and a validity theorem that decouples force from energy

The HMC wrapper currently passes one custom-gradient target: value and score
from the same adapter call, so MH corrects leapfrog error **relative to the
finite surrogate target** `U(theta) = -Lhat^N(theta)`. That is exact MCMC on
`exp(-U)` because value and gradient describe the same scalar.

**Proposition 1 (surrogate-force HMC validity).** Let `F: R^P -> R^P` be any
deterministic, momentum-independent field (not necessarily a gradient). Define
one leapfrog step `Psi`:

```text
p_1/2 = p_0 + (eps/2) F(theta_0)
theta_1 = theta_0 + eps M^{-1} p_1/2
p_1 = p_1/2 + (eps/2) F(theta_1)
```

and the proposal `S = flip ∘ Psi^L` with `flip(theta,p) = (theta,-p)`.
Then the MH kernel that accepts with probability
`min{1, exp(H(z) - H(S z))}`, `H(theta,p) = U(theta) + (1/2) p^T M^{-1} p`,
leaves `exp(-H)` invariant — **for any `F`**.

*Proof.* (i) Volume: each substep adds to one coordinate block a function of
the other block (a shear), so each substep Jacobian has unit determinant;
hence `|det D Psi| = 1` and `|det D S| = 1`. (ii) Involution: apply `Psi` to
`(theta_1, -p_1)`. The first half-kick gives
`-p_1 + (eps/2)F(theta_1) = -p_1/2`; the drift gives
`theta_1 - eps M^{-1} p_1/2 = theta_0`; the second half-kick gives
`-p_1/2 + (eps/2)F(theta_0) = -p_0`. So `Psi ∘ flip ∘ Psi ∘ flip = id`, and
`S` composed over `L` identical steps is an involution (`F` depends only on
`theta`, so the step sequence is palindromic automatically). (iii) A
deterministic involutive volume-preserving proposal with acceptance
`min{1, e^{H - H∘S}}` satisfies detailed balance for `exp(-H)` by the standard
change-of-variables argument. ∎

**Consequence.** Correctness needs only the **energy** `U` to be the exact
executed finite scalar. The **force** `F` may be any deterministic
`theta`-function: a lower-variance replacement score (Sect. 4), a damped or
clipped same-scalar score, a teacher score, or a Gaussian-approximation
(sigma-point) score — gradient error degrades acceptance, never invariance.
Requirements: freeze all seeds inside `F`; `F` must not depend on momentum or
trajectory randomness. This is implementable but **not current behavior**
(single custom-gradient target today; the Zhao–Cui result note records the
same fact). It converts "the score is too noisy" from a correctness crisis
into an efficiency knob.

## 2. Why the current score blows up: tangent-recursion anatomy

Vectorize the carried tangent `v_t = vec(d X_t^+ / d theta_p) ∈ R^{Nd}`. The
audit verified that the executed JVP implements exactly

```text
v_t = J_t v_{t-1} + b_t,                                            (2.1)
J_t = D_X [ R_theta( F_theta(X, e_t), alpha(F_theta(X, e_t)) ) ]    (stage Jacobian)
b_t = explicit theta-terms (transition, likelihood, initial scale)
```

so `v_T = sum_{s<=T} ( prod_{u=s+1}^{T} J_u ) b_s`, and the score is

```text
d lhat_t / d theta = sum_i alpha_{t,i} [ d_theta l_{t,i} + grad_x l_{t,i} · xdot_{t,i}^- ],
score = sum_t d lhat_t / d theta.                                    (2.2)
```

Across particle seeds, `J_u` and `b_s` are random (functions of `e`, initial
noise). Two regimes:

- **Contractive/neutral:** a strictly negative Lyapunov exponent together
  with bounded forcing controls tangent accumulation. Linear score-variance
  growth is only a conditional hypothesis: it additionally needs suitable
  stationarity, moment, and mixing assumptions on the random stages and
  additive terms. At `gamma = 0` the geometric bound is generally `O(T)`, not
  summable. The LGSSM `T=50` pattern is descriptive evidence, not a consequence
  of `gamma <= 0` alone.
- **Expansive (`gamma > 0`):** early tangent contributions are amplified
  `~ e^{gamma (T-s)}`; variance explodes with `T` and with conditioning. The
  Austria diagonal-only route (SD `3.4e3` vs pairwise `36.5` on identical
  data/seeds) is the signature of an unstable tangent mode. This is a
  **mechanism hypothesis**, directly measurable (below), not yet a proven
  classification.

**Stage-gain inventory** (each factor verified in the audited code; bounds are
worst-case operator norms, not typical values):

1. Weight normalization: `alphadot_i = alpha_i ( vdot_i - sum_j alpha_j vdot_j )`
   with `vdot_i = d_theta l_i + grad_x l_i · xdot_i`. Gain from particle
   tangents into weight tangents `~ max_i alpha_i ||grad_x l_i||` — large for
   concentrated likelihoods (Austria: 9 observed coordinates per step).
2. Sinkhorn kernel: `d exp(-C/(eps*cbar)) ~ kernel * (dC/(eps*cbar) - ...)` —
   gain `~ 1/(eps*cbar)`; row quotient divides by `m_i >= tau = 1e-7`.
3. Contract E affine: `dA = (dL_w - A dL_E) L_E^{-1}` with
   `||L_E^{-1}||_2 = lambda_min(P_tilde + lambda I)^{-1/2} <= lambda^{-1/2}
   ≈ 316` at `lambda = 1e-5`; Cholesky JVPs additionally scale with
   `kappa(L)`.
4. Damped Gauss–Newton (diagonal correction):
   `cdot = rho (J_a^T J_a + delta I)^{-1} ( rdot - Ndot N^{-1} r )`,
   `||(J_a^T J_a + delta I)^{-1}|| <= 1/delta = 1e5` at `delta = 1e-5` when
   `J_a` degenerates (clouds near the moment-feasibility boundary
   `k ≈ 1 + s^2` make `J_a` ill-conditioned).
5. Pairwise stage: normalization by `rms = sqrt(mean(proj^2) + delta_p)`
   contributes `1/rms <= delta_p^{-1/2}` factors in the tangent.

A product of `T` stages whose gains are bounded only by ridge/floor constants
can amplify multiplicatively. Conversely, the pairwise map plausibly *reduced*
`gamma`: linearizing one pairwise step in the shape-error directions gives
`u -> u + (rho_p / rms) * (bounded projected gradient step)`, i.e. local
Jacobian `≈ I - (rho_p/rms) H_pair` on those modes (`H_pair` the Gauss–Newton
Hessian of the pair objective), a contraction when
`0 < rho_p * lambda_H / rms < 2`. Heuristic linearization — testable, not
proved.

**Diagnostic D1 (same-run finite-time directional-growth diagnostic).** The tangent layout is already
`[N, d, P]`. Append `K` probe columns `w ~ N(0, I_{Nd})`, fixed seed, zero
`b_t` (i.e. propagate them through `J_t` only — pass zero explicit
theta-terms for the probe coordinates), and record per-step growth
`g_t = ||J_t w_t|| / ||w_t||`, renormalizing `w` each step. Then
`gammahat = T^{-1} sum_t log g_t` is a finite-horizon directional-growth
summary, not automatically the asymptotic top Lyapunov exponent. One probe can
miss the dominant direction and nonnormal transients can dominate. Use
multiple probes (or QR/subspace iteration) and report this diagnostic jointly
with score SD. It is explanatory evidence, not a promotion criterion. The
mechanism hypothesis predicts larger growth for Austria diagonal-only than
pairwise, with LGSSM as a control.

## 3. Same-scalar repairs (keep Object A, keep the HMC contract)

### 3.1 Root-cause repair: replace the cubature residual design by a fixed whitened Gaussian design

**The defect (derived).** The replicated cubature design has rows
`sqrt(d)(±e_a)`, each replicated `M = N/(2d)` times. Per axis `a`:

```text
N^{-1} sum_r Xi_{r,a}^2 = (2M · d) / (2dM) = 1                    (variance 1, by design)
N^{-1} sum_r Xi_{r,a}^4 = (2M · d^2) / (2dM) = d                  (kurtosis = d)
N^{-1} sum_r Xi_{r,a}^2 Xi_{r,b}^2 = 0   (a != b)                 (co-kurtosis = 0)
```

because every row has exactly one nonzero coordinate. Gaussian references are
`3` and `1`. **At `d = 18` the reset injects residuals with per-axis kurtosis
18 and zero cross co-kurtosis at every time step.** The injected cloud
`ytilde = y^+ + Xi B^T` inherits this fourth-moment artifact whenever
`||B||` is non-negligible, and the higher-moment correction then spends its
noisy-target, tuned-strength budget removing the design's own artifact each
step — feeding exactly the recycled shape error that the pairwise trial
identified (`E[z_i^2 z_j^2] = 0 vs 1` is quoted there; the kurtosis-`d`
diagonal defect derived here is additional and worse at `d = 18`).

**The repair.** Draw `Z ∈ R^{N×d}` standard normal **once** (fixed recorded
seed), and exactly whiten:

```text
Xi* = (Z - 1 zbar^T) Chat^{-T},   Chat Chat^T = N^{-1} (Z - 1 zbar^T)^T (Z - 1 zbar^T).
```

Then **exactly** `N^{-1} Xi*^T 1 = 0` and `N^{-1} Xi*^T Xi* = I_d` (the two
design identities the chapter requires), while the higher moments are those of
a standardized Gaussian sample:

```text
kurtosis_a  = 3 + O_p(N^{-1/2})   (studentized SD ≈ sqrt(24/N) ≈ 0.154 at N = 1008)
co-kurt_ab  = 1 + O_p(N^{-1/2})   (studentized SD ≈ sqrt(4/N)  ≈ 0.063 at N = 1008)
co-skew, skew ≈ 0 + O_p(N^{-1/2})
```

(These are first-order studentized-Gaussian heuristics; raw `m22` has
`sqrt(8/N)` SD, while full whitening has additional off-diagonal constraints.)
Exact whitening perturbs these at `O(1/N)`. The design
stays `theta`-independent, so the hard-zero design tangent remains **exact**
(`dXi = 0`) and the same-scalar total-derivative claim is untouched. The
`N ≡ 0 (mod 2d)` divisibility constraint disappears. Note the connection to
GenUT: the chapter proves a positive `2d+1`-atom rule with Gaussian kurtosis
is impossible for `d >= 4` (`w_0 = 1 - d/3 < 0`); the `N`-atom Monte Carlo
design is the positive, equal-weight way to get approximate Gaussian moment
matching at any `d`. This is a one-tensor swap in the existing route — the
cheapest intervention aimed directly at the measured mechanism. New tuning
scope per the per-scope tuning rule.

### 3.2 Gain caps: raise `lambda`, `delta`, `delta_p`

Section 2's worst-case gains are `lambda^{-1/2}`, `1/delta`,
`delta_p^{-1/2}`. Raising them trades moment-restoration accuracy for tangent
stability; they are already tuned controls, but tuning has so far minimized
value/score *variance proxies* or moment residuals — never a stability
functional. A retune with `gammahat` (D1) as an explicit veto ("no arm with
`gammahat > 0`") is the principled version.

### 3.3 Antithetic innovations

Replace the scalar by `Lhat_anti = (Lhat(Z) + Lhat(-Z))/2`. This is a
**different finite scalar**, still deterministic and same-scalar
differentiable (score = average of the two tangent runs). Engineering
evidence (mechanics-only after the non-DGP demotion): SD ratios
`0.35–0.58` with paired-bootstrap support. Composes with 3.1/3.2. Cost 2x.

### 3.4 Post-processing control variates (campaign estimator, not the scalar)

The per-time score increments are already emitted
(`diagnostics["score_increments"]`). Across seeds, regress the score error on
a cheap correlated statistic with known conditional mean (e.g. the SGQF or
sigma-point score at the same `theta`, which is deterministic):
`score_cv = score - beta (h - E h)` with `beta` fit on tuning seeds only.
Reduces comparison variance `∝ (1 - corr^2)`; does not touch any single-run
scalar or its derivative.

### 3.5 ESS-triggered reset skipping — allowed only as a fixed schedule

Skipping the OT/reset when `ESS_t >= tau N` removes amplifying maps. But an
`ESS_t(theta)` trigger makes the branch `theta`-dependent: `Lhat^N` becomes
discontinuous at crossings and the fixed-branch derivative claim fails there.
Either use a `theta`-independent schedule (reset every `k`-th step) or reject
this remedy for score-bearing runs.

## 4. Replacement estimators of the exact score (Object B)

### 4.1 Fisher identity

For `p_theta(y_{1:T}) = ∫ p_theta(x_{0:T}, y_{1:T}) dx`, under dominated
convergence (interchange of `∇_theta` and the integral),

```text
∇ log p(y_{1:T}) = ∫ ∇ p(x, y) dx / p(y)
                 = ∫ [∇ log p(x, y)] · p(x, y)/p(y) dx
                 = E[ ∇_theta log p_theta(x_{0:T}, y_{1:T}) | y_{1:T} ].       (4.1)
```

Markov structure makes the integrand additive:

```text
∇ log p(x, y) = s_0(x_0) + sum_{t=1}^T s_t(x_{t-1}, x_t),
s_0 = ∇ log mu_theta(x_0),
s_t = ∇_theta log f_theta(x_t | x_{t-1}) + ∇_theta log g_theta(y_t | x_t).    (4.2)
```

**No derivative of any resampling/transport map appears.** The score is a
smoothed additive functional; any consistent smoothing estimator gives a
consistent score estimator. For our adapters (additive Gaussian transitions
`x_t = m_theta(x_{t-1}) + Sigma_q^{1/2} eta`):

```text
∇_theta log f = (x_t - m_theta(x_{t-1}))^T Sigma_q^{-1} ∂_theta m_theta(x_{t-1})
                - (1/2) ∂_theta log det Sigma_q
                - (1/2) (x_t - m)^T ∂_theta(Sigma_q^{-1}) (x_t - m),
```

e.g. for `Sigma_q = q^2 I`: `∇_q log f = ||x_t - m||^2 / q^3 - d/q`. The
`∂_theta m_theta` terms reuse the existing adapter transition tangents with
zero state tangent (Austria: the RK4 tangent). All TF-native and batch-native.

### 4.2 Path-space (OPG) estimator — why it degenerates

With categorical resampling and ancestors `a_t(i)`, carry
`S_{t,i} = S_{t-1, a_t(i)} + s_t(x_{t-1, a_t(i)}, x_{t,i})` and report
`sum_i alpha_{T,i} S_{T,i}`. Failure mode: **ancestral coalescence** — for
`T - s` large the time-`T` population shares `O(1)` distinct time-`s`
ancestors, so early increments are effectively single-path (no `1/N`
averaging), and correlated across `t`; the variance grows quadratically in
`T`. Literature: Poyiadjis, Doucet & Singh (2011, Biometrika) prove the
`O(T^2)` rate for this estimator and `O(T)` for the marginal version below.
**Status: cited, not re-verified locally** — fetch the paper into
`.localresources/` before any implementation decision (local-copy policy);
the coalescence argument above is the qualitative derivation. Additional
obstruction here: our route has no ancestry (the reset is a deterministic
barycentric/moment map), so OPG would anyway require a categorical-resampling
arm. Not recommended.

### 4.3 Marginal / forward-smoothing estimator (O(N^2) per step) — the serious candidate

Define `T_t(x) = E[ sum_{s<=t} s_s | x_t = x, y_{1:t} ]`. By the Markov
property, conditionally on `x_t` the past depends on `y_{1:t-1}` and the
transition into `x_t`, through the backward kernel

```text
B_t(x_t, dx_{t-1}) ∝ p(x_{t-1} | y_{1:t-1}) f_theta(x_t | x_{t-1}) dx_{t-1},
```

and the tower property gives the exact recursion

```text
T_t(x_t) = E_{x_{t-1} ~ B_t(x_t, ·)} [ T_{t-1}(x_{t-1}) + s_t(x_{t-1}, x_t) ]. (4.3)
```

Particle version: with a filter approximation `sum_j beta_{t-1,j} delta_{z_{t-1,j}}`
of `p(x_{t-1}|y_{1:t-1})`,

```text
W_t(i,j) = beta_{t-1,j} f_theta(x_{t,i}^- | z_{t-1,j})
           / sum_k beta_{t-1,k} f_theta(x_{t,i}^- | z_{t-1,k}),
That_{t,i} = sum_j W_t(i,j) [ That_{t-1,j} + s_t(z_{t-1,j}, x_{t,i}^-) ],      (4.4)
Shat_T = sum_i alpha_{T,i} That_{T,i}.
```

Properties:

- **No transport/reset derivative anywhere.** The whole Jacobian-product
  channel of Sect. 2 is absent by construction. Variance grows linearly in
  `T` (cited as in 4.2, qualified).
- Cost `O(N^2)` transition-density evaluations per step: `N = 1008` gives
  `~1.0e6` Gaussian evaluations per step — a `[N, N]` kernel, trivially GPU
  batched; adopt the repository chunking discipline for the `N×N` workspace.
  Requires an evaluable transition density (true for all current adapters).
- **It estimates Object B, not Object A**. Consistency and any finite-`N`
  bias rate require the usual particle approximation assumptions and are not
  established for the OT/reset route here. The same-scalar contract is lost if
  used as the HMC gradient — use Proposition 1 (energy = `Lhat^N`, force =
  `Shat_T` with its own frozen seeds) to keep exactness.
- **Reset-compatibility caveat (important).** Using the pre-reset weighted
  cloud `(x_{t-1}^-, alpha_{t-1})` as `(z, beta)` keeps (4.4) well-defined on
  our route, but the forward particles `x_{t,i}^- = f(x_{t-1,i}^+)` are
  propagated from the **reset** cloud, whose barycentric/moment map is a
  changed object relative to categorical resampling (the chapter's
  `D_B` vs `D_sharp` discussion). The estimator then inherits the reset's
  changed-object bias through the atom locations. Therefore:
  - as a **runtime force** on the current route: acceptable (Prop. 1 makes
    any deterministic force valid);
  - as a **reference arm**, first run it on a plain multinomial PF (TF-native;
    large `N`; replicated) where the particle approximation is the standard
    one. Do not call either implementation an exact oracle or transfer a
    literature variance rate to the OT/reset program without a separate
    validation.

### 4.4 Full particle MCMC (pseudo-marginal) — out of near-term scope

PMMH requires a nonnegative **unbiased** likelihood estimator; the OT/moment
route is not unbiased (chapter statement), so PMMH would require the
categorical-resampling family and a different HMC-free sampler (or
correlated pseudo-marginal). Record as the boundary option; not proposed now.

## 5. Is the Zhao–Cui moment-teacher idea sound? Is it useful here?

**What it is.** A deterministic squared-TT density filter run in parallel,
`qtilde_t = h_t^2 + tau_t q_{0,t}` fitted by fixed-ALS to the square-root
pulled-back target; its standardized shape moments
`(gamma_i, kappa_i, C^(3)_{ij}, C^(4)_{ij})` and their **total** tangents
(sequential ALS JVP `H_u cdot_u = bdot_u - Hdot_u c_u`, carried-marginal
quotient tangent, defensive-term tangent) replace the empirical targets in the
higher-moment correction, while Contract E keeps the particle `(mu_w, P_w)`.
Labeled `extension_or_invention` — using TT moments as Contract E targets is
not a Zhao–Cui claim.

**Soundness verdict: sound as a finite program.** The chapter's propositions
are internally consistent and the 2026-07-30 mechanics campaign verified the
pieces: square-root target semantics (`h^2` represents `e^{-c} qbar`),
paired-core observable contraction `∫ g h^2 dnu = e_1^T ∏ A_k[g_k] e_1`
against dense quadrature, sequential fixed-ALS value/JVP by finite
differences, graph-native shape-target JVP parity, hybrid preservation of
`(mu_w, P_w)`, and the ordered/symmetric mask repair. The total-score
proposition correctly demands the teacher-target tangents be total (a frozen
teacher is a different, declarable program; a partially-differentiated one is
wrong relative to the total-score claim).

**Usefulness for THIS problem: partial — it addresses one of three channels.**
Decompose the observed score-variance mechanism:

1. **Target noise:** empirical `s_a, k_a, C^(3), C^(4)` are `O_p(N^{-1/2})`
   noisy and their weight-tangents are noisy; the correction chases cloud
   noise. Evidence this channel is real: the LGSSM cross-model result —
   empirical pairwise targets *hurt* a Gaussian scope where the true targets
   are the constants `(0, 1)`. **The teacher fixes exactly this channel**
   (smooth deterministic targets and tangents).
2. **Injected design artifact:** kurtosis `d`, co-kurtosis `0` per reset
   (Sect. 3.1). The teacher does **not** fix this; the design swap does, at
   near-zero cost.
3. **Jacobian-product amplification** through weights/OT/Contract E
   (Sect. 2). The teacher does **not** fix this either — and it adds more
   differentiated machinery to the route (though with smooth tangents), plus
   the same value-shift risk that failed the pairwise promotion gate
   (teacher-target corrections still move the carried cloud, and TT bias
   enters the finite value).

Cost/feasibility boundary: the exact transformed-SV particle/teacher
composition is now assembled and passed its hard feasibility gates, but it
showed no statistically supported improvement over empirical Contract E.
Austria `d = 18` TT rank behavior remains untested; TF32 remains disabled for
this lane; and the per-HMC-evaluation cost is one TT fit plus ALS JVP per time
step per leapfrog step, which may dominate the filter.

**The degenerate teacher that is free.** For near-Gaussian scopes the teacher
can be constants: explicit targets `(skew, kurt, co-skew, co-kurt) =
(0, 3, 0, 1)` with **zero tangents** — a fully declared different finite
program using the existing `explicit_target_*` hooks, deterministic, costless,
and exactly what the cross-model note proposed for LGSSM. The TT teacher's
genuine comparative advantage is confined to strongly non-Gaussian scopes
(SIR near boundaries, SV tails) where neither noisy empirical targets nor
Gaussian constants are right.

**Verdict.** Sound: yes (as mathematics and now as mechanics). Correctly
gated: yes (no model-scale claims made). Useful for the current score
problem: **not as the first-line repair** — it attacks channel 1 only, at the
highest cost of any option on the table, while channel 2 has a one-tensor fix
and channel 3 has measurement (D1) and damping/force-splitting responses.
Keep it as the research track for genuinely non-Gaussian shape targets,
behind three gates: (i) rank/cost feasibility at `d = 18`, (ii) an
LGSSM/dense-parity value+score gate of the assembled composition, (iii)
evidence that Gaussian-constant targets are insufficient on the target model.

## 6. Recommended order (each item needs its own plan; no run is authorized here)

| # | Action | Object | Cost | What it decides |
|---|---|---|---|---|
| 1 | Lyapunov probe D1 on Austria + LGSSM, diagonal vs pairwise arms | diagnostic | trivial (extra tangent columns) | is the instability a `gamma > 0` tangent mode; per-arm before/after number for every later repair |
| 2 | Whitened-Gaussian residual design swap (3.1), fresh tuning scope | same scalar (new route arm) | one tensor + retune | does removing the injected kurtosis-`d`/co-kurt-0 artifact collapse the variance without the pairwise value shift |
| 3 | Gaussian-constant explicit targets on LGSSM (5) | same-scalar family, declared program | free | validates/refutes the target-noise channel cleanly against the Kalman oracle |
| 4 | Antithetic overlay (3.3) on the winner of 2/3 | different declared scalar | 2x | supported variance reduction stacking |
| 5 | PDS forward-smoothing reference oracle (4.3) on a multinomial PF arm, large `N`, `R` replications | Object B reference with MCSE | `O(N^2 T)` offline | first nonlinear score reference with error bars; `log_kappa` bias localization |
| 6 | Split force/energy HMC (Prop. 1) if a cheaper/lower-variance force emerges from 2–5 | exact MCMC on `Lhat^N` | wrapper change | decouples correctness from score variance permanently |
| 7 | Zhao–Cui teacher integration behind its three gates (5) | changed finite program | highest | non-Gaussian shape targets, only where 3 shows constants fail |

## 7. Claim-class ledger

- **Derived in this note:** Prop. 1 and its proof; the tangent recursion
  (2.1)–(2.2) and stage-gain inventory (each factor matched to audited code);
  cubature design fourth-moment identities (kurtosis `= d`, co-kurtosis
  `= 0`); whitened-design exact first/second-moment identities; Fisher
  identity (4.1)–(4.2); backward-kernel recursion (4.3)–(4.4); the
  ESS-branch discontinuity; the `w_0 = 1 - d/3` infeasibility use.
- **Cited, not locally re-verified:** `O(T^2)` vs `O(T)` variance rates for
  path vs marginal score estimators (Poyiadjis–Doucet–Singh 2011; the
  coalescence argument above is the qualitative derivation). Action: store
  the paper in `.localresources/` and read the theorem/proof sections before
  implementing item 5. Chopin(20) is locally available in `docs/` for the
  smoothing-chapter cross-check; not yet inspected for this note.
- **Mechanism hypotheses (testable, not established):** `gamma > 0` on
  Austria diagonal-only; the pairwise map as a local contraction; the design
  artifact as the dominant recycled-shape-error source. Item 1–2 discriminate.
- **Empirical, scope-limited:** all Austria/LGSSM/KSC/predator-prey numbers
  quoted from the 2026-07-30 artifacts (16 fixed seeds, fixed data, declared
  bootstraps).
- **Not checked:** GPU/TF32 drift of any new arm; PDS estimator bias when run
  on the OT-reset route (Sect. 4.3 caveat); TT rank growth at `d = 18`.

## 8. Nonclaims

No exact-posterior score, no unbiasedness of `Lhat^N`, no method superiority,
no default/HMC/leaderboard promotion. Variance reductions cited are
scope-specific. Proposition 1 guarantees invariance of surrogate-force HMC,
not good mixing; acceptance quality under a replacement force is an empirical
question for its own plan.
