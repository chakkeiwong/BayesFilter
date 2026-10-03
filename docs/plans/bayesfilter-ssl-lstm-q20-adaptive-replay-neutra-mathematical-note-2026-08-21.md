# Adaptive replay and tempered transport ensembles for SSL-LSTM NeuTra: mathematics, proofs, corrections, and limits

Original date: 2026-08-21  
Substantive revision: 2026-08-28  
Status: `REPLAY_THEOREMS_RETAINED_HIGH_DIMENSIONAL_FOUNDATION_WITHDRAWN_TEMPERED_RKL_ENSEMBLE_PROPOSED`

## Revised verdict

A fixed-capacity, content-independently refreshed buffer fed by a fixed valid
block law can target the desired transport objective and can converge under
explicit assumptions. If the particle proposal adapts forever, a second
proved route uses a fresh valid block for the persistent gradient and permits
stale replay only with summable cumulative influence. Constant-weight stale
replay from an evolving proposal remains unproved. Combining either proved
route with fresh NeuTra base draws has an exact population optimum: if the flow
family contains a transport from the Gaussian base to the target, then the
hybrid objective is minimized exactly at that transport and the pulled-back
target is Gaussian.

This is a conditional mathematical result. It does **not** prove that the
existing six 100-particle SMC training populations, the implemented finite
dense IAF, the current optimizer, or the SSL-LSTM target satisfy the required
assumptions. In particular, the existing normalized-only SMC artifacts support
a finite empirical forward-KL fit; they do not provide an unbiased
unnormalized target-measure estimator.

The 2026-08-28 revision changes the scientific recommendation. The replay
theorems remain conditional mathematical statements, but proposal support and
asymptotic correctness do not imply usable finite-sample accuracy. If the
learned transport supplies the proposal used to build its own target-side
training measure, a mode that the transport assigns negligible probability can
be absent from every finite block. Reweighting cannot repair an unobserved
region, and using the resulting measure to validate the same transport is
circular. Under product mismatch, the second moment of importance weights can
grow exponentially with dimension even when every proposal density is positive
everywhere. Therefore adaptive particle replay is no longer the primary
proposal for future high-parameter SSL-LSTM NeuTra.

The new primary candidate retains the original paper-style reverse-KL objective
and its fresh IID Gaussian base draws. It trains a deliberately diversified
ensemble of invertible transports along a proper temperature path, combines
them as a probability mixture rather than an arithmetic average of maps, and
uses the frozen transports as multiple exact HMC coordinate charts inside
replica exchange. The mixture and temperature continuation are discovery and
geometry mechanisms. Exactness comes only from the transformed target,
Metropolis corrections, fixed state-independent mixtures of invariant kernels,
and the replica-exchange swap ratio. No finite ensemble or temperature ladder
proves exhaustive mode discovery.

The new construction and the correction of the former recommendation are
proved in Sections 15--25. The original replay analysis in Sections 1--13 is
preserved because it remains useful for a later optional correction or
diagnostic lane after independent coverage evidence exists.

## 1. Claimed target and notation

The target in this note is the fixed UKF-defined q=20 parameter posterior used
by the current BayesFilter target. It is not the exact nonlinear-state-space
posterior. Let

```text
Theta = R^d,                 d = 4 for the active q=20 parameter target,
tilde_pi(theta) > 0,         unnormalized target density,
Z = integral tilde_pi(theta) dtheta in (0, infinity),
pi(theta) = tilde_pi(theta) / Z,
rho(z),                      standard Gaussian density on R^d.
```

Let `T_phi : R^d -> R^d` be a continuously differentiable bijection with a
continuously differentiable inverse and nonzero Jacobian determinant. Its
pushforward density is

```text
q_phi(theta)
  = rho(T_phi^{-1}(theta))
    * abs(det D T_phi^{-1}(theta)).                         (1)
```

The target pulled back through the same map is

```text
pi_phi^z(z)
  = pi(T_phi(z)) * abs(det D T_phi(z)).                     (2)
```

The score with respect to the transport parameters, at fixed physical point,
is

```text
s_phi(theta) = grad_phi log q_phi(theta).                   (3)
```

### Assumption set A: density and differentiation

The identities below use the following assumptions where stated.

1. `tilde_pi` and `rho` are measurable, positive almost everywhere on their
   declared support, and `0 < Z < infinity`.
2. Each admitted `T_phi` is a `C^1` diffeomorphism.
3. `log q_phi(theta)` is differentiable in `phi` for almost every `theta`.
4. Differentiation may be interchanged with both the target-side integral in
   (4) and the base-side integral in (7), locally uniformly on the parameter
   set being considered. For example, `||s_phi(theta)||` has a pi-integrable
   envelope and

   ```text
   ||grad_phi log abs(det D T_phi(z))||
   + ||grad_theta log tilde_pi(T_phi(z))||
       * ||grad_phi T_phi(z)||
   ```

   has a rho-integrable envelope. The same base-side envelope is required for
   the conditional expectation in (28).
5. Every importance proposal covers the target: `pi << r`, with the stated
   second-moment conditions when a variance or stochastic-approximation claim
   is made.

These are proof assumptions, not established properties of the active q=20
runtime. Target-status failures or a non-differentiable numerical branch would
violate them.

## 2. The two population objectives

Define the mass-covering or forward objective

```text
F(phi) = KL(pi || q_phi)
       = integral pi(theta) log[pi(theta) / q_phi(theta)] dtheta.       (4)
```

Define the paper-style NeuTra or reverse objective

```text
R(phi) = KL(q_phi || pi).                                      (5)
```

The original NeuTra derivation samples fresh `z ~ rho`, maps it through the
current `T_phi`, and evaluates the target there. Equations (2)--(3) of Hoffman
et al. give the corresponding reparameterized ELBO. The locally stored source
is
`.localresources/papers/multimodal_hmc/hoffman-sountsov-dillon-2019-neutra.txt`,
lines 87--125.

### Proposition 1: change-of-variables identities

For every admitted `phi`,

```text
q_phi(T_phi(z)) * abs(det D T_phi(z)) = rho(z),              (6)

R(phi) = KL(rho || pi_phi^z).                                (7)
```

#### Proof

Substitute `theta = T_phi(z)` into (1). Since
`D T_phi^{-1}(T_phi(z)) = [D T_phi(z)]^{-1}`, the two Jacobian
determinants cancel and give (6). For (7), change variables
`theta = T_phi(z)` in (5), use (6), and recognize (2):

```text
R(phi)
  = integral rho(z)
      log {rho(z) / [pi(T_phi(z)) abs(det D T_phi(z))]} dz
  = KL(rho || pi_phi^z).
```

This proves both identities. `QED`

### Proposition 2: exact gradients of the two objectives

Under Assumption set A,

```text
grad F(phi) = -E_pi[s_phi(theta)],                            (8)
```

and, using fresh `z ~ rho`,

```text
grad R(phi)
  = E_rho grad_phi [
        log rho(z)
        - log abs(det D T_phi(z))
        - log tilde_pi(T_phi(z))
      ].                                                      (9)
```

The omitted `log Z` in (9) is constant in `phi`.

#### Proof

Only `-log q_phi(theta)` in (4) depends on `phi`. Differentiating under the
integral gives (8). Equation (7) expands to

```text
E_rho[log rho(z) - log pi(T_phi(z))
      - log abs(det D T_phi(z))].
```

Replace `log pi` by `log tilde_pi - log Z`, differentiate under the base
expectation, and obtain (9). `QED`

Equation (9) includes the target score through the total derivative of
`log tilde_pi(T_phi(z))`. Reusing a target value or score evaluated at an old
physical point `T_phi_old(z)` is wrong relative to (9) after `phi` changes.

## 3. What the current 600-row training computes

The current trainer computes

```text
F_hat_fixed(phi)
  = -sum_{i=1}^600 W_i log q_phi(theta_i),                    (10)
```

where the same rows and weights are used for every optimizer update. This is
implemented by `WeightedForwardKLNeuTraTrainer` in
`bayesfilter/inference/neutra_weighted_training.py`, lines 604--714, and the
q=20 runner calls it on the unchanged full batch in
`docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py`,
lines 343--414 and 653--700.

Equation (10) is a valid objective for the discrete empirical measure

```text
pi_hat_fixed = sum_i W_i delta_{theta_i}.                     (11)
```

It is not equal to (4) unless the empirical integration error happens to be
zero for the entire function class `{log q_phi : phi in Phi}`. Repeating (10)
reduces optimization error for (10); it does not reduce the Monte Carlo error
between (10) and (4).

## 4. Exact proposal-aware particle blocks

This section gives a forward-replay block whose gradient is unbiased up to the
single positive constant `Z`. Avoiding self-normalization is useful because
`Z` does not depend on `phi` and therefore need not be known to locate the
forward-KL stationary points.

### Definition 1: known-density deterministic-mixture block

At generation round `b`, freeze proposals
`r_{b1}, ..., r_{bH}` before drawing the block. Draw exactly `n_{bh}` samples
from proposal `r_{bh}`, set

```text
N_b = sum_h n_{bh},
alpha_{bh} = n_{bh} / N_b,
m_b(theta) = sum_h alpha_{bh} r_{bh}(theta),                 (12)
```

and assume `m_b(theta) > 0` wherever `tilde_pi(theta) > 0`.
For an integrable vector function `f`, define

```text
gamma_hat_b(f)
  = (1/N_b) sum_h sum_{i=1}^{n_{bh}}
      [tilde_pi(X_{bhi}) / m_b(X_{bhi})] f(X_{bhi}),          (13)

X_{bhi} independently distributed as r_{bh}.                 (14)
```

The proposal definitions, allocation fractions, behavior-transport identity
when applicable, and `log m_b(X)` must remain attached to the block.

### Theorem 1: conditional unbiasedness of a known-density block

Let the proposals in (12) be chosen from any past history, but frozen before
the new block is drawn. For every integrable `f` fixed before those draws,

```text
E[gamma_hat_b(f) | past and frozen proposals]
  = integral tilde_pi(theta) f(theta) dtheta.                 (15)
```

Consequently,

```text
G_hat_b^F(phi) = -gamma_hat_b(s_phi)                          (16)
```

is conditionally unbiased for `Z grad F(phi)` when `phi` is fixed before the
block is drawn.

#### Proof

Using `n_{bh}/N_b = alpha_{bh}` and linearity of expectation,

```text
E gamma_hat_b(f)
  = sum_h alpha_{bh}
      integral r_{bh}(theta)
        [tilde_pi(theta) / m_b(theta)] f(theta) dtheta

  = integral [sum_h alpha_{bh} r_{bh}(theta)]
      [tilde_pi(theta) / m_b(theta)] f(theta) dtheta

  = integral tilde_pi(theta) f(theta) dtheta.
```

The second conclusion follows from (8). `QED`

This theorem remains true for proposals centered on known modes or for a
proposal containing a frozen earlier transport, provided every density in
(12) is evaluated correctly. It does not authorize caller-stamped proposal
density values.

### Why the unknown normalizer is harmless for a pure forward gradient

By (8) and (15), the estimator in (16) targets `Z grad F`. Since `Z > 0` is a
constant, `Z grad F(phi) = 0` if and only if `grad F(phi) = 0`. It changes the
gradient scale, not the forward objective's stationary set.

When the forward gradient is combined with a reverse gradient, the effective
population objective is

```text
J_{a,b}(phi) = a Z F(phi) + b R(phi),     a > 0, b > 0,       (17)
```

not `a F + b R` unless `Z` has been estimated and absorbed into `a`. This scale
matters for optimization and must be tuned; it does not change the exact common
minimizer proved below.

## 5. SMC population blocks are a different evidence class

An SMC terminal population normally provides normalized weights
`W_{bi}` and the estimate

```text
pi_hat_b(f) = sum_i W_{bi} f(theta_{bi}).                     (18)
```

Under standard assumptions this estimator is consistent as the number of
particles grows. The stored Del Moral--Doucet SMC source describes normalized
estimators and their consistency at
`.localresources/papers/multimodal_hmc/del-moral-doucet-2002-smc-samplers-preprint.txt`,
lines 400--426. It separately discusses normalizing-constant estimation and
the conditions for unbiasedness at lines 428--470.

Normalized (18) is not generally an unbiased estimator of `E_pi[f]` at finite
particle count. Therefore, future SMC replay has two mathematically distinct
routes.

### Route SMC-U: proof-bearing unnormalized blocks

The SMC implementation may emit a proper unnormalized estimator
`gamma_hat_b(f)` satisfying

```text
E[gamma_hat_b(f) | frozen SMC protocol] =
  integral tilde_pi(theta) f(theta) dtheta.                   (19)
```

For example, a repository implementation could prove (19) by induction for
its exact Feynman--Kac construction and unbiased resampling scheme. The
algorithm, incremental potentials, normalizer estimator, pre/post-resampling
convention, and test functions must all be bound in the artifact. Merely
storing normalized terminal weights does not establish (19).

If (19) is established for the actual route, `-gamma_hat_b(s_phi)` can replace
(16) in the proof-bearing replay algorithm.

### Route SMC-N: normalized consistency-only blocks

Assume an SMC law of large numbers for the relevant transport-score class:

```text
sup_{phi in C} ||pi_hat_b(s_phi) - E_pi[s_phi]||
  -> 0 in probability as N_b -> infinity.                    (20)
```

Then the normalized population gradient

```text
-sum_i W_{bi} s_phi(theta_{bi})                               (21)
```

is uniformly consistent for `grad F` as `N_b -> infinity`.
This is a consistency statement, not finite-`N_b` unbiasedness.

### Proposition 3: why endlessly rotating fixed-size normalized blocks is not
an exactness proof

Suppose independent SMC-N blocks all have the same fixed particle count `N`.
Let `H_b(phi)` denote (21). If the blocks have finite first moment, averaging
infinitely many blocks gives

```text
(1/B) sum_{b=1}^B H_b(phi) -> E[H_1(phi)]                    (22)
```

by the law of large numbers. In general
`E[H_1(phi)] != grad F(phi)` because (18) is self-normalized. Increasing the
number of fixed-size runs removes between-run variance around the finite-`N`
expectation; it does not by itself remove finite-`N` bias.

#### Proof

Equation (22) is the ordinary law of large numbers. The second statement
follows because equality of the two expectations was not assumed and is not a
general property of a ratio estimator. Consistency in (20) removes the gap by
increasing `N`, not by asserting finite-`N` unbiasedness. `QED`

An explicit generic witness is already visible at `N=1`: the sole normalized
weight is `W_1=1` regardless of the unnormalized target weight. Thus
`E[pi_hat(f)] = E_r[f]`, which differs from `E_pi[f]` whenever the proposal
`r` differs from `pi` on the relevant function `f`. For `f=s_phi`, this gives
`E[H_1(phi)] = -E_r[s_phi]` rather than `grad F(phi)` generically. This is a
finite-`N` counterexample to automatic debiasing, not a numerical estimate of
the actual `N=100` artifact.

For the current artifacts, `N=100` and only normalized weights are loaded.
They are therefore SMC-N finite empirical blocks. They cannot be upgraded to
SMC-U by equalizing each run's total replay mass.

## 6. A fixed-capacity rotating replay buffer

The experience-replay proposal is now formalized at the population-block
level. Individual SMC rows are not the replacement unit.

### Definition 2: content-independent block refresh

Let the buffer have `K >= 1` slots. A block record contains a random function

```text
H(B, phi) = -gamma_hat_B(s_phi),                              (23)
```

where `gamma_hat_B` is either a known-density block satisfying Theorem 1 or an
SMC-U block satisfying (19). Let `P_B` be the fixed law of a fresh valid block.

At each refresh transition, each slot is independently replaced by a fresh
draw from `P_B` with probability `epsilon in (0,1]`; otherwise it is retained.
The refresh coin is independent of particle values, weights, target values,
mode labels, losses, and ancestry. Other content-independent schedules with a
uniform geometric refresh bound can be handled similarly.

The active forward gradient is

```text
G_F(phi, B_1:K) = (1/K) sum_{k=1}^K H(B_k, phi).              (24)
```

### Lemma 1: invariant law and geometric forgetting of the buffer

The refresh chain has unique invariant law `P_B^K`. For any two initial
buffers, a common-randomness coupling satisfies

```text
P(buffers have not coupled after t transitions)
  <= K (1 - epsilon)^t.                                      (25)
```

Consequently the buffer chain is uniformly geometrically ergodic.

#### Proof

Couple corresponding slots by using the same refresh coin and, on refresh,
the same new block. Once a slot refreshes, the two copies of that slot agree
forever under this coupling. A given slot has not refreshed after `t`
transitions with probability `(1-epsilon)^t`. The union bound over `K` slots
gives (25). A buffer of independent `P_B` blocks remains so after the
transition, proving invariance. The coupling bound gives uniqueness and
uniform geometric convergence to that invariant law. `QED`

### Corollary 1: correct stationary mean field

For every fixed `phi` satisfying the block integrability assumptions,

```text
E_{P_B^K}[G_F(phi, B_1:K)] = Z grad F(phi).                  (26)
```

#### Proof

Every slot has marginal law `P_B`; apply (15) or (19), then average. `QED`

This establishes the correct population mean. It does not make a reused
buffer gradient conditionally unbiased given transport parameters that were
trained on that same buffer.

## 7. Fresh reverse-KL queries plus rotating replay

At optimization step `t`, draw a fresh base batch
`z_{t1}, ..., z_{tM}` independently from `rho` after `phi_t` is fixed. Define

```text
G_R_hat(phi_t)
  = (1/M) sum_j grad_phi [
        log rho(z_{tj})
        - log abs(det D T_phi(z_{tj}))
        - log tilde_pi(T_phi(z_{tj}))
      ] at phi = phi_t.                                      (27)
```

By Proposition 2,

```text
E[G_R_hat(phi_t) | past] = grad R(phi_t).                    (28)
```

The combined update is

```text
phi_{t+1} = Project_C {
  phi_t - eta_t [a G_F(phi_t, B_t) + b G_R_hat(phi_t)]
},                                                           (29)
```

for positive coefficients `a,b` and a declared parameter set `C`.

### Theorem 2: a sufficient convergence theorem for fixed-capacity replay

Assume:

1. the block law `P_B` is fixed during the proof-bearing training phase and
   every block satisfies Theorem 1 or (19);
2. the buffer follows Definition 2 and is independent of the fresh base draws;
3. `C` is compact and convex, projection in (29) is Euclidean, and every
   unprojected update in the theorem remains in `C` almost surely, so the
   projection is inactive along the realized trajectory; this is an explicit
   sufficient interior-trajectory assumption, not a projected-boundary
   theorem;
4. `H(B,phi)` is uniformly bounded and uniformly Lipschitz in `phi` on `C`;
   Lemma 1 then makes the Poisson series (32) a bounded solution `u_phi` of
   (33), uniformly Lipschitz in `phi`;
5. the fresh reverse estimator is conditionally unbiased, uniformly Lipschitz
   in `phi`, and uniformly bounded; a bounded conditional second moment alone
   can be handled by a more general theorem but is not claimed here;
6. the step sizes are deterministic, nonincreasing, and satisfy
   `eta_t > 0`, `sum_t eta_t = infinity`, and
   `sum_t eta_t^2 < infinity`;
7. there are `phi_star in C` and `mu > 0` such that

   ```text
   grad J_{a,b}(phi_star) = 0,
   <phi - phi_star, grad J_{a,b}(phi)>
     >= mu ||phi - phi_star||^2       for every phi in C;     (30)
   ```

   this is a strong-stability hypothesis, not a property established for the
   finite dense IAF objective. For the implemented masked tanh network, it
   also cannot hold on any `C` containing two distinct symmetry-equivalent
   stationary points; the local-basin consequence is stated below; and
8. iterates remain in `C` and no target or numerical validity veto fires.

Then `phi_t -> phi_star` almost surely.

#### Proof

By Lemma 1, the buffer is uniformly geometrically ergodic with invariant law
`P_B^K`. By Corollary 1 and (28), the stationary mean of the bracketed update
in (29) is

```text
a Z grad F(phi) + b grad R(phi) = grad J_{a,b}(phi).          (31)
```

Geometric ergodicity and Assumption 4 imply that the centered buffer function
admits the convergent Poisson series

```text
u_phi(B) = sum_{j=0}^infinity
  [P^j G_F(phi,B) - Z grad F(phi)].                           (32)
```

Here is the regularity argument used by Theorem 2. Let `P` be the buffer
kernel, let `mu_B=P_B^K` be its invariant law, and let `tau` be the coupling
time in Lemma 1 between a chain started at `B` and a stationary copy. The
refresh coins and replacement blocks used in that coupling are independent of
`phi`. If `U_H` is a uniform bound for `||G_F(phi,B)||` and `L_H` is a uniform
Lipschitz constant in `phi`, then

```text
||P^j G_F(phi,B) - mu_B(G_F(phi,.))||
  <= 2 U_H K (1-epsilon)^j,

||u_phi(B)|| <= 2 U_H K / epsilon.
```

For two parameter values, couple the same two chains and apply the Lipschitz
bound only on the event that they have not yet coupled. The integrand vanishes
after coupling, so

```text
||[P^j G_F(phi,B)-mu_B G_F(phi)]
  -[P^j G_F(phi',B)-mu_B G_F(phi') ]||
  <= 2 L_H ||phi-phi'|| K (1-epsilon)^j,

||u_phi(B)-u_phi'(B)||
  <= (2 K L_H / epsilon) ||phi-phi'||.
```

Thus the series is bounded and uniformly Lipschitz. This argument depends on
the phi-independent refresh kernel in Definition 2; it is not available for
an evolving proposal kernel without a controlled-Markov proof.

It satisfies the Poisson equation

```text
u_phi - P u_phi = G_F(phi,.) - Z grad F(phi).                 (33)
```

Take the order of one iteration to be: update with `B_t`, then draw
`B_{t+1}` from the refresh kernel `P(B_t,.)`. Define

```text
h_phi(B) = G_F(phi,B) - Z grad F(phi),

M^B_{t+1}
  = u_{phi_t}(B_{t+1}) - P u_{phi_t}(B_t),

M^R_{t+1}
  = G_R_hat(phi_t) - grad R(phi_t).                          (33a)
```

Both `M^B_{t+1}` and `M^R_{t+1}` have conditional mean zero. Equation (33)
gives the exact decomposition

```text
h_{phi_t}(B_t)
  = u_{phi_t}(B_t) - u_{phi_t}(B_{t+1}) + M^B_{t+1}.        (33b)
```

Let `g = grad J_{a,b}`, let
`M_{t+1}=a M^B_{t+1}+b M^R_{t+1}`, and introduce the transformed iterate

```text
y_t = phi_t - a eta_t u_{phi_t}(B_t).
```

Because the projection is inactive by Assumption 3, substituting (33b) into
(29) gives

```text
y_{t+1}
  = y_t - eta_t g(phi_t) - eta_t M_{t+1} + r_{t+1},         (33c)

r_{t+1}
  = a [eta_t u_{phi_t}(B_{t+1})
       - eta_{t+1} u_{phi_{t+1}}(B_{t+1})].
```

Boundedness and Lipschitz continuity of `u`, bounded update directions, and
nonincreasing `eta_t` imply

```text
sum_t ||r_{t+1}||
  <= c_1 sum_t (eta_t-eta_{t+1})
     + c_2 sum_t eta_{t+1} eta_t
  < infinity.                                                (33d)
```

The bounded martingale differences also satisfy
`sum_t eta_t^2 E[||M_{t+1}||^2 | history] < infinity`, so their weighted
series converges almost surely.

It remains to identify the deterministic drift. With
`e_t = phi_t - phi_star`, (30) gives

```text
-2 eta_t <e_t, grad J_{a,b}(phi_t)>
  <= -2 mu eta_t ||e_t||^2.                                (33e)
```

Moreover `||y_t-phi_t|| <= c eta_t`. Expanding
`V_t=||y_t-phi_star||^2` in (33c), using (33e), and absorbing this
`O(eta_t)` displacement and the absolutely summable `r_{t+1}` terms gives the
almost-supermartingale drift inequality for a deterministic nonnegative
summable bound `d_t`:

```text
E[V_{t+1} | history through t]
  <= V_t - 2 mu eta_t ||e_t||^2 + c_3 eta_t^2 + d_t,

d_t >= 0,    sum_t d_t < infinity.                           (33f)
```

The absorption of the remainder is pathwise; it does not assume that
`r_{t+1}` is history-measurable or conditionally centered. Let `U` bound
`||u_phi||`, let `L_u` be the Lipschitz constant established above, and let
`D` bound the full update direction. From (33c),

```text
r_{t+1}
  = a [(eta_t-eta_{t+1}) u_{phi_t}(B_{t+1})
       + eta_{t+1}(u_{phi_t}(B_{t+1})
                   -u_{phi_{t+1}}(B_{t+1}))],

||r_{t+1}|| <= beta_t
  := a U (eta_t-eta_{t+1}) + a L_u D eta_t eta_{t+1},
```

where `beta_t` is deterministic and summable. Compactness and the bounded
direction assumptions bound every cross term containing `r_{t+1}` by a fixed
constant times `beta_t`; those bounds are included in `d_t`. The only
uncentered noise cross term that needs cancellation is
`-2 eta_t <y_t-phi_star,M_{t+1}>`, and its conditional expectation is zero
because `y_t` is history-measurable and `M_{t+1}` is a martingale difference.
Finally,

```text
<y_t-phi_star, g(phi_t)>
  = <e_t,g(phi_t)> + <y_t-phi_t,g(phi_t)>
  >= mu ||e_t||^2 - c eta_t,
```

so the second term contributes only another `O(eta_t^2)` quantity to (33f).

To see the last implication without using it as an extra optimization
assumption, add the tail of the summable positive errors to `V_t`; the result
is a nonnegative supermartingale up to the nonnegative decrement
`2 mu eta_t ||e_t||^2`. The supermartingale convergence argument therefore
says that `V_t` converges and
`sum_t eta_t ||e_t||^2 < infinity` almost surely.  If the limiting squared
distance were positive, the latter sum would diverge because
`sum_t eta_t = infinity`. Since `V_t` converges and
`||y_t-phi_t|| -> 0`, the squared distance has the same limit, which must
instead be zero. Hence
`||e_t|| -> 0`, and therefore `phi_t -> phi_star` almost surely. `QED`

Theorem 2 proves that a fixed-capacity buffer can be refreshed forever and
still converge under an explicit sufficient stability condition. It does
**not** establish that condition for the current nonconvex dense IAF, and it
does not cover a proposal/block generator that changes arbitrarily with the
learned transport, constant learning rates, target-based eviction, or finite
normalized SMC-N blocks. Those variants require a separate controlled-Markov
or finite-error argument.

### Symmetry boundary for the implemented dense IAF

The local qualification above is forced by exact parameter symmetries in the
active implementation. In `_dense_masks`
(`bayesfilter/inference/neutra_weighted_training.py:190-216`), hidden units with the same degree
have identical mask columns in their incoming layer and identical mask rows in
their outgoing layer. Permuting two such units, including their biases and
the corresponding incoming/outgoing parameter coordinates, defines an
orthogonal parameter map `S` with

```text
T_{S phi} = T_phi,    q_{S phi} = q_phi,    J_{a,b}(S phi) = J_{a,b}(phi).
```

The active q=20 runner selects `activation="tanh"`
(`docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py:361-376`).
With `activation="tanh"`, independently negating one hidden unit's incoming
weights and bias and its outgoing row is another exact symmetry because
`tanh(-x)=-tanh(x)`. For either symmetry, differentiability gives
`grad J_{a,b}(S phi) = S grad J_{a,b}(phi)`. If `phi_star` is stationary and
`S phi_star != phi_star`, then applying (30) at `phi=S phi_star` would require

```text
0 = <S phi_star - phi_star, grad J_{a,b}(S phi_star)>
  >= mu ||S phi_star - phi_star||^2 > 0,
```

which is impossible. Therefore Theorems 2 and 2A are global parameter-space
claims only for a symmetry-free or quotient/gauge-fixed parameterization. For
the current dense IAF they are, at most, conditional local-basin results on a
set `C` containing one representative of each relevant symmetry orbit. This
does not invalidate either theorem; it prevents applying (30) globally.

### Theorem 2A: continual adaptive generation with summable stale replay

Let the proposal library at update `t` be any function of the history through
`phi_t`. Freeze it, draw a fresh known-density block satisfying Theorem 1 (or
a fresh SMC-U block satisfying (19)), and evaluate
`G_new_t^F` at `phi_t`. Independently draw the fresh base batch for
`G_R_hat(phi_t)`. Let `R_t` be any history-measurable gradient-like quantity
computed from a fixed-capacity stale replay buffer, including a buffer whose
contents are continually replaced. Consider

```text
phi_{t+1} = Project_C {
  phi_t - eta_t [
    a G_new_t^F + b G_R_hat(phi_t) + lambda_t R_t
  ]
},                                                          (33g)
```

where the proposal and fresh reverse estimators are conditionally unbiased for
`Z grad F(phi_t)` and `grad R(phi_t)`, respectively. Assume the compact
interior-trajectory, deterministic step-size, and strong-stability conditions
of Theorem 2. Require `G_new_t^F` and `G_R_hat(phi_t)` to be uniformly bounded
over all admitted adaptive proposal histories. Also assume

```text
lambda_t >= 0,    lambda_t deterministic,
sup_t lambda_t < infinity,
sum_t eta_t lambda_t < infinity,    sup_t ||R_t|| < infinity. (33h)
```

An adapted random schedule could be admitted only under the separate
condition `sum_t eta_t lambda_t < infinity` almost surely and a corresponding
adapted-noise proof; it is not included in Theorem 2A as written.

Then `phi_t -> phi_star` almost surely, even though the fresh proposal may
adapt forever and the stale buffer may use a content-dependent replacement
rule.

#### Proof

By Theorem 1 or (19), conditional on the pre-draw history,

```text
xi_{t+1}
  = a [G_new_t^F - Z grad F(phi_t)]
    + b [G_R_hat(phi_t) - grad R(phi_t)]
```

is a bounded martingale difference. Projection is inactive under the stated
interior-trajectory assumption. With `e_t=phi_t-phi_star`, expand the squared
distance after (33g), condition on the pre-draw history, use (30), and bound
`||e_t||` by compactness:

```text
E[||e_{t+1}||^2 | history]
  <= ||e_t||^2
     - 2 mu eta_t ||e_t||^2
     + c_4 eta_t^2
     + c_5 eta_t lambda_t.                                  (33i)
```

The final two terms are summable by (33h) and Assumption 6. The same
nonnegative almost-supermartingale argument used after (33f) shows that
`||e_t||^2` converges and
`sum_t eta_t ||e_t||^2 < infinity`. Since `sum_t eta_t=infinity`, the limit is
zero. `QED`

Theorem 2A is a rigorous way to keep generating particles from evolving
proposals while retaining a replay buffer. Its price is that old, potentially
dependent replay has only a summable cumulative effect; the proof-bearing
fresh block supplies the persistent forward-gradient signal. Constant
positive weight on adaptively generated stale blocks is not covered by
Theorem 2 or 2A unless a controlled-Markov or another finite-error proof is
supplied.

## 8. Exact minimizer and Gaussianization

### Lemma 2: nonnegativity and equality condition for KL

For probability densities `p` and `q` with `p << q`,

```text
KL(p || q) >= 0,                                             (34)
```

with equality if and only if `p=q` almost everywhere.

#### Proof

Let `A={x:p(x)>0}` and `c=integral_A q`, so `0<c<=1`. Apply
Jensen's inequality to the convex function `-log`:

```text
KL(p||q) = E_p[-log(q/p)]
          >= -log E_p[q/p]
          = -log c
          >= 0.
```

Strict convexity gives equality only when `q/p` is constant `p`-almost
everywhere and `c=1`. Its `p`-expectation is then one, so the constant is one;
`c=1` also leaves no `q` mass outside `A`. Thus `p=q` almost everywhere.
The converse is immediate. `QED`

### Theorem 3: common exact minimizer of the hybrid objective

Let `a>0`, `b>0`, and suppose there exists `phi_star` in the admitted family
such that

```text
q_{phi_star} = pi almost everywhere.                         (35)
```

Then

```text
J_{a,b}(phi) = a Z KL(pi||q_phi) + b KL(q_phi||pi) >= 0,     (36)
```

`J_{a,b}(phi_star)=0`, and every global minimizer with value zero satisfies
`q_phi=pi` almost everywhere.

#### Proof

Both terms in (36) are nonnegative by Lemma 2 and their coefficients are
strictly positive. At (35), both KL divergences are zero. Conversely, a sum of
two nonnegative terms with strictly positive coefficients is zero only if both
terms are zero; Lemma 2 then gives `q_phi=pi` almost everywhere. `QED`

Thus an unknown `Z`, an arbitrary positive relative objective scale, and the
addition of either KL direction do not move the exact attainable solution.
They can materially change finite optimization behavior.

### Proposition 4: existence in an unrestricted triangular class

Assume `rho` and `pi` are strictly positive densities whose successive
conditional cumulative distribution functions are jointly continuously
differentiable in the conditioned and conditioning variables, strictly
increasing in the conditioned variable, and have jointly continuously
differentiable inverses. Then a triangular `C^1` diffeomorphism `T_star` exists
with
`(T_star)_# rho = pi`.

#### Proof

For a density `p`, define its Rosenblatt map

```text
R_p(x)_1 = P_p(X_1 <= x_1),
R_p(x)_j = P_p(X_j <= x_j | X_{1:j-1}=x_{1:j-1}),  j>1.      (37)
```

The stated strict monotonicity and smoothness make `R_p` a triangular `C^1`
bijection from `R^d` to `(0,1)^d` with a `C^1` inverse. The conditional
probability integral transform gives `(R_p)_# p = Uniform((0,1)^d)`. Therefore

```text
T_star = R_pi^{-1} o R_rho                                  (38)
```

is a triangular `C^1` diffeomorphism and

```text
(T_star)_# rho
  = (R_pi^{-1})_# (R_rho)_# rho
  = (R_pi^{-1})_# Uniform
  = pi.
```

`QED`

This proposition is an existence result for a broad triangular class. It does
not prove that the implemented finite-stage, finite-width, bounded-scale dense
IAF contains `T_star` or approximates it well enough under the available
budget.

### Corollary 2: exact Gaussian pullback

If (35) holds, then

```text
pi_{phi_star}^z(z) = rho(z).                                 (39)
```

#### Proof

Use (2), substitute `pi=q_{phi_star}`, and apply (6). `QED`

This is the precise whitening claim. It says the transformed *distribution*
is Gaussian. It does not say a finite HMC chain is IID.

### Corollary 3: an ideal Gaussian HMC kernel can be independent

For target `rho=N(0,I)`, identity mass, a complete momentum refresh, and exact
Hamiltonian integration for time `tau=pi/2`, successive position states are
IID `N(0,I)`.

#### Proof

The exact Gaussian Hamiltonian equations have solution

```text
z(tau) = z(0) cos(tau) + p(0) sin(tau).                      (40)
```

At `tau=pi/2`, `z(tau)=p(0)`. The refreshed momentum is independent
`N(0,I)`, so the new position is independent of the old position and has law
`rho`. Exact dynamics has acceptance probability one. `QED`

The repository uses discrete leapfrog integration and a tuned finite
trajectory. Corollary 3 proves possibility, not that the current fixed-HMC
kernel attains independence.

## 9. Counterexamples and forbidden shortcuts

### Counterexample 1: content-based eviction changes mode mass

Let the target live on two regions `A,B` with probabilities `p` and `1-p`,
where `p != 1/2`. A buffer policy that always retains equal counts from `A`
and `B`, followed by uncorrected uniform replay, converges to the measure with
masses `(1/2,1/2)`, not `(p,1-p)`. It optimizes a different forward objective.
Mode quotas may be used only with explicit selection-probability correction or
as a labeled non-posterior training heuristic.

### Counterexample 2: row deletion breaks a normalized SMC population

Consider one two-row population with normalized weights `(0.9,0.1)` and a test
function with values `(0,1)`. Its estimate is `0.1`. If the first row is
deleted and the survivor is renormalized, the estimate becomes `1`; if the
second is deleted, it becomes `0`. Neither operation preserves the original
weighted measure. Replacing an entire independently normalized population
avoids this undefined partial-population normalization.

### Counterexample 3: missing proposal support cannot be repaired by weights

If `pi(A)>0` but every replay proposal has probability zero on `A`, then no
particle appears in `A` and `tilde_pi/r` is undefined there. The estimator in
Theorem 1 integrates only over the proposal support and cannot equal the
target integral. Full support is a mathematical requirement, not an ESS
diagnostic.

### Counterexample 4: fixed replay does not learn an unobserved bridge

Let two target densities agree at every fixed replay point but differ on an
open region between the two observed particle clouds. Equation (10) is
identical for the two targets for every `phi`, because it uses only the stored
points and weights. Therefore no optimizer using only (10) can distinguish
their bridge geometry. Fresh target queries can distinguish them.

### Counterexample 5: fresh reverse KL does not guarantee mode coverage

On a two-point target with masses `(1/2,1/2)`, restrict the variational family
to the two point masses. Either candidate has reverse KL `log 2`, while its
forward KL is infinite because it assigns zero mass to one target point. Thus
reverse KL can admit a single-mode solution in a restricted family. Fresh
base draws reduce fixed-sample error; they do not prove multimodal discovery.

## 10. What may and may not be replayed

| Record | May be replayed? | Mathematical condition |
|---|---|---|
| Known-density importance block | Yes | Keep its frozen proposal mixture and source density; use (13) |
| Proof-bearing unnormalized SMC block | Yes | The exact SMC implementation has established (19) |
| Existing normalized q=20 SMC block | Yes, as an empirical/consistency-only anchor | Do not call its finite gradient unbiased; preserve the whole population |
| Old physical point generated by an earlier flow | Not naively for reverse KL | It is off-policy after the flow changes; use a valid behavior-density importance identity or do not use it |
| Old base point `z` | Algebraically reusable, but finite replay recurs | Re-evaluate `T_phi(z)`, target value/score, and Jacobian at current `phi`; fresh `z` remains the proof-bearing default |
| Mode label or ancestry | Diagnostic/stratification metadata | Never use it for uncorrected content-based retention or posterior mass assignment |

## 11. Proof-bearing algorithm and practical extension

### Algorithm A: theorem-covered rotating replay

1. Freeze a full-support known-density proposal library for one training phase,
   or use an SMC-U implementation with a checked unnormalized estimator.
2. Initialize `K` independent valid population blocks.
3. Refresh blocks by a content-independent rule satisfying Definition 2.
4. At every optimization update, draw a fresh batch from `rho`, transform it
   with the current map, and evaluate the exact batch-native target and score.
5. Update with (29), using a diminishing step-size schedule and the stability
   assumptions of Theorem 2.
6. Select only on disjoint fresh validation blocks and fresh base draws.
7. Freeze the transport before HMC; no replay or retuning occurs inside HMC.

### Algorithm B: adaptive practical extension

An evolving transport can be added as a new proposal because Theorem 1 permits
history-dependent proposals for fresh blocks. Theorem 2A covers a fresh valid
block at every update plus stale replay satisfying (33h). Persistent reuse of
old adaptive blocks with nonsummable positive influence is outside Theorems 2
and 2A unless one of these is supplied:

- a controlled-Markov stochastic-approximation proof;
- an increasing-buffer uniform law of large numbers;
- a cross-fitted phase construction with a fresh proof-bearing gradient; or
- a replay coefficient whose accumulated optimization perturbation is
  summable.

The (33h) variant is a proved conditional algorithm. The constant-weight stale
replay variant remains a plausible research candidate, not a proved default.

## 12. Required block provenance

Every future replay block should bind at least:

- target and prepared-input signatures;
- block type (`known_density_mis`, `smc_u`, or `smc_n`);
- generation round, seed, particle count, and source identity;
- points, exact target values/status, and proposal or SMC weights;
- complete proposal-mixture definition and behavior-transport identity;
- SMC schedule, incremental-weight convention, normalizer evidence, and
  pre/post-resampling semantics when applicable;
- ancestry/root identifiers and mutation sign-transition diagnostics;
- selection/eviction probability and the proof route that authorizes it; and
- hashes of the immutable records.

The mode label is diagnostic. It must not silently define training mass.

## 13. SSL-LSTM evidence gates after implementation

The mathematical result does not promote an implementation. A future reviewed
campaign would need three separate ledgers.

### Assumption-realism gate

The uniform boundedness assumptions in Theorems 2 and 2A are strong proof
conditions, not diagnostics that the current runtime is presumed to satisfy.
For a known-density block, the relevant risk is an unbounded

```text
[tilde_pi(X) / m_b(X)] ||s_phi(X)||
```

over the proposal law. For fresh reverse queries, Gaussian base draws can
produce unbounded target-score or Jacobian contributions even when the target
is finite. A future campaign must therefore report support, importance-ratio
tails, score/Jacobian envelopes, and finite-moment diagnostics on disjoint
calibration data. Passing those diagnostics nominates a route but does not
prove an essential supremum bound. If only finite second moments are supported,
the run must use a separately proved second-moment stochastic-approximation
result or remain diagnostic-only; clipping or truncation changes the estimator
and requires an explicit bias analysis.

### Engineering correctness

- batch-native TensorFlow/XLA target and training;
- GPU memory growth established before device initialization;
- exact target/score and transport/Jacobian parity;
- proposal-density or SMC-weight identity tests;
- whole-population replacement and untouched validation/audit blocks; and
- no stale reverse-KL target evaluation.

### Numerical and transport validity

- finite target/status for every fresh query;
- proposal support and weight-tail diagnostics by generation round;
- ancestry and genuinely new-particle counts;
- per-source and per-mode heldout losses with uncertainty;
- pullback mean, covariance, marginal/tail, and dependence diagnostics on fresh
  target-side and fresh-flow samples; and
- no candidate selection from training-buffer loss alone.

### Sampler and scientific validity

- one frozen transformed target and one common HMC kernel;
- sequential warmup with warmup draws excluded;
- parameter and sign-indicator rank/folded R-hat and bulk/tail ESS;
- direct per-chain cross-sign transitions and initialization forgetting;
- no pooling of mode-locked chains; and
- predictive analysis only after sampler admission.

## 14. Corrected answer to "will replay work?"

Theorems 1, 2, 2A, and 3 establish a conditional existence result: refreshed
replay can have the correct mean field and can converge under the stated
support, moment, stability, and stochastic-approximation assumptions. They do
not establish that the required estimator is usable at finite cost in high
dimension. In particular, they do not answer how a proposal that misses an
unknown mode will discover it.

The earlier recommendation treated this gap as an empirical gate while still
placing adaptive replay on the main repair path. That recommendation was too
optimistic for the intended future SSL-LSTM setting. When the proposal is
materially derived from `q_phi`, the construction can use `q_phi` to generate a
finite target-side measure, train `q_phi` on that measure, and then report
agreement with the same measure. The importance identity remains correct, but
the global validation argument is circular. A defensive component proves
support only; it does not prove an adequate second moment or a feasible sample
count.

Accordingly, replay is retained only as an optional later correction or
diagnostic after an independent coverage argument is available. It is not the
foundation of the new high-dimensional proposal. Sections 15--25 replace that
foundation with multiple reverse-KL transports, temperature continuation, and
exact replica-exchange HMC.

## 15. Why full support does not solve finite high-dimensional sampling

Let `pi_d` and `m_d` be normalized target and proposal densities and define the
importance ratio `w_d=pi_d/m_d`. Whenever `pi_d << m_d`,

```text
E_m[w_d] = 1,
E_m[w_d^2] = 1 + chi_square(pi_d || m_d)
             = exp(D_2(pi_d || m_d)),                       (41)
```

where `D_2` is the order-two Renyi divergence. The common population proxy for
the relative effective sample size is

```text
ESS_fraction_infinity = 1 / E_m[w_d^2].                    (42)
```

It is a second-moment diagnostic, not an exact finite-sample ESS identity.

### Proposition 5: product mismatch produces exponential weight degeneration

Suppose

```text
pi_d(theta) = product_{j=1}^d pi_j(theta_j),
m_d(theta)  = product_{j=1}^d m_j(theta_j),                 (43)
```

with `pi_j << m_j`, and let

```text
c_j = integral pi_j(x)^2 / m_j(x) dx.
```

Then

```text
E_m[w_d^2] = product_{j=1}^d c_j.                           (44)
```

If `c_j >= 1+delta` for every `j` and some `delta>0`, then

```text
ESS_fraction_infinity <= (1+delta)^(-d).                   (45)
```

#### Proof

The product assumptions give

```text
w_d(theta)^2 m_d(theta)
  = product_j [pi_j(theta_j)^2 / m_j(theta_j)].
```

Tonelli's theorem factorizes the nonnegative integral and yields (44). Equation
(45) follows by bounding every factor below by `1+delta` and taking the
reciprocal. `QED`

This proposition is deliberately elementary. Real posteriors need not
factorize, but the result proves that positivity of a proposal density does not
prevent catastrophic second-moment growth. A tail mismatch can instead make
one or more `c_j` infinite.

### Proposition 6: weights cannot repair an unobserved region

For a measurable region `A`, let `X_1,...,X_N` be IID from a proposal `m`.
Then

```text
P(no X_i lies in A) = (1-m(A))^N.                           (46)
```

On that event, every estimator supported only on the sampled rows assigns zero
empirical mass to `A`, regardless of the values that an importance ratio would
have taken there.

#### Proof

Independence gives the probability in (46). If no sampled row belongs to `A`,
every weighted atomic measure on those rows evaluates the indicator of `A` as
zero. `QED`

Equations (44)--(46) are the precise form of the chicken-and-egg objection. If
`m` is built mostly from a transport that gives an important region
astronomically small probability, formal support does not create usable
evidence about that region.

## 16. An ensemble is a mixture of pushforward laws, not an averaged map

Let `K>=1`. For each component `i`, let `T_i:R^d->R^d` be a `C^1`
diffeomorphism and define

```text
q_i(theta)
  = rho(T_i^{-1}(theta)) abs(det D T_i^{-1}(theta)).         (47)
```

Let `alpha_i>0` with `sum_i alpha_i=1`.

### Proposition 7: categorical selection gives the mixture density

Draw `I ~ Categorical(alpha_1,...,alpha_K)`, independently draw `Z~rho`, and
set `Theta=T_I(Z)`. Then

```text
q_alpha(theta) = sum_{i=1}^K alpha_i q_i(theta).             (48)
```

#### Proof

For every measurable `A`, condition on `I` and use the pushforward definition:

```text
P(Theta in A)
  = sum_i alpha_i P(T_i(Z) in A)
  = sum_i alpha_i integral_A q_i(theta) dtheta.
```

This identifies the density in (48). `QED`

### Proposition 8: arithmetic averaging of maps is not mixture sampling

In one dimension, take `T_1(z)=z-a`, `T_2(z)=z+a`, and equal weights. The
arithmetic average is `T_bar(z)=z`, whose pushforward is `N(0,1)`. The mixture
in (48) is

```text
0.5 N(-a,1) + 0.5 N(a,1),                                  (49)
```

which differs from `N(0,1)` for every `a != 0`. Moreover, an arithmetic average
of diffeomorphisms need not be invertible: averaging `T_1(z)=z` and
`T_2(z)=-z` gives the constant zero map.

#### Proof

The first claim follows from the means and variances: (49) has variance
`1+a^2`, whereas `T_bar(Z)` has variance one. The second claim is immediate
from the displayed maps. `QED`

The proposed ensemble therefore retains a discrete chart index. It is not a
single NeuTra bijector and must not be passed to an ordinary single-chart HMC
implementation as though it were one.

## 17. Mixture reverse KL still uses IID Gaussian base draws

For a proper target `pi_beta=tilde_pi_beta/Z_beta`, define

```text
R_beta(alpha,T_1:K) = KL(q_alpha || pi_beta).                (50)
```

### Proposition 9: Gaussian expectation for mixture reverse KL

Under the density and integrability assumptions above,

```text
R_beta(alpha,T_1:K)
  = sum_i alpha_i E_{Z~rho} [
      log q_alpha(T_i(Z)) - log tilde_pi_beta(T_i(Z))
    ] + log Z_beta.                                         (51)
```

Consequently the trainable part of (51) can be estimated using only IID draws
from `rho`, evaluations of every component density at transported points, and
evaluations of the unnormalized target. It requires no samples from `pi_beta`
and no particle approximation to `pi_beta`.

#### Proof

Expand the KL integral. Split its expectation under the mixture using (48),
then apply the change of variables `theta=T_i(z)` separately to each component.
Replacing `log pi_beta` by `log tilde_pi_beta-log Z_beta` gives (51). `QED`

If differentiation may be interchanged with the finite sum and Gaussian
expectations, enumerating the component index and drawing fresh Gaussian
batches gives an unbiased stochastic gradient of the displayed trainable
objective. A sampled categorical index is unnecessary for training because the
finite component sum can be evaluated directly.

## 18. What reverse-KL mixture weights do and do not estimate

The following idealized calculation isolates the mode-mass question. Let
`A_1,...,A_K` be a measurable partition with `p_i=pi(A_i)>0`, and define the
conditional target `pi_i=pi 1_{A_i}/p_i`. Suppose `q_i` is supported in `A_i`
and set

```text
delta_i = KL(q_i || pi_i).                                  (52)
```

This separated-support model is a mathematical limit; a full-support neural
flow generally only approximates it.

### Proposition 10: exact decomposition of separated mixture reverse KL

Under the separated-support assumptions,

```text
KL(q_alpha || pi)
  = sum_i alpha_i [log(alpha_i/p_i) + delta_i].              (53)
```

For fixed components, the unique minimizing weights are

```text
alpha_i_star
  = p_i exp(-delta_i) / sum_j p_j exp(-delta_j).             (54)
```

Thus `alpha_star=p` if every component exactly equals its regional target, but
an imperfect component is downweighted according to its local reverse-KL
error.

#### Proof

On `A_i`, `q_alpha=alpha_i q_i` and `pi=p_i pi_i`. Splitting the KL integral by
the partition gives (53). Introduce a Lagrange multiplier for
`sum_i alpha_i=1`. The stationary equation is

```text
log(alpha_i/p_i) + delta_i + 1 + lambda = 0.
```

Normalizing its solution gives (54). Strict convexity in the positive simplex
gives uniqueness. `QED`

This proposition corrects a tempting overclaim. Joint mixture reverse-KL can
recover regional masses in the exact separated-component limit. With unequal
component errors, its weights confound target mass and approximation quality.
They are variational mixture weights, not independently certified posterior
mode probabilities.

## 19. What multiple random starts can prove

### Proposition 11: conditional discovery probability

Suppose independent training initializations reach the basin associated with
mode `j` with probability `r_j`. With `K` independent runs, the probability of
missing mode `j` is

```text
(1-r_j)^K.                                                   (55)
```

For a finite declared collection of modes `1,...,J`, the probability of missing
at least one is at most

```text
sum_{j=1}^J (1-r_j)^K.                                      (56)
```

#### Proof

Equation (55) is independence. Equation (56) is the union bound. `QED`

This is not a discovery guarantee because the `r_j` are unknown and can be zero.
Changing only neural-network weight seeds may leave every transport near the
same initial physical region. A valid experiment must diversify initial affine
locations and scales as explicit hypotheses and preserve all lineages until a
held-out decision stage.

### Proposition 12: finite target queries cannot certify global mode completeness

Let an algorithm query an unnormalized smooth positive target and any finite
number of its derivatives at a finite set `S` in `R^d`. There exists another
smooth positive integrable target that agrees with every queried value and
derivative but assigns arbitrarily large additional unnormalized mass to a
region disjoint from `S`.

#### Proof

Choose an open ball `B` whose closure is disjoint from the finite set `S`, and
choose a nonzero nonnegative smooth bump `h` with compact support in `B`. For an
original target `tilde_pi_0`, define

```text
tilde_pi_c = tilde_pi_0 + c h,       c>0.                    (57)
```

The bump and all its derivatives vanish in a neighborhood of every point in
`S`, so all queried information agrees. Both targets are positive and
integrable. The added unnormalized mass is `c integral h`, which can be made
arbitrarily large. `QED`

Consequently neither a finite transport ensemble nor replica exchange can prove
the absence of every undiscovered mode without additional structural
assumptions. The research goal is evidence of exploration under declared model
structure, not a universal completeness certificate.

## 20. Tempered reverse-KL continuation

Let `g_0` be a normalized positive reference density and define the geometric
bridge

```text
tilde_pi_beta(theta)
  = g_0(theta)^(1-beta) tilde_pi(theta)^beta,
0 <= beta <= 1.                                              (58)
```

Assume every `Z_beta=integral tilde_pi_beta` is finite and positive. For a
posterior `tilde_pi(theta)=p(theta)L(y|theta)` with proper prior `p`, choosing
`g_0=p` gives the likelihood-tempered path

```text
tilde_pi_beta(theta) = p(theta) L(y|theta)^beta.             (59)
```

This proper-reference construction is required. A uniform `beta=0` endpoint on
`R^d` is not a probability distribution.

### Proposition 13: the bridge scales only the declared energy difference

Write `U_0=-log g_0` and `U_1=-log tilde_pi`. Up to an additive constant,

```text
U_beta(theta)=(1-beta)U_0(theta)+beta U_1(theta).             (60)
```

For two points `a,b`,

```text
U_beta(b)-U_beta(a)
  = (1-beta)[U_0(b)-U_0(a)]
    + beta[U_1(b)-U_1(a)].                                  (61)
```

For (59), the likelihood part of the difference is multiplied by `beta` while
the prior part is unchanged.

#### Proof

Take the negative logarithm of (58) and subtract its values at `a` and `b`.
The posterior factorization gives the final statement. `QED`

Tempering can reduce a likelihood-created barrier, but (61) also shows why no
monotone connectivity claim follows: the reference energy, mode volumes, and
relative regional masses change along the path.

### Proposition 14: continuation does not alter the final reverse-KL target

If the last temperature is `beta_L=1`, the objective in (50) at the last stage
is exactly `KL(q_alpha||pi)`. Earlier temperatures and warm starts alter the
optimization trajectory but not the final population objective.

#### Proof

Substituting `beta=1` in (58) gives `tilde_pi_1=tilde_pi`, hence
`pi_1=pi`. Apply definition (50). `QED`

This proposition licenses temperature continuation as an optimization and
discovery mechanism. It does not say that continuation finds the global
minimum or that different lineages occupy different modes.

## 21. Each frozen transport defines an exact coordinate-chart kernel

Fix a temperature `beta` and a transport `T_i`. Define its exact pullback

```text
pi_beta_i^z(z)
  = pi_beta(T_i(z)) abs(det D T_i(z)).                       (62)
```

Let `P_beta_i` be any Markov kernel preserving (62), such as a fixed,
Metropolis-corrected HMC kernel evaluated with the exact transformed target.
Define the physical-coordinate kernel

```text
K_beta_i(theta,A)
  = P_beta_i(T_i^{-1}(theta), T_i^{-1}(A)).                  (63)
```

### Proposition 15: chart pushforward preserves the physical target

The kernel `K_beta_i` preserves `pi_beta`.

#### Proof

Under `theta=T_i(z)`, the distribution `pi_beta(dtheta)` becomes
`pi_beta_i^z(dz)` by (62). Therefore

```text
integral pi_beta(dtheta) K_beta_i(theta,A)
 = integral pi_beta_i^z(dz) P_beta_i(z,T_i^{-1}(A))
 = pi_beta_i^z(T_i^{-1}(A))
 = pi_beta(A).
```

The middle equality is invariance of `P_beta_i`. `QED`

Training quality affects the efficiency of this kernel, not its invariant
target, provided the transport is frozen and the target, score, Jacobian, HMC
integration, and Metropolis correction are implemented correctly.

### Proposition 16: a fixed mixture of chart kernels is exact

Let `gamma_i>=0`, `sum_i gamma_i=1`, and assume the `gamma_i` are fixed and do
not depend on the current state. Then

```text
K_beta = sum_i gamma_i K_beta_i                             (64)
```

preserves `pi_beta`.

#### Proof

Linearity gives

```text
pi_beta K_beta
  = sum_i gamma_i (pi_beta K_beta_i)
  = sum_i gamma_i pi_beta
  = pi_beta.
```

`QED`

State-dependent chart weights are not automatically valid. For a two-state
uniform target, both the identity kernel and the flip kernel are invariant. If
the identity is selected at state zero and the flip at state one, both states
move to zero, so the state-dependent mixture is not invariant. Any adaptive or
state-dependent chart selector therefore requires its own correction proof.

## 22. Replica exchange with transport ensembles

Let `0<=beta_0<...<beta_L=1` and define the product target

```text
Pi(theta_0:L) = product_{ell=0}^L pi_beta_ell(theta_ell).     (65)
```

At temperature `ell`, use a fixed mixture of exact chart kernels as in (64).
For adjacent temperatures, propose exchanging `theta_ell` and
`theta_{ell+1}`.

### Proposition 17: the adjacent swap ratio satisfies detailed balance

For a symmetric adjacent-pair proposal, accept the exchange with probability

```text
a_swap = min(1,
  [tilde_pi_beta_ell(theta_{ell+1})
   tilde_pi_beta_{ell+1}(theta_ell)]
  /
  [tilde_pi_beta_ell(theta_ell)
   tilde_pi_beta_{ell+1}(theta_{ell+1})]).                  (66)
```

The swap kernel is reversible with respect to `Pi`.

#### Proof

All unaffected product factors cancel in the ratio of (65) at the swapped and
current states. The two unknown normalizing constants also cancel. Equation
(66) is therefore the ordinary Metropolis ratio for a symmetric involutive
proposal, which gives detailed balance. `QED`

### Theorem 4: tempered multi-transport HMC has the exact cold marginal

Suppose every within-temperature chart kernel satisfies Proposition 15, every
chart mixture has fixed state-independent weights as in Proposition 16, and
every exchange uses (66). Any fixed composition or random scan of these
within-temperature and exchange kernels preserves `Pi`. Consequently the
`beta_L=1` marginal is `pi`.

#### Proof

The product of the within-temperature mixtures preserves (65) because each
factor kernel preserves its corresponding factor. Proposition 17 shows that
every exchange kernel preserves the same product target. A composition or
fixed state-independent mixture of kernels sharing an invariant law preserves
that law. Finally, the last factor of (65) is `pi_beta_L=pi`. `QED`

The theorem removes the particle-measure circularity from correctness. It does
not prove irreducibility, useful swap rates, temperature round trips, hot-chain
basin forgetting, or cold-chain convergence.

### Proposition 18: invariance alone does not imply discovery

The identity kernel preserves every target but never changes state. More
generally, if all within-temperature kernels preserve a common region `A` and
every replica is initialized in `A`, replica exchange only permutes states in
`A`; no replica reaches `A^c`.

#### Proof

The identity statement is immediate. Under the second hypothesis, within-
temperature updates keep every state in `A`, and a swap changes only which
temperature owns each existing state. Induction over transitions proves that
all states remain in `A`. `QED`

This is why observed swaps, acceptance, or finite values are explanatory
diagnostics. Claim-bearing evidence requires replica-identity round trips,
hot-level basin forgetting, repeated cold-level transitions, initialization
forgetting, modern convergence diagnostics, and target-relevant downstream
agreement.

## 23. Optional mixture proposals after training

The learned mixture can also be used as an independence proposal, but this is
an optional global-move kernel rather than the foundation of the method.

### Proposition 19: Metropolis correction makes a mixture proposal exact

Assume `q_alpha(theta)>0` wherever `pi(theta)>0`. From current state `x`, draw
`y~q_alpha` and accept with probability

```text
a_ind(x,y) = min(1,
  [tilde_pi(y) q_alpha(x)] / [tilde_pi(x) q_alpha(y)]).       (67)
```

The resulting independence kernel is reversible with respect to `pi`.

#### Proof

For `x != y`, the accepted flow from `x` to `y` is

```text
pi(x) q_alpha(y) min(1,
  [pi(y)q_alpha(x)]/[pi(x)q_alpha(y)])
 = min(pi(x)q_alpha(y), pi(y)q_alpha(x)),
```

which is symmetric in `x,y`. The rejection mass completes detailed balance.
`QED`

This kernel uses exact density ratios rather than an importance estimate, but
its acceptance can still collapse in high dimension if `q_alpha` is a poor
global approximation. It must be compared with, not substituted for, the
multi-chart replica-exchange construction.

## 24. Corrected algorithmic proposal

### Algorithm C: tempered ensemble reverse-KL training

1. Define and test a proper bridge (58). Prefer the exact prior-likelihood path
   (59) when the target exposes that decomposition. Do not use an improper
   uniform endpoint.
2. Predeclare a temperature ladder for a bounded pilot. Treat its size and
   spacing as hypotheses until temperature-overlap evidence is available.
3. At the first level, initialize multiple transports with distinct stateless
   neural seeds and deliberately diversified affine locations and scales drawn
   from or constructed under the proper reference `g_0`. Random neural weights
   alone are not a diversity contract.
4. Train every component with the original reverse-KL Gaussian objective. At
   later levels, warm-start each lineage from the preceding temperature and
   apply predeclared independent perturbations or branching. Preserve all
   lineages; do not erase a component merely because another has lower
   descriptive loss.
5. Optionally refine all components and positive mixture weights using (51).
   Report the `alpha_i` as variational weights and audit the approximation-error
   confounding in (54).
6. Freeze transports, mixture weights, component identities, temperature path,
   target signatures, and selection rules before sampler validation.

Fully optimizing every component at `beta=0` makes all component distributions
target the same reference law and can erase the initial distributional
diversity. The endpoint is therefore a bridge and implementation check, not the
sole source of distinct lineages. The bounded design must compare pure
continuation with predeclared fresh restarts or branching at one or more
positive temperatures. These mechanisms change initialization paths, not the
reverse-KL objective.

### Algorithm D: exact tempered multi-chart HMC

1. For each temperature and retained component, construct the exact pullback
   target (62), including the temperature-specific target value and score and
   the transport log determinant.
2. Tune a fixed HMC kernel within each declared temperature/chart scope using
   disjoint tuning data. Warmup draws are never posterior draws.
3. At each within-temperature update, select a chart with fixed
   state-independent `gamma`; transform the current physical state through its
   inverse, run the corrected HMC transition in that chart, and map back.
4. Apply alternating adjacent exchanges using (66), retaining replica identity
   and complete swap telemetry.
5. Use only the `beta=1` retained draws for posterior inference. Require the
   repository sequential-HMC diagnostics plus replica round trips, hot-level
   basin forgetting, cold-level mode transitions, and downstream reference
   agreement.
6. Test (67) only as an optional separately labeled global proposal arm.

The two algorithms use IID Gaussian base draws for learned-transport training
and exact target evaluations for sampling. They do not require a particle
measure drawn from the unknown posterior.

The current public fixed-transport tuner binds one frozen transport and one
transformed target per artifact. Tuning every `(beta,i)` scope separately does
not by itself implement the multi-chart sequential controller in Algorithm D.
That controller, its canonical NeuTra-HMC route-ledger classification, and its
consumption of exact per-scope tuning artifacts are new implementation work.

## 25. Revised evidence boundary

The new candidate answers a narrower question than "solve multimodality in high
dimension." It asks whether a diversified temperature-continuation ensemble
creates useful complementary coordinate charts, and whether exact replica
exchange using those charts explores the declared SSL-LSTM target better than
the single-transport and physical-coordinate baselines under a common budget.

The following roles are fixed before implementation:

| Quantity | Evidentiary role |
|---|---|
| Exact density, inverse, Jacobian, score, HMC reversibility, and swap-ratio fixtures | hard implementation veto |
| Target/status finiteness and GPU/XLA/batch policy | hard execution veto |
| Single cold reverse-KL transport | required baseline |
| Physical-coordinate replica exchange with matched tempering target | required classical comparator |
| Cold multi-start ensemble without tempering | required ablation |
| Tempered ensemble without joint mixture refinement | plain proposed method |
| Tempered ensemble with joint mixture refinement | enhanced proposed method |
| Reverse-KL loss, latent mean/covariance, component separation, acceptance, and swap rate | explanatory or nomination only |
| Replica round trips, hot basin forgetting, cold retained R-hat/ESS, mode transitions, and reference/downstream agreement | sampler promotion criteria or vetoes as predeclared |
| Mixture weights `alpha` | variational quantities; not posterior mode-mass authority without the conditions of Proposition 10 |
| Fixed chart-selection weights `gamma` | algorithmic frequencies affecting efficiency, not posterior masses |

No finite pass establishes exhaustive mode discovery, universal high-
dimensional scaling, statistical superiority, or correctness of the underlying
UKF-defined approximation to the nonlinear state-space posterior. A future
dimension ladder must report how the required number of components,
temperatures, target evaluations, and wall time scale; q=20 success cannot by
itself promote the method for larger SSL-LSTM parameter spaces.

## Independent-audit status

The original bounded MathDevMCP audit is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathdevmcp-audit-2026-08-21.md`.
It covers the 2026-08-21 replay analysis. It certified the scalar
deterministic-mixture cancellation but did not certify
the measure-theoretic or stochastic-approximation arguments. A thorough
read-only independent review was completed in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-review-reply-2026-08-23.md`;
the resulting adjudication and plan amendment are in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-review-adjudication-plan-2026-08-23.md`.
The original request is preserved in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-handoff-2026-08-21.md`.

The 2026-08-28 correction and tempered-ensemble propositions have a separate
LaTeX audit surface and MathDevMCP record:

- `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.tex`;
- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-mathdevmcp-audit-2026-08-28.md`.

The implementation proposal and its document-alignment audit are separate from
the mathematical audit. A Claude handoff requests an independent review of the
revised note and plan; until that reply exists, the new route is a reviewed-by-
Codex candidate, not an independently accepted design.

## Sources and implementation anchors

- Hoffman et al., "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using
  Neural Transport," Section 2.2, equations (2)--(3), and Section 2.3,
  arXiv:1903.03704; local text at
  `.localresources/papers/multimodal_hmc/hoffman-sountsov-dillon-2019-neutra.txt`.
- Hukushima and Nemoto, "Exchange Monte Carlo Method and Application to Spin
  Glass Simulations," Section II, equations (2.1)--(2.7),
  arXiv:cond-mat/9512035; local PDF and inspection record in
  `.localresources/papers/multimodal_hmc/` and `CORPUS_AUDIT.md`.
- Parno and Marzouk, "Transport Map Accelerated Markov Chain Monte Carlo,"
  Section 3.1, especially equation (21), arXiv:1412.5492; local text at
  `.localresources/papers/multimodal_hmc/parno-marzouk-2018-transport-map-mcmc.txt`.
- The inspected multimodal-HMC source synthesis, including the replica-exchange
  product target, swap correction, and limitations:
  `docs/surveys/multimodal_hmc_survey.tex` and
  `.localresources/papers/multimodal_hmc/CORPUS_AUDIT.md`.
- Del Moral and Doucet, "Sequential Monte Carlo Samplers" preprint; local text
  at
  `.localresources/papers/multimodal_hmc/del-moral-doucet-2002-smc-samplers-preprint.txt`.
- Current weighted trainer:
  `bayesfilter/inference/neutra_weighted_training.py`.
- Current q=20 fixed-replay runner:
  `docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py`.
- Current q=20 batch-native likelihood-plus-Gaussian-prior target:
  `bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py`.
- Current single-map HMC interface and canonical sequential controller:
  `docs/reference/hmc-tuning-interface.md` and
  `bayesfilter/inference/neutra_hmc.py`.
- Diagnostic physical-coordinate pure-power replica exchange, which is not the
  proposed proper-reference multi-chart implementation:
  `bayesfilter/testing/distributed_replica_exchange_tf.py`.
- Governing plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematics-review-plan-2026-08-21.md`.
- MathDevMCP audit:
  `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathdevmcp-audit-2026-08-21.md`.
- Fable review handoff:
  `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-handoff-2026-08-21.md`.
- Review adjudication and plan amendment:
  `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-review-adjudication-plan-2026-08-23.md`.
- Corrected implementation plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`.
