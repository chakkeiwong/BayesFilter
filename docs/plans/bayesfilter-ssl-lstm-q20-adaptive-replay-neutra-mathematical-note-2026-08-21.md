# Adaptive replay for SSL-LSTM q=20 NeuTra: mathematics, proofs, and limits

Date: 2026-08-21
Status: `CONDITIONAL_MATHEMATICAL_VIABILITY_ESTABLISHED_EMPIRICAL_SUCCESS_UNPROVED`

## Verdict

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

## 14. Direct answer to "will it work?"

Theorems 1, 2, 2A, and 3 establish that refreshed replay **can** work:

1. valid fresh particle blocks can estimate the exact forward gradient;
2. a fixed-capacity content-independently refreshed buffer can converge under
   explicit stochastic-approximation assumptions;
3. an evolving proposal can generate a new proof-bearing block every update
   while stale replay is retained with summable influence;
4. fresh reverse-KL queries estimate the original NeuTra objective;
5. the hybrid has the correct exact solution whenever that solution is in the
   flow family; and
6. at that solution the target pullback is exactly Gaussian.

What remains unproved for SSL-LSTM q=20 is whether the active target is globally
regular, whether a finite dense IAF has adequate capacity, whether a
proof-bearing proposal/SMC block generator has acceptable variance and the
required envelopes, whether the nonconvex optimizer reaches a symmetry-free
useful basin, and whether the resulting finite HMC kernel crosses the sign
barrier. The strong-stability condition (30) cannot be applied globally to
the current symmetric parameterization. These are empirical research
questions and must remain gates rather than conclusions.

## Independent-audit status

The bounded MathDevMCP audit is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathdevmcp-audit-2026-08-21.md`.
It certified the scalar deterministic-mixture cancellation but did not certify
the measure-theoretic or stochastic-approximation arguments. A thorough
read-only independent review was completed in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-review-reply-2026-08-23.md`;
the resulting adjudication and plan amendment are in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-review-adjudication-plan-2026-08-23.md`.
The original request is preserved in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-handoff-2026-08-21.md`.

## Sources and implementation anchors

- Hoffman et al., "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using
  Neural Transport," arXiv:1903.03704; local text at
  `.localresources/papers/multimodal_hmc/hoffman-sountsov-dillon-2019-neutra.txt`.
- Del Moral and Doucet, "Sequential Monte Carlo Samplers" preprint; local text
  at
  `.localresources/papers/multimodal_hmc/del-moral-doucet-2002-smc-samplers-preprint.txt`.
- Current weighted trainer:
  `bayesfilter/inference/neutra_weighted_training.py`.
- Current q=20 fixed-replay runner:
  `docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py`.
- Governing plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematics-review-plan-2026-08-21.md`.
- MathDevMCP audit:
  `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathdevmcp-audit-2026-08-21.md`.
- Fable review handoff:
  `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-handoff-2026-08-21.md`.
- Review adjudication and plan amendment:
  `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-review-adjudication-plan-2026-08-23.md`.
