# Zhao-Cui High-Dimensional Frozen-Proposal APF Review Result

Date: 2026-07-22

Decision: `REVISE`

The frozen-proposal auxiliary particle filter (APF) is a viable
`extension_or_invention` architecture, but the handoff recommendation is not
ready to adopt as written. The finite value program and its fixed-branch score
can be made exact for a declared proposal branch. The proposal is not
source-faithful Zhao-Cui, a state-independent proposal does not solve the
high-dimensional particle-degeneracy problem by itself, and the simple score
recursion does not cover parameter-dependent or singular structural state
maps. The smallest justified implementation is therefore a deterministic
fixed-randomness APF kernel with an analytical score, followed by an offline
Zhao-Cui squared-TT proposal compiler and fresh ESS/rank validation.

This result does not claim a NAWM adapter, NAWM readiness, posterior
correctness, HMC convergence, source-faithful Zhao-Cui status, default
promotion, or superiority over direct fixed-TTSIRT.

## 1. Scope And Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can a high-dimensional proposal reduce online memory while retaining a target-consistent finite filtering value and a matching analytical score? |
| Candidate | Frozen full-support proposal (eventually compiled from squared-TT/KR geometry) plus a fixed-branch APF importance program. |
| Comparator A | Direct fixed-TTSIRT retained-density recursion with paired-core marginalization and a complete retained-object derivative. |
| Comparator B | Fixed-randomness bootstrap/SIS program with the same value/score contract. |
| Comparator C | Current GenUT/Contract-E dense and streaming OT routes, with dense OT treated as a memory/work warning rather than a mandatory baseline. |
| Primary promotion criterion | The implementation returns one explicitly defined finite scalar and an analytical derivative of that same scalar; value and score pass an exact-oracle or independent directional check. |
| Veto diagnostics | Non-finite values, invalid measure/support, omitted proposal or ancestor terms, branch changes during evaluation, singular-transition misuse, unbounded TT rank, or collapse on the declared fresh validation partition. |
| Explanatory diagnostics | ESS, weight entropy, log-weight spread, TT heldout density loss, rank, compile time, warmed time, allocator bytes, and score variance. These do not prove correctness or convergence. |
| Continuation veto | The finite scalar cannot be defined on the model support, or no exact derivative of that scalar exists under a reviewed fixed-branch formulation. Candidate failure alone is not a research-direction veto. |
| Nonclaims | No exact posterior or pseudo-marginal claim is made by the first implementation slice; no NAWM dimensions beyond checked observables/shocks are inferred. |

The skeptical audit passed for executing this review because the baseline
ladder includes direct TTSIRT, the proposal is labeled as an extension, score
dependence and singular measures are explicit gates, and memory/work are
separate criteria. It did not pass for algorithm promotion because particle
degeneracy, proposal refresh, and the complete derivative for structural
models remain unresolved.

## 2. Source Audit And Classification

### Zhao-Cui paper

The local full text is
`.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt`
and the local PDF is
`.local_sources/highdim_nonlinear_filtering/zhao_cui_tt_sequential_learning_jmlr_23-0743.pdf`.
The inspected technical anchors are:

| Anchor | Checked content | Classification for this work |
| --- | --- | --- |
| Eqs. (1)-(4), lines 47-104 | State-space transition/observation densities and posterior/evidence definition. | `source_faithful` model contract. |
| Eqs. (9)-(11), lines 339-366 | Adjacent-target recursion and retained previous marginal. | `source_faithful` target construction. |
| TT decomposition, lines 391-455 | Core shapes, integration by core contraction, and stated `O(m l r^2)`/`O(m l r^3)` costs. | `source_faithful` TT storage/marginalization accounting. |
| Eq. (13), lines 539-573 | Nonnegative squared-TT density with defensive reference mass and support domination. | `source_faithful` proposal-support mechanism. |
| Proposition 2, lines 592-626 | Squared-TT marginal through accumulated mass matrices/Cholesky factors. | `source_faithful` offline marginalization. |
| KR construction, lines 627-670 | Conditional densities, triangular map, and inverse/evaluation costs. | `source_faithful` offline map operations. |
| Eq. (15)-(16), lines 693-719 | Actual adjacent target is fitted, then marginalized. | `source_faithful` proposal training signal. |
| Proposition 4, lines 771-804 | Conditional KR inverse has the desired conditional density. | `source_faithful` conditional sampling identity. |
| Eq. (20)-(23), lines 807-887 | Upper conditional map, forward samples, and target/proposal correction. | `source_faithful` correction idea; not the frozen-HMC extension. |
| Algorithm 3, lines 890-923 | Per-particle conditional map and weight update. | `source_faithful` mechanics, with fixed-branch adaptation required. |
| Section 4/Appendices A-B | Approximation/error discussion and linear-Gaussian appendix. | Error context only; no bounded-rank NAWM guarantee. |

The paper explicitly states that maximal TT rank can depend on dimension and
can become large (lines 426-437). It therefore does not justify treating `r`
as a universal constant.

### Pinned author code

The audit snapshot is
`third_party/audit/zhao_cui_tensor_ssm_p10/source`, whose manifest binds
upstream commit `80034dccb99eb1d86284a1839b4a12067d13b9da`. The working tree
itself is an Octave-compatible snapshot at a later local commit; the manifest
and `octave_compatibility.patch` are preserved, so no byte-identical-upstream
claim is made.

| Source anchor | Observed operation | Classification |
| --- | --- | --- |
| `models/full_sol.m:21-43` | Push samples, compute ESS, reapproximate, sample through `eval_irt`, and correct with `exp(-fun_post)/eval_pdf`. | `source_faithful` proposal-correction mechanics. |
| `models/full_sol.m:46-130` | Weighted recenter/Cholesky, previous retained marginal, actual target construction, squared-TT fit, normalizer update. | `source_faithful` retained-object recursion. |
| `models/computeL.m:24-47` | Weight normalization, weighted center/covariance, Cholesky/scale. | `source_faithful` geometry; not a parameter score. |
| `eg3_sir/mainscript.m:14-22` | Austria SIR has `m=18`, `T=20`, no parameter block in this example. | Source-derived fixture dimensions only. |
| `eg3_sir/mainscript.m:39-55` | `N=5000`, Lagrange basis, max rank 40, squared route. | Historical author configuration, not a BayesFilter default. |
| `deep-tensor.dev/src/SIRT.m:1-48` | SIRT/TTSIRT map, marginal, density, and Jacobian API. | `source_faithful` map surface. |
| `@TTSIRT/marginalise.m` | Right/left core contractions and marginal object creation. | `source_faithful` marginalization. |
| `@TTSIRT/eval_rt_jac_reference.m` and `eval_irt_reference.m` | Reference-to-state map and Jacobian/inverse operations. | `source_faithful` map operations; no local HMC derivative closure. |

The author code adapts/rebuilds objects during `solve`; it does not establish
that those adaptive choices are differentiable through an HMC trajectory.

### Local evidence

The local records agree with this review:

- `docs/plans/bayesfilter-highdim-zhao-cui-p83-phase2-transport-marginalization-design-result-2026-06-22.md` preserves target/proposal correction and blocks base-density or tensor-grid substitutes.
- `docs/plans/bayesfilter-highdim-zhao-cui-p83-phase4-analytical-derivative-audit-result-2026-06-22.md` blocks source-route analytical derivative readiness because previous-marginal and transport derivative ownership is absent.
- `docs/plans/bayesfilter-highdim-zhao-cui-p91-phase9-final-decision-result-2026-06-29.md` promotes only a local complete-data component and explicitly leaves the full filtering score unclaimed.
- `docs/plans/bayesfilter-highdim-zhao-cui-p76-phase6-bounded-minibatch-pilot-result-2026-06-18.md` and the P76 phase-10 result establish finite target/metric plumbing for a one-step 36-dimensional pilot, not a filtering likelihood or HMC score.
- `bayesfilter/highdim/cubature_genut_filter.py` and the GenUT math/code audit support the dense `N^2 d_x p_b` memory warning.

## 3. Corrected Algorithm

### 3.1 Frozen branch

At an offline reference stage, construct a branch object `B` containing:

1. initial states `x0[i]` and normalized proposal density values `q0[i]`;
2. for each `t >= 1`, fixed ancestor indices `A[t,i]`, fixed positive auxiliary probabilities `a[t,j]`, current states `x[t,i]`, and proposal density values `q[t,i] = q_t(x[t,i] | x[t-1,A[t,i]], y_t)`;
3. all seeds, coordinate order, TT cores/ranks/bases, defensive mass, map settings, and invalid-state policy;
4. a common reference measure on which every model density and proposal density is defined.

For the first implementation, `B` is parameter-independent. A proposal may
be generated from a Zhao-Cui squared-TT/KR fit at a reference parameter, but
the realized states, genealogy, `q`, and `a` are constants during value/score
evaluation. A positive defensive component is required on the stochastic
support. A caller cannot stamp a proposal as source-faithful; the compiler
must issue its route identity from the actual callable and settings.

This is a fixed-HMC adaptation of the source operations, not the author's
adaptive sequential solver.

### 3.2 Exact finite scalar

Let `ell0_i(theta)` be

`log p_theta(x0_i) + log g_theta(y0 | x0_i) - log q0_i`.

Set

`c0 = logsumexp_i ell0_i - log N`,

`W0_i = exp(ell0_i - logsumexp_j ell0_j)`, and

`G0_i = h0_i - S0`, where
`h0_i = grad_theta[log p_theta(x0_i) + log g_theta(y0|x0_i)]` and
`S0 = sum_i W0_i h0_i`.

For `t >= 1`, define

`v_ti(theta) = W_{t-1,A_ti}(theta) * p_theta(x_ti|x_{t-1,A_ti}) * g_theta(y_t|x_ti) / (a_{t,A_ti} q_ti)`.

The step increment and normalized weights are

`c_t = logsumexp_i log v_ti - log N`,

`W_ti = exp(log v_ti - logsumexp_j log v_tj)`.

The reported finite likelihood estimator is

`log L_B(theta; u) = sum_{t=0}^{T-1} c_t`.

The `1/N` factor is part of the definition at every step. Omitting it changes
the scalar by a parameter-independent constant and still breaks reproducible
normalization/accounting, so it is not left implicit.

### 3.3 Analytical fixed-branch score

Because `B` is constant in theta, `log a`, `log q`, states, and ancestors have
zero derivative. Let `s_f` and `s_g` be the model-provided parameter scores.

`h_ti = G_{t-1,A_ti} + s_f(theta, x_{t-1,A_ti}, x_ti) + s_g(theta, x_ti, y_t)`.

Then

`S_t = sum_i W_ti h_ti`,

`G_ti = h_ti - S_t`,

and

`grad_theta log L_B = sum_t S_t`.

The implementation must compute this recursion directly, not use an admitted
finite-difference or autodiff score. Finite differences remain an external
diagnostic of the same scalar.

### 3.4 Parameter-dependent proposal repair

If states are reconstructed as `x_t = T_theta(u_t, x_{t-1}, y_t)`, the above
formula is incomplete. The admitted alternative must carry state tangents
`D_t = d x_t/d theta` (or an exactly equivalent forward/reverse adjoint) and
include total derivatives of `log f`, `log g`, `log q`, `log a`, and every map
Jacobian. A partial derivative through only the explicit model parameters is
wrong relative to the total derivative of that finite program. The first
implementation therefore rejects a parameter-dependent proposal rather than
silently using the simple recursion.

### 3.5 Singular structural dynamics

For deterministic/algebraic structural coordinates, an ordinary Lebesgue
density `f_theta(x_t|x_{t-1})` may not exist. The supported repair is an
innovation-coordinate representation:

`z_t ~ rho_t(z_t|x_{t-1},theta)`,

`x_t = F_t(x_{t-1}, z_t, theta)`,

with the proposal defined on `z_t` and a declared mixed measure for any
deterministic completion. The value must use the innovation density and the
Jacobian only on the stochastic subspace. Until an adapter supplies this
contract, the APF fails closed for a singular transition; it must not assign a
Lebesgue log density to a lower-dimensional manifold.

### 3.6 HMC and pseudo-marginal semantics

With one permanently fixed branch `u`, the target is the deterministic
approximate posterior proportional to `p(theta) exp(log L_B(theta;u))`. This
is not the exact posterior and is the only target claimed by the first kernel.

If the exact pseudo-marginal target is desired, `u` (including uniforms,
Gaussian bases, and genealogy randomness) must be part of the Markov state and
updated with a valid invariant kernel. It may be held fixed during a
conditional HMC trajectory, but it cannot be held fixed forever. A future
pseudo-marginal/particle-HMC implementation needs an extended-target proof;
the current implementation does not make that claim.

## 4. Proof Obligations And Results

### Proposition 1: finite scalar and proposal measure

For a fixed branch `B`, define the proposal path measure as the product of
`q0(x0)` and, at each step, `a_t(A_t) q_t(x_t|x_{t-1,A_t},y_t)`. The ratio for a
selected particle is

`p_theta(x0) g_theta(y0|x0)/q0(x0)` initially and

`W_{t-1,A} p_theta(x_t|x_{t-1,A}) g_theta(y_t|x_t)/(a_t(A) q_t(x_t|...))`

recursively. The displayed `c_t` is the log of the arithmetic mean of these
nonnegative ratios. Thus `log L_B` is a fully specified scalar of the branch,
not an unspecified "PF likelihood".

*Proof.* Divide the target path factor by the branch proposal factor at each
selected ancestor. The previous normalized empirical measure contributes
`W_{t-1,A}`; the proposal contributes `a_t(A)q_t`. Taking the arithmetic mean
over the `N` fixed particles gives `exp(c_t)`. Multiplication over time gives
the stated scalar. No unlisted base-density or Jacobian term is permitted.

### Proposition 2: conditional importance identity

If `q0 > 0` wherever the initial target is positive and `a_t(j)q_t > 0`
wherever the corresponding transition/observation target is positive, each
ratio is the Radon-Nikodym derivative of the target factor with respect to the
branch proposal factor on the declared measure.

*Proof.* The ratio is a product of the previous empirical mass and the local
target density divided by the selected ancestor/proposal density. Positivity
of the defensive proposal gives absolute continuity. The identity is
measure-specific; it is invalid if a singular transition is represented as an
ordinary Lebesgue density.

### Proposition 3: unbiasedness scope

The usual nonnegative likelihood-unbiasedness result is a statement about the
randomized estimator over branch randomness. It applies when branches are
generated by a valid randomized SMC proposal on the reference measure and the
recorded ancestor probabilities are the probabilities actually used. One
fixed branch is a deterministic realization of that random estimator; the
number itself is neither "unbiased" nor "biased" without averaging over its
branch law. Permanently conditioning HMC on that realization defines an
approximate-likelihood target, not an exact pseudo-marginal target.

*Proof.* Conditional expectation of a correctly weighted randomized proposal
integrates the next Feynman-Kac factor; induction gives the standard
likelihood-estimator expectation. A single frozen realization is one sample of
that estimator and is not itself an expectation. Fixing the random numbers
forever changes the HMC target to the branch-conditioned deterministic
posterior. This is why the implementation records the target class rather
than claiming exact pseudo-marginal inference.

### Proposition 4: recursive score identity

Assume all branch objects are theta-independent and all model scores are exact.
Then `S_t` above equals `grad log(exp(c_t))`, and the sum of `S_t` equals the
gradient of the exact finite scalar.

*Proof.* Write `ell_ti = log v_ti`. By induction,
`grad log W_{t-1,j} = G_{t-1,j}`. Therefore
`grad ell_ti = h_ti`. Differentiating
`c_t = logsumexp(ell_t) - log N` gives
`S_t = sum_i W_ti h_ti`. Differentiating
`log W_ti = ell_ti - logsumexp(ell_t)` gives `G_ti=h_ti-S_t`.
The base case is identical with `ell_0` and `h_0`; summing increments proves
the result.

### Proposition 5: support and defensive mass

For a squared-TT proposal `phi^2 + tau lambda` with `tau>0` and a reference
density `lambda` dominating the target on the stochastic support, the proposal
is positive wherever the target is positive. This is the support condition
needed by Proposition 2; it does not bound the importance-weight variance.

*Proof.* The defensive term is at least `tau lambda`. The paper's Eq. (13)
and lines 562-573 give the corresponding domination argument. Large target /
proposal ratios can still produce collapse, so support is a hard validity gate,
not a performance guarantee.

### Proposition 6: innovation-coordinate singular repair

If `x_t=F_t(x_{t-1},z_t,theta)` has stochastic innovation density `rho_t` and a
deterministic completion, the valid local Radon-Nikodym factor is the density
of `z_t` with respect to its innovation measure, times the declared
stochastic-subspace Jacobian when a change of variables is made. A full-state
Lebesgue density exists only when the map has full stochastic rank.

*Proof.* Push forward the innovation measure through `F_t`; deterministic
coordinates induce a singular component. Applying a full-dimensional
Jacobian formula would divide by directions with zero variance and is not a
valid density. The implementation therefore requires an innovation adapter
or fails closed.

### Proposition 7: TT approximation role

If the TT/KR proposal is evaluated exactly and included in the correction, TT
approximation changes proposal variance and computational cost, not the target
of the randomized importance program. If `q` is approximated, its realized
pointwise value is part of the proposal and any mismatch is a value error, not
merely variance.

*Proof.* Importance correction removes the proposal density that is actually
used. Replacing it by an unrecorded or base density changes the
Radon-Nikodym ratio. The paper's Eq. (23) uses the fitted conditional proposal
in the denominator; `models/full_sol.m:132-137` likewise corrects with
`eval_pdf`.

## 5. Memory Ledger

The online evaluator is streamed over time. Let `d=d_x`, `p_b` be the active
score block, `N` the particle count, and `b` bytes/scalar.

### Live online tensors

| Tensor/buffer | Scalars |
| --- | ---: |
| previous and current states | `2 N d` |
| log weights, increments, selected `log a`, selected `log q`, ancestor indices (integer bytes separate) | `6 N` |
| previous marks, local marks, current marks, weighted reduction workspace | `4 N p_b` |
| optional state tangent | `N d p_b` |
| TT metadata if a map is evaluated online (not required by the first kernel) | `O(d m r^2)` |

Thus a conservative simple-score bound is

`M_simple <= b N (2 d + 6 + 4 p_b) + M_constants`,

and a forward-tangent bound is

`M_tangent <= b N (2 d + 6 + 4 p_b + d p_b) + M_constants`.

`M_constants` includes observations and the streamed branch record; it is
separate from the device live set. If all `T` states are resident, add
`b T N d` host/device storage. The offline proposal artifact should normally
be stored on host and streamed one time step at a time.

### Comparator ledgers

| Route | Persistent peak | Temporary peak | Work warning |
| --- | --- | --- | --- |
| Dense GenUT/Contract-E | Arrays including `N x N x d`, `N x N x d x p_b`, and `N x N x p_b`. | Pairwise differences and tangent contractions. | At least quadratic in `N` per transport iteration. |
| Streaming GenUT/Contract-E | `O(Nd+Np_b)` plus chunk buffers. | `O(K^2 d p_b)` for a chunk. | Exact all-pairs work remains quadratic in `N`. |
| Direct fixed-TTSIRT | TT cores, retained marginal mass objects, rank/basis-dependent work. | Fit and marginal contractions. | `O(m l r^3)` marginal construction and rank-dependent fitting/map work. |
| Frozen APF simple | `O(Nd+Np_b)` live, plus streamed branch record. | `O(Nd+Np_b)` per step. | `O(TN)` model/proposal evaluation after compilation. |
| Frozen APF tangent | `O(Nd p_b)` extra. | Tangent/Jacobian contractions. | `O(TN d p_b)` absent a reverse/blocked adjoint. |
| Offline TT compiler | `O(d m r^2)` core storage plus samples/batches. | ALS/KR work buffers. | Paper reports `O(m l r^2)` function evaluations and `O(m l r^3)` flops for TT fitting; rank is not assumed bounded. |

The handoff's dense tangent lower bound `b N^2 d p_b` is correct. For
`N=5000,d=18,p_b=3`, FP32 requires about 5.03 GiB for that tensor alone;
FP64 doubles it to about 10.06 GiB. At `N=10000` the corresponding values
are about 20.12 and 40.23 GiB, before other arrays.

### Representative APF live-set sizing

These are formula-based capacity numbers, not measured benchmark results. A
hypothetical stress state `d=48` is used because NAWM's full state dimension is
not available from the checked paper. Values are GiB, using the simple bound
above; `T`-persistent branch storage is listed separately for `T=200`.

| N | FP32, p_b=1 live | FP32, p_b=16 live | FP64, p_b=16 live | FP32 branch `T=200` |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.00040 | 0.00062 | 0.00124 | 0.0358 |
| 5,000 | 0.00197 | 0.00309 | 0.00618 | 0.179 |
| 10,000 | 0.00395 | 0.00618 | 0.0124 | 0.358 |

These small live sets are the engineering advantage of the APF, not evidence
that the required `N` is small enough statistically. On this machine the
trusted GPU probe reported an RTX 4080 SUPER with 16,376 MiB VRAM; host memory
reported 45 GiB. The APF live set fits that capacity for the tabled cases,
while a dense `N=5000,d=48,p_b=16` tangent tensor would require about 71.5 GiB
FP32 and is impossible on either the GPU or current host memory without
chunking.

## 6. Work And Amortization Ledger

For fixed branch length `T`:

| Component | Work |
| --- | --- |
| Model transition/observation and analytical score | `O(T N C_model)`; for dense state covariance, include the factorization/solve cost. |
| APF weighting, normalization, score recursion | `O(T N p_b)` plus `O(TN)` log-sum-exp. |
| Fixed ancestor lookup | `O(TN)`. |
| TT fitting | `O(m l r^2)` target evaluations and `O(m l r^3)` flops per paper lines 426-437, multiplied by ALS/optimizer passes and fresh batches. |
| TT marginal construction | `O(m l r^3)` (paper lines 623-626). |
| KR evaluation/inversion | `O(m l r^3 + N m l r^2 + N m l(log l+r)+N m c)` for `N` samples (paper lines 651-670). |
| Dense OT | At least `O(I N^2 d)` work, and tangent variants `O(I N^2 d p_b)` arithmetic. |
| Streaming OT | `O(I (N/K)^2 K^2 d) = O(I N^2 d)` for exact all-pairs blocks, with bounded peak memory. |
| XLA | One-time shape/dtype/device compilation plus warmed `O(TN)` execution; compile time is measured, not inferred from asymptotics. |

Total campaign cost is

`C_total = C_offline + H * C_online + C_refresh`,

where `H` is the declared number of HMC value/score calls and `C_refresh` is
nonzero whenever a new anchor/proposal is required. The break-even count is

`H_break_even = (C_offline + C_refresh) / (C_comparator_online - C_online)`

only when the denominator is positive and both routes satisfy the same value/
score gates. No measured break-even claim is made in this review.

## 7. NAWM II Sizing Case

The checked primary source is the local PDF named in the handoff. Its Bayesian
estimation section states that the model uses the log-linear state-space
representation (PDF text extraction lines 1801-1820), retains 18 observables,
adds six series (lines 1837-1848 and 1888-1922), and has 24 distinct structural
shocks (lines 1925-1960). These support the following table only:

| Quantity | Checked value | Anchor | Status |
| --- | ---: | --- | --- |
| Observed series `d_y` | 24 | NAWM II Section 3.1.1, lines 1837-1848 and 1888-1922 | `NAWM-derived` |
| Distinct structural shocks `d_s` | 24 | Section 3.1.2, lines 1925-1960 | `NAWM-derived` |
| Full state dimension `d_x` | unavailable | Paper says log-linear state-space representation but does not provide a compact state-vector count in the inspected passages | `source gap` |
| Estimated parameter dimension `p` | unavailable | Estimation tables/equations require a model/code partition not present in the checked local adapter | `source gap` |
| Horizon `T` | data sample is described, but no BayesFilter claim horizon is fixed here | Section 3.1.1 | `not fixed` |
| deterministic/algebraic blocks | present in structural equations and recursive blocks | Sections 2.3.3-2.3.4 and Appendix A | `structural evidence; adapter absent` |

Hypothetical stress scenarios, explicitly not NAWM-derived, are:

| Scenario | `d_x` | `d_s` | `d_y` | `p` | `T` | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H1 | 48 | 24 | 24 | 16 | 200 | sparse/block state stress |
| H2 | 96 | 24 | 24 | 32 | 200 | companion/deterministic completion stress |
| H3 | 192 | 24 | 24 | 64 | 400 | pessimistic state embedding stress |

A stochastic-dimension proposal can reduce proposal-map cost only if the
deterministic completion is analytically reconstructed and the innovation
measure/Jacobian is supplied. It cannot be assumed from `d_s=24`; companion,
lag, measurement-error, mixed-frequency, and algebraic states may still enter
the transition/observation evaluation and score. No real NAWM execution is
authorized by this result.

For TT scenarios, `m=9,r=16` and `m=9,r=64` are illustrative basis/rank cases,
not selected settings. Core scalar storage is approximately `d m r^2`:
for `d=48`, this is 0.42 MiB FP32 at `r=16` and 6.75 MiB FP32 at `r=64`,
before boundary, mass, and training buffers. Rank growth must be recorded;
the pessimistic case is `r` increasing with dimension until the compiler's
rank budget veto fires.

## 8. Architecture Comparison

| Architecture | Value target | Score target | Peak memory | Total work | Main risk |
| --- | --- | --- | --- | --- | --- |
| Direct fixed-TTSIRT | Approximate retained-density likelihood | Full derivative through retained TT/marginal/transport, currently missing locally | rank/basis dependent | fitting and map dependent | derivative ownership and rank growth |
| Frozen TT-proposal APF | Fixed-branch importance estimator, with explicit `a` and `q` | Exact derivative of that finite estimator | particle-linear online; branch storage streamed | offline TT plus `O(TN)` online | weight collapse, anchor refresh, singular support |
| Fixed bootstrap/SIS | Standard fixed-branch particle estimator | Same finite estimator score | particle-linear | `O(TN)` | severe degeneracy and proposal mismatch |
| GenUT/Contract-E dense OT | Current reset scalar | Same finite scalar if all terms included | quadratic in `N` and tangent size | quadratic per OT iteration | memory/work explosion |
| GenUT/Contract-E streaming OT | Same scalar | Same | chunk bounded | still exact quadratic work | runtime explosion |
| Blockwise deterministic Gaussian filter | Gaussian approximate likelihood | analytical block recursion | block covariance dependent | block/cubic factorization | approximation bias and structural closure |

The APF is the best first engineering candidate for particle-linear memory, but
it is not the winner scientifically until the fresh high-dimensional ESS and
value/score tests are passed. Direct TTSIRT remains a serious comparator, not
a rejected historical route.

## 9. Minimal Validation Ladder

No long campaign was run for this review. The following ladder is the smallest
discriminating sequence.

| Rung | Configuration | Primary criterion | Veto | Budget/nonclaim |
| --- | --- | --- | --- | --- |
| 0 | 1D/3D and 24D LGSSM, `T=2,10`, `N=64,256`, fixed Gaussian proposal, 4 seeds, `p_b=1,4` | finite-program score agrees with an independent directional derivative and exact Kalman value within predeclared tolerance | wrong `1/N`, omitted ancestor/proposal term, nonfinite, branch drift | <= 5 min CPU; no posterior claim |
| 1 | exact scalar SV, `T=20`, `N=256,1024`, 8 seeds | same-scalar value/score identity and descriptive estimator variance | score mismatch or support failure | <= 10 min CPU; no HMC claim |
| 2 | reduced continuous SIR, `d=2/4`, `T=20`, `N=256,1024`, 8 seeds | comparison to a dense/high-particle reference plus ESS screen | collapse, invalid score, finite-value failure | <= 20 min CPU; no d=18 claim |
| 3 | singular structural fixture in innovation coordinates, `d_s=2,d_x=4`, `T=10`, `N=256` | mixed-measure value/score identity | any Lebesgue misuse or missing Jacobian | <= 10 min CPU; no NAWM claim |
| 4 | Austria SIR `d=18,T=20`, only after rungs 0-3, fresh tune/holdout partitions | analytical score/value contract and declared ESS/log-weight screen | collapse, rank budget, stale tuning, target mismatch | bounded campaign; no NAWM claim |
| 5 | synthetic NAWM-shaped sparse/block model (`d_y=24,d_s=24`, explicit hypothetical `d_x,p,T`) | structural adapter and capacity evidence | unsupported dimension substitution, singular-measure failure | no real NAWM readiness claim |

Each serious rung needs a manifest containing commit, command, environment,
dtype/TF32/XLA, memory policy, seeds, wall time, plan/result paths, and the
proposal branch hash. Means, ESS, tails, and runtimes from a few seeds are
descriptive only; they do not rank candidates.

## 10. Literature And Claim Ledgers

### Source-support ledger

| Source | Classification | Technical support | Limits |
| --- | --- | --- | --- |
| Zhao and Cui, JMLR 25 (2024) | `DIRECT_METHOD` | Eqs. (1)-(23), Proposition 2, Proposition 4, Algorithms 1-3, Section 4, Appendices A-B inspected locally. | Does not prove this frozen APF, HMC readiness, or bounded NAWM rank. |
| Pinned companion code | `IMPLEMENTATION_OR_SOFTWARE` | `full_sol`, `computeL`, TTSIRT map/marginal files, and Austria SIR script inspected. | Local snapshot is patched; code is not a mathematical oracle. |
| NAWM II, ECB Working Paper 2200 | `EMPIRICAL_EXAMPLE` / sizing source | Section 3.1.1-3.1.2 and log-linear-state-space discussion inspected. | Does not supply BayesFilter adapter dimensions or implementation. |
| Snyder et al. (2008) | `FOUNDATIONAL` / `COMPETITOR` | Local Sections 2-5 and equations around the `tau^2` collapse criterion inspected. | Warns about prior-like particle proposals; does not rule out all advanced proposals. |
| Poyiadjis/Del Moral derivative sources in local cache | `FOUNDATIONAL` | Available for future score-comparison audit. | Not used to certify the new kernel until its exact target is checked. |

### Citation/venue metadata ledger

The local OpenAlex cache was last recorded in the existing May/June 2026
ledgers. Zhao-Cui JMLR publication metadata and Snyder MWR metadata are
available; live citation counts/venue metrics were not queried for this review.
Cached counts are coverage signals only and are not correctness evidence.

### Backward snowball ledger

Zhao-Cui's related-work references inspected for this decision include Gordon
et al. and Doucet/Johansen (particle filtering), Pitt/Shephard (auxiliary PF),
Andrieu et al. (particle MCMC), Cui/Dolgov (squared inverse Rosenblatt), and
Spantini et al. (transport filtering). Pitt/Shephard, pseudo-marginal/particle
MCMC, and high-dimensional collapse are required follow-up sources for any
exact-posterior promotion. The quarantined Spantini 2016 workshop item is not
used as support.

### Forward snowball ledger

No live forward-citation query was run. Existing cached metadata is retained
as coverage context; follow-ups, corrections, and replications are
`not checked`, not zero.

### Claim-support ledger

| Claim | Support class | Status |
| --- | --- | --- |
| Zhao-Cui fits an actual adjacent target and marginalizes a squared TT | `PRIMARY_TECHNICAL_SUPPORT` + author-code evidence | checked |
| A defensive squared-TT proposal supplies support domination | `PRIMARY_TECHNICAL_SUPPORT` | checked under Eq. (13) |
| Frozen APF scalar and score above are algebraically identical | `PROJECT_DERIVATION` | established in this result; implementation test pending |
| APF is statistically viable at high dimension | `SOURCE_GAP_BLOCKER` | unknown until fresh ESS/rank evidence |
| NAWM full state/parameter dimensions | `SOURCE_GAP_BLOCKER` | unavailable from checked sources |

### Omitted-paper/reviewer-risk register

| Omission risk | Why it matters | Action |
| --- | --- | --- |
| Pitt-Shephard auxiliary PF foundations | ancestor/proposal normalization | inspect before randomized unbiasedness claim |
| Andrieu et al. particle MCMC and particle-HMC follow-ups | extended-target invariance | inspect before exact posterior claim |
| Snyder/Bengtsson high-dimensional collapse | particle-count feasibility | checked Snyder locally; retain as continuation veto context |
| singular/innovation-coordinate filtering | NAWM structural blocks | add synthetic fixture before NAWM-shaped run |
| TT rank/error bounds | prevents hidden bounded-rank assumption | paper states rank dependence; add heldout rank-growth diagnostics |
| low-rank/factored OT alternatives | fair comparator if transport remains online | inspect if APF/TT proposal fails |

## 11. Implementation Decision And Nonclaims

Implement the fixed-branch APF value/score kernel first, with a repository-owned
branch manifest and fail-closed support checks. Add a Gaussian oracle branch
and tests before wiring any TT/KR compiler. Then add an offline compiler
interface whose proposal implementation can be Zhao-Cui squared-TT/KR, while
keeping the online kernel independent of adaptive fitting.

Do not edit the existing source-route defaults, call the APF source-faithful
Zhao-Cui, run a long Austria/NAWM campaign, or make an HMC/pseudo-marginal
claim until the validation ladder passes. A failed APF candidate triggers
proposal/rank/coordinate repair and fresh tuning; it does not reject the
Zhao-Cui research direction.

The strongest alternative explanation for a later positive result is that the
Gaussian or low-dimensional proposal is unusually well matched; the result
would not transfer to NAWM without the structural adapter and fresh ESS/rank
evidence. The weakest current evidence is proposal quality in dimensions above
the 36-dimensional P76 one-step pilot.
