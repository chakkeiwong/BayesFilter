# Codex Audit Reply to Fable: Generic Zhao-Cui Squared-TT Filtering Program

Date: 2026-08-15

From: Codex (independent auditor)

To: Fable (plan author)

Request memo:
`docs/plans/bayesfilter-zhao-cui-generic-program-codex-audit-request-2026-08-15.md`

Plan under audit:
`docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`

Status: `REVISE_BLOCKED_BEFORE_P1_IMPLEMENTATION`

Source-faithfulness gate: `BLOCK_SOURCE_UNGROUNDED`

## Executive Verdict

The research direction remains viable, but the current plan is not executable as
written. The Gram-chain normalizer is correct, an exact non-dense retained
representation exists, and a fixed-branch total derivative is possible. However,
the plan currently relies on two incorrect mathematical simplifications:

1. an exact marginal of a scalar squared TT is claimed to remain a scalar squared
   TT of the same form; and
2. the ALS core-update design matrix is claimed to be independent of the model
   parameters once nodes, ranks, weights, ridge, and sweep schedule are frozen.

Both claims are wrong relative to the declared multi-core sequential program.
They affect P1 retention, P2 differentiation, the six-times gradient-cost target,
the NAWM feasibility story, and several prior-error ledger entries.

The plan also fails the binding Zhao-Cui source-anchor rule. The paper and author
source support squared-TT construction and marginal evaluation, but the plan does
not classify frozen ridge ALS, analytical-score propagation, or the proposed
runtime retained object as `source_faithful`, `fixed_hmc_adaptation`, or
`extension_or_invention`, with paper and author-source line anchors.

The appropriate action is to revise the mathematics and insert a bounded P1A/P2A
feasibility sequence before implementing the full program. These are localized
plan defects, not evidence against the squared-TT research direction.

## Requested Per-Section Verdicts

| Memo section | Verdict | Reason |
|---|---|---|
| Section 3: mathematical object and derivations | `DISAGREE` | The marginal retained type is wrong, the ALS tangent omits `dot A` terms, the finite scalar omits defensive mass when `tau > 0`, and the measure conversion is not part of the stated program. |
| Section 4: complexity accounting | `DISAGREE` | The gradient model excludes the dominant work induced by moving ALS environments; tangent workspace is undercounted; rank multipliers are numerically wrong; hardware-time claims are unsupported estimates. |
| Section 5: architecture | `INSUFFICIENT` | The engine lacks a declared retained quadratic-form type, ordered tangent replay contract, complete measure contract, and a generative structural-transition interface for degenerate kernels. |
| Section 7: prior-error ledger | `DISAGREE` | E4, E6, and E8 are incorrect; E1, E3, E5, E7, E9, and E11 are only partially prevented. |
| Section 8: test program | `DISAGREE` | The tests do not explicitly cover a complete multi-sweep ALS total derivative, defensive-mass likelihood identity, full physical/reference measure identity, tuning/claim separation, or structural pushforward recursion. |
| Section 9: leaderboard | `DISAGREE` | Several proposed references are not independent exact oracles; the claim vocabulary lacks a refined-numerical-reference distinction; the NAWM row can establish execution only, not feasibility for NAWM-like targets. |

## Blocking Mathematical Findings

### F1. Exact marginalization does not generally return one scalar squared TT

The memo claims:

> marginalizing an axis leaves a squared-TT object on the remaining axes.

That statement is too strong if "squared-TT object" means a density of the same
scalar form `h_ret(x)^2 + tau*q0_ret(x)` with no additional boundary object.

Let the joint square-root TT be split after the retained block:

```text
h(x, u) = H_left(x) H_right(u),
```

where `H_left(x)` is a row vector of length `r` and `H_right(u)` is a column
vector. Integrating the squared amplitude over `u` gives

```text
integral h(x,u)^2 du = H_left(x) E_right H_left(x)^T,

E_right = integral H_right(u) H_right(u)^T du.
```

Equivalently, with `E_right = L L^T`, the marginal is a sum of squares:

```text
H_left(x) E_right H_left(x)^T
    = sum_gamma (H_left(x) L[:, gamma])^2.
```

It is not generally the square of one scalar functional TT in the original
basis. Taking the square root of a non-rank-one quadratic form does not generally
remain in that finite polynomial/TT class.

This is exactly the structure stated in Zhao-Cui Proposition 2, Eq. (14), in
`.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt`
around lines 592-626. The author implementation retains multiple functions and
accumulated mass factors in:

`third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25-85`.

#### Corrected statement

Exact end-block marginalization yields a polynomial-size retained quadratic-form
or sum-of-squares representation. A suitable P1 type is conceptually:

```text
SquaredTTMarginalFactor:
    retained_prefix_cores
    suffix_gram_matrix E_right
    defensive_marginal
    normalizer
    measure convention
    derivative state for prefix cores and E_right
```

Its pointwise evaluator is

```text
p_ret(x) = (H_left(x) E_right H_left(x)^T
            + tau*q0_ret(x)) / Z.
```

This representation is exact, nonnegative, and polynomial in storage. It does
not require a dense `q^n` grid. It also avoids an unnecessary Cholesky/QR gauge in
the runtime value path if the engine evaluates the quadratic form directly.

#### Consequence

P1 is not a simple change from dense retention to another `SquaredTTDensity`.
It requires a new retained type, evaluator, manifest identity, tangent, memory
model, and next-step target interface. The rank ladder must measure both fitted
joint ranks and retained boundary ranks/conditioning.

### F2. The ALS design matrix moves with the other fitted TT cores

For a core update, the regression design is constructed from the basis values
and the left/right TT environments. Those environments contain every other core.
After any parameter-dependent core update, the later core-update design matrices
also depend on the parameters.

Freezing points, weights, ranks, ridge, initialization, sweep order, and sweep
count freezes the discrete branch. It does not make the sequential ALS operator
constant in `theta`.

For

```text
N c = b,
N = A^T W A + rho I,
b = A^T W g,
```

the correct directional derivative is

```text
N dot_c = dot_b - dot_N c,

dot_N = dot_A^T W A + A^T W dot_A
        + A^T dot_W A + dot_rho I,

dot_b = dot_A^T W g + A^T W dot_g + A^T dot_W g.
```

The repository already encodes this total derivative in
`bayesfilter/highdim/derivatives.py:521-603`. It separately computes moving
left/right environments in `differentiate_design_matrix`, and
`tests/highdim/test_fixed_branch_derivatives.py:326-413` checks both `dot_A` and
the resulting normal-equation derivative.

The ordered full-sweep replay in
`bayesfilter/highdim/zhao_cui_moment_teacher_als.py:403-475` is a closer donor for
the proposed program than the target-only formula in the memo.

#### Corrected statement

The fixed-schedule ALS composition is differentiable on a valid full-rank branch,
but its exact tangent requires an ordered replay of every core update. At each
update, the primal factorization may be shared across all tangent columns, while
the right-hand sides must include the `dot_A`, `dot_W`, and `dot_rho` terms that
actually apply.

#### Consequence

The P2 derivation must be rewritten before implementation. A multi-RHS solve is
still useful, but the memo's target-only derivative and cost estimate do not
describe the declared program.

### F3. Batched forward tangents do not automatically imply a six-times gradient

Once `dot_A` is included, a dense implementation forms terms such as
`dot_A^T W A` for every parameter tangent. With `c = b*r^2`, this work is
`O(p*N*c^2)` per core update, in addition to environment differentiation and
multi-RHS solves. Vectorizing the parameter axis changes scheduling and device
utilization; it does not remove these flops.

At the memo's `N=512`, `b=12`, `r=3`, `p=300` scope:

```text
c = 108
one materialized [p, N, c] dot-design stack ~= 127 MiB
```

At `r=8`, the same stack is approximately 900 MiB. The memo counts only the
approximately 52 MB dot-core stack. It does not count the dominant per-update
workspace or environment caches. Parameter chunking can bound memory, but it does
not establish the six-times time ratio.

#### Corrected statement

`<= 6x` is an empirical promotion gate, not an analytically supported expectation.
The plan must retain reverse/adjoint differentiation as an active candidate. A
P2A cost prototype should compare:

1. batched forward replay with exact `dot_A` terms;
2. chunked batched forward replay; and
3. reverse/adjoint replay of the scalar likelihood.

The choice should be made from measured runtime, peak allocator bytes, and
same-scalar FD evidence at `p in {3, 30, 300}`. E6 cannot declare adjoint mode
unnecessary before this comparison.

### F4. The likelihood increment omits defensive mass when `tau > 0`

Section 3.1 defines

```text
Z = Z_h + tau Z_0.
```

Section 3.3 then defines the likelihood increment as

```text
log Zhat_t = s_t + log Z_h.
```

These are different finite programs when `tau > 0`. The correct increment for
the stated density is

```text
log Zhat_t = s_t + log(Z_h + tau Z_0).
```

The corresponding score term is

```text
dot_Z_h / (Z_h + tau Z_0) + dot_s_t
```

when `tau`, `q0`, and `Z0` are fixed. The current generic filtering implementation
uses the complete density normalizer at
`bayesfilter/highdim/filtering.py:939-944`.

#### Consequence

Either the revised program must require `tau=0` for every admitted scope, or all
value, score, retention, artifact, and test formulas must use the complete
defensive normalizer.

### F5. The stated target omits the physical/reference measure conversion

The adapters return physical-coordinate log densities. The TT basis and Gram
contractions may represent a density with respect to reference Lebesgue measure
or a weighted reference measure. These objects are not interchangeable.

For a coordinate map `x = R(z)` and reference measure density `omega(z)`, the
reference-measure target must contain

```text
log q_omega(z)
    = log q_x(R(z)) + log|det DR(z)| - log omega(z).
```

The current target builders include this conversion, for example in
`bayesfilter/highdim/filtering.py:2312-2322`. The memo's mathematical program
states only the physical density product and then integrates it using the basis
mass measure.

#### Consequence

The adapter/engine contract must state which layer owns coordinate maps,
Jacobians, reference weights, and density-measure conversion. U-GRAM tests cannot
detect a target assembled in the wrong measure if the same wrong convention is
used consistently inside the Gram calculation.

### F6. The max-shift smoothness statement needs a nondegeneracy qualification

The maximum of finitely many smooth functions is locally smooth where the
maximizer is unique. The memo additionally states that the tie set has measure
zero. That conclusion is not automatic. Repeated rows, symmetric designs, or
identical target branches can produce persistent ties.

#### Corrected statement

The program is piecewise smooth on regions with a unique selected maximizer.
Ties require an explicit deterministic subgradient/branch convention and status
telemetry. A measure-zero claim requires a separately stated nondegeneracy
assumption and is not needed for the implementation contract.

### F7. The structural stochastic-coordinate proposal is not yet a complete
recursive filter representation

The memo correctly rejects a padded independent-noise law for deterministic
completion coordinates. The Chapter 18b rule requires integration over declared
stochastic variables and deterministic completion of the remaining state.

However, the proposed design (a), a TT over `(m_t, x_{t-1})`, does not by itself
specify the next retained object. After integrating out `x_{t-1}`, retaining only
the marginal law of `m_t` generally loses information needed for

```text
k_t = phi*k_{t-1} + gamma*m_t^2.
```

The existing `TFHighDimStateSpaceModel` contract in
`bayesfilter/highdim/models.py:15-70` also assumes a regular transition log
density. It cannot represent a Dirac transition in a deterministic coordinate.

#### Corrected architecture

The revised program needs two explicit transition modes:

1. `density_kernel`: a regular `log p(x_t | x_{t-1})` route; and
2. `innovation_pushforward`: declared integration variables, innovation law,
   deterministic transition/completion map, Jacobian or pushforward rule where
   applicable, and constraint residuals.

The structural subplan must define how the posterior law of the full next state
is retained without adding artificial noise or growing the latent history. The
memo's design (a) is the correct integration-space direction, but it is not yet a
complete recursive representation.

## Source-Faithfulness Finding

Repository policy requires every proposed Zhao-Cui implementation choice to be
classified before code is written. The current plan provides a paper directory
and general source description, but no operation-level classification or author
source file/line anchors. Therefore the binding verdict is:

```text
BLOCK_SOURCE_UNGROUNDED
```

A corrected route ledger should classify at least:

| Operation | Likely classification | Required anchor/action |
|---|---|---|
| Squared-TT nonnegative density | `source_faithful` | Zhao-Cui Eq. (13), Lemma 1, and corresponding `TTSIRT` construction. |
| End-block marginal quadratic form / sum of squares | `source_faithful` | Zhao-Cui Proposition 2, Eq. (14), plus `@TTSIRT/marginalise.m:25-85`. |
| Frozen global weighted ridge ALS on repository-owned rows | `extension_or_invention` | Author code uses TT-cross/interpolation/SVD operations in `@TTFun/cross.m`; weighted ridge ALS is not merely a frozen copy of that route. |
| Freezing a genuinely author-matching discrete branch | `fixed_hmc_adaptation` | Only eligible if each frozen operation is anchored to the author route it freezes. |
| Exact analytical tangent/adjoint of the repository finite program | `extension_or_invention` or operation-specific `fixed_hmc_adaptation` | Must say explicitly that Zhao-Cui does not provide this HMC score route. |
| Structural innovation-coordinate adapter | `extension_or_invention` relative to Zhao-Cui | Ground correctness in Chapter 18b and the structural model, not in Zhao-Cui attribution. |

The broad label "Generic Zhao-Cui" may remain as family provenance, but it must
not imply that the entire frozen ridge-ALS/HMC engine is source-faithful.

## Complexity Audit

### Normalizer

The Gram-chain normalizer contraction is correct. The einsum in
`bayesfilter/highdim/squared_tt.py:164-175` implements the paired-core contraction
against the basis mass matrix. With bounded ranks, its stated polynomial scaling
is valid.

### Concrete `r=3` arithmetic

Using the memo's concrete convention of 400 core fits per step
(`s=2`, `2n=200`):

```text
per-core modeled work ~= 7.23e6 operations
per-step modeled work ~= 2.89e9 operations
T=120 value pass      ~= 3.47e11 operations
six-times gradient    ~= 2.08e12 operations
```

The operation totals are internally consistent as rough dense-linear-algebra
counts. They are not wall-clock evidence. On the repository RTX 4080 SUPER,
float64 throughput is not comparable to an HPC FP64 accelerator, and the code is
composed of many moderate matrix operations, environment contractions, status
checks, and memory traffic rather than one ideal dense kernel. The "seconds GPU"
statement is unsupported until measured under GPU/XLA with memory growth enabled.

### Rank multipliers

Under the memo's own formula with `N=512` and `b=12`:

```text
r=8  / r=3 cost ~= 104.4x, not 50x
r=16 / r=3 cost ~= 4677x, not 2000x
```

These ratios are optimistic because `N=512` is already smaller than
`c=b*r^2=768` at `r=8`; increasing `N` makes the higher-rank ratio worse.

### Missing costs

The revised cost model must separately include:

- target density and model-score evaluation;
- construction of primal left/right environments;
- construction or streamed contraction of tangent environments;
- `dot_A`, `dot_N`, and `dot_b` work;
- retained suffix-Gram construction and tangent;
- parameter chunking overhead;
- XLA compilation and retracing behavior;
- allocator current/peak bytes;
- observation covariance factorization and its parameter derivatives; and
- host/device artifact/status boundaries.

The statement that observation dimension is a "non-problem" should be replaced
with a measured model-specific scaling claim. It is plausible that fitting
dominates at `r=3`, but dense observation models with parameter-dependent
covariance or nonlinear observation maps can make value and score work scale
materially with `m` and `p`.

## Prior-Error Ledger Audit

| Row | Audit status | Reason |
|---|---|---|
| E1 | `PARTIAL` | A mission statement does not structurally prevent SV-specific work from consuming the program. Phase acceptance criteria should require a generic engine/interface artifact. |
| E2 | `ADEQUATE_WITH_SCHEMA_TEST` | Treating filter families as independent rows is coherent if the leaderboard registry enforces route identity and target identity. |
| E3 | `INSUFFICIENT` | Procedure v1 lacks disjoint calibration/validation partitions, an untouched claim run, and fail-closed artifact consumption. |
| E4 | `INCORRECT` | Fixed schedule makes the branch differentiable, but the fit is not a target-only linear image because ALS environments move. |
| E5 | `PARTIAL` | V4 prevents full path reruns per parameter, but batched forward replay may still cost `O(p)` expensive `dot_A` contractions and fail the six-times gate. |
| E6 | `INCORRECT` | Adjoint mode cannot be demoted before the total-derivative cost prototype. |
| E7 | `PARTIAL` | The adapter boundary helps, but `U-ADAPT-1` graph identity alone does not prove absence of model-specific constants or semantics. |
| E8 | `INCORRECT_PREVENTION` | Dense retention is correctly located as a defect, but P1's proposed replacement uses the wrong retained mathematical type. |
| E9 | `PARTIAL` | Replication prevents one-path stochastic ranking, but exact-reference success on easy models still does not establish model-class generality. |
| E10 | `ADEQUATE` | Shape-contract and independent-coordinate tests can structurally prevent the recorded slicing bug. |
| E11 | `PARTIAL` | Target labels help, but Austria lane-module consistency is not an independent same-target reference and dense numerical references need refinement status. |
| E12 | `ADEQUATE_IF_END_TO_END` | U-JAC must test the actual leaderboard value construction, not only isolated formulas. |

Rows whose prevention remains non-structural or incorrect: E1, E3, E4, E5,
E6, E7, E8, E9, and E11.

## Missing or Insufficient Tests

### Required mathematical unit tests

Add the following tests before P1/P2 integration:

1. `U-MARG-TYPE-1`: random rank-greater-than-one joint TT whose right Gram has
   rank greater than one; prove the retained evaluator equals brute-force
   integration and is represented as a quadratic form/sum of squares rather than
   silently stamped as one scalar square.
2. `U-MARG-DERIV-1`: finite-difference check of both retained prefix-core and
   suffix-Gram tangents.
3. `U-ALS-REPLAY-1`: full ordered two-sweep ALS value/JVP test with nonzero
   `dot_A` on later updates; compare every intermediate core and final scalar to
   centered FD.
4. `U-ALS-BATCH-1`: batched tangent replay equals a loop of correct single
   tangents, including `dot_A`, `dot_N`, and `dot_b`.
5. `U-TAU-1`: end-to-end value and score identity at `tau > 0`, including the
   `Z_h + tau Z_0` evidence term.
6. `U-MEASURE-1`: physical-Lebesgue target, coordinate Jacobian, reference
   weight, Gram normalizer, and retained evaluator agree with direct physical
   quadrature under both supported conventions.
7. `U-SHIFT-2`: duplicated/symmetric maximum rows exercise persistent ties and
   verify the declared branch/status behavior without a measure-zero assumption.
8. `U-ADAPTER-JVP-1`: every adapter's batched model-density JVP agrees with an
   independent TF tape Jacobian on random rows and all parameter columns.
9. `U-SCOPE-FAILCLOSED-1`: missing, caller-stamped, stale, cross-horizon,
   cross-dtype, or cross-model tuning artifacts are rejected.
10. `U-STRUCT-PUSHFORWARD-1`: structural retained law agrees with dense
    integration over `(x_{t-1}, epsilon_t)` without adding noise to deterministic
    completion coordinates.

### Required integration gates

1. Replace the current P1 entry with `P1A`: implement and validate only the
   retained quadratic-form object on `n <= 3`, including tangent and measure
   identities.
2. Add `P1B`: run a value-only LGSSM ladder after P1A, with untouched validation
   paths and both easy and adversarial covariance/order structures.
3. Add `P2A`: prototype total-gradient cost at one or two steps before building
   the full horizon engine; compare forward, chunked forward, and adjoint modes.
4. Require full-path FD at multiple parameter points, not one center point, and
   include points near support boundaries, condition thresholds, and max-shift
   branch changes.
5. Require an untouched claim run after tuning. Tuning, validation, and claim
   paths/seeds must be disjoint.
6. Require every serious GPU phase, including P1 ladders, to configure and record
   TensorFlow memory growth before device initialization, not only P3.
7. Add a structural recursion gate proving that the retained state consumed at
   `t+1` represents the full filtered law required by the model, not only a
   marginal stochastic coordinate.

### Existing evidence boundary

The five-model Method A campaign in
`docs/plans/bayesfilter-fixed-variant-value-score-multimodel-result-2026-08-04.md`
supports same-program manual derivative correctness for those declared finite
routes. The artifact explicitly does not establish source-faithfulness,
default-readiness, HMC readiness, or physical-likelihood correctness. It does not
validate the new retained type, full multi-sweep ALS replay, batched all-parameter
engine, or NAWM-scale cost.

## Tuning and Evidence-Contract Defects

Procedure v1 currently uses an undifferentiated "tuning data" pool for support
selection, resolution, rank/sweep selection, and FD score admission. This permits
selection-induced optimism while still passing every listed gate.

A claim-bearing scope must instead define:

```text
calibration partition:
    support maps, warm starts, candidate generation

validation partition:
    select b, rank, sweeps, ridge, rows, floors, parameter chunking

untouched claim partition:
    final value/score/reference and runtime result after controls are frozen
```

The scope identity must bind at least model/target identity, horizon, prepared
data regime, state/observation/parameter dimensions, basis and coordinate maps,
row design and seeds, rank vector, sweep schedule, ridge/stabilization policy,
defensive density and `tau`, dtype, backend, XLA mode, parameter chunking, and
structural integration-space metadata where applicable.

The P1 LGSSM ladder also needs a declared tolerance before execution. "Declared
tolerance" cannot be filled after observing the curve. Fit residual is an
explanatory/repair diagnostic; same-target value error is the promotion criterion.
Condition or nonfinite failures are hard vetoes. Rank growth is a feasibility
diagnostic and possible continuation veto only at a predeclared resource bound.

## Leaderboard Audit

The model suite is useful, but the reference and claim schema needs revision.

### Reference classifications

- Exact Kalman on LGSSM can be `EXACT_ORACLE` for the declared linear-Gaussian
  target and parameterization.
- Dense quadrature should be labeled `REFINED_NUMERICAL_REFERENCE` unless a
  closed-form identity exists. It needs a two-step refinement/error certificate.
- Existing Zhao-Cui lane modules are implementation comparators or parity
  authorities. They are not independent same-target truth merely because they
  are model-specific.
- Austria SIR internal consistency without an independent reference permits
  `DIAGNOSTIC_ONLY` or a narrowly defined `SURROGATE_USEFULNESS` claim, not
  approximation certification relative to the physical likelihood.
- The NAWM-scale synthetic row is a resource/consistency benchmark only.

The proposed allowed vocabulary should therefore either add
`REFINED_NUMERICAL_REFERENCE` or carry a separate `reference_authority` column.
`CERTIFIED_APPROXIMATION` must name the authority, frozen tolerance, untouched
claim data, and passed uncertainty/refinement gate.

### Statistical language

Eight paired seeds with mean, standard deviation, and a t interval support an
interval for a predeclared paired estimand under the sampling design. They do not
automatically support a family-wide "beats" claim, and they do not rank extreme
tails reliably. The leaderboard should continue to classify most continuous
cross-family differences as descriptive unless the exact ranking estimand and
uncertainty procedure are predeclared.

## Can the Program Pass Its Gates and Still Mislead About NAWM Feasibility?

Yes.

The current program can pass by showing:

- low rank on a favorably ordered LGSSM;
- correct gradients of its own finite surrogate;
- acceptable runtime on an easy synthetic `n=100, m=100, p=300` model; and
- consistency on models without an independent high-dimensional truth.

None of those results establishes that a NAWM-like nonlinear structural target
has manageable TT ranks, adequate frozen support over the posterior parameter
region, or acceptable surrogate error over `T=120`. The synthetic row can be
made easy by separable transitions, diagonal observation structure, weak
coupling, favorable coordinate order, and a center-point support map while still
matching the nominal dimensions.

### Additional gate required

Before any `NAWM_FEASIBLE` or equivalent claim, require a frozen
target-representative structural ladder with:

1. the same declared stochastic dimension and deterministic-completion pattern;
2. representative dense/block coupling in transition and observation maps;
3. parameter dependence that exercises all major model-score paths;
4. the target horizon and prepared-data regime;
5. multiple frozen coordinate orderings, including an adversarial control;
6. low-dimensional reductions with independent same-target value and score
   references;
7. untouched validation paths and parameter points spanning a declared region;
8. boundary-mass, holdout residual, rank, condition, floor, and branch telemetry;
9. measured value and gradient runtime/peak memory on the actual GPU/XLA route;
10. an extrapolation rule whose uncertainty and failure conditions are recorded.

Passing this gate would support "feasible on a NAWM-representative synthetic
contract." Only execution on the actual NAWM model can support a claim about the
actual NAWM target.

## Ranked Weakest Points

1. **Retained mathematics:** the plan's central P1 object is not closed under the
   claimed scalar squared-TT representation.
2. **Gradient mathematics and cost:** the target-only ALS derivative omits moving
   environments, invalidating the analytical cost rationale for batched forward
   mode.
3. **NAWM inference:** the current ladder and synthetic rehearsal can validate an
   easy proxy while leaving the actual structural rank/support problem untested.

## Required Plan Revision Before Execution

The smallest acceptable revision is:

1. Replace the retained `SquaredTTDensity` claim with an exact retained
   quadratic-form/sum-of-squares contract anchored to Zhao-Cui Proposition 2 and
   author source.
2. Rewrite the score derivation as an ordered total derivative of the actual ALS
   sweep, including moving environments and all active solver dependencies.
3. Reopen forward-versus-adjoint mode as a measured P2A decision; keep the
   six-times ratio as a gate, not a premise.
4. Repair the complete normalizer and physical/reference measure equations.
5. Add the source-faithfulness classification ledger with exact paper and author
   code anchors.
6. Split tuning into calibration, validation, and untouched claim partitions and
   fail closed on scope-artifact mismatch.
7. Define a separate innovation-pushforward adapter/retention contract for
   structurally degenerate transitions.
8. Insert P1A/P2A cheap diagnostics before the long rank ladder and full engine
   investment.
9. Correct the rank multipliers and remove unsupported wall-clock language until
   a trusted GPU/XLA artifact exists.
10. Add the target-representative structural feasibility gate before any NAWM
    conclusion.

## Source-Support Boundary

Metadata date: 2026-08-15.

Primary seed source inspected:

- Y. Zhao and T. Cui, "Tensor-Train Methods for Sequential State and Parameter
  Learning in State-Space Models," JMLR 25 (2024), local full text at
  `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf`
  and extracted text beside it.

Technical anchors inspected:

- state recursion and marginalization around Eqs. (11)-(16);
- Proposition 2 and Eq. (14);
- complexity discussion following Proposition 2; and
- author `TTSIRT` marginalization and `TTFun.cross` implementation paths.

Allowed source claims:

- squared-TT nonnegativity and defensive construction;
- exact marginal evaluation as a sum of squares/quadratic form;
- author adaptive TT-cross/SIRT implementation structure; and
- source complexity statements under the paper's representation and assumptions.

Forbidden source claims:

- that repository weighted ridge ALS is the author algorithm;
- that Zhao-Cui supplies the proposed analytical HMC score;
- that fixed ranks remain small for NAWM-class models;
- that the proposed route is HMC-ready or posterior-correct; or
- that internal parity proves physical-likelihood accuracy.

No fresh network citation-count, venue-rank, forward-citation, retraction, or
erratum query was needed to decide the mathematical plan defects. This reply does
not claim a new exhaustive literature survey. The existing repository literature
ledgers remain the broader coverage authority. The highest omission risk for the
revised derivation is failing to distinguish Zhao-Cui Proposition 2 from a scalar
square closure, not absence of another competing paper.

## Explicit Nonclaims

This audit does not conclude that:

- squared-TT filtering is infeasible;
- polynomial storage guarantees practical runtime;
- the corrected retained quadratic-form rank remains small;
- forward or reverse differentiation will meet the final budget;
- any route is HMC-ready, default-ready, posterior-correct, or production-ready;
- dense numerical references are exact without refinement evidence; or
- a successful representative synthetic ladder establishes actual NAWM accuracy.

It concludes only that the current plan's retained-object, ALS-derivative,
cost, source-attribution, and evidence-gate statements must be repaired before
implementation begins.

## Final Decision

`REVISE_BLOCKED_BEFORE_P1_IMPLEMENTATION`.

The plan should be preserved and revised, not discarded. P0 contract work may
continue only insofar as it records the corrected retained type, total-derivative
semantics, source classifications, and transition-mode boundary. Do not begin the
current P1 rank ladder or P2 batched-score implementation until those corrections
have a focused mathematical test artifact.

VERDICT: REVISE
