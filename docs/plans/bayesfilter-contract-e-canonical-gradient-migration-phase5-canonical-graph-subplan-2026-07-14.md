# Phase 5 Subplan: Canonical One-Graph Value, JVP, And FD

Date: 2026-07-14

Status: `REVIEWED_ACTIVE`

Master program:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-master-program-2026-07-13.md`

## Phase Objective

Build one repository-owned LGSSM callable that reconstructs
candidate-dependent initialization, executes LEDH flow and likelihood
increments, normalizes corrected logits without a probability floor, applies
the Phase 4 row-quotient streaming transport and Contract E-Chol reset, and
returns both its literal primal and analytic parameter JVP. Every finite-
difference center and endpoint must invoke that same compiled callable and
return its primal; a separately compiled or algebraically reconstructed value
route is forbidden.

Phase 5 is an engineering graph-identity and local derivative phase. It does
not own the multi-seed Kalman comparison, production admission, nonlinear
migration, HMC, or leaderboard regeneration.

## Entry Conditions Inherited From Phase 4

- Contract E-Chol is the only canonical reset candidate; raw/v1 routes remain
  historical and ineligible.
- The Phase 4 quotient explicitly carries `Q,M,Y,dQ,dM,barQ,barM` without a
  floor or dense production state.
- Direct source/probability-weight and transport
  source/normalized-log-weight paths are separated and composed in the correct
  coordinates.
- Selected `B=1,N=10000,d=3` forward and analytic VJP graphs execute on trusted
  GPU/XLA/TF32, but that is feasibility evidence only.
- `valid_chart` is a hard veto. The particle-only convenience wrapper is not an
  admission gate.
- General dense/autodiff/chunk agreement, row/Sinkhorn/chunk adequacy, and all
  six reset-adequacy blockers remain unresolved.
- The production factory remains empty; no v2 artifact is admitted.
- The platform-blocked Claude route is not retried. Fresh bounded Codex review
  is the substitute.
- Historical LGSSM benchmark files are comparison/extraction sources, not the
  canonical route. Their raw-reset functions and separately compiled value/
  score entry points must not be relabeled or silently mutated into canonical
  authority.

## Confirmed Historical Defects To Remove

1. The old score shard returns the primal of a compact sensitivity graph while
   FD calls a separately compiled value graph. TF32 makes their center values
   differ, so `same_scalar_finite_difference` was false.
2. The old compact callable freezes center-prepared `initial_particles` in its
   primal while injecting `initial_noise * d_initial_std` into the tangent. The
   reported tangent therefore differentiates an operation absent from the
   literal perturbed primal.
3. Historical normalization forms `log(max(weights,floor))`; the canonical
   chart has no probability floor and must use
   `normalized_log_weights = corrected_logits - logsumexp(corrected_logits)`.
4. Historical transport carries the unnormalized numerator and raw reset. The
   canonical graph must call the Phase 4 quotient-plus-Contract-E composition.

## Frozen LGSSM Finite Scalar

The physical parameter vector has no transform in Phase 5:

```text
theta = (phi1, phi2, phi3, q_scale, r_scale).
```

Its valid chart requires `abs(phi_j)<1`, `q_scale>0`, and `r_scale>0`. Define

```text
F(theta) = diag(phi1, phi2, phi3)
Q(theta) = q_scale^2 * I_3
R(theta) = r_scale^2 * I_3
s0_j(theta) = q_scale / sqrt(1 - phi_j^2)
H = [[1.0, 0.25, -0.15],
     [0.2, 1.1, 0.3],
     [-0.1, 0.35, 0.9]].
```

There is no offset. For fixed standard-normal prepared arrays `z0` and `zt`,
the initial carried cloud and transition-first state at zero-based time `t` are

```text
X_{-1} = z0 * s0(theta)
logw_{-1,i} = -log(N) for every batch and particle
mu_t   = X^star_{t-1} F(theta)^T
X0_t   = mu_t + q_scale * zt[t].
```

At `t=0`, `X^star_{-1}` means `X_{-1}`; there is no likelihood contribution
before this transition. The linear observation is `H X0_t`, with prepared
observation `y_t`.

The finite LEDH flow maps `X0_t` to `X_t` and exposes the proposal density at
the pre-flow draw and the flow Jacobian. For each particle, the corrected logit
has these exact signs:

```text
a_ti = previous_log_weight_ti
       + log Normal(X_ti; mu_ti, Q(theta))
       + log Normal(y_t; H X_ti, R(theta))
       - log Normal(X0_ti; mu_ti, Q(theta))
       + forward_log_abs_det_J_ti.
```

Then

```text
Delta_tb = reduce_logsumexp_i(a_tbi)
logw_tbi = a_tbi - Delta_tb
w_tbi = exp(logw_tbi)
L_b(theta) = sum_{t=0}^{T-1} Delta_tb
objective(theta) = mean_b L_b(theta)
per_batch_score[b,k] = d L_b / d theta_k
score[k] = mean_b per_batch_score[b,k]
         = d objective / d theta_k.
```

After an active reset, the next log weights are exactly `-log(N)` for every
particle. After an inactive reset, carry `logw_t` unchanged. The likelihood
increment is always added before reset. These equations, model matrices,
zero-based convention, mean batch aggregation, and equal-weight value are part
of route identity. In particular, `previous_log_weight` in the `t=0` corrected
logit is exactly `-log(N)`; it is neither zero nor an omitted additive constant.

### Frozen Linear-Gaussian LEDH Map

For each batch, the affine flow uses no jitter, adaptive stabilization, or
explicit matrix inverse. Let

```text
LQ = chol(Q)
LR = chol(R)
Qinv = cholesky_solve(LQ, I_3)
Rinv = cholesky_solve(LR, I_3)
K = Qinv + H^T Rinv H
LK = chol(K)
C = cholesky_solve(LK, I_3)
LC = chol(C)
m_ti = C (Qinv mu_ti + H^T Rinv y_t)
A = LC * LQ^{-1}
X_ti = m_ti + A (X0_ti - mu_ti)
log|J_t| = logdet(LC) - logdet(LQ).
```

All applications use row-vector storage with the corresponding transposes; no
explicit inverse tensor is formed. `Q`, `R`, `K`, and `C` Cholesky factors must
be finite with positive diagonals or the flow chart is invalid. The manual JVP
must differentiate these exact Cholesky solves and products; it may not copy
the historical adaptive-jitter or explicit-`tf.linalg.inv` graph.

### Frozen Transport Geometry

For active reset input `X_t`, define population-coordinate standard deviations

```text
c = mean_N(X_t)
sigma_j = sqrt(mean_N((X_t[:,j]-c_j)^2))
D = max_j sigma_j
s = sqrt(3) * D
scaled_X = (X_t-c)/s.
```

The chart requires finite `D>0`. At ties, the JVP uses TensorFlow reduction
semantics represented explicitly by an equal split across exactly tied maxima;
the max mask/count is recorded at center and FD endpoints. Let

```text
r = max(scaled_X) - min(scaled_X)
epsilon0 = max(r^2, 1e-6).
```

The max/min masks and the `r^2 >= 1e-6` branch are recorded and must remain
identical at center/endpoints. The finite transport uses prepared scalar
`epsilon>0`, prepared `0<scaling<=1`, and a fixed integer step count. Its
annealing update is `running <- max(running*scaling^2, epsilon)` with every
active branch recorded. These declared nonsmooth branches are part of the
finite target; an unrecorded floor, stop-gradient, or schedule change is
forbidden.

## Research Intent Ledger

| Field | Binding Phase 5 intent |
| --- | --- |
| Main question | Does one finite TensorFlow graph return a primal and analytic JVP for the same Contract E LGSSM numerical program, with FD invoking that exact graph? |
| Candidate | Fixed-noise LGSSM LEDH recursion with fixed reset mask, finite Sinkhorn schedule, row quotient, fixed residual designs, and fixed prepared ridge. |
| Expected failure mode | Hidden graph duplication, candidate initialization outside the callable, double/omitted normalization derivative, invalid reset chart, or historical raw helper reachability. |
| Primary engineering criterion | Bitwise center-primal identity from repeated calls to one concrete callable; valid branches/charts; exact outer-wiring certificates; full tiny-graph analytic JVP bitwise equality with TensorFlow forward autodiff of the same primal core; same-callable FD ladder artifact. |
| Promotion veto | Any graph-identity, initialization, coordinate, quotient, reset, branch, finite, or prepared-input identity failure. |
| Continuation veto | Invalid target/harness, unavoidable historical raw route, missing fixed prepared inputs, concurrent in-scope edit, or campaign budget exhaustion. |
| Repair trigger | Local JVP/autodiff/FD mismatch with a valid target and chart; XLA compile/resource failure; review finding. |
| Explanatory diagnostics | Componentwise JVP/FD differences, FD step ladder, likelihood prefix values, reset residuals, condition proxies, runtime, and memory. |
| Must not be concluded | Kalman equivalence, FD statistical confidence, production admission, HMC readiness, nonlinear validity, or leaderboard readiness. |

## Canonical Callable Boundary

Create an owned TensorFlow graph module under
`bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py` and focused tests
under `tests/highdim/`. A benchmark harness may prepare deterministic inputs and
emit evidence, but it must call the owned module and cannot define the
canonical mathematics itself.

The prepared closure contains only parameter-independent data:

```text
observations              [T,m]
initial_noise             [B,N,d]
transition_noise          [B,T,N,d]
fixed_reset_mask          [B,T]
residual_design           [B,T,N,d]
prepared_ridge            [B,T]
finite Sinkhorn settings  scalars/static integers
```

The callable accepts `theta[p]` and returns a tensor-only record containing at
least:

```text
objective                 scalar
per_batch_log_likelihood  [B]
score                     [p]
per_batch_score           [B,p]
valid_chart               [B]
minimum_mass              [B]
branch_identity tensors   fixed mask/schedule/chart fields
```

Inside the callable, compute

```text
initial_particles(theta) = initial_noise * initial_std(theta)
```

before both primal and tangent state are initialized. The score is the JVP with
the identity parameter basis, not a gradient of a separately evaluated value.

The parameter-direction axis is always the final axis `p=5`:

```text
d_initial_particles       [B,N,d,p]
d_running_particles       [B,N,d,p]
d_running_log_weights     [B,N,p]
d_scaled_geometry         [B,N,d,p]
d_normalized_log_weights  [B,N,p]
d_normalized_weights      [B,N,p]
d_residual_design         [B,N,d,p] = 0
d_ridge                   [B,p] = 0
d_epsilon0                [B,p].
```

The identity basis is `I_5`; column `k` is direction `k`. Phase 4 receives all
five directions on that final axis and internally maps only over `p`. It must
never fold directions into `B`, `N`, geometry, or payload axes. Per-time
`d_incremental[B,p]` is accumulated into `per_batch_score[B,p]`, and
`score=reduce_mean(per_batch_score,axis=0)` exactly matches the declared mean
objective.

## Exact Weight Semantics

For corrected logits `a`:

```text
ell = reduce_logsumexp(a)
logw = a - ell
w = exp(logw)
dell = sum(w * da)
dlogw = da - dell
dw = w * dlogw
```

No probability floor, `log(max(w,floor))`, clip, or stopped normalization is
allowed. For a reverse-coordinate unit test, apply the unique pullback once:

```text
G_a = G_increment * w + G_logw_total
      - w * sum(G_logw_total)
```

The Phase 4 `G_logw_total` already includes `w * G_w_probability`; no second
simplex projection or probability-coordinate addition is allowed.

## Reset And Time Semantics

At every time step:

1. transition the carried particles using fixed transition noise;
2. run the candidate-dependent LEDH flow;
3. compute the exact transition, observation, proposal, and Jacobian terms;
4. normalize corrected logits with the equations above and add `ell` to the
   log likelihood;
5. if the frozen reset mask is active, derive scaled geometry and its tangent,
   call the Phase 4 Contract E streaming forward/JVP core with that time's fixed
   residual design and ridge, require `valid_chart`, carry equal weights, and
   continue from the reset particles;
6. otherwise carry post-flow particles and normalized log weights unchanged.

The reset mask, finite Sinkhorn schedule, chunks, residual design, and ridge are
fixed prepared inputs and identical at center/FD endpoints. Candidate-dependent
adaptive ridge or residual randomness is forbidden.

## Same-Graph FD Contract

Instantiate one `tf.function(jit_compile=True)` concrete callable returning
both primal and score. The FD harness must invoke that callable, including score
computation, at the center and every endpoint; it may discard the endpoint
score only after execution. It must not trace or call a value-only wrapper.

Before any derivative comparison:

- call the concrete function twice at the center;
- require bitwise-identical serialized primal and per-batch likelihood;
- require identical branch/chart ledgers and prepared-input hashes;
- compare the score artifact center primal with the FD artifact center primal
  bitwise; and
- invalidate the artifact before computing a relative error if any identity
  fails.

Use the Phase 1 central-FD step ladder and active-set report. The owner-directed
`0.05*sqrt(p)` relative rule is a heuristic FD screen only. It is not a general
gradient-accuracy, Kalman, or promotion threshold. Near-zero components require
the frozen absolute-scale reporting path; do not divide by a near-zero oracle.

## Non-Arbitrary Derivative Certificate

Phase 5 success does not depend on a fitted relative tolerance. Before reading
the full tiny-graph result, freeze exact outer-wiring microcertificates with
dyadic inputs for:

- the initialization dependency/Jacobian map at `phi=(0,0,0)` and positive
  dyadic `q_scale`;
- floor-free log-normalization JVP/VJP, additive-constant invariance, and
  likelihood-increment aggregation using directly supplied dyadic
  probabilities;
- the `p=5` direction-axis map through the frozen Phase 3/4 exact Contract E
  charts; and
- `B=2` sum-over-time and mean-over-batch objective/score aggregation.

Every microcertificate must pass exact executed equality. For the actual frozen
tiny LGSSM graph, compare the manual analytic JVP with TensorFlow
`ForwardAccumulator` applied to the exact same private primal core and identity
basis. The primary derivative gate is predeclared as bitwise equality (`0 ULP`)
for every per-batch and aggregated component. This is intentionally strict and
does not pretend that an unproved forward-error tolerance is rigorous.

If `0 ULP` cannot be achieved after repair, Phase 5 closes
`CANONICAL_GRADIENT_UNCERTIFIED_NUMERICAL_IDENTITY_BLOCKED`. The FD ladder and
`0.05*sqrt(p)` screen remain explanatory and cannot override that result.
Phase 6 fail-closed historical cleanup may proceed independently for safety,
but no canonical gradient artifact, admission, or later scientific claim may
advance from the blocked Phase 5 result.

The exact initialization dependency map is

```text
ds0_j/dphi_k = 0                                  for k != j
ds0_j/dphi_j = q_scale*phi_j/(1-phi_j^2)^(3/2)
ds0_j/dq     = 1/sqrt(1-phi_j^2)
ds0_j/dr     = 0.
```

Initialization tests require nonzero values only where this map and the frozen
chart make them nonzero and require exact structural zeros elsewhere. A
separate overall-primal perturbation test may cover all five parameters, but it
does not prove initialization wiring.

## Baseline Ladder

| Arm | Role | Promotion role |
| --- | --- | --- |
| Historical raw compact/value pair | Demonstrates the graph-identity and reset defect | Negative comparator only |
| Canonical no-reset one-graph arm | Localizes reset contribution | Explanatory diagnostic only |
| Canonical quotient plus Contract E one-graph arm | Phase 5 candidate | Primary engineering target |
| TensorFlow autodiff of the same tiny callable | Independent derivative comparator on CPU float64 | Local repair/veto diagnostic, not production promotion |

Kalman enters only as an explanatory value/gradient field on tiny LGSSM
fixtures. Phase 8 owns any equivalence decision.

## Skeptical Plan Audit

Decision: `PASS_FOR_BOUNDED_LOCAL_IMPLEMENTATION_AFTER_HANDOFF_REVIEW`.

| Risk | Finding and control |
| --- | --- |
| Wrong baseline | Same finite callable, its TensorFlow autodiff, and same-callable FD are the Phase 5 comparators; Kalman is not the derivative identity oracle here. |
| Proxy promotion | Center identity and tiny FD establish graph wiring only; they do not establish Kalman agreement or production admission. |
| Hidden graph duplication | FD executes the exact concrete value-and-score function and is forbidden from a value-only trace. |
| Missing initialization dependence | Initial particles are reconstructed from fixed noise inside the callable before primal/tangent initialization. |
| Coordinate error | Log normalization is expressed directly without a floor; JVP and the unique VJP projection each occur once. |
| Invalid reset hidden by particles-only wrapper | The graph calls diagnostic-returning cores and aggregates `valid_chart` as a hard output/veto. |
| Prepared-input drift | Residual/ridge tensors and finite settings are hashed and compared at center/endpoints. |
| Historical authority leakage | New canonical code lives under `bayesfilter/`; historical benchmark entry points remain unadmitted comparators. |
| Arbitrary FD threshold | Step ladder and branch identity precede output; `0.05*sqrt(p)` remains heuristic-only. |
| Environment mismatch | Begin CPU float64 with GPU hidden. Trusted GPU/XLA full-time work requires a later reviewed command and is not implied by Phase 4 one-step feasibility. |
| Candidate versus direction failure | A graph or derivative mismatch triggers repair; it does not reject Contract E unless the same finite target cannot be represented. |

No material wrong baseline, proxy criterion, missing stop condition, stale
context, or artifact mismatch remains in the draft. Review must still confirm
feasibility before code changes.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does one owned finite graph compute the Contract E LGSSM primal and its analytic JVP, with FD invoking that identical graph? |
| Exact comparator | Repeated same-concrete-function center calls; tensor hashes; tiny CPU-float64 TensorFlow autodiff; same-callable FD ladder; local JVP/VJP identities. |
| Primary engineering criterion | Bitwise center identity, valid fixed branches/charts, candidate initialization inside graph, exact normalization/axis/aggregation microcertificates, and `0 ULP` same-primal-core analytic-JVP versus forward-autodiff equality pass. |
| Hard vetoes | Any separate value graph, center/hash mismatch, missing initialization tangent, probability floor, double normalization, invalid chart, raw reset, nonfinite output, prepared-input drift, or wrong callable. |
| Promotion blockers | General derivative error bound, Phase 4 numerical blockers, full-time GPU feasibility, Kalman equivalence design, and all later-phase gates. |
| Repair triggers | Same-callable autodiff/FD discrepancy, active-set change, compile/resource failure, or review finding with a valid target. |
| Explanatory only | FD relative error, step-ladder shape, Kalman differences, timings, memory, residuals, and condition proxies absent a justified promotion design. |
| Not concluded | Kalman equivalence, stochastic accuracy, HMC/default/admission, nonlinear, leaderboard, or release readiness. |

## Required Artifacts

- Owned canonical LGSSM graph module and focused tests.
- Frozen tiny fixed-noise prepared-input fixture before outputs.
- Structured center-identity, branch, initialization, normalization, JVP/
  autodiff, and FD-ladder artifact.
- Source/prepared-input hashes and CPU-hidden run manifest.
- Phase 5 result or blocker with decision/inference tables and post-run red team.
- Phase 6 historical-route cleanup subplan.
- Updated master, ledger, and stop handoff.

## Required Checks, Tests, And Reviews

1. Inventory and hash every historical primitive proposed for extraction; do
   not import `docs/benchmarks` from the owned module.
2. Freeze `B=1`, small `N`, `T=1` and `T>1` fixtures with fixed noises, reset
   masks, residual designs, ridge, and finite transport settings before output.
3. Unit-test direct log-normalization primal/JVP/VJP identities, additive-
   constant invariance, and absence of floors/double projection.
4. Test candidate-dependent initialization against the frozen dependency map:
   accurate/nonzero derivatives only for `phi_j` and `q_scale` where declared,
   and structural zeros for cross-`phi` and `r_scale` entries. Test overall
   primal sensitivity to all parameters separately.
5. Test active and inactive reset branches, chart aggregation, invalid-chart
   veto, and exact prepared-input identity.
6. Run exact outer-wiring microcertificates, then compare one-step and short-
   prefix analytic JVP with TensorFlow forward autodiff of the exact same private
   primal core on CPU float64. Require bitwise (`0 ULP`) equality for Phase 5
   success; otherwise repair or close the canonical gradient uncertified.
7. Run the same-concrete-function FD step ladder only after center/branch/hash
   identity passes. Persist every endpoint primal and chart status.
8. Source-audit no NumPy, historical raw reset, probability floor, adaptive
   ridge, fresh randomness, value-only FD wrapper, or duplicate normalization.
9. Run deliberate CPU-XLA tiny wrappers and Phase 0-4 compatibility suites.
10. Run Python compilation, JSON/hash checks, and scoped `git diff --check`.
11. Obtain bounded fresh-Codex implementation and result/Phase 6 handoff
    reviews; repair up to five material rounds per blocker.

## Forbidden Claims And Actions

- Do not call algebraically equivalent or separately compiled functions the
  same scalar.
- Do not freeze center particles outside the callable while differentiating
  their candidate dependence.
- Do not use `log(max(weights,floor))`, probability clipping, or a hidden floor.
- Do not omit or apply log-normalization projection twice.
- Do not call the particles-only Phase 4 wrapper as an admission gate.
- Do not fall back to raw barycentric reset on an invalid Contract E chart.
- Do not make ridge/residual design candidate-dependent or draw new randomness.
- Do not import a benchmark module into the owned `bayesfilter` graph module.
- Do not register a route, admit v2 artifacts, run HMC, claim Kalman agreement,
  or regenerate leaderboard rows.

## Exact Next-Phase Handoff Conditions

### Certified Progression Handoff

Phase 6 may receive a certified canonical-gradient handoff only if:

- one owned callable literally returns the primal used by its analytic JVP and
  every FD center/endpoint;
- repeated center outputs and score/FD center artifacts are bitwise identical;
- candidate-dependent initialization is inside that callable for primal and
  tangent;
- corrected-logit normalization is floor-free and differentiated exactly once;
- active resets use row quotient plus Contract E-Chol and consume `valid_chart`;
- fixed residual/ridge/schedule/mask identities match at all calls;
- exact outer-wiring certificates and `0 ULP` same-primal-core analytic-JVP/
  forward-autodiff equality pass; an inconclusive derivative result cannot hand
  a certified canonical gradient to later phases;
- local CPU float64, same-callable FD, source, compatibility, and CPU-XLA checks
  pass their frozen engineering contracts;
- all Phase 4 and later numerical/scientific blockers remain explicit; and
- the Phase 5 result and Phase 6 cleanup subplan pass bounded handoff review.

Only this branch may later seek Phase 7 documentation reconciliation, Phase 8
scientific/Kalman evidence, canonical artifact admission, HMC use, or nonlinear/
leaderboard migration.

### Blocked Cleanup-Only Handoff

If the exact outer certificates pass but the `0 ULP` full tiny-graph derivative
gate remains nonzero after the repair loop, close Phase 5 with
`CANONICAL_GRADIENT_UNCERTIFIED_NUMERICAL_IDENTITY_BLOCKED`. Phase 6 may then
execute only fail-closed mechanical cleanup of historical raw/default/admission
reachability under a dedicated cleanup-only subplan and result. That handoff
must carry:

- no canonical gradient, score, v2 admission, default, HMC, Kalman, nonlinear,
  leaderboard, or release authority;
- the exact failed components/ULP evidence and source hashes;
- continued empty production factory and historical fail-closed behavior; and
- a stop after cleanup, before Phase 7 or any scientific/admission progression,
  until a newly reviewed Phase 5 repair resumes and passes the certified
  handoff.

## Stop Conditions

Stop and write a blocker result if a literal single callable cannot expose both
primal and analytic JVP, the target requires a historical raw reset, fixed
prepared inputs cannot be identified, every predeclared tiny chart is invalid,
a concurrent in-scope edit appears, a new scientific threshold/target decision
requires owner authority, five material repair rounds fail for the same
blocker, or the eight-hour campaign budget expires. Local mismatch and compile
failure are repair triggers first.

## Phase-End Protocol

1. Run CPU-hidden graph-identity, normalization, initialization, chart, JVP/
   autodiff, and same-callable FD checks.
2. Persist structured evidence and any same-phase repair records.
3. Write the Phase 5 result/blocker and manifest.
4. Draft Phase 6 cleanup subplan.
5. Review the result/handoff and repair if material.
6. Update master, ledger, and stop handoff.
7. Advance only on engineering evidence; preserve every scientific blocker.
