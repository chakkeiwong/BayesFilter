# Contract E--TP Phase 6 Zhao--Cui Comparator Certification Plan

metadata_date: 2026-07-15
status: READY_AFTER_SKEPTICAL_AUDIT
master_plan: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`
execution_target: explicit CPU-hidden TensorFlow float64 correctness ladder; trusted GPU/XLA deferred to Phase 9
campaign_budget: at most 24 CPU core-hours and three attempts per scalar row in this phase
output_root: `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase6_zhao_cui_comparators/`

## Phase Objective

Certify a Zhao--Cui comparator only when the executed finite program preserves
the adjacent-state squared-TT recursion of Zhao and Cui (2024), differentiates
the same executed scalar, and matches the frozen BayesFilter row target.  If a
row lacks such a route, emit an explicit unavailable or blocked record rather
than substituting a Kalman oracle, a retained-grid algorithm, a scoped
complete-data score, or a one-axis squared-density extension.

This phase separates two questions:

1. **Engineering correctness:** does a reported score equal the derivative of
   its own finite value program under the frozen branch?
2. **Comparator eligibility:** does that finite program retain every
   paper/author coordinate and operation needed by the claimed comparator, or
   must it be classified as an extension?

Passing the first question does not imply the second.

## Entry Conditions Inherited From Phase 5

- Contract E--Chol remains the only canonical LEDH reset route.
- Contract E--TP remains experimental.
- LGSSM Contract E--TP passed the exact-continuation `T=2,10,50` ladder.
- Actual SV, KSC-SV, and predator--prey have valid Contract E--TP short-prefix
  diagnostics.
- Generalized SV is a row-specific negative result for the tested progressive
  continuation feature family.  Its comparator may still be audited, but no
  `T=100` Contract E--TP run is allowed.
- Austria SIR observed-data comparison is blocked because the clipped simulator
  and Gaussian transition density define different probability measures.
- P90/P91 SIR evidence remains valid only for its local complete-data component
  and value-bridge scope.

## Research Intent Ledger

| Role | Binding statement |
| --- | --- |
| Main question | Which current or repaired fixed branches are legitimate same-target Zhao--Cui comparators with correct own-scalar scores? |
| Mechanism under test | Fixed basis, rank, quadrature, sweep order, and seeds applied to the paper's adjacent-state squared-TT approximation and marginalization recursion. |
| Expected failure mode | A route called `zhaocui` may fit only the already marginalized current-state density, omit the previous-state axis, or report a derivative of a different branch/scalar. |
| Promotion criterion | Target identity, source-route structure, realized branch identity, finite value/score, and all-coordinate own-scalar FD pass at `T=1,2,10`; no forbidden substitution occurs. |
| Promotion veto | Wrong target, missing previous-state marginalization, oracle alias, generic retained-grid route, scoped component promoted to observed-data score, nonfinite result, or own-scalar derivative failure. |
| Continuation veto | Shared fitting/marginalization algebra is invalid; branch identity cannot be proved; or the repaired route fails `T=1` after bounded root-cause repair.  A row-specific approximation failure is not a campaign veto. |
| Repair trigger | FD failure, fit/rank/condition failure, marginal mass failure, target timing mismatch, or mismatch between the paper route and executed variables. |
| Explanatory diagnostics | Values, scores, per-time increments, fit residuals, marginal mass, condition numbers, and cross-method gaps. |
| Forbidden conclusion | No source-faithful adaptive TT-cross claim, cross-method equivalence claim, superiority ranking, HMC readiness, default readiness, or full-leaderboard completeness. |

## Checked Source And Claim-Support Ledger

The literature-audit scope is deliberately bounded to the technical route
needed for implementation.  Live citation metadata and forward snowballing are
not needed to decide whether local code implements Algorithms 1--2; their
absence blocks a complete literature survey, not this source-route audit.

| Claim | Support class | Checked anchor | Allowed conclusion |
| --- | --- | --- | --- |
| The recursive posterior update uses the previous marginal, transition density, and observation density | `PRIMARY_TECHNICAL_SUPPORT` | Zhao--Cui Eq. (9)--(12), local text lines 339--520 | Defines the adjacent-state target and marginalization role. |
| The nonnegative route fits the square root of the adjacent-state target and marginalizes the previous state | `PRIMARY_TECHNICAL_SUPPORT` | Zhao--Cui Algorithm 2 and Eq. (15)--(16), local text lines 693--718 | Required comparator structure. |
| The squared-TT marginal is computed by paired-core mass contractions | `PRIMARY_TECHNICAL_SUPPORT` plus `IMPLEMENTATION_EVIDENCE` | Zhao--Cui Proposition 2; `bayesfilter/highdim/squared_tt.py::normalized_marginal_density_values` | Repository primitive implements the required fixed marginal contraction. |
| The author implementation reuses the previous marginal, builds TTSIRT, accumulates its normalizer, and carries it forward | `IMPLEMENTATION_EVIDENCE` | `third_party/audit/tensor-ssm-paper-demo/models/full_sol.m:72`, `:101`, `:124`, `:132` | Confirms the practical author route. |
| BayesFilter one-axis scalar route is not the adjacent-state Algorithm-2 route | `IMPLEMENTATION_EVIDENCE` | `bayesfilter/highdim/filtering.py:917`, `:944`, and explicit nonclaim `:977` | It may be retained as an extension diagnostic, but not certified as a Zhao--Cui source-route comparator. |
| The repaired scalar route treats `theta` as an external query rather than a TT coordinate | `IMPLEMENTATION_EVIDENCE` compared with `PRIMARY_TECHNICAL_SUPPORT` | repaired route; Zhao--Cui Eq. (15)--(16) includes `(x_t,theta,x_{t-1})` | Even with correct adjacent-state marginalization, it is `extension_or_invention`, not a fixed adaptation of the paper's parameter-learning route. |

Publication status is the local JMLR 2024 full text.  Retraction/erratum and
live citation metadata were not queried in this phase.  That omission is
recorded as a literature-coverage gap and is not used to imply source validity.

## Mathematical Comparator Contract

Let the frozen row use observations `y_0,...,y_{T-1}` and parameter `theta`.
BayesFilter's synthetic SV rows observe the simulated initial state, so the
first update is

\[
  q_0(x_0;\theta)
  =p_0(x_0;\theta)g_0(y_0\mid x_0;\theta).
\]

For `t>=1`, let `\widehat p_{t-1}` be the normalized marginal carried by the
previous fitted squared TT.  The required adjacent-state target is

\[
 q_t(x_t,x_{t-1};\theta)
 =\widehat p_{t-1}(x_{t-1};\theta)
   f_t(x_t\mid x_{t-1};\theta)
   g_t(y_t\mid x_t;\theta).
\]

A fixed design approximates its square root by `h_t` and defines

\[
  \widehat q_t(x_t,x_{t-1};\theta)
  =h_t(x_t,x_{t-1};\theta)^2
   +\tau_t\lambda_t(x_t,x_{t-1}).
\]

The finite-program evidence increment and carried density are

\[
  \widehat Z_t(\theta)=\iint \widehat q_t\,dx_tdx_{t-1},
  \qquad
  \widehat p_t(x_t;\theta)
  =\frac{\int \widehat q_t(x_t,x_{t-1};\theta)dx_{t-1}}
         {\widehat Z_t(\theta)}.
\]

The first step uses the analogous one-axis definitions.  With a stable target
shift `c_t(theta)`, the stored increment is

\[
  \widehat\ell_t(\theta)
  =\log \widehat Z_t^{\mathrm{scaled}}(\theta)+c_t(\theta),
  \qquad
  \widehat\ell_T=\sum_t\widehat\ell_t.
\]

The claimed score is the total derivative

\[
  \nabla_\theta\widehat\ell_T
  =\sum_t\left[
    \frac{\nabla_\theta\widehat Z_t^{\mathrm{scaled}}}
         {\widehat Z_t^{\mathrm{scaled}}}
    +\nabla_\theta c_t
  \right],
\]

where the derivative of `q_t` includes the derivative of the carried previous
marginal.  Omitting that term is a partial derivative and is wrong relative to
the total-score claim.

### Proposition 1: Required structural equivalence

For `t>=1`, a route that fits only a one-axis function of `x_t` after separately
integrating the transition against a retained grid does not execute the finite
program above and is not a fixed adaptation of Zhao--Cui Algorithm 2.

**Proof.** Algorithm 2(b) approximates a function whose arguments include both
`x_t` and `x_{t-1}`; Algorithm 2(c) then obtains the new marginal by integrating
the fitted squared TT over `x_{t-1}`.  A one-axis fit has no previous-state core
and therefore cannot perform that fitted-object marginalization.  It may target
a numerically related predictive density, but its approximation and
marginalization operators occur in a different order.  Since projection and
marginalization do not generally commute, the programs are not equal. `QED`

### Proposition 2: Own-scalar derivative necessity

Suppose all basis functions, ranks, quadrature points, sweep order, ridge,
coordinate maps, seeds, and runtime branches are frozen in a neighborhood of
`theta`.  If the score implementation differentiates every TensorFlow operation
of the finite program above, including the previous marginal, then its result is
the total derivative of that finite scalar wherever the fixed linear solves are
nonsingular.

**Proof.** Under the assumptions, each target evaluation, fixed least-squares
solve, paired-core mass contraction, normalization, and logarithm is a
differentiable finite-dimensional map.  Composition and the chain rule give the
displayed total derivative.  The previous marginal is an input to `q_t`, so its
derivative appears through that composition.  Nonsingularity makes each fixed
solve locally differentiable. `QED`

This proposition proves equality to the finite program, not accuracy relative
to the exact filter or the adaptive author implementation.

## Route Classification And Execution Matrix

| Row | Current route verdict | Phase 6 action | Eligible output |
| --- | --- | --- | --- |
| LGSSM `d=3,p=5` | Kalman adapter aliases the oracle; no real fixed adjacent-state TT | Do not execute an alias. Record unavailable pending a multidimensional fixed source-route implementation. | `zhaocui_comparator_unavailable` |
| Actual transformed SV | Repaired route restores adjacent-state axes but keeps `theta` external to the TT | Test the fixed-parameter adjacent-state extension at `T=1,2,10`. | certified `extension_or_invention`; not a Zhao--Cui source comparator |
| KSC-SV | Observation model is a BayesFilter mixture adaptation and current route is one-axis | Run only after actual-SV core passes; classify as `extension_or_invention` even with repaired adjacent-state core. | certified extension, not source-faithful |
| Generalized SV | Model is not an author-source operation and current Contract E--TP feature family failed | Own-scalar diagnostic only after shared core passes; no `T=100`. | certified extension diagnostic or row-specific negative result |
| Predator--prey | Current helper uses forbidden generic retained-grid route | Do not execute it. Record unavailable until a fixed-variant adjacent-state source route exists. | `zhaocui_comparator_unavailable` |
| Austria SIR observed-data | Target measure is unresolved and total derivative owners are missing | Preserve P90/P91 component evidence; do not execute observed-data comparison. | `blocked_target_measure_mismatch` |

## Required Artifacts

- this plan and a post-audit amendment if its mathematical contract changes;
- a Phase 6 source/route classification JSON;
- a TensorFlow scalar adjacent-state fixed squared-TT implementation;
- focused unit tests for target construction, Proposition-2 marginalization,
  total derivatives, and branch-identity failure;
- one structured result per attempted scalar row and one explicit unavailable
  record per non-executable row;
- run manifest with Git commit, command, conda environment, CPU/GPU status,
  seeds, wall time, artifacts, plan, and result;
- Phase 6 result/close record and Phase 7 handoff.

## Required Checks And Tests

1. Primitive two-axis target values match direct TensorFlow evaluation.
2. Squared-TT normalizer matches an independent dense quadrature at a tiny rung.
3. The carried marginal integrates to one and matches direct integration of the
   same fitted squared TT.
4. `T=1` repaired and legacy routes coincide because no adjacent transition has
   yet occurred, up to a conditioning/roundoff bound.
5. At `T=2`, the repaired route proves the previous-state axis is present and
   integrated; a deliberate one-axis substitution fails the identity gate.
6. TensorFlow total gradients, analytic/JVP components when available, and
   central FD of the same scalar are reported for every parameter.
7. Every FD rung `(1e-2,3e-3,1e-3,3e-4)` is preserved.  For each parameter,
   report

   \[
     r_j(h)=\frac{|s_j-\mathrm{FD}_j(h)|}
                  {\max(|s_j|,|\mathrm{FD}_j(h)|,10^{-12})}.
   \]

   The owner-selected screen `max_j r_j <= 0.05*sqrt(p)` applies only to this
   own-scalar comparison.  For every parameter, at least two adjacent steps in
   the declared descending ladder must be finite, branch compatible, and below
   that same threshold.  If their errors are `r_large` and `r_small`, the
   smaller-step error must satisfy

   \[
     r_{\rm small}\leq r_{\rm large}
       \quad\hbox{or}\quad
     |r_{\rm small}-r_{\rm large}|\leq
       0.1\{0.05\sqrt p\}.
   \]

   Thus one favorable step cannot hide an unstable ladder, and no unrelated
   absolute-error tolerance is introduced.
8. The compatibility identity excludes `theta` and numerical target values but
   includes realized dimensions, basis, ranks, quadrature nodes, sweep order,
   ridge policy, coordinate maps, seeds, observation count, target ids, and
   executed step structure.  Plus/minus/base identities must be independently
   derived.  Caller self-attestation is forbidden.
9. Cross-method value and score gaps are descriptive only because no justified
   row-specific equivalence margin is available.

## Evidence Contract

| Item | Contract |
| --- | --- |
| Scientific question | Does each admitted comparator execute the declared same-target adjacent-state fixed squared-TT program and differentiate its own scalar correctly? |
| Exact baseline | Direct dense quadrature of the same finite target at primitive/tiny rungs; model-specific exact/reference filters remain separate Phase 7 comparators. |
| Primary pass criterion | Structural route identity plus all-parameter own-scalar derivative pass at `T=1,2,10`. |
| Hard vetoes | Wrong target or timing; missing previous-state axis/marginalization; invalid branch identity; nonfinite; rank/solve failure; FD failure; oracle alias; retained-grid substitution; component-score promotion. |
| Explanatory only | Cross-method score gaps, runtime, fit residual trends below validity gates, and descriptive refinement behavior. |
| Not concluded | Exact filtering, equality with adaptive author code, cross-method equivalence, superiority, HMC/default/GPU readiness, or leaderboard completeness. |
| Preserved artifact | Versioned JSON results plus this phase result record. |

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| BayesFilter `y_0` initial-state timing | Dataset generator and target registry; binding target choice | Paper indexing silently drops or shifts an observation | `T=1` direct target identity |
| `(x_t,x_{t-1})` variable order | Zhao--Cui Algorithm 2 | Wrong axis marginalized | asymmetric two-axis marginal fixture |
| Existing FD step ladder | `FixedBranchDerivativeConfig`; reviewed diagnostic ladder | roundoff at small `h` or truncation at large `h` | preserve every rung and require stable adjacent window |
| `0.05*sqrt(p)` FD screen | owner decision; FD-only | mistaken cross-method tolerance | schema separates `own_scalar_fd` from `cross_method_gap` |
| Initial polynomial degree/rank/quadrature | inherited scalar comparator values; warm-start hypothesis only | underfit mistaken for algorithm failure | tiny dense fit/marginal residual and one-factor refinement |
| Fixed ALS branch with external `theta` | project extension | differs from adaptive author TT-cross and removes the parameter TT coordinate | classification remains `extension_or_invention`, never `fixed_hmc_adaptation` or `source_faithful` |
| Defensive `tau=0` at first diagnostic | current scalar comparator baseline | poor tail support or singular transport | density/mass/positivity check; later nonzero tau is a declared refinement, not silent repair |

## Skeptical Pre-Execution Audit

status: `PASS_AFTER_MATERIAL_ROUTE_AND_FD_IDENTITY_REPAIR`

The first Phase 6 concept was unsafe for two newly verified reasons:

1. The current scalar route fits the already marginalized current-state density
   and explicitly disclaims integrated-axis marginalization.  Calling it a
   fixed Zhao--Cui route would be wrong relative to Algorithm 2.
2. Its scalar FD path assigns the base compatibility hash to plus and minus
   runs instead of independently deriving their realized identities.  This
   cannot prove a frozen branch.

The execution order is therefore repaired:

1. emit the route-classification ledger and explicit unavailable rows;
2. repair independent scalar FD compatibility identities;
3. implement/test the two-axis adjacent-state value recursion;
4. establish its total derivative first by TensorFlow autodiff and same-scalar
   FD; manual/JVP decomposition is explanatory until separately proved;
5. run actual SV at `T=1,2,10`;
6. only if the shared core passes, run KSC and generalized-SV extensions at
   `T=1,2,10`;
7. stop each failed row at its smallest discriminating prefix and repair only
   the identified cause.

The audit also rejects a full-horizon run in this phase: it would spend compute
without answering source-route correctness, and full horizons belong to Phase
9 after Phase 7/8 scientific comparisons.

## Forbidden Claims And Actions

- Do not label any local fixed branch `source_faithful` or an adaptive
  TT-cross/SIRT reproduction.
- Do not certify the legacy one-axis scalar route as Zhao--Cui Algorithm 2.
- Do not use Kalman as the LGSSM Zhao--Cui cell.
- Do not execute or admit the generic multistate retained-grid route.
- Do not treat P90/P91 SIR component evidence as an observed-data score.
- Do not select the SIR clipped or Gaussian target without a new human-level
  scientific target decision.
- Do not use `0.05*sqrt(p)` for cross-method agreement.
- Do not hide an unstable FD ladder by reporting only its best step.
- Do not clip weights, stop feature gradients, switch active sets at runtime,
  or silently change basis/rank/quadrature after seeing audit results.
- Do not claim that an extension comparator closes a Zhao--Cui
  source-faithfulness gap.

## Exact Next-Phase Handoff Conditions

Phase 7 may start only with a row ledger containing exactly one of:

- `certified_extension_or_invention`: same engineering gates passed but model or
  route is outside author source;
- `zhaocui_comparator_unavailable`: no admissible implementation exists;
- `blocked_<reason>`: a target, derivative, or validity condition prevents a
  meaningful comparator.

Certified extensions may enter Phase 7 only as extension diagnostics; they do
not satisfy the requested Zhao--Cui source-comparator column.  A Phase 6 result
record must list exact commands, artifacts, failures/repairs, wall time,
remaining budget, and nonclaims.

## Stop Conditions

- Stop the whole phase if the adjacent-state target or Proposition-2 marginal
  primitive is mathematically wrong, artifacts are corrupted, or branch
  identity cannot be made non-self-attesting.
- Stop one row, not the campaign, when its target is invalid, its comparator is
  unavailable, or bounded refinement cannot pass `T=1,2,10`.
- Stop and request direction before changing the scientific target, model
  measure, parameterization, hardware class, campaign budget, or public/default
  policy.
- Do not stop merely because a candidate comparator fails; record the negative
  result and continue independently valid rows.
