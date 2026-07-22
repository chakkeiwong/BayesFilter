# Contract E--TP All-Model Implementation And Gradient-Comparison Plan

metadata_date: 2026-07-15
status: COMPLETE_WITH_NEGATIVE_RESULTS_AND_BLOCKERS
program_id: contract-e-tp-all-model-gradient-comparison
algorithm_id: contract_e_tp_experimental_v1
supervisor: Codex
campaign_authorization: owner requested execution on 2026-07-15
confirmed_compute_cap: 128 CPU core-hours, 64 trusted GPU-hours, three full-horizon attempts per model

terminal_result: `docs/plans/bayesfilter-contract-e-tp-phase10-terminal-synthesis-result-2026-07-15.md`
reset_memo: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-reset-memo-2026-07-15.md`

Completion note, 2026-07-15: all frozen rows and the structural extension are
classified as pass, negative result, or blocked in the terminal result. Program
completion means the planned questions have controlling evidence; it does not
mean every model passed, a complete leaderboard exists, or Contract E--TP is
canonical, default-ready, or HMC-ready.

## Outcome Sought

Implement the score-aware teacher-projection reset described in
`docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`, integrate it into one
finite LEDH value-and-total-score program, and test it on every executable
observed-data model family in the current high-dimensional leaderboard suite.
For each model, compare value and score against the strongest same-target
reference available and against a certified fixed-variant Zhao--Cui route where
one exists or can be implemented.

The central scientific question is:

> Does Contract E--TP preserve the value and total score of its declared finite
> teacher locally, remain recursively stable, and approach the same
> observed-data likelihood gradient as independently constructed references and
> fixed-variant Zhao--Cui approximations?

This is a research and implementation program. It does not authorize changing
the canonical Contract E--Chol policy, adding Contract E--TP to a leaderboard,
or claiming HMC readiness.

## Binding Decisions

1. Contract E--TP is a new experimental algorithm. Its initial route identity is
   `contract_e_tp_experimental_v1`; it is not an alias, revision, or fallback for
   `contract_e_chol_v1`.
2. `contract_e_chol_v1` remains the only canonical reset under repository
   policy. Contract E--TP artifacts must fail canonical, default, leaderboard,
   and HMC admission until a separate owner decision and evidence program.
3. The scalar target is the observed-data log likelihood produced by one fixed
   finite program. Its score is the total derivative of that same scalar. The
   teacher construction, feature reduction, chart solve, carried nonuniform
   weights, later LEDH steps, and all parameter-dependent preparation inside
   the declared target must be differentiated.
4. Zhao--Cui is a comparator, not an oracle. A route may be called
   `source_faithful` only after both paper and author-code anchors are checked.
   Frozen rank, basis, samples, pivots, and schedules are
   `fixed_hmc_adaptation`; generalized-model operations absent from the source
   are `extension_or_invention`.
5. The generic all-axes retained-grid functions
   `multistate_nonlinear_fixed_design_tt_value_path` and
   `multistate_nonlinear_fixed_design_tt_score_path` are diagnostic/historical
   only and may not support the primary comparison.
6. No fixed percentage defines cross-method gradient agreement. In particular,
   the earlier `0.05*sqrt(p)` screen belongs only to individual-direction
   same-scalar finite-difference checking. It is not a Contract E--TP versus
   Zhao--Cui equivalence margin.
7. Every nonlinear model gets a target-specific feature and preparation
   protocol. The seven-feature 2D LGSSM witness is a reference fixture, not a
   transferable default.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can the local finite-teacher projection preserve selected filtering features and their total tangents, and can the recursive method produce value/score estimates consistent with independent references? |
| Mechanism under test | Positive compression of a parent-by-innovation teacher through a fixed differentiable chart, followed by propagation of nonuniform weights through the next LEDH step. |
| Expected failure modes | Infeasible or nonpositive chart; feature-rank loss; omitted teacher/weight tangent; wrong time order; structural support violation; feature insufficiency; recursive drift; Zhao--Cui target mismatch; TT rank/fit error; excessive `N_parent*N_innovation*T` cost. |
| Promotion criterion | Per-model engineering and numerical gates pass; oracle/reference gates pass where available; the top two refinement rungs support a stable same-target result; and any claim of cross-method equivalence passes the predeclared simultaneous equivalence analysis. |
| Promotion veto | Nonfinite value/score, same-program AD/JVP/VJP/FD failure, target mismatch, feature residual, rank loss, nonpositive weight, structural-support failure, changed active chart, unaccounted randomness, or inadmissible Zhao--Cui route. |
| Continuation veto | Invalid model target/data, corrupted or incomplete artifacts, inability to differentiate the executed scalar, no feasible fixed chart on the declared parameter region, unavailable required comparator with no valid implementation path, or exhausted campaign budget. A candidate accuracy failure alone is not a continuation veto. |
| Repair trigger | Any localized implementation, graph, precision, chart-selection, feature-design, or comparator-wiring failure whose repair preserves the target and campaign budget. |
| Explanatory diagnostics | Cosine similarity, norm ratio, sign agreement, state moments, chart condition number, active-set margin, runtime, memory, and one-seed differences. |
| Forbidden conclusion | No exact nonlinear filtering, method superiority, production/default readiness, HMC readiness, NAWM readiness, or complete leaderboard claim follows from this program alone. |

## Scope And Comparator Gap Registry

The primary suite is the six executable observed-data families represented in
the current leaderboard. SIR d=18 already has a clean-room fixed-variant
Zhao--Cui implementation. Its P90/P91 evidence must be reused rather than
rebuilt: the existing route includes the 36-dimensional adjacent target for an
18-state model, fixed TTSIRT fitting/transport, deterministic resampling,
sequential retained objects, an author-formula value bridge, and a
three-parameter local complete-data score API. The parameterized SIR
leaderboard row remains scoped because the existing score omits the total
derivative through the previous marginal and fixed-TTSIRT proposal/transport;
that evidence boundary does not make the fixed-variant implementation itself
missing. DSGE/NAWM is a structural extension because BayesFilter currently has
adapter metadata gates, not a same-target executable Zhao--Cui likelihood
comparator.

| Family and row | State / parameter size | Contract E--TP role | Best independent reference | Zhao--Cui status before this program | Required action |
| --- | --- | --- | --- | --- | --- |
| LGSSM, `benchmark_lgssm_exact_oracle_m3_T50` | `d_x=3`, `p=5`, `T=50` | Primary oracle and recursion calibration | Exact differentiated Kalman filter | Current leaderboard Zhao--Cui cell is actually the Kalman oracle adapter, not TT execution | Implement a real fixed-variant TT route for this exact target or label the Zhao--Cui cell unavailable; never count the oracle twice |
| Actual non-Gaussian SV, `zhao_cui_sv_actual_nongaussian_T1000` | `d_x=1`, `p=2`, `T=1000` | Long-horizon nonlinear scalar test | Refined fixed-SGQF and high-accuracy teacher ladders; neither is exact | Scalar fixed-TT value/score adapter exists, but source/target/derivative identity must be recertified | Bind paper/source anchors, data, transforms, and same-scalar derivative before comparison |
| KSC-SV, `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | `d_x=1`, `p=2`, `T=1000` | Long-horizon mixture-surrogate test | Refined mixture enumeration/SGQF and teacher ladders | Scalar fixed-TT adapter exists; KSC target is an adaptation, not the paper's exact observation model | Classify honestly and certify same-target value/score |
| Generalized SV, `zhao_cui_generalized_sv_synthetic_from_estimated_values` | `d_x=1`, `p=3`, `T=1008` | Nonlinear observation/transition test | Native generalized-SV reference plus refined teacher | Current scalar fixed-design route is an extension; source-row comparator evidence is incomplete | Build or certify an `extension_or_invention` fixed-TT comparator; no source-faithful label |
| Predator--prey, `zhao_cui_predator_prey_T20` | `d_x=2`, `p=6`, `T=20` | Nonlinear RK4 with additive Gaussian real-plane state noise | Dense/SGQF lower rungs and refined teacher | Current leaderboard helper calls the forbidden generic retained-grid route; older tuned fixed-design evidence is not the required production source route | Implement a fixed-variant source-route comparator anchored to paper Section 6.4 and author `models/pp` code |
| Austria SIR, `zhao_cui_spatial_sir_austria_j9_T20` | `d_x=18`, `p=3` on the log-scale comparison surface, `T=20` | Full-rank high-dimensional source-semantics test (`Q=I_18`, initial covariance `I_18`) | Refined teacher, lower-dimensional closures, and model invariants after target binding | Existing clean-room fixed-TTSIRT source route: P90 matches the bound source scalar to author-formula replay, and P91 supplies the three-parameter local complete-data score, batched API, FD/identity diagnostics, and GPU/XLA evidence. The author clipped sampling push and Gaussian transition density define different measures. Full observed-data total-score certification also remains open because previous-marginal and fixed-TTSIRT proposal/transport derivative owners are explicitly blocked. | Preserve both source programs as distinct identities. Do not use structural completion. First retain P90/P91 component evidence at its scope; bind any full filtering comparison to either the Gaussian density target or a mathematically complete mixed clipped law before implementing missing derivative owners. Do not replace it with the retained-grid route. |
| Parameterized SIR component row | `d_x=18`, `p=3`, local component only | Existing fixed-variant derivative fixture | P91 direct local complete-data derivatives and score-identity artifact | Existing scoped fixed-TTSIRT component route, not a missing implementation | Preserve as a mandatory component-level test. Do not promote its conditioned local score to the marginal observed-data filtering score. |
| Structural deterministic fixture | Small stochastic subspace plus deterministic completion | Mandatory support proof before SIR/NAWM | Analytic structural identities | Not a Zhao--Cui row | Add as an exact engineering fixture |
| DSGE/NAWM extension | Model-owned dimension | Later structural stress | Model equations, structural completion identities, and any client oracle | No same-target production comparator | Implement only after a client adapter and target registry exist; report Zhao--Cui comparison unavailable unless a reviewed extension is built |

## Source Anchors And Claim Boundary

The Zhao--Cui technical spine used by this program is the inspected JMLR 2024
paper, especially equations (9)--(12), Algorithm 1, equation (13), Algorithm 2,
Proposition 2/equation (14), Algorithm 5/equations (30)--(35), Theorems 7--8,
and Sections 6.1--6.4. The pinned author source is commit
`80034dccb99eb1d86284a1839b4a12067d13b9da` under
`third_party/audit/zhao_cui_tensor_ssm_p10/source/`. The existing paper/code
crosswalk is
`docs/plans/bayesfilter-highdim-nonlinear-filtering-paper-first-scholarship-p10-zhao-cui-tt-paper-code-crosswalk-ledger-2026-05-30.md`.

The paper establishes recursive density approximations and filtering-density
error results. It does not prove that BayesFilter's frozen fixed-branch
likelihood derivative equals the exact likelihood score. That derivative is a
project construction and must pass same-scalar tests independently. The author
source is an audit source and must not be imported into `bayesfilter/` without a
separate license/clean-room decision.

For SIR d=18, the implementation anchors are already present in
`bayesfilter/highdim/source_route.py`. The controlling evidence is P90's
same-scalar value bridge and derivative-carry manifest plus P91's scoped score
contract and final decision. Those artifacts establish an implemented
fixed-variant route and a local component score. They also explicitly preserve:

```text
BLOCK_FIXED_TTSIRT_PREVIOUS_MARGINAL_DERIVATIVE_NOT_IMPLEMENTED
BLOCK_FIXED_TTSIRT_PROPOSAL_TRANSPORT_DERIVATIVE_NOT_IMPLEMENTED
```

Accordingly, this program distinguishes two SIR checks. The component check
evaluates the same conditioned transition/observation/prior scalar on shared
states and validates the existing three-parameter score. It is an adapter and
derivative diagnostic, not a comparison of filtering algorithms. The primary
algorithm comparison uses each method's marginal observed-data likelihood and
therefore requires the two blocked derivative owners to be implemented or the
SIR full-score comparison to remain explicitly blocked.

The source audit additionally found that author `st_process.mlx` clips
susceptible samples after Gaussian noise while author `transition.mlx` evaluates
an unclipped Gaussian density. These are different probability measures. The
Austria covariance is full rank, so the structural singular-dynamics interface
does not repair this mismatch. No Contract E--TP SIR observed-data score may be
admitted until the target identity selects one program and its proposal
accounting matches that program.

## Mathematical Program To Implement

At each reset time, with parent cloud `(x_i, a_i)` and fixed innovation rule
`(epsilon_j, b_j)`, execute:

```text
z_ij(theta)  = F_theta(x_i, epsilon_j)
u_ij(theta)  = a_i b_j c_t,ij(theta)
pi_ij(theta) = u_ij / sum_rs u_rs
b(theta)     = sum_ij pi_ij psi_theta(z_ij)
Phi_A(theta) = [psi_theta(z_a1), ..., psi_theta(z_aK)]
q(theta)     = solve(Phi_A(theta), b(theta))
```

The fixed feature vector starts from model-specific versions of:

```text
[mass, stochastic-coordinate mean, stochastic-coordinate second moments,
 next-step finite predictive contribution, reviewed model-specific features]
```

The next-step predictive contribution must be the exact fixed finite program
executed by the next LEDH step, including its transition, observation,
proposal-correction, fixed noise, and branch settings. It may not be replaced by
an analytic or Gaussian proxy unless that proxy is the declared scalar target.

The runtime chart uses fixed candidate indices. For a square chart,

```text
dq = solve(Phi_A, db - dPhi_A q).
```

If more anchors than features are needed, the only initial overcomplete option
is the fixed equality-constrained KKT projection already derived in the chapter.
Negative weights are a chart failure. Clipping, renormalizing after the solve,
switching bases based on the current parameter, or silently dropping a feature
changes the target and is forbidden.

The current increment is added before reset. The student cloud and `log(q)` are
then carried to the next step. Reverse mode must include direct dependence on
parents, incoming weights, innovation coordinates/weights when parameterized,
teacher correction factors, feature values, selected anchor locations, the
linear solve, and all later use of the carried cloud and weights.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Exact baseline | Differentiated Kalman likelihood for LGSSM; analytic structural identities for the singular fixture. |
| Numerical references | Dense teacher at small sizes; streaming teacher at larger sizes; model-specific refined SGQF/mixture/dense references only when target-compatible. |
| Algorithm ladder | No-reset LEDH diagnostic; canonical Contract E--Chol baseline; Contract E--TP; high-accuracy parent-by-innovation teacher; certified fixed-variant Zhao--Cui. Historical raw-barycentric and retained-grid routes are negative controls only. |
| Primary engineering gate | Dense/streaming primal, JVP, VJP, autodiff, and same-program FD tie out on small float64 fixtures; fixed chart and route identities fail closed. |
| Primary numerical gate | All increments, scores, weights, feature residuals, chart margins, and structural residuals are finite and within derived numerical error budgets. |
| Primary scientific gate | LGSSM agrees with the exact Kalman value and score under the frozen oracle design; nonlinear rows show reference/refinement convergence and, where claimed, pass the blinded equivalence analysis. |
| Hard vetoes | Target/data/coordinate mismatch; nonfinite; feature rank loss; nonpositive weight; chart switch; feature residual exceeding forward-error bound; structural-support violation; same-scalar derivative failure; inadmissible Zhao--Cui route; audit data used for preparation. |
| Statistical evidence | Paired data/evaluation points, disjoint preparation/validation/audit roles, per-replicate values and scores, simultaneous 95% intervals, and a recorded precision-stopping rule. |
| What is not concluded | A pass does not establish exact nonlinear filtering, superiority, posterior correctness, HMC readiness, canonical/default status, or NAWM readiness. |
| Artifact root | `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/<run_id>/` with unique run IDs and no overwrite. |

## Defining “Reasonably Match” Without Inventing A Threshold

Three different questions must remain separate.

1. **Derivative correctness within a method.** Compare AD/manual JVP/VJP to FD
   of the same finite scalar with common fixed randomness. The existing
   direction-FD policy may be used only here.
2. **Accuracy against an oracle or reference.** Compare each method separately
   to Kalman or another declared same-target reference. Report componentwise and
   standardized errors, not only a vector norm.
3. **Contract E--TP versus Zhao--Cui agreement.** Compare the two only after
   target, data, coordinates, horizon, initial law, and preparation roles match.

Before revealing method labels for a row, Phase 0 freezes parameter scales and
an equivalence rule. Parameter scales come from the declared unconstrained
coordinate map and model prior/reference scale, not from observed method
differences. Per-component equivalence margins must be derived from that row's
oracle error budget, reference-refinement error, or a downstream log-density
perturbation budget. If none can be justified, the row may report distances and
confidence intervals but cannot claim “reasonable agreement.”

For each comparison, emit:

```text
component difference:       g_TP,j - g_ZC,j
standardized component:     (g_TP,j - g_ZC,j) * parameter_scale_j
normalized RMS:             sqrt(mean_j(((g_TP,j-g_ZC,j)/score_scale_j)^2))
cosine agreement:           <g_TP,g_ZC> / (||g_TP|| ||g_ZC||)
norm ratio:                 ||g_TP|| / ||g_ZC||
per-time increment gap:     delta_g_TP,t,j - delta_g_ZC,t,j
```

Use paired bootstrap or paired Student intervals across independent
data/preparation replicates, with simultaneous component control selected
before results. A row is classified as:

- `equivalent_under_frozen_margin` only when every primary component's
  simultaneous interval lies inside its frozen equivalence interval and both
  methods pass their own reference/refinement gates;
- `different_under_frozen_margin` when an interval excludes the equivalence
  band or a directional hard veto fires;
- `unresolved_precision` when the interval is too wide;
- `comparison_invalid` when target, route, or derivative identity fails.

Cosine, sign, one-seed agreement, and descriptive mean gaps cannot produce the
equivalence label.

## Planned Implementation Surface

New owned modules:

- `bayesfilter/highdim/ledh_contract_e_tp_tf.py`: prepared chart dataclasses,
  dense and streaming teacher reductions, projection solve, reset result, JVP,
  and custom VJP.
- `bayesfilter/highdim/ledh_contract_e_tp_identity.py`: non-overridable
  experimental factory identity binding feature program, innovation rule,
  active indices, parameter region, scalar target, source closure, dtype, and
  branch settings.
- `bayesfilter/highdim/ledh_contract_e_tp_models.py`: protocol and registered
  model adapters; model mathematics remains in existing model modules.
- `bayesfilter/highdim/ledh_contract_e_tp_preparation.py`: offline chart search,
  train/validation/audit separation, fixed-region certificate, and serialization.
- `bayesfilter/highdim/ledh_contract_e_tp_artifact.py`: experimental schema,
  validation, diagnostics, and fail-closed non-admission status.

New runners and registries:

- `docs/benchmarks/configs/contract_e_tp_all_models_2026_07_15.json`:
  target/data/coordinate registry and per-row rung definitions.
- `docs/benchmarks/prepare_contract_e_tp_charts.py`: preparation-only chart and
  feature selection.
- `docs/benchmarks/run_contract_e_tp_model.py`: one model, one rung, one unique
  output directory.
- `docs/benchmarks/run_contract_e_tp_comparison.py`: artifact-only paired
  Contract E--TP/reference/Zhao--Cui analysis; it must not rerun methods.

Focused tests:

- `tests/highdim/test_ledh_contract_e_tp_primitives.py`
- `tests/highdim/test_ledh_contract_e_tp_derivatives.py`
- `tests/highdim/test_ledh_contract_e_tp_identity.py`
- `tests/highdim/test_ledh_contract_e_tp_lgssm.py`
- `tests/highdim/test_ledh_contract_e_tp_structural.py`
- `tests/highdim/test_ledh_contract_e_tp_model_adapters.py`
- `tests/highdim/test_ledh_contract_e_tp_zhaocui_comparators.py`
- `tests/highdim/test_ledh_contract_e_tp_artifacts.py`

Existing benchmark scripts remain unchanged until the experimental route passes
its own suite. No first implementation step may wire Contract E--TP into the
leaderboard or canonical artifact validators.

## Phase Plan

### Phase 0: Target, Source, And Statistical Freeze

**Objective.** Create the row registry before implementation or numerical
results can influence the comparison contract.

**Work.** For every row, hash observations and prepared inputs; record horizon,
initial law, transition/observation equations, parameter transforms and order,
value scalar, score coordinates, parameter region, feature candidates,
preparation/validation/audit seeds, comparator routes, source anchors, and route
classification. Copy the inspected Zhao--Cui paper into the project-local source
cache or record an explicit local-source blocker. Freeze the simultaneous
interval method and either a justified equivalence margin or
`descriptive_only_margin_unavailable` per row.

**Required artifacts.** Target registry JSON, source-support/claim ledger,
default-and-assumption ledger, comparison protocol, and Phase 0 result.

**Gate.** Every primary row has one unambiguous observed-data scalar. The LGSSM
oracle is not labeled Zhao--Cui. Predator--prey does not name the retained-grid
route. The SIR component row is excluded from primary completeness.

### Phase 1: Dense TensorFlow Contract E--TP Core

**Objective.** Reproduce the 2D witness in owned TensorFlow code and establish
the complete derivative of the finite projection.

**Work.** Implement dense teacher construction, normalized log weights, feature
matrix, square and overcomplete KKT charts, carried weights, JVP and VJP. Port
the existing 6561-to-7 witness as an independent NumPy/reference fixture only.
Test explicit `dq`, TensorFlow autodiff, reverse VJP, and common-random-number FD
for every parameter and random directions. Separate adjoints for incoming
parents, incoming weights, anchor positions, correction factors, and feature
parameters.

**Gate.** Float64 primal and derivative discrepancies are bounded by a
condition-number-based forward-error calculation, not a hand-picked decimal.
Mass, feature residual, positivity, and chart identity pass. Clipping and
runtime active-set selection are unreachable.

### Phase 2: Streaming Teacher And XLA Composition

**Objective.** Evaluate `N_parent*N_innovation` teachers without retaining a
dense candidate tensor and differentiate the same result.

**Work.** Stream teacher log-normalization and feature sums in two stable passes;
gather only frozen anchors; implement the transpose streaming reductions; trace
with `tf.function(jit_compile=True)`. Compare dense and streaming primal/JVP/VJP
over chunk sizes and permutations. Inspect the graph and measured memory.

**Gate.** No retained `O(N_parent*N_innovation*d_x)` or time-history tensor;
dense/streaming parity passes the derived rounding bound; XLA compiles; chunk
choice does not change the mathematical target.

### Phase 3: Recursive LGSSM Oracle Ladder

**Objective.** Prove that the implemented local mechanism survives recursive
LEDH use and identify where score drift enters.

**Work.** Run `T=1,2,5,50`, first on the documented 2D fixture and then on the
leaderboard `d_x=3,p=5` target. At every time step record current increment,
teacher/student predictive feature, its tangent, chart residual/margin,
incoming/outgoing weights, and cumulative score. Compare against exact Kalman
value and score and against Contract E--Chol/no-reset diagnostics. Audit center
and held-out parameter-region points.

**Gate.** Same-program derivatives pass; local retained-feature values and
tangents tie out; no chart changes occur; and the top refinement rungs satisfy
the frozen Kalman value/score criterion. A local tie-out with a failed cumulative
Kalman score is a scientific failure to diagnose, not a pass.

### Phase 4: Structural Singular-Dynamics Fixture

**Objective.** Establish the route needed by SIR and future NAWM-like models
without pretending a singular full-state covariance is positive definite.

**Work.** Construct the teacher in innovation coordinates, apply deterministic
completion to full state, choose anchors only from structural support, and
define features on the stochastic subspace plus declared observable
functionals. Verify completion identities and their derivatives. Compare with a
small all-stochastic control.

**Gate.** Exact structural residual and tangent identities pass; no artificial
full-rank jitter changes the model; every student point is a teacher candidate
on the declared support.

### Phase 5: Model-Specific Adapter And Preparation Protocols

**Objective.** Add actual SV, KSC-SV, generalized SV, predator--prey, and fixed
SIR without transferring unvalidated LGSSM defaults or conflating source
sampling and density programs.

**Work.** For each model, write a short preparation record that audits feature
set, innovation rule, chart size, parameter region, target scales, coordinate
transform, support map, capacity ladder, preparation budget, and held-out gate.
Use one-step then short-horizon then full-horizon rungs. Respect each target's
actual support: the additive-Gaussian predator--prey fixture has real-plane
support and must not be positivity-clipped; any genuinely constrained positive-
state model must generate teacher points on its declared support. Use Phase 4
structural coordinates only for a target with a proved singular stochastic
subspace. The Austria SIR fixture is full rank and instead requires a source
target/proposal semantics decision because clipping and Gaussian density differ.

**Gate.** Each adapter passes direct transition/observation derivative tests,
dense/stream teacher parity, chart-region audit, same-scalar AD/VJP/FD, and
short-horizon recursive tests before a full horizon is allowed. Failure of one
model blocks only that row unless it exposes a shared-core bug.

### Phase 6: Zhao--Cui Comparator Repair And Certification

Phase 5 handoff amendment, 2026-07-15: actual SV, KSC-SV, and predator--prey
have valid short-prefix Contract E--TP diagnostics. Generalized SV closes as a
row-specific negative result for the tested progressive continuation family and
must not proceed to `T=100`; its Zhao--Cui comparator may still be classified
and certified independently. Austria SIR is blocked on source target-measure
identity before an observed-data Contract E--TP route. These row-specific
outcomes do not block comparator certification for the valid rows.


**Objective.** Supply a legitimate same-target fixed-variant comparator for
each row or record a row-specific unavailable result.

**Work.** Implement the LGSSM fixed-TT route instead of reusing Kalman. Certify
actual-SV and KSC scalar adapters. Build/certify generalized-SV as an explicit
extension. Replace predator--prey retained-grid use with a clean-room
fixed-variant source-route implementation. For SIR, reuse the existing d=18
fixed-TTSIRT source route and its frozen identities. Run the P90/P91 local
component/value-bridge regressions first; do not rebuild or replace that route.
Then implement and test the two missing total-derivative owners for the previous
marginal and fixed-TTSIRT proposal/transport so that the cumulative
observed-data score differentiates the same executed value. If that extension
does not pass, retain the existing component evidence and mark only the SIR
full-filtering score comparison blocked. Bind every route to paper/source
anchors and a factory identity. Verify each score against FD of its own scalar
before any comparison.

**Gate.** No oracle alias, retained-grid route, scoped component, value/score
scalar mismatch, or unsupported source-faithfulness label enters the comparator
set. A missing comparator produces `zhaocui_comparator_unavailable`, not a
substitute algorithm.

For SIR specifically, this gate requires a two-row result:

- `sir_d18_fixed_variant_component_score`: must reuse and reproduce the P90/P91
  fixed-variant component evidence; it cannot support filtering equivalence.
- `sir_d18_fixed_variant_observed_data_total_score`: may enter the primary
  comparison only after the previous-marginal and proposal/transport derivative
  blockers close and same-scalar FD passes.

### Phase 7: Paired All-Model Comparison Ladder

Phase 6 handoff amendment, 2026-07-15: no current row has a Zhao--Cui
source-route parameter-learning comparator.  The repaired scalar route is a
fixed-parameter adjacent-state squared-TT `extension_or_invention` because it
fits `(x_t,x_{t-1})` at externally supplied `theta`, whereas Zhao--Cui
Algorithm 2 fits over `(x_t,theta,x_{t-1})`.  Contract E--Chol likewise has no
admissible same-target all-row artifacts in this campaign.  Phase 7 must expose
these cells as unavailable.  It may include the repaired scalar extension as a
separately labelled diagnostic, but it must not call that route Zhao--Cui or
silently substitute a historical retained-grid route.

**Objective.** Compare independently valid methods on identical scientific
targets.

**Work.** Execute, per row: exact/reference where available; high-accuracy
teacher; Contract E--Chol where an admissible artifact exists; Contract E--TP;
certified Zhao--Cui where an admissible source-route artifact exists; and the
fixed-parameter adjacent-state extension as a separately named diagnostic
where available. Use the same observations, evaluation points, horizon,
parameter chart, and initial law.
Where algorithms contain fixed random preparation, use paired role seeds when
the constructions admit it and otherwise report independent preparation
uncertainty. Preserve per-time score increments.

**Gate.** Fail closed on target-identity or derivative-identity mismatch.  When
no justified cross-method margin or replicated preparation ensemble exists,
emit `descriptive_only_margin_unavailable`, not an equivalence label.  Missing
methods must remain explicit unavailable cells. No row is ranked by runtime or
descriptive score gap. A failed candidate triggers feature/chart/refinement
diagnosis in Phase 8 unless a continuation veto fired.

### Phase 8: Repair, Refinement, And Sensitivity

**Objective.** Distinguish finite resolution, feature insufficiency, chart
failure, and genuine method disagreement.

**Work.** In this order, vary teacher resolution, student chart capacity,
look-ahead feature set, parameter-region width, and Zhao--Cui rank/fit budget.
Do not change multiple causes in the same diagnostic. Re-run the smallest failed
time prefix and parameter direction first. Use audit data only at the final
frozen candidate.

**Gate.** A repair is accepted only if it passes the original frozen target and
held-out criteria. If the best candidate remains different, record the negative
result; do not loosen the margin or relabel a proxy target.

### Phase 9: GPU/XLA Full-Horizon And Scaling Evidence

**Objective.** Determine whether the experimental algorithm is feasible on the
repository's default execution target.

**Work.** After correctness gates, run trusted GPU/XLA with float32/TF32 and an
FP64 reference subset. Use a ladder such as `(N_parent,N_innovation) =
(32,32),(64,64),(128,128)` only as an initial capacity hypothesis; Phase 0/5
must approve or replace it per model. Record compile time, runtime, peak memory,
chunking, device, TF32, and numerical drift.

**Gate.** Full horizon completes within the campaign budget, no CPU fallback or
non-XLA default path is mislabeled, and FP32/TF32 drift does not overturn the
float64 comparison classification. Scaling success is not HMC readiness.

### Phase 10: Structural Extension And Terminal Synthesis

**Objective.** Test the reusable structural interface on any executable
DSGE/NAWM client fixture, then close the campaign honestly.

**Work.** Require explicit stochastic/deterministic partition metadata,
innovation map, deterministic completion, observations, parameter chart, and a
same-target scalar. Run local structural/support and derivative tests before any
large client model. Compare with Zhao--Cui only if a reviewed fixed-variant
extension exists. Produce all-model, blocker, inference-status, and post-run
red-team tables.

**Gate.** Every in-scope row is `pass`, `negative_result`, or `blocked` with
evidence. “All rows attempted” must not be rewritten as “all models validated.”

## Test And Rung Matrix

Every primary row follows the same ordering, with target-specific sizes frozen
in its preparation record:

| Rung | Purpose | Minimum evidence |
| --- | --- | --- |
| Primitive | Catch algebra/wiring bugs | Dense float64 forward, explicit JVP/VJP, autodiff, coordinate FD, random-direction FD |
| `T=1` | Isolate teacher/reset | Teacher/student feature and tangent identity, carried-weight identity, support/rank/positivity |
| Short prefix | Detect recursive drift | Per-time value/score increments, chart margins, reference gaps |
| Resolution ladder | Estimate numerical error | At least three ordered teacher/capacity rungs when feasible; adjacent differences and cost |
| Parameter-region validation | Detect center-only chart | Frozen preparation region, disjoint held-out points, no chart switch |
| Full horizon pilot | Test feasibility | One diagnostic replicate, structured artifact, no scientific ranking |
| Replicated full horizon | Support inference | Paired replications or precision stopping, simultaneous intervals, final-only audit |
| GPU/XLA | Production-target feasibility | Device/XLA/TF32 manifest, FP64 subset, memory/runtime, no fallback |

For long SV rows, prefix rungs are `T=1,10,100` before `T=1000/1008`. For
predator--prey and SIR, use `T=1,2,5` before `T=20`. LGSSM uses
`T=1,2,5,50`.

## Default And Assumption Audit

| Choice | Provenance and status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| Mass, first moments, second moments, one-step predictive feature | Chapter derivation; starting hypothesis, not universal default | Insufficient future-score information or excessive feature count | LGSSM per-time tangent and feature ablation |
| Fixed square active chart | Proposition in chapter; primary implementation | Positivity/rank holds only at center | Prepared-region chart margin audit |
| Overcomplete KKT chart | Chapter derivation; fallback hypothesis | Negative weights despite equality fit | Dense Phase 1 positivity tests |
| Teacher-supported anchors | Structural-support requirement | Too little capacity for features | Rank/feasibility audit before runtime |
| Nonuniform carried weights | Algorithm requirement | Existing LEDH code silently resets to uniform | `T=2` carried-log-weight test |
| Innovation quadrature/design | Model-specific open choice | Biased or aliased teacher | Per-model three-rung refinement |
| Initial `(32,32),(64,64),(128,128)` GPU ladder | Convenience capacity hypothesis only | Under-resolved long-horizon score | Prefix convergence before full horizon |
| Six primary model families | Current leaderboard observed-data scope | Hidden model omitted or scoped component promoted | Phase 0 target registry diff against leaderboard rows |
| Zhao--Cui fixed branch | HMC-compatible comparison adaptation | Frozen design poorly approximates adaptive source method | Own-scalar FD plus rank/fit refinement; no adaptive-source claim |
| Simultaneous 95% intervals | Conventional inferential coverage, fixed before results | Too few replications or dependence ignored | Precision pilot and effective replicate audit |
| Existing 16-replicate precedent | May seed a pilot only | Underpowered final inference | Precision stopping; never force final `n=16` |

## Skeptical Plan Audit

Status: `PASS_AFTER_SCOPE_AND_BASELINE_REPAIR`.

The initial idea “implement and compare all rows with Zhao--Cui” was materially
unsafe for five reasons, now repaired in this plan:

1. The current LGSSM Zhao--Cui cell is the Kalman oracle adapter, so comparing
   against it would double-count the oracle rather than test Zhao--Cui.
2. The current predator--prey helper uses a repository-forbidden retained-grid
   route for production evidence.
3. SIR d=18 has a real fixed-variant implementation, but its admitted P91 score
   is a local complete-data component. Treating that score as the marginal
   observed-data filtering score would compare different scalars and would hide
   the documented previous-marginal and proposal/transport derivative terms.
4. Zhao--Cui's paper proves density-approximation results, not equality of the
   BayesFilter frozen-branch likelihood score.
5. A universal percentage for “reasonable gradient agreement” would be an
   arbitrary scientific default and would conflate FD correctness with
   cross-method accuracy.

Additional controls:

- **Wrong baseline:** Kalman, teacher, Contract E--Chol, Contract E--TP, and
  Zhao--Cui have distinct roles and identities.
- **Proxy promotion:** feature/tangent tie-out is an engineering gate, not exact
  filtering evidence. SGQF and teacher references are labeled approximations.
- **Unfair comparison:** target/data/coordinate/horizon/initial-law identity is
  checked before analysis.
- **Hidden active-set branch:** charts are prepared and frozen; runtime switching
  is a hard veto.
- **Environment mismatch:** correctness starts CPU-hidden float64; serious
  feasibility uses trusted GPU/XLA and records TF32.
- **Misleading successful command:** every runner writes per-time values,
  scores, identities, and veto diagnostics; a completion marker alone cannot
  pass.
- **Full-horizon waste:** small prefixes and resolution ladders precede T=1000
  or high-dimensional runs.
- **Stale context:** historical raw Contract E and generic retained-grid
  admission labels are ignored under current repository policy.

## Compute And Attempt Budget

At initial drafting, the plan did not itself launch a campaign. The owner then
requested execution on 2026-07-15, confirming this campaign cap:

- at most 128 CPU core-hours for reference, preparation, and focused tests;
- at most 64 trusted GPU-hours total across all models;
- at most three full-horizon attempts per model, including failed attempts;
- at most two post-audit repairs per model without a revised campaign plan;
- unique versioned output directories for every attempt;
- no package/environment mutation, external publication, or paid compute.

These are operational convenience caps, not scientific thresholds. If a row
cannot reach its precision target inside the cap, classify it
`unresolved_precision` or `under_budgeted`; do not weaken the evidence rule.

## Repair And Stop Rules

For every phase:

1. run the smallest discriminating test;
2. classify failure as shared-core implementation, model adapter, preparation,
   numerical resolution, comparator, artifact, or scientific-candidate failure;
3. patch only the responsible layer while preserving target and frozen criteria;
4. rerun focused regression, then the failed rung in a new output directory;
5. record attempt, wall time, failure, repair, and remaining budget;
6. continue to the next planned repair unless a continuation veto fires.

Stop a row when the target is invalid, the fixed chart has no feasible
parameter-region certificate, required evidence is missing/corrupt, the
comparator cannot be made same-target within scope, or its budget is exhausted.
Do not stop the whole program because one candidate/model fails unless it
reveals a shared-core or mathematical defect.

## Required Result Records

Every serious run manifest records git commit and dirty state, command,
environment/conda env, CPU/GPU/XLA/TF32 status, data and preparation hashes,
random seeds and roles, wall time, output paths, plan/result paths, route
identity, scalar target identity, parameter coordinates, feature/chart identity,
and source dependency closure.

Each model result includes:

- a decision table with primary criterion, vetoes, uncertainty, next action,
  and nonclaims;
- an inference-status table for hard veto, statistically supported equivalence,
  descriptive differences, default readiness, and next evidence;
- separate engineering, numerical, and scientific ledgers;
- strongest alternative explanation, weakest evidence, and what would overturn
  the decision;
- exact reasons for any unavailable Zhao--Cui comparator.

## Terminal Deliverables

1. Experimental Contract E--TP TensorFlow/XLA implementation and focused tests.
2. Frozen all-model target/comparator registry and source-anchor ledger.
3. Per-model chart/preparation certificates or blockers.
4. Per-model value/score/reference/Zhao--Cui artifacts with per-time diagnostics.
5. All-model comparison report that distinguishes equivalence, difference,
   unresolved precision, and invalid comparison.
6. Structural DSGE/NAWM extension result or explicit adapter/comparator blocker.
7. Reset memo stating what is implemented, what failed, what remains
   experimental, and why no canonical/leaderboard/HMC claim follows.
