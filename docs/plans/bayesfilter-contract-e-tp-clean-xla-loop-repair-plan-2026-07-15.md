# Contract E--TP Clean-XLA Loop Repair Plan

metadata_date: 2026-07-15
status: COMPLETE_PASS_CLEAN_XLA_COMPILE_EXPLOSION_REPAIRED
program_id: contract-e-tp-clean-xla-loop-repair
owner_request: repair and verify the XLA compilation issue, then write an all-model master program
compute_budget: at most 4 trusted GPU-hours and three GPU attempts per rung
artifact_root: `docs/benchmarks/artifacts/contract_e_tp_clean_xla_loop_repair_20260715/`

review_record: Claude bounded review round 1 returned `REVISE`; explicit
`T=1,2` semantics and quantitative graph-growth gates were added. Round 2
returned `VERDICT: AGREE`.

result: `docs/plans/bayesfilter-contract-e-tp-clean-xla-loop-repair-result-2026-07-15.md`
successor_master_program: `docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-master-program-2026-07-15.md`

## Objective

Replace the Python-unrolled time and finite-lookahead recursions in the
experimental LGSSM Contract E--TP GPU/XLA score route with functional
TensorFlow loops. Preserve the exact finite value program, its total derivative,
fixed Contract E--TP charts, fail-closed predicates, and controlling `T=10,50`
preparations while making XLA graph size depend primarily on one loop body rather
than on the observation horizon.

This phase repairs graph topology. It does not change the Contract E--TP feature
definition, quadrature order, lookahead length, charts, target, theta, dataset,
dtype, or scientific admission status.

## Entry Conditions

- The unrolled float64 route passes CPU/GPU value and score parity at `T=10,50`.
- The `T=50` unrolled route is numerically valid but takes `1694.94 s` for
  compile plus first execution and contains `140,307` graph operations.
- Static diagnosis shows zero functional `While` nodes, a Python time loop, and
  overlapping Python-unrolled finite-lookahead recursions.
- Contract E--TP remains experimental; Contract E--Chol remains canonical.
- Trusted GPU commands require escalated execution.

## Research And Engineering Question

Can the same finite LGSSM Contract E--TP value and total score be represented by
one fixed-shape intermediate filtering loop and one bounded batched lookahead
loop, so that `T=50` XLA compilation no longer exhibits static graph explosion?

## Frozen Baseline

| Item | Baseline |
| --- | --- |
| Target | deterministic LGSSM seed `81100`, physical theta `[0.72,0.55,0.35,0.35,0.45]` |
| Candidate | `contract_e_tp_experimental_v1`, finite lookahead 8, order-5 tensor Gauss--Hermite |
| Charts | controlling Phase 8B `T=10,50` active indices and row scales |
| Scalar/score oracle for topology repair | current unrolled Contract E--TP finite program, not Kalman |
| Independent scientific oracle | differentiated Kalman result already recorded; rerun only as a regression summary |
| Dtype/device | float64 trusted GPU/XLA; TF32 enabled but irrelevant to float64 |
| Unrolled graph | `24,787` nodes at `T=10`; `140,307` at `T=50` |
| Unrolled timing | `51.26 s` at `T=10`; `1694.94 s` at `T=50`; warmed `0.233/1.516 s` |

## Required Design

### Fixed-Lag Information Features

Implement a batched fixed-lag dynamic program over all start times. Carry one
batch of information matrices and vectors and iterate backward over at most
eight offsets with `tf.while_loop`. Use a validity mask for shorter terminal
windows. Do not rebuild one Python graph fragment per start time or call a
Python-unrolled backward recursion from inside a horizon loop.

The new function must match `_finite_lookahead_information_parameters` in
value and total derivative at multiple horizons and lookahead lengths before it
becomes the compiled route.

### Filtering Time Recursion

Use three stages:

1. a separate first step, because the parent cloud changes from the initial
   tensor-product size to the fixed feature-count chart;
2. one fixed-shape `tf.while_loop` body for times `1` through `T-2`, carrying
   parent points, parent log weights, accumulated objective, validity, and
   fixed-shape diagnostic histories; and
3. a separate terminal step with no projection.

Edge horizons are explicit. At `T=1`, execute only the initial-parent terminal
step and return empty projection histories. At `T=2`, execute the initial-parent
first projection and the fixed-chart terminal step, with zero intermediate-loop
iterations. At `T>=3`, execute all three stages. These branches are static
horizon dispatch, not Python time recursion, and must pass increment/final-state
parity at `T=1,2,3`.

The intermediate body must contain the same `_flow_correction`, log-weight
calculation, continuation feature evaluation, and fixed square projection as
the unrolled finite program. `maximum_iterations`, `parallel_iterations=1`, and
fixed shape invariants must be explicit. Histories may use fixed tensors or
fixed-size `TensorArray`; no Python tensor-record list is allowed in the
compiled route.

### Score

First test `GradientTape` over the functional loops. It is admissible only if
trusted XLA compiles it without a TensorList boundary failure and graph scaling
passes. If reverse autodiff fails or still causes material graph expansion,
use the existing explicit tangent/JVP ownership to carry all five parameter
directions through the same loop. Do not use `stop_gradient`, a transported-
state-only partial derivative, or a changed scalar.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Primary engineering criterion | compiled graph contains functional `While` operations; no Python time/window recursion in the new XLA route; `T=50` graph size and tracing no longer scale like 50 duplicated bodies |
| Numerical parity criterion | loop-native versus unrolled objective, increment history, score, chart validity, feature targets/residuals, and final log weights agree within machine-epsilon-scaled execution-equivalence bounds at `T=2,10,50` |
| Same-scalar derivative criterion | loop-native total score passes the existing individual-direction FD screen; `0.05*sqrt(p)` remains FD-only |
| Compiled fail-closed criterion | valid controlling charts pass; an invalid fixed chart produces `valid=false` and nonfinite poisoned carried output under XLA even though XLA ignores assertion operators |
| GPU criterion | real GPU placement, XLA true, exact preparation hash, finite value/score, no CPU fallback |
| Compile-scaling criterion | `T=50` compile plus first call is materially below the `1694.94 s` baseline and completes within the phase budget; report the value without inventing a universal threshold |
| Graph-scaling criterion | at `T=10,50`, record top-level nodes, function-library nodes, graph bytes, and functional loop operations; both graphs must contain at least two `While`/`StatelessWhile` operations, `T=50/T=10` top-level-node and function-node ratios must be `<=1.10`, and the GraphDef-byte ratio must be `<=1.25` |
| Explanatory diagnostics | trace time, compile time, warmed time, memory, HLO/GraphDef size, operation inventory, register-spill warnings |
| Artifact | versioned JSON result plus Markdown close record under the declared artifact root |

## Required Checks

1. Static AST/source guard: the new compiled route and its reachable loop-native
   helpers contain `tf.while_loop` and no Python `for`, `while`, `tf.unstack`,
   or reversed Python tensor-record recursion over time/windows.
2. CPU-hidden float64 parity for lookahead values and Jacobians at small
   horizons and for full candidate value/score at `T=2,10,50`.
3. Existing primitive, streaming, recursive, progressive, and Phase 7
   comparison regression suites.
4. Graph inventory at `T=10,50`, including `While`/`StatelessWhile`, graph
   nodes, graph bytes, function count, and function-library nodes.
5. Trusted GPU/XLA `T=10` compile/warm execution.
6. Trusted GPU/XLA `T=50` only after `T=10` passes.
7. Compiled invalid-chart negative control at the smallest feasible horizon.
8. `compileall`, JSON parsing, artifact hashes, and `git diff --check`.

## Skeptical Pre-Execution Audit

Status: `PASS_AFTER_BASELINE_AND_PROXY_REPAIR`.

- The baseline is the same finite Contract E--TP scalar, not Kalman; otherwise a
  scientific approximation difference could be mistaken for a topology defect.
- Warmed runtime alone is not the primary criterion because the observed
  failure is compilation latency and graph size.
- Returning fewer diagnostics is not proposed as the repair: measurement shows
  it changes `140,307` nodes only to `140,249`.
- A `tf.while_loop` token alone is not sufficient. Graph inventory must show
  functional loop operations and bounded horizon scaling.
- Replacing only the outer time loop is incomplete because the overlapping
  lookahead builder independently contributes `46,803` nodes at `T=50`.
- A failed candidate does not invalidate the mathematics. Value/score parity
  and fail-closed checks distinguish topology repair from target drift.
- Static parameter loops of fixed size five are not the horizon-scaling target,
  but no new Python parameter loop may be introduced in the compiled route.
- GPU compilation is attempted only after CPU parity and graph guards pass.

The graph ratios are engineering topology guards, not scientific tolerances.
With one fixed filter body and one fixed eight-offset body, increasing `T` five
fold should change tensor shapes and captured chart bytes but should not
duplicate operation bodies. Ten percent node/function-node headroom and 25%
GraphDef-byte headroom allow shape/constant metadata growth while rejecting the
observed `5.66x` unrolled node growth. Any unexpected excess triggers graph
inventory review rather than post-hoc relaxation.

## Defaults And Assumptions

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| split first/intermediate/terminal | required by parent-shape change; reviewed design | off-by-one likelihood or projection | `T=1,2,3` increment and final-cloud parity |
| batched eight-offset recurrence | algebraic re-expression of frozen windows | mask/order error near terminal | all-start comparison at `T=6`, lags `1,2,8` plus Jacobians |
| reverse autodiff through loop first | simplest same-scalar route; hypothesis | TensorList/XLA failure or retained-history blowup | CPU graph inventory then trusted `T=10` |
| fixed-size diagnostic histories | required for compiled result parity | history indexing drift | per-time array parity |
| float64 | frozen scientific baseline | does not answer TF32 readiness | preserve explicit nonclaim |

## Forbidden Claims And Actions

- Do not change charts, feature definitions, lookahead length, target data,
  theta, or quadrature to make compilation easier.
- Do not use NumPy/SciPy inside the gradient-bearing runtime path.
- Do not retain a Python-unrolled fallback as the default XLA factory.
- Do not call execution on GPU alone `clean XLA`; graph and static guards must
  pass.
- Do not claim float32/TF32, nonlinear-model, HMC, canonical, default, or
  leaderboard readiness.
- Do not overwrite Phase 9 artifacts.

## Repair Loop And Stop Conditions

Run the smallest parity and graph diagnostic after each layer. Repair localized
shape, TensorArray, autodiff, or XLA failures without changing the frozen
scientific contract. Use a fresh artifact path for every trusted attempt.

Stop and write a blocker if:

- loop-native and unrolled values or total scores cannot be reconciled;
- XLA cannot differentiate the functional loop and explicit JVP ownership
  cannot be preserved within budget;
- functional loops still produce horizon-proportional replicated graph bodies;
- invalid charts can pass or propagate finite carried state under XLA; or
- the four-hour trusted GPU or three-attempt-per-rung budget is exhausted.

## Exact Handoff

After implementation, local checks, trusted GPU evidence, and an execution
review pass, write:

- `docs/plans/bayesfilter-contract-e-tp-clean-xla-loop-repair-result-2026-07-15.md`;
- a versioned graph/timing JSON artifact; and
- a separate all-model Contract E--TP clean-XLA validation master program under
  `docs/plans`.

The all-model master program may inherit the repaired loop pattern but must not
assume model adapters, feature families, structural state, or comparators are
interchangeable. Each model requires its own shape, target, derivative, and
full-horizon evidence gates.
