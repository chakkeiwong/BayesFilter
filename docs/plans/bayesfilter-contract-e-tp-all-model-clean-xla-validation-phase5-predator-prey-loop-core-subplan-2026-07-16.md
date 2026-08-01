# Phase 5: Predator--Prey Loop-Native Core Subplan

Date: 2026-07-16

Status: `REVISED_PENDING_BOUNDED_REVIEW`

## Objective And Research Question

Determine whether the frozen real-plane predator--prey Contract E--TP scalar
can be expressed with bounded TensorFlow functional loops while preserving the
initial-observation-before-transition order, the same value and total score,
and the previously checked `T=2` exact / `T=5` approximate-reference behavior.
For the engineering question, extend the already observed order-5/lookahead-4
finite-program baseline to `T=20` as an explicitly inherited hypothesis. Run
adjacent refinements as descriptive diagnostics only; do not use them to claim
accuracy or select a scientifically preferred configuration without a
separately justified margin.

## Entry Conditions

- Phase 4 closed with KSC `T=1000` GPU/XLA pass and Actual-SV row rejection;
  neither result is a shared continuation veto.
- Predator target identity is `zhao_cui_predator_prey_T20`, `d_x=2`, `p=6`,
  real-plane additive Gaussian dynamics and observations, with `y_0` generated
  from the initial state before any transition.
- Existing `T=2` order-5/lookahead-1 evidence agrees with a converged
  semi-analytic oracle; existing `T=5` order-5/lookahead-4 evidence agrees
  closely with an approximate corrected-time-order fixed-SGQF reference.
- The existing recursive core, Gaussian-closure continuation, and RK4 dynamics
  contain reachable Python horizon/substep loops and are ineligible for clean
  XLA claims.
- Remaining budget is `95.42 CPU core-hours`, `31.94 trusted GPU-hours`, and
  three full-horizon attempts for predator--prey.

## Research Intent Ledger

| Role | Binding item |
| --- | --- |
| main question | can the same predator finite scalar/score use bounded graph-native time, lookahead, and RK4 recursion? |
| mechanism | static first/terminal dispatch plus fixed-shape `tf.while_loop` bodies; no target-law change |
| expected failure | reverse autodiff TensorLists, dynamic Cartesian shapes, chart positivity loss, or order/lookahead approximation drift |
| promotion criterion | exact loop/unrolled parity at `T=1,2,5`; same-scalar FD; real-plane/time-order identities; clean topology and fail-closed controls; explicit baseline-status `T=20` finite program |
| promotion veto | scalar/score mismatch, wrong time order/support, finite invalid output, source-audit failure, or silent promotion of the baseline to an accuracy/default claim |
| continuation veto | shared scalar/harness invalidity that cannot be localized inside the phase budget |
| repair trigger | localized XLA, graph, shape, autodiff, chart, or harness failure |
| explanatory only | SGQF gap at `T=5`, order/lookahead refinement gaps, compile/warm time |
| forbidden conclusion | exact nonlinear filtering, Zhao--Cui parity, HMC/default/leaderboard readiness |

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| real-plane support | frozen `PredatorPreySSM` additive Gaussian law; reviewed correction | clipping changes the target | source and negative-coordinate tests |
| observation before first transition | dataset/simulator identity and reviewed Phase 5 repair | historical route scores a different likelihood | `T=1` exact initial Gaussian identity |
| data/theta identity | `_predator_prey_dataset(81104)`, theta `[0.6,114,25,0.3,0.5,0.5]`; frozen `T=20` observation SHA-256 `dc63294b6e77913aef0c92796dd2d3c7a1721a766f976fcc392cd02a70754387`, theta SHA-256 `d07fb2bd6c5a225794ffa7e77cb7e7c524e90723b88475c44789a1fb4d6dee4f` | regenerated data or dirty-source drift changes the scalar | preparations serialize observations and generator/source/seed; harness recomputes and rejects any digest mismatch |
| teacher order 5 | successful `T=2,5` inherited engineering baseline; hypothesis only | inadequate scientific accuracy at `T=20` | order `7` is descriptive sensitivity, not a selector |
| continuation order 5/axis | successful approximate `T=5` inherited baseline; hypothesis only | Gaussian-closure quadrature error | order `7` is descriptive sensitivity, not a selector |
| lookahead 4 | full available future at `T=5`; inherited finite-program baseline at `T=20`, not a default | loses longer future score information | lookahead `8` at a CPU-feasible rung is descriptive sensitivity only |
| float64 | reference arm | performance only | timing/memory descriptive |
| FD step `1e-5*max(1,abs(theta_j))` | inherited predator float64 diagnostic | cancellation/truncation | fixed explanatory `3e-6` and `3e-5` scales only after primary failure |
| `0.05*sqrt(6)` | owner FD-only policy | could be misused as cross-method tolerance | label same-scalar individual-direction only |
| SciPy HiGHS chart selection and active cutoff `1e-10` | inherited viable `T=2,5` offline preparation semantics; SciPy `1.17.1`; convenience baseline, not a default | solver/version or numerical-zero classification changes active indices | record method/version/status/cutoff and active indices per time; same-environment rerun must reproduce chart-identity hash; cutoff sensitivity is descriptive and cannot replace baseline |
| parity `rtol=5e-11,atol=5e-11`; score `rtol=2e-8,atol=2e-9` | pre-run engineering convenience gates, deliberately looser than scalar-SV after 20 RK4 substeps; not a theorem or scientific margin | may hide an implementation-order mismatch | primitive RK4 and per-increment parity localize any failure; never relax post-result and do not call passing equality proof |
| graph ratios `1.10/1.10/1.25` | reviewed master-program topology policy | captured constants can inflate bytes without unrolling | inspect loop counts and captured bytes; topology-only, never numerical evidence |

The `T=20` engineering baseline is frozen before execution as teacher order 5,
continuation order 5 per axis, and lookahead 4, matching the existing viable
`T=5` finite-program family. This is cross-horizon extension of a row-specific
baseline, not scientific selection or promotion. Order-7 and lookahead-8 arms
may expose sensitivity or veto their own configurations, but no observed
difference can replace the frozen baseline or support ranking because no
accuracy margin exists. If the baseline fails, classify that candidate or
repair an implementation defect; do not substitute the most favorable arm.

## Required Implementation

1. Add a loop-native Gaussian-closure continuation using `tf.while_loop` and
   the same equations, state covariance symmetrization, quadrature, and common
   reference scaling.
   Use a fixed padded `[lookahead_steps,2]` window plus a scalar active count.
   The continuation body consumes only active entries. Preserve the existing
   analytic one-step formula through `tf.cond(active_count == 1, ...)`; counts
   2--4 use the identical Gaussian-closure recursion. `T=5` parity must exercise
   tail counts `4,3,2,1` explicitly.
2. Add a tensor-index-compatible positive-time teacher step. Preserve the
   initial step as a static special case; replace `tf.repeat`/`tf.tile`
   Cartesian expansion with static broadcast/reshape if reverse XLA requires
   it, preserving parent-major order.
3. Audit `PredatorPreySSM.transition_mean` and its RK4 helper. Any reachable
   Python RK4/substep loop must become a functional TensorFlow loop without
   changing step count, time increment, or equations.
4. Add `contract_e_tp_predator_prey_loop_core` and an XLA-default factory whose
   total score is reverse autodiff of the same objective. It returns objective,
   every increment, score, chart histories, final particles/weights, and a
   Boolean validity predicate. Invalidity poisons every claim-bearing numeric
   output.
5. Preserve the recursive route only as a short-rung finite-program oracle and
   make the source auditor reject it for clean execution.
6. Extend preparation to `T=10,20` for the frozen order-5/lookahead-4 baseline.
   Run bounded order-7/lookahead-8 sensitivity only if it fits the phase budget.
   Offline NumPy/SciPy remain preparation/reference tools, never gradient
   runtime.
7. Amend the preparation schema to serialize the exact theta and observations,
   their TensorFlow serialized-tensor SHA-256 hashes, generator/source identity,
   seed `81104`, time order, support, SciPy version, `linprog(method="highs")`,
   solver status, active cutoff `1e-10`, and a chart-identity hash over active
   indices/row scales. The harness regenerates nothing silently: it validates
   every serialized identity against the frozen values before tracing.

The clean factory may not hide a Python RK4 loop behind a call into
`bayesfilter/highdim/models.py`. Either make `PredatorPreySSM.transition_mean`
graph-native with eager/graph-safe validation or bind a repository-owned local
functional transition carrying the exact same 20 classical RK4 substeps. The
source evidence must audit both the predator module and every reachable model
helper that owns iteration. A single-file root-token audit is insufficient.
The current model parameter-box validator returns a Python Boolean through
`.numpy()`. The compiled route must instead carry an equivalent graph-native
box predicate and poison outputs when it is false; eager public validation may
retain its existing exception behavior. A Python trace-time validation result
is not a same-factory negative control.

## Required Checks And Artifacts

Fresh root:
`docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/`.

Required artifacts:

- fresh `T=1` plus existing hashed `T=2,5` preparation identities and
  loop/unrolled parity result;
- value, increment, score, final-state, chart, FD, support, time-order, source,
  graph, warm-replay, and invalid-control evidence;
- descriptive sensitivity table for adjacent teacher/continuation orders and
  lookahead hypotheses, with no selection/ranking field;
- fresh frozen-baseline `T=10,20` preparations and CPU/XLA core results;
- source/graph inventory comparing `T=5` to `T=20`, with top/function ratios
  `<=1.10` and GraphDef-byte ratio `<=1.25`;
- focused tests, exact command log, hashes, run manifest, result record, and
  reviewed Phase 6 subplan.

Local checks precede any full-horizon GPU work; Phase 5 itself is CPU-focused.
At minimum:

- analytic `T=1` initial-observation identity;
- loop versus recursive value/increment/state/chart parity at `T=1,2,5` with
  float64 `rtol<=5e-11`, `atol<=5e-11` for scalars/arrays and score
  `rtol<=2e-8`, `atol<=2e-9`;
- `T=2` result remains consistent with the existing semi-analytic artifact;
- `T=5` result remains consistent with the existing finite-program artifact;
- six individual-coordinate same-scalar central FDs pass the owner FD-only
  rule with all twelve endpoints valid;
- exact compiled invalid theta
  `[0.05,114.0,25.0,0.3,0.5,0.5]` is frozen before serious execution. Its sole
  mechanism is the graph-native lower bound `r>=0.1`, while the arithmetic
  remains finite. A separate preflight artifact binds theta, mechanism,
  preparation hashes, horizons, callable identity, and expected false validity;
  the exact compiled factory must poison objective, every increment, score,
  final particles/weights, and every floating chart diagnostic without aborting;
- clean source closure has no Python time/lookahead/RK4/solver loop; historical
  roots are rejected;
- CPU-hidden XLA compile and warm replay pass at the frozen-baseline longest
  rung.

## Exact Commands And Timeouts

Implementation adds
`docs/benchmarks/run_contract_e_tp_clean_xla_phase5_predator_prey.py`,
`docs/benchmarks/freeze_contract_e_tp_phase5_predator_invalid_control.py`, and
`tests/highdim/test_ledh_contract_e_tp_predator_prey_loop.py`. These paths do
not exist at plan-review time and are required implementation artifacts before
the exact commands run. Preparation continues to use
`docs/benchmarks/prepare_contract_e_tp_predator_prey_charts.py`, extended only
to emit the exact frozen baseline at `T=10,20`. All commands use fresh
create-or-fail leaf directories; logs are preserved beside structured JSON.

```bash
mkdir -p docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey
mkdir docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716
CUDA_VISIBLE_DEVICES=-1 timeout 1800 pytest -q tests/highdim/test_ledh_contract_e_tp_predator_prey_loop.py tests/highdim/test_contract_e_tp_clean_xla_guardrails.py tests/highdim/test_ledh_predator_generalized_fd_root_cause_diagnostic.py > docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/focused-tests.log 2>&1
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 timeout 600 python docs/benchmarks/prepare_contract_e_tp_predator_prey_charts.py --time-steps 1 --teacher-order 5 --continuation-order 5 --lookahead-steps 4 --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t1_preparation.json > docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t1-preparation.log 2>&1
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 timeout 3600 python docs/benchmarks/prepare_contract_e_tp_predator_prey_charts.py --time-steps 10 --teacher-order 5 --continuation-order 5 --lookahead-steps 4 --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t10_preparation.json > docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t10-preparation.log 2>&1
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 timeout 7200 python docs/benchmarks/prepare_contract_e_tp_predator_prey_charts.py --time-steps 20 --teacher-order 5 --continuation-order 5 --lookahead-steps 4 --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t20_preparation.json > docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t20-preparation.log 2>&1
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig timeout 600 python docs/benchmarks/freeze_contract_e_tp_phase5_predator_invalid_control.py --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t1_preparation.json --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t10_preparation.json --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t20_preparation.json --theta 0.05,114,25,0.3,0.5,0.5 --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/invalid-control.json > docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/invalid-control.log 2>&1
mkdir docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-cpu-20260716
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 timeout 10800 python docs/benchmarks/run_contract_e_tp_clean_xla_phase5_predator_prey.py --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t1_preparation.json --preparation docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_predator_prey_t2_order5_analytic_lookahead1_preparation_20260715.json --preparation docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_predator_prey_t5_order5_gaussian_closure_lookahead4_stabilized_preparation_20260715.json --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t10_preparation.json --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/t20_preparation.json --invalid-control docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-preparation-20260716/invalid-control.json --t2-reference docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_predator_prey_t2_order5_analytic_lookahead1_result_20260715.json --t5-reference docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_predator_prey_t5_order5_gaussian_closure_lookahead4_stabilized_result_20260715.json --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-cpu-20260716/result.json > docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-05/predator-prey/attempt-01-cpu-20260716/result.log 2>&1
```

The harness binds the fresh `T=1`, existing hashed `T=2,5`, and fresh `T=10,20`
preparations. Before serious
execution, focused tests must prove CLI identity validation, output overwrite
refusal, complete invalid-output poisoning, current-factory `T=5/T=20` graph
comparison, and exact artifact schema fields. Any descriptive sensitivity arm
gets a separate reviewed command and fresh root; it is not implicit in this
initial ladder.

Every preparation and harness manifest records git commit and dirty-status
digest, exact semantic command, Python/platform/conda environment, TensorFlow,
NumPy and SciPy versions where applicable, intentional `CUDA_VISIBLE_DEVICES=-1`,
dtype/XLA, theta/data/generator/seed identity, preparation/chart hashes, wall
time, attempt, output/log paths, controlling plan, and result path. The harness
rejects any old `T=2,5` preparation lacking the frozen observation digest unless
it independently recomputes the old artifact's theta, seed, horizon, active
indices, row scales, and resulting chart-identity hash before use.

## Skeptical Pre-Execution Audit

Status: `PASS_DRAFT_FOR_REVIEW`.

The audit found and repaired an unsupported post-result selection rule. No
target-specific accuracy margin exists, so adjacent order/lookahead differences
cannot nominate a preferred configuration. The plan now freezes the existing
order-5/lookahead-4 family only as the engineering baseline, prohibits
substitution after results, and adds cross-module RK4 source coverage plus exact
commands and timeouts.

- The baseline is the same finite scalar, not SGQF or Zhao--Cui.
- SGQF and order/lookahead differences are explanatory diagnostics and cannot
  silently become promotion margins or configuration selectors.
- Real-plane support prohibits positivity clipping even if a chart is easier.
- `T=1` and `T=2` do not contain an intermediate filter loop, so graph growth
  uses `T=5` versus `T=20` only after the same configuration is available.
- A numerical or chart rejection of the frozen `T=20` baseline is a candidate
  negative result and excludes it from Phase 6. Only a proved implementation,
  harness, or serialization defect may be repaired inside this baseline. An
  order/lookahead refinement is a separately reviewed diagnostic/new candidate
  and cannot make the frozen baseline pass. Candidate rejection does not reject
  the research direction unless the target/scalar is wrong.
- Phase 6 GPU launch is not authorized until this phase writes its result and a
  reviewed Phase 6 subplan passes readiness.

## Budget And Repair Reserve

Phase cap: `24 CPU core-hours`, zero trusted GPU-hours, and zero full-horizon
GPU attempts. Minimum entry plus reserve: `16 CPU core-hours`. All serious CPU
commands bind BLAS/OpenMP and TensorFlow intra/inter-op threads to one and are
charged conservatively at two core-hours per wall-hour. No command timeout may
exceed the remaining phase cap.

## Forbidden Actions And Claims

- Do not clip states, enforce positivity, change the target law, or restore the
  historical transition-before-`y_0` order.
- Do not transfer scalar-SV lookahead/order settings.
- Do not replace the frozen `T=20` baseline with a descriptively favorable
  refinement after observing results.
- Do not use Python time/lookahead/RK4 loops in a clean factory, stop gradients,
  disable XLA for production evidence, or use NumPy/SciPy in the gradient path.
- Do not claim exact nonlinear accuracy, cross-method equivalence, Zhao--Cui
  source-faithfulness, HMC, canonical/default, leaderboard, or production
  readiness.

## Exact Phase 6 Handoff

At close, write the Phase 5 result, hash all controlling artifacts, draft or
refresh
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase6-predator-prey-gpu-xla-subplan-2026-07-16.md`,
and review it for the exact frozen baseline configuration, CPU entry evidence, trusted
GPU memory policy, same-factory FD/fail-closed gates, fresh paths, budget, and
nonclaims. Continue automatically only if every `NEXT_PHASE_READINESS` clause
passes. A Phase 5 row rejection still hands off a bounded Phase 6 blocker result
or proceeds to Phase 7 as the reviewed program specifies; it does not cause a
false GPU launch.

## Stop And Repair Conditions

Repair and retry localized graph, XLA, shape, autodiff, serialization, or
preparation failures under fresh roots. Stop Phase 5 on wrong target/time order,
unresolved scalar mismatch, inability to preserve total score, phase budget
exhaustion, or an underdetermined scientific choice that materially changes the
finite program. Stop the whole program only for a proved shared invalidity,
total campaign budget exhaustion, or an external/irreversible boundary.
