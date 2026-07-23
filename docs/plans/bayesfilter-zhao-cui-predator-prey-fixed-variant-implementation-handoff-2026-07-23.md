# Zhao-Cui Predator-Prey Fixed-Variant Implementation Handoff

Date: 2026-07-23  
Owner: Zhao-Cui implementation agent  
Task type: implementation, mathematical/source audit, focused validation, and
leaderboard integration  
Current status: `IMPLEMENTATION_GATES_PASS_NOT_ADMITTED`

## Executive Verdict

Zhao-Cui is unavailable for the predator-prey leaderboard row even though a
finite Zhao-Cui-labelled predator-prey function exists. The existing function
is the generic all-axes retained-grid evaluator:

- `bayesfilter/highdim/filtering.py::multistate_nonlinear_fixed_design_tt_value_path`;
- `bayesfilter/highdim/filtering.py::multistate_nonlinear_fixed_design_tt_score_path`;
- historical route ID
  `zhao_cui_predator_prey_t20_multistate_fixed_design_tt`.

That route declares
`route_role = diagnostic_historical_retained_grid` and
`leaderboard_admission = not_admitted_for_production_leaderboard_use_fixed_variant_zhao_cui`
at `bayesfilter/highdim/filtering.py:39-43`. It retains a deterministic
tensor-product state grid and is explicitly described as a tiny diagnostic at
`bayesfilter/highdim/filtering.py:1761-1781`. Repository policy therefore
classifies it as historical in `bayesfilter/highdim/source_route.py`, and
`docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py::_zhao_cui_predator_prey_tt_cell`
correctly fails closed at lines 1276-1340.

The implementation now provides a predator-prey-specific, memory-bounded,
fixed-variant filtering route that:

1. evaluates the sealed `x0 -> transition -> y1` T20 target;
2. freezes every approximation and random/discrete branch required for HMC;
3. returns one finite filtering value program;
4. returns the analytical total score of that exact same finite program;
5. avoids the generic all-axis retained tensor-product grid; and
6. has a repository-issued route identity that the leaderboard can admit.

The model-density derivatives already exist. The missing work is the fixed
filtering approximation, its source-order recursion, its same-program score,
and its admission wiring. Do not "repair" the row by relabelling or calling the
historical retained-grid function.

## Mandatory Reading And Source Gate

Before editing implementation code, inspect and cite both the paper and pinned
author source. This is required by the repository's Zhao-Cui source-anchor
gate. A source-faithfulness verdict without both classes of anchor is invalid
and must stop with `BLOCK_SOURCE_UNGROUNDED`.

### Paper

Read the local paper text, not only its abstract or conclusions:

- `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt`
- squared-TT defensive construction, Equation (13): lines 539-573;
- paired-core marginalization, Proposition 2, conditional densities, and KR
  construction: lines 592-670;
- filtering proposal and correction, Algorithm 3: lines 890-924;
- recursive/preconditioned construction: lines 1656-1718;
- predator-prey model and experiment: lines 2412 onward, including the model,
  parameterization, and experiment configuration.

The relevant mathematical facts are:

- the square root of the target is approximated by a TT and then squared;
- the positive defensive term gives support and a valid proposal density;
- paired-core mass contractions produce marginals and conditional densities;
- the conditional KR map samples the new state;
- importance weights correct proposal-approximation error.

These facts apply to individual operations. They do not make every local
assembly source-faithful.

### Pinned author source

Inspect and cite exact lines or extracted MATLAB-live-script sections from:

- `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg4_predatorprey/mainscript.m`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pre_sol.m`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/computeL.m`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/setup.mlx`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/transition.mlx`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/st_process.mlx`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/ob_process.mlx`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/like.mlx`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/SIRT.m`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/TTSIRT.m`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m`;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m`.

Important visible anchors include:

- `mainscript.m:12-17`: predator-prey identity, `d=6`, `m=2`, `n=2`, and
  `T=20`;
- `mainscript.m:45-67`: source particle count, polynomial, mapping,
  preconditioning, and TT settings;
- `mainscript.m:69-79`: seeded bounded and preconditioned solves;
- `ssmodel.m:34-42`: `x0`, then transition, then observation ordering;
- `full_sol.m:21-43`: sequential push, reapproximation, transport sampling,
  and correction;
- `full_sol.m:46-129`: ESS repair, weighted centering, fitted SIRT, and
  normalizer update;
- `SIRT.m:51-85`: square-root potential-to-density construction;
- `@TTSIRT/marginalise.m:19-85`: direction-dependent paired-core marginal;
- `@TTSIRT/eval_cirt_reference.m:43-100`: prefix conditional evaluation and
  proposal-density calculation.

The `.mlx` files are ZIP/XML MATLAB live scripts, not plain text. Extract or
inspect their embedded MATLAB/document XML rather than treating binary terminal
output as a source audit.

## Route Classification Requirement

For every material operation and for the assembled route, use exactly one of:

- `source_faithful`: matches a cited paper and author-source operation;
- `fixed_hmc_adaptation`: freezes randomness, ranks, maps, schedules, samples,
  or another branch while preserving the cited author algorithmic route;
- `extension_or_invention`: not present in the paper/source or changes the
  algorithmic route.

The current reusable compiler explicitly records the following distinctions at
`bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py:443-516`:

| Operation | Current classification |
| --- | --- |
| Squared-TT defensive density | `source_faithful` operation only |
| Paired-core prefix conditional | `source_faithful` operation only |
| Frozen randomness/settings | `fixed_hmc_adaptation` |
| Reordered `(x_previous,x_current)` compiler | `extension_or_invention` |
| Finite-grid CDF/bisection inverse | `extension_or_invention` |
| Fixed-branch APF finite value and score | `extension_or_invention` |

Therefore, the existing assembled compiler is
`extension_or_invention`. Do not call it source-faithful. A new implementation
may change individual classifications only after checking the cited paper and
author source and documenting the exact reason.

The practical implementation target in this handoff is a Zhao-Cui-derived
fixed-HMC extension using the source squared-TT, paired-core conditional, and
importance-correction operations. If leaderboard governance requires the whole
route, rather than its operations, to be `source_faithful` or only
`fixed_hmc_adaptation`, stop before admission and report that policy mismatch.
Do not hide it by changing a label.

## Sealed Target Contract

The implementation must use this exact target. No P47 two-observation fixture,
reduced model, initial-observation-first dataset, or regenerated dataset may
fill this row.

| Field | Sealed value |
| --- | --- |
| Leaderboard row | `zhao_cui_predator_prey_T20` |
| Model | `bayesfilter.highdim.models.PredatorPreySSM` |
| State | `(P,Q)`, dimension 2 |
| Observation | dimension 2 |
| Physical parameter order | `(r,K,a,s,u,v)` |
| Truth | `(0.6,114.0,25.0,0.3,0.5,0.5)` |
| Initial mean | `(50.0,5.0)` |
| Initial covariance | `I_2` |
| Process covariance | `4 I_2` |
| Observation covariance | `4 I_2` |
| Horizon | `T=20` |
| Event order | `x0 -> 20 transitions -> observations y1:y20` |
| ODE interval | `delta=2.0` |
| RK4 internal step | `0.1` |
| RK4 substeps | `20` |
| Dataset seed | `81104` |
| Target ID | `zhao_cui_predator_prey_tf_seed81104_x0_then_y1_y20_v1` |
| State SHA-256 | `63cc7d7e8e3a251f76ebb607b152b58b59cd8ceda4489057e60070b44ab1d2ec` |
| Observation SHA-256 | `fea0681d43a4bd502d1f5a90e04f58da435c6e891e72d9da4d54f4cf0584f00a` |

The sealed dataset factory is
`bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py::generate_source_order_predator_prey_dataset_tf`.
It verifies both hashes. The existing source-order SGQF route ID is
`fixed_sgqf_zhao_cui_predator_prey_t20_transition_then_observe_physical_score_v1`.

The current UKF target in
`bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py` observes the initial
state before the first transition. It is not a same-target comparator. Do not
use it until a separately tested source-order UKF route exists.

## Existing Correct Building Blocks

Reuse reviewed mechanics where their contracts fit; do not copy them into a
parallel implementation without need.

- `bayesfilter/highdim/models.py::PredatorPreySSM`
  - RK4 transition;
  - `transition_mean_parameter_jacobian`;
  - `initial_log_density_parameter_score`;
  - `transition_log_density_parameter_score`;
  - `observation_log_density_parameter_score`.
- `bayesfilter/highdim/squared_tt.py`
  - squared-TT density and paired-core marginal operations.
- `bayesfilter/highdim/transport.py`
  - fixed TTSIRT transport and conditional proposal-density mechanics.
- `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py`
  - prepared fixed branches;
  - repository-computed branch/program identity;
  - fixed TTSIRT proposal compilation;
  - analytical target-density score recursion.
- `bayesfilter/highdim/zhao_cui_coupled_nonlinear.py`
  - useful engineering examples only, not a predator-prey model or source
    implementation.
- `tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py`
  - independent scalar and same-program score tests.
- `tests/highdim/test_zhao_cui_frozen_ttsirt_apf_compiler.py`
  - marginal, conditional, density, branch, and classification tests.

The current coupled nonlinear rung-2 work is a synthetic SIR-inspired block
test and is explicitly `extension_or_invention`. It is not predator-prey and
its tuned settings are not transferable defaults. Do not modify its concurrent
plan/result files as part of this task.

## Critical Source-Order Repair

Do not call `FrozenProposalAPFProgram._evaluate_core` unchanged for this row.
That evaluator currently stores one observation for every state cloud and
assimilates `observations[0]` at `states[0]` at
`bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py:736-775`. That is an
initial-observation-first program. The sealed row has no observation at `x0`.

Implement a source-order branch/program, preferably in a target-specific module
such as:

`bayesfilter/highdim/zhao_cui_predator_prey_fixed_variant_tf.py`

Its shapes must be explicit:

- states: `[T+1, N, 2]`, containing `x0:xT`;
- observations: `[T, 2]`, containing `y1:yT`;
- initial proposal log density: `[N]`;
- ancestors: `[T, N]`;
- frozen auxiliary log probabilities: `[T, N]`;
- transition proposal log density: `[T, N]`;
- score: `[6]` in physical `(r,K,a,s,u,v)` coordinates.

The source-order evaluator first corrects the `x0` proposal to the initial
density without an observation, and then performs exactly `T` transition and
observation updates. Use a TensorFlow time loop (`tf.while_loop` or an
equivalent graph-native scan), not a Python sample loop or a time loop unrolled
by Python in the XLA path.

## Mathematical Value And Score Contract

Let the entire proposal branch be prepared offline and independent of the
runtime parameter `theta`. This includes particles, TT cores, coordinate maps,
conditional proposal densities, reference points, ancestor uniforms,
ancestors, and auxiliary categorical laws. Let `q0(x0^i)` and
`qt(xt^i | x_{t-1}^{a_t^i})` be the exact proposal densities evaluated by the
same transport code that generated the fixed particles. Let `alpha_t` be the
frozen normalized auxiliary categorical law.

### Initial correction

For `i=1,...,N`, define

```text
ell_0^i(theta) = log p_theta(x_0^i) - log q_0(x_0^i)
c_0(theta)     = logsumexp_i ell_0^i(theta) - log N
W_0^i(theta)   = exp(ell_0^i(theta) - logsumexp_j ell_0^j(theta)).
```

There is no observation term in `ell_0`.

### Transition-observation correction

For `t=1,...,T`, sample index `i`, and fixed ancestor `a_t^i`, define

```text
ell_t^i(theta)
  = log W_{t-1}^{a_t^i}(theta)
  + log p_theta(x_t^i | x_{t-1}^{a_t^i})
  + log p_theta(y_t | x_t^i)
  - log alpha_t^{a_t^i}
  - log q_t(x_t^i | x_{t-1}^{a_t^i}),

c_t(theta)   = logsumexp_i ell_t^i(theta) - log N,
W_t^i(theta) = exp(ell_t^i(theta) - logsumexp_j ell_t^j(theta)).
```

The fixed finite value is

```text
L_N(theta) = c_0(theta) + sum_{t=1}^T c_t(theta).
```

This is the claimed target of the analytical score. It is not the exact
observed-data likelihood, and the fixed finite program alone does not establish
unbiased pseudo-marginal or posterior correctness.

### Analytical recursive score

Because `q0`, `qt`, `alpha_t`, states, and genealogy are fixed with respect to
runtime `theta`, their runtime derivatives are zero. Let

```text
s_0^i = partial_theta log p_theta(x_0^i),
d_0^i = s_0^i - partial_theta c_0,

m_t^i = d_{t-1}^{a_t^i}
        + partial_theta log p_theta(x_t^i | x_{t-1}^{a_t^i})
        + partial_theta log p_theta(y_t | x_t^i).
```

Then

```text
partial_theta c_0 = sum_i W_0^i s_0^i,
partial_theta c_t = sum_i W_t^i m_t^i,
d_t^i             = m_t^i - partial_theta c_t,
partial_theta L_N = partial_theta c_0
                    + sum_{t=1}^T partial_theta c_t.
```

Use the model's manual density-score methods to compute these terms. Runtime
finite differences and `tf.GradientTape` are forbidden. A central finite
difference of `L_N` at a small set of representative parameter points is an
independent diagnostic only. A `GradientTape` comparison of individual model
density scores may appear only in tests and is never runtime score provenance.

If the chosen proposal or auxiliary law depends on runtime `theta`, the formula
above is incomplete. Either freeze it, or derive and implement every pathwise
and proposal-density derivative. Do not omit these terms while claiming a total
score.

## Research Intent Ledger

| Item | Contract |
| --- | --- |
| Main question | Can a target-specific fixed Zhao-Cui-derived TTSIRT proposal provide a finite, deterministic, memory-bounded source-order T20 predator-prey value and exact analytical score of the same finite program? |
| Candidate | Fixed squared-TT/TTSIRT proposal branch with positive defensive mass, paired-core conditionals, frozen genealogy, exact proposal correction, and source-order APF recursion. |
| Expected failure modes | Wrong time order; density/measure mismatch; proposal-tail collapse; low ESS; incorrect auxiliary correction; score missing normalized-weight recursion; parameter-dependent branch; retained-grid fallback; XLA host fallback; excessive fit memory. |
| Primary implementation criterion | Source-order `L_N` and all six score coordinates are finite, deterministic, same-program, and satisfy the declared score audit at representative points. |
| Promotion criterion | The untouched N>1000 target run passes identity, fixed-branch, support, conditional-density, score, ESS, memory, GPU/XLA, and regression gates below. |
| Promotion veto | Any target/hash/time-order mismatch, retained-grid marker, runtime FD/autodiff, branch change with theta, non-finite value/score, invalid proposal density, zero defensive mass, failed same-program score audit, or missing GPU/XLA evidence. |
| Continuation veto | Source/paper mismatch that invalidates the proposed route; inability to define the proposal density used in correction; corrupted sealed dataset; campaign budget exhausted; or the assembled extension being disallowed as a Zhao-Cui leaderboard route. |
| Repair trigger | Candidate ESS/fit/conditioning/score failure with valid harness. Retune or repair on fresh calibration data within budget; do not tune on the untouched claim row. |
| Explanatory diagnostics | Runtime, allocator peak, conditional log-density error, rank/degree stability, weight spread, SGQF/GenUT differences, and historical retained-grid output. |
| Must not be concluded | Exact likelihood, source-faithful assembled route, unbiased pseudo-marginal target, posterior correctness, HMC convergence, cross-model default readiness, or statistical superiority from one seed. |

An implementation candidate failing an ESS or approximation gate is a candidate
failure, not a rejection of the Zhao-Cui research direction. Continue to the
predeclared tuning/repair phase unless a continuation veto fires.

## Skeptical Plan Audit

The plan is executable only after the implementing agent rechecks this audit
against the then-current tree.

| Risk checked | Required resolution |
| --- | --- |
| Wrong baseline | The old retained-grid result is historical diagnostic evidence, not a baseline or oracle. Use same-target SGQF and GenUT only as descriptive comparators. |
| Wrong target | Bind target ID, event order, parameter order, seed, and both hashes in route identity and artifacts. |
| Wrong scalar | Add an explicit source-order APF program; do not reuse the initial-observation-first evaluator. |
| Local score mistaken for filtering score | Validate the full recursive normalized-weight score of `L_N`; local complete-data density scores are inputs only. |
| Proposal dependence omitted | Require a parameter-independent prepared branch or implement all omitted proposal/pathwise derivatives. |
| Proxy promoted | FD, fit loss, KL, ESS, and one-seed comparator gaps diagnose the implementation; none alone proves likelihood accuracy or superiority. |
| Stale transferred defaults | Author and rung-2 settings are warm starts. Tune rank, degree, map, defensive mass, L1, auxiliary law, and fit budget for this target. |
| Environment mismatch | Offline CPU/FP64 reference checks are labelled reference-only; the claim-bearing online route is TensorFlow FP32, TF32 enabled, GPU, XLA, and memory-growth verified. |
| Non-answering artifact | The final JSON must contain value, six-coordinate score, branch/program IDs, target hashes, fixed controls, same-program audit, ESS/weight diagnostics, memory, device/XLA/TF32 status, and comparator values. |
| Hidden scaling failure | Report analytic and measured memory; forbid all-grid tensor-product storage and sample-wise XLA loops. |

This audit passes as a handoff design because the target and comparator roles
are explicit, the wrong existing route is excluded, the source-order mismatch
has an early unit test, and the artifacts directly answer availability. It does
not pre-approve any future implementation or leaderboard result.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| `T=20`, model dimensions | author `mainscript.m:12-17` and sealed row | reviewed target | wrong row | hash/time-order test |
| Physical parameter truth | `PredatorPreySSM.true_parameters()` | reviewed target | coordinate mismatch | manual score order and chain-rule test |
| `N=10000` | author `mainscript.m:45` | source warm start / final ladder arm | expensive or low effective diversity | first run N=1002 feasibility, then bounded ladder |
| N>1000 for numerical tests | owner instruction | required claim scope | tiny fixture accidentally promoted | manifest veto |
| TT degree/rank/source map | author `mainscript.m:48-67` | warm starts only | poor local fit or memory blow-up | target-specific validation and rank/degree ladder |
| Rung-2 Gaussian map and fit controls | synthetic rung-2 artifact | convenience warm start only | cross-model transfer fails | compare maps on calibration conditional loss and ESS |
| Positive defensive mass | paper Equation (13) | required mechanism | unsupported tails or invalid weights | support, normalization, and tail tests |
| Frozen predictive auxiliary | local extension | hypothesis | poor ESS away from reference theta | representative-theta ESS sweep |
| L1 regularization | repository Zhao-Cui policy | tuned default procedure | under/over-regularized TT | calibration/validation L1 grid with audit holdout |
| FP32 + TF32 + XLA online | repository default | required production-target arm | score/normalization precision loss | CPU FP64 tie-out then GPU tolerance audit |
| Same-program FD | numerical diagnostic | audit only | step-size artefact or accidental tuning to FD | multiple step sizes and untouched representative points |

## Proposed Architecture

### 1. Source-order data and model adapter

Use the sealed dataset factory. Add only the minimal adapter needed for the
fixed APF protocol, preferably without duplicating predator-prey dynamics. The
adapter must delegate densities and manual scores to `PredatorPreySSM` and bind:

- full-state Lebesgue measure;
- analytical manual-score backend;
- physical parameter order;
- exact model/source hashes.

### 2. Offline fixed proposal compiler

Build one fixed initial squared-TT/TTSIRT approximation and one fixed
adjacent-state proposal per observation time or a reviewed reusable stationary
proposal. The source operation approximates an adjacent target and obtains a
conditional through paired-core marginalization. Record the exact axis order.

The local compiler currently uses `(x_previous,x_current)` for prefix
conditioning, whereas the Zhao-Cui source orders variables differently. If
this order is retained, keep `extension_or_invention`. Never erase the reorder
from the manifest.

Freeze and hash at least:

- basis family and polynomial degree;
- TT rank caps and realized core shapes;
- L1 weight, ridge, fit samples, quadrature nodes, and fit sweeps;
- coordinate-map family and all scales/locations;
- positive defensive mass;
- fitted TT cores and mass contractions;
- conditional CDF/inverse schedule and tolerances;
- initial and transition reference points;
- ancestor uniforms and resulting ancestors;
- auxiliary categorical laws;
- generated state clouds and evaluated proposal log densities;
- dataset and source dependency closure.

All preparation occurs before XLA tracing. Runtime evaluation must not refit,
adapt rank, redraw, resample, retune, or select a parameter-dependent branch.

### 3. Online source-order value/score program

Implement the mathematical recursion above with TensorFlow tensor operations.
The default evaluator uses `tf.function(..., jit_compile=True)`. Requirements:

- no NumPy import or NumPy numerical computation in runtime, tuning,
  admission, artifact construction, or XLA paths;
- no Python sample loop;
- no Python time loop in the XLA path;
- no `tf.map_fn`/`tf.vectorized_map` wrapper around a scalar sample evaluator;
- no runtime `GradientTape` or finite differences;
- no host materialization used to make numerical decisions inside the route;
- no fallback to the generic retained-grid evaluator.

### 4. Repository identity and admission

Issue route identity from repository-owned callables and frozen settings. Do
not accept a caller-supplied canonical identity. Only after all gates pass:

- add the new route ID to
  `ZHAO_CUI_FIXED_VARIANT_ROUTE_IDS` in
  `bayesfilter/highdim/source_route.py`;
- add any required exports in `bayesfilter/highdim/__init__.py`;
- add the route to the leaderboard route mapping;
- replace the blocker in
  `_zhao_cui_predator_prey_tt_cell()` with the new evaluator;
- retain the old route in `ZHAO_CUI_HISTORICAL_ROUTE_IDS`;
- add a negative test proving the historical route still cannot fill the cell.

Adding an ID to the allow-list before the implementation gates pass is
forbidden.

## Memory And Performance Contract

The route must not materialize an `M^d` state grid, all tensor-product retained
states, or dense all-axis transition pairs. Linear storage in particles and
state dimension is acceptable for the fixed branch.

For dtype byte width `b`, report at least these theoretical terms:

```text
state clouds:              (T+1) * N * d_x * b
transition proposal logs:  T * N * b
auxiliary log laws:        T * N * b
ancestor indices:          T * N * 4
score working set:         O(N * p * b)
TT cores:                  sum_j ell_j * r_{j-1} * r_j * b
paired-core contractions:  O(sum_j r_j^2 * b), excluding fit workspace
```

Also report offline fit workspace and measured TensorFlow allocator current and
peak bytes. At `T=20`, `N=10000`, `d_x=2`, FP32 state clouds alone are about
1.68 MB; the route should remain linear rather than turning this small target
into a retained-grid benchmark.

Use `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import and the repository
memory-policy helper before any logical device or tensor initializes the GPU.
Fail closed if growth cannot be verified. Record TF32, XLA, physical/logical
device, memory-policy schema, allocator current/peak, compile time, steady-state
runtime, and wall time.

The high-dimensional motivation is scalability, but this task tests only the
predator-prey model. Do not claim NAWM or another high-dimensional nonlinear
model has been tested. Report the asymptotic memory formula so later work can
judge transfer.

## Tuning Protocol

Tuning is offline and target-specific. Author and synthetic rung-2 settings are
warm starts, not defaults.

1. Create deterministic, disjoint calibration, validation, and untouched audit
   designs. The sealed seed-81104 claim data must not be used to choose controls.
2. Evaluate a bounded ladder over coordinate map/scale, polynomial degree, TT
   rank, positive defensive mass, L1 weight, ridge, quadrature/fit budget, and
   auxiliary-law choice.
3. Respect the Zhao-Cui L1 policy: include positive L1 arms and a zero-L1
   comparator, then select on validation without using the audit set.
4. Use proposal conditional log-density/normalization error, heldout fit error,
   ESS, weight spread, and memory as tuning quantities. Do not require an exact
   likelihood or score oracle.
5. Use same-program FD at a small, randomly selected set of representative
   interior parameter points as a score-implementation diagnostic. Do not use
   FD at runtime or tune the final scientific estimate to minimize an oracle
   score gap.
6. Freeze the selected controls and branch before the untouched target run.
7. If the untouched claim fails, preserve it and retune on fresh calibration
   partitions; do not tune on that failed claim artifact.

Suggested bounded numerical ladder:

- primitive/unit fixtures may use tiny N and CPU FP64 but cannot support a
  numerical or leaderboard claim;
- first source-order GPU feasibility: `N=1002`, matching the existing GenUT
  one-seed feasibility scope and satisfying N>1000;
- stability arm: at least one larger N, recommended `N=5000`;
- source-setting arm: `N=10000`, from author `mainscript.m:45`, if the earlier
  arms pass and the campaign remains within budget.

Do not infer superiority from one seed or from monotone-looking N differences.

## Implementation Phases

### Phase 0: source audit and current-tree inventory

- Recheck all source anchors and write an operation-classification ledger.
- Recheck `git status`; preserve unrelated work and the concurrent rung-2
  files.
- Confirm no other agent has already added a predator-prey fixed route.
- Freeze the target ID, hashes, event order, and expected route class.
- Record a concise campaign plan, total attempt/compute budget, versioned output
  root, and stop conditions before long experiments.

Exit: target/source/classification ledger reviewed; otherwise
`BLOCK_SOURCE_UNGROUNDED`.

### Phase 1: source-order branch and scalar

- Add source-order branch shapes `[T+1,N,d]` and `[T,...]` updates.
- Implement an independent tiny scalar statement in a test.
- Prove by test that no observation is assimilated at `x0`.
- Prove the old initial-observation-first evaluator gives a different program
  on a nondegenerate fixture.

Exit: independent source-order value parity and deterministic replay pass.

### Phase 2: analytical score

- Implement the normalized-weight derivative recursion.
- Use only the three manual model density-score methods.
- Test each of six physical coordinates against same-program central FD on tiny
  and representative fixtures.
- Independently compare local model density scores to `GradientTape` in tests
  only.
- Bind value and score to the same branch/program ID.

Exit: all six coordinates pass reviewed FP64 and FP32 diagnostic tolerances;
runtime provenance contains no FD/autodiff.

### Phase 3: target-specific TTSIRT proposal

- Build initial and adjacent-state squared-TT approximations.
- Verify positive defensive mass, paired-core marginals, conditional
  normalization, inverse/forward consistency, and proposal log-density
  consistency.
- Classify every deviation from author ordering or CDF inversion.
- Ensure no generic retained-grid storage marker is present.

Exit: proposal mechanics and source-classification tests pass.

### Phase 4: offline tuning and freeze

- Run the bounded target-specific ladder on calibration/validation partitions.
- Tune L1 explicitly.
- Select controls without inspecting untouched claim outputs.
- Emit a repository-owned frozen tuning/branch artifact and exact hashes.

Exit: selected branch is noncollapsed, deterministic, and within memory budget.

### Phase 5: CPU reference and GPU/XLA feasibility

- Run tiny/representative CPU FP64 reference checks, explicitly labelled
  reference-only.
- Run FP32, TF32-enabled, GPU/XLA N=1002 feasibility with memory growth.
- If valid, run larger predeclared N arms within budget.
- Record compile and steady-state timing separately.

Exit: target run is finite and every hard implementation veto passes.

### Phase 6: comparator and regression evidence

- Compare descriptively with same-target SGQF and GenUT.
- Do not use the wrong-time-order UKF.
- Preserve the historical retained-grid result as a labelled negative control,
  not a comparator eligible for promotion.
- Run focused regression tests for LGSSM, actual SV, KSC-SV, generalized SV,
  spatial SIR, predator-prey SGQF, and GenUT route availability/identity.

Exit: no route-registry, target-identity, or existing-model regression.

### Phase 7: admission and leaderboard wiring

- Add route/export/mapping only after Phases 0-6 pass.
- Make the leaderboard execute the new fixed route and fail closed if its
  identity, target, score provenance, or branch hash is absent.
- Add a test proving caller-stamped or historical IDs cannot gain admission.
- Generate a new one-seed leaderboard feasibility artifact in a fresh output
  directory.

Exit: the Zhao-Cui predator-prey cell reports `executed_value_score` from the
new route and all admission tests pass.

## Required Tests And Acceptance Gates

### Identity and semantics

- exact target ID, state hash, observation hash, seed, T, and event order;
- exact physical parameter order and truth;
- route ID issued from actual callables and settings;
- branch/program identity changes if any frozen numerical object changes;
- replay determinism across repeated evaluations;
- no initial observation at `x0`;
- no retained-grid route marker or fallback.

### Proposal mathematics

- squared-TT density nonnegative;
- defensive mass strictly positive;
- mass/measure/Jacobian conventions consistent;
- marginal normalization valid;
- conditional density integrates to one within reviewed tolerance;
- conditional generation and evaluated log proposal agree;
- ancestor auxiliary law normalized and correction indexes the same ancestor;
- inverse/forward roundtrip valid;
- finite proposal log density for every realized state;
- ESS and maximum log-weight spread emitted at each time.

### Value and score

- finite scalar value and six finite physical score coordinates;
- independent tiny value implementation agrees;
- increment sum equals total value;
- increment-score sum equals total score;
- score and value share program/branch identity;
- representative same-program central FD diagnostic passes in FP64 and FP32;
- runtime score uses only manual analytical recursion;
- no `GradientTape`, runtime finite difference, adaptive refit, or
  parameter-dependent discrete branch;
- local complete-data score is not reported as the filtering score.

### TensorFlow/XLA/runtime

- TensorFlow/TFP only in candidate runtime;
- no NumPy dependency in runtime, tuning, admission, or artifact construction;
- no Python/sample-wise loop in XLA;
- JIT defaults true;
- TF32 enabled for the claim-bearing GPU arm;
- output tensors placed on GPU;
- memory growth configured and verified before initialization;
- allocator current/peak, device, XLA, TF32, dtype, compile time, and steady
  runtime recorded;
- no all-grid tensor-product or dense all-pairs retained transition storage.

### Statistical interpretation

- hard veto status reported first;
- one-seed continuous differences labelled descriptive only;
- no ranking unless a later predeclared multi-seed uncertainty analysis supports
  it;
- candidate rejection distinguished from direction rejection;
- no default-readiness claim from this one model.

## Existing Same-Target Comparator Evidence

These values are useful only as descriptive plausibility checks. Neither route
is an exact predator-prey likelihood oracle.

From
`docs/benchmarks/artifacts/one_seed_four_filter_feasibility_20260722/attempt03/result.json`:

| Route | N | Value | Score `(r,K,a,s,u,v)` |
| --- | ---: | ---: | --- |
| Fixed SGQF | deterministic quadrature | `-102.6227035213447` | `(-27.64114285, 0.08410678, -0.08414332, 0.85569906, 17.52559777, -22.63497837)` |
| GenUT | `1002` | `-102.58187866210938` | `(-26.75030708, 0.17595635, -0.09248146, 0.56542557, 19.85107040, -25.52731323)` |

The old retained-grid route reported approximately:

```text
value = -179.9239394489
score = (141.1773081, 6.6510046, 0.1645909,
         -61.2771959, -5.4411600, 6.2934740)
```

That evidence is preserved in
`docs/plans/bayesfilter-highdim-leaderboard-remaining-blockers-phase1-predator-prey-result-2026-07-02.md`.
It is historical diagnostic evidence only. Its large disagreement motivated
the current scrutiny, but it cannot be relabelled as the required route or used
as an oracle.

## Expected Artifacts

Use fresh, versioned paths. Do not overwrite any existing artifact.

Required outputs:

- source and operation-classification ledger under `docs/plans/`;
- implementation plan/evidence contract under `docs/plans/` before serious
  execution;
- target-specific tuning result and frozen branch manifest;
- CPU reference JSON/Markdown;
- GPU/XLA feasibility JSON/Markdown and run manifest;
- final implementation result note with decision and inference-status tables;
- reset memo stating exact route availability and remaining nonclaims;
- focused test logs or structured summaries;
- leaderboard result artifact after admission.

Every serious run manifest must include Git commit and dirty state, exact
command, conda environment, device and memory policy, dtype/TF32/XLA, seed,
particle count, target/data hashes, branch/program IDs, frozen tuning artifact,
wall time, output paths, plan path, and result path.

## Forbidden Fallbacks

- Do not call or wrap the generic multistate retained-grid value/score path.
- Do not remove its historical label or add its ID to the fixed-route allow-list.
- Do not use the P47 two-observation fixture.
- Do not use initial-observation-first data or UKF as a same-target comparator.
- Do not return only the sum of local complete-data scores.
- Do not use runtime finite differences or autodiff.
- Do not rebuild TT fits, ranks, maps, random points, or genealogy as `theta`
  changes.
- Do not silently ignore proposal/auxiliary derivatives if those objects depend
  on runtime `theta`.
- Do not use NumPy in runtime/tuning/admission/artifact code.
- Do not add a Python sample loop or scalar row mapper to the XLA path.
- Do not transfer synthetic rung-2 controls as reviewed predator-prey defaults.
- Do not add the route ID to admission registries before its gates pass.
- Do not claim source-faithfulness for an assembled extension.
- Do not claim an exact likelihood, pseudo-marginal correctness, HMC
  convergence, NAWM validation, or statistical superiority.

## Stop Conditions And Handoff Result

Stop and return a blocker note instead of forcing leaderboard availability if:

- the paper/author-source audit contradicts the proposed proposal/correction;
- exact proposal densities for generated states cannot be computed;
- the source-order finite scalar cannot be defined consistently;
- the score requires omitted runtime-parameter dependence that is not derived;
- a target/hash/time-order mismatch appears;
- the only working implementation falls back to retained-grid storage;
- the assembled `extension_or_invention` route is not admissible under the
  current Zhao-Cui leaderboard policy;
- the bounded campaign budget is exhausted.

The implementation is complete only when the new target-specific fixed route,
not the historical retained-grid route, produces the predator-prey T20 value
and manual same-program score on GPU/XLA and the leaderboard admits that exact
repository-issued identity.
