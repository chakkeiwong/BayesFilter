# Predator-Prey SGQF Leaderboard Repair: Agent Handoff

Date: 2026-07-22

Status: `READY_FOR_INDEPENDENT_AGENT_EXECUTION`

## Assignment

Take ownership of the predator-prey SGQF comparison lane while the primary
agent continues GenUT work. Audit and promote the existing correct
initial-observation-first PP-SGQF value/manual-score implementation into a
reusable repository-owned runtime route, issue an inspectable route identity,
and wire it into the canonical predator-prey T20 leaderboard cell. Write a
self-contained result note under `docs/plans/` when finished.

Do not modify GenUT code or artifacts. Do not change the generic SGQF time
convention for other models. Do not revive the historical wrong predator-prey
SGQF value or historical Zhao-Cui retained-grid route.

## Why This Work Is Needed

The canonical leaderboard row is:

```text
zhao_cui_predator_prey_T20
```

Its timing is:

1. draw `x_0 ~ N([50,5], I)`;
2. assimilate `y_0 | x_0 ~ N(x_0, 4I)`;
3. for `t=1,...,19`, propagate the predator-prey RK4 flow and add
   `N(0,4I)` process noise, then assimilate `y_t`.

The generic route
`bayesfilter/nonlinear/fixed_sgqf_tf.py::tf_fixed_sgqf_filter` instead applies
the transition before every observation, including `y_0`. Its previously
reported predator-prey value `-171.368581` and associated score are therefore
wrong relative to the canonical target, not merely low-accuracy
approximations. The leaderboard now blocks this route at:

```text
docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py::_predator_reference_and_sgqf
tests/test_two_lane_highdim_leaderboard_phase4.py
```

The block must remain until a correct replacement passes all gates below.

Exact source anchors for the mismatch are:

```text
bayesfilter/nonlinear/fixed_sgqf_tf.py:976    generic value entry point
bayesfilter/nonlinear/fixed_sgqf_tf.py:1039   time loop starts at observation 0
bayesfilter/nonlinear/fixed_sgqf_tf.py:1073   unconditional transition
bayesfilter/nonlinear/fixed_sgqf_derivatives_tf.py:225  generic score entry
bayesfilter/nonlinear/fixed_sgqf_derivatives_tf.py:285  score loop at 0
```

The canonical `PredatorPreySSM` model and simulation order are anchored in
`bayesfilter/highdim/models.py:1517` and `bayesfilter/highdim/models.py:1724`.

## Important Discovery: The Correct Algorithm Already Exists

Do not reimplement the recurrence blindly. A correct graph-native,
initial-observation-first PP-SGQF implementation already exists in:

```text
bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py
```

Important entry points are:

```text
pp_sgqf_likelihood_value_score_status        line 95
pp_sgqf_likelihood_value_only_status         line 408
make_predator_prey_sgqf_neutra_adapter       line 727
make_predator_prey_sgqf_target_contract      line 763
```

The exact analytic `y_0` block is at lines 111-142; the score loop is initialized
at index 1 at lines 372-388. The independent value-only route also starts at
index 1 at lines 538-552. Treat line numbers as review anchors, not immutable
identity fields; the issued route identity must bind source content.

The score route:

- analytically assimilates `y_0` under the initial Gaussian;
- initializes the filtered mean/covariance and their zero parameter tangents;
- starts its TensorFlow time loop at index `1`;
- propagates exact RK4 state and source-parameter sensitivities;
- differentiates Cholesky factors, sigma-point means/covariances, innovation
  covariance, gain, filtered moments, and every log-likelihood increment;
- uses no runtime autodiff or finite differences for the score; and
- is batch-native and XLA-compatible.

This code currently lives under `bayesfilter/testing`, so it must not be
silently treated as an admitted reusable leaderboard runtime. Audit it, extract
or promote it into an appropriate non-testing TensorFlow runtime module, and
leave the existing testing/NeuTra adapter as a compatibility wrapper if needed.

Do not use
`tf_predator_prey_to_fixed_sgqf_model(..., with_derivatives=True)` as the
replacement score path. Its derivative adapter uses `tf.GradientTape` in
`bayesfilter/nonlinear/fixed_sgqf_structural_adapter_tf.py:94-111`, which is
wrong relative to the required manual analytical runtime-score architecture.

## Mathematical Target

Let physical parameters be

```text
theta = (r, K, a, s, u, v)
      = (0.6, 114, 25, 0.3, 0.5, 0.5).
```

The continuous-time drift is

```text
d prey/dt = r prey (1 - prey/K) - s prey predator/(a + prey)
d predator/dt = u prey predator/(a + prey) - v predator.
```

The discrete transition mean is the RK4 solution over `delta=2.0` using 20
substeps of size `0.1`. The state and observation equations are

```text
x_t = RK4_theta(x_{t-1}) + eta_t,   eta_t ~ N(0,4I),  t >= 1
y_t = x_t + epsilon_t,              epsilon_t ~ N(0,4I).
```

For the first observation, with `m_0=[50,5]`, `P_0=I`, and `R=4I`, compute

```text
S_0 = P_0 + R
ell_0 = log N(y_0; m_0, S_0)
G_0 = P_0 S_0^{-1}
m_0^+ = m_0 + G_0 (y_0-m_0)
P_0^+ = P_0 - G_0 S_0 G_0^T.
```

These quantities do not depend on `theta`, hence their physical-parameter
tangents and the score contribution from `ell_0` are zero. The SGQF recursion
then starts from `(m_0^+,P_0^+)` at `t=1`.

For each later step, the same finite scalar must produce both value and score:

```text
ell(theta) = ell_0 + sum_{t=1}^{19} ell_t(theta)
score(theta) = d ell(theta) / d theta.
```

The runtime score must be the hand-derived recursive derivative of this exact
finite SGQF value program. Finite differences are validation only.

### Coordinate Conversion

The existing PP-SGQF implementation accepts an unconstrained six-probit
coordinate `z` and emits `d ell/dz`. The leaderboard uses physical coordinates.
The chart is

```text
theta_j = lower_j + (upper_j-lower_j) Phi(z_j),
d theta_j/dz_j = (upper_j-lower_j) phi(z_j).
```

For interior physical parameters,

```text
z_j = Phi^{-1}((theta_j-lower_j)/(upper_j-lower_j)),
d ell/d theta_j = (d ell/dz_j) / (d theta_j/dz_j).
```

Implement this conversion analytically in TensorFlow. The leaderboard
likelihood route must not add the Uniform prior or chart-Jacobian terms used by
the posterior/NeuTra adapter. Verify that the emitted value is the likelihood
only and the score is in physical order `(r,K,a,s,u,v)`.

## Prior Evidence To Reuse, Not Silently Promote

The July 15 admission campaign selected sparse level 2:

```text
docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/
  phase-p4/PP-SGQF/target-admission/
  attempt-01-20260715T123720Z/result.json
```

SHA-256:

```text
6eea9ab2f4cf0e5a23262ed450d8b85add27289ab4aeef3a9661b95d46c358c4
```

Its target identity is:

```text
docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/
  phase-p4/PP-SGQF/target-admission/
  attempt-01-20260715T123720Z/target_identity.json
```

SHA-256:

```text
9b9f31f773791e094a39be5cf821ebdd0f24c0197f5b32b114aaf249f4b3c4f4
```

The admission result records:

```text
selected level                         2
level-2 point count                    5
maximum level-2 vs level-5 value gap   0.0001784821
maximum level-2 vs level-5 score gap   0.0003973026
truth-point level-2 likelihood         -103.1378913390
truth-point level-3 likelihood         -103.1378763997
truth-point level-5 likelihood         -103.1378763997
```

This evidence is strong but historical relative to the current leaderboard
integration. Recheck source hashes, target/data identity, current runtime
policy, and current tests. Do not copy the old typed NeuTra identity and call it
a leaderboard route identity.

The prior result and design notes are:

```text
docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p4-pp-sgqf-level-design-2026-07-15.md
docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p4-pp-sgqf-comparator-result-2026-07-16.md
docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p4-predator-prey-result-2026-07-16.md
docs/benchmarks/run_multimodel_neutra_p4_predator_prey_sgqf_admission.py
tests/test_predator_prey_sgqf_neutra_target.py
```

## Current Same-Target Numerical Anchors

The July 22 GenUT campaign provides an untouched canonical seed-81104
comparison artifact:

```text
docs/benchmarks/artifacts/
  genut_predator_prey_leaderboard_continuation_20260722/
  attempt01/result.json
```

SHA-256:

```text
45fc63debeb0e53cc8d2f28ed56d60adf9cf5689dafa98a8d5d32e8db8607e74
```

At the truth point:

```text
refined bootstrap PF N=262144 mean   -103.1376759768
bootstrap PF 95% CI                  [-103.1444706,-103.1308814]
principal-square-root UKF            -103.1378616014
correct historical level-2 SGQF      -103.1378913390
GenUT N=1002 mean                    -103.1618270874
GenUT 95% CI                         [-103.3303054,-102.9933487]
```

The corrected SGQF value should be near `-103.13789`, not `-171.37`.

Reference implementations:

```text
bayesfilter/testing/predator_prey_bootstrap_pf_reference_tf.py
bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py
```

Bootstrap PF is a stochastic value authority only; it supplies no score truth.
UKF is a same-target analytical approximation diagnostic, not an oracle.

## Required Work Phases

### Phase 0: Audit And Freeze

Write a concise experiment plan before any serious GPU run. Freeze:

- target row, observation hash, state hash, timing, physical parameter order;
- existing level-2/3/5 cloud identities and source closure;
- exact distinction between likelihood and posterior targets;
- score coordinate convention;
- comparators and their diagnostic roles; and
- compute budget, versioned output root, and stop conditions.

Skeptically audit wrong baselines, stale evidence, source/test environment
drift, claim-data leakage, and whether each command can answer its question.

### Phase 1: Source And Mathematics Audit

Audit the existing corrected recurrence line by line. Confirm:

- `y_0` is assimilated exactly once before the time loop;
- the loop begins at `t=1` and ends after `t=19`;
- RK4 state and parameter tangents match the physical model;
- covariance, Cholesky, innovation, gain, and filtered-moment derivatives are
  complete;
- score increments differentiate the exact same finite scalar;
- no posterior prior/Jacobian terms enter the likelihood endpoint;
- level/node weights and negative-weight handling match the frozen SGQF rule;
- no NumPy, runtime finite differences, runtime autodiff, Python sample loop,
  `tf.map_fn`, `tf.vectorized_map`, or callback enters the active route; and
- XLA shapes are static where required.

Write the checked derivation in the result note or a dedicated mathematical
note. Directly classify any mismatch as wrong relative to the target.

### Phase 2: Runtime Extraction And Identity

Extract the reusable likelihood value/manual-score kernel from
`bayesfilter/testing` into an appropriate TensorFlow runtime module. Prefer a
small model-specific module over changing generic SGQF semantics globally.

Requirements:

- TensorFlow/TFP only; no NumPy;
- XLA JIT enabled by default;
- graph-native batch support;
- repository-owned immutable route identity binding target/data hash, sparse
  level, cloud construction, timing, parameter chart, value/score callable
  source closure, dtype, XLA, and backend;
- caller cannot stamp or override identity;
- physical likelihood endpoint and physical analytical score endpoint; and
- existing testing/NeuTra imports remain compatible or are migrated with
  focused tests.

The shortest bounded implementation may wire the existing corrected likelihood
endpoint first and record the `bayesfilter/testing` namespace as migration debt.
The preferred terminal implementation extracts the kernel into a non-testing
module and keeps compatibility imports. In either case, do not duplicate the
recurrence.

Suggested route identifier:

```text
predator_prey_initial_observation_first_fixed_sgqf_level2_manual_score_v1
```

Do not use this identifier until the repository factory actually binds the
callable and settings.

### Phase 3: Focused Correctness Tests

At minimum add tests for:

1. exact analytic `y_0` value and filtered moments;
2. `T=1` performs no transition, equals `log N(y0;[50,5],5I)`, and has zero
   physical likelihood score for every interior parameter point;
3. `T=2` performs exactly one transition;
4. full `T=20` timing and observation hash;
5. likelihood-only value excludes prior and chart-Jacobian terms;
6. physical score order and analytic source-to-physical chain rule;
7. recursive score versus representative-point central FD of the same scalar,
   with FD diagnostic-only;
8. value endpoint equals score endpoint's scalar;
9. levels 2/3/5 reproduce the convergence ladder;
10. batch permutation, CPU-hidden reference, GPU/XLA, finite/status and
    positive covariance gates;
11. repository identity rejects caller stamping, stale source, wrong target,
    wrong data, wrong level, and wrong timing; and
12. the old generic transition-before-`y_0` route remains blocked for this row.

Also use an autodiff-runtime sentinel or equivalent source/runtime guard. An
existing pattern is `AutodiffRuntimeSentinel` in
`scripts/audit_ledh_no_autodiff.py:504`. AST checks alone are useful but do not
replace executing the sentinel around the score endpoint.

### Phase 4: Fresh Bounded Evidence

Run a fresh target-matched claim on seed 81104 after source extraction. Use the
current refined bootstrap-PF value artifact and current UKF analytical
diagnostic. Reproduce at least levels 2, 3, and 5 at the truth point and the
existing four frozen audit points.

Primary promotion criteria:

- level 2 value and every physical score coordinate agree with level 5 within
  predeclared margins no weaker than the July 15 gates (`0.25` value, `0.5`
  score coordinate);
- truth-point level-2 value lies inside the current refined PF 95% interval
  expanded only by a predeclared practical margin justified before results;
- analytical score passes same-scalar FD audit at representative points;
- value and score are finite and target/status/covariance gates pass;
- trusted GPU, XLA, memory growth, device and allocator evidence are recorded;
  and
- fresh route identity validates.

PF and UKF comparisons are diagnostics. Do not claim SGQF superiority merely
because its truth-point value is close to PF or UKF.

### Phase 5: Leaderboard Integration

Replace the blocked fixed-SGQF predator-prey cell only after all earlier gates
pass. Update:

```text
docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py
tests/test_two_lane_highdim_leaderboard_phase4.py
tests/test_two_lane_highdim_leaderboard_analytical_scores.py
```

If the GenUT continuation harness still emits a deterministic SGQF comparator,
update its stale blocked diagnostic consistently without changing any GenUT
algorithm, tuning, value, score, or artifact:

```text
docs/benchmarks/run_genut_predator_prey_leaderboard_continuation.py
```

The admitted row must report:

- `comparison_status = executed_value_score`;
- canonical target and data identities;
- physical parameter score and coordinate label;
- one value/score route identity for the same scalar;
- manual/recursive analytical derivative provenance;
- sparse level and cloud identity;
- FP/GPU/XLA status appropriate to the actual claim artifact;
- reference/diagnostic roles without calling PF or UKF an oracle; and
- explicit nonclaims.

Run focused leaderboard completeness and analytical-score tests. Do not alter
GenUT integration or Zhao-Cui status in this assignment.

## Evidence Contract

| Field | Requirement |
|---|---|
| Scientific question | Can the already-correct initial-observation-first level-2 PP-SGQF likelihood/manual-score route be promoted into the canonical predator-prey T20 leaderboard without changing target, scalar, or derivative semantics? |
| Baseline | Level-5 SGQF consistency authority; refined bootstrap PF for value only; UKF for analytical same-target diagnostic only. |
| Primary criterion | Fresh level-2 value/physical-score consistency with level 5 plus same-scalar analytical-score validation and valid route identity. |
| Hard vetoes | Wrong `y_0` timing, posterior/likelihood substitution, partial or autodiff runtime score, nonfinite/status/covariance failure, stale/caller-stamped identity, source/data mismatch, CPU fallback in GPU claim, or failed value/score scalar identity. |
| Explanatory only | Runtime, UKF agreement, PF point difference, HMC history, and sparse-level point count. |
| What must not be concluded | Exact likelihood or score, SGQF superiority, broad nonlinear validity, default promotion, GenUT result, Zhao-Cui validity, or high-dimensional readiness. |
| Required artifact | Fresh versioned JSON result, serious-run manifest, raw level rows, source hashes, route identity, and self-contained Markdown result note. |

## Default And Assumption Audit

| Choice | Provenance | Current status | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Sparse level 2 | July 15 frozen level ladder | Strong warm start, not silently current default | Source/target drift invalidates old convergence | Reproduce levels 2/3/5 first |
| Initial observation first | Canonical dataset/target | Required target fact | Transition-before-`y_0` computes another likelihood | `T=1` zero-transition test |
| Six-probit source chart | Existing HMC adapter | Existing internal coordinate only | Posterior or source score leaks into physical leaderboard row | Physical/source chain-rule test |
| Float64 SGQF | Existing admitted implementation | Baseline hypothesis | Environment or GPU kernel mismatch | CPU-hidden/GPU-XLA parity |
| PF practical margin 1.0 | July 15 admission plan | Historical reviewed margin, not automatic | Overly wide margin masks error | Predeclare or justify before fresh claim |
| Level 5 authority | July 15 convergence ladder | Deterministic approximation authority | Shared SGQF bias across levels | Keep PF/UKF diagnostics and nonclaims |

## Implementation Constraints

- Preserve unrelated dirty worktree changes.
- Use `apply_patch` for edits.
- TensorFlow/TFP is the algorithmic backend.
- No NumPy in runtime, tuning, identity, or leaderboard paths.
- No runtime finite-difference score and no autodiff score.
- XLA JIT defaults on.
- GPU work requires trusted/escalated execution and verified memory growth.
- Do not change `tf_fixed_sgqf_filter` globally unless a separate audit proves
  every caller's timing semantics and tests all affected models. A
  predator-prey-specific extracted kernel is the safer default.
- Do not modify GenUT files or artifacts.
- Do not unblock the leaderboard from historical evidence alone.

## Expected Result Note

Write:

```text
docs/plans/bayesfilter-predator-prey-sgqf-leaderboard-repair-result-2026-07-22.md
```

It must include:

- claimed target versus quantity actually computed;
- derivation/source anchors;
- files changed;
- commands and environment;
- per-level value and physical-score table;
- PF and UKF diagnostic comparison with correct roles;
- recursive-score/FD validation table;
- route-identity and leaderboard status;
- engineering, numerical, and scientific ledgers;
- decision and inference-status tables;
- strongest alternative explanation and overturning evidence; and
- remaining nonclaims/gaps.

If a gate fails, keep the leaderboard cell blocked, preserve the failure
artifact, identify whether the failure is implementation, tuning, diagnostic,
or evidence against level-2 SGQF, and state the smallest justified repair.
