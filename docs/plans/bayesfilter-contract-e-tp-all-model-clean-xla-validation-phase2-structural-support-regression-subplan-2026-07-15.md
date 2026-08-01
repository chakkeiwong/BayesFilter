# Phase 2: Structural Support Regression Subplan

Date: 2026-07-15

Status: `PASS_CLOSED_HANDOFF_READY`

Review record: bounded review round 1 returned `REVISE` on fixed-index scope,
the carried-state total tangent, unsupported floating-point bounds, exact
negative-control factory identity, manifest fields, and budget wording. The plan
was patched. A focused second review isolated an unsupported `tanh` kernel error
assumption; the guard was narrowed to algebraic recombination conditional on
opaque recorded TensorFlow kernel values/Jacobians. Final focused review returned
`VERDICT: AGREE`.

## Objective And Claimed Target

Extend the exact two-state structural fixture from one teacher/projection step
to a fixed-shape recursive TensorFlow program using `tf.while_loop`. Verify at
every step that candidates and selected students remain on the declared
innovation-space deterministic-completion support and that the total tangent of
the executed finite scalar matches an unrolled reference and same-scalar FD.

The target is this finite engineering fixture only. It is not a DSGE/NAWM
likelihood, SIR repair, nonlinear filtering accuracy result, or model adapter.

## Entry Conditions

- Phase 1 is `PASS_CLOSED_HANDOFF_READY`; its final guardrail JSON SHA-256 is
  `344d8480621affbb89c048bdaf61d4e9660ebc2a4ac002c82e41594b38cd370a`.
- The existing one-step fixture passes support, exact completion, total tangent,
  positive fixed chart, off-route rejection, and no-NumPy/jitter checks.
- The fixture has four parents, three innovation points, four selected anchors,
  state dimension two, innovation dimension one, and parameter dimension four.
- Contract E--TP remains experimental; no full model row is admitted by this
  fixture.

## Mathematical Invariants

Let the carried cloud at step `t-1` be `x_{t-1}^i` with positive normalized
weights `w_{t-1}^i`, and let `(e^j,v^j)` be the fixed innovation rule. The
teacher is

`X_t^{ij} = F_theta(x_{t-1}^i,e^j)`,
`W_t^{ij} = w_{t-1}^i v^j`.

The deterministic residual is `c_theta(x_{t-1}^i,e^j,X_t^{ij})`. By
construction it must vanish up to evaluated floating-point roundoff. Contract
E--TP uses caller-prepared indices that are fixed before execution and selects
only indices of this teacher, so every carried student point is one of the
`X_t^{ij}` and remains on support. No runtime index or chart selection is part
of the claimed program. Repeating this argument proves the support invariant by
induction for this fixed-index, fixed-chart finite recursion.

For a carried parent `x_{t-1}(theta)`, define the checked recursive identity

`g_t(theta) = c_theta(x_{t-1}(theta), e,`
`                         F_theta(x_{t-1}(theta),e)) = 0`.

Its total derivative is

`D g_t = partial_theta c + partial_x c D x_{t-1}`
`        + partial_X c (partial_theta F + partial_x F D x_{t-1}) = 0`.

The tested tangent must contain every displayed term through the carried state
and transition. It is the derivative of the fixed-index, fixed-chart executed
finite program; a partial derivative holding parents fixed or a derivative of a
selection-changing program is wrong relative to this target. The compiled
scalar is a declared smooth function of projected matched targets, and its
reverse autodiff must match the same unrolled finite program and central FD.

### Numerical Support Guards

The mathematical target is exact zero. Floating evaluation uses two
implementation-specific forward-error envelopes, which are numerical guards,
not scientific tolerances or theorems about arbitrary TensorFlow kernels.

Let `u = eps_float64/2` and `gamma_k = k*u/(1-k*u)`. This guard makes no claim
about the error of TensorFlow/XLA's `tanh` kernel relative to mathematical
`tanh`. Instead, record the evaluated kernel value `z=TF_tanh(s)` and require
the transition and residual to consume the same stored `s` and reproduce the
same bitwise `z`. Conditional on these opaque evaluated kernel outputs, an
audited scalar recombination of transition plus residual uses at most 16 rounded
arithmetic operations per point. Define

`tau_value = gamma_16 * max(1, sum(abs(value-expression terms)))`.

For the explicitly expanded total-tangent identity above, treat TensorFlow's
evaluated `tanh` autodiff Jacobian as another recorded opaque kernel output. The
hand-counted algebraic recombination graph after those outputs uses at most 64
rounded arithmetic operations per parameter component. Define

`tau_tangent = gamma_64 * max(1, sum(abs(expanded tangent terms)))`.

The harness must emit the operation-count ledger, bitwise repeated-kernel-output
check, and every term scale used in these envelopes; tests bind counts `16` and
`64` to the exact fixture formula. Changing the formula requires recounting and
review. The primary checks are
`abs(c)<=tau_value` and `abs(Dg)<=tau_tangent`, plus loop/unrolled parity. These
envelopes may be conservative; they are not cross-model defaults and do not
establish transcendental-kernel or mathematical-`tanh` accuracy.

The off-support perturbation is `delta=1e-3` in the deterministic coordinate.
Before accepting it as a negative control, the harness must verify and record
`delta >= 10^6 * max(tau_value)` on the actual frozen fixture. If that
separation does not hold, the control is invalid evidence rather than a reason
to change the bound after observing results.

## Required Implementation

Add a loop-native structural fixture core in
`bayesfilter/highdim/ledh_contract_e_tp_structural_tf.py` with:

- static first/intermediate state shapes and `tf.while_loop`;
- parent-by-innovation expansion, normalized product weights, model-owned
  parameterized transition/residual/features, fixed per-step charts and row
  scales;
- histories for support residual, support bound, chart validity, minimum weight,
  feature residual, and scalar increment;
- Boolean validity combined from support, chart, finite, and positive-weight
  predicates;
- numerical poisoning of carried parents, weights, scalar, and score when
  invalid, including under XLA where assertion operators may be ignored;
- a `tf.function(..., jit_compile=True)` factory returning value, total score,
  diagnostics, and graph state.

The fixed active indices, charts, row scales, horizon, and all valid/invalid
inputs are bound through one repository-owned factory. The compiled negative
control calls the exact same concrete factory/configuration and differs only in
an input tensor that injects the declared deterministic-coordinate
perturbation. It must return `valid=false` and nonfinite scalar, score, carried
state, and carried weights. An alternate negative-control branch or factory is
not evidence for this gate.

No Python horizon/history loop may be reachable from the compiled core. The
unrolled reference remains in tests/reporting only.

The inherited one-step indices `[1,4,6,11]` are a step-0 regression only. A
pre-execution diagnostic showed they produce a negative student weight at step
1. This is a preparation failure, not evidence against structural support. The
repair is offline and frozen before candidate results: at the center and each
of five sequential teacher clouds, enumerate the 4-of-12 index sets, retain
only full-rank strictly positive charts, maximize the minimum student weight,
and break exact ties lexicographically. Row scales are the componentwise maximum
of absolute teacher features and target magnitude. Preserve the preparation
JSON and use its fixed prefixes unchanged for `T=1,2,5`; runtime selection and
post-result chart changes remain forbidden.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | does functional-loop recursion preserve exact structural support and its total tangent step by step? |
| Baseline | existing one-step fixture and an independently written unrolled finite recursion for `T=1,2,5` |
| Primary criterion | loop/unrolled value, increments, final state/weights, residuals, features, validity, and total score agree at roundoff; support residual/tangent bounds pass |
| Vetoes | artificial full-state noise, off-support finite propagation, Python dynamic loop, partial tangent, nonpositive chart, nonfinite happy path, or compiled invalid control returning finite state |
| Explanatory | graph nodes/bytes, compile/warm time, exact residual magnitudes |
| Not concluded | general structural filtering, NAWM/SIR validity, scientific accuracy, canonical/default/HMC/leaderboard readiness |
| Artifacts | fresh local JSON/log and trusted compiled JSON/log plus close record/hashes |

Every JSON contains a run manifest with git commit, dirty-worktree status,
exact command, conda/environment identity, CPU/GPU mode and trust basis,
TensorFlow/XLA/dtype settings, seeds or `N/A`, wall time, input/preparation
hashes, output paths, phase plan, and phase result path. The close record must
contain the engineering/numerical/scientific decision table, an inference-status
table, attempts/repairs, budget consumption, nonclaims, and a post-run red team.

## Artifacts And Commands

Local root:
`docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-02/structural/attempt-01-local-20260715/`.

Trusted compiled root:
`docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-02/structural/attempt-01-gpu-20260715/`.

Create a dedicated harness
`docs/benchmarks/run_contract_e_tp_clean_xla_phase2_structural.py` supporting
`--device cpu|gpu`, `--invalid-control`, `--output`, and fresh-path refusal.

1. CPU-hidden focused tests and local `T=1,2,5` parity/tangent artifact.
2. Phase 1 source guard over the new core and GraphDef functional-loop count.
3. Trusted/escalated `nvidia-smi` and TensorFlow device probe.
4. Smallest trusted GPU/XLA `T=2` happy path, then the same concrete compiled
   factory/configuration with only the declared invalid input perturbed.
5. Existing structural suite, Phase 1 guardrail suite, compileall, JSON/hash
   parsing, and `git diff --check`.

GPU commands are run only with trusted/escalated permission. CPU commands set
`CUDA_VISIBLE_DEVICES=-1` before TensorFlow import. No full-horizon model attempt
is consumed because this is a small shared fixture, not an eligible model row.

## Defaults And Assumptions

| Choice | Status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| existing four-parent/three-innovation fixture | exact inherited fixture | tests a convenient geometry only | retain engineering-fixture nonclaim |
| horizons `1,2,5` | edge/intermediate loop coverage | no stress-scale claim | GraphDef plus per-step parity |
| fixed charts from existing active indices | inherited exact support selection | chart switches hide derivative | runtime selection forbidden |
| per-step maximum-minimum-weight chart repair | predeclared after step-1 inherited-chart failure | center overfit or unstable chart | preserve all candidate margins; fixed prefix parity only |
| float64 | numerical reference | no TF32 claim | explicit artifact dtype |
| roundoff envelopes above | fixture-specific implementation guards | too loose to detect leakage | operation ledger plus `1e-3` separation ratio |

## Repair And Stop Conditions

Repair localized loop shape, TensorArray/history, fixture harness, or XLA
poisoning defects and retry in a fresh directory within budget. A failed
off-support control or mismatch with the unrolled total tangent is a shared
continuation veto: do not begin scalar-SV GPU work until repaired.

Stop for an unreconciled support/tangent mismatch, invalid chart propagating
finite state under XLA, inability to express the fixed shape with functional
control flow, corrupted evidence, or phase budget exhaustion. Do not stop merely
because a first compile/harness attempt fails locally.

Attempt record before serious execution: the inherited repeated one-step chart
passed step 0 (`minimum_weight=0.1166384`) and failed step 1 with a negative
weight (`-0.1599078`). No parity, score, or GPU claim was produced from that
invalid chart. The offline fixed-sequence preparation above is the localized
repair and does not alter the transition, residual, features, target scalar, or
runtime selection policy.

## Budget

The governing phase cap is 8 CPU core-hours and 1 trusted GPU-hour, with two
trusted fixture attempts. Planned minimum entry spend is 2 CPU core-hours and
0.20 GPU-hours; the separately held repair reserve is 2 CPU core-hours and 0.20
GPU-hours. Their componentwise sum is `4 CPU,0.40 GPU`, which is below both the
phase cap and available campaign budget `95.97 CPU,32 GPU`.
Entry budget gate: `PASS`.

## Exact Phase 3 Handoff

Before close, create
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase3-scalar-sv-loop-core-subplan-2026-07-15.md`
from the actual structural and guardrail results. It must keep actual SV and
KSC-SV separate, bind their target/time-order identities, audit target-specific
feature/order/chart choices, start at `T=1,2,10`, and exclude generalized SV.

The Phase 2 result and ledger must contain `NEXT_PHASE_READINESS`. Automatic
continuation is allowed only if the structural shared continuation veto is clear
and the Phase 3 subplan passes bounded review.

## Skeptical Pre-Execution Audit

Status: `PASS_DRAFT_FOR_REVIEW`.

- The exact structural identity, not a full-state Gaussian proxy, is the target.
- Support and tangent residuals are primary; graph and timing are explanatory.
- The numerical residual/tangent guards use predeclared fixture-specific
  operation-count envelopes and are labeled implementation guards, not fitted
  scientific thresholds or general error theorems.
- Projection selects teacher support, so induction is testable step by step.
- The off-support control distinguishes a vacuous always-valid predicate.
- The fixture cannot authorize NAWM, SIR, or nonlinear scientific claims.
