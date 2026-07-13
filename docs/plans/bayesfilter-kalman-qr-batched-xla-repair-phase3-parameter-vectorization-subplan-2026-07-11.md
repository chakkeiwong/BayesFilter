# Phase 3 Subplan: Analytical Parameter-Axis Vectorization

Date: 2026-07-11
Status: `PHASE_CLOSED_PHASE4_REVIEW_PENDING`

## Phase Objective

Remove the only three static Python `range(P)` loops in the true-batched
analytical score path by treating parameter dimension `P` as a TensorFlow batch
axis in the first-order QR, Cholesky, and factor-to-covariance derivative
helpers. Preserve the existing first-derivative formulas, tensor order
`[B,P,...]`, dtype, QR sign convention, jitter semantics, and public score
target.

This phase may validate the existing batch-native analytical score and run one
tiny GPU-hidden CPU XLA smoke after non-JIT correctness passes. It does not
change scalar/Hessian paths, repair batched autodiff, change timing boundaries,
run GPU, compare performance, or launch a grid.

## Entry Conditions Inherited From Phase 2

- Phase 2 strict diagnostic schema
  `bayesfilter.kalman_qr_batched_xla_repair.phase2.v1` has `state=passed`.
- All eight 16-tensor parity rows, both nested fixture records, and the exact
  `B=1/4/16` GraphDef gate passed.
- Phase 2 final focused suite passed `76` tests; compile and scoped diff checks
  passed.
- All 15 historical hashes and all three read-only QR/Kalman hashes matched.
- No Kalman benchmark worker was active at Phase 2 close.
- Current Phase 3 algorithmic baseline hash is
  `bayesfilter/linear/kalman_qr_derivatives_tf.py = 9434c3e0...`.
- Git HEAD movement from unrelated authorized work is ignored. This lane stops
  only if another lane changes a declared Phase 3 write/read-only path.
- Claude Opus remains platform-policy-blocked before probe; fresh bounded Codex
  substitute review is required and labeled weaker than Claude.

## Current Defect And Exact Scope

The batch-native score path
`tf_qr_sqrt_kalman_score_batched_static` has a TensorFlow `while_loop` over
time and tensor algebra elsewhere. Its only Python parameter-axis loops are:

1. `_batched_stack_qr_lower_factor_first_derivatives`, current lines 214-219;
2. `_batched_cholesky_factor_first_derivatives`, current lines 269-282;
3. `_batched_factor_covariance_first_derivatives`, current lines 299-306.

Other `range(parameter_dim)` loops in the module belong to scalar score,
masked score, or Hessian routes and are read-only in Phase 3. They must not be
used to fail the batch-native first-order structural gate or be refactored as
scope creep.

## Locked Formula And Shape Ledger

Notation uses arbitrary matrix dimensions where valid, not only square fixture
examples.

### Batched QR first derivative

- `stack`: `[B,N,K]` with binding precondition `K>=N`.
- `dstack`: `[B,P,N,K]`.
- `matrix = matrix_transpose(stack)`: `[B,K,N]`.
- `dmatrix = matrix_transpose(dstack)`: `[B,P,K,N]`.
- `q`: `[B,K,N]`; `r`: `[B,N,N]` from the unchanged positive-diagonal QR.
- Broadcast `q[:,None,:,:]` and `r[:,None,:,:]` across `P`.
- `dmatrix_r_inv = dmatrix @ inv(r[:,None,:,:])`: `[B,P,K,N]`, implemented by
  the existing triangular right solve.
- `a = q[:,None,:,:]^T @ dmatrix_r_inv`: `[B,P,N,N]`.
- `omega = tril_strict(a) - tril_strict(a)^T`: `[B,P,N,N]`.
- `dr = (a - omega) @ r[:,None,:,:]`: `[B,P,N,N]`.
- `factor = r^T`: `[B,N,N]`; `dfactor = matrix_transpose(dr)`:
  `[B,P,N,N]`.

The helper must fail closed for `K<N` before applying the square-R triangular
solve. Tests require a clear contract error for a statically known invalid
shape and a TensorFlow assertion for a dynamically supplied invalid shape; no
alternate wide-matrix derivative is introduced in this phase.

### Batched Cholesky first derivative

- `covariance`: `[B,N,N]`; `dcovariance`: `[B,P,N,N]`.
- `factor=L`: `[B,N,N]` from unchanged symmetrization, jitter, and Cholesky.
- `left = solve(L[:,None,:,:], sym(dcovariance))`: `[B,P,N,N]`.
- `b = left @ inv(L[:,None,:,:]^T)`: `[B,P,N,N]`.
- `g = tril(b) - 0.5*diag(diag(b))`: `[B,P,N,N]`.
- `dfactor = L[:,None,:,:] @ g`: `[B,P,N,N]`.

### Batched factor-to-covariance first derivative

- `factor`: `[B,N,N]`; `dfactor`: `[B,P,N,N]`.
- `covariance = factor @ factor^T`: `[B,N,N]`.
- `dcovariance = sym(dfactor @ factor[:,None,:,:]^T +
  factor[:,None,:,:] @ dfactor^T)`: `[B,P,N,N]`.

No transpose may reorder `B` or `P`; every transpose above is
`tf.linalg.matrix_transpose`, affecting only the final two axes. No `tf.map_fn`,
`tf.vectorized_map`, `tf.numpy_function`, explicit inverse, NumPy algorithmic
path, or new approximation is allowed.

## Required Artifacts And Write Set

Allowed Phase 3 writes:

- `bayesfilter/linear/kalman_qr_derivatives_tf.py`, limited to the three
  batched first-derivative helpers and directly required private broadcast
  support;
- new focused tests in
  `tests/test_linear_qr_batched_parameter_vectorization_tf.py`;
- every Phase 3 non-JIT correctness, dtype, shape, jitter, reverse-mode, and
  source-contract check must live in
  `tests/test_linear_qr_batched_parameter_vectorization_tf.py`;
- Phase 3 diagnostic mode/helpers only in
  `scripts/benchmark_kalman_qr_parameter_count_scaling.py` and version/schema
  constants only in `scripts/kalman_qr_benchmark_contract.py` if required;
- strict diagnostic JSON:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase3_parameter_graphdef_2026-07-11.json`;
- verbose logs only under `/tmp/kalman_qr_phase3_vectorization/`;
- Phase 3 result:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase3-parameter-vectorization-result-2026-07-11.md`;
- refreshed Phase 4 subplan and at most five deterministic Phase 4 review
  records if reached.

Read-only in Phase 3:

- `bayesfilter/linear/qr_factor_tf.py` scalar first-order helpers, which are the
  independent per-`(B,P)` formula references;
- `bayesfilter/linear/kalman_qr_tf.py` value/autodiff backend;
- all scalar, masked, and Hessian implementations in
  `bayesfilter/linear/kalman_qr_derivatives_tf.py`;
- `tests/test_linear_kalman_qr_derivatives_tf.py` and
  `tests/test_linear_qr_factor_tf.py`;
- `tests/test_linear_qr_batched_analytical_score_tf.py`, except that its exact
  existing CPU-XLA test named in section 6 may run after all non-XLA gates;
- Phase 0-2 artifacts and all historical `*2026-07-09*` paths.

## Required Checks, Tests, And Review

### 1. Helper formula parity

For both `float32` and `float64`, deterministic nondegenerate cases must compare
each vectorized helper against unchanged scalar references applied independently
for every `B` row and `P` direction:

- `B=1,3`, `P=1,4`;
- QR stack uses `N=3,K=5`, full column rank, positive-diagonal convention;
- QR contract tests reject `N=4,K=3` in both direct/static and dynamic-signature
  calls with the declared `K>=N` error;
- Cholesky covariance uses deterministic SPD `N=3` tensors and symmetric
  derivative directions;
- factor covariance uses nonsingular lower `N=3` factors and arbitrary first
  factor derivatives;
- exact output shapes/dtypes;
- factor/covariance base outputs use exact equality where operation order is
  unchanged, otherwise the same declared helper tolerances apply;
- locked helper tolerances: float32 `rtol=atol=2e-5`, float64
  `rtol=atol=2e-12`.

These tolerances are prospective conservative roundoff envelopes, not promotion
thresholds inherited from the historical benchmark. They may not be loosened
after observing a failure without stopping and refreshing/reviewing this plan.

### 2. Dynamic-P trace reuse

Wrap each helper in `tf.function(jit_compile=False)` with a `None` parameter
dimension in its input signature. Call the same wrapper at `P=1` and `P=4` and
require one concrete function plus exact output shapes. This gate proves the
helper no longer requires `_static_dim(..., axis=1)` and does not retrace by P.

### 3. Source structure

AST/source inspect exactly the three helpers and directly called private
first-order helpers. Require:

- no `for`, `while`, list/set/dict/generator comprehension;
- no `_static_dim` for parameter dimension;
- no `tf.map_fn`, `tf.vectorized_map`, or `tf.numpy_function`;
- no call to scalar score/Hessian routes.

Do not apply this gate to separate scalar or Hessian functions.

### 4. End-to-end analytical parity

Run focused GPU-hidden checks whose callables are explicitly
`jit_compile=False` before any XLA call:

- batch-native analytical versus scalar analytical rows at `B=1,3`, distinct
  `B/P/state` axes, float64 using existing `value atol=1e-10`, score
  `rtol=1e-8,atol=1e-9`;
- batch-native analytical versus existing reverse-mode reference at float32
  using the existing `rtol=atol=2e-4` diagnostic tolerance;
- Phase 2 fixture end-to-end smoke at `dimension=3,T=4,P=3,B=1/4`, both
  dtypes, requiring finite `[B]` value and `[B,P]` score and scalar analytical
  row parity under the same dtype-specific existing test tolerances;
- existing dtype, shape-rejection, jitter, and no-scalar-wrapper tests.

The float32 reverse-mode comparison is a correctness diagnostic, not a Phase 4
promotion of the currently broken benchmark probe.

The read-only historical tests in
`tests/test_linear_qr_batched_analytical_score_tf.py` call a public function
decorated `jit_compile=True`, including tests whose names do not mention XLA.
They must not be included in the pre-GraphDef pytest command. The new focused
test file must reproduce the required scalar, reverse-mode, dtype, shape, and
distinct-axis checks through `.python_function` or wrappers explicitly marked
`jit_compile=False`; it must also carry every Phase 3 source-contract check.
No node from the historical file may run before the GraphDef gate. After all
non-XLA gates pass, only the exact existing CPU-XLA node named in section 6 may
run.

The read-only scalar/Hessian suite in
`tests/test_linear_kalman_qr_derivatives_tf.py` is also excluded from the
pre-GraphDef command. It contains explicit CPU-XLA nodes and expensive
scalar/Hessian graph-reuse checks outside the three batched first-order helpers.
An attempted combined run completed 63 of 94 collected nodes and then hit its
prospective 180-second safety cap; it is recorded as incomplete, not passed.
Phase 3 instead rehashes that read-only file and relies on the focused batched
tests plus unchanged scalar helper references in `tests/test_linear_qr_factor_tf.py`.
The full scalar/Hessian suite is not waived globally; it is simply not evidence
for this pre-GraphDef batched-vectorization gate.

### 5. End-to-end GraphDef gate

Trace only the batch-native analytical score wrapper with
`tf.function(jit_compile=False)` at `dimension=10,T=8,B=4,float32` for nested
`P=50/150` fixtures. Do not execute the score callable.

Record for each P: node count, serialized bytes, ordered op sequence digest,
op histogram, constant count, input/output static shapes, trace wall time,
fixture/source/version hashes, and environment identity.

The strict JSON must also record schema/state, exact command argv, git commit,
declared-path status and SHA-256 hashes, interpreter/conda/runtime versions,
`CUDA_VISIBLE_DEVICES`, CPU thread settings, `jit_compile=False`, XLA not run,
the Phase 2 fixture-version triple, plan/result/log paths, every gate boolean,
and all Phase 3 nonclaims. It must fail closed with `state=failed` and nonzero
exit if any structural gate is false.

Implement the structural decision as a pure gate evaluator that receives two
already-traced metadata rows and returns named check booleans plus
`state=passed|failed`. Tests must start from a passing P=50/150 metadata pair
and independently mutate node count, one ordered op entry/digest, one
op-histogram count, and constant count. Every mutation must make its named
check false, overall `state=failed`, and the diagnostic return code nonzero. A
same-row/swap test must also prove that the evaluator requires distinct rows
labeled `parameter_count=50` and `150` instead of comparing one graph twice.

Boundary tests must monkeypatch score outputs to objects that raise on
materialization/execution and monkeypatch `_select_device` plus TensorFlow
device-enumeration helpers to raise if called. The diagnostic may obtain and
serialize concrete GraphDefs only; it must neither execute/materialize the
score callable nor invoke device enumeration.

Hard structural gates:

- node count exactly equal at `P=50/150`;
- ordered op sequence digest exactly equal;
- op histogram and constant count exactly equal;
- expected output shapes `[4]` and `[4,P]`;
- source gate above passes.

Shape metadata and derivative constant payloads necessarily differ with P and
are not normalized into a false whole-GraphDef equality claim. Raw GraphDef
bytes and trace duration are explanatory only. The historical 5,912-node
`B=4,P=50,T=120` and 16,568-node `B=16,P=150,T=120` observations are historical
context only, not like-for-like promotion baselines.

### 6. Tiny CPU XLA compatibility smoke

Only after sections 1-5 pass, run the existing single test
`test_batched_qr_score_cpu_xla_preserves_dtype_and_signature` with GPU hidden,
one CPU thread, and a 120-second command timeout. This is a small compatibility
smoke under the repository XLA-default policy. It is not the Phase 6 CPU XLA
scaling gate and cannot establish compile scalability or runtime readiness.

### 7. Exact local commands

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  bayesfilter/linear/kalman_qr_derivatives_tf.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  tests/test_linear_qr_batched_parameter_vectorization_tf.py

CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_linear_qr_batched_parameter_vectorization_tf.py \
  tests/test_linear_qr_factor_tf.py \
  tests/test_kalman_qr_batched_fixture.py

git diff --check -- \
  bayesfilter/linear/kalman_qr_derivatives_tf.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  scripts/kalman_qr_benchmark_contract.py \
  tests/test_linear_qr_batched_parameter_vectorization_tf.py
```

After the tests pass, generate the non-JIT graph artifact through a dedicated
fail-closed mode that branches before device enumeration and score execution:

```bash
mkdir -p /tmp/kalman_qr_phase3_vectorization
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  --phase3-parameter-graph-diagnostic --device cpu --cpu-threads 1 \
  --no-jit-compile \
  --output-json \
  docs/benchmarks/kalman_qr_batched_xla_repair_phase3_parameter_graphdef_2026-07-11.json \
  --phase3-log-path \
  /tmp/kalman_qr_phase3_vectorization/phase3_parameter_graphdef.log \
  > /tmp/kalman_qr_phase3_vectorization/phase3_parameter_graphdef.log 2>&1
```

Strict-read and summarize that JSON before the XLA smoke. Then run the one XLA
test with a bounded log:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 timeout 120 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_linear_qr_batched_analytical_score_tf.py::test_batched_qr_score_cpu_xla_preserves_dtype_and_signature \
  > /tmp/kalman_qr_phase3_vectorization/cpu_xla_smoke.log 2>&1
```

Exit `124` is a timeout repair trigger, not a pass. Preserve the full logs and
show only bounded tails on failure.

### 8. Close and review

Strict-read the Phase 3 JSON, rehash all read-only paths and historical paths,
check no worker remains, write the Phase 3 result, refresh Phase 4 from actual
evidence, and obtain bounded read-only review of the exact Phase 4 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the exact current first-derivative formulas evaluate P as TensorFlow batch algebra without static P graph duplication? |
| Exact baseline | The unchanged scalar first-order helpers for formula parity, current batch-native analytical score for end-to-end parity, and Phase 2 nested fixtures. |
| Primary criterion | Helper scalar parity, dynamic-P one-trace reuse, exact source gate, end-to-end parity, exact P=50/150 node/op structure gate, strict artifact validity, and tiny CPU XLA compatibility smoke all pass. |
| Promotion vetoes | Formula/shape/dtype mismatch; QR sign drift; non-finite output; scalar/autodiff parity failure; residual batched P loop; P-dependent node/op/constant-count growth; stale/corrupt artifact. |
| Repair triggers | Any in-scope helper, source, parity, trace, graph, or XLA-smoke failure that can be localized without changing the target. |
| Explanatory only | Raw GraphDef bytes, trace/XLA smoke duration, historical graph sizes, and observed residual maxima below locked tolerances. |
| Not concluded | Warm-runtime improvement, analytical/autodiff ranking, CPU/GPU scalability, HMC/posterior/default/production/scientific validity. |

## Skeptical Pre-Execution Audit

- The baseline is unchanged scalar first-order math and end-to-end score output,
  not historical compile time or GraphDef bytes.
- Equal node/op structure plus dynamic-P trace reuse directly tests removal of
  static P unrolling; smaller bytes alone cannot pass.
- Scalar and Hessian loops are outside the batch-native first-order question and
  cannot be silently expanded into this write set.
- Locked helper cases use distinct B/P/matrix axes to catch accidental axis
  exchange and broadcasting that only works when dimensions coincide.
- QR sign convention and Cholesky jitter remain unchanged.
- Non-JIT diagnostics are explicit localization/reference exceptions. The tiny
  XLA smoke is not promoted into Phase 6 evidence.
- No timing comparison is performed, so no replication/ranking rule is needed.
- A helper bug or one XLA-smoke failure is a repair trigger. It invalidates the
  implementation candidate or CPU-XLA compatibility arm, not the underlying
  Kalman target unless parity cannot be restored without changing formulas.

Audit status: `PASSED_AFTER_CODEX_SUBSTITUTE_REVIEW_ROUND5`.

## Forbidden Claims And Actions

- Do not edit scalar, masked, Hessian, value, QR-factor reference, or public API
  semantics.
- Do not vectorize second-order parameter grids.
- Do not use an explicit matrix inverse or an approximate derivative.
- Do not repair the batched-autodiff tape/reduction bug; that is Phase 4.
- Do not change timing/materialization or supervisor semantics.
- Do not run GPU, full CPU XLA scaling, score timing, or any comparison grid.
- Do not call graph-size reduction speed evidence or call the CPU smoke GPU
  readiness.
- Do not weaken tolerances or structural gates after seeing results.
- Do not modify/revert unrelated dirty work or historical artifacts.

## Exact Next-Phase Handoff Conditions

All conditions are conjunctive:

- All helper formula, dynamic-P, source, end-to-end, dtype/shape, and existing
  focused tests pass.
- Strict Phase 3 JSON has `state=passed` and every structural gate true.
- P=50/150 node count, ordered op digest, histogram, and constant count match.
- The bounded CPU XLA smoke passes within 120 seconds after non-XLA gates.
- Compile, focused pytest, scoped diff, strict JSON, read-only/historical hashes,
  and no-worker checks pass.
- Phase 3 result maps each loop to formula/test/graph evidence and preserves all
  nonclaims.
- Refreshed Phase 4 subplan receives exact `VERDICT: AGREE` from the available
  bounded reviewer, explicitly weaker than Claude if substitute review is used.

## Stop Conditions

- Scalar first-order parity cannot be restored within locked tolerances without
  changing formula, QR sign, jitter, dtype, or target semantics.
- Static P graph growth remains after all three in-scope loops are removed and
  the cause lies outside the reviewed write set.
- Another lane changes a declared Phase 3 path and overlap cannot be reconciled.
- A required artifact/provenance contract cannot be satisfied.
- New package/network/model-file/default/product/scientific authority is needed.
- The same material review blocker does not converge after five rounds.

A normal broadcasting mistake, focused test regression, GraphDef mismatch, or
tiny CPU XLA failure is a repair trigger first. Localize and patch within the
write set; stop only if it becomes one of the continuation vetoes above.

## Mandatory Phase-End Sequence

1. Run every required local check.
2. Write the Phase 3 result/close record.
3. Refresh the Phase 4 subplan from actual evidence.
4. Review Phase 4 for consistency, correctness, feasibility, artifact coverage,
   and boundary safety; visibly repair and recheck before advancing.
