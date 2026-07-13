# Phase 4 Subplan: Correct True-Batched Autodiff Comparator

Date: 2026-07-11
Status: `REVIEW_CONVERGED_FOR_CPU_XLA_TENSORLIST_BOUND_REPAIR_EXECUTION_ACTIVE`

## Phase Objective

Replace the known-broken diagnostic probe with a fail-closed true-batched
reverse-mode score method. The repaired method must differentiate the vector
batch likelihood without a reduction created outside `GradientTape`, prove
batch-row independence through the full likelihood Jacobian, match scalar
autodiff and the Phase 3 analytical score, and become the primary autodiff
method identity in the method-isolated harness.

This phase repairs comparator correctness and identity only. It does not change
the Kalman likelihood backend, separate timing/materialization, run a CPU/GPU
scaling ladder, tune XLA, or make performance or scientific claims.

## Entry Conditions Inherited From Phase 3

- Phase 3 strict schema
  `bayesfilter.kalman_qr_batched_xla_repair.phase3.v1` has `state=passed` and
  every one of its seven structural checks is true.
- P=50/150 analytical graphs both have 884 nodes, the same ordered-op digest,
  exact op histogram, and 573 constants.
- Final Phase 3 non-XLA suite passed 77 tests; the exact bounded analytical
  CPU-XLA smoke passed one test.
- Analytical source hash is `d24ae4363...`; harness hash is `17f03ab7...`;
  value backend hash is the unchanged `cc99674d...`.
- All Phase 3 read-only and 14 unique Phase 0 historical hashes matched; no
  Python/pytest benchmark worker remained.
- Git HEAD movement from unrelated authorized work is ignored. This lane stops
  only if another lane changes a declared Phase 4 write/read-only path.
- Claude Opus remains platform-policy-blocked before probe. Fresh bounded Codex
  substitute review is required and explicitly weaker than Claude.

## Current Defect And Root-Cause Contract

`build_batched_autodiff_probe_fn` currently executes:

1. watch `params: [B,P]` inside `GradientTape`;
2. compute `value: [B]` inside the tape;
3. leave the tape;
4. create `tf.reduce_sum(value)` outside the tape;
5. request its gradient and replace `None` with NaNs.

The outside-tape reduction is not recorded. The observed `None`/NaN result is a
harness defect, not evidence against QR likelihood differentiability.

The repaired production builder must use the already-recorded vector target:

```text
score = tape.gradient(
    value,
    params,
    output_gradients=tf.ones_like(value),
)
```

If `score is None`, construction must fail closed with a clear error. No NaN,
zero, stopped-gradient, finite-difference, or scalar-row fallback is allowed.

## Locked Gradient And Row-Independence Ledger

For `params` of shape `[B,P]` and likelihood vector `v` of shape `[B]`, define:

```text
J[i,j,p] = d v[i] / d params[j,p]    shape [B,B,P]
```

The intended score is `J[b,b,:]`, shape `[B,P]`. The VJP with all-one output
cotangents is `sum_i J[i,j,p]`. It equals the intended per-row score only when
all off-diagonal blocks `J[i,j,:]`, `i != j`, are zero. Therefore finite VJP
output alone cannot pass Phase 4.

Focused tests and the strict diagnostic must require:

- full Jacobian shape `[B,B,P]` at `B=4,P=3`;
- every diagonal block matches the returned score;
- every off-diagonal block is zero within prospective dtype tolerance;
- perturbing one parameter row changes only the corresponding likelihood and
  score row within the same tolerance;
- batch-native values/scores match independently computed scalar-row autodiff;
- values/scores also match the Phase 3 analytical score.

The test-only Jacobian may use a persistent tape on the tiny diagnostic case.
The production/timed builder must use one ordinary reverse-mode VJP and must not
form the full Jacobian.

## Method Identity And Artifact Migration

The closed harness method contract must become:

```text
PRIMARY_METHOD_IDS = (
    "batch_native_analytical_qr_score",
    "batch_native_autodiff_qr_score",
)
REFERENCE_METHOD_IDS = (
    "scalar_analytical_row_loop",
    "autodiff_row_loop_qr_score",
)
METHOD_IDS = PRIMARY_METHOD_IDS + REFERENCE_METHOD_IDS
```

Requirements:

- Rename the builder to `build_batch_native_autodiff_fn` and remove the broken
  `batched_static_autodiff_probe` method identity.
- The supervisor default must be exactly `PRIMARY_METHOD_IDS`, not a positional
  slice of `METHOD_IDS`.
- Scalar methods remain explicit small-case references and are never the
  default large-batch comparator.
- Bump the strict method-record schema from v2 to v3 and add an explicit
  `METHOD_CONTRACT_VERSION = "batch-native-autodiff-phase4-v1"` as a required
  `CONFIG_FIELDS` field and schedule-root field. It must therefore participate
  in both config and schedule fingerprints so older method artifacts cannot
  resume.
- Tests must show a synthetically valid v2/old-method record is rejected with a
  named schema/method/config reason; each stale-field test changes exactly one
  field while all other identity fields remain current, and hashes the record
  before and after the rejected reuse attempt to prove no overwrite.
- Method selection must still construct exactly one selected builder and record
  exactly one invoked method ID.
- Aggregate parity must compare the primary analytical/autodiff pair using the
  prospective dtype tolerances below, not exact JSON equality. Explicitly
  requested reference methods may join the same correctness comparison under
  the exact schedule semantics below.

### Schedule and aggregate semantics

Every schedule records one of two modes:

- `primary_pair`: both IDs in `PRIMARY_METHOD_IDS` are present. Pair completeness
  and comparator parity are mandatory aggregate checks. Any requested reference
  method is also compared to its corresponding primary method.
- `method_local_only`: one or both primary IDs are absent. The artifact must
  record `primary_pair_complete=false`, `comparator_parity_applicable=false`,
  `comparator_parity=null`, and a named reason. Pair checks are not included in
  that schedule's mandatory checks, so a valid method may retain method-local
  viability, but neither the status nor any result may call it comparison-
  complete or satisfy the Phase 4 handoff.

The default supervisor schedule is always `primary_pair`. Thus a missing
comparator can never silently pass a comparison, while later method-isolated
CPU/GPU localization does not erase a valid sibling method record.

For a comparison-eligible case, analytical output is the numerical reference.
For every scalar element `reference` and `candidate`, require finite values and:

```text
abs(candidate - reference) <= atol + rtol * abs(reference)
```

Use the requested case dtype only after both output metadata records exactly
match that dtype and expected shapes. Use separate value and score tolerances
from section 2. Reference-method mappings and directions are exact:

| Candidate method | Numerical reference method |
| --- | --- |
| `batch_native_autodiff_qr_score` | `batch_native_analytical_qr_score` |
| `scalar_analytical_row_loop` | `batch_native_analytical_qr_score` |
| `autodiff_row_loop_qr_score` | `batch_native_autodiff_qr_score` |

The left column is always `candidate` and the right column is always
`reference` in the asymmetric formula above. Tests for every mapping must cover
exact equality, just-inside and just-outside each value/score tolerance,
sign/asymmetric-magnitude cases that would differ if direction were reversed,
and NaN/Infinity rejection.

The legacy all-method v1 grid remains historical/non-promoting. Phase 4 may
update stale diagnostic labels needed to remove the broken probe, but it must
not revive that coupled grid as the execution path.

## Required Artifacts And Write Set

Allowed Phase 4 writes:

- `bayesfilter/linear/kalman_qr_tf.py`, limited to supplying the existing
  dynamic `n_timesteps` tensor as `maximum_iterations` on the batch-static
  likelihood `tf.while_loop`; no recursion math, public signature, static-loop
  route, scalar route, dtype, jitter, or numerical semantics may change;
- `scripts/benchmark_kalman_qr_parameter_count_scaling.py`, limited to the
  batch-native autodiff builder, Phase 4 diagnostic, method dispatch/labels,
  and directly required parity support;
- `scripts/kalman_qr_benchmark_contract.py`, limited to the v3 method identity,
  method-contract version, and stale-record validation;
- `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`, limited to
  primary defaults, schedule identity, and tolerance-aware aggregate parity;
- new focused tests in
  `tests/test_kalman_qr_batch_native_autodiff.py`;
- directly required contract/harness regression tests in
  `tests/test_kalman_qr_benchmark_contract.py` and
  `tests/test_kalman_qr_parameter_count_scaling_harness.py`;
- strict diagnostic JSON
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_2026-07-11.json`;
- durable CPU-XLA smoke JSON
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_cpu_xla_smoke_2026-07-11.json`;
- verbose logs only under `/tmp/kalman_qr_phase4_autodiff/`;
- Phase 4 result, refreshed Phase 5 subplan, and at most five deterministic
  Phase 5 review records if reached.

Read-only in Phase 4:

- every part of `bayesfilter/linear/kalman_qr_tf.py` except the single dynamic
  batch-loop bound explicitly admitted above;
- `bayesfilter/linear/kalman_qr_derivatives_tf.py` and all Phase 3 analytical
  formulas;
- `bayesfilter/linear/qr_factor_tf.py`;
- Phase 0-3 result/JSON artifacts and every historical `*2026-07-09*` path.

## Required Checks, Tests, And Review

### 1. Root-cause and fail-closed regression

- A tiny synthetic tape test proves a reduction created outside the tape yields
  `None`, while direct vector VJP or an inside-tape reduction is finite.
- Source/AST inspection proves the batch-native builder has no outside-tape
  `reduce_sum(value)`, no NaN/zero fallback, and no scalar score/value call.
- Monkeypatch the batch likelihood to a disconnected value and require builder
  construction/tracing to fail clearly when the gradient is `None`.

### 2. Value and score parity

Use deterministic `dimension=2,T=4,P=3`, `B=1/4`, both float32 and float64.
Compare:

- batch-native likelihood against scalar row likelihood;
- batch-native VJP score against scalar row autodiff;
- batch-native value/score against Phase 3 analytical value/score;
- exact output shapes `[B]`, `[B,P]`, requested dtype, and finite values.

Prospective tolerances, fixed before execution:

- float32 value/score: `rtol=atol=2e-4`;
- float64 value: `rtol=1e-10, atol=1e-10`;
- float64 score: `rtol=1e-8, atol=1e-9`.

They may not be loosened after observing a failure without stopping and
refreshing/reviewing this subplan.

### 3. Full Jacobian and perturbation gate

At `B=4,P=3,dimension=2,T=4`, both dtypes:

- full likelihood Jacobian is `[4,4,3]` and finite;
- diagonal blocks match returned score under the dtype tolerance;
- maximum absolute off-diagonal block is at most `2e-6` for float32 and
  `2e-12` for float64;
- perturb row 2 by deterministic `[+0.01,-0.015,+0.02]` and require rows 0,1,3
  values/scores unchanged under dtype tolerance;
- perturbed row 2 remains finite; no minimum-change claim is required because
  a near-stationary direction is possible.

The off-diagonal limits are conservative zero/roundoff screens, not evidence
thresholds for ranking, and cannot be loosened retrospectively.

### 4. Method-contract and stale-resume gates

- `PRIMARY_METHOD_IDS` and `REFERENCE_METHOD_IDS` are exact and disjoint.
- Default supervisor schedule contains only the two primary methods.
- Explicit reference selection remains possible.
- Selected-builder tests cover every method and prove sibling builders are not
  called.
- v2, old method ID, old method-contract version, stale source, and stale
  schedule records all fail reuse with named reasons. Each mutation changes one
  field only, leaves every other identity field current, and preserves an exact
  before/after artifact hash.
- Strict JSON rejects NaN/Infinity and duplicate keys as before.
- Tolerance-aware aggregate parity passes a within-tolerance primary pair and
  fails independently mutated value, score, shape, dtype, missing-output, and
  non-finite cases. Boundary tests cover just-inside/outside value and score
  tolerances plus analytical-reference direction.
- A default schedule requires the complete primary pair. Reference-only,
  single-primary, and mixed incomplete-primary schedules are explicitly
  `method_local_only`, expose null/not-applicable parity, and cannot satisfy
  the Phase 4 handoff even if their method-local records pass.

### 5. Strict Phase 4 diagnostic

Add a dedicated `--phase4-autodiff-diagnostic` mode that branches before device
enumeration and all timing/grid paths. With GPU hidden and JIT off, record the
four `(dtype,B)` rows above plus both Jacobian rows.

The strict JSON must include:

- schema/state and every named gate boolean;
- exact argv, git commit, declared-path status/hashes, interpreter/conda/runtime;
- `CUDA_VISIBLE_DEVICES`, CPU thread settings, `jit_compile=false`, XLA not run,
  no device enumeration, deterministic fixture/version hashes;
- per-row shapes/dtypes/finite checks and value/score residuals;
- per-dtype Jacobian shape, diagonal residual, off-diagonal maximum, and
  perturbation residuals;
- method schema/version/primary/reference identities;
- plan/result/log paths and Phase 4 nonclaims.

The diagnostic gate evaluator must receive the observed rows/manifest and an
independently constructed expected contract. It must expose named hard gates
for every evidence-bearing field, including:

- row finite/dtype/shape, scalar parity, analytical parity, diagonal-Jacobian,
  off-diagonal-Jacobian, and perturbation checks;
- exact method schema, method-contract version, primary/reference identities,
  fixture/parameter-batch/observation-generation versions, and fixture hashes;
- declared-path set/status/SHA-256, source fingerprint, nonempty git commit,
  exact diagnostic argv/mode/output/log paths, complete runtime identity, and
  plan/result paths;
- requested CPU device, `CUDA_VISIBLE_DEVICES=-1`, requested/effective thread
  settings, `jit_compile=false`, `xla_execution=not_run`, and TF32-not-queried
  status.

`no device enumeration` means the Phase 4 diagnostic branch calls neither the
harness selector nor TensorFlow physical/logical device-enumeration APIs. Tests
must monkeypatch all of those calls to raise. TensorFlow import-time CUDA log
messages are recorded separately and do not count as harness enumeration or GPU
evidence.

It must write `state=failed` and return nonzero if any hard gate is false. Pure
gate tests must start from one passing raw payload and independently mutate each
numerical, method, fixture/version/hash, declared-path hash/status, argv/path,
runtime, git, JIT/XLA/device/CUDA/thread/TF32 field. Every raw-field mutation
must make its named check false, overall `state=failed`, and return code nonzero;
no precomputed boolean-only mutation is sufficient.

### 6. Tiny CPU-XLA compatibility smoke

Only after sections 1-5 pass, run one GPU-hidden, one-thread, 120-second XLA
test at `dimension=2,T=4,P=3,B=4,float32`. Compare the compiled batch-native
autodiff outputs to its non-JIT outputs under float32 tolerance and require one
concrete function. Exit 124 is a repair trigger, not a pass.

The smoke must also write the durable strict JSON declared above. It records
schema/state, exact command argv, source/config/runtime/fixture hashes,
GPU-hidden CPU provenance, JIT/TF32/thread settings, compiled and non-JIT
shapes/dtypes, value/score residuals and tolerances, concrete-function count,
internal wall time, plan/result/log paths, and nonclaims. The phase close record
must preserve both separate outer command manifests defined after the command
blocks below. A passing CLI summary, pytest line, or ephemeral `/tmp` log alone
cannot close the XLA gate.

The XLA smoke has its own pure fail-closed evaluator, separate from the non-JIT
diagnostic evaluator. It receives the raw observed smoke payload and an
independently constructed expected contract. Named hard gates cover exact
schema/method/fixture/source/runtime identity, declared-path hashes/status,
argv/output/log/plan/result paths, GPU-hidden CPU provenance and thread/TF32
settings, `jit_compile=true`, `xla_execution=executed`, compiled and non-JIT
shapes/dtypes/finite status, value/score residual formula and locked tolerances,
one concrete function, and nonempty wall time. Starting from one passing raw
payload, tests must independently mutate every XLA-specific field, including
JIT, execution status, each compiled/non-JIT shape and dtype, finite flags,
value residual, score residual, each tolerance, concrete-function count, and
identity/provenance fields. Every mutation must make its named gate false,
overall `state=failed`, and the CLI return nonzero. Precomputed booleans cannot
substitute for raw-field evaluation.

This smoke is not Phase 6 CPU-XLA scaling evidence and cannot establish GPU
viability or runtime readiness.

#### Observed repair trigger: dynamic-loop TensorList bound

The first reviewed CPU-XLA producer failed, non-timeout, before numerical
comparison with `InvalidArgumentError: XLA compilation requires a fixed tensor
list size` at a reverse-mode accumulator under the batch likelihood's dynamic
`tf.while_loop`. The scalar sibling already supplies
`maximum_iterations=n_timesteps`; the batch sibling does not.

The bounded repair is to add exactly that existing bound to the batch dynamic
loop. This is execution-structure metadata required by XLA reverse mode, not a
change to likelihood recursion or a switch to the static Python time loop.
Focused source/tests must prove:

- the dynamic batch loop retains `parallel_iterations=1` and gains exactly
  `maximum_iterations=n_timesteps`;
- non-JIT float32/float64 value and score evidence remains within the original
  locked tolerances;
- the static-horizon likelihood is not substituted into the primary builder;
- the original failed artifact/log hashes are preserved in the Phase 4 result;
- the exact CPU-XLA CLI and pytest gates are rerun without changing tolerances.

Using the static-horizon route as the XLA comparator is forbidden because it
would unroll `T=120` and could hide rather than repair the Phase 6 graph-scaling
blocker.

### 7. Exact local commands

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_batch_native_autodiff.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py

CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  --deselect=tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value \
  tests/test_kalman_qr_batch_native_autodiff.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_linear_qr_batched_parameter_vectorization_tf.py \
  tests/test_kalman_qr_batched_fixture.py

git diff --check -- \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_batch_native_autodiff.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
```

Diagnostic:

```bash
mkdir -p /tmp/kalman_qr_phase4_autodiff
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  --phase4-autodiff-diagnostic --device cpu --cpu-threads 1 \
  --no-jit-compile \
  --output-json \
  docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_2026-07-11.json \
  --phase4-log-path /tmp/kalman_qr_phase4_autodiff/phase4_autodiff.log \
  > /tmp/kalman_qr_phase4_autodiff/phase4_autodiff.log 2>&1
```

Strict-read the non-JIT JSON before the XLA smoke. Add a dedicated
`--phase4-autodiff-xla-smoke` mode plus the exact regression node
`test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value`.
The CLI mode writes the durable XLA JSON; the focused test validates the same
compiled/non-JIT contract. Run the CLI through the bounded command:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 timeout 120 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  --phase4-autodiff-xla-smoke --device cpu --cpu-threads 1 \
  --jit-compile \
  --output-json \
  docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_cpu_xla_smoke_2026-07-11.json \
  --phase4-log-path /tmp/kalman_qr_phase4_autodiff/cpu_xla_smoke.log \
  > /tmp/kalman_qr_phase4_autodiff/cpu_xla_smoke.log 2>&1
```

Then run the exact focused regression node separately:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 timeout 120 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value \
  > /tmp/kalman_qr_phase4_autodiff/cpu_xla_pytest.log 2>&1
```

Strict-read the durable JSON. Both commands and the strict read must pass.

The Phase 4 result preserves two separate outer command manifests:

1. CLI XLA producer: exact timeout-wrapped command, exit code versus timeout,
   outer wall time, full CLI log path/hash, durable JSON path/hash, and strict-
   read result.
2. Pytest XLA regression: exact timeout-wrapped command, exit code versus
   timeout, outer wall time, full pytest log path/hash, exact node ID, and
   passed/failed/collected test count.

Neither command may inherit the other's success. Missing or failed evidence in
either manifest is a Phase 4 gate failure.

The pre-XLA command deliberately deselects the exact XLA regression node and
must report exactly one deselected item. This prevents execution of the XLA test
before the strict non-JIT diagnostic passes; the later exact-node command
remains mandatory and runs it separately.

### 8. Close and review

Strict-read the Phase 4 JSON, rehash every read-only/historical path, verify no
worker remains, write the Phase 4 result, refresh Phase 5 from actual evidence,
and obtain bounded read-only review of the exact Phase 5 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the true-batched QR likelihood produce the intended finite, row-independent `[B,P]` reverse-mode score and become the fair primary autodiff comparator? |
| Exact baseline | Scalar row autodiff for score correctness, Phase 3 analytical score, and the unchanged batch-static value backend. |
| Primary criterion | Root-cause regression, finite/dtype/shape parity, full Jacobian diagonal/off-diagonal gate, perturbation isolation, method-contract migration, strict artifact, and tiny CPU-XLA smoke all pass. |
| Promotion vetoes | `None` or fallback gradient; non-finite value/score/Jacobian; cross-row coupling; scalar/analytical parity failure; old probe/default scalar comparator remains; stale record resumes; corrupt artifact. |
| Repair triggers | Any localized tape-target, method-identity, parity, Jacobian, artifact, or tiny CPU-XLA failure within the write set. |
| Explanatory only | Diagnostic trace/XLA duration and observed residuals below prospective limits. |
| Not concluded | Warm-runtime improvement, CPU/GPU scalability, method ranking, HMC/posterior/default/production/scientific validity. |

## Skeptical Pre-Execution Audit

- The baseline is independently computed scalar autodiff plus the Phase 3
  analytical score, not merely disappearance of NaNs.
- A finite VJP is insufficient: the full Jacobian and row perturbation gates
  directly test the row-independence assumption required to interpret it as
  `[B,P]` per-row scores.
- The method schema/version/default migration prevents the repaired probe from
  remaining diagnostic-only or old scalar methods from silently staying primary.
- Non-JIT correctness precedes the one bounded XLA compatibility smoke. Neither
  trace nor compile survival is promoted into runtime evidence.
- Tolerances and off-diagonal screens are prospective and dtype-specific.
- No timing is interpreted, so stochastic ranking and replication rules do not
  apply in this phase.
- A current comparator failure triggers repair; it does not invalidate the
  Kalman target unless value/scalar references or row independence cannot be
  restored without changing the target.

Audit status: `CPU_XLA_TENSORLIST_BOUND_REPAIR_CODEX_ROUND1_AGREE`. The
earlier execution-order repair passed bounded Codex round 2. A later CPU-XLA
run then triggered the fixed-TensorList-size error above. The proposed repair
uses the same dynamic horizon bound already present in the scalar sibling and
explicitly rejects the misleading static-unroll workaround. A fresh bounded
bounded Codex substitute reviewer agreed that the bound is semantics-preserving
and appropriately empirical: it may or may not satisfy XLA, and failure remains
a repair trigger. Claude remained platform-policy-blocked before probe, so this
review is explicitly weaker.
audit found that the original pre-XLA command would collect the XLA regression
from the same test file. The repaired command now uses the exact node ID to
deselect only that test's execution and preserves the required non-JIT-before-
XLA gate order. Claude remained platform-policy-blocked before its liveness
probe, so no content was sent; bounded Codex substitute review is explicitly
weaker and returned `VERDICT: AGREE` in repair round 2.

## Forbidden Claims And Actions

- Do not edit Kalman value recursion, Phase 3 analytical formulas, scalar
  reference math, public API semantics, or any value-backend line except the
  admitted dynamic batch-loop `maximum_iterations` bound.
- Do not replace reverse mode with finite differences, forward sensitivities,
  scalar row loops, stopped gradients, or fallback values in the primary method.
- Do not form the full Jacobian in the production/timed builder.
- Do not retain `batched_static_autodiff_probe` as a valid method identity or
  describe scalar row autodiff as the primary comparator.
- Do not change timing/materialization boundaries; that is Phase 5.
- Do not run GPU, a CPU scaling ladder, HLO grid, score timing, or comparison
  benchmark.
- Do not weaken parity or row-independence limits after seeing results.
- Do not overwrite historical artifacts or modify/revert unrelated dirty work.

## Exact Next-Phase Handoff Conditions

All conditions are conjunctive:

- Root-cause/fail-closed, parity, Jacobian, perturbation, method-contract, stale-
  resume, and aggregate mutation tests pass.
- Strict non-JIT and CPU-XLA JSON artifacts have `state=passed` and every named
  gate true. The result preserves distinct CLI-producer and pytest-regression
  outer command manifests with their own exit/timeout, wall time, log hash, and
  respectively durable JSON/hash/strict-read or exact test node/count.
- The method schema is v3, the method-contract version is current, and default
  schedule contains exactly the analytical/autodiff primary pair.
- The bounded CPU-XLA smoke passes within 120 seconds after non-XLA gates.
- Compile, focused pytest, scoped diff, strict JSON, read-only/historical hashes,
  and no-worker checks pass.
- Phase 4 result records exact root cause, method migration, residuals, run
  manifest, decision table, uncertainties, and all nonclaims.
- Refreshed Phase 5 subplan receives exact `VERDICT: AGREE` from the available
  bounded reviewer, explicitly weaker than Claude if substitute review is used.

## Stop Conditions

- The corrected direct-vector VJP remains `None` or non-finite after localized
  repair without changing the batch likelihood target.
- Full Jacobian off-diagonal blocks or perturbation checks show genuine cross-row
  coupling that cannot be removed within fixture/harness semantics.
- Scalar autodiff and analytical parity cannot be restored within prospective
  tolerances without changing the target or reference math.
- Another lane changes a declared Phase 4 path and overlap cannot be reconciled.
- Required artifact/provenance cannot be satisfied or new human authority is
  needed.
- The same material review blocker does not converge after five rounds.

A normal tape-construction, shape, tolerance implementation, schema migration,
or tiny CPU-XLA failure is a repair trigger first. Stop only if it becomes one
of the continuation vetoes above.

## Mandatory Phase-End Sequence

1. Run every required local check.
2. Write the Phase 4 result/close record.
3. Refresh the Phase 5 subplan from actual evidence.
4. Review Phase 5 for consistency, correctness, feasibility, artifact coverage,
   and boundary safety; visibly repair and recheck before advancing.
