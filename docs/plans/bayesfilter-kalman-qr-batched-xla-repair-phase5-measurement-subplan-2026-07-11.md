# Phase 5 Subplan: Compile And Runtime Measurement Separation

Date: 2026-07-11
Status: `REVIEW_CONVERGED_ROUND_4_EXECUTION_READY`

## Phase Objective

Replace the historical end-to-end `_time_call` measurement with explicit,
fail-closed stages for TensorFlow trace, first executable call, synchronized
warm execution, host materialization, and artifact/report writing. Preserve the
same method outputs and method isolation while making every duration's boundary
auditable.

This phase validates measurement mechanics only. It does not run the target
CPU/GPU grid, compare method speed, choose a winner, tune XLA, or change a
default.

## Entry Conditions Inherited From Phase 4

- Phase 4 result is `LOCAL_GATE_PASSED_PHASE5_REVIEW_PENDING`.
- Strict non-JIT and CPU-XLA JSON artifacts both have `state=passed` and all 22
  independently recomputed gates true under declared-source fingerprint
  `3c8f75ca...`.
- The v3 primary pair is exactly batch-native analytical and batch-native
  autodiff; scalar methods remain explicit correctness references.
- The production autodiff callable uses one vector VJP and no fallback.
- The dynamic batched value loop has `maximum_iterations=n_timesteps`; the
  static Python time-loop route is not the primary comparator.
- Final non-XLA suite passed `236` tests with exactly one reviewed XLA-node
  deselection; independent CLI and exact-node CPU-XLA gates passed.
- All 14 historical anchors matched; no benchmark worker remained.
- Other authorized lane work is ignored. Stop only for overlap on a declared
  Phase 5 path that cannot be reconciled by exact hashes.
- Claude is platform-policy-blocked before probe. Bounded Codex substitute
  review is required and explicitly weaker.

## Measurement Contract

For one selected method/case, define these disjoint records:

1. `trace_seconds`: wall time to construct the selected `tf.function` and call
   `get_concrete_function` only. It is not XLA compilation time.
2. `first_executable_call_seconds`: wall time from invoking the concrete
   callable through completion of device work, before host serialization. It
   may include XLA compile and first execution and must be labeled exactly so.
3. `warm_execution_seconds`: synchronized device-execution durations after the
   first executable call. Each repeat returns device tensors and synchronizes
   only the minimal scalar sentinel needed to establish completion.
4. `materialization_seconds`: one host transfer and conversion of the complete
   `[B]` value plus `[B,P]` score after all timed warm calls.
5. `payload_encoding_seconds`: strict JSON encoding of an immutable measurement
   payload, outside every kernel and materialization duration.
6. `artifact_write_seconds`: atomic write of those already encoded payload bytes
   to a method-local sidecar, outside every kernel, materialization, and encoding
   duration.
7. The outer method-record envelope is written after the timed sidecar write. It
   records the sidecar path, SHA-256, encoding/write durations, and stage ledger,
   but its own write is deliberately untimed and labeled
   `envelope_write_measured=false`. The immutable sidecar does not contain its
   own encoding/write duration, avoiding a self-referential rewrite.

`first_executable_call_seconds - warm_execution_seconds` is not compilation
time and must not be emitted or described as such. Full `.numpy().tolist()` is
forbidden inside first/warm execution timing. Device tensors from the final
warm call may be materialized exactly once for parity/reporting. Every method is
measured in a fresh child process. An invocation counter starts at process entry
and covers fixture setup, builder selection, tracing, GraphDef inspection, and
execution; it must remain zero until the timed first concrete call and equal
`1 + repeats + 1` only after the post-measurement untimed reference call.
No selected-method callable invocation is allowed before the timed first call;
`get_concrete_function` and GraphDef inspection may trace but must not invoke the
concrete callable.

### Synchronization contract

- Prefer `tf.experimental.async_wait()` only if present and callable in the
  installed TensorFlow runtime.
- Otherwise synchronize by reducing the returned value and score to one scalar
  TensorFlow sentinel on the same device and materializing only that scalar.
- Record `synchronization_method` and the exact sentinel definition.
- Record `scalar_synchronization_materialization_count` separately from
  `full_output_materialization_count`. First/warm durations include host dispatch
  and the selected synchronization overhead; they are not kernel-only durations.
- Record `parity_residual_materialization_count=1`: after the untimed reference
  call, value and score maximum residuals are packed into one two-scalar tensor
  and transferred once. This count is separate from timed-call synchronization
  and full-output materialization.
- If `async_wait` is selected, the fresh method child must have no other pending
  TensorFlow work because `async_wait` may wait for all executor work.
- Never materialize the full value/score inside first/warm timing.
- Tests must prove the selected synchronization path is called once per timed
  execution and full `_materialize` is not called until the materialization
  stage.

## Required Artifacts And Write Set

Allowed Phase 5 writes:

- `scripts/benchmark_kalman_qr_parameter_count_scaling.py`, limited to reusable
  synchronization/timing primitives and one selected-method measurement path;
- `scripts/kalman_qr_benchmark_contract.py`, only if a closed v4 measurement
  record schema or timing-boundary version is required;
- `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`, limited to
  passing/validating measurement identity and preserving isolated failure stages;
- new `tests/test_kalman_qr_measurement_boundaries.py`;
- directly required regressions in
  `tests/test_kalman_qr_parameter_count_scaling_harness.py` and
  `tests/test_kalman_qr_benchmark_contract.py`;
- strict tiny CPU measurement-smoke JSON
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json`;
- immutable method-local timed payload sidecars under
  `/tmp/kalman_qr_phase5_measurement/methods/`, referenced by SHA-256 from the
  strict smoke envelope;
- exact runtime artifacts:
  `/tmp/kalman_qr_phase5_measurement/schedule.json`,
  `/tmp/kalman_qr_phase5_measurement/status.json`, child envelopes under
  `/tmp/kalman_qr_phase5_measurement/methods/<20-hex>.json`, immutable payload
  sidecars under
  `/tmp/kalman_qr_phase5_measurement/methods/<20-hex>.payload.json`, and
  progress journals under
  `/tmp/kalman_qr_phase5_measurement/progress/<16-hex-case>-<method-id>-<32-hex-attempt>.jsonl`;
- logs under `/tmp/kalman_qr_phase5_measurement/`;
- Phase 5 result
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase5-measurement-result-2026-07-11.md`;
- refreshed Phase 6 subplan
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-subplan-2026-07-11.md`;
- at most five Phase 6 review records named
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-subplan-codex-substitute-review-round<N>-2026-07-11.md`.

Read-only:

- every `bayesfilter/linear/*.py` algorithm source;
- Phase 4 numerical/XLA JSON artifacts and result;
- all historical `*2026-07-09*` artifacts.

## Required Checks, Tests, And Review

### 1. Pure boundary tests

- Fake callable tests prove trace, first, warm, materialization, and write
  stages are ordered and non-overlapping.
- Use `time.perf_counter_ns` for a raw ordered `stage_events` ledger. Every event
  has a closed stage name, `entered_ns`, `finished_ns`, and strictly increasing
  sequence index; intervals must be nonnegative and match the exact stage list,
  and `event[i].finished_ns <= event[i+1].entered_ns` must hold for every pair.
- Failure injection covers builder construction, `get_concrete_function`, first
  invocation, first synchronization, every warm invocation/synchronization,
  full materialization, strict encoding, temporary-file write/fsync, and atomic
  replacement. Each records the exact failure stage and cannot produce a passed
  record. When the child cannot write its envelope, the supervisor synthesizes
  a stage-specific failure from the durable progress journal and preserves any
  passed sibling record.
- Failure injection also covers progress-journal exclusive creation, append,
  flush/fsync, partial/corrupt last lines, and outer-envelope encoding,
  temporary-file write/fsync, and atomic replacement. A journal failure is a
  fail-closed supervisor structural failure, never a guessed child stage.
- Full materialization is called exactly once and only after warm timing.
- Scalar synchronization materializations equal `1 + repeats` for the sentinel
  path and zero for `async_wait`; full-output materialization count is one; and
  post-measurement parity-residual materialization count is one.
- Direct parity and GraphDef extraction occur outside every timed execution,
  materialization, encoding, and sidecar-write interval.
- Artifact serialization is outside all kernel/materialization timings. The
  timed immutable payload is encoded once, written once, hashed after close, and
  never rewritten; the outer envelope write is untimed.
- Every required duration is a finite nonnegative number; missing, boolean,
  negative, NaN, or infinite values fail validation.
- `warm_execution_seconds` has exactly the requested repeat count.
- No field named or described as pure compilation time or
  `first_minus_warm` remains in the v4 measurement artifact.

### 2. TensorFlow measurement tests

Use deterministic `dimension=2,T=4,P=3,B=4,float32` on GPU-hidden CPU:

- both primary methods return exact `[4]` and `[4,3]` float32 finite outputs;
- timed and direct non-timed outputs match at Phase 4 float32 tolerances. The
  comparison uses the already materialized final warm outputs against a
  post-measurement untimed reference evaluation in the same fresh process. The
  reference comparison is computed on device and reduced to scalar residuals;
  only those residual sentinels are materialized, so it does not perform a
  second full output transfer;
- one full output materialization occurs after the synchronized warm calls;
- `jit_compile=false` smoke validates boundaries without XLA;
- one bounded `jit_compile=true` CPU-XLA smoke validates the same boundary
  schema but is compatibility evidence only.

### 3. Graph and method isolation

- Trace records GraphDef node count and serialized bytes for only the selected
  method; these are explanatory in Phase 5.
- Selected-builder tests cover all four methods and prove sibling builders are
  never constructed.
- The default supervisor remains exactly `PRIMARY_METHOD_IDS`.
- A method-local failure does not erase a passed sibling artifact.

### 4. Strict raw evaluator

The smoke artifact must be evaluated from raw fields against an independently
constructed expected contract. Named gates cover:

- schema/timing-boundary/method/fixture/source/runtime identities;
- exact argv, output/log/plan/result paths, git commit, CPU/GPU-hidden/thread,
  JIT/XLA/TF32, and synchronization method;
- exact raw stage-event intervals, payload-sidecar path/hash, immutable
  payload-write count, untimed-envelope marker, scalar synchronization count,
  parity-residual materialization count, and full-output materialization count;
- finite nonnegative durations, requested warm count, shapes/dtypes/finiteness,
  direct-output parity, GraphDef metadata, and nonclaims.

One-field raw mutations, including every event boundary/count and sidecar
identity field, must make the corresponding gate false, overall `state=failed`,
and return code nonzero. Forged precomputed `checks=true` cannot rescue invalid
raw evidence.

### 5. Exact commands

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py

CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_batch_native_autodiff.py \
  --deselect=tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value

git diff --check -- \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py

sha256sum \
  bayesfilter/linear/kalman_qr_tf.py \
  bayesfilter/linear/kalman_qr_derivatives_tf.py \
  bayesfilter/linear/qr_factor_tf.py

pgrep -af 'run_kalman_qr_batched_xla_repair|benchmark_kalman_qr_parameter_count_scaling|pytest' || true
```

Before any edit, record SHA-256 for every allowed-write path that exists and the
three read-only algorithm paths. At close, list the actual changed path set,
verify it is a subset of the allowed writes, and verify the three complete
read-only hashes remain exactly:

- `bayesfilter/linear/kalman_qr_tf.py`:
  `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b`;
- `bayesfilter/linear/kalman_qr_derivatives_tf.py`:
  `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57`;
- `bayesfilter/linear/qr_factor_tf.py`:
  `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401`.

The pre-edit ledger is
`/tmp/kalman_qr_phase5_measurement/pre_edit_path_hashes.sha256`; create it before
source/test edits with this exact command after the directory exists:

```bash
sha256sum \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  scripts/kalman_qr_benchmark_contract.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py \
  docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-subplan-2026-07-11.md \
  bayesfilter/linear/kalman_qr_tf.py \
  bayesfilter/linear/kalman_qr_derivatives_tf.py \
  bayesfilter/linear/qr_factor_tf.py \
  > /tmp/kalman_qr_phase5_measurement/pre_edit_path_hashes.sha256

printf '%s\n' \
  'ABSENT tests/test_kalman_qr_measurement_boundaries.py' \
  'ABSENT docs/benchmarks/kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json' \
  'ABSENT docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase5-measurement-result-2026-07-11.md' \
  >> /tmp/kalman_qr_phase5_measurement/pre_edit_path_hashes.sha256
```

The three round-1 through round-3 Phase 5 review records already exist before
implementation and are review evidence, not implementation writes; record their
hashes separately in the Phase 5 result. Any later Phase 5 review record must be
listed explicitly at creation. The result must preserve the pre-edit ledger
SHA-256 and compare every opening path to its closing hash or explicitly classify
it as an allowed Phase 5 change. Repository-wide status is descriptive only
because the other authorized lane is active; the exact declared path set is
binding.

After the local checks pass, run this exact dedicated smoke from the repository
root; the supervisor launches one fresh method child for each primary method:

```bash
mkdir -p /tmp/kalman_qr_phase5_measurement
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  timeout 210s /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --dimensions 2 \
  --parameter-counts 3 \
  --timesteps 4 \
  --batch-size 4 \
  --dtype float32 \
  --device cpu \
  --cpu-threads 1 \
  --repeats 2 \
  --timeout-seconds 90 \
  --methods batch_native_analytical_qr_score batch_native_autodiff_qr_score \
  --output-dir /tmp/kalman_qr_phase5_measurement \
  --no-resume \
  --jit-compile \
  --tf32-enabled \
  > /tmp/kalman_qr_phase5_measurement/smoke.log 2>&1
```

The 210-second cap is a convenience bound for two sequential 90-second method
children plus at most 30 seconds of supervisor startup, validation, and durable
write overhead. Each child timeout is expected to produce a structured
stage-specific supervisor record. The outer `timeout 210s` is an emergency cap:
if it fires, only its nonzero exit and `smoke.log` are durable evidence; do not
claim a structured status record, do not run the export evaluator, and do not
close Phase 5. Repair the supervisor/runtime path prospectively or write a
blocker. Neither timeout proves that XLA is impossible.

After a zero exit, run the exact strict evaluator/export entry point:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --evaluate-phase5-smoke \
  --phase5-input /tmp/kalman_qr_phase5_measurement/status.json \
  --phase5-log /tmp/kalman_qr_phase5_measurement/smoke.log \
  --phase5-output docs/benchmarks/kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json
```

The strict evaluator writes the validated aggregate envelope to the declared
repository path and embeds each immutable payload sidecar's exact strict-JSON
content plus SHA-256. The repository artifact is therefore the durable evidence;
the unique no-resume `/tmp/kalman_qr_phase5_measurement/` directory remains a
reproducibility aid but is not required for later verification. The evaluator
records `/tmp/kalman_qr_phase5_measurement/smoke.log` as the external log.
The exact argv recorded in the artifact excludes shell-only environment,
redirection, and outer `timeout`; those are recorded separately in the run
manifest. A non-JIT escape hatch is debug-only and cannot close Phase 5.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does each recorded duration measure its declared stage without silently including full-output serialization or another method? |
| Exact baseline | Historical `_time_call`, which includes full host materialization in first and warm durations. |
| Primary criterion | Boundary tests, selected-method isolation, strict raw evaluator, and tiny JIT-on CPU smoke all pass with one post-timing materialization. |
| Promotion vetoes | Hidden full host transfer; unsynchronized execution; ambiguous first-call label; overlapping stages; invalid duration; sibling method construction; stale or corrupt artifact. |
| Repair triggers | Local timing helper, synchronization, stage labeling, or artifact-schema defects within the write set. |
| Explanatory only | Observed stage durations and GraphDef size from the tiny smoke. |
| Not concluded | Method speed ranking, CPU/GPU scalability, compile-time estimate, HMC/posterior/default/production/scientific validity. |

## Skeptical Pre-Execution Audit

- The baseline is the known materializing timer, not an assumed accurate
  measurement.
- The primary criterion is boundary correctness, not a favorable duration.
- First executable call is explicitly allowed to mix XLA compile and execution;
  no subtraction proxy is promoted to compile time.
- Full output transfer is directly counted and ordered, not inferred from a
  timing difference.
- One tiny smoke cannot rank methods or establish target-scale feasibility.
- Failure of a timing helper triggers repair and does not invalidate the Phase 4
  numerical target or comparator.

Audit status: `PASSED_AFTER_BOUNDED_CODEX_SUBSTITUTE_REVIEW_ROUND_4`.

## Forbidden Claims And Actions

- Do not call first-minus-warm or any subtraction pure compilation time.
- Do not describe requested TensorFlow threads as pinned physical cores.
- Do not time or serialize more than one selected method per child.
- Do not materialize full outputs in first/warm execution timings.
- Do not invoke the selected method before its timed first executable call or
  reuse a process across methods.
- Do not claim the outer envelope write is measured; only the immutable payload
  sidecar encoding/write stages have measured durations.
- Do not run GPU, the target CPU ladder, HLO dumps, or comparison grid.
- Do not edit Kalman/QR algorithm sources or change Phase 4 tolerances.
- Do not use tiny-smoke durations to rank methods or change defaults.
- Do not modify/revert unrelated dirty work.

## Exact Next-Phase Handoff Conditions

All conditions are conjunctive:

- Compile, focused tests, scoped diff, strict JSON, and no-worker checks pass.
- The strict JIT-on CPU smoke has `state=passed` and every named gate true.
- Trace, first executable, each warm call, materialization, and artifact write
  have distinct finite nonnegative fields and exact boundary definitions.
- Full output materialization count is exactly one and occurs after warm calls.
- The raw monotonic stage ledger proves interval ordering/non-overlap; scalar
  synchronization counts match the selected method; the immutable payload
  sidecar hash matches; and the outer envelope is explicitly untimed.
- Both primary methods pass the same tiny measurement schema in isolated child
  records; observed durations remain descriptive only.
- Phase 5 result records exact commands, source/artifact/log hashes, decision
  table, run manifest, uncertainty, and nonclaims.
- Phase 6 is refreshed from actual graph/timing feasibility and receives exact
  `VERDICT: AGREE` from the available bounded reviewer.

## Stop Conditions

- No reliable supported synchronization method exists and scalar-sentinel
  synchronization cannot establish completion without contaminating full-output
  timing.
- Timing stages cannot be separated without changing computed outputs.
- A declared path overlaps irreconcilably with the other lane.
- Required provenance/artifact cannot be made fail closed.
- New human authority is required.
- The same material review blocker does not converge after five rounds.

## Mandatory Phase-End Sequence

1. Run all required local checks and strict smoke.
2. Write the Phase 5 result/close record.
3. Draft or refresh Phase 6 from actual evidence.
4. Review Phase 6 for consistency, correctness, feasibility, artifact coverage,
   and boundary safety; repair visibly until convergence or five rounds.
