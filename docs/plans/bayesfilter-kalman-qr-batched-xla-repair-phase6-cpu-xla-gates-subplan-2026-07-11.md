# Phase 6 Subplan: GPU-Hidden CPU Trace And XLA Gates

Date: 2026-07-11
Status: `ROUND4_REPAIRED_AWAITING_FINAL_BOUNDED_REVIEW_ROUND5`

Round-1 opening SHA-256:
`e1b72b39bc4c37409e748c51fee3ff242008d2832cb70139b02cfef430318e91`.

Round-2 reviewed SHA-256:
`25bbd3f21af4bed07e7db8834d5a2ff0c85ffa8b4f3d0591e86629c821d4c294`.
Round-2 review record SHA-256:
`bf558874a15239cc3cc7cb3418087cf807f5a99ac98e6d69327f088ff4afbe0d`.

## Phase Objective

Establish bounded GPU-hidden CPU structural trace, XLA execution, and numerical
validity evidence for the repaired batch-native analytical and batch-native
autodiff paths across the inherited target `dimension/P/B` cells. Classify
common invalidity separately from CPU-backend, method, and current-cell
failures before any trusted GPU run.

Phase 6 is not a timing comparison. It does not run requested thread settings
`4/16`, rank methods, tune XLA, dump HLO, change defaults, or make a CPU
production claim.

## Entry Conditions Inherited From Phase 5

- Phase 5 result is `LOCAL_GATE_PASSED_PHASE6_REVIEW_PENDING`.
- Strict Phase 5 repository artifact
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json`
  has SHA-256
  `a74be199826f12b2c7931e7bb8d82d510826b69fcb7232ca3ad0b255b90ce74d`,
  `state=passed`, and 42/42 gates true.
- Final GPU-hidden non-XLA suite passed `207` tests with exactly one reviewed
  CPU-XLA node deselected; log SHA-256 is
  `3ddb89c43839be3c190cb1026754baa27340271e7fbe4ff3fdd6181f9523459e`.
- The v4 selected-method contract is
  `measurement-boundaries-phase5-v1`; methods execute in fresh isolated
  children and use separated trace/first/warm/materialization/reporting stages.
- Both primary methods passed the tiny `dimension=2,T=4,P=3,B=4,float32`
  CPU-XLA smoke. Child wall times were about five seconds, but these are not
  target-scale timeout estimates.
- Final execution source fingerprint is
  `56f0a447f1a12516a78ae5c98d64ed2f5f2c6f611d8f0e2c5d83f67d95b5fbc6`.
- Read-only algorithm hashes are exactly:
  - `kalman_qr_tf.py`: `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b`;
  - `kalman_qr_derivatives_tf.py`: `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57`;
  - `qr_factor_tf.py`: `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401`.
- Claude remains platform-policy-blocked before probe. Bounded Codex substitute
  review is required and explicitly weaker.
- Other authorized lane work is ignored. Stop only for irreconcilable overlap
  on a declared Phase 6 path, detected by exact opening/closing hashes.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | After the fixture, analytical parameter-axis, autodiff, and measurement repairs, which target-grid cells trace and execute under GPU-hidden CPU XLA within bounded resources while preserving finite/dtype/shape/parity validity? |
| Candidate mechanisms | B-independent TensorFlow fixture algebra, vectorized analytical derivative helpers, one-VJP batch-native autodiff, dynamic time loop with a TensorList bound, method-isolated fresh children, and separated synchronization/materialization. |
| Expected failure mode | Graph/HLO/codegen may still grow with dimension, `T=120`, `P=150`, or `B=16`; CPU XLA may time out/OOM/crash even when common math is correct. |
| Promotion criterion | The structural trace census is valid and each launched XLA cell is honestly classified. A cell is CPU-XLA viable only if both primary methods pass finite/dtype/shape/parity/provenance gates on that same cell. |
| Promotion veto | Invalid/stale artifact, source/config drift, non-finite output, dtype/shape/parity failure, hidden full materialization, wrong device/JIT/thread identity, or method coupling. |
| Continuation veto | Common harness/fixture/math invalidity; corrupted required artifact; source/read-only overlap; or new authority boundary. CPU-backend or one-method failure is lane-local, not automatically a continuation veto for Phase 7. |
| Repair trigger | Structural graph growth inconsistent with batch/parameter vectorization; missing failure stage; a smallest cell failure attributable to an in-scope harness defect; or strict evaluator mismatch. |
| Explanatory diagnostics | GraphDef nodes/bytes/trace time, first executable call, warm calls, child wall time, failure stage/error tail, requested affinity/thread settings. |
| Must not conclude | CPU/GPU scalability, method superiority, pure compilation time, physical-core scaling, production/default/HMC/posterior/scientific readiness. |

## Required Artifacts And Write Set

Allowed Phase 6 writes:

- `scripts/benchmark_kalman_qr_parameter_count_scaling.py`, limited to a
  Phase 6 trace-only diagnostic/evaluator and directly required v4 target-cell
  metadata;
- `scripts/kalman_qr_benchmark_contract.py`, only for a closed Phase 6 schema or
  fail-closed evaluator fields that cannot live in the runner;
- `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`, limited to
  target-cell schedule ordering, branch pruning, CPU-XLA result aggregation,
  and strict Phase 6 export;
- new `tests/test_kalman_qr_phase6_cpu_xla_gates.py`;
- directly required regressions in the three Phase 5 harness/contract tests;
- structural artifact
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_trace_census_2026-07-11.json`;
- immutable Gate B budget proposal
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_pilot_budget_2026-07-11.json`;
- detached Gate B review and runtime attestation
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-budget-review-round1-2026-07-11.md`
  and
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_budget_attestation_2026-07-11.json`;
- immutable pilot artifact
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_pilot_2026-07-11.json`;
- reviewed remaining-budget contract
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_remaining_budget_2026-07-11.json`;
- detached Gate C review record
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-budget-review-round1-2026-07-11.md`;
- detached Gate C runtime attestation
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gatec_budget_attestation_2026-07-11.json`;
- durable `P=150` routing artifact
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_p150_routing_2026-07-11.json`;
- target scalar-reference artifact
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_scalar_references_2026-07-11.json`;
- final CPU-XLA artifact
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_2026-07-11.json`;
- logs/runtime artifacts under `/tmp/kalman_qr_phase6_cpu_xla/`;
- Phase 6 result
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-result-2026-07-11.md`;
- refreshed Phase 7 subplan and at most five bounded review records.

Read-only:

- algorithm sources `bayesfilter/linear/kalman_qr_tf.py`,
  `bayesfilter/linear/kalman_qr_derivatives_tf.py`, and
  `bayesfilter/linear/qr_factor_tf.py`;
- direct execution dependencies `bayesfilter/__init__.py`,
  `bayesfilter/linear/__init__.py`, `bayesfilter/diagnostics.py`,
  `bayesfilter/structural.py`, `bayesfilter/results_tf.py`,
  `bayesfilter/linear/dtypes_tf.py`, and
  `bayesfilter/linear/types_tf.py`;
- Phase 4/5 results and JSON artifacts;
- every historical `*2026-07-09*` artifact.

The existing v4 source fingerprint remains preserved for Phase 5 historical
identity, but it is not sufficient Phase 6 provenance because the algorithm
modules import repository dependencies. Gate A adds a closed
`phase6_execution_dependency_manifest` to every Phase 6 proposal, child,
ledger, and evaluator.

Before each budget proposal is frozen, an import-only GPU-hidden discovery
child starts from a clean interpreter, imports the exact supervisor/benchmark
modules without building a fixture, tracing, or invoking a selected method,
then records every `sys.modules` entry whose resolved regular-file `__file__`
is under the resolved repository root. The manifest contains module name,
resolved repository-relative path, SHA-256, discovery command/environment,
and canonical digest. It must include the three algorithm files, the direct
dependencies listed above, and the three allowed-write execution paths; missing
required entries fail proposal construction.

Every trace/XLA/scalar child emits the same actual loaded-repository-module
manifest both immediately before selected-method construction and after its
terminal stage. The strict evaluator requires every actual module/path/hash to
belong to and match the reviewed discovery closure; a newly loaded repository
module, alias resolving outside/inside unexpectedly, symlink/non-regular path,
duplicate module path with conflicting bytes, or changed hash is invalid
evidence and stops the gate. Unused entries in the reviewed discovery closure
may remain. This binds the actual transitive repository-import closure without
claiming that the handwritten direct list is exhaustive.

Opening hashes are frozen in the reviewed Gate B/C proposal. Any drift before
launch invalidates the proposal and requires a regenerated proposal and review;
drift during a gate terminalizes the active child, marks that gate invalid, and
forbids mixing pre/post-drift records. This lane never edits or reverts a
read-only dependency, even if another authorized lane owns it.

## Target Cells And Branch-Pruning Contract

The inherited target cells are:

- `dimension in {10,20,30}`;
- `T=120`;
- `P in {50,150}`;
- `B in {1,4,16}`;
- `dtype=float32`;
- `device=cpu`, `CUDA_VISIBLE_DEVICES=-1`, requested threads `1`;
- primary methods only for XLA execution.

### Structural trace census

Trace both primary methods without XLA execution for all 18 `dimension/P/B`
cells, one fresh process per method/cell with a 60-second execution deadline and
70-second lifecycle cap. Record
GraphDef nodes, deterministic serialized bytes, ordered-op digest/histogram,
input/output signatures, trace time, method/fixture/source identity, and exact
failure stage. This artifact is explanatory/structural and cannot promote a
cell to CPU-XLA viability.

Structural gates:

- every target method/cell is represented by a passed or honest bounded
  failure record;
- within each fixed `dimension/method` six-cell cohort, exact topology is
  invariant across `B=1/4/16` and `P=50/150`: top-level/function node count,
  node/function names and order, op types, input/control-edge lists, device
  strings, attribute keys, and output tensor-index mapping all match;
- the cohort-level typed GraphDef diff contains only prospectively accepted
  `B/P` static-shape specialization described below. Any topology growth,
  extra op/function/control-flow body, float/string/bool constant change, or
  unexplained attribute/payload difference is a repair trigger;
- output signatures are exactly `[B]` and `[B,P]`, dtype float32;
- no selected method executes, no explicit TensorFlow device-enumeration API is
  called, and CUDA is hidden before TensorFlow import. A TensorFlow import-side
  CUDA diagnostic is recorded as an import side effect, not GPU evidence.

### Lossless GraphDef extraction and closed typed diff

Every trace child constructs a separate fixed-input concrete function for its
exact named `TensorSpec(shape=[B,P], dtype=float32, name="parameters_batch")`.
It calls `get_concrete_function()` with no runtime tensor and exactly once, then
extracts `concrete.graph.as_graph_def(add_shapes=True)`. It never invokes the
concrete function. The artifact records, in concrete order:

- the one structured user input and every captured input, each with tensor
  name including output index, dtype, and full shape;
- every concrete output with tensor name including output index, dtype, full
  shape, and result position (`value=0`, `score=1`);
- the complete `GraphDef` deterministic bytes, base64 encoded losslessly, byte
  count, SHA-256, GraphDef versions, and extraction-version string; and
- the complete parsed top-level nodes and function library as a derived typed
  token stream. Tokens retain protobuf field path, repeated-field index, wire
  type, dtype, tensor shape, raw `tensor_content`, and all repeated scalar
  values. The token-stream digest is explanatory; the embedded GraphDef bytes
  are authoritative and let the strict evaluator independently parse and
  recompute every derived field.

No GraphDef is destructively normalized. For each fixed `dimension/method`, the
strict evaluator jointly parses all six raw graphs and emits an exhaustive typed
diff. A differing coordinate is accepted only when all of these hold:

1. the node/function topology fields listed in the structural gate are exact;
2. the field is a TensorShape dimension, an `_output_shapes`/`shape` dimension,
   or a function `ArgDef` shape dimension;
3. across the full cohort, its distinct values track exactly one declared axis:
   `B -> {1,4,16}` at fixed `P`, or `P -> {50,150}` at fixed `B`; values cannot
   be accepted from a single pairwise comparison;
4. every `Const` dtype, tensor shape, `tensor_content`, and repeated scalar value
   is byte-identical. No axis-correlated integer constant is accepted in this
   first census; a required constant exception is a rejected diff and needs a
   future path/consumer-specific plan repair and review before XLA;
5. float, complex, string, bool, resource, variant, and raw undecodable payloads
   are byte-identical. Node names, edge names, op names, devices, attribute keys,
   and all non-shape attributes are byte-identical.

Function definitions are equal modulo only the accepted, tagged
`ArgDef`/node-shape coordinates above: topology, signature names/dtypes,
function attributes, node order/ops/edges/attribute keys, control returns, and
all other bytes are exact. The evaluator produces a canonical comparison copy
only after it records each raw coordinate/value: accepted shape dimensions are
replaced by axis-specific `B_SENTINEL` or `P_SENTINEL`, deterministic bytes must
then match, and the authoritative raw function bytes remain embedded unchanged.

The artifact preserves every accepted and rejected diff coordinate, its six
raw values/digests, axis classification, and rule ID. Zero rejected differences
and exact output mappings are required. Node/op/count summaries cannot rescue a
rejected typed diff. This gate tests absence of graph construction growth while
retaining visible static shape specialization; it does not claim that fixed
`[B,P]` GraphDefs are byte-identical.

Before JSON parsing, `stat` must show a regular non-symlink file no larger than
512 MiB. Before base64 decoding, each string must be canonical RFC 4648 base64
with no whitespace, declared decoded length at most 16 MiB, encoded character
length exactly `4*ceil(decoded_length/3)`, correct terminal padding, and total
declared decoded bytes at most 256 MiB across all 36 records. Decode uses fixed
1 MiB encoded chunks into a size-limited spool while incrementally counting and
hashing; it aborts before writing past either declared or hard limits and only
then passes at most 16 MiB to the protobuf parser. Prospective evidence limits
are therefore 16 MiB decoded per GraphDef, 256 MiB decoded total, and 512 MiB
for the complete trace JSON on disk.
These are repository/memory safety ceilings, not hypotheses about algorithmic
scaling. They are conservative relative to the retained 0.20-0.30 MiB tiny
graphs but are not target-size evidence. A limit exceedance terminalizes the
record as `graphdef_evidence_size_cap_exceeded`, makes
`trace_common_valid=false`, prunes pilot XLA, and requires a visible prospective
plan/review before any cap change; the evaluator never partially decodes or
silently drops the offending graph/token fields.

A structural mismatch is a repair trigger, not a speed result. No accepted
typed-diff rule may be added after observing target GraphDefs; an unclassified
difference requires a visible plan repair and rereview before any XLA launch.

Gate B executes all 36 trace children first, terminalizes the complete trace
roster, and runs the final trace evaluator before constructing or launching
either pilot XLA child. Pilot XLA is eligible only when the final trace artifact
has `trace_common_valid=true`, zero pending/running/rejected-diff records, and
the exact reviewed source/runtime/schedule identities. A trace timeout, crash,
invalid record, or rejected typed diff produces durable trace evidence but
prunes both pilot XLA children as `not_launched_trace_gate_not_passed`; it cannot
be treated as permission to gather XLA evidence first and repair structure
later.

### XLA execution lattice

Execute method-isolated cells in this order, one requested CPU thread:

1. `dimension=10,P=50,B=1`;
2. `dimension=10,P=50,B=4`;
3. `dimension=10,P=50,B=16`;
4. `dimension=10,P=150,B=1`;
5. `dimension=10,P=150,B=4`;
6. `dimension=10,P=150,B=16`;
7. repeat rungs 1-6 for `dimension=20`;
8. repeat rungs 1-6 for `dimension=30`.

Each cell runs both primary methods in fresh sequential children. Dependencies
are method-local: one method's failure never prunes its sibling method.

Branch-pruning rules:

- If a method fails at `B=1` for a fixed `dimension/P`, do not launch its larger
  `B` cells for that branch; write explicit `not_launched_after_smaller_failure`
  records while the sibling method may continue.
- If both methods pass `B=1` but one fails at `B=4`, do not launch only that
  method at `B=16`; preserve sibling evidence.
- A `P=150,B=k` method child is eligible only if that method's `P=50,B=k`
  child passed and, for `k>1`, its preceding `P=150` batch rung passed.
- A CPU-specific timeout/crash/OOM/codegen failure does not block the smallest
  trusted GPU Phase 7 gate when common correctness remains passed.

### Exact `P=150` routing

The routing artifact has one required record keyed
`dimension=<d>/batch=<b>/method=<method>`. Each record contains the smaller
cell ID/state/digest, preceding `P=150` dependency when applicable, immutable
source/config/runtime/fixture/schedule digests, exact rule ID, and final action.
It is preallocated as `pending_dependency`, atomically persisted after each
dependency closes, and strictly reparsed before an eligible larger child starts.

| Dependency state | Exact `P=150` action |
| --- | --- |
| Same-`B` `P=50` passed and preceding `P=150` batch rung passed or is not applicable at `B=1` | `eligible_under_gate_c_budget` |
| Same-`B` `P=50` was itself pruned after a smaller-`B` failure | `not_launched_p50_dependency_not_launched` |
| Same-`B` `P=50` timed out, crashed, failed, or was interrupted | `not_launched_p50_dependency_failed:<exact_class>` |
| Same-`B` `P=50` evidence is invalid/stale/corrupt | `not_launched_invalid_dependency_evidence` and continuation veto pending repair |
| Preceding `P=150` batch rung did not pass | `not_launched_after_smaller_p150_batch_failure:<exact_class>` |
| Common invalidity already fired | `not_launched_common_invalidity` and continuation veto |

There is no exceptional post-failure `P=150` launch in Phase 6. This avoids a
post-result authority cycle and prevents a larger parameter cell from being
used to reinterpret a smaller failure. The trace census still observes both
`P` values and can localize graph specialization without XLA execution.

## Prospective Numeric Defaults

| Number | Provenance/classification | Use | Failure handling |
| --- | --- | --- | --- |
| Trace execution deadline 60 s; lifecycle cap 70 s | Convenience; Phase 3 target-P traces were seconds, not target `T/B/dimension` evidence | One method/cell trace-only process; TERM at 60 s, KILL no later than 65 s, reap/prove gone by 70 s | Record timeout at `trace`; inspect smallest case before changing prospectively |
| XLA execution deadline 60 s; lifecycle cap 70 s | Conservative hypothesis from Phase 5 five-second tiny children and historical one-hour failures | One method/cell full v4 child; TERM at 60 s, KILL no later than 65 s, reap/prove gone by 70 s | Timeout is lane/cell evidence, not impossibility proof; no silent increase |
| Supervisor emergency cap 160 s per two-method cell | Two sequential 70 s lifecycle caps plus 20 s persistence/evaluation overhead | Prevent orphaned long cells | If it fires, terminate/reap, persist interruption, and write blocker |
| Warm repeats 2 | Inherited Phase 5 measurement-mechanics minimum | Compatibility diagnostics only | No ranking/statistical inference |
| Requested CPU threads 1 | Inherited Phase 6 first gate | Avoid thread-grid confounding | Do not describe as core pinning |
| Float32 parity | Locked Phase 4 `rtol/atol`: value and score `2e-4/2e-4` | Directed analytical-reference parity | No post-result tolerance change |

Phase 6 does not run the entire 18-cell XLA lattice in one opaque command. The
supervisor persists each child immediately and checks source/schedule identity
before the next child. Runtime authority is split into the following gates:

| Gate | Authorized work | Prospective hard ceiling | Authority condition |
| --- | --- | --- | --- |
| A: implementation only | Implement commands, schemas, evaluators, atomic persistence, and pure/tiny tests | No target trace, target XLA, or target scalar reference | This repaired subplan must first receive bounded agreement |
| B: pilot | Full 36-child trace census plus only `dimension=10,P=50,B=1` for both XLA methods | Trace lifecycle maximum `36*70=2520`; XLA cell 160; supervisor/evaluator reserve 320; shell TERM deadline 3000 plus 45-second KILL grace; hard outer ceiling 3045 seconds | Exact implemented commands/artifacts and opening hashes must be refreshed in this same subplan and receive a new bounded review |
| C: remaining | Reviewed branch-pruned remaining XLA lattice and target scalar references | Remaining XLA TERM deadline 2700 plus 45-second KILL grace; scalar TERM deadline 330 plus 45-second KILL grace; combined hard ceiling at most 3120 seconds | Pilot artifact/result and an immutable machine-readable budget proposal must be hashed, this subplan and exact proposal must receive separate bounded reviews, and detached attestations must bind those immutable digests |

Gate C review may narrow the schedule or ceilings after pilot evidence but may
not increase any per-child cap or the 3120-second hard overall ceiling. Any
increase requires a new human-visible plan/review and is not authorized by this
subplan.

Gate B and Gate C each use the same cycle-free authority protocol. The budget
JSON is an immutable proposal and contains no future review hash. The reviewer
reviews exactly that proposal path/digest plus the immutable plan path/digest.
After agreement, the Codex supervisor writes a detached strict-JSON attestation
containing the proposal path/digest, plan path/digest, review-record path/digest,
exact verdict, review strength, and timestamp. Runtime accepts explicit
proposal and attestation paths, independently hashes all bound files, requires
`AGREE`, and embeds both payloads/digests in its envelope. Neither proposal nor
plan is modified after attestation; any change invalidates authority and
requires a new proposal/review/attestation. This breaks the review-hash cycle.

These are maximum timeout budgets, not predicted runtimes or evidence of
feasibility. A timeout honestly classifies the launched work under its cap.

## Required Checks And Tests

### Pure/contract tests

- Closed Phase 6 trace and XLA artifact schemas reject missing/extra fields,
  non-finite durations, stale source/config/runtime/schedule identities,
  duplicate/missing cells, wrong method ordering, wrong device/thread/JIT/dtype,
  forged checks, and invalid not-launched reasons.
- One-field raw mutations make the named gate false, exported state failed, and
  evaluator return nonzero.
- Branch pruning preserves a passed sibling artifact and emits exact planned
  not-launched records without executing the pruned child.
- Timeout/crash/OOM/signal tests recover the last durable stage and do not
  misclassify CPU-backend evidence as common mathematical invalidity.
- Trace builder selection covers both primary methods and proves no sibling
  builder, device enumeration, concrete invocation, or XLA execution.

### Small correctness references

- Re-run the final Phase 5 local suite.
- Run scalar analytical/autodiff references only at
  `dimension=10,T=120,P=50,B=1/4,float32`, non-JIT and GPU-hidden, outside target
  XLA timings. Each of the four method/batch children has a 60-second execution
  deadline/70-second lifecycle cap and the scalar-reference supervisor has a
  330-second TERM deadline plus 45-second KILL grace. They are Gate C
  correctness references only. If infeasible under those reviewed caps, retain
  Phase 4 small-fixture references and label each target scalar reference
  `not_checked_timeout` rather than substituting a proxy or raising the cap.
- The four children are exactly
  `scalar_analytical_row_loop@B=1/4` and
  `autodiff_row_loop_qr_score@B=1/4`. After the remaining XLA command closes,
  the evaluator pairs each completed reference with the same-cell batch-native
  output: analytical-to-analytical and autodiff-to-autodiff. It also compares
  the two scalar reference methods.
- Every directed value and score comparison uses the reference operand in
  `abs(candidate-reference) <= 2e-4 + 2e-4*abs(reference)` elementwise with
  exact float32 `[B]`/`[B,50]` shapes and finite values. Stored comparison
  booleans cannot substitute for recomputation from embedded outputs.
- If both scalar references complete and disagree, target scalar-reference
  validity is `failed_scalar_reference_disagreement_unlocalized` and Phase 6
  stops for localization; the evidence does not identify which reference is
  wrong. If scalar references agree but exactly one completed batch/reference
  pair disagrees, that batch method receives a method-local numerical veto. If
  both completed batch/reference pairs disagree while the scalar pair agrees,
  status is `failed_common_or_cpu_xla_backend_unlocalized` and Phase 6 stops for
  focused localization; it must not be attributed to the fixture, harness,
  backend, or both methods without discriminating evidence.
  A timed-out/missing reference or unavailable batch output is
  `not_checked_missing_evidence`, cannot promote the affected cell, and is not
  itself mathematical invalidity.
- The aggregate `target_scalar_status` is `not_checked_timeout` only when all
  four reference children timed out validly and no comparison completed; it is
  `partial_missing_evidence` for every other valid mixture containing an
  individual `not_checked_missing_evidence` and no higher-precedence completed
  disagreement. Completed disagreement statuses take precedence over missing
  evidence. These mappings are recomputed from child records, not stored flags.

### Trace census checks

- Strict raw evaluator decodes the embedded raw GraphDef bytes and recomputes
  every extraction and typed-diff gate; stored metadata or `checks=true` is
  ignored.
- Source manifest and read-only algorithm hashes match opening values at close.
- The separate Phase 6 execution-dependency manifest matches the reviewed gate
  proposal before every child and at gate close; v4 source identity alone cannot
  satisfy this check.
- No output execution/materialization fields appear in trace-only records.

### XLA cell checks

- v4 measurement record, exact method/fixture/source/runtime/config/schedule
  identity, JIT on, CPU/GPU-hidden/thread identity, finite float32 `[B]` and
  `[B,P]`, direct repeat parity, immutable sidecar equality, and expected
  terminal/failure state.
- A two-method cell closes viability only if directed analytical/autodiff parity
  passes at the locked tolerance. One-method passes remain valid method-local
  engineering evidence but cannot close a fair-pair cell.

### Durable child-evidence protocol

`/tmp` journals/logs are recovery diagnostics only and cannot satisfy a Phase 6
handoff. After every child exit, timeout, signal, or launch failure, the
supervisor must:

1. read the child payload sidecar and append-only stage journal when present;
2. embed their parsed content, exact raw bytes encoded losslessly, SHA-256,
   byte count, exit status/signal, stdout/stderr tail plus hashes, and supervisor
   timestamps in the applicable repository JSON child record;
3. write a strict-JSON sibling temporary file, `fsync` the file, atomically
   `os.replace` the durable artifact, `fsync` its parent directory, then re-open
   and run the strict evaluator before launching another child; and
4. preserve the pilot artifact immutably after Gate B. Gate C reads it by path
   and digest and writes a separate final artifact rather than rewriting it.

An absent sidecar is valid only for a predeclared launch/timeout/crash stage
whose journal/exit evidence proves why it is absent. Malformed or inconsistent
sidecar data is invalid evidence, not a method failure. Resume begins only from
the last strictly reparsed repository artifact; `/tmp` alone is never resumed.

Every child is launched in a new process session/process group. The supervisor
owns the group ID and, on child cap, outer TERM, KeyboardInterrupt, or supervisor
exception, sends TERM to the whole group, waits at most five seconds for an
ordinary child cap, sends KILL to any surviving group, and reaps the direct
child by the 70-second lifecycle cap before a terminal ledger update. On outer
TERM it reserves at most 20 seconds for group TERM/KILL/reap, at most 10 seconds
for an atomic terminal ledger update/reparse, and leaves 15 seconds unallocated
for scheduling and `fsync` margin before the shell's 45-second KILL deadline.
Tests use real
harmless subprocess trees to prove no descendant survives timeout/interrupt.
Failure to prove reap is invalid supervisor evidence and stops further launches.

On resume, a durable `running` entry is never reused or silently reset. After
verifying that its recorded process group no longer exists, the supervisor
transitions it exactly once to terminal state `interrupted` with exact reason
`supervisor_recovery`, embedding the last valid stage journal and recovery
check. If the group still exists, resume
first terminates/reaps it under the same TERM/KILL protocol. If process identity
cannot be safely established, write a blocker and do not signal an unrelated
PID or launch another child.

### Preallocated ledger and state transitions

Each trace, pilot, scalar, routing, and final artifact is created before its
first child with the complete deterministic identity roster. Every roster entry
starts as the exact closed `pending` variant with null execution/evidence
fields. The only legal transitions are:

```text
pending -> running -> passed|failed|timed_out|crashed|interrupted
pending -> not_launched:<predeclared_reason>
```

At most one entry is `running`; terminal and `not_launched` entries are
immutable. Each update increments `update_index`, appends an event containing
the identity, prior/new state, UTC timestamp, and SHA-256 of embedded child
bytes, and rewrites the complete ledger atomically. Schedule-order and
dependency rules forbid advancing past an unresolved predecessor.

The in-progress evaluator requires the exact schema/roster, legal prefix state,
legal transition log, at most one running child, strict terminal variants,
source/config/runtime identity, and consistency of every embedded byte/hash. It
permits future `pending` entries and returns `state=running`, never `passed`.
The final evaluator uses the same raw ledger but requires no `pending` or
`running` entries, exact justified `not_launched` variants, all aggregate gates,
and `state=passed|complete_with_failures|failed`. The supervisor runs the
in-progress evaluator after each durable update and the final evaluator only at
closure. A failed reparse stops before the next child.

## Predeclared Commands

Before Gate A implementation, create
`/tmp/kalman_qr_phase6_cpu_xla/pre_edit_path_hashes.sha256` with exact hashes for
all existing allowed-write/read-only paths and explicit `ABSENT` lines for new
test/artifacts/result.

Final local checks:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py

CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_batch_native_autodiff.py \
  --deselect=tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value

git diff --check -- \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py
```

The exact prospective Gate B/C CLI shapes are below. Gate A must implement these
flags without weakening them. Before either command first executes, replace the
prospective marker with the exact `--help`-verified command, bind the opening
source/test hashes and applicable artifact hashes, and rereview this same
subplan.

Gate B trace census and pilot, prospectively:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  timeout --signal=TERM --kill-after=45s 3000s \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-pilot \
  --dimensions 10 20 30 --parameter-counts 50 150 \
  --batch-sizes 1 4 16 --timesteps 120 \
  --dtype float32 --device cpu --cpu-threads 1 --jit-compile \
  --trace-child-timeout-seconds 60 --xla-child-timeout-seconds 60 \
  --xla-cell-timeout-seconds 160 \
  --budget-contract docs/benchmarks/kalman_qr_batched_xla_repair_phase6_pilot_budget_2026-07-11.json \
  --budget-attestation docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_budget_attestation_2026-07-11.json \
  --trace-output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_trace_census_2026-07-11.json \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_pilot_2026-07-11.json
```

Gate C scalar references, prospectively:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  timeout --signal=TERM --kill-after=45s 330s \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-scalar-references \
  --dimensions 10 --parameter-counts 50 --batch-sizes 1 4 \
  --timesteps 120 --dtype float32 --device cpu --cpu-threads 1 \
  --no-jit-compile --child-timeout-seconds 60 \
  --budget-contract docs/benchmarks/kalman_qr_batched_xla_repair_phase6_remaining_budget_2026-07-11.json \
  --budget-attestation docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gatec_budget_attestation_2026-07-11.json \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_scalar_references_2026-07-11.json
```

Gate C branch-pruned remaining lattice, prospectively:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  timeout --signal=TERM --kill-after=45s 2700s \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-remaining \
  --dimensions 10 20 30 --parameter-counts 50 150 \
  --batch-sizes 1 4 16 --timesteps 120 \
  --dtype float32 --device cpu --cpu-threads 1 --jit-compile \
  --child-timeout-seconds 60 --cell-timeout-seconds 160 \
  --trace-input docs/benchmarks/kalman_qr_batched_xla_repair_phase6_trace_census_2026-07-11.json \
  --pilot-input docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_pilot_2026-07-11.json \
  --scalar-reference-input docs/benchmarks/kalman_qr_batched_xla_repair_phase6_scalar_references_2026-07-11.json \
  --budget-contract docs/benchmarks/kalman_qr_batched_xla_repair_phase6_remaining_budget_2026-07-11.json \
  --budget-attestation docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gatec_budget_attestation_2026-07-11.json \
  --routing-output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_p150_routing_2026-07-11.json \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_2026-07-11.json
```

Gate C consumes both scalar and remaining commands under one reviewed combined
3120-second hard ceiling including both 45-second KILL-grace windows. Neither command is authorized
before the pilot closes. The immutable budget proposal records the exact order,
planned cells/dependencies, per-child caps, TERM deadlines, KILL grace, hard
combined ceiling, and pilot/trace/subplan digests; it contains no review fields.
The detached attestation binds the separately written review record as specified
above. The scalar command cannot be run early as an allegedly small check.
The proposal predeclares command order `scalar_references` then
`remaining_lattice`, and the Gate C attestation contains one shared authority
ID. Both artifacts record monotonic start/end nanoseconds and the remaining
command verifies that scalar elapsed time plus its own prospective TERM/KILL
allowance cannot exceed 3120 seconds from the scalar command's recorded start.
An outer timeout or interrupted first command consumes actual elapsed time; it
does not reset the Gate C budget. If the remaining allowance is insufficient,
write deterministic `not_launched_global_budget_exhausted` records.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Which repaired target cells have stable batch/parameter graph structure and which methods execute under GPU-hidden CPU XLA within reviewed caps while preserving validity? |
| Baseline | Historical retained artifacts: parameter/batch static-unrolling and CPU XLA compile/codegen failures recorded in the reset memo. |
| Primary criterion | Complete honest trace census plus branch-pruned method-isolated XLA classifications; fair-pair viability requires both primary methods and locked parity on one cell. |
| Hard vetoes | Common invalidity, stale/corrupt artifact, non-finite/dtype/shape/parity failure, hidden materialization, wrong provenance, or boundary violation. |
| Lane-local vetoes | CPU XLA timeout/crash/OOM/codegen or one-method failure on a valid common harness. |
| Explanatory only | GraphDef nodes/bytes/trace time, first/warm durations, child wall time, and error tails. |
| Not concluded | CPU production target, GPU readiness, method superiority, physical-core scaling, HMC/posterior/default/production/scientific validity. |

## Skeptical Pre-Execution Audit

- Baseline is the historical invalid compile/codegen route, not its partial pass
  labels or runtime values.
- Structural GraphDef stability is a repair gate, not a runtime promotion
  criterion.
- The XLA lattice compares equivalent true-batched primary methods; scalar
  paths are small correctness references only.
- Timeouts are convenience hypotheses and classify cells under a cap; they do
  not prove impossibility.
- Branch pruning prevents known smaller failures from consuming hours while
  preserving sibling and lane-local evidence.
- `T=120` is inherited target scope. The trace-only census precedes XLA so a
  graph-scaling defect is localized cheaply.
- Requested threads are recorded settings, not physical-core pinning.
- CPU failure does not invalidate a later GPU repair phase unless common
  correctness/artifact assumptions fail.
- Exact target commands and total budget still require post-implementation
  refresh/review, so no target execution is currently authorized.
- Gate B cannot imply Gate C authority: pilot evidence must first be persisted,
  interpreted without ranking, and used to freeze the remaining budget.
- Losslessly embedded GraphDefs plus a closed cohort-level typed diff include
  internal attrs/shapes/constants, preventing node-count or op-histogram
  equality from concealing specialization.
- Target scalar references are target work and are therefore governed by Gate C,
  not by the pure/local-check exception.

Audit status: `PENDING_BOUNDED_REVIEW_AND_COMMAND_REFRESH`.

## Visible Round-1 Repair Record

The bounded Codex substitute round-1 review returned `REVISE`. This same
subplan was visibly patched as follows:

| Finding | Repair |
| --- | --- |
| Pilot and remaining-lattice authority conflicted | Split Gates A/B/C with separate immutable budget proposals, detached attestations, and non-increasable 3045/3120-second hard ceilings including KILL grace |
| Graph summaries could conceal specialization | Added losslessly embedded deterministic GraphDefs and an exhaustive closed cohort-level typed-diff contract |
| `P=150` failure routing was discretionary | Added a durable keyed routing artifact and deterministic dependency pruning; no post-failure exceptional launch remains |
| Target scalar references escaped command/budget review | Added exact prospective command, four 60-second execution/70-second lifecycle children, 330-second TERM plus 45-second grace, durable artifact, and Gate C prohibition |
| Phase 7 handoff implicitly required a CPU-XLA pass | Added an explicit all-CPU-XLA-lane-local-failure fallback below |
| `/tmp` evidence was not durable | Added lossless embedding, checksums, atomic repository persistence, strict reparse, and immutable pilot protocol |

## Visible Round-2 Repair Record

The bounded Codex substitute round-2 review returned `REVISE`. Its durable
record is
`docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-subplan-codex-substitute-review-round2-2026-07-11.md`.
This same subplan was visibly patched as follows:

| Finding | Repair |
| --- | --- |
| Gate C and exceptional-route review hashes were cyclic | Added immutable Gate B/C proposals plus detached review/runtime attestations; removed exceptional launches |
| Routing lacked pause/resume and pruned-dependency handling | Replaced it with deterministic same-`B`/preceding-rung dependency rules and exact not-launched classes |
| TERM deadlines omitted KILL grace | Declared hard ceilings 3045 and 3120 seconds, child lifecycle caps, outer cleanup/slack reserves, and cumulative Gate C accounting |
| Scalar references had no acceptance contract | Added exact method-matched comparisons, locked directed tolerances, and common/method/missing-evidence classifications |
| GraphDef recomputation was impossible from hashes | Embedded raw deterministic protobuf bytes and exact extraction/output-index rules |
| Incremental writes conflicted with final schema | Added complete preallocated rosters, legal state transitions, and distinct in-progress/final evaluators |
| Phase 7 fallback predicates were undefined | Added the machine-readable handoff contract below |

## Visible Round-3 Repair Record

Two focused bounded Codex substitute reviews of SHA-256
`71c9eb963e42b1bf0842d23ca589431c1446190a59948d2cfaf74725d72caae8`
returned `REVISE`. Their durable records are the round-3 runtime and evidence
review files under `docs/reviews/`. This same subplan was visibly patched:

| Finding | Repair |
| --- | --- |
| Child lifecycle left no KILL/reap interval | Set 60-second execution, five-second TERM, five-second KILL/reap; lifecycle 70 seconds; recomputed cell/Gate B caps |
| Outer grace left no scheduling/fsync slack | Set 45-second grace: at most 20 cleanup, at most 10 durable closure, 15 unallocated slack; recomputed Gate B/C hard ceilings |
| Decode caps applied too late | Added pre-parse file stat, pre-decode canonical base64/encoded-length/aggregate checks, and bounded chunked decoding |
| Function-body equality contradicted allowed shape diffs | Defined exact equality modulo recorded/tagged shape coordinates and canonical sentinel comparison |
| Axis-correlated integer constants could pass | Forbid every differing Const byte/value/shape; any exception requires a prospective path/consumer review |
| Provenance list was not transitive | Added import-only discovery and per-child actual loaded-repository-module closure reconciliation |
| Phase 7 mapping was partial/overrideable | Added first-match total mapping, selected-cell field, expansion false, and exact nonclaims |

## Visible Round-4 Repair Record

Two focused bounded Codex substitute reviews of SHA-256
`cf82af78ae4c9f9d2b608d6ed0dd102affde03f37676001cd9aebc8eef996159`
returned `REVISE`. This same subplan was visibly patched:

| Finding | Repair |
| --- | --- |
| CLI still used a 150-second cell cap | Changed both prospective Gate B/C cell-cap flags to the reviewed 160 seconds |
| Total Phase 7 table omitted scalar-passed/all-CPU-lane-local fallback | Added an ordered diagnostic-only fallback when no fair-pair exists and every nonpass CPU-XLA record is lane-local |

## Forbidden Claims And Actions

- Do not launch target trace/XLA commands until their exact implemented CLI,
  paths, and total budget replace the placeholders and receive bounded review.
- Do not launch target scalar-reference commands before the same Gate C review.
- Do not run GPU, requested threads `4/16`, HLO dumps, or the comparison ladder.
- Do not use trace size, compile survival, one warm call, or one failed cell to
  rank methods.
- Do not change Phase 4 tolerances or v4 measurement definitions after results.
- Do not edit `bayesfilter/linear/*.py` in Phase 6.
- Do not treat CPU timeout/crash as proof that XLA or the method is universally
  impossible.
- Do not modify/revert unrelated dirty work.

## Exact Next-Phase Handoff Conditions

All are conjunctive:

- Local compile/focused tests/scoped diff/read-only hashes/no-worker checks pass.
- Strict trace census is complete or an honest structural blocker result exists.
- Every launched XLA child has an immutable method record or a durable
  stage-specific failure record; every pruned cell has an exact reason.
- Result classifies common invalidity, CPU-backend failure, method failure, and
  current-cell failure separately.
- Phase 6 decision and inference-status tables state hard vetoes, viable cells,
  absence/presence of supported ranking, descriptive-only differences, and next
  evidence.
- Phase 7 is refreshed with exact trusted GPU commands/timeouts based on the
  smallest valid common-correctness/structural evidence and durable CPU-local
  failure records, and receives exact bounded `VERDICT: AGREE`.

No passing CPU-XLA cell is required for Phase 7 handoff when every CPU-XLA
failure is classified as lane-local and common validity remains established.
The strict Phase 6 handoff evaluator emits exactly these raw-derived fields:

| Field | True/allowed condition |
| --- | --- |
| `phase45_common_correctness_valid` | Bound Phase 4 diagnostic and Phase 5 strict smoke hashes/evaluators pass; read-only algorithm hashes match |
| `trace_common_valid` | Complete 36-record trace roster, every child passed extraction, every fixed-dimension/method cohort has zero rejected typed diffs, and exact input/output mapping passed |
| `target_scalar_status` | One of `passed`, `partial_missing_evidence`, `not_checked_timeout`, `failed_scalar_reference_disagreement_unlocalized`, `failed_common_or_cpu_xla_backend_unlocalized`, or `failed_method_local:<method>` under the exact scalar rules |
| `cpu_xla_common_invalidity` | True only for raw common fixture/harness/math/provenance invalidity, never merely timeout/OOM/compiler/backend/signal failure |
| `cpu_xla_lane_local_only` | True iff at least one CPU-XLA child has a valid launched terminal record, every nonpass launched terminal is a CPU-backend/current-method/current-cell class, all not-launched records have valid dependency/budget reasons, and `cpu_xla_common_invalidity=false`; it is false for an entirely unlaunched lattice |
| `selected_phase7_cell` | Exact `{dimension,parameter_count,batch_size,dtype}` or null, derived by the precedence table below |
| `phase7_scope` | Exactly `target_numerical_gate`, `diagnostic_smallest_gpu_only`, or `blocked` under the total precedence table below |
| `phase7_expansion_authorized` | Always false at Phase 6 handoff; a later Phase 7 review is required even after one diagnostic pass |
| `phase7_nonclaims` | Exact list: no target numerical claim when scalar evidence is missing; no GPU readiness; no CPU/GPU scalability; no method ranking; no HMC/posterior/default/production/scientific claim |

The strict evaluator applies this first-match precedence table; a later row
cannot override an earlier blocker:

| Condition | `phase7_scope` | Selected cell |
| --- | --- | --- |
| Phase 4/5 validity false; dependency/provenance invalid; trace invalid; CPU-XLA common invalidity true; scalar status `failed_scalar_reference_disagreement_unlocalized` or `failed_common_or_cpu_xla_backend_unlocalized` | `blocked` | null |
| Scalar status `failed_method_local:<method>` | `blocked` pending focused localization; the sibling's evidence remains method-local only | null |
| Scalar status `passed` and at least one `dimension=10,P=50,B in {1,4},float32` fair-pair CPU-XLA cell has its same-cell completed scalar comparisons | `target_numerical_gate` | Lexicographically smallest such same-cell-complete fair-pair cell |
| Scalar status `passed`, no same-cell-complete fair-pair CPU-XLA cell exists, Phase 4/5 and trace valid, and `cpu_xla_lane_local_only=true` | `diagnostic_smallest_gpu_only` | `dimension=10,P=50,B=1,float32` if its cohort is valid; otherwise lexicographically smallest valid target cohort |
| Scalar status `partial_missing_evidence` or `not_checked_timeout`, Phase 4/5 and trace valid, and `cpu_xla_lane_local_only=true` | `diagnostic_smallest_gpu_only` | `dimension=10,P=50,B=1,float32` if its cohort is valid; otherwise lexicographically smallest valid target cohort |
| Any other combination, including inconsistent raw status booleans | `blocked` | null |

For `diagnostic_smallest_gpu_only`, Phase 4 small-fixture evidence authorizes
only Phase 7's single smallest method-isolated trusted-GPU diagnostic. It does
not establish target numerical parity, GPU readiness, or permission to expand.
The artifact binds all missing scalar/CPU-XLA records into the Phase 7 entry
ledger. If the canonical smallest trace cohort is invalid, the evaluator itself
selects the lexicographically smallest valid cohort; Phase 7 cannot substitute a
different cell. If no target cohort is valid, `phase7_scope=blocked` and Phase 6
writes a blocker rather than treating the tiny Phase 5 smoke as target evidence.

## Stop Conditions

- Common harness/fixture/math/artifact invalidity remains after in-scope repair.
- A declared path overlaps irreconcilably with the other lane.
- Target commands/global budget fail to converge in review after five rounds.
- Required provenance or durable failure evidence cannot be made fail closed.
- New human authority is required.

CPU-backend or candidate-method failure alone is not a stop condition for the
program. Write the negative/lane-local result and continue to the reviewed Phase
7 repair when common correctness remains passed.

## Mandatory Phase-End Sequence

1. Run required local checks and reviewed bounded trace/XLA commands.
2. Write the Phase 6 result/close or blocker record.
3. Refresh Phase 7 from actual lane-local evidence.
4. Review and repair Phase 7 for consistency, correctness, feasibility,
   artifact coverage, and boundary safety before advancing.
