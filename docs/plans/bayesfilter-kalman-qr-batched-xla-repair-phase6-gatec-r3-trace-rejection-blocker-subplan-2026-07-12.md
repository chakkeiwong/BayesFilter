# Phase 6 Gate C R3 Trace-Rejection Blocker Diagnostic Subplan

Date: 2026-07-12

Status: `OFFLINE_DIAGNOSTIC_REVIEW_GATED_GATE_C_RUNTIME_BLOCKED`

Supervisor/executor: Codex in the current conversation.

Reviewer: fresh native Codex read-only substitute, recorded as
`codex_substitute_weaker`. The previously attempted bounded Claude review was
blocked by managed external-disclosure policy before repository content was
sent. That denial must not be retried or routed around.

Parent result:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-trace-pilot-result-2026-07-12.md`.

## Phase Objective

Using only the 36 preserved R3 GraphDefs, determine whether the six Gate B
cohort rejections arise from:

1. expected `B`/`P`-dependent shapes or integer constants whose role can be
   identified without weakening value/topology checks;
2. positional protobuf-token alignment cascades caused by repeated-field
   insertion or reordering; or
3. genuine graph topology, dependency, operation, function-body, or value
   specialization.

This is a diagnostic and attribution phase. It must not change the production
evaluator, make the gate pass, trace a new graph, execute a selected method, or
cross Gate C. The filename records the blocked handoff location; it does not
authorize Gate C runtime.

## Entry Conditions Inherited From Gate B R3

All conditions are conjunctive:

- the parent result exists and truthfully classifies the run as a valid
  structural trace rejection;
- trace ledger SHA-256 remains
  `7444fb41ef9d125990dee93a5370227c4b9ec0987ee37cb9ab7dfd362281d2b6`,
  byte count `221375005`, state `passed`, update index `72`, with 36 passed
  trace records and `trace_common_valid=false`;
- pilot ledger SHA-256 remains
  `1344f701eabfbec56e447b2cf40f3a8a4dd6cb79ef195659a50dfa4f03fb8ea2`,
  byte count `543644337`, state `complete_with_failures`, and exactly two
  `not_launched:trace_gate_not_passed` records;
- proposal SHA-256 remains
  `dd3a9495585a2f4b2995f1910da7d7b733f68467a285fa1856b5acf881f3886d`,
  attestation SHA-256 remains
  `9399be89c2263b0898c2a1c7718ca8484a81cb3fea31e8740c31750a0147e60f`,
  and authority ID remains
  `7d630ff42cc759c02d3e6618c90b97923ec9a9e8cba5b99dd41ee94e09347a33`;
- the budget state remains `closed`, the lease remains `released`, and no exact
  target worker survives;
- protected algorithm hashes remain
  `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b`,
  `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57`,
  and `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401`;
- R2 archive, import discovery, running state, and released lease remain at
  `40e6a186a28cd15d4ab3901f516854a6d84065fcb9759716108ab8e103e7834d`,
  `8ae6086bd6b8bbebd7bf236536a80cb6b8befa993a9e686801c451e8fec4c8ac`,
  `a4cc284b64d6527a7357171f4c47395a7f29f7fed7e50b15563257feae09390f`,
  and `ae711efe84056ae416d5fe2d2d40751b91afaa7f3a2e3530f095fb501a03b456`;
- no `pilot/`, `children/`, or `progress/` directory appears in the R3 root;
- both non-selected handoff pairs named below remain absent; and
- this exact subplan receives bounded read-only review and converges within at
  most five material rounds before diagnostic implementation or execution.

If any inherited identity or closure condition fails, write a blocker result
and stop. Do not repair or overwrite the frozen runtime generation.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Does stable-identity, consumer-aware GraphDef comparison preserve the Gate B rejection, or show that some/all raw positional rejections are evaluator artifacts? |
| Candidate/mechanism | A read-only offline diagnostic that aligns named graph entities, classifies axis-correlated data, and verifies topology/value sensitivity with mutations. |
| Expected failure mode | Stable keys are not unique; TensorFlow naming changes encode real topology differences; constant-consumer analysis is ambiguous; a keyed comparator hides an edge/op/value mutation; or preserved evidence cannot answer attribution. |
| Promotion criterion | No direct runtime promotion exists. The phase passes only if it produces a complete, reproducible classification with mutation controls and an exact prospective handoff. |
| Promotion veto | Missing/changed source GraphDef, lossy decoding, dropped entity/token, ambiguous duplicate key, undetected topology/value mutation, unexplained residual, or claim beyond the offline evidence. |
| Continuation veto | Frozen artifact drift/corruption, unsafe process state, inability to bind all 36 graphs, missing required diagnostics, observed in-scope concurrent write, or a new human/runtime/model/default/scientific boundary. |
| Repair trigger | Any diagnostic implementation defect, incomplete entity coverage, non-deterministic result, failed mutation control, or mismatch between raw and recomputed Gate B counts. |
| Explanatory diagnostics | Raw positional counts, stable-key counts, entity-set deltas, axis correlations, consumer classes, graph fingerprints, and representative paths. |
| Must not conclude | No Gate B pass, Gate C authorization, XLA viability, memory/performance repair, method ranking, CPU/GPU scalability, production/default/HMC/posterior/scientific validity. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Exact research question above. |
| Exact baseline | The byte-bound R3 trace ledger and its 36 embedded GraphDefs; current `compare_graphdef_cohort` output is the comparator to reproduce, not a truth label. |
| Primary pass/fail criterion | All source bindings match; current six-cohort counts reproduce; every graph/entity/token is accounted for; stable alignment and consumer-aware classifications are deterministic; all negative controls are detected. |
| Promotion veto diagnostics | Artifact/hash drift, decode mismatch, incomplete coverage, duplicate/ambiguous stable keys, missed mutation, or an unexplained residual presented as harmless. |
| Continuation veto diagnostics | Frozen evidence invalidity, unsafe process state, missing artifact/check, concurrent in-scope mutation, or required authority outside this plan. |
| Explanatory only | Number and percentage of collapsed positional differences, file/graph sizes, runtime/RSS of the offline tool, and method/dimension contrasts. |
| What will not be concluded | Even a proven evaluator false positive authorizes only a separate reviewed evaluator-repair phase with new authority; it does not retrospectively pass Gate B or authorize XLA. |
| Preserved artifact | Strict JSON diagnostic, durable check manifest/logs, and phase result. Exactly one selected next-subplan/review pair is also required only on a valid classified handoff. A stop before handoff drafting requires one blocker close record and no next subplan; review nonconvergence after drafting preserves that draft/review lineage but authorizes no handoff. |

Raw rejection totals are descriptive. They must never be interpreted as
independent defect counts or a method ranking.

## Diagnostic Design

### Source And Reproduction Layer

- Read the bounded trace JSON without writing it.
- Revalidate its exact byte count/digest, final ledger contract, all embedded
  GraphDef byte records, declared and decoded total, and source identities.
- Recompute the current `evaluate_phase6_trace_census` result and require the
  exact six accepted/rejected counts recorded in the parent result.
- Bind each decoded graph by identity ID, decoded byte count, and SHA-256 in the
  new diagnostic artifact.

Failure to reproduce the current gate is diagnostic invalidity, not evidence
that the gate has improved.

### Stable-Identity Layer

Build a separate diagnostic view without mutating any authoritative graph:

- align top-level `NodeDef` entries by unique `name`;
- align `FunctionDef` entries by unique `signature.name`;
- within each function, align `node_def` entries by unique `name` and function
  signature arguments/returns by their declared names where unique;
- retain ordered input edges, control edges, output indices, op names, device
  strings, function references, attribute key sets, and raw attribute values;
- record duplicate/missing keys as hard diagnostic failures rather than falling
  back silently to positional alignment; and
- separately record whether protobuf repeated-field order changed, because the
  parent Gate B contract required names and order to match.

The diagnostic must account for every raw token/entity. Keyed alignment may
explain a positional cascade but may not erase an insertion, deletion,
reordering, edge, op, function, device, dtype, or value difference.

### Axis And Constant Layer

For every differing shape coordinate or integer-valued `Const` payload:

- test correlation jointly across all six `(P,B)` cohort records, never a
  selected pair;
- classify exact `B`, `P`, deterministic products/shape vectors, or neither;
- decode dtype, tensor shape, `tensor_content`, and repeated scalar fields
  without float coercion;
- record every direct and transitive consumer needed to determine whether the
  constant participates only in shape/control metadata or can affect numeric
  computation; and
- mark ambiguous, mixed-use, or value-path constants as unsafe residuals.

Axis correlation and a shape-only consumer are explanatory findings, not an
accepted production normalization rule. Any future exception must be narrow,
path/consumer-specific, prospective, and separately reviewed.

### Topology Layer

For each fixed dimension/method cohort, record:

- named top-level and function entity sets, counts, order, and operation maps;
- data/control-edge maps with tensor output indices;
- function-call and function-body fingerprints;
- non-axis attribute and constant digests; and
- residual insertion/deletion/reorder/op/edge/value changes after alignment.

The final classification for each cohort is one of:
`evaluator_alignment_artifact_established`,
`expected_axis_data_requires_prospective_rule`,
`true_structural_specialization_established`,
`mixed_causes`, or `undetermined`.
The artifact must state the evidence supporting the label and may use
`mixed_causes` whenever more than one class remains.

### Required Negative Controls

Tests must build independent small GraphDefs and demonstrate that the
diagnostic:

- reduces a single inserted/reordered entity to the correct bounded entity
  delta instead of a positional cascade, while still reporting the insertion
  or order change;
- detects node deletion, op substitution, data-edge rewiring, control-edge
  rewiring, function-body change, dtype change, and device change;
- detects an input tensor output-index change, a function-call target change,
  function signature argument/return changes, `ret` and `control_ret` changes,
  and a cross-function call/dependency change;
- detects scalar/list/tensor shape-attribute changes and a raw non-shape list
  attribute change;
- detects a same-shape non-axis numeric constant mutation;
- does not classify an axis-correlated integer constant on a numeric/value path
  as harmless shape metadata, including when that numeric consumer is reached
  through a called function;
- rejects duplicate stable node/function keys and incomplete token coverage;
- distinguishes `B` from `P` only from the complete six-cell lattice; and
- is deterministic under two independent offline invocations.

These are necessary diagnostic-validity checks. Passing them does not validate
the Kalman result or production evaluator.

The test artifact must include a closed field-coverage matrix. Each promised
semantic class maps to at least one named mutation test, the expected detected
classification, and the actual result: entity insertion/deletion/order, op,
data edge and output index, control edge, function call target/body/signature,
`ret`, `control_ret`, dtype, device, shape attributes, non-shape scalar/list
attributes, `Const` tensor value, cross-function numeric consumer reachability,
duplicate keys, incomplete coverage, and full-lattice axis classification.
Every row must pass. Unlisted protobuf fields remain raw-digest residuals and
cannot be normalized by this phase.

## Required Artifacts

Allowed new writes are limited to:

- `docs/benchmarks/diagnose_kalman_qr_phase6_gateb_r3_trace_rejections_2026_07_12.py`;
- `tests/test_kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic.py`;
- `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_2026-07-12.json`;
- exact durable logs:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_py_compile_2026-07-12.txt`,
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_focused_pytest_2026-07-12.txt`,
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_run1_2026-07-12.txt`,
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_run2_2026-07-12.txt`,
  and
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_durable_run_2026-07-12.txt`;
- strict check manifest
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_check_manifest_2026-07-12.json`;
- phase result
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-trace-rejection-blocker-result-2026-07-12.md`;
- review records for this subplan under `docs/reviews/`; and
- exactly one selected next subplan and its bounded review record only on a
  valid classified handoff. A stop before handoff drafting writes only the
  blocker close record and leaves every prospective next subplan absent; review
  nonconvergence after drafting preserves the draft/review lineage but
  authorizes no handoff.

Ephemeral writes are limited to
`/tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/`: two diagnostic
outputs, `pycache/`, and `.pytest_cache/`. They are scratch artifacts only and
must not replace the durable logs, strict JSON, or manifest above.

Read-only inputs include the R3 trace/pilot ledgers, R3/R2 authority lineage,
current evaluator source/tests, protected algorithm sources, parent plans, and
the R3 work root. The diagnostic may import TensorFlow protobuf definitions
with GPU deliberately hidden. Tests may construct independent protobuf-only
synthetic `GraphDef` unit fixtures that do not import or call any Kalman model,
benchmark fixture builder, selected method, or concrete-function tracer. The
diagnostic may not build a Kalman/benchmark fixture, trace a concrete function,
invoke a selected method, initialize XLA, or enumerate/use a GPU.

## Required Checks, Tests, And Reviews

Before implementation/execution:

- bounded fresh Codex read-only review of exactly this subplan for consistency,
  correctness, feasibility, artifact coverage, and boundary safety;
- exact inherited hashes, file types, sizes, authority validation, final-ledger
  checks, budget/lease closure, namespace inventory, and no-worker scan.

After implementation:

- exact compile command:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  PYTHONPYCACHEPREFIX=/tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  docs/benchmarks/diagnose_kalman_qr_phase6_gateb_r3_trace_rejections_2026_07_12.py \
  tests/test_kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic.py \
  > docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_py_compile_2026-07-12.txt \
  2>&1
```

- exact focused GPU-hidden test command containing every negative control and
  using only the declared scratch cache:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  PYTHONPYCACHEPREFIX=/tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  -o cache_dir=/tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/.pytest_cache \
  tests/test_kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic.py \
  > docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_focused_pytest_2026-07-12.txt \
  2>&1
```

- before any diagnostic invocation, a closed no-target static/call-shape scan
  confirming the new diagnostic entrypoint cannot import or call target
  benchmark builders, fixture builders, selected methods, concrete-function
  tracers, XLA paths, GPU enumeration, or the Gate C runner;
- two independent GPU-hidden offline invocations against the same immutable
  trace:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  PYTHONPYCACHEPREFIX=/tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/diagnose_kalman_qr_phase6_gateb_r3_trace_rejections_2026_07_12.py \
  --trace-input docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_census_2026-07-12.json \
  --output-json /tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/run1.json \
  > docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_run1_2026-07-12.txt \
  2>&1

env CUDA_VISIBLE_DEVICES=-1 \
  PYTHONPYCACHEPREFIX=/tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/diagnose_kalman_qr_phase6_gateb_r3_trace_rejections_2026_07_12.py \
  --trace-input docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_census_2026-07-12.json \
  --output-json /tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/run2.json \
  > docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_run2_2026-07-12.txt \
  2>&1
```

  Their `diagnostic_payload_sha256` fields must match. The hashed diagnostic
  payload excludes exactly `run_manifest.started_utc`,
  `run_manifest.finished_utc`, `run_manifest.wall_seconds`, and
  `run_manifest.output_path`; no evidence/classification/count field is
  excluded. After equality passes, run the exact durable invocation:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  PYTHONPYCACHEPREFIX=/tmp/kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic/pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/diagnose_kalman_qr_phase6_gateb_r3_trace_rejections_2026_07_12.py \
  --trace-input docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_census_2026-07-12.json \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_2026-07-12.json \
  > docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_durable_run_2026-07-12.txt \
  2>&1
```

  This third invocation must produce the same diagnostic payload digest;
- strict parse/schema/content validation of the durable diagnostic artifact;
- exact reproduction of all six current cohort count pairs;
- complete graph/entity/token coverage and zero unexplained dropped data;
- recheck all inherited hashes, final-ledger bindings, live authority,
  budget/lease closure, exact no-worker state, protected/R2 hashes, and R3
  namespace inventory;
- `git diff --check` where applicable, per-untracked-file
  `git diff --no-index --check /dev/null <path>`, and closed trailing-whitespace
  scans; and
- strict check manifest binding every literal command above plus every static,
  hash, ledger, authority, closure, process, namespace, and whitespace command;
  for each command it records argv or exact shell text, working directory,
  environment overrides, start/finish UTC, wall seconds, exit status, durable
  stdout/stderr log path where applicable, log byte count/SHA-256, and the
  exact pass predicate. It also binds all three output paths, byte counts,
  SHA-256 values, and `diagnostic_payload_sha256` values;
- phase result with a run manifest containing git commit, exact commands and
  manifest SHA-256, Python/conda environment, CPU/GPU visibility and deliberate
  GPU-hidden status, XLA/TF32 status, data/fixture version (`N/A` for preserved
  GraphDefs), random seeds (`N/A`; deterministic diagnostic), wall time,
  input/output artifact paths and hashes, subplan path/hash, result path/hash
  exemption, and trust/claim boundary. The result records its own path but must
  state that a self-hash cannot be embedded without recursion; the frozen result
  SHA-256 is bound instead by a detached final review/close record that is not
  recursively bound back into the result. The result also contains a decision
  table, inference-status table, candidate-versus-direction distinction, and
  post-run red-team note.

At phase end, if and only if the diagnostic is valid and selects a classified
handoff branch, draft exactly one branch-specific next subplan, review it with a
fresh bounded Codex substitute, visibly repair material findings, and stop
after five rounds for the same unresolved blocker. If a diagnostic-invalidity,
undetermined, or other stop branch fires before handoff drafting, write the
blocker close record and stop with every prospective next-subplan path absent;
do not invent a handoff merely to satisfy the phase-end procedure. If a drafted
next subplan later fails to converge within five rounds, preserve that draft and
all review records, write a blocker close record, and state that no handoff is
authorized; do not delete or rename the failed review lineage.

## Forbidden Claims And Actions

- Do not edit or regenerate either R3 ledger, its 108 trace work files, proposal,
  attestation, review, audit, budget, lease, R2 lineage, or protected algorithm
  source.
- Do not edit `compare_graphdef_cohort`, `evaluate_phase6_trace_census`, any
  production/runtime evaluator, benchmark builder, Kalman implementation, or
  existing Phase 6 test in this diagnostic phase.
- Do not create Kalman/benchmark fixtures, trace new concrete functions,
  execute a selected method, compile/run XLA, run Gate C, use/enumerate GPU, run
  Phase 7, or launch memory/performance/scaling benchmarks. Protobuf-only
  synthetic GraphDefs are permitted solely in the named unit test as negative
  controls and carry no target evidence.
- Do not delete, compact, deduplicate, rename, or rewrite large artifacts.
- Do not add a normalization rule because it makes `trace_common_valid=true`.
- Do not use graph size, rejection count, or a collapsed keyed count as a pass
  criterion or method ranking.
- Do not call an axis-correlated constant harmless without proving its consumer
  boundary and passing the value-path negative control.
- Do not treat a name-aligned match as sufficient if order, edge, op, function,
  device, dtype, or raw value differs.
- Do not retrospectively change R3 authority or call a prospective diagnostic
  an authorized Gate B rerun.
- Claude or Codex reviewers are read-only and cannot authorize crossing human,
  runtime, model-file, funding, product/default, release, or scientific-claim
  boundaries.

Both non-selected branch pairs must remain absent during this phase:

- `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-scalar-remaining-subplan-2026-07-12.md` and its review; and
- `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r4-harness-repair-subplan-2026-07-12.md` and its review.

## Exact Next-Phase Handoff Conditions

Exactly one branch may be selected after the diagnostic result is frozen:

| Diagnostic result | Required handoff |
| --- | --- |
| False-positive evaluator defect established with all negative controls passing and no true/ambiguous residual | Draft/review a prospective R4 trace-evaluator repair subplan. It must require narrow rules, negative mutation tests, a fresh proposal/review/attestation/authority generation, and no reuse of R3 runtime authority. |
| Expected axis data is established, but accepting it would change the reviewed Gate B normalization semantics | Draft/review a prospective gate-semantics proposal subplan. It must enumerate exact path/consumer rules and adversarial controls and must request human approval before any relaxation of the originally reviewed no-`Const` rule. Until approval and a separate repair converge, Gate B remains rejected. |
| Genuine source graph specialization established | Draft/review a graph-structure localization/repair subplan. It may inspect algorithm/build paths but cannot authorize target execution until its own evidence and review gates pass. |
| Mixed causes | Draft/review the smallest next diagnostic or repair subplan that preserves the true residual and does not generalize from the false-positive portion. |
| Undetermined, invalid diagnostic, frozen evidence drift, or failed negative control | Write a blocker result and stop for human direction; no next subplan or runtime subplan is authorized. |

All valid classified handoffs additionally require:

- the diagnostic JSON and result are strict, durable, and hash-bound;
- every required local check passes;
- frozen R3/R2/protected bytes and closure state still match;
- the selected next subplan states objective, inherited entry conditions,
  artifacts, checks/reviews, evidence contract, forbidden actions, exact
  handoff, and stops; and
- its final bounded review ends `VERDICT: AGREE` within five material rounds.

Exact successful handoff state:
`OFFLINE_TRACE_REJECTION_CLASSIFIED_NEXT_SUBPLAN_REVIEWED_RUNTIME_STILL_BLOCKED`.

## Stop Conditions

- This subplan fails to converge within five material review rounds for the same
  blocker: no diagnostic execution is authorized. A selected next subplan that
  fails to converge within five material rounds is preserved with its review
  lineage, but no handoff or downstream execution is authorized.
- Any frozen R3/R2/protected byte or closure condition drifts.
- A required GraphDef cannot be decoded, bound, or completely accounted for.
- Stable keys are duplicate/ambiguous and no non-lossy comparison can be made.
- A negative mutation is missed, deterministic reruns disagree, or current Gate
  B counts cannot be reproduced.
- The evidence remains mixed or underdetermined and the next smallest safe
  diagnostic cannot be specified without human choice.
- A target worker appears, a process cannot be safely attributed, or another
  lane changes an in-scope read-only input during the phase.
- Continuing would require target/XLA/GPU execution, an evaluator/algorithm
  edit, package/network/model-file/funding/default/product/release change, or a
  scientific claim outside this plan.

Candidate rejection is not research-direction rejection. A valid finding that
the current evaluator is wrong triggers a separate repair; a valid finding of
real specialization triggers localization. Neither finding alone invalidates
the Kalman mathematics, repaired harness, or broader program.

## Skeptical Pre-Execution Audit

Status: `PASS_FOR_SUBPLAN_REVIEW_ONLY_OFFLINE_EXECUTION_PENDING_AGREEMENT`.

| Audit risk | Assessment |
| --- | --- |
| Wrong baseline | Controlled. The exact immutable R3 GraphDefs are the baseline; historical graph summaries and current evaluator output are comparators, not ground truth. |
| Proxy promoted to criterion | Controlled. Rejection counts, graph sizes, and keyed-count reductions are explanatory only. Complete coverage and mutation sensitivity are the diagnostic gate. |
| Missing stop conditions | Controlled. Hash drift, incomplete coverage, ambiguous keys, missed mutations, nondeterminism, concurrent writes, and boundary expansion are explicit stops. |
| Unfair comparison | Controlled. Every classification is joint within each exact six-cell fixed-dimension/method cohort; no selected pair or method is privileged. |
| Hidden assumptions | Exposed. Names may not be unique or semantically stable, order may remain contract-relevant, axis correlation may not imply shape-only use, and consumer reachability may be ambiguous. Each becomes a checked field or hard residual. |
| Stale context | Controlled. The phase revalidates ledger bytes, live authority bindings, source hashes, budget/lease closure, namespace, and processes before and after work. |
| Environment mismatch | Controlled. TensorFlow is used only for protobuf decoding with GPU hidden. No result is described as repository-default GPU/XLA evidence. |
| Artifact fitness | Controlled. Preserved raw GraphDefs plus entity/consumer/mutation evidence directly answer evaluator-versus-structure attribution. Timing or memory data would not. |
| Pass while misleading | Controlled. A smaller stable-key diff cannot pass the phase unless order and every entity/edge/op/value remain visible and adversarial mutations are detected. |
| Fail for tooling rather than graph structure | Controlled. Decode, uniqueness, coverage, determinism, and independent fixture mutations distinguish diagnostic invalidity from a graph finding. |

The audit passes only for obtaining review of this subplan. Offline execution
may begin after exact `VERDICT: AGREE` and refreshed entry checks. Runtime
execution remains forbidden regardless of the review verdict.
