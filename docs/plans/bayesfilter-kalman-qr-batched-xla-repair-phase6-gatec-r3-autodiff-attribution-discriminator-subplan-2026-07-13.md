# Phase 6 Gate C R3 Autodiff Attribution Discriminator Subplan

Date: 2026-07-13

Status: `REVIEW_CONVERGED_OFFLINE_IMPLEMENTATION_AUTHORIZED_RUNTIME_BLOCKED`

Supervisor/executor: Codex in the current conversation.

Reviewer: fresh native Codex read-only substitute with
`codex_substitute_weaker` provenance. Claude remains unavailable because the
managed external-disclosure gate rejected the call before a liveness probe; it
must not be retried in this phase.

Parent result:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-result-2026-07-13.md`.

## Phase Objective

Use only the immutable Gate B R3 GraphDef bytes, the reviewed 904-target
localization, frozen local source text, and exact installed TensorFlow source
text to determine which preserved graph region owns each atomic graph
occurrence. The parent contains 12,316 target-observation records: 6,716
top-node, 5,000 function-body, 144 function, 24 graph-order, and 432
integer-constant observations. The 432 integer-constant observations contain
2,386 nested graph occurrences; replacing those outer containers with their
nested entries yields exactly 14,270 atomic graph occurrences to classify.
The discriminator tests whether the observed batch-dependent autodiff structure
can be separated into:

- local pre-while model-tensor construction reachable from the exact
  `parameters_batch` input;
- the forward Kalman `While` call and its exact cond/body functions;
- the value-output projection from the forward `While`;
- top-level reverse-VJP setup upstream of the generated reverse `While`;
- the generated reverse-while call and its exact cond/body functions;
- top-level post-reverse VJP propagation or aggregation;
- score-path propagation not structurally downstream of the reverse `While`;
- constant/shape materialization with no unique generating operation;
- a cross-region composite such as graph order; or
- an unresolved or invalid region.

Region ownership is a graph-structural statement, not a root-cause statement.
For every occurrence, preserve the complete forward/backward witness, exact
function-call/capture binding, and any populated `NodeDef.experimental_debug_info`
or `GraphDef.debug_info`. Empty debug fields are a valid observed outcome but
must be counted across all 18 autodiff GraphDefs rather than assumed from a
probe. Names may corroborate a structural result but cannot determine it.

The phase may nominate exactly one smallest counterfactual control point only
when graph structure and source anchors uniquely bound that control point.
It cannot establish avoidability, inherence, a TensorFlow defect, numerical
equivalence, memory improvement, or performance improvement. The analytical
constant lane remains unresolved. No source, evaluator, fixture, trace, or
authority byte is edited. Bounded in-memory protobuf test graphs are the sole
fixture exception and are never serialized or used as evidence.

## Entry Conditions Inherited From The Previous Phase

All entry conditions are conjunctive. They are rechecked before implementation,
before loading phase code, and after the durable evidence run.

### Reviewed parent lineage

| Artifact | Required SHA-256/state |
| --- | --- |
| Parent result path above | `0dfaccf6ecb3150290323d6efdd9248b4d6ceefd3c2520d5f1ca8086901ac9a4`; current reviewed result |
| `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-result-review-final-2026-07-13.md` | `fc5d4a16c256a259037880f8309d5e82dd9b1d233cda809583a7d80e3fcb1774`; exact `VERDICT: AGREE` on the parent hash |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_structure_localization_2026-07-13.json` | `ee2903381039f7cf15a4ec5112304232ae138eebacef8e0858da1fda5f7452c1`; 68,112,660 bytes; canonical payload `f29bea2cb4f26e6e26f1606149652eb98f4145099a3568e7874d67824f166e1c` |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_structure_localization_2026-07-13_check_manifest.json` | `8aa6f6079a38f0e66e90a8b5964f25f146b5fdeb94585510d7d26133d7028bbf`; every check true |
| `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-subplan-2026-07-13.md` | `88db6519ca3d1a668ef9565506b539c1bd4cd672f000424c35a4be6d5581a949` |
| `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-subplan-review-final-2026-07-13.md` | `ab8bc7613ec1b547cba58ccdb3419cf46ec37b6af2be94b267e46b55e331d2eb` |

The parent localization must remain
`passed_complete_causally_ambiguous`: exactly 904 unique targets, 12,316
occurrences, zero `mapped_exact`, 904 `enumerated_causally_ambiguous`, and zero
`missing_or_incomplete`. Its target-kind counts must remain 295 top-level
nodes, 444 function-body nodes, 9 functions, 12 graph-order targets, and 144
normalized integer-constant targets. Its decision must remain
`next_branch=autodiff_attribution_discriminator`, Gate B `still_rejected`,
Gate C `blocked`, runtime unauthorized, and analytical lane unresolved.

The superseded Round 1 parent-result review does not authorize this phase.

### Frozen decoder, corpus, and authority lineage

| Artifact | Required SHA-256/state |
| --- | --- |
| `docs/benchmarks/localize_kalman_qr_phase6_gateb_r3_autodiff_structure_2026_07_13.py` | `f743479564e54bb1dfa9651e724964c863ecbd7f9d20498e68b64a24b0da4ab9` |
| `docs/benchmarks/run_guarded_kalman_qr_phase6_gateb_r3_autodiff_structure_2026_07_13.py` | `5d989656c2923f473c2a00810c082e9e28bc5529cf617c8ae9fa608108476e27` |
| `tests/test_kalman_qr_phase6_gateb_r3_autodiff_structure_localization.py` | `d0bda71ce885d2edf27d5b9306fc2e1fa2e6f55bdfdf4b24f3405203435b0b1c` |
| Parent localization authorized snapshot | `64b677d61e0f76c581e7975990f8f9941ebb2919de4556db29728061e4099e7f` |
| Final R3 trace-rejection diagnostic | `637273af37ed2606b9bd0bc4868a1719a65ad17d89d94ab018e5678082fb25ff`; canonical payload `30a2753246d4c86a6952268fad5a49d8e77991084f4100a45d1eca051c710cd7` |
| Final diagnostic check manifest | `fcc5ad1ea9cf6f06ce4ae0dc83e0da00d2ce08b9bfb805f3220748cf9bc1e54f` |
| R3 trace census | `7444fb41ef9d125990dee93a5370227c4b9ec0987ee37cb9ab7dfd362281d2b6` |
| R3 CPU-XLA pilot | `1344f701eabfbec56e447b2cf40f3a8a4dd6cb79ef195659a50dfa4f03fb8ea2`; exactly two `not_launched:trace_gate_not_passed` records and zero XLA calls |
| R3 budget | `dd3a9495585a2f4b2995f1910da7d7b733f68467a285fa1856b5acf881f3886d` |
| R3 budget attestation | `9399be89c2263b0898c2a1c7718ca8484a81cb3fea31e8740c31750a0147e60f` |

`/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/` must still contain exactly
`import_discovery.json`, `budget_state/`, and 108 files under `trace/`.
`pilot/`, `children/`, and `progress/` remain absent. The recorded Gate B
budget state remains `closed`, its lease remains `released`, no target worker
survives, and all forbidden authority branch pairs remain absent. The phase
reads the existing 36 JSON trace records containing GraphDef bytes; it never
writes the R3 root.

The same private `google.protobuf` descriptor decoder and complete 12-file
installed-schema ledger used by the parent localizer are reused read-only.
Every descriptor path, byte count, hash, dependency order, protobuf version
`6.33.5`, decoded raw GraphDef hash, and diagnostic binding must reproduce the
parent ledger exactly. Loading any `tensorflow.*_pb2` module or importing
TensorFlow is forbidden.

### Frozen local and installed source provenance

The complete parent local-source ledger remains binding. At minimum, these
mechanism anchors and their containing-file hashes must match before and after:

| Path | Required SHA-256 | Required anchor |
| --- | --- | --- |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `baf62b85f885073d0b72b5c13af0463ac5566f2429c16d5c98a542aa24c8eec9` | `_broadcast_vector_basis`, `_broadcast_matrix_basis`, and `_batched_model_tensors`, lines 712-784; one-VJP wrapper, lines 1893-1945 |
| `bayesfilter/linear/kalman_qr_tf.py` | `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b` | batched static while route, lines 621-763 |
| `tensorflow/python/eager/backprop.py` | `c9a461d06085be50a2235e23e6c32a4649805fa2b5fb43d50e7fb421f92eed78` | exact VJP/zero-gradient construction anchors from the parent ledger |
| `tensorflow/python/ops/gradients_util.py` | `1b4cf14a574b45f708ec4fba67bd450336a6b8e63f494ddf791fc5de5d981e98` | exact graph-gradient and zero-materialization anchors from the parent ledger |
| `tensorflow/python/ops/while_v2.py` | `756344a1c87911ca4a0678bea1388d070c7218afb1ec34a3f07c03473804719a` | `_WhileGrad` lines 322-437, `_preprocess_grad`/`_zeros_like` lines 523-580, `_grad_fn` lines 694-730, and capture/rewrite optimization lines 950-1009 |

TensorFlow provenance remains distribution `tensorflow==2.20.0`, Python
3.13.13, with the exact `METADATA` and `WHEEL` hashes inherited from the
parent. A source anchor can explain a structurally matched operation; it cannot
turn name similarity or reachability into proof of origin, avoidability,
inherence, or defect.

The exact subplan must receive fresh read-only review agreement within five
material rounds. Any entry mismatch produces a blocker result and stops without
loading implementation code or changing frozen inputs.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can complete graph-internal provenance separate all 14,270 atomic graph occurrences represented by the 12,316 reviewed target-observation records among local pre-while construction, forward while/value projection, reverse-VJP setup, generated reverse while, post-reverse VJP, other score path, constant/shape, composite, and unresolved regions? |
| Candidate/mechanism | Deterministic offline GraphDef decoding; exact call attrs; function ownership; call-input/function-argument and function-return/call-output bindings; complete forward/backward reachability; parameter/value/score boundary anchors; debug-info census; and exact installed/local source anchors. |
| Expected failure | Debug metadata is empty; call or argument binding is lossy; a target occurrence belongs to multiple regions; terminal constants have no generator witness; graph-order targets are not reducible to one region; or source/target/corpus bytes drift. |
| Primary criterion | All 12,316 parent observation records reproduce exactly. Every one of the derived 14,270 atomic graph occurrences is represented exactly once and receives one allowed region state plus a complete witness or an explicit unresolved/composite witness; all 904 target aggregates are derived deterministically from those atomic occurrences; zero observation or atomic occurrence is missing or duplicated. |
| Promotion veto | Missing/duplicate occurrence; target-key or observation drift; lossy edge/call/capture mapping; a classification determined by names; stale/false anchor; unsupported causal language; failed mutation; or nondeterministic payload. |
| Continuation veto | Frozen-input drift, unsafe process state, in-scope concurrent write, failed pre-load guard, unexpected write, or need for TensorFlow/trace/XLA/GPU/runtime/source-edit authority. |
| Repair trigger | Exactly one bounded local control point is structurally upstream of every target in a declared repair cohort, no competing control point remains for that cohort, and the result only nominates a later semantics-preserving counterfactual test. |
| Explanatory only | Region counts, path lengths, op/name families, debug-field population counts, source proximity, dominance ratios, GraphDef size, and offline elapsed time. |
| Must not conclude | No root cause, avoidability/inherence, TensorFlow bug, evaluator exception, numerical validity, Gate B pass, Gate C authority, XLA/GPU viability, memory/performance repair, ranking, or readiness claim. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Exact research question above. |
| Baseline/comparator | The current reviewed 904-target localization and its exact 12,316 observation records, including the exact 2,386 nested integer-constant graph occurrences, rebound to the same 18 immutable autodiff GraphDefs. Analytical GraphDefs are provenance controls only and remain unresolved. |
| Primary pass criterion | A total, mutually exclusive partition of all 14,270 atomic graph occurrences with complete structural witnesses; exact 904-target, 12,316-observation, and nested-occurrence parity; deterministic canonical digest across two scratch runs and one durable run; all negative controls pass. Unresolved/composite is an allowed honest classification, not a promotion. |
| Promotion vetoes | Any primary-criterion failure, false unique control point, debug/name inference represented as provenance, or missing exact source/framework anchor for a source-language claim. |
| Continuation vetoes | Entry/closure drift, guard failure, unauthorized write/import/call, active target process, or requirement for new runtime evidence. |
| Explanatory diagnostics | Per-region counts, path lengths, source-anchor coverage, debug-info census, op/name summaries, and ratios. |
| Not concluded | Structural ownership does not establish construction origin, semantics, counterfactual effect, compile-memory effect, runtime effect, framework necessity, or a defect. |
| Preserved result | Strict discriminator JSON, exact logs/check manifest, phase result, detached agreeing result review, then exactly one reviewed next subplan or a blocker-only stop. |

## Required Artifacts And Write Set

Before implementation, create a new authorized-state snapshot at:

`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_attribution_discriminator_authorized_snapshot_2026-07-13.json`.

It records existence, size, SHA-256, and Git status for every phase-owned path
and finite stem, the parent artifacts and protected inputs, the exact R3
inventory, and the preexisting unrelated repository status list/digest. Other
lane changes are evidence-only and must not be modified, staged, cleaned, or
represented as phase work.

Only these exact new repository paths or finite stems may be written:

- `docs/benchmarks/discriminate_kalman_qr_phase6_gateb_r3_autodiff_attribution_2026_07_13.py`;
- `docs/benchmarks/run_guarded_kalman_qr_phase6_gateb_r3_autodiff_attribution_2026_07_13.py`;
- `tests/test_kalman_qr_phase6_gateb_r3_autodiff_attribution_discriminator.py`;
- the authorized snapshot path above;
- `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_attribution_discriminator_2026-07-13.json`;
- the same benchmark stem ending `_static_scan.txt`, `_py_compile.txt`,
  `_focused_pytest.txt`, `_run1.txt`, `_run2.txt`, `_durable_run.txt`, or
  `_check_manifest.json`;
- `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-attribution-discriminator-result-2026-07-13.md`;
- paths under `docs/reviews/` with exact filename prefix
  `bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-attribution-discriminator-subplan-review-`
  and suffix in `{round1,round2,round3,round4,round5,final}-2026-07-13.md`;
- paths under `docs/reviews/` with exact filename prefix
  `bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-attribution-discriminator-result-review-`
  and suffix in `{round1,round2,round3,round4,round5,final}-2026-07-13.md`;
- only after detached result agreement, exactly one of these next-plan paths:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-source-counterfactual-repair-subplan-2026-07-13.md`,
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-framework-proof-subplan-2026-07-13.md`, or
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-minimal-counterfactual-trace-subplan-2026-07-13.md`;
- for the selected next-plan path only, review records under `docs/reviews/`
  with the exact selected basename prefix, then `-review-`, and suffix in
  `{round1,round2,round3,round4,round5,final}-2026-07-13.md`; or
- the discriminator result path as a blocker record, with no next subplan.

No existing phase/source/test/evidence file may be edited. The only new scratch
paths are:

- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_attribution_discriminator/run1.json`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_attribution_discriminator/run2.json`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_attribution_discriminator/py_compile/discriminator.pyc`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_attribution_discriminator/py_compile/guard.pyc`; and
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_attribution_discriminator/py_compile/test.pyc`.

The supervisor may create only the empty scratch root and its empty
`py_compile/` directory before checks. A preexisting or nonempty root is an
entry veto and is not removed. Every invocation uses
`PYTHONDONTWRITEBYTECODE=1`; pytest caches, capture files, `__pycache__`, and
filesystem temp fixtures are forbidden. The guard and closure inventory reject
every undeclared scratch/repository write. No `/tmp` convenience or status file
is permitted.

## Required Classifier And Witness Contract

The discriminator must re-decode all 36 bound GraphDefs but classify only the
18 autodiff graphs. It must independently reconstruct the parent observation
keys and prove exact equality to the reviewed 12,316-observation multiset before
classification. It then flattens each non-integer observation to one atomic
graph occurrence and each integer-constant observation to its exact nested
`occurrences` entries, producing exactly 14,270 atomic graph occurrences.
Atomic keys bind target key, outer observation canonical digest, and, when
present, nested occurrence canonical digest; positional index alone is not an
identity. Reusing parent decoder helpers is permitted only through one exact
post-guard load of the hashed parent localizer; parent payload conclusions may
not replace reconstruction.

For every autodiff graph, derive boundaries from bytes and trace evidence:

- exact structured user input `parameters_batch` and exact concrete value/score
  output node names from the trace record, never from a hard-coded suffix;
- every top-level `While`/`StatelessWhile` call and its `cond`/`body` function
  attrs, with unique function definitions, ordered call inputs, function input
  args, return map, ordered outputs, and consumers;
- forward-while versus reverse-while identity from graph structure. The forward
  call is the unique while call in the inclusive ancestor set of the concrete
  value output. A reverse candidate is a different while call in the inclusive
  ancestor set of the concrete score output that has a complete call/function
  binding and consumes at least one exact forward-call output or a top-level
  descendant of such an output. This is a saved-forward/capture binding; an
  edge from the concrete value-output wrapper is neither required nor assumed.
  The reverse call is identified only when exactly one candidate remains.
  Empty or multiple candidate sets produce the honest
  `structural_boundary_ambiguous` state for affected occurrences with the full
  candidate ledger; they do not permit name-based selection. Names are
  negative-control perturbation targets, not identity evidence;
- complete top-level forward and backward reachability, including data and
  control edges, and explicit treatment of constants that are only linked as
  producers;
- exact function ownership and cross-boundary argument/return witnesses for
  every function/function-body occurrence; and
- exact debug-info census for every node and graph, preserving populated values
  losslessly and recording empty coverage without treating emptiness as a
  classifier.

Let `A(x)` and `D(x)` be inclusive data-and-control ancestor and descendant sets
in the top-level graph; `P`, `V`, and `S` are the exact parameter, value-output,
and score-output nodes from trace metadata; and `F` and `R` are the uniquely
resolved forward and reverse calls above. Every function occurrence is first
bound to its exact caller through a concrete `cond` or `body` attr; an uncalled
or multiply bound function is not assigned by normalized name or body
similarity.

Each atomic graph occurrence receives exactly one state from this fixed
first-match order. Later predicates are evaluated after subtracting all earlier
members:

- `cross_region_composite`: a graph-order occurrence, represented by both exact
  ordered graph/function ledgers and their digests;
- `forward_while_call_or_function`: the exact `F` call or a function/function
  body bound by `F.cond` or `F.body`;
- `reverse_while_call_or_function`: the exact `R` call or a function/function
  body bound by `R.cond` or `R.body`;
- `local_pre_forward_while`: a top-level node in
  `(D(P) intersect A(F)) - {F}`;
- `forward_while_setup_not_parameter_descendant`: a top-level node in
  `A(F) - D(P) - {F}`;
- `forward_value_projection`: a top-level node in
  `(D(F) intersect A(V)) - {F}`;
- `reverse_vjp_setup`: a top-level node in `A(R) - {R}` after subtraction of
  all earlier states;
- `post_reverse_vjp`: a top-level node in
  `(D(R) intersect A(S)) - {R}` after earlier-state subtraction;
- `score_path_not_reverse_descendant`: a top-level node in `A(S)` after
  earlier-state subtraction;
- `constant_or_shape_origin_unresolved`: an otherwise unassigned
  integer-constant atomic occurrence, or an otherwise unassigned zero-input
  literal tensor/shape producer. The witness must preserve the parent constant
  record, literal attributes, and all consumers; op or name alone is
  insufficient;
- `structural_boundary_ambiguous`: a complete occurrence not matched above,
  including an uncalled/multiply bound function or an occurrence affected by a
  nonunique `F`/`R` candidate set. Its witness contains every candidate and the
  failed uniqueness predicate; or
- `unresolved_invalid`: a malformed, missing, duplicate, lossy, or internally
  inconsistent occurrence. This is a continuation veto, not an honest
  ambiguity state.

If `F` is not unique, every occurrence whose predicate depends on `F` or `R`
is `structural_boundary_ambiguous`; if `F` is unique but `R` is not, only
forward-region predicates are evaluated and every reverse-dependent residual
is `structural_boundary_ambiguous`. The exact frozen corpus is still expected
to resolve both boundaries, but the rule does not force that outcome.

Every occurrence has a common witness containing atomic key, target key, outer
observation digest, optional nested-occurrence digest, graph SHA-256, concrete
scope, entity digest, trace input/output anchors, `F`/`R` candidate ledgers,
and the canonical predicate-result vector. State-specific witnesses add:

- the complete induced reachability slice and its canonical digest for every
  reachability predicate; for readability only, also preserve the shortest
  path selected by `(edge_count, canonical node-key sequence)` so traversal
  order cannot change the witness;
- caller node, attr key, function input/output signature, ordered argument/
  return binding, and function digest for while-function states;
- both order ledgers for `cross_region_composite`;
- literal tensor/shape fields and all consumer paths for
  `constant_or_shape_origin_unresolved`; and
- the complete candidate set and exact failed predicate for
  `structural_boundary_ambiguous`.

Every non-invalid state requires the deterministic witness schema above. A
target-level aggregate is `uniform_<state>` only when every atomic occurrence
has that same state; otherwise it is `mixed_regions`. Graph-order targets are expected to be
`cross_region_composite` unless a lossless region-specific order witness exists.
No target is discarded because it is metadata, a constant, an order target, or
an unchanged-name changed-body occurrence.

The strict result validator must reject overlap, gaps, fabricated call attrs,
removed edges, shuffled call arguments/returns, false output anchors, renamed
structural identities, changed function ownership, changed target/observation
keys, false debug values, stale source hashes, unsupported unique-control-point
nomination, and causal or performance language outside the explicit nonclaims.

## Required Checks, Tests, And Reviews

### Before implementation

- preserve every review round visibly and obtain exact `VERDICT: AGREE` on this
  subplan within five material rounds;
- rerun every parent/artifact/hash/authority/R3-inventory/no-worker check;
- validate the new authorized-state snapshot and the exact empty scratch root;
- record the skeptical audit below after reviewer repairs; and
- confirm the phase code can answer the structural question using only the
  standard library, `google.protobuf`, and one guarded exact load of the frozen
  parent localizer.

### Before any phase-code import or test

Run and preserve a CPython AST boundary scan over exactly the discriminator,
guard, and focused test. It must reject:

- TensorFlow or BayesFilter algorithm/benchmark imports; direct import of the
  parent localizer; any non-whitelisted dynamic import;
- target builders/methods, `tf.function`, `GradientTape`, concrete-function or
  trace APIs, XLA/JIT APIs, device/GPU enumeration, subprocess/process/shell,
  network, package, model, or external-review calls;
- `eval`, `exec`, writes outside the exact output argument and authorized test
  mutation objects, or any test using a filesystem temp fixture; and
- classifiers based on string fragments such as `gradient_tape`, `_grad_`,
  `rewritten`, `while`, `Shape`, or `zeros` rather than graph witnesses.

Only after the scan passes may explicit bytecode compilation run. It calls
`py_compile.compile(..., cfile=<exact path>, doraise=True)` separately for the
three files and creates only the three authorized `.pyc` files. Implicit
`__pycache__` output is forbidden.

Every focused-test and evidence invocation enters through the new pre-load
guard. The guard installs import, selected-call, device, subprocess, network,
and write boundaries before loading the phase code. It permits exactly one
post-guard dynamic load of the exact hashed parent localizer and the exact
focused test. It asserts TensorFlow is absent before and after.

The focused tests must include at least:

- exact 904-target, 12,316-observation, 2,386 nested integer occurrence, and
  14,270 atomic occurrence reconstruction with duplicate/gap rejection;
- populated and empty debug-info lossless census tests;
- data/control reachability and constant-producer treatment;
- forward/reverse structural identity with all semantic names replaced,
  including a reverse call fed by saved forward outputs but not by the concrete
  value-output wrapper, plus zero/multiple reverse-candidate ambiguity;
- call-input/function-argument and function-return/call-output binding;
- all first-match states, precedence overlaps, honest structural ambiguity,
  and invalid-state fixtures;
- target uniform/mixed aggregation;
- mutation rejection for call attrs, edge direction, argument/return order,
  output anchors, function ownership, target key, debug value, source hash, and
  unique-control-point nomination;
- canonical-digest determinism with declared run metadata excluded; and
- guard rejection of TensorFlow import, selected method/build/trace surfaces,
  subprocess/network/device calls, and undeclared writes.

### Offline evidence and closure

- run two independent GPU-hidden scratch invocations and one durable invocation
  through the pre-load guard;
- require identical canonical payload digests and identical occurrence/target
  partitions, excluding only declared timestamps, wall time, and output path;
- require all 36 raw GraphDef hashes and all 18 autodiff boundary records to
  match the trace/localization ledgers;
- require exact totality over 904 targets, 12,316 parent observations, 2,386
  nested integer-constant graph occurrences, and 14,270 atomic graph
  occurrences; zero `unresolved_invalid`; allowed honest unresolved/composite
  states do not count as invalid;
- rerun all entry and protected-source checks and compare only the authorized
  write set against the snapshot; inventory exact finite stems and scratch
  paths without materializing convenience files;
- write the result with a run manifest, decision table, inference-status table,
  candidate-versus-direction distinction, and post-run red team;
- freeze the result bytes and obtain an independent detached fresh-Codex review
  agreeing on target/occurrence coverage, structural witnesses, source anchors,
  causal boundaries, hashes, and write/runtime boundaries; and
- only after result agreement, draft and review exactly one next subplan selected
  by the handoff table.

## Forbidden Claims And Actions

- Do not edit algorithm, benchmark, evaluator, existing test, parent tool,
  frozen evidence, trace, R3/R2 authority, budget, lease, or work-root bytes.
- Do not create a target/runtime or filesystem fixture, invoke a target builder
  or selected method, import or initialize TensorFlow, acquire a new concrete
  function/trace, compile/run XLA,
  enumerate/use GPU, run Gate C/Phase 7, benchmark, install packages, access the
  network, or launch a subprocess from phase code.
- Bounded in-memory protobuf graphs constructed inside the exact authorized
  focused test are permitted only for classifier, mutation, and guard tests.
  They are not benchmark/target fixtures, may not be serialized to disk, and
  cannot support an evidence or promotion claim.
- Do not classify by names, op type alone, source proximity, elapsed time,
  frequency, dominance, GraphDef size, or empty debug metadata.
- Do not call any construction avoidable, inherent, erroneous, benign, a root
  cause, a TensorFlow bug, a local bug, or an evaluator exception.
- Do not claim numerical equivalence, compile feasibility, memory reduction,
  performance improvement, scalability, ranking, default/production readiness,
  HMC/posterior correctness, or scientific validity.
- Do not resolve the analytical constant lane by analogy.
- Reviewers are read-only and cannot authorize runtime, source edits, human,
  model-file, funding, product/default, release, or scientific boundaries.

## Exact Next-Phase Handoff Conditions

After detached result agreement define:

- `complete`: exact 904-target, 12,316-observation, 2,386 nested-occurrence, and
  14,270 atomic-occurrence parity, deterministic payload, zero
  `unresolved_invalid`, all checks pass, and result review agrees;
- `unique_local_control_point`: `complete`, one exact local source operation is
  structurally upstream of every occurrence in one declared repair cohort,
  every path witness is present, and no competing local/framework control point
  remains for that cohort;
- `framework_proof_question`: `complete`, at least one cohort is uniformly
  `reverse_while_call_or_function` or `constant_or_shape_origin_unresolved`, no
  local control point is nominated for it, and one exact installed-source
  question could discriminate necessity without claiming it; and
- `minimal_runtime_counterfactual_needed`: `complete`, neither predicate above
  holds, but exactly one bounded counterfactual trace design can distinguish the
  remaining graph regions. This predicate grants no runtime authority.

The handoff is first-match ordered and therefore mutually exclusive:

| Reviewed result | Exact handoff |
| --- | --- |
| `unique_local_control_point` | Draft/review exactly `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-source-counterfactual-repair-subplan-2026-07-13.md`. It may authorize only a later semantics-preserving prospective test under its own boundary. Gate B remains rejected and Gate C/runtime remains blocked. |
| Else `framework_proof_question` | Draft/review exactly `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-framework-proof-subplan-2026-07-13.md`. It remains offline installed-source analysis unless separately reviewed. |
| Else `minimal_runtime_counterfactual_needed` | Draft/review exactly `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-minimal-counterfactual-trace-subplan-2026-07-13.md`. The plan must state a fresh command/budget/authority gate; drafting it does not authorize execution. |
| Otherwise | Write the discriminator blocker result and stop. No next subplan and no runtime/source authority. |

Every next subplan must state phase objective, inherited entry conditions,
required artifacts, required checks/tests/reviews, evidence contract, forbidden
claims/actions, exact next-phase handoff, and stop conditions. A next phase may
start only after its own exact `VERDICT: AGREE` and all separate authority gates.

Successful phase state:
`AUTODIFF_DISCRIMINATOR_RESULT_REVIEWED_NEXT_SUBPLAN_REVIEWED_RUNTIME_STILL_BLOCKED`.

## Stop Conditions

- subplan or result review does not converge within five material rounds;
- any parent, localization, target, occurrence, GraphDef, source, descriptor,
  authority, protected, or review byte drifts;
- any occurrence is missing, duplicated, overlapping, or
  `unresolved_invalid`;
- call/function/capture/output binding, deterministic rerun, negative control,
  static scan, pre-load guard, mutation suite, or scoped closure check fails;
- the scratch root is preexisting/nonempty, any undeclared path is written, a
  target worker appears, or another lane changes an in-scope read-only path;
- the result reviewer rejects the structural attribution or causal boundary;
  or
- continuation requires source/evaluator edits, a new trace, TensorFlow/XLA/GPU
  runtime, package/network/model/funding/default/product/release action, or
  scientific authority not granted by a later reviewed plan.

## Skeptical Pre-Execution Audit

Status: `PASSED_AFTER_ROUND1_REPAIR_AND_ROUND2_AGREEMENT`.

| Risk | Assessment |
| --- | --- |
| Wrong baseline | Controlled by the current reviewed parent result/artifact and exact 12,316 observations, not the superseded parent review or earlier diagnostic-only summaries. |
| Proxy promotion | Controlled: names, debug-field population, counts, path lengths, op families, elapsed time, and dominance ratios are explanatory only. Total graph witnesses are the primary criterion, and structural ownership still cannot prove causation. |
| Missing stop conditions | Controlled by exact parity, zero-invalid, drift, pre-load guard, mutation, determinism, closure, review, and authority stops. |
| Unfair comparison | No performance comparison occurs. All 18 autodiff GraphDefs and every reviewed occurrence are classified under one rule; analytical graphs are controls only. |
| Hidden assumptions | Exposed: debug metadata may be empty; graph ownership may remain composite; constants may lack a unique generator; generated reverse-while structure can be triggered by local VJP choices; ownership is not root cause. |
| Stale context | Controlled by exact parent/result/review/artifact/tool/source/framework/descriptor/R3 hashes before and after. |
| Environment mismatch | The installed TensorFlow 2.20.0 files and frozen GraphDefs are provenance-bound, but TensorFlow runtime is never imported. GPU is deliberately hidden. |
| Artifact fitness | Complete call/argument/return and reachability witnesses can separate graph regions and nominate the next discriminating question. They cannot establish a valid repair or memory/performance effect, which remain later gates. |

Offline implementation begins only after exact `VERDICT: AGREE`. Gate B remains
rejected and Gate C/runtime remains blocked regardless of plan review.
