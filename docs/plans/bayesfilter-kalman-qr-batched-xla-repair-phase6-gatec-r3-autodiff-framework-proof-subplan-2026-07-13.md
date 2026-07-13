# Phase 6 Gate C R3 Autodiff Framework Proof Subplan

Date: 2026-07-13

Status: `SUPERSEDED_UNEXECUTED_BY_LEAN_COUNTERFACTUAL_PLAN_2026_07_13`

Supersession note: no framework-proof implementation, test, snapshot, or result
artifact was created from this subplan. Its reviewed source analysis remains
useful context, but the user chose proportional academic governance and a
direct bounded counterfactual experiment. Do not resume this subplan unless the
user explicitly reactivates it.

Supervisor/executor: Codex in the current conversation.

Reviewer: fresh native Codex read-only substitute with
`codex_substitute_weaker` provenance. Claude is not retried after the managed
external-disclosure denial recorded by the parent phase.

Parent result:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-attribution-discriminator-result-2026-07-13.md`.

## Phase Objective

Use only the frozen discriminator artifact, its exact 420-target framework-proof
candidate set, immutable local source text, and immutable installed TensorFlow
2.20.0 Python source text to answer two bounded questions:

1. What does the exact installed source conditionally construct when the
   TensorFlow gradient registry dispatches a functional `While` or
   `StatelessWhile` operation to `while_v2._WhileGrad`, and which serialized
   structural signatures of that construction are present in all 18 frozen
   autodiff GraphDefs?
2. Can exact installed-source call and branch preconditions uniquely bind any of
   the 420 candidate targets to that construction, or do delegated op-gradient
   constructors, competing zero/shape constructors, native tape dispatch, and
   missing runtime branch evidence leave construction origin unresolved?

The candidate set contains exactly 344
`uniform_reverse_while_call_or_function` targets with 2,138 atomic occurrences
and 76 `uniform_constant_or_shape_origin_unresolved` targets with 1,242 atomic
occurrences, for 420 targets and 3,380 atomic occurrences. The broader
discriminator contains 1,270 constant/shape-state atomic occurrences; 28 of
those belong to mixed or other target aggregates and are controls, not members
of the 76-target candidate cohort.

This phase is a source derivation and frozen-signature consistency check. It may
prove a conditional statement of the form:

> Given registry dispatch to the exact installed `_WhileGrad` function, its
> preconditions, a nonempty differentiable `(ys, xs, grads)` set, and successful
> completion through `_build_while_op`, the function constructs one reverse
> functional while envelope with a gradient cond/body and resolves captured
> forward tensors as specified by the cited source.

It may not infer from a matching GraphDef signature that the Python route was
the unique historical constructor. It must separate the framework-built reverse
envelope from body contents delegated to `_GradientsHelper` and individual
registered op-gradient functions. It must also separate zero/shape constructors
in `while_v2.py`, `gradients_util.py`, `default_gradient.py`, `math_grad.py`, and
`linalg_grad.py`; node names or op types alone cannot select among them.

No source, evaluator, benchmark, test outside the finite write set, frozen
artifact, trace, GraphDef, R3 authority byte, or TensorFlow installation byte is
edited. TensorFlow is not imported or initialized. No target builder, trace,
XLA, GPU, Gate C, or performance run is authorized.

## Entry Conditions Inherited From The Previous Phase

All entry conditions are conjunctive and are checked before implementation,
before any phase-code load, and at closure.

### Reviewed parent lineage

| Artifact | Required SHA-256/state |
| --- | --- |
| Parent result path above | `6b2ca70580df40629444b1dc3e705dc81ddf6dd2ce04389951728bb3ce99b5e1`; reviewed result |
| Parent result Round 1 review | `65e75aa1dd6bd423d41f0d9a77c3a2635d033e73a79a0003826cd38018c417e5`; `VERDICT: AGREE` |
| Parent result final review | `969479011a4a93e1a029ea94a065d7e798e0c0bcdac840caf866ccf69057e6a8`; exact `VERDICT: AGREE` on the parent hash |
| Durable discriminator JSON | `8494478c0c9857818d72222c7a5966039083a8c16c44d35ab1dab24f74c22265`; 62,349,977 bytes; canonical payload `5ba99bfca3c3afa28d6593d48ca6a85e428e992d663b0b68d7f87e22bad4b2e3` |
| Discriminator check manifest | `55fc3687f0a772faa5b83f793d1281a02e477c900976f96b8e239db7adf81449`; every check true |
| Parent discriminator subplan | `ce14737c2bee978e4fc1fe6134c5b306d6bc6c39de95b78658b46b49c5a8247b` |
| Parent subplan final review | `24c74a874ef62034b607fbca5c2fddbb69647d3ec6a50d4ac95e2ae2f9af0e9e`; `VERDICT: AGREE` |

The parent state must remain
`passed_complete_structural_region_partition`: 904 targets, 12,316 parent
observations, 2,386 nested integer-constant occurrences, 14,270 atomic
occurrences, and zero `unresolved_invalid`. Its decision must remain no unique
local control point, Gate B `still_rejected`, Gate C `blocked`, runtime
unauthorized, analytical lane unresolved, and
`next_branch=autodiff_framework_proof`.

The candidate set must reproduce exactly from the parent decision and target
ledgers: 344 reverse-while targets / 2,138 atomics and 76 constant/shape targets
/ 1,242 atomics. Target identity is the exact `target_key`; occurrence identity
is the exact parent `atomic_key`. Names are preserved for identity and reporting
but never treated as source provenance.

### Frozen local, source, environment, and authority lineage

The complete parent descriptor, GraphDef, local-source, installed-source,
distribution, R3 work-root, budget, lease, no-worker, and protected algorithm
ledgers remain binding. At minimum:

| Path | Required SHA-256 | Role |
| --- | --- | --- |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `baf62b85f885073d0b72b5c13af0463ac5566f2429c16d5c98a542aa24c8eec9` | local one-VJP wrapper and batched tensor construction |
| `bayesfilter/linear/kalman_qr_tf.py` | `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b` | local forward functional while |
| `tensorflow/python/eager/backprop.py` | `c9a461d06085be50a2235e23e6c32a4649805fa2b5fb43d50e7fb421f92eed78` | tape Python callback, VSpace, and public gradient entry |
| `tensorflow/python/eager/imperative_grad.py` | `d31d9d2db5a5dde739fade26d7ae233d2bba88776d7a003af33f5ec2f4814737` | native tape dispatch boundary |
| `tensorflow/python/framework/ops.py` | `f795a78cce2827ce2b50892e20c01b628a07a71fa463ece766f0de9e2ec7677d` | gradient registry and lookup |
| `tensorflow/python/ops/gradients_util.py` | `1b4cf14a574b45f708ec4fba67bd450336a6b8e63f494ddf791fc5de5d981e98` | graph-gradient traversal and zero materialization |
| `tensorflow/python/ops/while_v2.py` | `756344a1c87911ca4a0678bea1388d070c7218afb1ec34a3f07c03473804719a` | functional while gradient construction |
| `tensorflow/python/ops/default_gradient.py` | `33f44537f8942aff124272dd3879f55db897c3935a9a7088d37750b6d39e675d` | default zero/one constructors |
| `tensorflow/python/ops/math_grad.py` | `61e4299fb89e265a8c7678411abb04b7bc08b38e61236e60be72fd9bc609aa46` | broadcast Add/Mul/Sum shape constructors |
| `tensorflow/python/ops/linalg_grad.py` | `41b41e93cac2dfce5bcd223bcc5498d2161e4f45ddb41ab51fceb42bbf8bd447` | Einsum gradient shape constructors |

TensorFlow distribution remains `tensorflow==2.20.0`; `METADATA` SHA-256
`aadf1cb4d0afeaaa947c7b32a8e9299cef3261137c16dd710bcf804fb6b4844c`
and `WHEEL` SHA-256
`3a52126eda4371f6a03eb2f01bb5ada5c65b3d3527a0a3e7c29840ff6e9f36a1`.
Python remains 3.13.13 in `/home/ubuntu/anaconda3/envs/tfgpu`.

`/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/` must still contain exactly
`import_discovery.json`, `budget_state/`, and 108 files under `trace/`.
`pilot/`, `children/`, and `progress/` remain absent. Gate B budget remains
closed, lease released, no target worker survives, and no forbidden authority
branch appears. The phase reads no R3 trace file directly; all frozen GraphDef
evidence enters through the reviewed discriminator JSON.

The new scratch root
`/tmp/kalman_qr_phase6_gateb_r3_autodiff_framework_proof/` must be absent before
the supervisor creates its empty root and empty `py_compile/` directory. Any
other preexisting content is an entry veto and is not removed.

The exact subplan must obtain a fresh read-only `VERDICT: AGREE` within five
material rounds before implementation. Any entry mismatch produces a blocker
result and stops without loading implementation code.

## Source Closure And Anchor Contract

The original eight framework anchors are binding:

| Anchor | Excerpt SHA-256 | Question |
| --- | --- | --- |
| `backprop.py:627-680` | `123ffb6717272921fa13d28cedbc24c96e16e5b59c99172bdaee15eab52c124d` | VSpace zero/one/shape callbacks |
| `backprop.py:960-1072` | `0c2f0fc25780baba2e8b8616fd2b31c799950a18fefe37dccc3c694b09b4533c` | `GradientTape.gradient` argument normalization and native call |
| `gradients_util.py:506-750` | `455868bf431dec460abfb835d9de2b57956652839537041f2d6cb376ccf1a97f` | graph traversal, registry lookup, zero output grads, and gradient invocation |
| `gradients_util.py:858-883` | `81f78d82d30501881d2f10cf2192f662dfd0fec82e43cf446a8ea6ce525a4605` | unconnected-zero return branch |
| `while_v2.py:322-437` | `d06db84a4a76cbbe1d983d2a47a6787244b7e6aa841ac961ec416e7ad96b9d28` | registered `_WhileGrad`, optional forward rewrite, capture resolution, reverse while build |
| `while_v2.py:535-580` | `f1b35c61f8e0f992504e6c3362737d682ed81273df5d38a5a75970e9c7f68a73` | resource/variant incoming-gradient zero branch |
| `while_v2.py:713-730` | `2c38c84dd6abca31f4984c737407df0e16fb32b1936b3aaea666c37e252e77e9` | body differentiation through `_GradientsHelper(..., unconnected_gradients="zero")` |
| `while_v2.py:970-1009` | `55b4ddf6e9bdb04d4a19f96b2f3667d66e7e35955a13f51da97604dcf23d97af` | conditional move-to-forward optimization and XLA exclusion |

The following exact closure anchors are additionally required because the eight
original excerpts do not close tape dispatch, registry, or competing
constructor calls:

| Anchor | Excerpt SHA-256 | Role |
| --- | --- | --- |
| `imperative_grad.py:29-72` | `6e89e4328210a19f1d3bee930bac148f0bf2a0ebce52a8b8703f9f59d072fd6f` | Python-to-native tape boundary |
| `backprop.py:118-153` | `7a53ac28c4dd3c3c5d3733a99007a04bbd49cad2d20df787506157a4f36bc1d8` | native callback registry lookup and gradient-function call |
| `ops.py:1694-1744` | `ce910da4d08ae3c2afe0e61a29e1a9b91a4ed8d74d53528d1d4ca750ece69e2e` | `RegisterGradient` semantics |
| `ops.py:1787-1800` | `789334b0634c0f55c7895aef49b30eb359d26ec53e9342e57d6aa98e94e980f4` | graph-mode registry lookup |
| `default_gradient.py:22-80` | `11e87b22896cf43365115cae76e06bc17e391cc5d12eb0079a4828aa9e31938f` | resource-aware zero/one branch alternatives |
| `math_grad.py:50-220` | `837c8eeccced3779fa5dfa7839bbfd65beb85ee2d94c4b0867bb7730199d03f0` | dynamic broadcast and Sum shape constructors |
| `math_grad.py:1357-1435` | `6cf38186ee815ac5fa9cc468c2b272eea7a074714cb1f92a8d8635c79cb96a98` | Add/Mul static-fast-path versus dynamic broadcast branches |
| `linalg_grad.py:59-372` | `87d9626f6188e58fcf43fad5633d070836d6d58da63bc7ea33a3458aa31d1fc7` | complete Einsum gradient constructor including its registration decorator |
| `while_v2.py:440-477` | `5a45ab2a9fe9eb49c6bf5f70a7a3cd712ff9627569dcb7a47e2d1fc0b890d1b1` | complete `_build_while_op` definition and stateful/stateless constructor branch |
| `while_v2.py:624-691` | `3ddc0795a8f96982889283e34b1fba5daadbe259ae272c9eb2f0e2d4cc06ea84` | complete `_create_grad_func` definition and capture-output construction |
| `while_v2.py:733-771` | `2d17dfa2e1c7464d0c7ce1653d5134e02fc9ff1b6f472d00dbfa26d4f2a78d98` | complete `_resolve_grad_captures` definition |

Only these 19 excerpts and their complete containing files are authorized
installed-source inputs. The analyzer may follow an additional source call only
when an AST-resolved call/import edge from one of these excerpts is required to
evaluate a declared branch and the plan is visibly amended and rereviewed
first. It may not perform an open-ended TensorFlow source search during
evidence generation.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Does installed TensorFlow 2.20.0 source conditionally force the frozen reverse-while envelope under explicit dispatch/preconditions, and can source plus frozen structural signatures uniquely attribute any candidate cohort without runtime evidence? |
| Candidate/mechanism | AST-bound call/registration graph; control-flow branch table; conditional source theorem; frozen forward/reverse while, call/function/capture, op, target, and atomic witnesses; competing-constructor ledger. |
| Expected failure | Native tape boundary does not preserve Python constructor identity; multiple registered op-gradient functions can emit the same node/op family; constant nodes lack producer edges; branch predicates are not serialized; or the frozen signature is compatible but not uniquely attributable. |
| Primary criterion | Reproduce exactly 420 candidate targets and 3,380 atomics; construct a complete source call/branch graph for all 19 anchors; state every theorem with antecedent, consequent, cited lines, and limitation; classify every candidate target once under the allowed source-evidence states; preserve competing constructors; deterministic payload across three runs. |
| Promotion veto | Missing/duplicate candidate; false call edge; stale anchor; source theorem without explicit antecedent; unique-origin or necessity claim from signature/name/op similarity; collapsed competing constructors; nondeterministic payload; or failed mutation. |
| Continuation veto | Input/source/authority drift, guard failure, unauthorized write/import/call, need for TensorFlow/runtime/new trace, in-scope concurrent write, or nonconverged review after five rounds. |
| Repair trigger | A reviewed result proves a concrete source-branch uncertainty that one bounded prospective counterfactual trace can discriminate without source editing. This triggers a later design subplan only; it is not runtime authority. |
| Explanatory only | Cohort sizes, atomic counts, node/op/name families, matching signature counts, branch counts, source proximity, and elapsed time. |
| Must not conclude | No unique historical constructor unless exact evidence proves it; no framework defect, avoidability, inherence, local defect, repair, numerical equivalence, Gate B pass, Gate C/runtime, memory/performance effect, ranking, or readiness. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Exact main question above. |
| Baseline/comparator | The reviewed 420-target/3,380-atomic candidate set and all 18 frozen boundary ledgers, compared against exact TensorFlow 2.20.0 source control flow. The other 484 targets and 10,890 atomics are provenance controls only. |
| Primary pass criterion | Exact target/atomic parity; total mutually exclusive source-evidence classification; complete 19-anchor ledger; source theorem and branch table pass all static/mutation checks; identical canonical digest and classification across two scratch and one durable run. |
| Promotion vetoes | Any primary failure; inference from names; theorem antecedent omitted; signature match represented as origin/necessity; missing alternative constructor; unsupported source line; or causal/performance overclaim. |
| Continuation vetoes | Entry/closure drift, unsafe process state, pre-load guard failure, TensorFlow import, target/runtime call, undeclared write, or need for new evidence beyond this reviewed offline scope. |
| Explanatory diagnostics | Signature-match counts, constructor-candidate counts, AST/call edges, op families, and branch coverage. |
| Not concluded | A conditional source theorem is not proof that the frozen graph historically followed that route; source compatibility is not necessity, defect, or repair evidence. |
| Preserved result | Strict proof JSON, exact logs/check manifest, phase result, detached agreeing result review, then exactly one reviewed next subplan or blocker-only stop. |

## Allowed Source-Evidence States

The validator first computes `I(t)`, which is true for a missing/duplicate
target or atomic, stale source, false call graph, inconsistent witness, or
unsupported classification input. Any `I(t)` is `invalid_or_incomplete` and a
continuation veto; invalid targets never enter valid-state classification.

For each valid target `t`, compute these base facts without semantic names:

- `E(t)`: the exact entity digest and call/function binding identify the reverse
  while call or one of its exact called cond/body function envelopes, and every
  serialized consequent of the `_WhileGrad` theorem holds;
- `G(t)`: exactly one bound Add/Mul/Sum/Einsum constructor has every required
  antecedent serialized and satisfied and every required consequent matched;
- `C(t)`: at least two bound zero/shape constructors remain structurally
  compatible, or a required predicate that would choose among compatible
  zero/shape constructors is absent from frozen evidence;
- `X(t)`: at least one bound constructor is applicable, every applicable
  constructor has complete serialized antecedent and consequent evidence, and
  every applicable constructor contradicts a required serialized consequent;
- `D(t)`: the exact function-owner/call binding puts the target inside the
  reverse body and construction is delegated through `_GradientsHelper` or an
  op-specific gradient function; and
- `A(t)`: at least one source branch is structurally applicable. Applicability
  requires entity/edge/function/attribute/tensor evidence, never a name or op
  type alone.

The six valid state predicates are explicitly disjoint:

1. `reverse_envelope_conditional_framework_theorem` iff `E(t)`.
2. `op_gradient_conditional_constructor_candidate` iff
   `not E(t) and G(t) and not C(t)`. Native dispatch and historical origin stay
   unobserved; this is a conditional candidate, not unique provenance.
3. `zero_or_shape_competing_framework_constructors` iff
   `not E(t) and C(t)`. This includes a unique-looking signature whose deciding
   source predicate is not serialized.
4. `source_signature_incompatible` iff
   `not E(t) and not G(t) and not C(t) and X(t)`. This is only a repair trigger
   when all evidence remains valid; it is not a framework-defect finding.
5. `reverse_body_delegated_constructor_unresolved` iff
   `not E(t) and not G(t) and not C(t) and not X(t) and D(t)`.
6. `source_evidence_insufficient` iff
   `not E(t) and not G(t) and not C(t) and not X(t) and not D(t)`. This covers
   `not A(t)` and any otherwise unevaluable non-zero/shape case.

States 1 and 2 establish only conditional source facts. States 3, 5, and 6 are
honest unresolved outcomes. The validator must assert that exactly one of the
six raw valid-state predicates is true for every valid target and that every
pairwise intersection is empty; choosing the first true state is forbidden.
Names may be reported after classification, but mutation of all non-contract
names must leave base facts and state unchanged.

## Required Artifacts And Write Set

Before implementation, create an authorized-state snapshot at:

`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_framework_proof_authorized_snapshot_2026-07-13.json`.

It records every entry hash/state, exact 420-target/3,380-atomic candidate digest,
19 excerpt hashes, complete containing-file hashes, R3 inventory and authority,
empty scratch state, all phase-owned path/stem absence, and the unrelated
dirty-worktree status/digest. Other-lane paths remain evidence-only and may not
be modified, staged, cleaned, or represented as phase work.

Only these exact new repository paths or finite stems may be written:

- `docs/benchmarks/prove_kalman_qr_phase6_gateb_r3_autodiff_framework_path_2026_07_13.py`;
- `docs/benchmarks/run_guarded_kalman_qr_phase6_gateb_r3_autodiff_framework_proof_2026_07_13.py`;
- `tests/test_kalman_qr_phase6_gateb_r3_autodiff_framework_proof.py`;
- the authorized snapshot path above;
- `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_framework_proof_2026-07-13.json`;
- the same benchmark stem ending `_static_scan.txt`, `_py_compile.txt`,
  `_focused_pytest.txt`, `_run1.txt`, `_run2.txt`, `_durable_run.txt`, or
  `_check_manifest.json`;
- `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-framework-proof-result-2026-07-13.md`;
- review paths under `docs/reviews/` with the exact framework-proof subplan or
  result basename, `-review-`, and suffix in
  `{round1,round2,round3,round4,round5,final}-2026-07-13.md`;
- only after detached result agreement, exactly one of:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-minimal-counterfactual-trace-design-subplan-2026-07-13.md`,
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-source-closure-extension-subplan-2026-07-13.md`, or
  the framework-proof result path refreshed as a blocker with no next plan;
- reviews for only the selected next-plan basename under the same finite review
  suffix set.

No existing file may be edited. The only scratch paths are:

- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_framework_proof/run1.json`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_framework_proof/run2.json`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_framework_proof/py_compile/proof.pyc`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_framework_proof/py_compile/guard.pyc`; and
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_framework_proof/py_compile/test.pyc`.

Every command uses `CUDA_VISIBLE_DEVICES=-1` and
`PYTHONDONTWRITEBYTECODE=1`. Pytest cache, implicit `__pycache__`, filesystem
fixtures, convenience files, and arbitrary `/tmp` output are forbidden.

## Required Analyzer And Witness Contract

The analyzer uses only the Python standard library. It parses installed source
with `ast` and reads the frozen JSON with strict finite-number JSON decoding.
It never imports any source file as a module.

It must:

- revalidate every parent/result/review/artifact/manifest/source/distribution
  hash before reading conclusions;
- reconstruct the exact candidate set independently from both the parent
  decision target list and `partition.targets`, proving equality of 420 target
  keys, 3,380 atomic keys, aggregate states, and per-target atomic counts;
- preserve a control ledger for the 484 noncandidate targets and prove none is
  promoted into the candidate set;
- parse all 19 excerpts and containing files into an AST-bound definition,
  decorator, import alias, call edge, return, and branch-predicate ledger;
- derive the `_WhileGrad` theorem as explicit antecedent/consequent clauses,
  distinguishing unconditional statements inside the function from optional
  `while_op_needs_rewrite`, skip-input, trainability, default-zero,
  XLA-exclusion, and op-gradient branches;
- map serialized GraphDef evidence only to theorem consequents that are actually
  present in the frozen boundary/target witnesses;
- build a competing-constructor ledger for every constant/shape target from
  exact AST branches and available structural fields;
- preserve a per-target missing-evidence ledger, including unobserved native
  dispatch, Python branch values, tensor shape objects, registry identity, and
  historical call stack;
- classify all 420 targets once by evaluating the six disjoint valid-state
  predicates above after the invalidity veto; and
- emit a strict JSON payload with canonical SHA-256 excluding only start time,
  finish time, wall time, and output path.

Every source theorem witness includes source file SHA-256, excerpt SHA-256,
definition/decorator identity, exact line interval, AST digest, antecedents,
consequents, exception/early-exit paths, delegated calls, and prohibited
converse. Every target witness includes target key, all atomic keys, aggregate
state, frozen structural facts used, source theorem/branch references,
competing constructors, missing facts, classification predicate vector, and
claim boundary.

The validator rejects any state decided by target name fragments, op type alone,
source proximity, frequency, or cohort uniformity. It rejects the words
`necessary`, `inherent`, `avoidable`, `bug`, `fixed`, `improved`, or equivalent
in affirmative conclusions unless they appear in an explicit nonclaim or
quoted source text with a claim-boundary label.

## Required Checks, Tests, And Reviews

### Before implementation

- preserve all review rounds and obtain exact `VERDICT: AGREE` on this subplan
  within five material rounds;
- rerun the full entry/hash/authority/R3/no-worker/scratch checks;
- validate the new authorized snapshot and skeptical audit; and
- confirm the question is answerable offline with only standard-library AST and
  JSON parsing.

### Before any phase-code import or test

Run and preserve a CPython AST boundary scan over exactly the analyzer, guard,
and focused test. It must reject TensorFlow/BayesFilter imports, dynamic imports,
`eval`, `exec`, target/trace/XLA/device/runtime surfaces, subprocess/network,
package/model/external-review calls, filesystem test fixtures, undeclared
writes, open-ended source discovery, and classifiers based on semantic target
names.

Only after the scan passes, explicitly compile the three sources with
`py_compile.compile(..., cfile=<exact authorized path>, doraise=True)`. Every
test and evidence invocation enters through the pre-load guard. The guard
installs import, call, device, process, network, source-read, and write
boundaries before loading phase code and asserts TensorFlow is absent before and
after.

Focused tests must include at least:

- exact 420-target, 3,380-atomic reconstruction and 484-target control coverage;
- duplicate, missing, wrong-aggregate, and candidate-list disagreement
  rejection;
- excerpt/file/distribution hash drift and line-range drift rejection;
- AST registration and dispatch edges, `_WhileGrad` theorem clauses, optional
  rewrite, reverse build, capture resolution, delegated `_GradientsHelper`, and
  native boundary representation;
- competing zero/shape constructors and absent branch-value handling;
- invalidity as a pre-classification veto; all six valid Boolean state
  predicates; pairwise raw-predicate exclusivity; totality; and rejection of a
  classifier that merely selects the first true predicate;
- semantic-name perturbation with unchanged classification;
- mutations of decorator, called function, branch predicate, return/build call,
  capture call, source line, target structural fact, atomic identity, and claim
  boundary;
- rejection of converse reasoning from signature match to unique origin;
- canonical determinism and declared metadata exclusions; and
- guard rejection of TensorFlow import, open-ended source reads, target/runtime,
  subprocess/network/device calls, and undeclared writes.

### Offline evidence and closure

- run two independent GPU-hidden scratch invocations and one durable invocation
  through the guard;
- require identical canonical payloads, theorem ledgers, candidate partitions,
  and classifications across all three runs;
- rerun entry, source, authority, process, repository-write, scratch, and
  other-lane closure checks;
- write a result with run manifest, decision table, inference-status table,
  candidate-versus-direction distinction, repair-loop disclosure, and post-run
  red team;
- freeze the result and obtain a detached fresh native Codex review agreeing on
  source derivation, target/atomic coverage, competing constructors, converse
  boundary, hashes, and runtime/write authority; and
- only after result agreement, draft and review exactly one next subplan chosen
  by the first-match handoff below.

## Forbidden Claims And Actions

- Do not edit local algorithm, benchmark, evaluator, parent tools/tests/results,
  installed TensorFlow, traces, GraphDefs, R3 authority, budget, lease, or work
  root.
- Do not import/initialize TensorFlow, invoke BayesFilter target code, build a
  concrete function, trace, compile/run XLA, enumerate/use GPU, run Gate C or
  Phase 7, benchmark, install packages, access network, launch subprocesses from
  phase code, or use filesystem fixtures.
- Do not perform open-ended TensorFlow source discovery. Amend and rereview this
  subplan before adding a source file or excerpt.
- Do not use names, op type alone, uniformity, source proximity, or source
  comments as unique provenance.
- Do not infer the converse of the conditional source theorem. A frozen
  signature compatible with `_WhileGrad` does not prove it was historically or
  uniquely constructed by `_WhileGrad`.
- Do not call a structure necessary, inherent, avoidable, erroneous, benign, a
  TensorFlow/local bug, fixed, or an evaluator exception.
- Do not claim numerical equivalence, compile feasibility, memory reduction,
  performance improvement, scalability, ranking, default/production readiness,
  HMC/posterior correctness, or scientific validity.
- Do not resolve the analytical lane by analogy.
- Reviewers remain read-only and cannot authorize source edits, runtime, human,
  model-file, funding, product/default, release, performance, or scientific
  boundaries.

## Exact Next-Phase Handoff Conditions

After detached result agreement define:

- `complete`: exact candidate/control reconstruction, all 420 targets and 3,380
  atomics classified exactly once, complete 19-anchor theorem/branch ledger,
  deterministic payload, zero invalid targets, and all checks/review pass;
- `bounded_runtime_discriminator_ready`: `complete`, at least one material source
  uncertainty is stated as two mutually exclusive prospective trace outcomes,
  each observable through one bounded new trace without source edits, and the
  source result does not require broader source closure;
- `source_closure_extension_needed`: `complete`, no bounded runtime discriminator
  is yet well-defined, but exactly one AST-resolved call/import edge identifies
  one finite additional installed-source excerpt that can decide a material
  branch offline; and
- `blocker`: not `complete`, or neither discriminating predicate holds.

The handoff is first-match ordered:

| Reviewed result | Exact handoff |
| --- | --- |
| `bounded_runtime_discriminator_ready` | Draft/review exactly `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-minimal-counterfactual-trace-design-subplan-2026-07-13.md`. It designs command, budget, comparison, stop, and authority gates only. Drafting/reviewing it does not authorize runtime. |
| Else `source_closure_extension_needed` | Draft/review exactly `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-source-closure-extension-subplan-2026-07-13.md`. It remains offline and adds only the exact justified excerpt. |
| Otherwise | Write/refresh the framework-proof blocker result and stop. No next subplan and no runtime/source-edit authority. |

Every next subplan must state phase objective, inherited entry conditions,
required artifacts, required checks/tests/reviews, evidence contract, forbidden
claims/actions, exact handoff, and stop conditions. It begins only after its own
exact `VERDICT: AGREE` and separate authority gates.

Successful phase state:
`FRAMEWORK_PROOF_RESULT_REVIEWED_NEXT_SUBPLAN_REVIEWED_RUNTIME_STILL_BLOCKED`.

## Stop Conditions

- subplan or result review does not converge within five material rounds;
- any parent/result/artifact/source/distribution/authority/protected hash drifts;
- candidate target/atomic/control reconstruction is incomplete or inconsistent;
- a theorem omits an antecedent, asserts a converse, or lacks exact source
  anchors;
- a target is missing, duplicated, overlaps states, or is
  `invalid_or_incomplete`;
- a competing constructor is omitted or classification depends on names/op type
  alone;
- static scan, guard, mutation suite, determinism, source closure, process,
  scratch, repository closure, or other-lane boundary check fails;
- another lane changes an in-scope read-only path; or
- continuation requires source edit, TensorFlow/runtime/new trace, XLA/GPU,
  package/network/model/funding/default/product/release, performance claim, or
  scientific authority not granted here.

## Skeptical Pre-Execution Audit

Status: `PASSED_AFTER_ROUND1_AND_ROUND2_REPAIRS_ROUND3_AGREEMENT`.

| Risk | Assessment |
| --- | --- |
| Wrong baseline | Controlled by the reviewed 420-target/3,380-atomic candidate set, all 18 frozen boundaries, and 484-target control ledger; not by target names or the whole 14,270-atomic partition as if all were framework candidates. |
| Proxy promotion | Controlled: source-signature matches, names, op families, counts, comments, and elapsed time are explanatory only. The primary criterion is a complete conditional theorem/branch ledger with explicit missing facts and competing constructors. |
| Missing stop conditions | Exact parity, source/hash drift, false call/theorem, converse, competing-constructor, guard, mutation, determinism, write, review, and authority stops are declared. |
| Unfair comparison | No performance comparison occurs. Every candidate target uses the same six disjoint valid-state predicates; no ordered selection or first-match classification occurs. Noncandidate targets are controls. |
| Hidden assumptions | Exposed: native tape dispatch is not serialized; registry identity and Python branch values may be absent; reverse envelope does not determine body constructors; constants can have multiple constructors; source compatibility is not origin. |
| Stale context | Controlled by exact parent/result/review/artifact/manifest/source/distribution/R3 hashes at entry and closure. |
| Environment mismatch | Installed TensorFlow 2.20.0 source is provenance-bound but never imported. The phase deliberately cannot observe native/runtime behavior. GPU is hidden. |
| Artifact fitness | AST and frozen GraphDef evidence can prove conditional source control flow and serialized signature consistency. They cannot establish historical origin, counterfactual effect, repair, or performance; those remain explicit nonclaims. |

Implementation begins only after this audit is updated to a reviewed pass and
the exact plan receives `VERDICT: AGREE`. Gate B remains rejected and Gate
C/runtime remains blocked regardless of plan review.
