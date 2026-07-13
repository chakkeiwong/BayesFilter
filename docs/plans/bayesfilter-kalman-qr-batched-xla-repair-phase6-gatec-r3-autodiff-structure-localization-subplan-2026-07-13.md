# Phase 6 Gate C R3 Autodiff Structure Localization Subplan

Date: 2026-07-13

Status: `ROUND3_REVISED_REVIEW_REQUIRED_OFFLINE_ONLY_RUNTIME_BLOCKED`

Supervisor/executor: Codex in the current conversation.

Reviewer: fresh native Codex read-only substitute with
`codex_substitute_weaker` provenance. Claude is not retried after the managed
external-disclosure denial.

Parent result:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-trace-rejection-blocker-result-2026-07-12.md`.

## Phase Objective

Use only frozen source text and preserved R3 GraphDef bytes to enumerate and
attribute the observed batch-dependent autodiff entities. The target universe
is every top-level insertion/deletion/change, generated-function
insertion/deletion/change, changed function-body entity, and unresolved
autodiff residual reported by the final diagnostic across all three dimensions.

For every target entity, record exactly one mutually exclusive coverage state:

- `mapped_exact`: the observed construction has a complete preserved-graph
  neighborhood and exact local or installed-framework source anchors;
- `enumerated_causally_ambiguous`: the entity and neighborhood are complete,
  but two or more bounded source/framework explanations remain and none is
  promoted; or
- `missing_or_incomplete`: the entity, edge/function neighborhood, provenance,
  or source-anchor coverage is absent or incomplete.

This phase may nominate a bounded semantics-preserving repair hypothesis. It
cannot establish that construction is avoidable or inherent: avoidability
requires a later counterfactual repair test, while inherent TensorFlow behavior
requires a separately reviewed framework-source/counterfactual proof gate.
The analytical constant lane remains unresolved. No source or evaluator rule is
edited here.

## Entry Conditions Inherited From The Previous Phase

All conditions are conjunctive and must be rechecked before implementation and
after the durable run.

### Parent and diagnostic lineage

| Artifact | Required SHA-256 |
| --- | --- |
| Parent result path above | `7fd716362dc0c53bf4e1a10bb7412510ebb1a3afb1eabe27ba8ef7c61a6ad390` |
| `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-trace-rejection-blocker-result-review-round1-2026-07-13.md` | `69d006a89f3e9749ca04a1f3ee74b5a07ab284d467938982833a2624fa4bb51c` |
| `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-trace-rejection-blocker-result-review-final-2026-07-13.md` | `27e1f97c0113b1ab53124f0bc35735217eaefcb1d0a6930a265740cfbe2dbaab` |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_2026-07-12.json` | `637273af37ed2606b9bd0bc4868a1719a65ad17d89d94ab018e5678082fb25ff`; 5,780,057 bytes; canonical payload `30a2753246d4c86a6952268fad5a49d8e77991084f4100a45d1eca051c710cd7` |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_check_manifest_2026-07-12.json` | `fcc5ad1ea9cf6f06ce4ae0dc83e0da00d2ce08b9bfb805f3220748cf9bc1e54f` |
| `docs/benchmarks/diagnose_kalman_qr_phase6_gateb_r3_trace_rejections_2026_07_12.py` | `7e0f1818bb5bc349adea2b2ba703cdd4d552c3e6ffb195f9613f791cd6710c4a` |
| `tests/test_kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic.py` | `c902fc65c513960de7bcef36c493057aeb05577946df94cb62cdfb2d432ee44c` |

The diagnostic must still be `passed`, all ten strict validations must remain
true, and its six cohort classifications must remain three analytical
`undetermined` plus three autodiff `mixed_causes`.

### R3 and R2 authority lineage

| Artifact | Required SHA-256/state |
| --- | --- |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_census_2026-07-12.json` | `7444fb41ef9d125990dee93a5370227c4b9ec0987ee37cb9ab7dfd362281d2b6` |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_cpu_xla_pilot_2026-07-12.json` | `1344f701eabfbec56e447b2cf40f3a8a4dd6cb79ef195659a50dfa4f03fb8ea2`; exactly two `not_launched:trace_gate_not_passed` records and zero XLA calls |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_2026-07-12.json` | `dd3a9495585a2f4b2995f1910da7d7b733f68467a285fa1856b5acf881f3886d` |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_attestation_2026-07-12.json` | `9399be89c2263b0898c2a1c7718ca8484a81cb3fea31e8740c31750a0147e60f` |
| `/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/import_discovery.json` | `6db78f6a610e10681a10f4231ba0b32f798264065b12fe6f67290b8e5663719c` |
| `/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/budget_state/gate_b-7d630ff42cc759c02d3e6618c90b97923ec9a9e8cba5b99dd41ee94e09347a33.json` | `a44904233d54b5906a96c0ba26c6e75e8ef33980e531ea7546d43099cbdefc28`; `closed` |
| Same path with suffix `.lease` | `8005baec0a329003b46049b2aa21cfaf887e1934f8b60f8499a1f3aeb10ac14b`; `released` |
| `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_invalid_harness_archive_2026-07-12.json` | `40e6a186a28cd15d4ab3901f516854a6d84065fcb9759716108ab8e103e7834d` |
| `/tmp/kalman_qr_phase6_cpu_xla_gateb_r2/import_discovery.json` | `8ae6086bd6b8bbebd7bf236536a80cb6b8befa993a9e686801c451e8fec4c8ac` |
| R2 budget-state path recorded by the R3 trace ledger | `a4cc284b64d6527a7357171f4c47395a7f29f7fed7e50b15563257feae09390f` |
| Same R2 path with suffix `.lease` | `ae711efe84056ae416d5fe2d2d40751b91afaa7f3a2e3530f095fb501a03b456` |

The R3 work root must contain exactly `import_discovery.json`, `budget_state/`,
and `trace/`; `trace/` must contain exactly 108 files. `pilot/`, `children/`,
and `progress/` must be absent. No target worker may survive, the authority
namespace must remain exact, and every forbidden branch pair must remain absent.

### Source and installed-framework provenance

The localization JSON must bind and revalidate this complete local dependency
ledger before and after execution:

| Path | Required SHA-256 |
| --- | --- |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `baf62b85f885073d0b72b5c13af0463ac5566f2429c16d5c98a542aa24c8eec9` |
| `scripts/kalman_qr_benchmark_contract.py` | `f52a20624eb3c8c72c59cc2809f4cd870de4c3c84276fed97f308bc4f0a75e64` |
| `bayesfilter/__init__.py` | `986fd24cc5c86812c53fff10f2e169525783921496ab7997d988d5423ff9663b` |
| `bayesfilter/diagnostics.py` | `da00bf6421d55952d6e0a4ee58e4402bfeee414c90d44e2745485b73b73e1fc4` |
| `bayesfilter/linear/__init__.py` | `df9e248a8fc24063112d3bdcfbff0b1e46ef30781c13d05b36d932909c5bb46e` |
| `bayesfilter/linear/dtypes_tf.py` | `de534d5411a372e0344b1248e1c192dcc0206b21a8bef86c13cff15024ef960d` |
| `bayesfilter/linear/kalman_qr_derivatives_tf.py` | `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57` |
| `bayesfilter/linear/kalman_qr_tf.py` | `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b` |
| `bayesfilter/linear/qr_factor_tf.py` | `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401` |
| `bayesfilter/linear/types_tf.py` | `6f79ae42472ecd304e6012bdf3c1fba13e97ac1431d537d8152ecb96133f8af8` |
| `bayesfilter/results_tf.py` | `e09be453fa62e1b3a3ef16c542ca3782f82c1967303509430b5676168e91cea6` |
| `bayesfilter/structural.py` | `3a181ef4c8f9e67ab8b923b5c016d6df3da13e4b77d4e940f0c07a60fc37fd4f` |

TensorFlow provenance is distribution `tensorflow==2.20.0`, Python 3.13.13.
The wheel `METADATA` and `WHEEL` hashes are respectively
`aadf1cb4d0afeaaa947c7b32a8e9299cef3261137c16dd710bcf804fb6b4844c`
and `3a52126eda4371f6a03eb2f01bb5ada5c65b3d3527a0a3e7c29840ff6e9f36a1`.
If framework-origin language is used, it must cite exact line anchors in the
hashed installed files: `tensorflow/python/eager/backprop.py`
(`c9a461d06085be50a2235e23e6c32a4649805fa2b5fb43d50e7fb421f92eed78`),
`tensorflow/python/ops/gradients_util.py`
(`1b4cf14a574b45f708ec4fba67bd450336a6b8e63f494ddf791fc5de5d981e98`),
or `tensorflow/python/ops/while_v2.py`
(`756344a1c87911ca4a0678bea1388d070c7218afb1ec34a3f07c03473804719a`).

GraphDef decoding must not import TensorFlow. It may build a private
`google.protobuf` descriptor pool from the exact generated descriptor literals
in this complete installed-schema ledger:

| Exact installed path | Required SHA-256 |
| --- | --- |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/types_pb2.py` | `da2051ab56bacdd352d423406f61e6a400ee7c5ccd697d438a4af2347a38b950` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/tensor_shape_pb2.py` | `f2270dddfc27b1fc4e73147eab9bac16b494e44bc582fc37c6b73732e898f360` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/resource_handle_pb2.py` | `2cf571a9bbd6933942fa1ff965eef70f6b744278c293dd817993aeb49ae5638d` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/full_type_pb2.py` | `ad3195d1f19f5194092ea95cf2b138a50a8d526fd79acf6715d07a19e15975a8` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/tensor_pb2.py` | `08e6ffb6ada2798aecc04c8ee963b3d2f14a9ca5e1bb2c516937bb0ac0f9ad64` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/attr_value_pb2.py` | `cfabd36cba17dbeb2c7d8a2f12f4212c65a2aa19d011dfcb3940a0a5d37bcdbf` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/op_def_pb2.py` | `9b4fcdb6a51416554488d21b29d3672dff4f58b5054cf08027b818b957977075` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/node_def_pb2.py` | `ba780b9f27bd40b9e405b7aa2b772095f6807674831d19b4b6fc107097a8bad7` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/function_pb2.py` | `ff554bd811b3f068893847b6e85393968f409963225c84034696dc6acbcdcc73` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/graph_debug_info_pb2.py` | `33abded1da2221458edc7e14aa9fff0e286d8b7339be43792bf063e3aae9422c` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/versions_pb2.py` | `d47bf8af3dc59be01e31356403a27285c43d28c27a775bbf476a8e6b1f9a7508` |
| `/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/core/framework/graph_pb2.py` | `ef332be91bd9fd86fdb3dd58953b870a14443a73fc3894e5a5ec58792b85d437` |

Every path/hash must match before the localizer is implemented or loaded and
again after the durable run. The durable artifact must also record sizes,
dependency order, protobuf version `6.33.5`, and prove decoded raw hashes,
entity counts, and stable deltas agree with the final diagnostic. Descriptor
extraction may use `ast.literal_eval`; importing any `tensorflow.*_pb2` module
is forbidden.

The exact subplan must converge under bounded read-only review within five
material rounds. Any entry mismatch writes a blocker result and stops without
modifying frozen inputs.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Which exact preserved graph neighborhoods and source/framework operations account for every observed batch-dependent autodiff entity and residual? |
| Candidate/mechanism | Pure offline GraphDef decoding, complete structural-delta inventory, dependency/call slices, source AST anchors, and installed TensorFlow text anchors. |
| Expected failure | Incomplete descriptor closure; lost graph edge/function target; nonunique source mapping; source drift; or a target entity omitted from the partition. |
| Primary criterion | Every target entity is present exactly once in a complete partition and is `mapped_exact` or `enumerated_causally_ambiguous`; zero `missing_or_incomplete`; deterministic evidence; exact provenance. |
| Promotion veto | Any missing/duplicate target, incomplete slice, stale/false anchor, unsupported causal language, failed mutation, or source/evidence drift. |
| Continuation veto | Frozen input drift, unsafe process state, missing artifact, in-scope concurrent write, failed runtime guard, or need for target/XLA/GPU/runtime authority. |
| Repair trigger | Complete exact attribution nominates a bounded local semantics-preserving counterfactual; nomination is not evidence that the repair is valid or avoidable. |
| Explanatory only | Entity counts, slice sizes, name patterns, source proximity, and offline elapsed time. |
| Must not conclude | No avoidable/inherent claim, source bug, evaluator exception, Gate B pass, Gate C authority, XLA/GPU viability, memory/performance repair, ranking, or production/scientific readiness. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Exact research question above. |
| Baseline | Final diagnostic bytes plus all 36 preserved R3 raw GraphDefs, with the 18 autodiff GraphDefs as the attribution target and analytical records retained only to preserve the unresolved boundary. |
| Pass criterion | Complete mutually exclusive target partition, exact graph and source/framework anchors, deterministic canonical digest, and all negative controls passing. |
| Veto diagnostics | `missing_or_incomplete`, duplicate target, lossy decoder/slice, diagnostic mismatch, stale source, runtime-guard call, or unsupported avoidable/inherent claim. |
| Explanatory diagnostics | Counts, neighborhood depth, name families, occurrence patterns, and source-line density. |
| Not concluded | Attribution alone does not establish semantics, numerical correctness, repairability, XLA viability, memory/performance improvement, or readiness. |
| Preserved result | Strict localization JSON, exact logs/check manifest, phase result, detached agreeing result review, then exactly one reviewed classified handoff or blocker-only stop. |

## Required Artifacts And Write Set

Before implementation, create
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_structure_localization_authorized_snapshot_2026-07-13.json`.
It records existence, size, hash, and Git status for every authorized path and
tight stem below, plus the preexisting unrelated repository status hash. The
unrelated file list is evidence only and cannot be cleaned or modified.

Only these exact new paths or finite stems may be written:

- `docs/benchmarks/localize_kalman_qr_phase6_gateb_r3_autodiff_structure_2026_07_13.py`;
- `docs/benchmarks/run_guarded_kalman_qr_phase6_gateb_r3_autodiff_structure_2026_07_13.py`;
- `tests/test_kalman_qr_phase6_gateb_r3_autodiff_structure_localization.py`;
- the authorized snapshot path above;
- `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_structure_localization_2026-07-13.json`;
- the same benchmark stem ending `_static_scan.txt`, `_py_compile.txt`,
  `_focused_pytest.txt`, `_run1.txt`, `_run2.txt`, `_durable_run.txt`, or
  `_check_manifest.json`;
- `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-result-2026-07-13.md`;
- paths under `docs/reviews/` with exact filename prefix
  `bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-subplan-review-`
  and suffix in `{round1,round2,round3,round4,round5,final}-2026-07-13.md`;
- paths under `docs/reviews/` with exact filename prefix
  `bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-result-review-`
  and suffix in `{round1,round2,round3,round4,round5,final}-2026-07-13.md`;
- after the result review agrees, exactly one next-plan path:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-source-counterfactual-repair-subplan-2026-07-13.md`
  or
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-attribution-discriminator-subplan-2026-07-13.md`;
- for the selected next-plan path only, paths under `docs/reviews/` with exact
  filename prefix
  `bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-source-counterfactual-repair-subplan-review-`
  or
  `bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-attribution-discriminator-subplan-review-`
  and suffix in `{round1,round2,round3,round4,round5,final}-2026-07-13.md`; or
- a localization blocker result under the exact localization-result path, with
  no next subplan.

The scratch write set is exact. Only these paths are authorized:

- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_structure_localization/run1.json`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_structure_localization/run2.json`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_structure_localization/py_compile/localizer.pyc`;
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_structure_localization/py_compile/guard.pyc`; and
- `/tmp/kalman_qr_phase6_gateb_r3_autodiff_structure_localization/py_compile/test.pyc`.

The supervisor may create only the empty scratch root and its empty
`py_compile/` directory before checks. A preexisting or nonempty scratch root is
an entry veto and is not removed. All test/evidence runs use
`PYTHONDONTWRITEBYTECODE=1`; `__pycache__`, pytest cache, capture files, and
pytest temp/base-temp paths are forbidden. Tests may not request `tmp_path`,
`tmp_path_factory`, `tmpdir`, or any filesystem temp fixture. The guard and
closure inventory reject every other scratch path. No R3 work-root file may be
written or regenerated.

## Required Checks, Tests, And Reviews

### Before implementation

- preserve every review round visibly and obtain exact `VERDICT: AGREE` on this
  path within five material rounds;
- rerun every entry hash/state/namespace/no-worker/branch-absence check;
- create and validate the scoped authorized-state snapshot;
- record a skeptical audit against wrong baseline, proxy promotion, unfair
  comparison, hidden assumptions, stale context, environment mismatch, and
  artifacts unable to answer causation; and
- confirm the localizer imports only standard-library modules and
  `google.protobuf`; the guard may import pytest only after its guards are
  installed in exact test mode. Neither may import TensorFlow or BayesFilter
  algorithm/benchmark code.

### After implementation and before any import or test

Run the static boundary scan first over the localizer, guard harness, and test,
and preserve its exact log. It must reject:

- imports of `tensorflow`, the selected benchmark module, or BayesFilter
  algorithm modules;
- dynamic imports except the guard harness's exact post-guard load of the exact
  localizer in localizer mode and pytest's post-guard collection of the exact
  test path in test mode, `eval`, `exec`, target builders/methods, `tf.function`,
  `GradientTape`, concrete-function/tracing APIs, XLA/JIT APIs, device/GPU
  enumeration, subprocess/process/shell/network calls, and writes outside the
  exact output argument; and
- any test not confined to pure decoder, graph/source-attribution, mutation,
  deterministic-payload, provenance, and boundary behavior.

Only after the scan passes may bytecode compilation and focused GPU-hidden
tests run. Compilation must call `py_compile.compile(..., cfile=<exact path>,
doraise=True)` separately for the localizer, guard, and test and produce only
the three exact `.pyc` paths above. `python -m py_compile` and implicit
`__pycache__` output are forbidden.
Every test and evidence invocation must enter through the exact guard harness,
which installs guards before loading the test or localizer. The guards fail on
any TensorFlow import, selected builder/Kalman call surface,
subprocess/process/shell/network call, device enumeration, or write outside the
exact invocation output and declared scratch paths. In `--mode localizer`, the
harness may use only one exact post-guard `runpy.run_path` load of the localizer.
In `--mode test`, it may import pytest only post-guard and call `pytest.main`
with exactly the declared flags and exact test path; the test must verify the
guard token before using one exact `runpy.run_path` load of the localizer. Both
modes verify afterward that no `tensorflow` module entered `sys.modules`.
Focused tests run
with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONDONTWRITEBYTECODE=1`,
`-p no:cacheprovider`, and `--capture=no`; they import the localizer only after
the harness guards are active and the guard forbids every test write. Required
mutations cover truncated and unknown descriptor data,
missing/duplicate entity, changed edge/op/function target, stale/false source
anchor, incomplete slice, duplicate graph identity, diagnostic/hash mismatch,
ambiguous alternatives, false exact mapping, and unsupported avoidable/inherent
classification.

### Offline invocation and closure

- run two independent GPU-hidden scratch invocations and one durable invocation,
  all through the pre-load guard harness with `PYTHONDONTWRITEBYTECODE=1`;
- require identical canonical payload digests excluding only declared run
  timestamps and output paths;
- reproduce all 36 raw GraphDef bindings and final-diagnostic entity/delta
  summaries before interpreting the 18 autodiff graphs;
- partition every declared target exactly once into `mapped_exact`,
  `enumerated_causally_ambiguous`, or `missing_or_incomplete`;
- require exact graph neighborhood and source/framework anchors for every exact
  mapping and bounded alternatives for every ambiguity;
- rerun every entry check and compare only the authorized write set against the
  snapshot; run whitespace checks only on that set and reject any unexpected
  path under the finite stems;
- write the result with run manifest, decision table, inference-status table,
  candidate-versus-direction distinction, and post-run red team;
- freeze the result bytes and obtain an independent detached review agreeing on
  entity coverage, source anchors, ambiguity states, causal language, artifact
  hashes, and boundary compliance; and
- only after that agreement, draft and review exactly one valid next subplan.

## Forbidden Claims And Actions

- Do not edit algorithm, benchmark, builder, evaluator, existing test, frozen
  evidence, R3/R2 authority, budget, lease, or work-root bytes.
- Do not create a fixture, trace a function, invoke a selected method, initialize
  TensorFlow, compile/run XLA, enumerate/use GPU, run Gate C/Phase 7, benchmark,
  install packages, access the network, or launch a subprocess from phase code.
- Do not call a construction `avoidable`, `inherent`, erroneous, benign, or a
  source bug from attribution, names, local source proximity, or artifact
  mutation. Artifact mutations test the validator, not TensorFlow causation.
- Do not resolve analytical constants by analogy or propose an evaluator
  exception.
- Do not erase entity insertions, deletions, order, edges, ops, calls, values, or
  function bodies through normalization.
- Do not claim memory/performance repair, XLA viability, method ranking,
  production/default readiness, HMC/posterior correctness, or scientific
  validity.
- Reviewers are read-only and cannot authorize runtime, human, model-file,
  funding, product/default, release, or scientific boundaries.

## Exact Next-Phase Handoff Conditions

Define these predicates after detached result agreement:

- `complete`: the target partition is exact, no target is
  `missing_or_incomplete`, and every required check/review agrees;
- `repair_eligible`: `complete`, every target is `mapped_exact`, and a bounded
  local semantics-preserving counterfactual hypothesis is nominated; and
- `discriminator_eligible`: `complete`, not `repair_eligible`, and a bounded
  safe next offline discriminator or separately reviewed framework-proof
  question exists.

The predicates below are mutually exclusive and exhaustive:

| Localization result after detached result agreement | Exact handoff |
| --- | --- |
| `repair_eligible` | Draft/review the exact `autodiff-source-counterfactual-repair` subplan. It must test the hypothesis prospectively and may reject it. The analytical lane remains unresolved; Gate B remains rejected and Gate C/runtime remains blocked. |
| `discriminator_eligible` | Draft/review the exact `autodiff-attribution-discriminator` subplan containing the smallest safe offline discriminator or separately defined framework-proof gate. This includes every complete non-repair-eligible result for which such a discriminator exists, including all-exact attribution with no bounded local repair hypothesis. The analytical lane remains unresolved; Gate B remains rejected and Gate C/runtime remains blocked. |
| Not `repair_eligible` and not `discriminator_eligible` | Write the localization blocker result and stop. This includes every incomplete/invalid/drifted/failed-guard result and every complete non-repair result for which no safe discriminator exists. No next subplan and no runtime authority. |

No next subplan may be drafted from an unreviewed result. Every valid next
subplan must state its objective, inherited entry conditions, artifacts,
checks/tests/reviews, evidence contract, forbidden actions, exact handoff, and
stops. Successful state:
`AUTODIFF_ATTRIBUTION_RESULT_REVIEWED_NEXT_SUBPLAN_REVIEWED_RUNTIME_STILL_BLOCKED`.

## Stop Conditions

- plan or result review fails to converge within five material rounds;
- any required input, source, descriptor, authority, protected, or review byte
  drifts;
- any target is `missing_or_incomplete` or appears in multiple states;
- decoder parity, deterministic rerun, negative control, static scan, runtime
  guard, or scoped cleanliness check fails;
- target worker/process state becomes ambiguous or another lane changes an
  in-scope read-only path;
- the result reviewer does not agree on coverage and causal boundaries; or
- continuing requires source/evaluator edits, new trace/XLA/GPU/Gate C runtime,
  package/network/model/funding/default/product/release changes, or scientific
  authority.

## Skeptical Pre-Execution Audit

Status: `ROUND3_REVISED_REVIEW_PENDING`.

| Risk | Assessment |
| --- | --- |
| Wrong baseline | Controlled by the final conservative diagnostic and all 36 preserved raw GraphDefs, not the superseded pre-red-team artifact. |
| Proxy promotion | Controlled: names, counts, and source proximity are explanatory; complete entity partition and graph/source anchors are required. |
| Missing stops | Controlled by the hard `missing_or_incomplete`, drift, guard, mutation, deterministic, review, and authority stops. |
| Unfair comparison | Controlled: all three dimensions and both parameter counts/batch patterns are covered within the same autodiff method. |
| Hidden assumptions | Exposed: GraphDef-to-source mapping can be nonunique; attribution cannot prove avoidability/inherence; descriptor decoding may be incomplete. |
| Stale context | Controlled by exact parent/result/review, source, framework, descriptor, R3/R2, and authority hashes before and after. |
| Environment mismatch | No TensorFlow runtime is imported; GPU is hidden; installed TensorFlow 2.20.0 text/descriptors are provenance-bound read-only inputs. |
| Artifact fitness | Full graph neighborhoods plus exact source/framework anchors can localize observed construction and nominate hypotheses; they cannot validate a repair or performance claim. |

Offline implementation begins only after exact `VERDICT: AGREE`. Gate B remains
rejected and Gate C/runtime remains blocked regardless of review.
