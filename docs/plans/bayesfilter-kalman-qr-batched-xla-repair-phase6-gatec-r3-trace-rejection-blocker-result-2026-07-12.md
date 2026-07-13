# Phase 6 Gate C R3 Trace-Rejection Blocker Diagnostic Result

Date: 2026-07-13

Status: `OFFLINE_DIAGNOSTIC_PASSED_MIXED_CAUSES_GATE_B_REJECTED_GATE_C_BLOCKED`

Parent subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-trace-rejection-blocker-subplan-2026-07-12.md`.

Parent subplan SHA-256:
`0afb2d033e62035c032a82db48ffce949a72776109ac9c37c97f28a04f3b3929`.

Final agreeing subplan review SHA-256:
`9defdc687ad2d8ec265a71c8d68745c1c1e2eba9976fa62837a6816fbfadba0a`.

## Result

The offline preserved-GraphDef diagnostic passed its engineering evidence
contract and classified the R3 structural rejection as `mixed_causes`.

- The analytical method has invariant named entity sets, entity counts, order,
  operation/dependency structure, and function topology across the six `P/B`
  records at every dimension. Its 46 noncanonical positional rejections are all
  inside `Const` fields, its 692 accepted coordinates are all declared `B/P`
  shape dimensions, and its 40 differing integer constants track `B`, `P`, or
  both. However, all 40 constants per cohort retain ambiguous consumer
  boundaries under the conservative reachability analysis. The analytical
  classification is therefore `undetermined`, not a proved evaluator false
  positive and not an approved normalization exception.
- The autodiff method has genuine batch-dependent graph construction in every
  dimension. Named top-level nodes grow with `B`, generated gradient functions
  change, and some function bodies add/remove nodes. It also contains
  axis-correlated constants and unresolved residuals. Each autodiff cohort is
  classified `mixed_causes`.

The large positional rejection totals overcount independent semantic changes,
but they were not purely evaluator artifacts. Gate B remains rejected, Gate C
runtime remains blocked, and no XLA or Kalman target ran.

## Durable Evidence

Diagnostic artifact:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_2026-07-12.json`.

| Field | Value |
| --- | --- |
| SHA-256 | `637273af37ed2606b9bd0bc4868a1719a65ad17d89d94ab018e5678082fb25ff` |
| Byte count | `5780057` |
| Canonical diagnostic payload SHA-256 | `30a2753246d4c86a6952268fad5a49d8e77991084f4100a45d1eca051c710cd7` |
| State | `passed` |
| Classification | `mixed_causes` |
| Graph coverage | 36 of 36 unique identities, all raw bytes bound |
| Strict validation | all ten checks `true` |
| Gate B | `still_rejected` |
| Gate C | `blocked` |
| Runtime authorized | `false` |

Strict check manifest:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_check_manifest_2026-07-12.json`, byte count `10942`, SHA-256
`fcc5ad1ea9cf6f06ce4ae0dc83e0da00d2ce08b9bfb805f3220748cf9bc1e54f`.

## Cohort Findings

| Dimension | Method | Stable entity counts in `P50/B1,B4,B16,P150/B1,B4,B16` | Axis-correlated integer constants | Ambiguous consumer constants | Classification |
| ---: | --- | --- | ---: | ---: | --- |
| 10 | analytical | `1377,1377,1377,1377,1377,1377` | 40 | 40 | `undetermined` |
| 20 | analytical | `1377,1377,1377,1377,1377,1377` | 40 | 40 | `undetermined` |
| 30 | analytical | `1377,1377,1377,1377,1377,1377` | 40 | 40 | `undetermined` |
| 10 | autodiff | `1399,1417,1431,1399,1417,1431` | 112 | 71 | `mixed_causes` |
| 20 | autodiff | `1399,1431,1431,1399,1431,1431` | 112 | 71 | `mixed_causes` |
| 30 | autodiff | `1399,1431,1431,1399,1431,1431` | 112 | 71 | `mixed_causes` |

At `d=10`, autodiff `B=4` adds 16 named
`gradient_tape/add*/Shape{,_1}` nodes relative to `B=1`. `B=16` adds those 16
plus 14 `zeros_{5,7,9,11,13,14,15}/{Const,shape_as_tensor}` nodes. At
`d=20/30`, both `B=4` and `B=16` add all 30 nodes relative to `B=1`. Generated
`while_body_*_grad_*` and `while_cond_*_grad_*` function identities/bodies also
change. These are named entity and function-body differences, not positional
index cascades.

The exact source operation that causes TensorFlow to generate those nodes is
not yet established. The next phase localizes them without editing source or
running a new target trace.

## Baseline Reproduction

The current production evaluator was recomputed before stable-key analysis and
reproduced all six existing count pairs exactly:

| Dimension | Method | Accepted | Rejected |
| ---: | --- | ---: | ---: |
| 10 | analytical | 692 | 47 |
| 20 | analytical | 692 | 47 |
| 30 | analytical | 692 | 47 |
| 10 | autodiff | 255 | 18998 |
| 20 | autodiff | 357 | 12252 |
| 30 | autodiff | 357 | 12252 |

Raw counts remain descriptive coordinate counts, not independent defect counts
and not method rankings.

## Required Checks

Final compile command passed in `0.061549363` seconds. Its empty durable log has
SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The final GPU-hidden focused suite passed `11 passed in 1.79s`; outer elapsed
time was `3.144399157` seconds. Durable log SHA-256:
`b6fc5e0b2a2e217683a2277fb19aff365537a246eb8e02a102728aa3ac702954`.

The suite covers entity insertion/deletion/order, op substitution, data edge
and output index, control edge, function target/body/signature/return/control
return, dtype/device/shape/list attributes, `Const` values, cross-function
numeric consumers, duplicate keys, incomplete coverage, full-lattice axis
classification, residual partition completeness, deterministic payload hashing,
and an AST no-target call-surface contract. The only subprocess call permitted
by the AST test is exact `git rev-parse HEAD`.

Three independent GPU-hidden offline invocations completed with wall times
`41.74633153900504`, `41.071354658924975`, and `41.71222290198784` seconds.
Their JSON file hashes differ because timestamp/output-path fields remain
visible, but all three canonical evidence payloads equal
`30a2753246d4c86a6952268fad5a49d8e77991084f4100a45d1eca051c710cd7`.

Final whitespace, per-untracked-file `git diff --no-index --check`, strict JSON,
schema/content, GraphDef coverage, ledger, live authority, budget/lease,
namespace, no-worker, protected hash, and R2 hash checks passed. The R3 root
still contains only `import_discovery.json`, `budget_state/`, and `trace/` with
108 preserved files.

## Visible Repair Loop

The first scratch implementation recomputed downstream consumer reachability
quadratically for every integer constant. Codex interrupted it before it emitted
a JSON artifact and preserved its 330-byte log at
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_attempt0_interrupted_2026-07-12.txt`, SHA-256
`384739f73471fbc320263bc2afe7ed5792f429dcf34e09853d1389a349a1a612`.
The implementation was repaired by indexing each graph once and traversing
only differing constants; focused tests were rerun.

An initial valid three-run diagnostic then reported analytical axis data without
proving that every positional residual and consumer path fit that explanation.
The post-run red team rejected that attribution. The complete pre-red-team
artifact is preserved as superseded evidence at
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_rejection_diagnostic_pre_redteam_2026-07-12.json`, byte count `5134362`, SHA-256
`06d9f976d642dd9a972cc651dacc44dc36b6c74e2dcd3cb13e79a55843602847`.
It is not promotion evidence.

The diagnostic was strengthened to require a complete positional residual
partition, parse FunctionDef `node:output:index` edges, and mark any unproved
constant consumer boundary unsafe. Compile, the expanded tests, and all three
offline invocations were rerun. Only the final hashes in the check manifest
carry the evidence burden.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Commands | Exact five final commands in the strict check manifest |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; CPython 3.13.13; Linux 6.8.0-124-generic x86_64 |
| CPU/GPU | CPU protobuf/JSON analysis; `CUDA_VISIBLE_DEVICES=-1`; GPU deliberately hidden; no enumeration |
| XLA/TF32 | not initialized or invoked; TF32 not queried |
| Data/fixture | `N/A`; immutable preserved R3 GraphDefs |
| Seeds | `N/A`; deterministic diagnostic |
| Wall time | final diagnostic runs 41.7463 s, 41.0714 s, and 41.7122 s; final compile 0.0615 s; final tests 3.1444 s |
| Inputs | R3 trace SHA `7444fb41...`; parent subplan SHA `0afb2d03...`; all frozen identities in manifest |
| Outputs | diagnostic SHA `637273af...`; check manifest SHA `fcc5ad1e...`; durable logs in manifest |
| Plan | parent subplan path above |
| Result | this path; self-hash intentionally omitted to avoid recursion and bound by detached review/close record |
| Trust/claim boundary | offline engineering attribution only; no repository-default GPU/XLA evidence |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep Gate B rejected and Gate C blocked; select offline source/local-graph attribution | Passed: exact counts, complete graph/token coverage, stable-key comparison, eleven mutation/boundary tests, and deterministic payload digest | Autodiff structural promotion veto established; analytical consumer safety unresolved | Which source/autodiff operations produce the extra gradient/zeros nodes, and whether analytical constants can ever receive a narrow reviewed rule | Inspect source and preserved graph neighborhoods under the dedicated next subplan | No XLA viability, numerical failure, memory/performance repair, ranking, GPU, production, HMC, posterior, or scientific claim |

## Inference-Status Table

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Gate B structural veto remains. Autodiff entity/function growth is supported. No numerical/XLA/memory veto was tested. |
| Statistically supported ranking | None; this was deterministic structural analysis, not a stochastic or timing comparison. |
| Descriptive-only differences | Positional rejection counts, graph/entity counts, artifact sizes, and offline wall times. |
| Default readiness | Not established; Gate C, GPU, Phase 7, default, and production gates remain blocked. |
| Next evidence needed | Source-anchored mapping from extra autodiff nodes/functions to exact operations plus an analytical consumer-path proof or explicit unresolved record. |

## Candidate Versus Direction

The current strict gate candidate rejected correctly under its reviewed rules.
The diagnostic weakened the hypothesis that all thousands of autodiff
coordinates represent independent defects, but it also rejected the hypothesis
that the entire Gate B failure is a positional evaluator artifact. This result
does not reject Kalman mathematics, TensorFlow as the repository backend, the
repaired runtime harness, or the broader batched-XLA repair direction.

## Post-Run Red Team

The strongest alternative explanation for the autodiff entity growth is benign
TensorFlow gradient specialization required by fixed batch shapes rather than a
repairable source bug. That still violates the current exact-topology gate and
may affect compile size, but it must not be called erroneous until source/local
graph attribution distinguishes framework necessity from avoidable source
construction.

The strongest alternative explanation for the analytical constants is that
they are shape-only metadata that the initial no-`Const` rule intentionally
rejected. The diagnostic did not prove their transitive consumer boundary, so
no exception is proposed here. Evidence that would overturn this result is a
complete source/consumer proof plus adversarial mutation tests, or a source
mapping showing that the named autodiff insertions are artifacts of unstable
generated naming with no entity/dependency difference after a stronger
non-lossy correspondence. The weakest current evidence is semantic causation;
the strongest is byte identity, entity presence, deterministic reproduction,
and mutation sensitivity.

## Forbidden Conclusions

No new target graph, fixture, selected method, XLA compile/runtime, GPU,
memory, performance, scalability, numerical parity, HMC, posterior, default,
production, release, or scientific evidence was produced. The original memory
and performance problems remain unresolved.

## Handoff

Selected next subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-subplan-2026-07-13.md`.

It authorizes only offline source and preserved-GraphDef localization. Gate C
runtime remains blocked.
