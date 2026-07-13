# Phase 6 Gate C R3 Autodiff Structure Localization Result

Date: 2026-07-13

Status: `LOCALIZATION_PASSED_COMPLETE_CAUSALLY_AMBIGUOUS_GATE_B_REJECTED_GATE_C_BLOCKED`

Parent subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-structure-localization-subplan-2026-07-13.md`.

Parent subplan SHA-256:
`88db6519ca3d1a668ef9565506b539c1bd4cd672f000424c35a4be6d5581a949`.

Final agreeing subplan review SHA-256:
`ab8bc7613ec1b547cba58ccdb3419cf46ec37b6af2be94b267e46b55e331d2eb`.

## Result

The offline preserved-GraphDef localization passed its engineering evidence
contract, but it did not establish a unique source or framework cause for any
target. It partitioned `904` unique targets with exact graph occurrence
evidence:

| Entity kind | Unique targets |
| --- | ---: |
| Top-level node | 295 |
| Function | 9 |
| Function-body node | 444 |
| Graph order | 12 |
| Integer constant | 144 |
| **Total** | **904** |

Every target is `enumerated_causally_ambiguous`; none is `mapped_exact`, and
none is `missing_or_incomplete`. The artifact retains `12,316` occurrence
observations, including both graph sides for changed/order entities, exact
function summaries/callers, node producers/consumers, 36 raw GraphDef bindings,
12 descriptor files, 17 source/distribution files, three local source anchors,
and eight installed-framework anchors.

The correct handoff is therefore
`autodiff_attribution_discriminator`, not a source counterfactual repair. Gate B
remains rejected, Gate C/runtime remains blocked, and the analytical lane
remains unresolved.

## Durable Evidence

Localization artifact:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_structure_localization_2026-07-13.json`.

| Field | Value |
| --- | --- |
| SHA-256 | `ee2903381039f7cf15a4ec5112304232ae138eebacef8e0858da1fda5f7452c1` |
| Byte count | `68112660` |
| Canonical payload SHA-256 | `f29bea2cb4f26e6e26f1606149652eb98f4145099a3568e7874d67824f166e1c` |
| State | `passed_complete_causally_ambiguous` |
| Classification | `complete_attribution_inventory_causal_ambiguity_retained` |
| Target count | `904` |
| Coverage | `904` ambiguous; `0` exact; `0` incomplete |
| Gate B | `still_rejected` |
| Gate C | `blocked` |
| Runtime authorized | `false` |

Strict check manifest:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_structure_localization_2026-07-13_check_manifest.json`, byte count
`12192`, SHA-256
`8aa6f6079a38f0e66e90a8b5964f25f146b5fdeb94585510d7d26133d7028bbf`.

## Constant Coverage

The frozen final diagnostic contains 144 differing integer-constant records per
autodiff dimension, not 32 total residuals:

| Dimension | `B` | `P` | `B+P` | Presence change | Total records |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 96 | 8 | 8 | 32 | 144 |
| 20 | 96 | 8 | 8 | 32 | 144 |
| 30 | 96 | 8 | 8 | 32 | 144 |

The normalized constant target keys collapse identical cross-dimension names
to 144 unique targets while preserving a separate observation for every
dimension and graph occurrence. No constant is called shape-only safe,
avoidable, inherent, benign, erroneous, or an evaluator exception here.

## Checks

The pre-load AST scan passed over exactly the localizer, guard, and focused test.
It found no forbidden import/call, no test write or temp fixture, one exact
guarded localizer output write, and only the reviewed post-guard dynamic-load
exceptions. Log SHA-256:
`393403d15be2821627dfddce7f49408dc9613e4a743e5ac7f207d805197f59ce`.

Three explicit `py_compile.compile(..., cfile=..., doraise=True)` calls passed
and produced only the authorized `localizer.pyc`, `guard.pyc`, and `test.pyc`
paths. Compile-log SHA-256:
`7d829761b4e1e87cb63cae770a6223146a3687e9f0bca79bfe31e2b6fe6fe96d`.

The exact guarded focused suite passed `9 passed in 49.24s`. It covered:

- truncated, missing, and unknown descriptor data;
- missing/duplicate graph entities and duplicate graph identities;
- changed edge, operation, function target, and function-body entities;
- stale/false source anchors and incomplete/one-sided slices;
- diagnostic/hash mismatch and deterministic payload behavior;
- bounded ambiguity, false exact mapping, and forbidden avoidable/inherent
  claims;
- process/import/device/write guard violations; and
- the complete real 36-GraphDef partition.

Focused-log SHA-256:
`bfb33233694435130d53ec996a23ab825839bab87dca1fa72cacdfcce8a1f5df`.

Two independent scratch localizations and one independent durable localization
all produced canonical payload digest
`f29bea2cb4f26e6e26f1606149652eb98f4145099a3568e7874d67824f166e1c`.
Their partitions, checks, and decisions are identical. Raw JSON hashes differ
only because the four declared timestamp/wall-time/output-path fields remain
visible.

Closure revalidated every frozen parent, diagnostic, R3/R2 authority, source,
framework, descriptor, and protected algorithm hash. The R3 work root still
contains exactly `import_discovery.json`, `budget_state/`, and `trace/`; trace
still contains 108 files; budget remains closed; lease remains released; the
two XLA pilot records remain `not_launched:trace_gate_not_passed`; and no target
worker survives.

The shared worktree remains dirty. After removing only authorized localization
paths from porcelain status, the unrelated lane remains exactly 428 paths with
the entry digest
`d5d248105bfaa996c625474246f9b83b2d4086a5aa42f0dbe8efa48ac606652c`.
Those paths were neither modified nor cleaned by this phase.

## Visible Repair Loop

The first static-scan implementation falsely treated read-only `Path.open` and
string `.replace` as writes and omitted the explicitly authorized post-guard
pytest import. It failed before any subject import/compile. The scanner was made
mode-aware and rerun to pass.

The first compile wrapper contained a syntax typo in its inline manifest and
stopped before producing any bytecode. It was corrected; the three exact cfiles
were then produced and bound.

The first guarded pytest setup exposed pytest's default `/dev/null` file logging
handler. Rather than authorizing any test write, the guard replaced that handler
with an in-memory stream after guarded pytest import. Static scan, compile, and
tests were rerun.

The second guarded suite passed seven tests and exposed two real defects: the
device guard did not match `/dev/nvidia0`, and the localizer expected the stale
label `base64-standard-v1` instead of the frozen corpus's exact
`base64-rfc4648`. Both were repaired. Static scan, compile, and all nine tests
were rerun before any evidence localization.

During closure, the supervisor briefly materialized a filtered Git-status digest
and a later no-index diff-check capture at undeclared `/tmp` paths. Both were
deleted; the status comparison and result whitespace check were rerun without
creating another file. The reviewed scratch root is exact. Only the final
passing hashes carry the evidence burden.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Commands | Exact final command contracts, exit codes, logs, and outputs in the strict check manifest |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`; CPython 3.13.13; Linux 6.8.0-124-generic x86_64 |
| CPU/GPU | CPU-only protobuf/JSON analysis; `CUDA_VISIBLE_DEVICES=-1`; no device enumeration |
| XLA/TF32 | not initialized or invoked; TF32 not queried |
| Data/fixture | immutable Gate B R3 36-GraphDef corpus |
| Seeds | `N/A`; deterministic offline analysis |
| Wall time | focused suite `49.24s`; durable localization `49.6237s`; scratch runs same bounded offline class |
| Plan | parent subplan path above |
| Result | this path; self-hash intentionally omitted and bound by detached review |
| Trust/claim boundary | offline engineering localization only |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the complete localization inventory; keep Gate B rejected and Gate C/runtime blocked | Passed: all 904 declared targets appear exactly once, zero incomplete, deterministic three-run digest, exact provenance, and all negative controls pass | Source-repair promotion veto remains because zero targets have a unique exact causal anchor; no runtime continuation authority exists | Which bounded discriminator can distinguish local broadcast/VJP specialization from TensorFlow reverse-while/zero-shape generation without target runtime or source edits | Detached result review, then draft/review exactly one offline `autodiff-attribution-discriminator` subplan if the result agrees | No avoidable/inherent construction, source bug, evaluator exception, memory/performance repair, XLA/GPU viability, ranking, default/production readiness, HMC/posterior correctness, or scientific validity |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No missing entity, provenance drift, guard violation, failed mutation, or deterministic mismatch remains. Gate B's prior structural veto remains active. |
| Statistically supported ranking | None; this is deterministic structural evidence, not a stochastic or timing comparison. |
| Descriptive-only differences | Target/observation counts, artifact sizes, and offline wall times. |
| Default readiness | Not established; Gate B, Gate C, GPU, XLA, production, and scientific gates remain blocked. |
| Next evidence needed | A reviewed offline discriminator or framework-proof gate that can uniquely eliminate one of the bounded alternatives for a material target family. |

## Candidate Versus Direction

The localization candidate succeeded at complete enumeration but failed to
produce exact causal attribution. That blocks a source counterfactual repair
nomination; it does not reject the broader Kalman QR memory/performance repair
direction. The observed ambiguity is precisely the input to the next
discriminating phase, not evidence that the underlying memory/performance issue
is fixed, inherent, unfixable, or caused by TensorFlow.

## Post-Run Red Team

The strongest alternative explanation is that the 904-target inventory is a
complete catalog of correlated manifestations, not 904 independent causal
constructions. The unique-key partition intentionally deduplicates across
pairs/dimensions, but it cannot infer causal independence from names or graph
proximity. A smaller number of local broadcast/VJP operations, TensorFlow
reverse-while specializations, or both could explain many targets.

Evidence that would overturn this result is a missing target, a lossy function
or order slice, a stale source/framework anchor, a non-deterministic canonical
payload, or an exact mapping that survives the negative controls with one unique
source anchor. None was observed. The weakest evidence is causal attribution;
the strongest is frozen-byte provenance, exact entity/occurrence coverage,
three-run determinism, and adversarial validator/guard behavior.

## Forbidden Conclusions

No target builder or selected method was called. TensorFlow was not imported.
No new trace, fixture, XLA compile/runtime, GPU enumeration/use, Gate C run,
benchmark, numerical parity test, memory measurement, performance measurement,
scalability comparison, HMC/posterior check, default change, production claim,
release action, or scientific claim occurred. The original memory and
performance problems remain unresolved.

## Handoff

This result must receive detached agreeing review before a next subplan is
drafted. If review agrees, the only valid branch is:

`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-attribution-discriminator-subplan-2026-07-13.md`.

That next phase remains offline-only unless separately reviewed. Gate B remains
rejected and Gate C/runtime remains blocked.
