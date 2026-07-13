# Phase 6 Gate C R3 Autodiff Attribution Discriminator Result

Date: 2026-07-13

Status: `STRUCTURAL_PARTITION_PASSED_NO_UNIQUE_REPAIR_POINT_GATE_B_REJECTED_GATE_C_BLOCKED`

Parent subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-attribution-discriminator-subplan-2026-07-13.md`.

Parent subplan SHA-256:
`ce14737c2bee978e4fc1fe6134c5b306d6bc6c39de95b78658b46b49c5a8247b`.

Final agreeing subplan review SHA-256:
`24c74a874ef62034b607fbca5c2fddbb69647d3ec6a50d4ac95e2ae2f9af0e9e`.

## Result

The guarded offline discriminator passed its engineering evidence contract. It
reconstructed all `904` targets and `12,316` parent observations, expanded the
`2,386` nested integer-constant occurrences, and assigned exactly one allowed
structural state to all `14,270` atomic occurrences. No occurrence was missing,
duplicated, or `unresolved_invalid`.

This is not a repair result. The partition did not establish one unique local
control point, construction origin, root cause, or counterfactual effect. Gate B
therefore remains rejected, Gate C/runtime remains blocked, and the original
memory/performance problems remain unresolved.

The first-match handoff predicate is `framework_proof_question`. Exactly `420`
targets form uniform installed-framework-source candidate cohorts: `344`
uniform reverse-while targets and `76` uniform unresolved constant/shape
targets. Their uniformity permits one bounded offline source question; it does
not prove that TensorFlow requires, originates, or incorrectly creates them.

## Durable Evidence

Discriminator artifact:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_attribution_discriminator_2026-07-13.json`.

| Field | Value |
| --- | --- |
| SHA-256 | `8494478c0c9857818d72222c7a5966039083a8c16c44d35ab1dab24f74c22265` |
| Byte count | `62349977` |
| Canonical payload SHA-256 | `5ba99bfca3c3afa28d6593d48ca6a85e428e992d663b0b68d7f87e22bad4b2e3` |
| State | `passed_complete_structural_region_partition` |
| Classification | `graph_region_ownership_only_causation_unresolved` |
| Targets / observations | `904` / `12,316` |
| Nested / atomic occurrences | `2,386` / `14,270` |
| Invalid occurrences | `0` |
| Unique local control point | `false` |
| Next branch | `autodiff_framework_proof` |
| Gate B / Gate C | `still_rejected` / `blocked` |
| Runtime authorized | `false` |

Strict check manifest:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_autodiff_attribution_discriminator_2026-07-13_check_manifest.json`.
Byte count `15081`; SHA-256
`55fc3687f0a772faa5b83f793d1281a02e477c900976f96b8e239db7adf81449`.

## Structural Partition

| Atomic state | Count |
| --- | ---: |
| `structural_boundary_ambiguous` | 3,122 |
| `forward_while_setup_not_parameter_descendant` | 2,184 |
| `reverse_while_call_or_function` | 2,138 |
| `forward_while_call_or_function` | 2,016 |
| `post_reverse_vjp` | 1,716 |
| `constant_or_shape_origin_unresolved` | 1,270 |
| `local_pre_forward_while` | 942 |
| `score_path_not_reverse_descendant` | 468 |
| `reverse_vjp_setup` | 366 |
| `forward_value_projection` | 24 |
| `cross_region_composite` | 24 |
| `unresolved_invalid` | 0 |
| **Total** | **14,270** |

All 18 autodiff GraphDefs had exactly one structurally identified forward
while and one structurally identified reverse while. The artifact preserves the
complete boundary, call/function binding, region-slice, target, occurrence, and
source-anchor ledgers. It classified by structural predicates and explicit
precedence, not semantic name fragments.

All preserved `GraphDef.debug_info` and `NodeDef.experimental_debug_info`
fields were empty across the 18 autodiff graphs and their 25,430 nodes. This is
an observed coverage result, not evidence that any state is safe, necessary,
or attributable to a particular source line.

Target aggregates include `344` uniform reverse-while targets, `76` uniform
constant/shape-unresolved targets, `129` uniform structural-boundary-ambiguous
targets, `12` uniform cross-region composites, and `2` mixed-region targets.
Only the first two cohorts satisfy the reviewed framework-proof-question
predicate.

## Checks

The pre-load AST boundary scan passed over exactly the discriminator, guard,
and focused test. It found no forbidden import, runtime surface, subprocess,
network call, device use, undeclared write, or semantic-name classifier. Log
SHA-256:
`25b1c31cda53451042aa9e27da2675d2332cd7c4b697b2d0427ff8501ef1f3b6`.

Three explicit `py_compile.compile(..., cfile=..., doraise=True)` calls passed
and wrote only the declared `discriminator.pyc`, `guard.pyc`, and `test.pyc`.
Compile-log SHA-256:
`65bfafc84670bc03b0a77c964b8af7772cc9ad6dbebe4841af50ef97b968a014`.

The exact guarded focused suite passed `9 passed in 107.67s`. It covered:

- guard-before-load behavior and authority boundaries;
- semantic-name perturbation and saved-forward reverse binding;
- mutually exclusive first-match states and complete path witnesses;
- function ownership, call/argument/return binding, and lossless debug data;
- zero/multiple reverse-candidate ambiguity;
- edge, call-output, function, count, duplicate, target, and witness mutations;
- canonical-payload exclusions and deterministic hashing; and
- complete real-corpus reconstruction and partition validation.

Focused-log SHA-256:
`d0989226c5bf408210ead3e60d170662cdfd153c9c5e9ff8784189b40efaf242`.

Two independent scratch runs and one durable run produced the same canonical
payload digest and identical partitions. Their raw JSON hashes differ only in
the four declared run-metadata fields: start time, finish time, wall time, and
output path.

Closure revalidated the reviewed plan, review, parent result/localization,
local source, installed TensorFlow source, descriptor, R3 authority, budget,
lease, and protected algorithm hashes. The R3 root remains exactly
`import_discovery.json`, `budget_state/`, and `trace/`; the trace inventory is
still 108 files; forbidden runtime directories remain absent; and no target
worker survives.

The authorized snapshot recorded 430 unrelated dirty-worktree paths with
digest `c17205537cee17f3ee91f7892ee397f3a1ef8ca090764adfa44c03923206bb2d`.
After excluding this phase's exact authorized paths, closure has one additional
unrelated path:
`docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-implementation-codex-substitute-review-2026-07-12.md`.
That path belongs to the other lane. It was not read as evidence, modified,
staged, cleaned, or represented as phase work. No protected in-scope path
drift occurred.

## Visible Repair Loop

Plan review Round 1 required disjoint state predicates and witnesses, a
saved-forward reverse binding that does not assume an edge from the concrete
value wrapper, and an explicit bounded in-memory protobuf fixture exception.
The subplan was visibly repaired; Round 2 and exact-hash confirmation agreed.

The first focused fixture renamed the contract-bound `parameters_batch` input.
The fixture was repaired to perturb only non-bound semantic names, then focused
checks were rerun.

The first complete-corpus test rebuilt graph adjacency for every occurrence and
was interrupted as non-viable. Boundary and adjacency structures were cached by
graph identity; the full guarded suite then passed.

The first static scan rejected a forbidden semantic-name fragment used only in
explanatory source-anchor text. The prose was replaced with exact source-anchor
wording, and the scan was rerun before evidence.

The first scratch artifact repeated large graph ledgers per atomic occurrence
and reached approximately 356 MB. That scratch-only output was deleted. Shared
ledgers were deduplicated without changing any classification, and two scratch
runs plus the 62,349,977-byte durable run were regenerated. Only the final
passing hashes carry evidence.

Claude was not retried after the managed external-disclosure gate rejected the
call before review. The converged subplan review used a fresh native Codex
reviewer with `codex_substitute_weaker` provenance. That review cannot grant
runtime, source-edit, framework-necessity, memory/performance, product, or
scientific authority.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Commands | Exact final commands, command contracts, logs, hashes, and outputs are in the strict check manifest |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`; CPython 3.13.13; Linux 6.8.0-124-generic x86_64 |
| CPU/GPU | CPU-only offline protobuf/JSON analysis; `CUDA_VISIBLE_DEVICES=-1`; no device enumeration |
| XLA/TF32 | not initialized or invoked; TF32 not queried |
| Data/fixture | immutable Gate B R3 36-GraphDef corpus |
| Seeds | `N/A`; deterministic offline analysis |
| Wall time | focused suite `107.67s`; scratch runs `76.4106s` and `77.2730s`; durable run `77.3632s` |
| Plan | parent subplan path above |
| Result | this path; self-hash intentionally omitted and bound by detached review |
| Trust boundary | offline engineering structural attribution only |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close the discriminator as an engineering pass but reject repair eligibility | Passed: exact total, mutually exclusive, deterministic partition with complete witnesses and zero invalid occurrences | No implementation/evidence veto fired; causal and runtime boundaries remain active | Structural ownership does not establish construction origin, necessity, avoidability, or counterfactual effect | Draft and review exactly the offline installed-source framework-proof subplan | No Gate B pass, repair, framework defect/necessity, memory/performance improvement, runtime feasibility, ranking, or readiness |

## Inference Status

| Evidence class | Status | Interpretation |
| --- | --- | --- |
| Hard veto screen | Passed for the offline discriminator only; no drift, invalid occurrence, nondeterminism, guard failure, or unauthorized runtime |
| Statistically supported ranking | Not applicable; this deterministic structural phase compares no stochastic candidates |
| Descriptive-only differences | State counts, cohort sizes, path lengths, graph sizes, debug-field emptiness, and elapsed times explain the artifact but cannot promote a repair |
| Default-readiness | Not established; Gate B is rejected and Gate C/runtime is blocked |
| Next evidence needed | A reviewed offline derivation connecting the exact 420-target cohorts to installed TensorFlow control-flow branches, with explicit necessity/non-necessity and uncertainty states |

## Candidate Versus Direction

The current local-repair candidate failed: the discriminator found no unique
local control point. That rejects immediate source-counterfactual repair; it
does not reject the broader repair direction.

The next framework-proof phase is designed to discriminate the surviving
uncertainty. It may show that the exact installed-source branch constructs a
cohort under stated preconditions, that more local binding evidence is needed,
or that installed source cannot decide the question. A candidate failure in
that phase is not a research-direction veto unless source drift, inconsistent
bindings, invalid evidence, or another stated continuation veto fires.

The analytical constant lane remains separately unresolved. Nothing in this
autodiff partition resolves it by analogy.

## Post-Run Red Team

The strongest alternative explanation is that uniform structural ownership is
an artifact of this frozen GraphDef corpus while the decisive construction
choice lies in a local VJP wrapper, an installed framework branch, or an
interaction between them. The current artifact cannot distinguish those
causes.

Evidence that would overturn the handoff is an exact witness establishing one
uncontested local control point for a declared cohort, or a proof that the
installed source question cannot be answered without new runtime evidence.
Neither is present.

The weakest evidence is the jump from uniform cohort membership to selecting
installed-source inspection. That selection is justified only as the smallest
offline discriminating question. It is not evidence for framework necessity or
a framework defect.

## Forbidden Conclusions

- The memory and performance problems are not fixed.
- Gate B did not pass; Gate C, XLA, GPU, and runtime execution remain blocked.
- Structural ownership does not prove origin, root cause, avoidability,
  inherence, error, benignness, or an evaluator exception.
- Empty debug metadata does not prove absence of provenance or safety.
- The 420-target cohort does not prove a TensorFlow requirement or defect.
- No numerical equivalence, compile feasibility, scalability, ranking,
  default/production readiness, HMC/posterior correctness, or scientific
  validity claim is supported.

## Handoff

After detached result review agreement, draft and review exactly:

`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-autodiff-framework-proof-subplan-2026-07-13.md`.

That subplan must remain offline installed-source analysis. It must bind the
exact result/artifact/manifest hashes and the eight installed-source anchors in
`backprop.py`, `gradients_util.py`, and `while_v2.py`; state all required phase
fields; distinguish construction sufficiency from necessity; and forbid source
edits, TensorFlow import/runtime, new traces, XLA/GPU, Gate C, and
memory/performance claims.

No framework-proof implementation or analysis begins until that exact subplan
has its own converged review agreement. If result review rejects target or
occurrence coverage, structural witnesses, causal boundaries, hashes, or
write/runtime boundaries, patch this result visibly and rerun focused closure
checks. Stop after five material rounds for the same unresolved blocker.
