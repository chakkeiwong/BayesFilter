# Phase 6 Gate B R3 Mixed-Format Binding Repair Subplan

Date: 2026-07-12

Status: `DRAFT_REVIEW_REQUIRED_NO_TARGET_AUTHORITY`

Supervisor/executor: Codex in the current conversation.

Reviewer: Claude Opus at max effort, read-only and advisory. Claude cannot
authorize runtime or any human, model-file, funding, product/default, release,
or scientific boundary.

Execution authorization: the user's 2026-07-12 instruction, "do as you
suggested," authorizes Codex to execute this already-described bounded repair
and Gate B sequence. Codex must record at the final runtime audit that this
authorization is still current and that scope has not expanded. No plan review,
proposal, attestation, audit, or handoff token grants execution permission. If
the requested command or scope changes materially, stop for fresh user
authorization.

Parent result:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-trace-pilot-result-2026-07-12.md`.

## Phase Objective

Repair the real-proposal runtime binding contradiction exposed by `r2`, ensure
all deterministic trace-launch bindings and the initial trace ledger are
constructed and validated before opening durable budget state, preserve `r2`
as immutable failed-harness evidence, and create a fresh disjoint `r3`
authority. Only after
focused and consolidated checks, a repair result, bounded reviews, a strict
`r2` archive, a fresh proposal review, detached attestation, and a final
skeptical audit may Codex use the still-current user authorization to run the
exact Gate B target command again.

This is a harness-schema and transaction-order repair. It does not alter Kalman
mathematics, target grid, methods, tolerances, schedules, device, dtype, JIT,
timeouts, or the 3045-second budget.

## Entry Conditions Inherited From R2

All conditions are conjunctive:

- `r2` proposal SHA-256 remains
  `187594f66a2a87e237d697d52085318731efea986e077b2972e7a1cf44b46359`;
- `r2` attestation SHA-256 remains
  `4fa7b0cbef59c826804dc9e156fffe9660aabdb05d26b0d88458893b89d566cd`;
- `r2` proposal review remains at SHA-256
  `c4c7055eb5c416310867c831e93fb3cb111d1da76c6d2b3e6cf25b409d940acf`,
  strength `claude_opus_max`, ending `VERDICT: AGREE`;
- `r2` skeptical audit remains at SHA-256
  `b95d56f31c5a5f47eba601df6695070c98e12ebac70e0ebc0f43f207709b566b`;
- `r2` import discovery, budget state, and lease remain at SHA-256
  `8ae6086bd6b8bbebd7bf236536a80cb6b8befa993a9e686801c451e8fec4c8ac`,
  `a4cc284b64d6527a7357171f4c47395a7f29f7fed7e50b15563257feae09390f`,
  and `ae711efe84056ae416d5fe2d2d40751b91afaa7f3a2e3530f095fb501a03b456`;
- the `r2` budget state remains truthfully `running`, its lease remains
  `released`, trace/pilot outputs remain absent, and no target worker survives;
- the immutable `r1` archive remains at SHA-256
  `caacd7144a0e6b7767487d7cc3a48145702983487ac1ab6885f5f97ba2f9607a`;
- protected Kalman source hashes remain exactly:
  `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b`,
  `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57`,
  and `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401`;
- the parent `r2` result exists and no `r3` work root, proposal, attestation,
  trace, pilot, budget state, lease, or child artifact exists;
- this exact subplan receives bounded read-only `VERDICT: AGREE` before edits.

## Required Artifacts

### Immutable R2 Lineage

- strict archive manifest:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_invalid_harness_archive_2026-07-12.json`;
- the archive inclusion set is exactly the following ordered paths, with exact
  absolute path, byte count, SHA-256, role, format, and immutable disposition:
  1. `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-harness-repair-subplan-2026-07-12.md`;
  2. `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-harness-repair-result-2026-07-12.md`;
  3. `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-repair-subplan-codex-review-round3-2026-07-12.md`;
  4. `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-repair-result-review-round2-2026-07-12.md`;
  5. `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r1_invalid_harness_archive_2026-07-12.json`;
  6. `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_2026-07-12.json`;
  7. `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-budget-review-round1-2026-07-12.md`;
  8. `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_attestation_2026-07-12.json`;
  9. `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-final-skeptical-runtime-audit-2026-07-12.md`;
  10. `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-trace-pilot-result-2026-07-12.md`;
  11. `/tmp/kalman_qr_phase6_cpu_xla_gateb_r2/import_discovery.json`;
  12. `/tmp/kalman_qr_phase6_cpu_xla_gateb_r2/budget_state/gate_b-4807a429ce935c95392f6af62266ef53a2e6165c8b8cc5e0cb415ba80fb26096.json`;
  13. `/tmp/kalman_qr_phase6_cpu_xla_gateb_r2/budget_state/gate_b-4807a429ce935c95392f6af62266ef53a2e6165c8b8cc5e0cb415ba80fb26096.json.lease`;
- the archive absence set is exactly the `r2` trace output, pilot output,
  `trace/`, `pilot/`, `children/`, and `progress/` paths plus any entry other
  than `import_discovery.json` and `budget_state/` directly under the `r2` work
  root; the `budget_state/` directory must contain exactly the state and lease
  above;
- the archive scope is the complete direct `r2` launch-generation inventory,
  not a repository-wide or transitive authority inventory; referents embedded
  in proposal/source manifests remain protected by their own digest validators;
- the archive records process absence, root-cause localization, and explicit
  nonclaims;
- archive validation must rehash every listed file and must never mutate `r2`.

### R3 Repair And Review

- this subplan;
- preserved subplan review rounds at
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-bindings-repair-subplan-review-round<N>-2026-07-12.md`;
- final agreeing subplan review record:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-bindings-repair-subplan-review-final-2026-07-12.md`;
- repair result:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-mixed-format-bindings-repair-result-2026-07-12.md`;
- preserved repair-result review rounds at
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-bindings-repair-result-review-round<N>-2026-07-12.md`;
- final agreeing repair-result review record:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-bindings-repair-result-review-final-2026-07-12.md`;
- focused/consolidated logs under
  `/tmp/kalman_qr_phase6_gateb_r3_binding_repair/`.

### Fresh R3 Authority And Runtime Namespace

- work root: `/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/`;
- import discovery:
  `/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/import_discovery.json`;
- proposal:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_2026-07-12.json`;
- proposal review:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-budget-review-round1-2026-07-12.md`;
- detached attestation:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_attestation_2026-07-12.json`;
- final skeptical runtime audit:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-final-skeptical-runtime-audit-2026-07-12.md`;
- trace ledger:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_census_2026-07-12.json`;
- pilot ledger:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_cpu_xla_pilot_2026-07-12.json`;
- fresh budget state/lease and child artifacts under the `r3` root;
- final runtime result:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-trace-pilot-result-2026-07-12.md`.

Exactly one branch-specific next subplan and its fixed final review may be
created after the runtime result:

- valid Gate B pass:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-scalar-remaining-subplan-2026-07-12.md`
  and
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-scalar-remaining-subplan-review-final-2026-07-12.md`;
- valid structural trace rejection:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-trace-rejection-blocker-subplan-2026-07-12.md`
  and
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-trace-rejection-blocker-subplan-review-final-2026-07-12.md`;
- `r3` harness invalidity:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r4-harness-repair-subplan-2026-07-12.md`
  and
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r4-harness-repair-subplan-review-final-2026-07-12.md`.

The two non-selected branch path pairs must remain absent. None of these review
paths can authorize execution.

The fresh `r3` proposal authority inputs are exactly this ordered list:

1. `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_invalid_harness_archive_2026-07-12.json` (`json`);
2. `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-mixed-format-bindings-repair-result-2026-07-12.md` (`markdown`);
3. `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-bindings-repair-subplan-review-final-2026-07-12.md` (`markdown`);
4. `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-bindings-repair-result-review-final-2026-07-12.md` (`markdown`).

No round-specific review file may substitute for either fixed final review path.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can mixed-format path/digest authority inputs be preserved byte-for-byte in runtime bindings without requiring semantic JSON parsing, while all semantically parsed evidence remains strict and deterministic trace-launch binding/ledger validation occurs before budget mutation? |
| Candidate mechanism | Remove only the authority-input `strict_json` requirement; retain valid/present blob and exact proposal path/digest equality; construct and validate trace bindings/initial ledger before budget lease/open; parent revalidation immediately before spawn plus a child-entry snapshot guard before target work; fresh `r3` namespace and authority. |
| Exact baseline | Immutable `r2` proposal and failed pre-ledger state, plus the diagnostic showing one JSON and three Markdown authority inputs. |
| Expected failure mode | Validation is weakened for proposal/attestation/Phase 4/5/predecessor evidence; text tampering passes; path order is ignored; another hidden binding predicate remains; budget opens before deterministic trace preflight completes; an authority input mutates between preflight and spawn; `r2` is accidentally resumed or overwritten. |
| Primary criterion | Real-format one-JSON/three-Markdown bindings validate; byte/path/order/digest mutations fail; parsed artifacts remain strict; initial trace ledger is constructed before budget open; parent-detected boundary mutation prevents spawn; child-entry drift prevents fixture/method/trace work; drift present at terminal revalidation invalidates evidence fail-closed; focused and consolidated gates pass; `r2` hashes remain unchanged; fresh `r3` authority validates. |
| Promotion veto | Any broad blob-validation weakening, accepted missing/tampered/reordered authority input, lost semantic JSON requirement, `r2` mutation/reuse, budget mutation on deterministic preflight failure, protected drift, failed tests, invalid authority, or surviving process. |
| Continuation veto | Corrupt/ambiguous immutable evidence, unsafe ownership, inability to create a disjoint namespace, five non-converging review rounds for one blocker, or a new human/package/network/model/default/scientific boundary. |
| Repair trigger | Any focused/consolidated/review/archive/namespace/authority failure. |
| Explanatory only | Test duration, file size, traceback length, and later target trace/runtime metrics. |
| Must not conclude | No target correctness, XLA viability, memory/performance repair, scalability, GPU, HMC/posterior, default/production, or scientific claim from this repair. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Exact main question above. |
| Baseline/comparator | Byte-identical `r2` failed authority and synthetic plus real-format binding fixtures. |
| Primary pass/fail criterion | Mixed-format bytes are accepted only through exact blob/path/digest binding; semantically parsed artifacts remain JSON; deterministic trace binding and initial-ledger validation precede budget mutation; parent-detected mutation cannot spawn, child-entry-detected mutation cannot begin target work, and drift present at terminal revalidation cannot yield valid evidence. |
| Veto diagnostics | Missing/reordered/substituted/tampered input, path/digest mismatch, semantic JSON weakening, stale namespace, `r2` drift, budget opened on preflight failure, invalid proposal/attestation, or process survival. |
| Explanatory only | Durations, counts, and target metrics. |
| Not concluded | No method/backend outcome or memory/performance claim. |
| Preserved result | Archive, tests/logs, repair result/reviews, fresh authority, and branch-specific runtime artifacts. |

## Repair Design

1. In `_phase6_bindings_valid`, keep `phase6_blob_record_valid` and
   `present is True` for every authority input, but do not require
   `strict_json`. Exact ordered `{path, sha256}` projection must continue to
   equal proposal inputs. Do not alter the blob schema or hash/base64 checks.
2. Keep strict JSON mandatory for proposal, attestation, Phase 4/5 evidence,
   and runtime predecessors because their contents are semantically inspected.
3. Add real-format tests containing one JSON archive and three Markdown files.
   Prove construction, ledger creation, and launch revalidation pass; mutate
   bytes, path, digest, order, presence, base64, and JSON-required categories
   independently and prove fail-closed behavior.
4. Before budget lease/open, construct the trace bindings, run
   `_phase6_bindings_valid`, run `phase6_revalidate_launch_authority`, construct
   the complete initial trace ledger in memory, and validate its closed schema.
   Pass that exact preconstructed ledger into execution or use an equivalent
   pure preflight token bound to its canonical digest. Do not create any trace
   output before budget open.
5. After lease acquisition and budget open, persist the preconstructed initial
   trace ledger, then rehash/revalidate proposal, attestation, Phase 4/5
   evidence, all four authority inputs, source/runtime identity, and schedule
   immediately before each child spawn. A mutation detected by this final
   parent guard launches no child, durably terminalizes/prunes the ledger as
   common invalidity, and closes the budget command truthfully.
6. Add one integration test where deterministic preflight fails and assert no
   budget directory/state/lease, trace, pilot, or child artifact is created.
   Add a separate parent-boundary test that mutates one authority input after
   budget open but before the final parent guard and proves zero spawn, durable
   common-invalidity trace evidence, released lease, and closed budget state.
7. Trace-static pre-budget validation includes proposal/attestation, the trace
   schedule, four authority inputs, Phase 4/5 evidence, current source/runtime,
   exact live supervisor argv, namespace, and initial trace ledger. Pilot-static
   proposal/schedule checks are also validated, but a complete pilot binding and
   ledger cannot be constructed before runtime because its exact predecessor is
   the final trace ledger. Construct pilot bindings only after the final trace,
   bind that predecessor, and repeat launch-authority validation before any
   pilot spawn.
8. Add a child-entry authority-snapshot guard whose exact inputs are carried in
   the reviewed child command or a digest-bound launch manifest. Before fixture
   construction, selected-method construction, concrete-function tracing, XLA,
   or Kalman execution, the child reads and validates proposal, attestation,
   Phase 4/5 evidence, the four authority inputs, source/runtime identity, and
   its schedule row against the parent-validated digests. A child-entry mismatch
   exits with typed common-invalidity evidence and no target work. Test mutation
   after the parent's final validation but before this child guard and assert a
   subprocess may exist but fixture/method/trace/XLA invocation counts remain
   zero.
9. After the child guard captures exact governance bytes and source/runtime/
   schedule identity in memory, target work must not reopen proposal,
   attestation, Phase 4/5 evidence, or the four authority-input files. The child
   uses its loaded modules, in-memory schedule/config, and captured identity.
   Audit this no-reopen boundary in source/tests. Parent terminal revalidation
   detects drift that is present at that checkpoint, rejects the child evidence
   as common invalidity, prunes remaining work, and prevents a valid Gate B
   artifact. Test persistent terminal drift separately from the zero-target-work
   child-entry test.
10. An ABA mutation introduced and restored entirely between checkpoints is not
    detectable by path hashing and is outside the claimed guarantee. These runs
    assume cooperative repository writers; any independently observed
    concurrent write to an in-scope source or authority path is a continuation
    veto, even if bytes are later restored. Do not claim continuous filesystem
    immutability or adversarial-writer resistance.
11. Preserve `r2`; change active constants and closed CLI paths only to `r3`.
   Proposal inputs must equal the exact ordered four-path list above.

## Editable Baseline Hashes

These paths may change only within the write set and must have before/after
hashes plus an approved-diff disposition in the repair result:

| Path | Pre-edit SHA-256 |
| --- | --- |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `d299c5ffed10a30c93a0e8685b47bfe9433efb66cf722fe64b53bf4ef39651fc` |
| `scripts/kalman_qr_benchmark_contract.py` | `1f81eb7453ec58a4b0b0227897fb9252fed2ffdacf6fa0d8c0392e5f056f783b` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `197bd4357e42aa308dfd44c99c47fbb948534593c4d141a9c4ba073a45c1c07a` |
| `tests/test_kalman_qr_phase6_cpu_xla_gates.py` | `7251aa289ff2d1041a51d5d79f2c478686b000f31e213fc35b6af3063bc22f20` |
| `tests/test_kalman_qr_phase6_gatea_runtime_controls.py` | `1d29f3063a47967e542fd3c97730caf4c0eadd258ecd6f3feeb3dac4263298d0` |
| `tests/test_kalman_qr_phase6_import_discovery_cli.py` | `0f8a97c0f4e20cd1958d83ae9b9e5dbf36972efc6e6626c7c2902d8dd104b830` |
| `tests/test_kalman_qr_phase6_gateb_runtime_repair.py` | `d035c68471a7e2493ef9c9c2afce07920ff6d9c85ac7d138c63bc6964677f51c` |

## Write Set

- `scripts/kalman_qr_benchmark_contract.py`;
- `scripts/benchmark_kalman_qr_parameter_count_scaling.py`;
- `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`;
- `tests/test_kalman_qr_phase6_cpu_xla_gates.py`;
- `tests/test_kalman_qr_phase6_gatea_runtime_controls.py` only if required for
  budget-order integration coverage;
- `tests/test_kalman_qr_phase6_gateb_runtime_repair.py`;
- `tests/test_kalman_qr_phase6_import_discovery_cli.py` only for active-path
  updates;
- newly created `r3` repair subplan review/result/review logs, exact `r2`
  archive, and fresh `r3` proposal/review/attestation/audit/runtime/result
  artifacts at the fixed paths declared above;
- exactly one selected branch-specific next-subplan/review pair at the fixed
  paths above. Immutable `r1`/`r2` inputs and both non-selected branch pairs are
  excluded from the write set even though they are named in this subplan.

Protected `bayesfilter/linear/*.py` algorithm files and unrelated lanes are
read-only.

## Required Checks, Tests, And Reviews

- strict `r2` archive reconstruction/revalidation and no-worker check;
- focused tests covering every Repair Design item;
- exact compilation and scoped `git diff --check`/whitespace checks;
- consolidated GPU-hidden Gate A/Gate B harness suite, with target CPU-XLA node
  explicitly deselected until fresh authority;
- source-shape check proving only authority inputs lost the JSON requirement;
- protected and exact immutable-lineage hashes before/after tests; editable
  paths use the baseline table and approved before/after diff ledger;
- repair result with run manifest, decision/inference tables, evidence ledgers,
  alternative explanation, and nonclaims;
- bounded review of this subplan before edits and bounded review of the repair
  result after checks; patch and rerun focused checks on `REVISE`, maximum five
  material rounds for one blocker;
- fresh `r3` proposal construction only from an absent work root, strict
  proposal review, detached attestation, authority validation, and final
  skeptical runtime audit before target execution.

## Skeptical Pre-Execution Audit

Status: `PASS_FOR_SUBPLAN_REVIEW_ONLY`.

- Wrong baseline: `r2` is preserved as failed-harness evidence; no target
  outcome is inferred.
- Proxy promotion: local binding tests cannot establish XLA or memory repair.
- Missing stops: immutable drift, broad validation weakening, budget mutation
  on preflight failure, stale namespace, failed tests/reviews, and unsafe
  process ownership are explicit stops.
- Hidden assumption: Markdown is not trusted semantically; it is trusted only
  as exact bytes already bound by reviewed path/digest authority.
- Stale context: every exact archive-inclusion path and protected hash is
  rechecked before edits, after tests, before proposal, and after runtime;
  editable paths use the declared before/after diff ledger.
- Environment: local checks are GPU-hidden/no-target; later Gate B remains the
  reviewed GPU-hidden CPU-XLA diagnostic lane.
- Artifact fitness: real-format binding tests answer this repair; only final
  trace/pilot ledgers can answer Gate B.

## Forbidden Claims And Actions

- Do not alter, resume, close, delete, rename, overwrite, import, or reuse any
  `r1` or `r2` runtime artifact, budget state, lease, proposal, or attestation.
- Do not weaken proposal, attestation, Phase 4/5 evidence, runtime predecessor,
  blob hash/base64, exact path/digest/order, or live-source validation.
- Do not treat Markdown bytes as parsed semantic authority beyond their exact
  reviewed digest.
- Do not change target grid, methods, ordering, budgets, timeouts, tolerances,
  device, threads, dtype, JIT, or promotion criteria.
- Do not run target trace/XLA, Gate C, GPU, HMC, or comparisons before fresh
  `r3` authority and exact audit `PASS`.
- Do not claim memory/performance repair or use review as runtime authority.
- Do not claim that proposal/review/attestation/audit/handoff artifacts authorize
  execution; only the still-current user instruction plus Codex's final scope
  check authorizes the exact bounded command.

## Exact Next-Phase Handoff Conditions

All are conjunctive:

- `r2` archive strictly validates and all immutable hashes match;
- mixed-format binding and pre-budget validation-order repairs pass focused and
  consolidated checks without protected drift;
- repair result and agreeing implementation/result review are frozen;
- fresh `r3` proposal/review/attestation/authority/audit all pass;
- the final audit records that the user's existing execution authorization is
  still current, no scope expanded, and no artifact is being treated as
  permission; only then may Codex run the exact proposal-bound command once;
- runtime branch is classified exactly as valid pass, valid structural trace
  rejection, or harness invalidity; only valid pass can refresh executable Gate
  C planning.

Pre-runtime handoff:
`GATE_B_R2_ARCHIVED_R3_BINDINGS_REPAIRED_AUTHORITY_VALID_RUNTIME_READY`.

## Stop Conditions

- Any immutable `r1`/`r2` or protected hash drifts.
- Correctness requires accepting unbound or tampered authority bytes.
- Deterministic preflight cannot be moved before durable budget mutation without
  weakening lifecycle accounting.
- Required local checks or review fail after bounded repair.
- Fresh `r3` namespace is nonempty or not disjoint.
- Authority, process ownership, or required evidence cannot be validated.
- Any independently observed concurrent write to an in-scope authority or
  source path during runtime, including an observed ABA write/restore.
- Continuing requires new package/network/credential/model-file/funding,
  product/default/release, scientific-claim, or other human authority.

## Mandatory Close Sequence

1. run required local/artifact checks;
2. write the `r3` repair result;
3. construct and review fresh `r3` authority or write a blocker;
4. after authorized runtime, write the exact branch result;
5. draft/refresh and review the branch-specific next subplan;
6. advance only when exact handoff conditions hold.
