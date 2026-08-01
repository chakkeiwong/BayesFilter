# Phase 6 Subplan: Separate Authority And Tiny Actual-Target CPU/XLA Smoke

Date: 2026-07-11

Status: `ATTEMPT1_IMPLEMENTATION_FAILURE_REPAIR_QUALIFICATION_NO_RUNTIME`

## Phase Objective

Preserve the completed Phase 5 preflight, add a narrow smoke-only runtime
authority mechanism and dedicated-output launcher, and, only after a separate
exact human approval, run one tiny actual-target two-worker CPU/XLA HMC
mechanics smoke. Phase 6 tests initialization, compilation, worker persistence,
finite transition outputs, coordinate mapping, deterministic seed/topology
wiring, and artifact writing. It does not test convergence or authorize
serious Phase 7.

## Entry Conditions Inherited From Phase 5

- The Phase 5 result and this subplan pass fresh independent review.
- The Phase 5 terminal manifest re-opens and verifies the exact V2 config,
  adoption record, and preflight bytes.
- The active V2 config remains exactly:
  - embedded artifact hash
    `sha256:bd127c2eb4e554c241a9f38111b5d832cb8ae9132429332abec724b9b2d39a6a`;
  - exact file SHA-256
    `9270ec429a4b49e19f5ac6492e146bb1010e07c4ea0aa17600294e6c41db7ca8`;
  - byte count `12560`;
  - `runtime_authority=false`.
- The Phase 5 terminal manifest remains exactly:
  - embedded artifact hash
    `sha256:32e97d69595029423fbe4e22f714c3596c8a4c3d9b3aabd6b2c600d279355bc0`;
  - exact file SHA-256
    `41426951d25d02a7efbbd595e6edb0a81039fb297ce3ebd05e1695207fda4871`;
  - byte count `1196`.
- Live V2 preflight still reports all typed identity and integrity checks true,
  `runtime_authority=false`, and `runtime_executed=false`.
- The historical V1 config remains byte-identical and its named validator still
  fails exactly with `public final kernel hash mismatch`.
- One separately approved smoke launch was attempted on 2026-07-12. Its
  permanent claim consumed the V2 authority before worker initialization. The
  attempt failed with `runtime_error:BrokenProcessPool` because the retained
  child source loader did not set `module.__file__`; the controller therefore
  raised `NameError: name '__file__' is not defined` while importing. It
  produced zero worker PIDs, no burn-in or retained check, no diagnostic, no
  transition, and no private sample bytes. No serious runtime has run.
- The V2 approval, authority, claim, result, progress, output manifest, log,
  and zero-byte reservations are immutable consumed attempt-1 evidence. They
  must never be deleted, replaced, rewritten, or reused.
- Unrelated dirty LEDH/QR work remains outside scope and must be preserved.

The approved Phase 5 baseline-migration statement is not Phase 6 runtime
authority. The V2 smoke approval is also no longer authority for another
launch because its one allowed launch was consumed. Attempt 2 requires a new
V3 terminal proposal manifest and the exact new human decision defined below.
Silence, a request to continue, approval of this document as prose, repetition
of the V2 statement, or any earlier approval is insufficient.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Bind authority to the exact terminal Phase 5 manifest, V2 config, adoption record, preflight, transition hash, and smoke-execution hash; verify live identities again immediately before workers. |
| Preflight mistaken for smoke | Record preflight as already passed and runtime-free. The smoke result must carry worker PIDs and finite runtime evidence; preflight cannot satisfy the Phase 6 runtime criterion. |
| Serious run accidentally authorized | Keep V2 `runtime_authority=false`; add a separate closed smoke-authority record whose only mode is `smoke`, and require `run_phase7(..., smoke=True, smoke_authority=...)`. Reject missing authority, `smoke=False`, unknown fields, changed command, or changed artifacts before output/worker creation. |
| Historical or Phase 5 evidence mutated | Never edit V1, V2, the adoption record, preflight, or Phase 5 manifest. New Phase 6 artifacts point backward; completed Phase 5 artifacts never point forward. |
| Shared outputs overwritten | Smoke uses dedicated repository artifact paths, distinct from `burnin_sampling.json`, `burnin_sampling_progress.json`, and the serious retained-samples path. The launcher fails if any declared smoke output already exists. |
| Static approval replayed | The authority record binds one exact permanent launch-claim path. After all pre-runtime checks, the launcher atomically creates that claim with `O_CREAT|O_EXCL`, writes and fsyncs its authority/output binding, and never deletes it, including after crash or failure. An existing or malformed claim means the authority is consumed. |
| Writable or in-process claim replay | Create the durable claim at owner-read-only mode `0400`, retain an `O_RDONLY` descriptor, and consume a process-local prepared-context capability at each transition from verified preparation to reserved outputs to runtime entry. A caller-constructed context is rejected before output or worker creation. |
| Failure writes before authority | Authority verification, full live V2 preflight, output alias/nonexistence checks, and atomic claim creation occur outside the controller's output-writing `try`. Pre-claim failure creates no log, result, progress, sample, worker, or claim. |
| Output path replaced after preflight | Before claiming, open and pin every repository-relative parent directory component with `O_DIRECTORY|O_NOFOLLOW`, record parent device/inode identity, and reject any replacement. After the durable claim, reserve every log/progress/result/manifest/private/emergency output with `openat`, `O_NOFOLLOW`, and `O_EXCL`; retain descriptors for all writes and revalidate both parent and final-component device/inode identity before every update. |
| Launcher or control-flow failure after claim | Reserve distinct infrastructure-failure and infrastructure-manifest descriptors before ordinary outputs. Wrap reservation, redirection, controller, log sealing, and manifest construction/writing in one post-claim supervisor. Before hashing emergency evidence, flush only streams whose descriptors still name the reserved log inode, detach those streams, and fsync the held log descriptor so traceback/shutdown output cannot mutate referenced bytes. Preserve a valid primary result; bind exact held bytes for every reserved ordinary output plus a separate path-integrity flag; never overwrite the primary result with launcher-failure semantics. Preserve the original `BaseException`, seal emergency evidence best-effort with bounded idempotent retries, then re-raise control-flow exceptions. |
| Worker failure leaves a peer hung | On success, ordinary failure, timeout, or `BaseException`, signal every worker before waiting for any worker, share one teardown deadline bounded by the global wall-time cap and five seconds, kill survivors, and call only `shutdown(wait=False)`. Never use `shutdown(wait=True)` or replace the original control-flow exception. |
| Smoke emits serious semantics | Smoke has a separate strict result/failure schema, decision, progress schema, and ordered nonclaims. It must never emit `PASS_PHASE7_TO_PHASE8_APPROVAL_BOUNDARY` or copy V2's pre-runtime `not Phase 7 smoke or serious execution` nonclaim. |
| Parent/worker TOCTOU | The proposal pins a deterministic transitive closure of static BayesFilter imports from the Phase 6 controller, authority module, benchmark driver, and three command entrypoints, plus the eight exact Phase 2-6 review tests and Python executable. Unrelated repository Python files are outside this lane and may coexist. Each child rechecks the exact bound inventory, rejects any unbound BayesFilter or `docs.*` module that actually loads, then reopens every governed source reference and reconstructs the expected transition identity before runner construction or compilation. Added or changed files inside the bound closure are a veto; unrelated-lane additions are not. Parent success alone is insufficient. |
| Proxy promoted | Smoke checks mechanics and finite diagnostics only. The unchanged serious thresholds are reported as explanatory values, but neither top-level nor per-parameter `passed` applies those thresholds. R-hat, ESS, acceptance, runtime, and smoke pass cannot establish convergence, recovery, ranking, or serious readiness. |
| Environment mismatch | Establish CPU hiding and the exact five thread variables at each Phase 6 command entrypoint before importing BayesFilter, NumPy, TensorFlow, or TFP; verify observed rather than hard-coded parent values; freeze two spawned workers, two chains per worker, sequential compilation, versions, float64, XLA/JIT, threads, and smoke counts in the adopted smoke execution identity. |
| Invalid scientific stop | A smoke failure blocks serious runtime but does not reject the target or scientific direction unless it proves a target/math invalidity. Classify implementation, environment, artifact, and candidate failures separately. |
| Missing stop | Stop on absent approval, artifact drift, authority mismatch, preflight mismatch, output collision, nonfinite value, XLA/JIT fallback, worker/process failure, timeout, invalid artifact, or review blocker. Always stop after smoke assessment before serious Phase 7. |
| Consumed authority reused after repair | Treat every attempt-1 path as protected immutable evidence, use distinct V3 proposal and attempt-2 authority/output paths, and require a new exact manifest-bound approval. Hold shared locks as cooperative exclusion, but also pin each file's first post-lock device/inode/mode/link/owner/size/mtime/ctime signature and reject every later descriptor or pathname signature that differs; this detects an advisory-lock-ignoring writer that restores exact bytes when the mutation remains visible in retained inode/path metadata. This does not claim protection against privileged metadata forgery or a rewrite the filesystem does not expose through bytes or the pinned signature. The V2 authority and claim remain terminally consumed even though the defect is localized and fixable. |
| Historical evidence cannot be reverified after current-path changes | Keep proposal parsing archival and self-contained while keeping current command/path/inventory/byte requirements in the live candidate verifier. A historical proposal may verify its own terminal output but must fail the current attempt-2 authorization gate. |

Audit verdict:
`PASS_FOR_ATTEMPT1_PRESERVATION_AND_LOCAL_ATTEMPT2_REPAIR_QUALIFICATION_ONLY`.
The localized failure did not invalidate the target, data, math, typed
identity, Phase 5 evidence, XLA mechanics, or HMC transition because execution
stopped during child module import. Actual attempt-2 HMC runtime remains
blocked until a V3 proposal is frozen, reviewed, and separately approved.

The first complete independent Codex pre-runtime audit returned
`VERDICT: REVISE`. It found unbounded teardown, an incomplete implementation
inventory, accidental per-parameter R-hat/ESS gating, asymmetric emergency
retry, missing preclaim and child-race coverage, pathname-replacement evidence
that could not verify, incomplete serious-path protection, asserted rather
than observed parent thread settings, a writable claim inode, and a
caller-forgeable runtime context. The repair loop addressed each finding and
added focused no-runtime tests. The original review snapshot is not a passing
review because source bytes changed during repair; a fresh frozen independent
Codex review remains mandatory before proposal materialization.

## Research Intent Ledger

| Field | Declaration |
| --- | --- |
| Main question | Can the approved refreshed typed transition execute through the exact Phase 7 two-worker CPU/XLA path for tiny bounded counts and emit finite, internally valid mechanics artifacts? |
| Candidate/mechanism | V2 transition `sha256:10d9a9d2...f16d6a` under smoke execution `sha256:fc85f9b1...6ac604`. |
| Exact baseline/comparator | The immutable Phase 5 V2 config, adoption record, preflight, terminal manifest, and the existing Phase 7 controller smoke semantics. No sampler-ranking comparator exists. |
| Expected failure mode | Worker/XLA initialization, replay reconstruction, process persistence, finite target/log-accept/sample state, coordinate mapping, or artifact writing fails. |
| Promotion criterion | Smoke-only engineering pass: exact authority and live preflight pass; two persistent workers initialize; four chains execute 4 burn-in and 8 retained transitions per chain; XLA/JIT and CPU hiding are confirmed; required runtime values and diagnostics are finite; dedicated artifacts validate. |
| Promotion veto | Any required mechanics/artifact check fails. This veto prevents serious Phase 7 planning from advancing automatically. |
| Continuation veto | Missing/mismatched approval, Phase 5 drift, output collision, live typed/integrity mismatch, nonfinite state/sample/target/log-accept, available nonzero divergence telemetry, XLA/JIT fallback, process failure, timeout, public/private leak, or invalid result artifact. |
| Repair trigger | A localized authority, launcher, worker, XLA, serialization, resource, or artifact defect with intact Phase 5 evidence. Repair must be separately reviewed and rerun only under still-valid or renewed exact approval as declared by the authority record. |
| Explanatory diagnostics | Acceptance, first/warm call timing, worker PIDs, diagnostic values, compile trace counts, and elapsed time. |
| Must not be concluded | No convergence, posterior recovery, calibrated uncertainty, serious-run readiness by smoke alone, ranking, production/default, GPU, DSGE, NeuTra, or scientific validity. |

## Attempt 1 Failure And Repair Contract

The immutable consumed attempt-1 evidence is:

| Artifact | Embedded hash | Exact file SHA-256 | Bytes | Mode |
| --- | --- | --- | ---: | ---: |
| Original proposal | `sha256:57b9434a54c3c2ac9c67ddf57a54caaf00feb9dcf9910a0fb41b03e44bad653a` | `16df0bdb62f45e9b2c304a7030c5c7d08497720f42c43dbf489b694dc9497d0d` | 193504 | `0600` |
| Original proposal manifest | `sha256:9db02019042769750a731dbc849746c5e3380a8883e03167553d7829acf0f1c7` | `b31d93a568bd30458c56bc87d9eca17ea73ea3579f973591e00d0a9a80696c3c` | 848 | `0600` |
| V2 attempt-1 proposal | `sha256:6ab3167abd521c6c41fc481cfed75d4ffae613cc672d49019bedbf8490639ced` | `f8c1d301186e9b1df390dbc4248c95932737bf2a7d8f50c6af985129bc7755c8` | 30416 | `0600` |
| V2 attempt-1 proposal manifest | `sha256:e8e913e005423da1da87bfa1f5a8e832f7b32d8a8a90172aa81fefea8607bc3b` | `29dbba924ce899189e178d624ddc26c1fdfaaf46674244c3547f44c7ee591527` | 847 | `0600` |
| Attempt-1 authority | `sha256:6206888214a63b5b0c56a776d27f2a520880b0645ebf7e2afc280f000cfe4c58` | `e6be84a875ded5b880eef7d7445e645aefdf86c61dcbc5b4ad744d6d1bec126c` | 1712 | `0600` |
| Attempt-1 permanent claim | `sha256:6cddfe35278935f9be30a65bf2b481ac53eaaba59818de56855b55b53d73b2ad` | `d1424c0cf4bc616bcab1de7efda29a9f0c465d0496cb8c1eb6d720faba8d54d3` | 1886 | `0400` |
| Attempt-1 failure result | `sha256:68bfd9078c9f187874d2a2334f8353fbc5d4f4736e52dd58c42764782bbcd275` | `28f7866d6e2fc1419a010b70a9b9e4f9f45da3f2c7c0f259c48b47fc9bf09fe9` | 5668 | `0400` |
| Attempt-1 terminal progress | `sha256:01ae35db4edf24e150ded5f11f1e1af8949091f657c7a34437f10207a725ed5a` | `55a973c4df278ce137793df767ff80864f17e5c90a130c3a59b4fa2cecfed24c` | 949 | `0400` |
| Attempt-1 output manifest | `sha256:714eb599b2ddc70607ff94a6b2ee963266618b38802687152707fc89a2634099` | `c144596ac8f0d2be33e6cf65d4c60d64cc2703b0990ee37005707c6a4f773900` | 3836 | `0400` |
| Attempt-1 infrastructure-failure reservation | N/A | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | `0400` |
| Attempt-1 infrastructure-manifest reservation | N/A | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | `0400` |
| Attempt-1 private-sample reservation | N/A | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | `0400` |
| Attempt-1 log | N/A | `6dee7ec170811c18c87fc1ee3fa0397213325363a5c1e4e2c294874cc5e7bf80` | 2564 | `0400` |

A first fresh independent review of the attempt-2 repair returned
`VERDICT: REVISE`: the paths were protected against aliasing, but V3 candidate
verification did not mechanically recheck the immutable attempt-1 hash/size/
mode ledger. The repair adds one shared fail-closed integrity verifier covering
all 13 original-proposal, V2, and attempt-1 files. It exact-byte and
mode checks each regular non-symlink file, semantically revalidates both
proposal manifests plus the terminal output bundle, confirms the exact
implementation-failure classification, and rereads all evidence after semantic
verification. The V3 proposal builder, authority builder, and launcher all
enter through the live candidate verifier and therefore run this gate before
any new artifact or claim write. A fresh review on the repaired bytes remains
required.

The second fresh review also returned `VERDICT: REVISE`. It found that the
first verifier could miss a same-size overwrite or concurrent mode change,
accepted hard links, released its checks too far before irreversible claim
creation, lacked direct race/semantic tests, and documented only part of the
13-file ledger. The repaired design now opens every file and parent without
following symlinks, requires owner/group identity and exactly one hard link,
holds nonblocking shared locks and pinned descriptors, compares complete stat
signatures and two exact reads, and retains the evidence session across each
proposal/authority write and from preclaim through launcher teardown. It
rechecks immediately before and after claim creation, before/during/after
output reservation, on every secure output descriptor access, during prepared
context validation, after the first progress write, and immediately before
process-pool creation. A reservation-time mismatch consumes the authority and
may leave only the permanent claim plus already-created empty reservations; it
must create no worker and write no nonempty attempt-2 output bytes. A later
mismatch may preserve valid output bytes that were durably completed before
detection, such as the initial preflight progress record, but it must cause no
further output write after detection. The controller must tear down any workers
already created, rethrow the typed drift without controller failure
classification, and the launcher must not seal an infrastructure terminal.
Direct tests cover same-size and concurrent overwrite, mutate-and-restore
between verification calls, size, mode, symlink, hard-link, lock conflict,
semantic and terminal-classification drift, plus preclaim, postclaim,
pre-reservation, final-reservation, controller-entry, and post-progress outcomes
with descriptor closure.
Fresh full-gate and independent review on these bytes remain required.

The repair is limited to:

1. set `module.__file__` to the bound retained source filename before executing
   a bundled child module;
2. treat a zero-byte reserved private-sample file as unavailable in pathname
   output-manifest reconstruction, matching descriptor-backed construction;
3. separate archival proposal parsing from current live-candidate validation
   so attempt-1 terminal evidence remains verifiable without making V2 current
   or reusable; and
4. move every new proposal, authority, claim, public result/progress/manifest,
   emergency reservation, private sample, and log to distinct V3/attempt-2
   paths.

No target, data, transform, transition, execution identity, smoke topology,
counts, seeds, dtype, XLA/JIT setting, threshold, package, or serious artifact
may change under this repair.

## Required Artifacts

Before requesting runtime approval:

- Strict Phase 6 authority-proposal, proposal-manifest, smoke-authority,
  launch-claim, smoke-result/failure, smoke-progress, output-manifest,
  infrastructure-failure, and infrastructure-manifest schemas, parsers,
  verifiers, and no-runtime tests.
- A narrow smoke launcher path that cannot select serious mode or serious
  output paths. It performs all pre-runtime checks and the atomic claim before
  opening its log or entering any output-writing controller block.
- A narrow authority-materialization CLI that accepts only the exact
  manifest-bound approval statement plus a canonical approval date, rechecks
  the complete proposal/Phase 5/live-preflight/environment bundle, and creates
  the authority artifact exclusively and crash-durably. Restart recovery may
  accept only byte-identical complete authority bytes; it must never replace a
  partial, malformed, symlinked, or different authority artifact.
- New attempt-2 smoke authority proposal after the source-loader/output-
  manifest repair:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_authority_proposal_v3.json`.
- New attempt-2 terminal proposal manifest:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_authority_proposal_manifest_v3.json`.
- The original and V2 proposal pairs plus all attempt-1 runtime artifacts
  remain immutable superseded/consumed evidence. They must not be overwritten,
  deleted, or treated as current authority.
- Terminal proposal manifest that binds the exact reviewed subplan, Phase 5
  terminal manifest, V2 config, adoption record, preflight, the deterministic
  transitive static-import closure rooted at the authority module, controller,
  exact benchmark driver, launcher, proposal builder, and authority builder,
  the eight exact Phase 2-6 review tests, Python executable, fixed command,
  output paths, timeout, and nonclaims. This is a scoped runtime/review
  inventory, not the complete repository Python/test inventory; any unbound
  BayesFilter or `docs.*` module that actually loads in a child is rejected.
- Focused implementation-review record.

Only after the exact human approval:

- Terminal attempt-2 smoke authority record written as
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2_authority.json`.
  It binds the exact new approval statement and V3 proposal-manifest hash. It
  must not modify or be referenced by completed Phase 5 or attempt-1 artifacts.
- Permanent atomic launch-consumption claim:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2_launch_claim.json`.
  It remains present on pass, failure, timeout, interruption, or crash. A
  partial/malformed existing claim still means the authority was consumed.
- Dedicated public result:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2_result.json`.
- Dedicated public progress:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2_progress.json`.
- Dedicated protected retained-sample artifact:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/private_diagnostics/phase6_typed_identity_smoke_attempt2_retained_samples.npz`.
  Only its hash, byte count, shape/finite checks, and
  `path_publicized=false` may be exposed publicly.
- Full log, opened by the launcher only after the permanent claim exists:
  `docs/plans/logs/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2.log`.
- Phase 6 attempt-2 output-integrity manifest:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2_output_manifest.json`.
  It terminally binds the authority record, permanent launch claim, public
  progress/result, log reference, and bounded private-sample reference without
  a circular hash edge. A normal manifest also binds the two empty emergency
  reservations.
- On any catchable post-claim launcher/reservation failure, a separate strict
  infrastructure-failure artifact and infrastructure manifest at
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2_infrastructure_failure.json`
  and
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2_infrastructure_manifest.json`.
  They bind the exact held descriptor bytes for each reserved ordinary output, record
  whether each reviewed pathname remains intact, preserve any valid primary
  controller result without overwrite, and classify unreserved paths
  explicitly rather than inventing evidence.
- Phase 6 result/close record and drafted Phase 7 serious subplan. The Phase 7
  subplan must remain non-executable pending a new human serious-run approval.

## Separate Smoke Authority Contract

Use this exact decision identifier:

`AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE`

The approval request presented after implementation/review must include the
exact terminal Phase 6 authority-proposal-manifest embedded hash. The accepted
statement must be exactly:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE bound to Phase 6 authority proposal manifest <exact sha256:...>.`

The authority record must be a closed, exact schema and bind:

1. decision identifier, exact approval statement, and canonical `YYYY-MM-DD`
   approval date;
2. exact Phase 6 authority-proposal-manifest reference;
3. exact V2 config/adoption/preflight/Phase 5 terminal-manifest references;
4. transition identity
   `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a`;
5. smoke execution identity
   `sha256:fc85f9b1e0bb406593de9f5b8195ced6e86b10ee8fd549b1ecd1a8a24d6ac604`;
6. mode `smoke`, two workers, two chains per worker, 4 burn-in and 8 retained
   transitions per chain, CPU hiding, float64, XLA/JIT, versions, threads,
   seeds, and sequential worker compilation;
7. the exact launcher command, permanent claim path, dedicated output/log
   paths, and bounded timeout;
8. `serious_runtime_authority=false`, `phase8_authority=false`, and
   `neutra_authority=false`;
9. ordered nonclaims and its own embedded artifact hash.

The runtime API must verify this record and every live reference after deciding
the selected mode but before creating any output or worker. The immutable V2
config remains `runtime_authority=false`; the separate record is eligible for
one smoke launch only. Eligibility is consumed by an atomic permanent claim,
not by trusting a static boolean or approval string. No boolean edit or copied
approval string in the V2 config is permitted.

## Atomic Consumption And Pre-Runtime Ordering

The launcher must apply this exact order before any HMC compile transition:

1. parse the CLI and require the exact reviewed smoke mode, authority path,
   command, claim path, timeout, and output/log paths;
2. parse and verify the terminal authority record and its proposal-manifest
   reference;
3. verify Phase 3-5 terminal evidence and rebuild the full live V2 preflight;
4. verify the selected transition/smoke-execution identities, environment, and
   source references;
5. resolve all governed, serious, Phase 5, claim, log, public, and private paths;
   reject duplicates, aliases, symlink escapes, or existing smoke outputs;
6. open and pin every reviewed output parent component from a repository-root
   directory descriptor with `O_DIRECTORY|O_NOFOLLOW`; record and recheck its
   device/inode identity; atomically create the claim through the pinned parent
   with `openat(O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW, 0o400)`, write a strict claim binding the exact authority artifact hash,
   proposal-manifest hash, command, output/log paths, PID, and start time, then
   flush and `fsync` it and `fsync` the pinned parent directory so the new
   directory entry is crash-durable; reopen and retain a read-only claim
   descriptor for exact post-claim verification;
7. never remove, truncate, replace, or reuse the claim. If claim-file or
   parent-directory sync is interrupted or fails and the path exists, that
   path permanently consumes this authority; do not retry, repair, or delete
   it in place;
8. only after the claim is durable, exclusively reserve the infrastructure
   failure/manifest, normal output manifest, log, progress, result, and private
   sample files through their pinned parents with `O_NOFOLLOW|O_EXCL`; retain
   every descriptor and recheck parent plus file identity before every write;
9. enter one post-claim supervisor covering reservation, log redirection,
   controller runtime, log sealing, manifest construction, and manifest write;
   on any failure, preserve the original exception, write the separate
   infrastructure terminal lane best-effort with bounded idempotent retries,
   preserve a valid primary result, bind raw held bytes for partial files, and
   re-raise `BaseException` control-flow causes after sealing;
10. build terminal evidence from captured pre-claim proposal/manifest/
    authority/preflight payloads plus pinned post-claim descriptors; do not
    reopen governed paths after claim consumption; and
11. never call the generic serious-result writer from smoke mode.

Steps 1-5 fail with no new artifact or worker. Step 6 creates only the permanent
claim. Step 8 creates only exclusive empty reservations before a worker. Any
failure after step 6 consumes the approval and requires a newly
reviewed proposal and new explicit human approval before another smoke attempt.
This is intentionally fail-closed; deleting a claim is forbidden.

If the pinned attempt-1 evidence drifts after step 6, the typed drift path is
stricter than ordinary infrastructure-failure handling. Reservation-time drift
preserves only the permanent claim and any empty reservations already created.
A later detection may preserve valid bytes completed before detection and must
tear down any workers already created. In both cases, close all retained
descriptors, perform no further output write after detection, and do not
construct or seal controller/infrastructure terminal artifacts. The explicit
post-progress/pre-pool check therefore permits the already-durable initial
progress record but creates no worker.

The claim prevents concurrent or later replay of the same authority. The
launcher must also refuse an existing log/result/progress/private-sample path
before claiming. It must not use shell redirection because that would create
the log before authority consumption; the launcher opens/duplicates the log
descriptor only after the durable claim.

## Smoke-Specific Result Contract

Smoke success must use a dedicated schema and exact decision such as
`PASS_PHASE7_TYPED_IDENTITY_SMOKE_MECHANICS_ONLY_STOP_BEFORE_SERIOUS_APPROVAL`.
Smoke failure must use its own schema/decision and must not imply that Phase 7
serious sampling ran or failed. Both must bind the authority and claim hashes,
record `smoke=true`, `serious_runtime_executed=false`, `phase8_executed=false`,
and `neutra_executed=false`, and use fixed smoke-specific nonclaims.

The smoke nonclaims must say the tiny mechanics smoke executed, historical
typed equality remains unsupported, serious convergence thresholds were not
evaluated, and no serious/Phase 8/NeuTra/default/scientific authority or claim
follows. They must not copy V2 nonclaims that say no smoke executed. The
existing generic `PHASE7_RESULT_SCHEMA`,
`PASS_PHASE7_TO_PHASE8_APPROVAL_BOUNDARY`, and generic V2 nonclaim tuple are
forbidden in smoke terminal artifacts.

Authority/reference/live-preflight/collision failures occur before claim and
must not write a smoke failure result. Controller runtime failures after a
durable claim may write a smoke-specific primary failure artifact. Catchable
launcher/reservation/log/manifest failures use the distinct infrastructure
failure lane and must never overwrite a valid primary controller result. Every
post-claim failure leaves the claim in place.

## Child-Side Identity Gate

The worker request must carry the exact governed source references and expected
transition identity from the verified authority/V2 bundle plus the exact
proposal implementation-reference inventory. On initialization, each child
must verify that the inventory role set still exactly equals the live closed
inventory and that every referenced file and executable matches, then reopen
and exact-byte/canonical/reference-verify the governed sources, reconstruct
`FrozenHMCTransitionIdentityV1` from the child's loaded replay, and match the
approved transition hash before building the chunk runner or calling the
one-transition compile probe. It must return separate verified implementation,
governed-source, and transition status in bounded metadata. A mismatch is a
hard veto and no compile transition may run in that child.

This child gate closes the parent-preflight-to-worker-read gap. After the child
gate, the transition uses the already verified in-memory objects; it must not
reread a governed source later in the worker lifetime.

## Fixed Smoke Execution

| Field | Fixed value |
| --- | --- |
| Target | Approved deterministic `T=120`, 18-parameter LGSSM replay |
| Device | CPU only; `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import in parent and workers |
| Workers/chains | Two persistent spawned workers, two chains each, four chains total |
| Compilation | Sequential worker initialization; `tf.function(..., jit_compile=True)` / XLA only |
| Dtype | `float64` |
| Burn-in | Exactly 4 transitions per chain; no extension |
| Retained | Exactly 8 transitions per chain; no extension |
| Seeds | Adopted root `(20260711, 701)` and existing deterministic derivation, including distinct compile-probe seeds |
| Diagnostics | Finite-input and finite-diagnostic engineering screen only; configured serious thresholds remain unchanged but are not applied as smoke promotion gates |
| Timeout | A reviewed bounded machine timeout recorded in the authority proposal; timeout is an engineering veto, not a scientific result |
| Outputs | Dedicated Phase 6 paths only; fail before claim/runtime if any exists or aliases a Phase 3-5/serious/authority path |

## Required Local Checks Before Approval Request

- Parse and live-verify Phase 3, Phase 4, and Phase 5 terminal manifests.
- Recompute current V2, adoption, preflight, Phase 5 manifest, launcher, and
  test exact file hashes.
- Test missing, malformed, stale, copied-path, rehashed-tampered, wrong-mode,
  wrong-command, wrong-output, wrong-count, wrong-version, wrong-identity, and
  unknown-field authority records.
- Test proposal/authority exclusive creation, file and parent `fsync`, exact-byte
  restart recovery, and rejection of partial, different, symlinked, or
  noncanonical-date authority materialization.
- Test that `smoke=False`, missing authority, V2 boolean tamper, output
  collision, and serious-path aliasing fail before output or worker creation.
- Test atomic `O_EXCL` claim behavior, claim-file and parent-directory `fsync`,
  concurrent launch rejection, partial or malformed existing-claim rejection,
  permanent consumption after simulated file-sync, directory-sync, or later
  failure, and the prohibition on claim deletion/reuse.
- Inject separate claim-file and parent-directory `fsync` failures; in each
  case require the claim path to remain, retry to raise `FileExistsError`, and
  no delete/reuse path to exist.
- Race symlink creation and pathname replacement for the log, progress,
  result, output manifest, and protected NPZ. Test parent-directory
  replacement separately. Require `O_NOFOLLOW`/device-inode validation to
  veto, preserve the outside target bytes, and leave the claim consumed.
- Inject post-claim reservation, redirection, controller, log-sealing, and
  manifest construction/write failures, including one-shot emergency artifact
  construction/write/`fsync` failures and `KeyboardInterrupt`. Require a strict
  infrastructure failure plus terminal infrastructure manifest whenever their
  reserved descriptors remain usable; require any valid primary result bytes
  to remain unchanged, bounded idempotent retry to accept already-complete
  descriptor bytes, and the original control-flow exception to be re-raised.
- Test that pre-authority, preflight, and collision failures create no claim,
  log, result, progress, sample, or worker; test that the log is opened only
  after the durable claim.
- Test smoke success/failure exact schemas, decisions, and nonclaims; reject
  `PASS_PHASE7_TO_PHASE8_APPROVAL_BOUNDARY`, generic serious-result schema, or
  V2's `not Phase 7 smoke or serious execution` nonclaim in smoke output.
- Test child-side source/reference and transition tampering between parent
  preflight and child initialization, plus project implementation-reference
  tampering; assert rejection before runner construction or `runner.run`.
- Test a failing worker with a hung peer and `KeyboardInterrupt` through the
  real controller teardown structure using fake executors; require all peers
  to be signaled before a shared bounded join, survivors killed,
  `shutdown(wait=False)`, and original control flow preserved.
- Test a fully finite diagnostic payload whose R-hat/ESS values fail the
  unchanged serious thresholds; require both top-level and per-parameter smoke
  pass under `finite_diagnostics_only_non_promoting` while preserving the
  observed explanatory values.
- Test that valid authority can reach a fully mocked smoke path only with
  `smoke=True`; no local pre-approval test may execute a real transition.
- Test terminal authority/proposal/output-manifest acyclicity and exact-byte
  tamper detection.
- Test public redaction and private-sample reference boundaries.
- Run Python compilation, focused Phase 2-6/controller pytest, forbidden
  bypass/repin scan, and scoped `git diff --check`.
- Obtain fresh independent Codex review. The managed Claude disclosure
  rejection remains binding and must not be retried.
- Verify the immutable attempt-1 output manifest from exact bytes and test that
  its V2 proposal cannot pass the current attempt-2 live candidate gate.
- Test that the shared attempt-1 integrity gate covers all 13 original/V2/
  runtime evidence files and rejects byte, size, mode, symlink, semantic, or
  terminal-classification drift before any attempt-2 artifact is written.
- Write/update the pre-runtime Phase 6 result section, ledger, runbook, and
  handoff to `AWAITING_HUMAN_PHASE7_SMOKE_APPROVAL_V3_ATTEMPT2`.
- Stop and request the exact approval. Do not execute smoke in the same
  authorization scope that created or reviewed the proposal.

## Required Runtime Checks After Exact Approval

Immediately before launch:

1. run only the exact reviewed smoke launcher command; the launcher verifies
   approval, authority, Phase 3-5 artifacts, live identities, paths, and the
   environment before atomically consuming authority;
2. confirm the permanent claim was durably written before log/output/worker;
3. confirm each child independently verified governed sources and transition
   identity before its compile probe;
4. validate public result/progress embedded hashes, private artifact hash/size/
   shape/finiteness, worker PIDs, two-worker persistence, XLA/JIT metadata,
   counts, seeds, CPU hiding, smoke-specific semantics, and no serious/Phase
   8/NeuTra execution;
5. classify any failure before considering repair. Never delete the claim or
   reuse the approval for a retry.

The expected smoke command may be frozen only after the launcher and authority
proposal exist. It must select `burnin_sampling`, `--phase7-smoke`, the exact
authority record, and dedicated Phase 6 repository output paths. It must not
re-run kernel tuning or use an ad hoc `python -c` controller call.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can the approved typed V2 transition complete the tiny exact Phase 7 two-worker CPU/XLA mechanics route under a separately verified smoke-only authority? |
| Exact baseline | Immutable Phase 5 V2/adoption/preflight/terminal-manifest evidence and live transition/smoke-execution identities. |
| Primary criterion | Authority and preflight pass before claim/workers; a permanent claim consumes the approval once; both children verify their own loaded sources/transition before compile; fixed two-worker smoke executes exact tiny counts with CPU/XLA/JIT/float64 provenance; all required mechanics values and diagnostics are finite; smoke-specific dedicated artifacts validate. |
| Hard vetoes | Approval/reference mismatch, Phase 5 drift, typed/integrity mismatch, existing/replayed/invalid claim, output collision/alias, pre-claim output, child identity mismatch, generic serious semantics in smoke output, nonfinite runtime value, available nonzero divergence, XLA/JIT fallback, wrong process/topology/count/seed, worker failure, timeout, invalid artifact, or public disclosure. |
| Explanatory only | Acceptance, R-hat/ESS values from 8 retained draws, compile/runtime timings, PIDs, and legacy whole-payload differences. |
| Not concluded | No convergence, recovery, posterior correctness, ranking, serious readiness by smoke alone, production/default, GPU, Phase 8, NeuTra, or scientific claim. |
| Preserving artifact | Immutable Phase 5 bundle; Phase 6 proposal/authority, dedicated smoke artifacts/log/manifest, result, review, ledger, and handoff. |

## Forbidden Claims And Actions

- Do not run any HMC transition or create workers before exact separate human
  smoke approval.
- Do not mutate V1, V2, adoption, preflight, Phase 5 manifest, Phase 3/4
  evidence, tuning artifacts, target, transforms, mechanics, thresholds,
  topology, versions, counts, seeds, threads, or serious artifact paths.
- Do not let a general runtime boolean, prior approval, CLI flag alone, or
  review verdict authorize smoke.
- Do not delete, chmod for mutation, rewrite, rename, truncate, or reuse any
  attempt-1 proposal, authority, claim, output, log, or zero-byte reservation.
- Do not interpret the repeated V2 approval as authority for attempt 2.
- Do not represent a static authority record plus a file-existence check as
  one-use authority. Do not delete or reuse the permanent launch claim.
- Do not create the log or any smoke output before authority, live preflight,
  collision/alias checks, and durable atomic claim succeed.
- Do not emit the serious result schema/decision/nonclaims from smoke mode.
- Do not let a child take its compile transition using sources verified only by
  the parent.
- Do not allow the smoke authority record to authorize `smoke=False`, serious
  burn-in/sampling, output reuse, Phase 8, NeuTra, package/network action, or a
  product/default-policy change.
- Do not interpret finite smoke diagnostics, acceptance, R-hat, ESS, or timing
  as convergence, recovery, comparison, or readiness evidence.
- Do not retry Claude while the managed external-disclosure rejection remains
  binding.

## Exact Next-Phase Handoff Conditions

Phase 7 serious planning may begin only after:

1. Phase 6 attempt-2 authority/launcher repair and local no-runtime tests pass;
2. the exact new human smoke approval is recorded against the V3 terminal proposal
   manifest;
3. the exact smoke command runs once and only in smoke mode;
4. all smoke mechanics, environment, process, finite-value, redaction, and
   artifact-integrity gates pass;
5. the Phase 6 result and Phase 7 serious subplan pass independent review;
6. the supervisor records that smoke is engineering evidence only; and
7. the supervisor stops and requests a new exact human approval for serious
   Phase 7 burn-in and retained sampling.

Phase 7 serious runtime must not execute merely because the smoke passes.

## Stop Conditions

- Independent review has not agreed or exact smoke approval is absent,
  ambiguous, denied, or bound to another proposal manifest.
- Any attempt-1 artifact differs from the immutable hash/size table, cannot be
  reverified, or is selected as an attempt-2 output or authority path.
- Any immutable Phase 3-5 artifact, governed input, source identity, or live
  typed identity changes.
- Authority cannot be made mode-specific and fail-closed before outputs/workers.
- Atomic authority consumption cannot be made permanent and race-safe, or any
  code path can delete/reuse the claim.
- Any declared smoke output exists, aliases another governed artifact, or
  cannot preserve the public/private boundary.
- A real transition occurs during pre-approval local tests.
- The approved smoke returns a hard veto, invalid artifact, timeout, or
  unclassified failure.
- Repair would change the target, transition, serious thresholds, execution
  identity, package environment, or approval scope.
- The same substantive review blocker remains after five repair rounds.
- The smoke passes: write/review the result and stop before serious runtime.
