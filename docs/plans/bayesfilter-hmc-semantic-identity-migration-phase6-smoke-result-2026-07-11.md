# Phase 6 Result: Attempt-2 Typed-Identity Mechanics Smoke Passed

Date: 2026-07-11

Updated: 2026-07-12

Status: `PHASE6_CLOSED_MECHANICS_ONLY_PASS`

## Direct Verdict

Attempt 2 consumed the exact V3-manifest-bound one-use smoke authority and
completed the fixed two-worker, four-chain, CPU-hidden Host-XLA mechanics smoke.
Both children verified the exact retained source bundle and transition identity
before compilation. Each chain completed 4 burn-in and 8 retained transitions;
all required mechanics values, diagnostics, samples, and artifact cross-links
were finite and valid. The terminal decision is
`PASS_PHASE7_TYPED_IDENTITY_SMOKE_MECHANICS_ONLY_STOP_BEFORE_SERIOUS_APPROVAL`.

Attempt 1 remains immutable failure evidence: it consumed its V2 authority and
claim, then failed before worker initialization because the retained child
loader omitted `module.__file__`. The localized repair succeeded in attempt 2;
attempt 1 is still an implementation failure, not evidence against the target,
HMC, XLA, or the scientific direction.

The smoke authority is permanently consumed and cannot be reused. Serious
Phase 7 requires a distinct reviewed one-use authority mechanism and a new
exact human approval. Phase 8 and NeuTra remain unapproved.

## Inherited Approval Scope

The baseline-migration authority consumed before Phase 6 was:

`I approve PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION bound to certificate sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f.`

That statement authorized the typed V2 baseline migration and bounded local
Phase 5/6 proposal preparation. It did not authorize a smoke or any serious
runtime. A later V2-manifest-bound smoke approval authorized exactly one launch;
attempt 1 permanently consumed it. Repetition of that V2 statement does not
authorize attempt 2.

## Historical Pre-Runtime Skeptical Audit And Evidence Contract

This table records the proposal/pre-attempt gate as it stood before either
runtime attempt. Its no-runtime statements are historical and are superseded
for terminal mechanics status by the attempt-2 closeout below.

| Field | Pre-runtime result |
| --- | --- |
| Engineering question | Can a fail-closed, proposal-bound, one-use authority mechanism safely delimit the exact tiny two-worker CPU/XLA mechanics smoke before any runtime occurs? |
| Exact baseline | Immutable Phase 5 V2, adoption, preflight, and terminal manifest; transition `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a`; smoke execution `sha256:fc85f9b1e0bb406593de9f5b8195ced6e86b10ee8fd549b1ecd1a8a24d6ac604`. |
| Primary pre-runtime criterion | Strict proposal/manifest, closed source inventory, one-use authority/claim mechanism, dedicated descriptor-backed outputs, bounded teardown, non-promoting smoke semantics, local tests, and independent reviews pass without runtime. |
| Hard veto status | No veto fired during proposal preparation. Attempt 1 later fired an implementation/runtime veto before worker initialization. HMC transition diagnostics remain unevaluated because no transition ran. |
| Explanatory only | Test counts, implementation hashes, planned topology/counts, and successful proposal verification. They do not establish worker/XLA/HMC feasibility. |
| Not concluded | No smoke pass, convergence, posterior recovery, ranking, serious readiness, production/default, GPU, Phase 8, NeuTra, or scientific validity. |

The plan audit passed for proposal-only execution. It found and repaired wrong
trust boundaries rather than weakening them: caller-forgeable contexts,
incomplete source inventory, import/restore and namespace-parent races,
unbounded teardown, writable claims, output-path TOCTOU, emergency-evidence
gaps, accidental serious-diagnostic gating, and incomplete collision/race
coverage. The final implementation and proposal artifact reviews both ended
`VERDICT: AGREE`.

## Superseded Original Proposal Evidence

| Artifact | Embedded artifact hash | Exact file SHA-256 | Bytes |
| --- | --- | --- | ---: |
| `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_authority_proposal.json` | `sha256:57b9434a54c3c2ac9c67ddf57a54caaf00feb9dcf9910a0fb41b03e44bad653a` | `16df0bdb62f45e9b2c304a7030c5c7d08497720f42c43dbf489b694dc9497d0d` | 193504 |
| `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_authority_proposal_manifest.json` | `sha256:9db02019042769750a731dbc849746c5e3380a8883e03167553d7829acf0f1c7` | `b31d93a568bd30458c56bc87d9eca17ea73ea3579f973591e00d0a9a80696c3c` | 848 |

The original proposal was `pending_human_smoke_approval`, bound 533 exact
implementation/runtime references, and records:

- mode `smoke` only;
- two spawned workers and two chains per worker;
- 4 burn-in and 8 retained transitions per chain;
- CPU hiding, float64, TensorFlow/TFP, XLA/JIT, fixed threads and seeds, and
  sequential worker compilation;
- a wall-time cap of 28800 seconds;
- dedicated output paths and permanent one-use claim; and
- `serious_runtime_authority=false`, `phase8_authority=false`, and
  `neutra_authority=false`.

The proposal graph is acyclic: it binds completed Phase 5 and frozen source
bytes, and the terminal proposal manifest binds the complete proposal bytes.
Neither artifact points forward to a future authority or runtime result.

## Implementation Review And Repairs

The implementation review record is
`docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase6-preruntime-implementation-codex-review-2026-07-11.md`.
The proposal artifact review record is
`docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase6-proposal-artifacts-codex-review-2026-07-11.md`.

The final frozen implementation review confirmed:

- full live verification precedes either proposal write;
- parent and child execute project code from retained approved bytes;
- child import bootstrap owns code-free `docs` namespace parents and rejects
  unapproved project/docs modules;
- proposal/authority/claim/output references are backward-bound and acyclic;
- launch context and output-session capabilities cannot be caller-forged or
  replayed in-process;
- the permanent `0400` claim is atomically created and never deleted;
- all output writes are descriptor-based through pinned parents;
- success, ordinary failure, timeout, and control-flow teardown are bounded;
  and
- smoke diagnostics are finite-only engineering evidence and cannot emit the
  serious Phase 7 decision.

## Historical Pre-Attempt Checks

These checks preceded attempt 1 and are retained as historical proposal
evidence. They were deliberate CPU-hidden engineering checks with
`CUDA_VISIBLE_DEVICES=-1` and did not execute a worker or HMC transition.

| Check | Result |
| --- | --- |
| Targeted source-loader/loaded-module regressions | `4 passed` |
| Final focused Phase 2-6/controller gate | `207 passed, 2 warnings` |
| Final combined eight-module gate | `230 passed, 2 warnings in 37.45s` |
| Warnings | Existing TFP `distutils.version` deprecations only |
| Python compilation | Passed |
| Scoped tracked and untracked whitespace checks | Passed |
| Authority-literal and bypass/repin scans | Passed |
| Live proposal/manifest/Phase 5/inventory verification | Passed |
| Independent frozen implementation review | `VERDICT: AGREE` |
| Independent exact proposal artifact review | `VERDICT: AGREE` |
| HMC transition, worker, XLA compile transition, or sampling | Not run |

TensorFlow emitted CUDA factory-registration and `cuInit` messages while GPU
devices were deliberately hidden. Those messages are not GPU health evidence
and do not alter the CPU-only test classification.

## Historical Pre-Attempt Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Worktree | Dirty; in-scope migration files plus unrelated user/agent work preserved |
| Command class | CPU-hidden pytest, compilation, hashing, proposal construction, strict live verification, and read-only review only |
| Python | `3.11.14`; proposal command resolves to `/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11` |
| TensorFlow / TFP | `2.19.1` / `0.25.0` |
| CPU/GPU status | CPU only by explicit `CUDA_VISIBLE_DEVICES=-1`; no GPU evidence |
| XLA/JIT | Required and identity-bound for the future smoke; no compile transition ran |
| Data | Immutable Phase 5 governed LGSSM fixture/replay references |
| Random seeds | Future smoke root seed `[20260711, 701]`; no stochastic transition ran |
| Final combined-test wall time | `37.45 s` |
| Plan | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-subplan-2026-07-11.md` |
| Result | This file |

## Current Decision Table

| Decision | Status |
| --- | --- |
| Phase 6 attempt-2 repair | Passed local and independent frozen review |
| Attempt 1 | Implementation failure before worker initialization; authority permanently consumed |
| V3 proposal and terminal manifest | Passed live and independent exact-artifact review; approval consumed by attempt 2 |
| Attempt-2 smoke runtime | Passed mechanics-only gate; authority permanently consumed |
| Serious Phase 7 | Not authorized; Phase 7 subplan is non-executable pending review and separate authority |
| Phase 8 / NeuTra | Not authorized |
| Next justified action | Review the terminal Phase 6 closeout and non-executable Phase 7 serious subplan; then design/freeze a separate serious authority proposal and request a new exact manifest-bound approval |

## Superseded Original Stop And Approval Request

Before the 2026-07-12 concurrent-lane drift, the requested statement was:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE bound to Phase 6 authority proposal manifest sha256:9db02019042769750a731dbc849746c5e3380a8883e03167553d7829acf0f1c7.`

That statement was received, but pre-authority verification vetoed it after
the original broad inventory drifted. It is historical evidence only and
cannot authorize the refreshed V2 proposal below.

The original and V2 approval statements are both historical and non-actionable.
The later exact V3-bound approval was consumed once by attempt 2. The
post-smoke serious-runtime stop is now active.

## Residual Risk And Nonclaims

The child import, retained-source verification, Host-XLA compilation,
persistent-worker topology, HMC transition mechanics, private-sample writing,
and terminal artifact path were exercised successfully. The remaining risk is
the serious controller: the smoke used only 4 burn-in and 8 retained
transitions per chain, so it did not test long-run convergence, caps, resource
behavior, or serious output replacement/recovery.

No historical-private transition equality, convergence, recovery, statistical
ranking, production/default readiness, GPU readiness, Phase 8, NeuTra,
package, product, or scientific claim is made.

## 2026-07-12 Concurrent-Lane Proposal Refresh

The human supplied the exact approval bound to the original terminal proposal
manifest:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE bound to Phase 6 authority proposal manifest sha256:9db02019042769750a731dbc849746c5e3380a8883e03167553d7829acf0f1c7.`

Authority materialization failed safely before writing any authority because
two unrelated test files had been added by another active repository lane
after the original proposal review. The original proposal bound the complete
repository test tree, so those unrelated additions changed its role set. No
authority, permanent claim, output, log, private sample, worker, or HMC
transition was created. The smoke was not attempted.

The root cause was a governance-scope defect: repository-wide Python inventory
was broader than the Phase 6 runtime trust boundary. The repaired inventory is
a deterministic static-import closure rooted at the six Phase 6 runtime
entrypoints, plus eight exact review tests and the Python executable. It binds
71 roles: 62 runtime sources, 8 tests, and 1 executable. Unrelated lane files
may coexist; any unbound BayesFilter or `docs.*` module that actually loads in
a child still fails closed.

The original proposal pair remains immutable superseded evidence. The
refreshed versioned pair is:

| Artifact | Embedded artifact hash | Exact file SHA-256 | Bytes |
| --- | --- | --- | ---: |
| `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_authority_proposal_v2.json` | `sha256:6ab3167abd521c6c41fc481cfed75d4ffae613cc672d49019bedbf8490639ced` | `f8c1d301186e9b1df390dbc4248c95932737bf2a7d8f50c6af985129bc7755c8` | 30416 |
| `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_authority_proposal_manifest_v2.json` | `sha256:e8e913e005423da1da87bfa1f5a8e832f7b32d8a8a90172aa81fefea8607bc3b` | `29dbba924ce899189e178d624ddc26c1fdfaaf46674244c3547f44c7ee591527` | 847 |

Targeted tests passed (`8 passed`). The complete eight-module gate passed
(`231 passed, 2 warnings in 388.79s`), with only the existing TFP deprecation
warnings. Python compilation, whitespace, authority/bypass scans, clean-process
import-closure coverage, unrelated-file stability, live proposal verification,
and original-artifact integrity passed. Independent implementation and exact
artifact reviews both ended `VERDICT: AGREE`.

The original approval was inapplicable to changed proposal bytes and was
rejected by the authority builder. At that historical V2 gate, the statement
requested next was:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE bound to Phase 6 authority proposal manifest sha256:e8e913e005423da1da87bfa1f5a8e832f7b32d8a8a90172aa81fefea8607bc3b.`

That V2 statement was received and consumed by attempt 1. It cannot authorize
attempt 2. Serious Phase 7, Phase 8, and NeuTra remain separate boundaries.

## 2026-07-12 Attempt-1 Failure, Repair, And V3 Proposal Gate

Attempt 1 created and permanently consumed V2 authority
`sha256:6206888214a63b5b0c56a776d27f2a520880b0645ebf7e2afc280f000cfe4c58`
and launch claim
`sha256:6cddfe35278935f9be30a65bf2b481ac53eaaba59818de56855b55b53d73b2ad`.
Its terminal failure result is
`sha256:68bfd9078c9f187874d2a2334f8353fbc5d4f4736e52dd58c42764782bbcd275`:
stage `preflight_passed`, reason `runtime_error:BrokenProcessPool`, zero worker
PIDs, no diagnostics, no burn-in or retained checks, and no private sample
bytes. The immutable 13-file ledger in the reviewed Phase 6 subplan is the
binding attempt-1 evidence set.

The repair set:

- supplies `module.__file__` in the retained child loader;
- treats zero-byte private reservations as unavailable;
- separates archival proposal verification from current authorization;
- moves every retry path to V3/attempt 2;
- pins, locks, hashes, double-reads, semantically verifies, and retains all 13
  attempt-1 evidence descriptors across proposal/authority/launch boundaries;
- makes typed evidence drift bypass all later controller and infrastructure
  writes while preserving bounded worker teardown; and
- closes same-size overwrite, mode, symlink, hard-link, lock, claim-boundary,
  reservation, controller-entry, and post-progress adversarial cases.

The final no-runtime verification was:

| Check | Result |
| --- | --- |
| Stage-specific drift matrix | `3 passed, 2 warnings` |
| Complete authority module | `106 passed, 2 warnings in 312.64s` |
| Combined eight-module migration gate | `251 passed, 2 warnings in 340.78s` |
| Python compilation and scoped whitespace | Passed |
| Immutable attempt-1 ledger | `INTEGRITY_OK 13` |
| Frozen attempt-2 repair review | `VERDICT: AGREE` |
| Exact V3 artifact review | `VERDICT: AGREE` |

The warnings were existing TensorFlow Probability `distutils.version`
deprecations. TensorFlow also emitted CUDA factory-registration and `cuInit`
noise while devices were deliberately hidden. These are CPU-hidden test logs,
not GPU health or runtime evidence. No test created a worker or HMC transition.

The reviewed V3 proposal pair is:

| Artifact | Embedded artifact hash | Exact file SHA-256 | Bytes | Mode |
| --- | --- | --- | ---: | ---: |
| `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_authority_proposal_v3.json` | `sha256:d2aff98cb93b85527bd71a206af5244aa18e373ae8a3bd7897b8fc3c841d0395` | `7a5c093a42d7b373d1711c29ed073eb46954f3517d4246878a5d1ff20df40880` | 30498 | `0600` |
| `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_authority_proposal_manifest_v3.json` | `sha256:9f026fcf4382e77df5e5e4adff97ac63ceed918717e3be88f611eac7f1a2c3d0` | `e15cd087fa40e91acb875d88d948fc185a0e6bf1eabc17841111aa9048a7d503` | 847 | `0600` |

At that pre-runtime gate the proposal was
`pending_human_smoke_approval`, bound 71 exact live implementation references
and distinct attempt-2 paths, and granted no serious, Phase 8, NeuTra,
GPU/default/product, or scientific authority. All attempt-2 authority, claim,
output, log, infrastructure, and private-sample paths were then absent.

## Historical Exact Attempt-2 Approval Gate

The statement accepted for attempt 2 was exactly:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE bound to Phase 6 authority proposal manifest sha256:9f026fcf4382e77df5e5e4adff97ac63ceed918717e3be88f611eac7f1a2c3d0.`

At that gate, no attempt-2 artifact or transition was permitted until the exact
statement was received. It was then consumed by the single terminal launch
documented below. It cannot be used again. The V2 approval bound to
`sha256:e8e913e005423da1da87bfa1f5a8e832f7b32d8a8a90172aa81fefea8607bc3b`
is terminally consumed and cannot be reused.

## 2026-07-12 Attempt-2 Terminal Closeout

The exact V3-bound approval was received and materialized as one smoke-only
authority. Prelaunch verification passed the 71-role live implementation
closure, all immutable Phase 5 references, all 13 attempt-1 evidence files,
exact command/path constraints, output absence, and process absence. The exact
reviewed launcher then ran once. The permanent claim consumed the authority
before output or worker creation.

### Terminal Artifacts

| Artifact | Embedded artifact hash | Raw SHA-256 | Bytes | Mode |
| --- | --- | --- | ---: | ---: |
| Attempt-2 authority | `sha256:1f3b8f6b92fda72221fa5036ad752c997d75e4e975b0e0c83afe116eef5e0e9b` | `0ec56084480028932761b29ed16d18310919323f1a879c20a8806d8937154a66` | 1712 | `0600` |
| Permanent claim | `sha256:7c3b9ec793eb5dffc5f8b0471ba839cbda7684b2d794c172c51c7df50e93f5ca` | `af813e923269b7547c7cdaab8c2d46f256c2620181dba4d067030fbc6d1608ae` | 1967 | `0400` |
| Result | `sha256:e7584e3c3d62e0a2370a33c1a77c8b9c6b1e157d1199cea4ceb9fd749a7a576d` | `ebd870daba9aaff60708327efba3c9fcaeec0624e396be59b34ba61c245e3397` | 16505 | `0400` |
| Progress | `sha256:698818a54380c2f2207c35a122201c000111a63c8d52c9d256c98e9051370e05` | `251d91ddf7f7c86c8a35b1b01a2e1ae91afa138a82bf5b44bb255f1a34bbe730` | 1560 | `0400` |
| Output manifest | `sha256:805312c66c742cf2f7bce6da9c8e585a2bc99350ebd3bd65f474fd063eba51a8` | `6faa154774f99b7cadd7ea9665501b34c329c9663793ea9f3487e8d0507a0384` | 4100 | `0400` |
| Log | N/A | `1bb925e287b15ef927219c709e5f887a9708d7125d9c61d15842037476ebe08e` | 5494 | `0400` |
| Private sample | N/A | `d46514c6fad6dd0b55f9563f9686fee0436034a448677e295062ec899c24393f` | 4392 | `0400` |
| Infrastructure-failure reservation | N/A | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | `0400` |
| Infrastructure-manifest reservation | N/A | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | `0400` |

The ordinary output manifest is terminal. Neither emergency reservation was
used. No smoke process remained after completion.

### Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Worktree | Dirty; semantic-identity lane plus unrelated concurrent user/agent work preserved |
| Command | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 scripts/run_hmc_phase6_typed_identity_smoke.py --stage burnin_sampling --phase7-smoke --phase7-smoke-authority docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase6_smoke_attempt2_authority.json` |
| Environment | Conda `tf-gpu`; Python `3.11.14`; TensorFlow `2.19.1`; TFP `0.25.0` |
| Device | CPU only by deliberate `CUDA_VISIBLE_DEVICES=-1`; this is not GPU evidence |
| XLA/JIT | Host XLA, `jit_compile=true`; one compile trace per worker |
| Data | Immutable governed deterministic `T=120`, 18-parameter LGSSM replay |
| Seeds | Root `(20260711, 701)` with the fixed smoke derivation |
| Workers/chains | Two persistent workers, PIDs `21` and `121`; two chains each |
| Counts | 4 burn-in and 8 retained transitions per chain |
| Wall time | `25.814051708206534` seconds |
| Plan | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-subplan-2026-07-11.md` |
| Result | This file |

### Independent Terminal Verification

The strict result and progress parsers passed. The output manifest rebuilt
exactly from current bytes. Authority, claim, V3 proposal manifest, result,
progress, log, private sample, and emergency-reservation cross-links passed.
All 13 attempt-1 files reverified. The protected NPZ independently contained:

- `retained_raw_samples`: finite `float64`, shape `(8, 4, 18)`;
- `final_worker_states`: finite `float64`, shape `(2, 2, 18)`;
- config hash equal to the terminal result config hash; and
- private replay hash equal to
  `preflight_before_runtime.private_replay_artifact_hash`.

The NPZ replay value is the embedded private replay artifact hash
`sha256:ce878e2a28c49256ac2d75f3b3d8e2207a5a106e0c9e0175dfcf43020799b867`.
It is intentionally distinct from the separately recorded canonical replay
payload hash `sha256:14f252578e6d8e6f09194a0e1bef6af04f24fa18183dfae3620628c74f473c45`.

### Diagnostic Interpretation

The smoke hard-veto screen passed. Retained explanatory diagnostics were:

| Diagnostic | Observed |
| --- | ---: |
| Maximum R-hat | `3.685359225168008` |
| Minimum bulk ESS | `13.697247858180793` |
| Minimum tail ESS | `8.0` |

These values use only 8 retained draws per chain. They are explanatory only.
The serious thresholds R-hat `<=1.01`, bulk ESS `>=1000`, and tail ESS `>=400`
were deliberately not smoke promotion criteria. The observed values therefore
do not constitute a convergence failure, candidate rejection, ranking, or
serious-readiness result.

CUDA factory-registration and `cuInit` messages appeared despite deliberate
GPU hiding. Both workers initialized XLA for the Host platform. These messages
are not GPU evidence and do not invalidate the CPU-hidden smoke.

### Decision And Handoff

| Decision field | Terminal status |
| --- | --- |
| Engineering mechanics | Passed |
| Primary criterion status | Passed: exact authority/preflight, two persistent workers/four chains, fixed tiny counts, Host XLA/JIT, finite mechanics/sample state, and strict terminal artifacts verified |
| Hard veto screen | Passed; no nonfinite, identity, worker, XLA, or artifact veto |
| Main uncertainty | Only 8 retained draws per chain; long-run mixing, serious controller caps/resources, convergence, and serious output replacement remain unevaluated |
| Serious convergence criterion | Not evaluated |
| Statistically supported ranking | None; no comparison and insufficient draws |
| Descriptive-only differences | Smoke R-hat/ESS, acceptance, compile/warm timing, PIDs, and elapsed time are explanatory only |
| Default readiness | Not evaluated and not supported |
| Serious Phase 7 authority | Absent |
| Phase 8 / NeuTra authority | Absent |
| What is not concluded | No convergence, posterior recovery, calibrated uncertainty, ranking, production/default readiness, GPU readiness, Phase 8, NeuTra, or scientific validity |
| Next evidence | Separately authorized serious burn-in/retained sampling under the reviewed Phase 7 subplan |

Phase 6 is closed at a mechanics-only pass. Fresh review of this result and the
next subplan converged with `VERDICT: AGREE`:
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-subplan-2026-07-11.md`.
It is non-executable until its separate serious authority implementation and
proposal converge in review and the human approves the exact terminal serious
proposal-manifest hash. The earlier V3 smoke approval cannot cross that
boundary.
