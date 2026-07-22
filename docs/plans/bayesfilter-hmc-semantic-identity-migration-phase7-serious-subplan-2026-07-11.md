# Phase 7 Subplan: Typed-Identity Serious Burn-In And Sampling

Date: 2026-07-11

Updated: 2026-07-12

Status: `NON_EXECUTABLE_PENDING_REVIEW_AND_SEPARATE_SERIOUS_RUNTIME_AUTHORITY`

## Phase Objective

Run the approved typed HMC transition through the fixed serious deterministic
LGSSM burn-in and retained-sampling controller exactly once, under a separate
one-use human authority. Preserve the existing all-parameter convergence
thresholds, two-worker CPU/XLA execution contract, private/public boundary,
and eight-hour machine cap.

This phase asks whether the fixed transition can produce a valid retained
sample artifact under the predeclared serious controller. It does not ask
whether the tiny Phase 6 smoke converged, whether one sampler is superior, or
whether the posterior recovers truth. Posterior recovery remains a Phase 8
question and Phase 8 remains unauthorized.

## Entry Conditions Inherited From Phase 6

All of the following are required before any Phase 7 source or authority work:

1. The Phase 6 result records
   `PASS_PHASE7_TYPED_IDENTITY_SMOKE_MECHANICS_ONLY_STOP_BEFORE_SERIOUS_APPROVAL`.
2. The terminal Phase 6 attempt-2 result, progress, and output manifest parse
   and verify against current exact bytes:
   - result `sha256:e7584e3c3d62e0a2370a33c1a77c8b9c6b1e157d1199cea4ceb9fd749a7a576d`;
   - progress `sha256:698818a54380c2f2207c35a122201c000111a63c8d52c9d256c98e9051370e05`;
   - output manifest `sha256:805312c66c742cf2f7bce6da9c8e585a2bc99350ebd3bd65f474fd063eba51a8`.
3. The attempt-2 authority
   `sha256:1f3b8f6b92fda72221fa5036ad752c997d75e4e975b0e0c83afe116eef5e0e9b`
   and permanent claim
   `sha256:7c3b9ec793eb5dffc5f8b0471ba839cbda7684b2d794c172c51c7df50e93f5ca`
   remain present and consumed. They cannot authorize or be reused by Phase 7.
4. The complete 13-file attempt-1 ledger still passes exact-byte, size, mode,
   link, and semantic verification.
5. The Phase 5 V2 config, adoption record, live preflight, and terminal manifest
   remain exact and reconstruct:
   - transition identity
     `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a`;
   - serious execution identity
     `sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4`.
6. Both smoke workers independently verified the retained implementation
   source bundle and transition identity before XLA compilation, and no worker
   or smoke process remains alive.
7. The Phase 6 result and this subplan pass fresh independent read-only review.
8. Claude remains unavailable under the binding managed external-disclosure
   rejection. A fresh Codex reviewer is the required substitute unless that
   external boundary is explicitly changed.

The smoke's observed R-hat and ESS values are not entry vetoes. They came from
only eight retained draws per chain under a finite-only mechanics screen.

## Existing Serious-Path Collision

The configured public serious result path already contains the historical
pre-migration blocker:

- path:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/burnin_sampling.json`;
- embedded hash:
  `sha256:e7ae6f73b92f66e2a346823b3323f419cab57cbc3731dd5e267b7af8e60269bd`;
- raw SHA-256:
  `3b34cf56062950a9ba835f6b4839421510a8921545a0edd36203f39eac4ec0d6`;
- byte count: `2378`;
- meaning: typed migration had not yet occurred and neither smoke nor serious
  sampling executed.

That file is evidence, not an expendable placeholder. Before a serious
authority proposal may be finalized, Phase 7 must create an immutable exact
archive plus a terminal archive manifest under the semantic-identity migration
artifact directory. The archive must verify the original bytes before and
after copying. The serious launcher may replace the configured live result
path only after a durable one-use claim and only if the human-approved serious
authority proposal explicitly binds the archive manifest, the expected old
bytes, and the replacement action. It may never delete or overwrite the
archive. If that controlled replacement cannot be made crash-safe and
reviewable, stop and version the serious output paths through a separately
reviewed execution-contract migration instead.

The serious progress and retained-sample paths are currently absent. Their
absence must be rechecked immediately before authority materialization and
again before claim creation.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Rebuild the live transition and serious execution identities from the exact Phase 5 V2 sources; never use smoke identity or legacy whole-payload hashes as the serious baseline. |
| Smoke proxy promoted | Phase 6 established mechanics only. Its R-hat, ESS, timing, and acceptance values cannot authorize or predict serious convergence. |
| Missing stop | Identity, integrity, authority, collision, finite-value, XLA, worker, divergence, diagnostic-cap, wall-time, and artifact failures are explicit stops. |
| Unfair comparison | No sampler or candidate ranking occurs in Phase 7. The only comparator is the predeclared all-parameter convergence contract. |
| Hidden assumption | CPU hiding, two persistent workers, four chains, float64, XLA/JIT, seeds, thread settings, counts, thresholds, and output paths are bound by the serious execution identity and authority. |
| Stale context | Reverify the live source closure, Phase 5 bundle, Phase 6 terminals, historical blocker bytes, output absence, environment, and process table before proposal and launch. |
| Environment mismatch | Record Python, TensorFlow, TFP, Host XLA, CPU hiding, thread settings, and no non-JIT fallback in the authority, workers, result, and manifest. |
| Artifact cannot answer question | Require per-check diagnostics, terminal decision, protected raw retained samples, provenance verification, exact manifest, and a result note. |
| Static boolean becomes authority | Keep V2 `runtime_authority=false`; accept serious mode only through an unforgeable verified launch context backed by a separate human-bound authority and permanent claim. |
| Historical result overwritten | Archive and terminally bind the exact blocker before any approved replacement of its configured live path. |

Audit verdict:
`PASS_FOR_PHASE7_PLAN_REVIEW_AND_NO_RUNTIME_AUTHORITY_DESIGN_ONLY`.
Serious execution remains prohibited until the reviewed authority proposal is
bound to a new exact human approval.

## Research Intent Ledger

| Field | Declaration |
| --- | --- |
| Main question | Can the fixed typed-identity LGSSM HMC transition satisfy the fixed serious burn-in and retained-sampling controller under all-parameter convergence gates? |
| Candidate/mechanism | Transition `sha256:10d9a9d2...f16d6a` under serious execution `sha256:ceefd154...87e4`. |
| Exact baseline | Phase 5 V2/adoption/preflight/manifest evidence and the frozen serious controller contract below. There is no ranking comparator. |
| Expected failure mode | Slow mixing reaches a burn-in or retained cap with high R-hat or low ESS; alternatively a worker, XLA, identity, finite-value, wall-time, or artifact veto fires. |
| Promotion criterion | Every one of 18 raw parameters passes R-hat `<= 1.01`, bulk ESS `>= 1000`, and tail ESS `>= 400` on the final retained check; all engineering and numerical vetoes pass; the private sample and terminal manifest verify. |
| Promotion veto | Failure of any all-parameter retained diagnostic at the retained cap blocks Phase 8. |
| Continuation veto | Authority/reference mismatch, source or identity drift, output/archive mismatch, nonfinite state/sample/target/log-accept, available nonzero divergence telemetry, XLA/JIT fallback, worker/process failure, timeout, artifact corruption, public leak, or unclassified failure. |
| Repair trigger | A localized authority, archive, launcher, worker, serialization, resource, or artifact defect with intact transition and serious execution identities. A diagnostic-cap failure requires a separately reviewed tuning repair and is not repairable inside this run. |
| Explanatory diagnostics | Acceptance, intermediate R-hat/ESS trajectories, compile/runtime timing, PIDs, and descriptive sample summaries. |
| Must not be concluded | No posterior truth recovery, calibrated uncertainty, sampler superiority, production/default readiness, GPU readiness, NeuTra readiness, Phase 8 pass, or broad scientific validity. |

## Fixed Serious Runtime Contract

| Field | Fixed value |
| --- | --- |
| Target | Approved deterministic `T=120`, 18-parameter LGSSM replay |
| Transition | `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a` |
| Serious execution | `sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4` |
| Device | Deliberate CPU-only serious exception; `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import in parent and workers |
| Workers/chains | Two persistent spawned workers, two chains per worker, four chains total in stable order |
| Compilation | Sequential worker initialization; `tf.function(..., jit_compile=True)` / Host XLA only |
| Dtype | `float64` |
| Root seed | `(20260711, 701)` with the existing deterministic stage/check/worker derivation |
| Burn-in | Initial `2000`, window `1000`, extension `1000`, cap `16000` transitions per chain |
| Retained | Initial `4000`, check interval `2000`, extension `2000`, cap `40000` draws per chain |
| Diagnostics | Every raw parameter: R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400` |
| Timeout | `28800` seconds; machine protection only, not a scientific diagnostic |
| Serious artifacts | Configured `burnin_sampling.json`, `burnin_sampling_progress.json`, and protected `private_diagnostics/phase7_retained_samples.npz`, subject to the historical-result archive gate above |

This is an explicit CPU-hidden serious HMC exception. It does not change the
BayesFilter GPU/XLA default, provide GPU evidence, or authorize CPU NeuTra
training. NeuTra training remains a separate GPU-only lane.

## Separate Serious Runtime Authority

The active V2 config must remain byte-identical with
`runtime_authority=false`. Do not edit it to true, create a V2 copy with a
forged true value, or let a CLI flag bypass the controller refusal.

The Phase 7 pre-runtime implementation must introduce a distinct closed-schema
authority path with decision identifier:

`AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS`

The mechanism must include:

1. A pending serious authority proposal and terminal proposal manifest. They
   bind this reviewed subplan, the Phase 6 terminal output manifest and result,
   the Phase 5 V2 bundle, transition and serious execution identities, exact
   source/runtime inventory, historical-result archive manifest, fixed command,
   fixed output paths, counts, thresholds, timeout, public/private boundary,
   and ordered nonclaims.
2. A human statement accepted only in this exact shape:

   `I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS bound to Phase 7 authority proposal manifest <exact sha256:...>.`

3. A terminal serious authority record that binds the exact statement and
   manifest and grants one serious launch only. It grants no Phase 8 or NeuTra
   authority.
4. A permanent atomic `0400` launch claim created and crash-durably synced
   after all preclaim checks but before live-result replacement, log/output
   creation, worker creation, or an HMC transition. The claim is never deleted,
   rewritten, or reused, including on failure or interruption.
5. An unforgeable prepared serious launch context consumed once by
   `run_phase7`. Generic `run_phase7(smoke=False)` with the V2 config must keep
   failing while no verified serious context is supplied.
6. Descriptor-backed result, progress, log, private-sample, emergency-failure,
   and terminal output-manifest handling with exact path/parent/inode checks.
7. Child-side exact source and transition-identity verification before each
   worker's compile probe, matching or exceeding the Phase 6 smoke boundary.

Review agreement cannot grant this authority. The serious authority proposal
may be materialized only after implementation and no-runtime tests converge;
runtime begins only after the exact manifest-bound human statement is received.

## Required Artifacts

Before requesting serious runtime approval:

- immutable archive of the historical `burnin_sampling.json` blocker and a
  terminal archive manifest;
- separate serious authority implementation, parsers, verifiers, launcher,
  one-use claim, secure output session, and no-runtime tests;
- serious authority proposal and terminal proposal manifest;
- frozen implementation review and exact proposal-artifact review;
- refreshed Phase 7 result stub/ledger entry stating no serious runtime yet.

Only after exact human approval:

- terminal serious authority record;
- permanent serious launch claim;
- serious public progress and result at the reviewed configured paths;
- protected `phase7_retained_samples.npz` on success only;
- bounded serious execution log;
- ordinary terminal output-integrity manifest, or a distinct infrastructure
  failure plus infrastructure manifest if sealing the ordinary lane fails;
- Phase 7 result/close record with run manifest, decision table, inference
  status, three-ledger separation, and post-run red team;
- drafted and reviewed Phase 8 closeout/boundary subplan, still non-executable
  without separate Phase 8 approval.

## Required Checks, Tests, And Reviews

Before proposal materialization:

1. Reverify all Phase 5, Phase 6 attempt-1, and Phase 6 attempt-2 terminal
   evidence from exact bytes and hashes.
2. Verify the historical serious blocker, its immutable archive, and archive
   manifest before and after every proposal/authority boundary.
3. Test wrong/missing/stale approval, wrong manifest, wrong decision, wrong
   mode, static-boolean bypass, forged context, copied authority, claim replay,
   output collision, alias/symlink/hard-link, path replacement, partial write,
   and interruption cases without a real worker or transition.
4. Test that every count, threshold, seed, topology, environment, XLA/JIT,
   dtype, timeout, serious identity, and path mutation fails before claim or
   worker creation.
5. Test crash-durable one-use claim creation, bounded teardown, emergency
   evidence, preservation of a valid primary result, and no claim deletion.
6. Test historical-result archival and controlled live-path replacement. A
   failure before durable claim leaves the original live path unchanged; a
   failure after claim must leave either the original or a strict terminal
   result, never an unclassified partial replacement.
7. Test child-side source/transition drift before compile, worker persistence,
   stable chain order, deterministic seed derivation, exact state handoff, and
   no non-JIT fallback using mocks only before approval.
8. Run CPU-hidden focused and combined tests, Python compilation, forbidden
   bypass/repin scans, scoped whitespace checks, strict artifact parsing, and
   process-absence checks.
9. Obtain fresh bounded read-only review of the implementation and exact
   authority proposal. Use fresh Codex substitute review while the Claude
   disclosure rejection remains binding.

After the approved run:

1. Parse and cross-verify authority, claim, progress, result, private sample,
   log, and terminal manifest.
2. Independently inspect retained-sample shape, finiteness, config/replay
   provenance, raw hash, and protected file mode.
3. Verify two persistent worker PIDs, four chains, Host XLA/JIT, CPU hiding,
   versions, thread settings, compile traces, exact counts, and teardown.
4. Apply hard vetoes before interpreting diagnostics. Do not rank or promote
   on acceptance, timing, intermediate checks, or descriptive summaries.
5. Write the Phase 7 close record and review it with the non-executable Phase 8
   subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific/engineering question | Can the fixed typed transition complete serious burn-in and retained sampling under deterministic all-parameter gates? |
| Exact baseline | Phase 5 transition/serious identities and the fixed contract in this subplan. |
| Primary pass criterion | Terminal serious result passes every all-parameter retained R-hat/ESS threshold, all vetoes, protected-sample verification, and terminal-manifest verification. |
| Promotion vetoes | Any parameter fails R-hat, bulk ESS, or tail ESS at the retained cap; required diagnostic missing or nonfinite. |
| Continuation vetoes | Authority, identity, integrity, archive/output, finite-value, divergence, XLA, worker, timeout, public-boundary, or artifact veto listed above. |
| Explanatory only | Smoke diagnostics, intermediate serious checks, acceptance, PIDs, compile/runtime timing, and descriptive posterior summaries. |
| What passing will not prove | Posterior recovery, calibrated uncertainty, superiority, production/default readiness, GPU readiness, Phase 8 success, NeuTra readiness, or broad validity. |
| Preserving artifacts | Phase 5/6 terminals; historical blocker archive; serious proposal/authority/claim/progress/result/private sample/log/manifests; Phase 7 result and reviews. |

## Forbidden Claims And Actions

- Do not execute serious HMC before exact proposal-manifest-bound human
  approval.
- Do not mutate or reuse either smoke authority or claim.
- Do not edit the active V2 config's `runtime_authority=false` value.
- Do not overwrite the historical serious blocker before its exact immutable
  archive and archive manifest pass review and are bound by the human-approved
  proposal.
- Do not change transition mechanics, target, transforms, counts, thresholds,
  seeds, topology, threads, CPU hiding, XLA/JIT, dtype, timeout, or serious
  artifact paths inside the run.
- Do not manually extend, thin, exclude a chain, retune, restart from partial
  samples, or change a threshold after observing diagnostics.
- Do not treat smoke R-hat/ESS, acceptance, timing, or a hard-screen pass as
  evidence of convergence or superiority.
- Do not run Phase 8, posterior-recovery evaluation, NeuTra training, package
  installation, network fetches, default-policy changes, or unrelated lanes.
- Do not describe the CPU-hidden run as GPU evidence or as a change to the
  repository's GPU/XLA default.

## Exact Next-Phase Handoff Conditions

Phase 8 planning may begin only when:

1. the serious authority proposal and implementation passed local and fresh
   independent reviews;
2. the exact human approval bound to the terminal serious proposal manifest was
   recorded and consumed once;
3. the serious run reached a strict terminal result under the unchanged caps;
4. authority, claim, result/progress, sample, log, and output manifest verify;
5. every engineering/numerical hard veto and every final all-parameter
   convergence criterion passed;
6. the Phase 7 result distinguishes convergence evidence from posterior
   recovery and all broader nonclaims; and
7. the Phase 7 result and non-executable Phase 8 subplan pass review.

Even then, stop before Phase 8 runtime and request its separate human
authority. A Phase 7 diagnostic-cap failure hands off to a separately reviewed
tuning-repair plan, not Phase 8.

## Stop Conditions

- Any inherited Phase 5 or Phase 6 artifact, identity, mode, or cross-link
  drifts.
- The historical serious blocker cannot be archived and preserved exactly.
- A one-use serious authority cannot be made independent of the V2 static
  boolean and smoke authority.
- The exact serious proposal-manifest-bound human approval is absent,
  ambiguous, denied, stale, or bound to another manifest.
- Any serious output exists unexpectedly, aliases protected evidence, or
  cannot be handled without unreviewed destructive replacement.
- Any target, transition, execution contract, package environment, or public/
  private boundary would need to change.
- Any real worker or transition occurs during pre-approval checks.
- A hard continuation veto, diagnostic cap failure, invalid terminal artifact,
  timeout, or unclassified failure occurs during the approved run.
- The same substantive review blocker remains after five repair rounds.
- The serious run passes: write and review the close record, then stop before
  Phase 8 authority.
