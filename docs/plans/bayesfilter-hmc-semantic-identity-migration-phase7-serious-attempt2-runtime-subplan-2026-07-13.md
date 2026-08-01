# Phase 7 Serious Attempt-2 Runtime Subplan

Date: 2026-07-13

Status: `SUPERSEDED_BY_ACADEMIC_CAMPAIGN_SUBPLAN`

Governance supersession, 2026-07-13: the owner retired exact hash-bound
approval, one-use authority/claim, and per-launch security ceremony for trusted
local academic work. This document is historical. The active Phase 7 plan is
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-subplan-2026-07-13.md`.

## Phase Objective

Consume one exact attempt-2 manifest-bound human approval, materialize one
closed-schema one-use authority, and execute the unchanged serious
two-worker CPU-hidden Host-XLA/JIT HMC controller exactly once. Preserve all
attempt-1 terminal evidence and use only the versioned attempt-2 output paths.

The phase asks whether the fixed typed transition can complete serious burn-in
and retained sampling under the predeclared all-parameter convergence gates.
It does not ask whether the posterior recovers truth or whether this sampler is
superior, production-ready, GPU-ready, or suitable for NeuTra.

## Entry Conditions Inherited From The Repair Phase

1. The attempt-2 pre-runtime result records
   `PASS_PHASE7_SERIOUS_ATTEMPT2_PRERUNTIME_GATE_STOP_BEFORE_AUTHORITY_AND_RUNTIME`.
2. Proposal
   `sha256:e851b313f08e935f6bf4d67dca22448862e072dffc0fe32609580327e95182f4`
   remains exactly 39904 bytes with file SHA-256
   `cb026193af3506719ecc17858979b4005b6a19a8eb2b8ad6d34a3800c60d0ab7`
   and mode `0600`.
3. Terminal proposal manifest
   `sha256:64774b7c949386daf42d73291dbe2cccdc535625e92ab98ed349337c4d46e15e`
   remains exactly 869 bytes with file SHA-256
   `e7aa19fb234dd3eff960e97c0c50a643c98663a6e87c98170a9c0f09c9a991b6`
   and mode `0600`.
4. Exact proposal and terminal-manifest reviews end `VERDICT: AGREE` after the
   documented canonical-hash correction.
5. Attempt-1 authority and claim remain consumed; its complete terminal graph,
   empty reservations, archive, historical subplan, and result note verify
   exactly. The v1 approval is permanently non-actionable.
6. Every attempt-2 authority, claim, result, progress, sample, log, ordinary
   manifest, and infrastructure path remains absent.
7. The proposal-bound source inventory, documents, Phase 5/6 artifacts,
   transition identity, serious execution identity, interpreter, environment,
   command, and paths remain exact.
8. No matching serious launcher, controller, worker, or attempt-2 process is
   alive.
9. This subplan has passed fresh bounded read-only review.
10. The owner supplies exactly:

    `I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS_ATTEMPT2 bound to Phase 7 attempt-2 authority proposal manifest sha256:64774b7c949386daf42d73291dbe2cccdc535625e92ab98ed349337c4d46e15e.`

No prior approval, paraphrase, placeholder, different decision identifier, or
different manifest hash satisfies entry condition 10.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Reverify the exact proposal-bound transition, serious execution identity, source inventory, Phase 5/6 artifacts, and complete attempt-1 terminal graph before authority and claim. |
| Proxy promoted | Phase 6 smoke and pre-runtime checks establish mechanics only. Final promotion requires every declared all-parameter serious diagnostic and hard veto. |
| Missing stop | Authority, preclaim, output, worker, XLA, finite-value, divergence, timeout, diagnostic-cap, artifact, public-boundary, Phase 8, and NeuTra stops are explicit. |
| Unfair comparison | No candidate ranking occurs. The run is evaluated only against its fixed convergence and integrity criteria. |
| Hidden assumption | The proposal fixes target, seeds, counts, thresholds, topology, threads, CPU hiding, dtype, XLA/JIT, timeout, command, paths, and nonclaims. |
| Stale context | Immediately recheck every proposal-bound reference and all attempt-2 absences before authority creation and again before launch. Drift blocks; it is never silently repinned. |
| Environment mismatch | Use only the proposal-bound interpreter and launcher, with `CUDA_VISIBLE_DEVICES=-1` and fixed thread variables set before framework import. |
| Artifact cannot answer question | Require a strict terminal result or infrastructure terminal, permanent claim, bounded log, protected samples on success, and a terminal output manifest. |
| Attempt-1 mutation | Active attempt-2 output handling uses exclusive creation only. Any attempt-1 byte, mode, link, path, archive, or required-absence drift blocks. |

Audit verdict: `PASS_CONDITIONED_ON_EXACT_APPROVAL_AND_FINAL_ENTRY_RECHECKS`.

## Research Intent Ledger

| Field | Declaration |
| --- | --- |
| Main question | Can the fixed typed LGSSM HMC transition complete the serious controller and satisfy all final convergence gates? |
| Candidate/mechanism | Transition `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a` under serious execution `sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4`. |
| Expected failure mode | Slow mixing reaches a burn-in or retained cap; alternatively an authority, worker, XLA, finite-value, timeout, or artifact veto fires. |
| Promotion criterion | All 18 parameters pass final R-hat `<=1.01`, bulk ESS `>=1000`, and tail ESS `>=400`; all hard vetoes pass; protected samples and terminal manifest verify. |
| Promotion veto | Any required final diagnostic is missing, nonfinite, or fails its threshold at the retained cap. |
| Continuation veto | Authority/reference drift, source/identity drift, attempt-1 drift, output collision, nonfinite state/sample/target/log-accept, available nonzero divergence telemetry, non-JIT fallback, worker failure, timeout, public leak, invalid terminal artifact, or unclassified failure. |
| Repair trigger | A localized launcher, authority, worker, serialization, resource, or artifact defect with intact target and typed identities. A diagnostic-cap failure requires a separate tuning plan. |
| Explanatory only | Acceptance, intermediate R-hat/ESS, PIDs, compile/runtime timing, and descriptive sample summaries. |
| Must not be concluded | Posterior recovery, calibrated uncertainty, sampler superiority, production/default/GPU readiness, Phase 8 success, NeuTra readiness, or broad scientific validity. |

## Fixed Runtime Contract

| Field | Fixed value |
| --- | --- |
| Target | Deterministic `T=120`, 18-parameter LGSSM replay |
| Device | Deliberate CPU-only serious exception; `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import in parent and workers |
| Workers/chains | Two persistent spawned workers, two chains per worker, four chains in stable order |
| Compilation | Sequential worker initialization; Host XLA through `jit_compile=True`; no fallback |
| Dtype | `float64` |
| Root seed | `(20260711, 701)` with the proposal-bound derivation |
| Burn-in | Initial `2000`, window `1000`, extension `1000`, cap `16000` transitions per chain |
| Retained | Initial `4000`, interval `2000`, extension `2000`, cap `40000` draws per chain |
| Diagnostics | Every raw parameter: R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400` |
| Wall-time cap | `28800` seconds, a machine-protection veto rather than a scientific diagnostic |
| Output policy | Exclusive creation of all versioned attempt-2 outputs; no historical replacement |

## Required Artifacts

- exact attempt-2 proposal and terminal proposal manifest;
- one terminal attempt-2 authority at the proposal-bound path;
- one permanent mode-`0400` attempt-2 launch claim;
- versioned public progress and terminal result, or a strict infrastructure
  failure plus infrastructure manifest;
- protected versioned private retained samples on successful sampling only;
- bounded versioned serious log;
- ordinary terminal attempt-2 output manifest on normal controller completion;
- Phase 7 attempt-2 runtime result/close record with run manifest, decision
  table, inference status, three ledgers, and post-run red team; and
- drafted and reviewed Phase 8 closeout subplan only after a valid Phase 7 pass.

## Required Commands And Execution Order

1. Re-run exact read-only entry verification with the proposal-bound
   interpreter and environment. Do not create authority if any check drifts.
2. After matching the exact approval statement, run the authority builder once:

   ```bash
   /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 scripts/build_hmc_phase7_serious_authority.py \
     --approval-statement '<exact approved statement>' \
     --approval-date '<YYYY-MM-DD receipt date>'
   ```

3. Strictly parse and cross-check the authority, then recheck all remaining
   attempt-2 runtime-path absences and process absence.
4. Run the exact proposal-bound launcher once, visibly in the current session:

   ```bash
   /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 \
     scripts/run_hmc_phase7_typed_identity_serious.py \
     --stage burnin_sampling \
     --phase7-serious-authority \
     docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority.json
   ```

5. Do not rerun the launcher, including after an infrastructure or diagnostic
   failure. The authority and permanent claim are one-use evidence.

## Required Checks, Tests, And Reviews

Before authority:

1. Strictly parse and reconstruct the proposal and terminal manifest from one
   pinned evidence session.
2. Verify exact source/document/file hashes, schemas, byte counts, modes,
   parent identities, attempt-1 terminal semantics, archive, and absences.
3. Verify the exact interpreter, TensorFlow/TFP versions, CPU hiding, thread
   environment, XLA/JIT setting, runtime counts, seeds, topology, thresholds,
   timeout, command, and paths.
4. Verify all attempt-2 authority/claim/runtime paths and filtered processes
   are absent.
5. Do not rerun the full test suite or Phase 6 smoke unless drift makes the
   proposal invalid. Proposal-bound drift is a stop, not a reason to repin.

After authority but before launch:

1. Parse the closed-schema authority and verify its exact proposal-manifest
   reference, statement, date, one-launch scope, and Phase 8/NeuTra denials.
2. Reverify proposal-bound evidence, runtime-path absence except authority, and
   process absence.

After the one launch:

1. Parse and cross-verify authority, permanent claim, progress, result or
   infrastructure terminal, log, private sample if present, and terminal
   manifest.
2. Independently inspect protected sample mode, hash, shape, finiteness,
   config, replay provenance, worker metadata, and stable chain ordering.
3. Verify two persistent worker PIDs, four chains, CPU hiding, float64 Host
   XLA/JIT, fixed thread settings, compile traces, exact counts, and teardown.
4. Apply hard vetoes before interpreting diagnostics. Never promote acceptance,
   timing, smoke metrics, or intermediate convergence checks.
5. Write and review the terminal attempt-2 result. Draft/review Phase 8 only if
   the strict serious pass criterion is met; otherwise write the appropriate
   blocker or tuning-repair handoff.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific/engineering question | Can the fixed typed transition complete serious burn-in and retained sampling under deterministic all-parameter gates? |
| Exact comparator | The fixed convergence and integrity contract in the approved attempt-2 proposal; no sampler ranking comparator. |
| Primary pass criterion | Strict terminal result passes all 18-parameter R-hat/ESS gates, all hard vetoes, private-sample verification, and terminal-manifest verification. |
| Promotion vetoes | Any final parameter diagnostic missing, nonfinite, or outside its threshold at the retained cap. |
| Continuation vetoes | Authority, identity, attempt-1 integrity, output, finite-value, divergence, XLA, worker, timeout, public-boundary, artifact, or unclassified failure. |
| Explanatory only | Smoke results, intermediate checks, acceptance, PIDs, timing, and descriptive posterior summaries. |
| What passing will not prove | Posterior recovery, calibrated uncertainty, superiority, production/default/GPU readiness, Phase 8 success, NeuTra readiness, or broad validity. |
| Preserving artifact | Attempt-2 authority/claim/progress/result/sample/log/manifests and the terminal result note. |

## Forbidden Claims And Actions

- Do not accept the consumed attempt-1 approval or any non-exact attempt-2
  statement.
- Do not create authority, claim, outputs, or workers before exact approval.
- Do not edit or regenerate proposal-bound source, tests, documents, artifacts,
  environment, identities, command, paths, counts, thresholds, or nonclaims.
- Do not mutate, restore, delete, chmod, relink, or reuse attempt-1 evidence.
- Do not rerun Phase 6 smoke or attempt 1.
- Do not rerun attempt 2 after its authority or claim is consumed.
- Do not switch off JIT, expose a GPU, change threads, retune, thin, exclude a
  chain, restart from partial samples, or change thresholds after diagnostics.
- Do not treat a smoke pass, hard-screen pass, acceptance, or timing as evidence
  of convergence, recovery, or superiority.
- Do not run Phase 8, posterior-recovery evaluation, NeuTra training, package
  installation, network fetches, default-policy changes, or unrelated lanes.
- Do not describe this CPU-hidden exception as GPU evidence or a repository
  default change.

## Exact Next-Phase Handoff Conditions

Phase 8 planning may begin only if:

1. the exact attempt-2 approval was consumed once;
2. the one serious run reached its strict normal terminal result;
3. authority, claim, result, progress, log, private sample, and ordinary output
   manifest verify exactly;
4. every engineering and numerical continuation veto passed;
5. all 18 final parameter diagnostics passed their fixed thresholds;
6. the terminal result separates convergence evidence from recovery and all
   broader nonclaims; and
7. the attempt-2 result and a non-executable Phase 8 subplan pass review.

Even then, stop before any Phase 8 runtime or NeuTra action. A diagnostic-cap
failure hands off to a separately reviewed tuning-repair plan; an
infrastructure failure hands off to a separately reviewed mechanical repair.

## Stop Conditions

- The exact attempt-2 approval is missing, stale, ambiguous, or mismatched.
- Any proposal-bound byte, hash, schema, source, document, identity,
  environment, command, path, or contract value drifts.
- Any attempt-1 artifact or required absence drifts.
- Any attempt-2 runtime path exists unexpectedly before its permitted stage.
- A worker or transition occurs before the permanent claim is durably consumed.
- Authority construction or prelaunch verification fails.
- The one launcher exits with an infrastructure, controller, diagnostic-cap,
  timeout, integrity, or unclassified failure.
- A valid terminal pass is produced; write/review the close record and stop
  before Phase 8.
- The same material review blocker remains after five substantive repair rounds.
