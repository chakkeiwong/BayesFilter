# Phase 7 Academic Serious-HMC Campaign Subplan

Date: 2026-07-13

Status: `TERMINAL_DIAGNOSTIC_CAP_FAILURE_NO_RETRY`

## Phase Objective

Run the fixed typed-identity deterministic LGSSM HMC serious campaign under
academic-research governance, using versioned outputs and the unchanged
scientific criteria. Retire the old per-launch authority/claim protocol from
the active route without deleting its historical artifacts.

The user supplied the later plain-language request to continue Phase 7 and the
rest of the runbook on 2026-07-13. That request authorizes this fixed campaign
within the declared three-launch and eight-hour budget.

## Entry Conditions

- Phase 5 typed transition identity remains
  `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a`.
- Serious execution identity remains
  `sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4`.
- Phase 6 mechanics smoke passed with two workers, four chains, CPU-hidden
  float64 Host XLA/JIT, and finite tiny-smoke samples.
- Phase 7 serious attempt 1 failed before workers at output reservation; it
  produced no sampler evidence.
- Existing proposal/authority/claim artifacts remain historical and are not
  overwritten or treated as active execution permission.
- No serious Phase 7 process is running when the campaign begins.

## Scientific Contract

| Field | Fixed value |
| --- | --- |
| Question | Can the fixed typed transition complete serious burn-in and retained sampling and pass every all-parameter convergence gate? |
| Target | Deterministic `T=120`, 18-parameter LGSSM replay |
| Device | Deliberate CPU-only HMC exception with `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import |
| Workers/chains | Two persistent workers, two chains per worker, four chains in stable order |
| Compilation | TensorFlow/TFP float64 Host XLA with `jit_compile=True`; no non-JIT fallback |
| Root seed | `(20260711, 701)` with the existing deterministic derivation |
| Burn-in | Initial `2000`, window `1000`, extension `1000`, cap `16000` transitions per chain |
| Retained | Initial `4000`, interval/extension `2000`, cap `40000` draws per chain |
| Promotion criterion | Every raw parameter passes R-hat `<=1.01`, bulk ESS `>=1000`, and tail ESS `>=400`; all engineering and numerical vetoes pass |
| Campaign budget | Eight total wall-clock hours across at most three launches, including infrastructure repair and retry |
| Output policy | New unique run directory per launch; never overwrite a prior run or historical authority artifact |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Exact comparator | The fixed transition and serious scientific contract above; there is no sampler-ranking comparator |
| Promotion veto | Any required final parameter diagnostic is missing, nonfinite, or outside its threshold at the retained cap |
| Continuation veto | Invalid target/config, source or typed-identity drift affecting computation, nonfinite state/target/sample/log-accept, available nonzero divergence telemetry, XLA/JIT fallback, worker failure not repairable within budget, corrupted/missing result artifacts, public/private leak, or total budget exhaustion |
| Repair trigger | A localized launcher, multiprocessing, serialization, path, or resource failure while target, method, criteria, hardware class, privacy, and total budget remain unchanged |
| Explanatory only | Smoke diagnostics, intermediate R-hat/ESS, acceptance, PIDs, and compile/runtime timings |
| Not concluded | Posterior recovery, calibrated uncertainty, sampler superiority, production/default/GPU readiness, Phase 8 success, NeuTra readiness, or broad validity |

## Active Implementation Route

Before the first launch, replace the legacy authority-dependent entry point with
the smallest local academic campaign wrapper that:

1. loads and validates the existing fixed V2 config and typed identities;
2. sets the fixed CPU/thread environment before TensorFlow import;
3. creates a unique versioned run directory using exclusive directory creation;
4. writes an ordinary run manifest with commit, command, config/identity hashes,
   environment, seeds, budget, attempt number, and artifact paths;
5. invokes the existing serious controller without a cryptographic approval or
   permanent launch-claim dependency;
6. writes progress, result, protected retained samples, and a final ordinary
   checksum manifest; and
7. preserves every historical Phase 6/7 artifact without mutation.

Focused tests must cover config/identity drift, output-directory collision,
attempt numbering, campaign-budget accounting, failure classification, and
ordinary terminal artifact validation. Do not extend the legacy authority,
claim, inode, or reservation implementation for this active route.

## Historical Repair And Retry Contract

An infrastructure failure may be repaired and retried automatically when:

- no scientific or numerical transition result invalidates the campaign;
- target, data, method, identities, counts, thresholds, hardware class, privacy,
  and total budget are unchanged;
- the failed attempt has a terminal result or failure note;
- a focused regression covers the repair; and
- the next launch uses a fresh run directory.

Stop after three launches or eight cumulative wall-clock hours. A change to the
scientific contract or budget requires new plain-language user direction, not a
magic approval phrase.

Attempt 1 ended at `diagnostic_cap_failure`, not infrastructure failure. The
retry route above is therefore closed regardless of the unused nominal budget.

## Review Policy

No additional procedural review is required before implementing or launching
this already-audited fixed campaign. Use independent review only if the active
implementation changes the scientific contract, introduces a material
numerical risk, or supports a publication/default claim. Reviewer unavailability
does not block the campaign when focused local checks pass.

## Pre-Launch Skeptical Audit

The 2026-07-13 pre-launch audit initially found four material evidence flaws:

- a strict result could survive failure to write terminal progress, leaving a
  stale or incomplete progress artifact;
- strict validation did not bind burn-in/retained counts, the full diagnostic
  object, and the private archive to the fixed serious schedule;
- some governed-source or identity drift could be classified as retryable
  infrastructure failure; and
- a dispatched worker transition could be recorded as unexecuted when parent
  response validation failed.

The repair was limited to terminal progress/result cross-validation, complete
diagnostic and schedule validation, conservative continuation-veto
classification, truthful transition-dispatch accounting, and focused tests.
The scientific target, thresholds, worker topology, CPU/XLA execution route,
seeds, and campaign budgets remain unchanged.

The repaired implementation received a scoped convergence verdict of
`VERDICT: AGREE`. The complete CPU-hidden local gate then passed `31` tests in
`32.90 s`; `py_compile` and `git diff --check` also passed. The earlier
incomplete pytest invocation was a test-runner/tool-session anomaly and did not
recur under verbose bounded execution. All four audit findings are closed, so
the skeptical audit now passes for launch of the fixed campaign.

## Result Requirements

The terminal campaign note must include:

- every launch command, commit, environment, seeds, attempt wall time, and
  cumulative wall time;
- unique output paths and ordinary SHA-256 checksums;
- failure/repair history;
- final hard-veto and all-parameter diagnostic status;
- decision and inference-status tables;
- separation of engineering correctness, sampler validity, and scientific
  interpretation; and
- a post-run red team and explicit nonclaims.

## Stop And Handoff

- Stop for invalid scientific inputs, material contract drift, missing required
  diagnostics, unrepairable artifact corruption, privacy leakage, or budget
  exhaustion.
- Do not stop merely because a legacy approval token, claim file, proposal hash,
  descriptor invariant, or reviewer verdict is absent.
- On a strict Phase 7 pass, write the terminal result and draft the Phase 8
  scientific evaluation plan. Phase 8 execution remains a separate research
  campaign, not a security-authority ceremony.
- On diagnostic-cap failure, write a tuning/research result rather than
  misclassifying the candidate failure as infrastructure failure.

## Terminal Outcome

Attempt 1 reached the `16000` burn-in cap. Diagnostics were finite, bulk ESS
and tail ESS passed, and no hard veto was recorded, but eight parameters failed
R-hat `<=1.01`; maximum R-hat was `1.043456525609825`. Retained sampling did
not begin. The checksum-verified result is recorded in
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-result-2026-07-13.md`.

The fixed candidate is rejected under this screen. Do not retry this campaign
or execute Phase 8 scientific runtime or NeuTra. Phase 8 may only close the
documentation and handoff boundary.
