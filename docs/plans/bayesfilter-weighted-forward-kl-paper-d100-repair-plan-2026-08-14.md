# Paper d100 repair plan (2026-08-14)

Status: `EXECUTED_PARTIAL_TERMINAL_2026-08-14`

This plan continues the terminal d100 result without overwriting any prior
artifacts. It addresses the user's concern about false rejection and separates
statistical-screen calibration from genuine transport failures.

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | Does a conservative 99.9% individual interval change the preserved Gaussian decisions, and which remaining funnel/runtime failures can be repaired under evidence rather than by relaxing a failed target law? |
| Mechanisms under test | (A) exact archive re-adjudication at one frozen 99.9% interval for every Gaussian structural diagnostic; (B) independent exact-law calibration and profile diagnostics for the funnel screen; (C) one bounded reverse-funnel tail-repair training hypothesis only if the cheap diagnostics support it. |
| Baseline | The frozen d100 target, replay, transport states, HMC archives, and 99% results from the 2026-08-13 terminal campaign. No prior sample or state is overwritten. |
| Promotion criterion | A candidate must pass sampler gates plus all predeclared target-law intervals at the new 99.9% level. The interval level applies uniformly to all arms in this plan. |
| Promotion veto | Hash mismatch, nonfinite value, invalid archive, failed sampler gate, or any target-law interval excluding its exact value. |
| Continuation veto | Shared target/runtime defect, corrupted archive, inability to reproduce deterministic diagnostics, or budget exhaustion before a bounded repair can answer its question. |
| Repair trigger | Gaussian 99.9% re-adjudication remains rejected; reverse funnel remains tail-biased after a cheap exact-law/profile check; or forward training profile identifies a bounded host/device bottleneck with a correctness-preserving fix. |
| Explanatory only | Runtime, retracing count, clipping, acceptance, selected `L`, individual discrepancies, and objective loss. |
| Must not be concluded | A 99.9% pass is not proof of superiority or exact equality; a Gaussian pass after interval widening is not evidence that the original 99% screen was wrong; a reverse-funnel repair is not a method-wide claim. |

## Default and assumption audit

| Choice | Provenance/status | Failure mode | Early check | Promotion status |
|---|---|---|---|---|
| 99.9% interval | New user-requested conservative hypothesis; critical value `3.2905267315` | Hides a real bias or gives an overly permissive per-test gate; multiplicity is still not formally controlled | Re-adjudicate every preserved Gaussian arm uniformly; compare effect/MCSE and 99% result | Reviewed diagnostic policy for this repair only |
| No Gaussian HMC rerun for first step | Derived: analytic interval is deterministic from archived draws | Re-running could change the question and spend budget without resolving calibration | Hash-bound archive verification and exact recomputation | Correct first diagnostic |
| Exact iid Gaussian calibration | Derived reference check | A chain-aware HMC discrepancy may not appear in iid draws | Generate independent exact draws and run the same structural summaries | Explanatory calibration |
| Reverse-funnel repair | Hypothesis from observed tail compression; not a default | More training may improve central loss without fixing tails | Profile tail coverage and evaluate selected checkpoints on exact tail summaries | Candidate repair only |
| Forward runtime profiling | Engineering diagnostic | Optimization may change numerical behavior or XLA contract | One bounded profiler run, no default change | Explanatory only |

## Staged execution

1. Implement a hash-bound adjudicator option for a uniform interval level and
   re-adjudicate Gaussian forward and reverse archives at 99.9%. Preserve the
   original 99% decisions and write new roots.
2. Generate a small independent exact Gaussian diagnostic calibration with the
   same retained shape and calculate the empirical false-rejection rate of the
   11-summary screen at 99% and 99.9%. This is calibration evidence, not a
   posterior pass.
3. Apply the same 99.9% gate mathematically to the preserved funnel reverse and
   forward archives. Do not relabel the funnel reverse as fixed if its tail
   deficit remains outside the widened interval.
4. Profile one forward-KL training update and one heldout-likelihood evaluation
   with TensorFlow profiler/XLA metadata, recording device placement and host
   synchronization. Do not alter the training objective or default path.
5. If reverse funnel tail failure remains and the profile confirms no harness
   defect, run one bounded tail-aware repair canary using the existing frozen
   target/replay contract. The repair may alter only the training objective
   weighting or architecture if the canary plan records it; it must use fresh
   output roots and downstream HMC before any claim.
6. Write a terminal repair result and update the reset memo with a decision table,
   inference-status table, strongest alternative explanation, and next action.

## Skeptical plan audit

| Audit question | Finding |
|---|---|
| Wrong baseline? | No. All first-step comparisons use the exact preserved archives and the same target law. |
| Proxy promoted? | No. 99.9% intervals remain a hard screen; calibration and runtime remain explanatory. |
| Missing stop condition? | No. Archive/hash failure stops adjudication; failed reverse repair remains candidate evidence and does not reject forward KL generally. |
| Unfair comparison? | No. The new level is applied uniformly to Gaussian and funnel arms; no selective rescue is allowed. |
| Hidden assumption? | Yes, the 99.9% level is a policy hypothesis, not a mathematically multiplicity-adjusted joint test. It is recorded explicitly and cannot support superiority. |
| Will artifacts answer the question? | Yes. New versioned adjudication, calibration, profile, and repair roots preserve commands, hashes, seeds, environment, and prior decisions. |
| Could a pass mislead us? | Yes: wider intervals can accept a biased candidate. The exact effect/MCSE table and independent calibration are required before interpretation. |

Audit verdict: `PASS_FOR_STAGED_EXECUTION`. The deterministic interval and iid
calibration phases completed. The GPU profile phase is pending trusted GPU
approval; its failed authorization is an infrastructure blocker, not evidence
about the algorithm. No reverse tail-repair training was launched because the
profile could not be completed under the declared execution contract.

## Execution outcome

See `docs/plans/bayesfilter-weighted-forward-kl-paper-d100-repair-result-2026-08-14.md`.

- 99.9% was applied uniformly to Gaussian and funnel archives.
- Gaussian reverse passed; Gaussian forward remained rejected.
- Funnel reverse remained rejected; funnel forward remained passed.
- iid calibration completed with 32 replications of 4,000 exact draws:
  all-11 pass rate `0.78125` at 99% and `0.96875` at 99.9%.
- The trusted GPU profile was attempted once and rejected by the approval
  service with HTTP 502 before process creation. No workaround or scientific
  claim was made from that missing profile.
