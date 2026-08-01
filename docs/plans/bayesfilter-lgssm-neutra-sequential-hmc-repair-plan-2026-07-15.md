# LGSSM NeuTra Sequential Warm-Up And Sampling Repair Plan

Date: 2026-07-15  
Status: `READY_AFTER_SKEPTICAL_AUDIT`  
Parent: `docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-plan-2026-07-15.md`

## Problem And Objective

The executed NeuTra Phase 4 used a fixed 1,000 burn-in transitions and then a
fixed 1,000 retained draws per chain. It discarded the burn-in path and treated
a terminal modern R-hat miss as candidate rejection. That is a planning and
integration error: BayesFilter already has sequential retained-draw machinery,
but this TensorFlow-only NeuTra route bypassed it, and the existing reusable
verifier still accepts only a fixed, discarded initial burn-in.

Repair the NeuTra route so that it:

- retains and archives every warm-up draw while excluding warm-up from
  posterior estimation;
- checks a recent warm-up window after sequential chunks and continues until
  the maximum of rank-normalized split and folded rank-normalized split R-hat
  is at most `1.05`, or a 10,000-transition-per-chain warm-up cap is hit;
- collects posterior draws in sequential chunks and recomputes the same modern
  R-hat cumulatively until it is at most `1.01`, or 10,000 retained draws per
  chain are collected; and
- preserves all old artifacts as historical evidence while superseding their
  invalid terminal admission decision.

## Research Intent And Evidence Contract

| Item | Frozen contract |
| --- | --- |
| Main question | Do either of the two already trained and frozen NeuTra candidates support a healthy fixed HMC kernel when warm-up and retained sampling use sequential modern-R-hat stopping? |
| Candidate/mechanism | Existing `dense_seed1201` and `dense_seed1202` transports; selected HMC step size `0.8`; 10 leapfrog steps; no retraining or retuning |
| Exact baseline | Historical Phase 4 fixed 1,000/1,000 execution, retained only as incomplete diagnostic evidence; tuned plain-HMC remains the later posterior comparator |
| Warm-up readiness criterion | After at least 2,000 transitions per chain, maximum modern R-hat `<=1.05` on the latest 1,000 archived warm-up transitions in raw model coordinates |
| Tuning admission criterion | After warm-up passes, cumulative retained draws reach at least 1,000 per chain and maximum modern R-hat `<=1.01` in raw model coordinates |
| Hard vetoes | Nonfinite state, target, log acceptance, or R-hat; invalid target-status telemetry; energy-error divergence screen; no chain movement; identity/artifact drift; warm-up or retained cap reached without its R-hat pass |
| Repair trigger | Local schema, archive, XLA, or controller defect under the unchanged target/kernel/budget triggers focused repair and a fresh versioned attempt |
| Explanatory only | Acceptance, runtime, individual rank/folded R-hat values, and the number of chunks needed |
| Forbidden conclusions | Warm-up R-hat is not posterior convergence; passing tuning admission is not posterior correctness, recovery, superiority, calibration, robustness, production readiness, or default readiness |

Warm-up samples are retained as diagnostic artifacts, not mixed into posterior
draws. The readiness window is recent rather than cumulative because early
nonstationary warm-up values should not permanently veto a later stable kernel.
Both rank and folded rank-normalized split R-hat are checked: folded-only would
miss location nonconvergence. The term "modern R-hat" below always means the
per-parameter maximum of those two quantities.

## Default And Assumption Audit

| Choice | Provenance and status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Warm-up minimum 2,000, window/chunk 1,000 | BayesFilter Phase 7 controller pattern, adapted as a target-specific hypothesis | Window too short or readiness passes by chance | preserve every check; later retained R-hat remains an independent gate |
| Warm-up threshold `1.05` | Owner-proposed predetermined readiness threshold | Too loose for posterior claims | warm-up is readiness only; retained threshold remains `1.01` |
| Warm-up cap 10,000 | Conservative interpretation of the owner maximum and bounded-compute policy | A viable slow-mixing kernel may fail at cap | classify as kernel/candidate failure, not target or NeuTra-direction failure |
| Retained chunk/minimum 1,000 | Continuity with prior tuning admission and existing sequential verifier | A short stochastic pass may not persist | later confirmatory ESS/R-hat/agreement/recovery phase uses fresh evidence |
| Retained cap 10,000 | Explicit owner directive and existing generic verifier hard cap | Candidate fails despite eventual convergence beyond cap | reject only this fixed kernel under this budget |
| Raw-coordinate R-hat | Existing Phase 4 contract | Latent mixing can hide model-coordinate behavior | compute diagnostics after frozen transport forward map |
| Fresh repair seeds | Avoids optional stopping on already observed fixed-run draws | Seed-specific pass/fail | run both independently trained candidates and forbid ranking |

## Skeptical Plan Audit

Audit date: 2026-07-15. The plan was checked for wrong baselines, proxy
promotion, missing stop conditions, unfair comparison, stale assumptions,
environment mismatch, and artifacts that would not answer the question.

Findings and repairs:

1. Reusing the old fixed 1,000 retained arrays and merely appending draws would
   condition the new stopping rule on an already inspected terminal result.
   The repair uses fresh warm-up and retained seed roots.
2. Treating folded R-hat alone as sufficient would miss location failures.
   Both rank and folded components remain mandatory.
3. Treating all cumulative warm-up as stationary evidence would make early
   transient draws contaminate readiness. All warm-up is archived, but the
   declared readiness diagnostic uses the latest 1,000-draw window.
4. Feeding warm-up samples into posterior estimation would change the sampled
   distribution. Warm-up and retained archives are separate and only the latter
   may enter posterior summaries.
5. The existing generic sequential verifier implements the retained loop but
   fixes and discards initial burn-in; importing it would also violate this
   campaign's TensorFlow-only import closure. The repair adds the missing
   controller to the active TensorFlow/TFP route and tests its stopping logic.

Audit verdict: `PASS_AFTER_REVISION`. The commands below can answer the stated
question without changing the target, transports, fixed kernel, hardware class,
or scientific promotion thresholds.

## Compute And Attempt Budget

- Focused CPU-hidden unit/integration checks: at most 30 minutes.
- One fresh sequential attempt per candidate, four batched chains, XLA,
  float64, up to 10,000 warm-up plus 10,000 retained transitions per chain.
- Aggregate campaign wall ceiling for both candidates: 6 hours.
- One localized harness/XLA/archive retry per candidate in a fresh versioned
  directory; no retry for a genuine health or R-hat cap failure.
- No retraining, package/environment mutation, network fetch, GPU job, paid
  compute, destructive action, public act, or default-policy change.

## Phase R0 - Documentation And Controller Repair

Objective: supersede the invalid fixed-budget decision and implement a
TensorFlow-only sequential warm-up/retained controller.

Entry conditions: Phase 1-3 candidate identities passed; historical Phase 4
files are immutable; both selected repair-grid kernels are healthy and use step
size `0.8`, 10 leapfrog steps.

Required artifacts: this plan; amended parent/result/reset notes; controller
code; focused tests for early pass, extension, cap failure, warm-up retention,
warm-up/posterior separation, max(rank, folded) use, and 10,000 caps.

Checks/review: compile; focused pytest; TensorFlow-only import audit; diff
check; one bounded read-only material plan review when Claude is available.

Handoff: continue to R1 only if local checks pass and no review finding exposes
a target, diagnostic, seed, artifact, or budget defect. Procedural reviewer
unavailability is recorded and is not a scientific blocker.

Stop: broken modern-R-hat implementation, inability to archive warm-up, target
identity drift, or a required change beyond this contract.

## Phase R1 - Fresh Sequential Verification

Objective: run both existing frozen candidates through the corrected
controller without rerunning training or the already completed finite grid.

Fresh seed roots:

| Candidate | Warm-up root | Retained root |
| --- | --- | --- |
| `dense_seed1201` | `(20260715, 4101)` | `(20260715, 4201)` |
| `dense_seed1202` | `(20260715, 4301)` | `(20260715, 4401)` |

Required artifacts: versioned per-candidate result; separate TensorFlow tensor
archives for every warm-up and retained chunk; cumulative final warm-up and
retained archives; per-check R-hat/health rows; exact target, transport,
adapter, kernel, seed, environment, XLA/CPU-hidden, wall-time, plan, and output
manifest fields; aggregate Phase R1 decision.

Checks: warm-up must pass before retained sampling starts; all chunks must pass
health; retained admission uses cumulative raw draws; archive hashes and shapes
must verify; no output overwrite; total wall budget must remain.

Handoff: if at least one candidate passes, freeze its corrected kernel identity
and refresh the confirmatory Phase 5 subplan to use its still-unused original
confirmation seed. If none passes, write a candidate/kernel failure result;
do not infer target or research-direction failure.

Stop: true hard veto, either cap failure for that candidate, corrupted artifact,
or total campaign budget exhaustion. One candidate failure does not stop the
other.

## Phase R2 - Confirmatory Sampling And Closeout

Objective: for each admitted candidate, run the parent plan's fresh
confirmatory posterior checks, increasing retained draws under the same 10,000
cap if modern R-hat is not yet sufficient, then apply the already declared
bulk ESS, tail ESS, posterior agreement, recovery, health, and divergence gates.

Entry: immutable R1 admission and fixed kernel; unused confirmation seed;
refreshed Phase 5 subplan with exact remaining budget.

Handoff: write the terminal decision/inference tables and reset memo. A positive
claim is limited to the exact favorable LGSSM fixture and recorded candidate.

Stop: confirmatory hard veto, cap failure, comparator mismatch, missing MCSE or
ESS evidence, corrupted artifact, or exhausted total budget.

## Phase Procedure

At the end of each phase: run local checks; write a phase close record; refresh
the next subplan from observed evidence and remaining budget; review its
suitability; and continue unless a real scientific, numerical, artifact,
hardware, privacy, external, or budget blocker exists.
