# BayesFilter NeuTra HMC Core Consolidation And Robustness Program

Date: 2026-07-15  
Status: `PROGRAM_COMPLETE_WITH_DISCLOSED_COMMAND_TRANSCRIPT_CAVEAT`  
Supervisor/executor: Codex  
Claude role: bounded read-only reviewer  
Active output root:
`docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/`

## Objective

Consolidate the sequential warm-up and retained-sampling repair proved by the
LGSSM NeuTra campaign into a shared TensorFlow/TensorFlow Probability inference
API, make that API the default for claim-bearing NeuTra HMC, classify and
migrate the repository's NeuTra HMC routes without rewriting historical
evidence, and run a bounded robustness ladder beyond the two original training
seeds and favorable fixture.

This is a phased program because it combines a cross-cutting API/default change
with new stochastic research evidence. Engineering consolidation must close
before robustness experiments begin.

## Research Intent Ledger

| Item | Contract |
| --- | --- |
| Main engineering question | Can all active claim-bearing NeuTra HMC use one TensorFlow-only sequential controller that archives warm-up, excludes it from posterior draws, and grows retained sampling under declared convergence gates? |
| Main scientific question | Does the positive `dense_seed1201` exact-fixture result persist for an additional independent training seed, and does NeuTra remain viable on one genuinely new LGSSM data fixture after target-specific training and a valid tuned plain-HMC comparator? |
| Mechanism | Shared fixed-kernel batched HMC controller plus fresh target-specific dense-IAF training and downstream HMC validation |
| Expected engineering failure | Campaign-specific artifact assumptions leak into core; fixed burn-in/fixed retained counts remain reachable as serious defaults; NumPy enters active inference; warm-up is accidentally pooled with posterior draws |
| Expected scientific failure | A new training seed or new fixture yields energy errors, nonfinite target status, failed modern R-hat/ESS, posterior disagreement, or poor recovery |
| Promotion criterion | Report three independent outcomes: engineering consolidation, same-fixture additional-seed evidence, and new-fixture evidence. Engineering promotion requires C0-C2. A positive joint robustness answer requires both S1 and F2 to complete and pass their own target-specific downstream HMC contracts; one passing arm supports only that arm's narrow claim. |
| Promotion veto | Any claim-bearing route still bypasses the shared controller; core imports NumPy or host callbacks; archive/seed/target identity drift; invalid diagnostic; hard HMC health veto; comparator mismatch |
| Continuation veto | Broken target or harness; corrupted artifacts; no valid comparator for the new fixture; unavailable trusted GPU for authorized training; total campaign budget exhaustion |
| Repair trigger | Local import, schema, serialization, XLA, archive, route-dispatch, or reporting defect under unchanged target/method/budget |
| Explanatory diagnostics | Acceptance, energy-error magnitudes/counts, number of chunks, runtime, training/heldout loss, per-parameter R-hat/ESS/agreement/recovery |
| Forbidden conclusions | One or two additional arms do not prove calibration, broad robustness, superiority, production readiness, or universal NeuTra reliability |

## Canonical Shared NeuTra HMC Policy

For claim-bearing BayesFilter NeuTra HMC:

- use TensorFlow/TFP batched chains and XLA by default;
- retain and archive every warm-up chunk, but never include warm-up in
  posterior estimates;
- use at least four chains for modern R-hat decisions;
- check warm-up on a predeclared recent window using the per-parameter maximum
  of rank-normalized split and folded rank-normalized split R-hat;
- default warm-up readiness to at least 2,000 transitions per chain, latest
  1,000-transition window, threshold `<=1.05`, and maximum 10,000 per chain;
- grow retained sampling cumulatively, with a minimum set by the evidence role
  and maximum 10,000 per chain;
- use modern R-hat `<=1.01` for tuning admission; confirmation additionally
  requires bulk ESS `>=1000` and tail ESS `>=400` for every parameter;
- apply finite state/target/log-acceptance, target-status, movement, and declared
  energy-error divergence vetoes to every chunk;
- allow acceptance to nominate a fixed kernel but never to establish
  convergence; and
- write immutable, versioned artifacts with separate warm-up and retained
  archives.

Fixed discarded burn-in and fixed terminal retained counts remain permitted
only for explicitly labeled smoke, mechanics, reference, or debugging code that
cannot support convergence, posterior, robustness, production, or default
claims.

## Route Inventory Boundary

The migration must classify every repository-owned Python route containing
NeuTra plus HMC/sampling behavior into exactly one class:

1. `active_claim_bearing`: must call the shared core controller;
2. `historical_or_superseded`: preserved for artifact reproducibility, must
   state that it is not the active default and must not be launched for new
   claim-bearing work;
3. `smoke_mechanics_or_reference`: may keep bounded fixed counts when its
   nonclaims explicitly forbid convergence/posterior promotion; or
4. `training_or_non_hmc`: out of migration scope.

The executed 2026-07-15 LGSSM gap-closure route is the first
`active_claim_bearing` migration target. The older serious-validation and
target-specific-protocol HMC phases are historical/superseded by the terminal
gap-closure result. Phase 18-20 mechanics/reference helpers remain bounded
engineering evidence, not serious sampling defaults.

## Evidence Contract

Engineering pass requires:

- a shared module under `bayesfilter/inference` with no NumPy import/call and no
  `tf.numpy_function`/`tf.py_function`;
- generic chain count, target adapter, raw-coordinate transform, diagnostic
  callback, and archive callback interfaces;
- default 10,000 warm-up and retained caps and modern R-hat definitions;
- unit tests for warm-up retention/exclusion, recent-window readiness,
  cumulative retained checks, full-convergence extension, health vetoes, seed
  separation, caps, no overwrite through the model wrapper, and public/private
  result separation;
- the claim-bearing LGSSM route imports the shared controller instead of owning
  its algorithm; and
- a checked route inventory with no unclassified serious NeuTra HMC path; and
- a ledger-driven machine-checkable guard that fails if an
  `active_claim_bearing` route does not import/call the shared core controller,
  contains a prohibited fixed burn-in/fixed terminal sampling implementation,
  or lacks the declared warm-up/retained policy binding. Explicit
  `historical_or_superseded` and `smoke_mechanics_or_reference` exceptions must
  be enumerated rather than inferred from filenames; and
- a persistent discovery-completeness check that scans repository-owned Python
  sources using declared NeuTra plus HMC/sampling syntax markers, compares the
  discovered set with the committed ledger, and fails on an unledgered
  qualifying route, a stale ledger path, or multiple classifications. The
  discovery rules and explicit false-positive exclusions are versioned in the
  ledger rather than hidden in test code.

Scientific robustness evidence is arm-specific:

- same-fixture seed arm: fresh 5,000-step GPU/XLA training seed, fresh tuning,
  sequential admission, and fresh confirmation under the existing comparator;
- new-fixture arm: independently generated deterministic observations with a
  new target signature, target-specific geometry/training, a valid tuned
  plain-HMC comparator for that fixture, fresh NeuTra tuning, and fresh
  confirmation against that comparator.

Training/heldout loss is nomination/explanation only. Downstream HMC is the
promotion criterion. A hard-vetoed candidate is rejected without retry; a
localized infrastructure defect may use one fresh attempt under the unchanged
contract.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Shared module separate from `hmc.py` | 2026-07-15 audit | Existing verifier uses NumPy diagnostics and fixed discarded burn-in | Importing it violates active NeuTra policy or keeps wrong semantics | AST/import audit and core Gaussian test | reviewed design choice |
| Warm-up 2,000/1,000/10,000 and `1.05` | Corrected LGSSM campaign and owner direction | Passed real execution and separates readiness from confirmation | Recent window passes by chance | fresh retained full convergence remains independent | reviewed default for NeuTra HMC |
| Retained cap 10,000 | Owner direction and existing verifier cap | Bounded serious sampling | Viable slow mixer rejected at cap | classify candidate failure, not target/direction failure | fixed policy |
| Modern max(rank, folded) R-hat | Repository convergence implementation | Detects location and scale nonconvergence | Folded-only or legacy R-hat silently substitutes | exact definition assertions | fixed policy |
| Third training seed | Missing seed-reliability evidence | Smallest extension of training-seed evidence | Still too few seeds for population reliability | report as one additional arm, no rate estimate | bounded robustness arm |
| New simulation seed | Deterministic driver supports fixture-bound data seeds | Creates genuinely different observations without changing model family | New draw is not materially harder | predeclare difficulty diagnostics and call it new, not necessarily harder, unless observed metrics support harder label | hypothesis |
| Non-truth-centered training geometry on new fixture | Avoid favorable truth-centered initialization | Tests knowledge transfer under less privileged initialization | Offset makes training fail for initialization rather than method | affine-only baseline, finite/status checks, bounded screen | explicit stress mechanism |
| Existing `wide_2x_lr5e3` recipe as new-fixture baseline | Current target winner | Reasonable warm start, not self-justifying | Cross-fixture transfer is poor | target-specific short screen against source/lower-rate comparators | transferred hypothesis only |
| Plain-HMC comparator required per fixture | Scientific evidence discipline | Prevents truth recovery alone from acting as posterior correctness | Comparator itself fails convergence | comparator veto stops the new-fixture lane | fixed gate |

## Skeptical Plan Audit

Audit date: 2026-07-15. The plan was challenged for stale baselines, proxy
promotion, unfair comparisons, hidden defaults, environment mismatch, missing
stop conditions, and commands whose artifacts would not answer the question.

Findings and corrections:

1. Moving the campaign code into legacy `hmc.py` would retain NumPy diagnostics
   and fixed discarded burn-in. A new TensorFlow-only module is required.
2. Repo-wide migration cannot mean rewriting historical evidence or claiming
   every file with “NeuTra” is an active sampler. A route classification ledger
   and active-default test replace indiscriminate rewrites.
3. Initial-state or affine-center perturbations do not create a new posterior
   fixture. The new-fixture arm requires new observations and a new target
   signature.
4. Reusing the old tuned plain-HMC comparator on new observations would be
   wrong. The new-fixture arm cannot train or promote NeuTra until a new tuned
   plain-HMC comparator passes.
5. Reusing `wide_2x_lr5e3` as an unquestioned default on a new target violates
   cross-task transfer discipline. It is only one target-specific screen arm.
6. Calling a new simulation draw “harder” in advance is unsupported. The plan
   records it as a new fixture; the “less favorable” designation is earned only
   if predeclared diagnostics show reduced truth-centered privilege or greater
   posterior/training difficulty.
7. Running broad multi-model robustness now would exceed a bounded campaign.
   This program uses one additional same-fixture seed and one new-fixture ladder;
   further models become a later program based on these results.
8. The first draft allowed one robustness arm to promote the program and relied
   on audits rather than durable default enforcement. The contract now reports
   engineering/S1/F2 separately, requires both scientific arms for a joint
   robustness answer, and requires a ledger-driven failing guard in C2.
9. The first enforcement revision could still be evaded by omitting a new route
   from the ledger. C0-C2 now require a persistent source-discovery versus
   ledger completeness invariant and an omitted-route negative fixture.

Audit verdict: `PASS_AFTER_REVISION`. Engineering consolidation can execute
immediately. Scientific phases execute only after their preceding target,
comparator, and budget entry gates pass.

## Compute And Attempt Budget

- Phases C0-C2 engineering: CPU-hidden tests and smokes, at most 60 minutes.
- Same-fixture third seed: one 5,000-step trusted GPU/XLA training job, at most
  30 minutes; CPU/XLA tuning and confirmation, at most 3 hours.
- New fixture and plain-HMC comparator: CPU-hidden deterministic fixture,
  target, geometry/tuning/sampling ladder, at most 4 hours.
- New-fixture training: target-specific short screen of at most three recipes at
  500 steps each, then one 5,000-step GPU/XLA candidate, aggregate GPU wall at
  most 60 minutes.
- New-fixture NeuTra HMC: tuning and confirmation, at most 4 hours.
- One localized infrastructure retry per job in a fresh versioned directory.
- No package/environment mutation, network fetch, paid compute, destructive
  operation, public act, or silent change to the model/prior/scientific gates.

## Phase C0 - Inventory And Core Contract

Objective: freeze the route inventory, public API, defaults, artifact roles,
and migration boundary.

Entry: terminal exact-fixture result and sequential repair artifacts pass
integrity checks.

Artifacts: route inventory JSON/Markdown; core API contract tests; policy text
in `AGENTS.md` and `CLAUDE.md`; phase result and C1 subplan.

Checks: every discovered NeuTra HMC route classified exactly once; discovery
rules and exclusions are explicit; stale and unledgered paths fail; historical
and smoke nonclaims verified; public/private tensor boundary specified.

Handoff: exact symbols and consumers frozen for C1. Stop for an unresolvable
active-route ambiguity that would change scientific meaning.

## Phase C1 - Shared TensorFlow-Only Controller

Objective: implement the generic sequential controller under
`bayesfilter/inference`.

Artifacts: implementation, lazy inference exports, focused unit tests, Gaussian
CPU/XLA integration smoke, C1 result, and C2 subplan.

Checks: no NumPy/host callback; one compiled program per chunk size; warm-up
retention and posterior exclusion; modern diagnostic semantics; health vetoes;
10,000 caps; deterministic separated chunk seeds; diagnostic/archive callbacks;
no model-specific paths or schemas in core.

Handoff: core tests and smoke pass. Stop for wrong sampling semantics,
nonfinite/diagnostic defects, or XLA incompatibility.

## Phase C2 - Active Route Migration And Default Enforcement

Objective: make the LGSSM claim-bearing route a thin consumer of the shared
controller and prevent new claim-bearing routes from using fixed discarded
burn-in/fixed terminal sampling.

Artifacts: migrated route; compatibility tests; route-policy audit; historical
classification notes; a committed route-classification ledger; a
machine-checkable enforcement and discovery-completeness test bound to that
ledger; C2 result; S1 subplan.

Checks: existing terminal artifact readers remain valid; no historical artifact
is rewritten; active route behavior is equivalent on deterministic test
fixtures; all serious defaults come from core; smoke/reference exceptions stay
explicit. The enforcement test must fail on fixtures that (a) add an active
route without a core call, (b) embed fixed `num_burnin_steps` plus fixed terminal
`num_results` in an active route, (c) omit the canonical policy identifier, or
(d) add a qualifying NeuTra-HMC route that is absent from the ledger. It must
also fail on stale ledger paths and duplicate classifications.

Handoff: engineering consolidation complete. Stop for behavior drift in target,
kernel, seed derivation, diagnostics, or artifact contents.

## Phase S1 - Third Training Seed On Validated Fixture

Objective: test training-seed robustness on the existing exact fixture with one
fresh seed `(20260715, 1203)`.

Artifacts: target-specific 5,000-step GPU training result, fresh fixed-kernel
tuning, sequential admission, confirmation, comparator/recovery result, S1
close record, and F0 subplan.

Checks: same selected recipe identity; no weight reuse; GPU memory growth; XLA;
finite/status/parity; acceptance nomination only; modern warm-up/retained gates;
full convergence, comparator agreement, and recovery.

Handoff: report the new seed independently. A hard-vetoed seed does not stop
the new-fixture engineering lane unless it exposes a common harness/target
defect.

## Phase F0 - New Fixture And Comparator Admission

Objective: generate one genuinely new deterministic T=120 observation fixture,
bind a new exact-target signature, and produce a tuned plain-HMC comparator.

The simulation seed is frozen before execution in the F0 subplan. The model,
prior, parameter order, and truth remain unchanged so the question isolates
data-realization and initialization privilege. A non-truth-centered geometry
stress is defined separately and does not alter the posterior target.

Artifacts: config, fixture, target identity, difficulty diagnostics relative to
the original fixture, XLA value/score gate, tuned plain-HMC run, retained
archive, convergence/recovery result, F0 close record, and F1 subplan.

Checks: target signature differs for the right fixture-bound reason; comparator
passes all modern R-hat/ESS/health/recovery gates; no old comparator or target
signature reused.

Handoff: only an admitted comparator allows F1. Comparator cap/health failure is
a true stop for the new-fixture scientific lane.

## Phase F1 - Target-Specific New-Fixture Training

Objective: screen the inherited recipe against at least source-width and
lower-learning-rate comparators using the new target and non-truth-centered
initial geometry, then train one independently initialized 5,000-step candidate.

Artifacts: screen ledger, selected recipe with uncertainty/nonclaims, GPU run
manifest, frozen payload, parity/health result, F1 result, and F2 subplan.

Checks: target-specific objective/scaling; no screen-weight reuse; GPU memory
growth/XLA; all-finite/status; heldout nomination only; frozen parity.

Handoff: at least one engineering-valid frozen candidate. Zero valid candidates
is a training/candidate failure and stops F2 without rejecting NeuTra generally.

## Phase F2 - New-Fixture NeuTra HMC

Objective: tune and confirm the F1 candidate using the shared controller and
the F0 comparator.

Artifacts: tuning grid/admission, separate warm-up/retained archives,
confirmation, posterior agreement/recovery, result and inference tables.

Checks: unchanged modern thresholds and caps; fresh seeds; no acceptance-based
promotion; all parameter gates; no descriptive method ranking.

Handoff: pass supports viability on two observation fixtures; failure records
the precise candidate/tuning/numerical/statistical class.

The terminal program must not collapse S1 and F2 into a single pass flag. Joint
robustness is `pass` only when both pass; otherwise report the completed arm
outcomes independently and state that joint robustness was not established.

## Phase A - Terminal Drift And Omission Audit

Objective: review the implementation and all executions against this program.

Required audit questions:

- Did any active route bypass core or import NumPy/host callbacks?
- Were warm-up samples archived and excluded everywhere?
- Were recent-window and cumulative diagnostics applied to the intended stages?
- Did any threshold, cap, target, seed, comparator, kernel, recipe, hardware
  class, or artifact root drift?
- Did a proxy become a promotion criterion?
- Were candidate failures separated from target/research-direction failures?
- Were all required manifests, close records, next subplans, and nonclaims
  written?
- Does any positive claim exceed the number of fixtures/seeds actually tested?

Artifacts: terminal drift matrix, missed-item repair record, final result, reset
memo, local checks, and one bounded Claude terminal review when available.

Stop: any unrepairable evidence corruption or unsupported claim. Localized
documentation/reporting omissions are patched and rechecked before close.

## Phase Procedure

At the end of every phase:

1. run required local checks;
2. write a phase result/close record;
3. draft or refresh the next subplan from actual evidence and remaining budget;
4. review the next subplan for consistency, correctness, feasibility, artifact
   coverage, default assumptions, and boundary safety; and
5. continue unless a real scientific, numerical, artifact, hardware, privacy,
   external, or budget blocker fires.

Claude review is advisory. A material scientific/numerical finding must be
repaired; reviewer unavailability or procedural disagreement does not block
trusted local execution when local evidence is adequate.
