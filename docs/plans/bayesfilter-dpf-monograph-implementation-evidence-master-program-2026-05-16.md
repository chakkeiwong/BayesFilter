# Master program: DPF monograph implementation-and-evidence

## Date

2026-05-16

## Status

Proposed non-production implementation-and-evidence program for the
reviewer-grade DPF monograph lane.

This program follows the reviewer-grade writing closeout.  It does not reopen
the P0--P13 exposition round.  Its purpose is to execute the Chapter 26
diagnostic contract and create evidence before any empirical, banking-facing,
or production-facing claims are strengthened.

## Governing Motivation

The reviewer-grade DPF monograph round closed with:

- reviewer-grade exposition complete;
- empirical DPF-HMC validation not complete;
- banking or production readiness not claimed.

The remaining gap is not prose quality.  The gap is executable evidence:

- no code-level diagnostic harness has executed the Chapter 26 checks;
- no posterior-comparison package exists for PF-PF, EOT/Sinkhorn, learned OT,
  or DPF-HMC;
- DPF-HMC for nonlinear DSGE or MacroFinance targets remains unvalidated;
- source provenance remains bibliography-spine support unless a future
  ResearchAssistant source-review artifact upgrades it.

This program therefore builds a bounded evidence spine that can later inform
BayesFilter-owned implementation work and monograph claim updates.  Student
DPF baselines remain out of lane.

## Lane Boundary

Owned planning surfaces:

- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-*`;
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-reset-memo-2026-05-15.md`,
  only for continuity updates.

Proposed non-production evidence surface:

- `experiments/dpf_monograph_evidence/`.

Read-only source surfaces:

- reviewer-grade DPF chapters under `docs/chapters/ch19*.tex` and
  `docs/chapters/ch32_diff_resampling_neural_ot.tex`;
- reviewer-grade P0--P13 plans, results, audits, and final readiness report;
- `docs/references.bib`;
- ResearchAssistant local summaries, if present and read-only.

Forbidden without a later explicit transition plan:

- production `bayesfilter/` edits;
- production API changes;
- production `bayesfilter/` imports from the evidence harness;
- student-baseline files and vendored student snapshots;
- controlled student-baseline experiments;
- live API, network fetch, GPU-required runs, long HMC chains, broad dependency
  installs, or notebook conversion;
- claims of banking, model-risk, production, or empirical validation.

No commit or push is authorized by this plan.

## Clean-Room Import Policy

The default execution contract is clean-room evidence only.  IE2--IE8 may use
standard scientific Python dependencies that already exist in the environment,
but they must not import production `bayesfilter/` modules or student-baseline
modules.  This means the program validates equation-level diagnostics,
controlled fixtures, and harness semantics; it does not validate the production
BayesFilter implementation.

If a later worker concludes that production imports are required, that worker
must stop and write a separate transition plan with a narrow read-only import
whitelist, mutation guard, and acceptance criteria.  The current program does
not authorize that transition.

## Evidence Standard

Each phase must produce evidence that is:

1. deterministic or seed-controlled;
2. small enough for normal repository history;
3. tied to a chapter equation, claim, or diagnostic table;
4. explicit about what it proves and what it does not prove;
5. classified with a structured exit label;
6. recorded in Markdown plus machine-readable JSON when execution occurs.

The default interpretation is conservative: failure produces a blocker map, not
a tuning campaign.

## Required Research-Run Governance Artifacts

Every executed phase must include the following in its result artifact or a
linked machine-readable record:

1. skeptical plan-audit status;
2. research-intent ledger stating the diagnostic question, intended use, and
   prohibited use;
3. evidence contract with comparator or baseline id, tolerance, source-support
   class, and promotion rule;
4. pre-mortem listing expected failure modes and how they will be classified;
5. run manifest with command, commit, branch, environment, CPU/GPU policy,
   random seed policy, wall-clock cap, and artifact paths;
6. result decision table with pass criterion, promotion veto, continuation
   veto, repair trigger, and explanatory-only diagnostics;
7. stochastic inference-status table when stochastic outputs are reported,
   including replication count, uncertainty status, and whether the evidence is
   descriptive only;
8. post-run red-team note describing the strongest reason not to over-read the
   result.

IE0 is the plan-audit gate.  IE2 must encode the shared evidence contract and
run-manifest schema before IE3--IE8 execute.

## Phase Taxonomy

| Phase | Subplan | Role |
| --- | --- | --- |
| IE0 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie0-preflight-plan-2026-05-16.md` | scope, worktree, evidence inventory, plan audit |
| IE1 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie1-source-review-intake-plan-2026-05-16.md` | optional source-review upgrade from bibliography-spine support |
| IE2 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie2-diagnostic-harness-design-plan-2026-05-16.md` | Chapter 26 diagnostic harness design |
| IE3 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie3-linear-gaussian-recovery-plan-2026-05-16.md` | PF and EDH linear-Gaussian recovery tests |
| IE4 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie4-affine-flow-pfpf-plan-2026-05-16.md` | PF-PF density, log-det, and weight checks |
| IE5 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie5-resampling-sinkhorn-plan-2026-05-16.md` | soft-resampling and Sinkhorn controlled tests |
| IE6 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie6-learned-ot-plan-2026-05-16.md` | learned-OT teacher/student/OOD residual tests |
| IE7 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie7-hmc-value-gradient-plan-2026-05-16.md` | same-scalar HMC value-gradient and repeatability tests |
| IE8 | `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie8-posterior-sensitivity-governance-plan-2026-05-16.md` | posterior sensitivity and research evidence note |

## Execution Order

Run phases in order.  IE1 may exit as `ie1_source_review_deferred` if source
intake is not authorized; that exit does not block IE2, but it preserves the
current bibliography-spine caveat.

IE2 is mandatory before all executable diagnostic phases.  IE3--IE7 may not
invent independent schemas; they must use the harness contract from IE2 or stop
with a schema blocker.  IE8 is the only phase that may summarize cross-phase
readiness.

Posterior sensitivity in IE8 is restricted to analytic or controlled fixtures
validated in this same harness.  It must not run real DPF-HMC, DSGE, or
MacroFinance target chains unless a new actual-target-instantiation phase is
created and accepted first.

## Global Cycle

Each phase must follow:

```text
preflight -> phase-local plan -> execute -> test -> audit -> tidy -> update reset memo
```

Continue automatically only when:

- the phase primary criterion passes;
- no veto diagnostic fires;
- generated artifacts stay within size limits;
- the next phase remains justified by recorded evidence.

## Common Result Labels

Allowed final labels:

- `ie_phase_passed`;
- `ie_phase_passed_with_caveats`;
- `ie_phase_deferred_with_recorded_reason`;
- `ie_phase_blocked`;
- `ie_phase_rejected_for_lane_drift`.

IE8 must end with one of:

- `dpf_monograph_evidence_program_complete`;
- `dpf_monograph_evidence_program_complete_with_blockers`;
- `dpf_monograph_evidence_program_blocked`.

Every phase must also classify each diagnostic row using:

- `promotion_pass`;
- `promotion_veto`;
- `continuation_veto`;
- `repair_trigger`;
- `explanatory_only`.

## Global Veto Diagnostics

A phase must stop if it:

1. edits production `bayesfilter/` without explicit transition authorization;
2. uses student-baseline code or results as monograph validation evidence;
3. describes bibliography-spine support as ResearchAssistant-reviewed support;
4. runs GPU, network, API, broad dependency install, notebook conversion, or
   unbounded experiments;
5. tunes an HMC or DPF method before same-scalar value/gradient checks pass;
6. reports only qualitative success where the diagnostic requires a scalar,
   residual, tolerance, or finite/shape check;
7. omits seed/randomness policy for stochastic diagnostics;
8. writes artifacts too large for normal repository history;
9. strengthens banking, model-risk, production, or empirical validation claims;
10. skips reset-memo continuity after a completed or blocked phase.
11. omits CPU-only declaration or GPU visibility policy from the run manifest;
12. omits source-support class from a Markdown result summary or IE8 aggregate
    table.

## Shared Artifact Contract

The planned implementation root is:

```text
experiments/dpf_monograph_evidence/
```

Execution phases should use these conventional subdirectories unless their
result artifact records a reason to deviate:

- `fixtures/`
- `diagnostics/`
- `runners/`
- `reports/`
- `reports/outputs/`

Canonical final outputs after IE8:

- `experiments/dpf_monograph_evidence/reports/dpf-monograph-research-evidence-note.md`;
- `experiments/dpf_monograph_evidence/reports/outputs/dpf_monograph_evidence_summary.json`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-final-closeout-{YYYY-MM-DD}.md`.

## Validation Contract

Before IE8 can close the program, the following must be true or blocked with a
structured reason:

- every IE2 diagnostic schema is represented in a JSON result or blocker
  record;
- all successful numerical diagnostics report seed policy, tolerance,
  finite-value status, shape status, and runtime;
- all results report comparator or baseline id, diagnostic role classification,
  promotion status, continuation status, repair trigger, source-support class,
  environment, commit, command, wall time, artifact paths, CPU/GPU policy, and
  uncertainty status;
- same-scalar HMC checks pass before any HMC tuning or posterior summary is
  interpreted;
- every posterior sensitivity claim is labeled as exploratory evidence unless a
  trusted reference is present;
- no production or student-lane imports are introduced into the evidence root;
- `git diff --check` passes over the implementation-evidence docs and evidence
  root.

## Relationship To Monograph Claims

Passing this program would not automatically change the monograph.  It would
authorize a later claim-update plan that can cite bounded evidence.  Failing or
blocked phases should become explicit limitations and implementation tasks.
