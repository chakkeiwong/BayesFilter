# Phase IE8 plan: posterior sensitivity and research-evidence note

## Date

2026-05-16

## Purpose

Integrate IE1--IE7 evidence into a research evidence note.  IE8 may run small
posterior sensitivity checks only on analytic or controlled fixtures whose
prerequisite diagnostics passed in this same harness.  It must preserve the
distinction between exploratory evidence, research implementation evidence, and
bank-facing validation.

## Allowed Write Set

- `experiments/dpf_monograph_evidence/reports/`;
- `experiments/dpf_monograph_evidence/reports/outputs/`;
- implementation-evidence result/closeout files;
- reviewer-grade reset memo continuity section.

## Prerequisites

- IE2 harness ready;
- IE3--IE7 completed or blocked with structured records;
- IE7 must pass before any HMC posterior sensitivity run is attempted;
- every posterior sensitivity run must have a reference class:
  `analytic_reference`, `controlled_simulator_reference`,
  `exact_discrete_baseline`, or `no_trusted_reference_exploratory_only`.

## Tasks

1. Inventory all diagnostic records and blockers.
2. Run zero posterior sensitivity checks by default.  Posterior sensitivity
   execution is allowed only in a future accepted phase that names a trusted
   posterior-capable fixture and comparator.  IE8's current role is aggregate
   governance and evidence closeout.
3. Compare against trusted analytic or controlled references when available.
4. Classify every claim as exploratory, controlled-fixture-supported, blocked,
   or not tested.  `Controlled-fixture-supported` means the claim is supported
   only on the stated analytic or controlled fixture; it is not validation for
   banking, production, model-risk, or real DPF-HMC targets.
5. Produce the final research evidence note and closeout artifact.
6. Update the reviewer-grade reset memo with the completed or blocked program
   state.

## Posterior Sensitivity Execution Gate

IE8 must not run HMC chains, DPF-HMC targets, DSGE targets, MacroFinance
targets, posterior summaries, or posterior sensitivity experiments in this run.

The only currently grounded reference classes are:

| Completed phase | Grounded reference class | Scope |
| --- | --- | --- |
| IE3 | `analytic_reference` | one-step linear-Gaussian Kalman posterior and descriptive bootstrap PF/EDH recovery |
| IE4 | `analytic_reference` | deterministic affine density/log-det/PF-PF algebra parity |
| IE5 | `analytic_reference` | deterministic relaxed-target arithmetic and finite-budget Sinkhorn marginal residuals |
| IE6 | `no_trusted_reference_exploratory_only` | learned-OT residuals deferred because no approved artifact exists |
| IE7 | `no_trusted_reference_exploratory_only` | fixed-scalar value-gradient gate only; no posterior reference |

`controlled_simulator_reference` and `exact_discrete_baseline` are allowed
vocabulary for future work but are not instantiated by the current artifacts.

## Summary JSON Schema

IE8 must write:

- `experiments/dpf_monograph_evidence/reports/outputs/dpf_monograph_evidence_summary.json`.

The summary JSON must include:

- `program_exit_label`;
- `program_status`;
- `source_support_ceiling`;
- `posterior_sensitivity_executed`;
- `posterior_sensitivity_execution_reason`;
- `canonical_diagnostic_rows`;
- `artifact_inventory`;
- `master_program_compliance`;
- `non_implication`;
- `post_run_red_team_note`.

`canonical_diagnostic_rows` must contain one row per canonical diagnostic id.
Each row must include:

- `diagnostic_id`;
- `phase_id`;
- `coverage_status`: one of `passed`, `blocked`, `deferred`, `missing`;
- `phase_status`;
- `source_support_class`;
- `comparator_or_reference_id`;
- `reference_class`;
- `trusted_reference_present`;
- `execution_occurred`;
- `claim_ceiling`: one of `controlled_fixture_supported`,
  `exploratory_only`, `deferred_evidence_gap`, `blocked`, or `not_tested`;
- `why_not_validation`;
- `non_implication`;
- `artifact_paths`;
- `promotion_criterion_status`;
- `promotion_veto_status`;
- `continuation_veto_status`;
- `repair_trigger`.

When `trusted_reference_present=false`, the row's `claim_ceiling` must not
exceed `exploratory_only`, except for explicit `deferred_evidence_gap` or
`blocked` rows.

## Coverage And Narrative Label Mapping

Canonical machine statuses are:

- `passed`;
- `blocked`;
- `deferred`;
- `missing`.

Narrative labels map as follows:

- `controlled-fixture-supported` -> `passed` with
  `claim_ceiling=controlled_fixture_supported`;
- `exploratory` -> `passed` or `missing` with
  `claim_ceiling=exploratory_only`;
- `blocked` -> `blocked`;
- `deferred` -> `deferred`;
- `not tested` -> `missing`.

IE6 must remain visible as `deferred` with repair trigger
`missing artifact provenance`.  IE8 must not collapse deferred into blocked or
not-tested language.

## Source-Support Ceiling

The final evidence note and summary JSON must state:

```text
Program-level source-support ceiling: bibliography-spine support unless a row
explicitly records a stronger reviewed-source artifact. IE1 did not identify
reviewed local DPF source summaries, so this program does not upgrade
source provenance to paper-reviewed support.
```

No final table may imply primary-paper review, banking validation, model-risk
validation, production readiness, real DPF-HMC validation, or posterior
agreement.

## Final Closeout Requirements

The final closeout artifact must include:

- decision table;
- inference-status table;
- canonical diagnostic inventory;
- artifact inventory;
- explicit `what was not concluded` section;
- source-support ceiling;
- IE6 deferred evidence-gap treatment;
- statement that no IE8 posterior sensitivity was executed;
- master exit-label rationale;
- post-run red-team note.

## Run Manifest If A Runner Is Used

If IE8 uses a local runner to produce the summary JSON, it must record CPU-only
and provenance fields in the summary JSON:

- command;
- branch;
- commit;
- dirty-state summary;
- Python version;
- package versions;
- `cpu_only=true`;
- `cuda_visible_devices="-1"`;
- `gpu_devices_visible=[]`;
- `gpu_hidden_before_import=true`;
- `pre_import_cuda_visible_devices="-1"`;
- `pre_import_gpu_hiding_assertion=true`;
- started/ended UTC timestamps;
- artifact paths.

The runner must set `CUDA_VISIBLE_DEVICES=-1` before importing scientific
dependencies.

## Exit Label Selection

- `dpf_monograph_evidence_program_complete`: allowed only if every canonical
  diagnostic is passed with claim ceilings and no deferred/blocked/missing row
  remains.
- `dpf_monograph_evidence_program_complete_with_blockers`: use when the
  aggregate ledger and closeout artifacts are complete but at least one
  canonical diagnostic is deferred, blocked, missing, or exploratory-only.
- `dpf_monograph_evidence_program_blocked`: use if IE8 cannot produce the
  aggregate ledger without hiding, ambiguating, or misclassifying evidence gaps.

Given current IE6 deferred status and no IE8 posterior-sensitivity execution,
the expected exit label is
`dpf_monograph_evidence_program_complete_with_blockers` if aggregation succeeds.

## Primary Criterion

The program ends with a defensible evidence ledger that states what can inform
future BayesFilter implementation and what remains unvalidated.

## Veto Diagnostics

- posterior sensitivity is run despite failed same-scalar HMC checks;
- exploratory checks are described as validation;
- blockers are hidden in aggregate summaries;
- report omits source-support caveats;
- reset memo is not updated.
- a claim above exploratory is made without a trusted reference class;
- real DPF-HMC, DSGE, or MacroFinance targets are sampled without a new
  accepted actual-target-instantiation phase.

## Outcome Classification

- Promotion/pass criterion: IE8 produces a complete evidence ledger and any
  posterior sensitivity row has a trusted reference class or is explicitly
  exploratory-only.
- Promotion veto: source-support, reference class, or blocked-prerequisite
  status is missing from the aggregate table.
- Continuation veto: IE7 failed for any HMC row that IE8 proposes to run.
- Repair trigger: missing artifacts, inconsistent diagnostic labels, or
  overbroad readiness language.
- Explanatory-only diagnostics: posterior sensitivity without trusted reference
  and all controlled-fixture checks that do not instantiate real DPF-HMC
  targets.

## Expected Artifacts

- `experiments/dpf_monograph_evidence/reports/dpf-monograph-research-evidence-note.md`;
- `experiments/dpf_monograph_evidence/reports/outputs/dpf_monograph_evidence_summary.json`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-final-closeout-{YYYY-MM-DD}.md`;
- optional phase result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie8-posterior-sensitivity-governance-result-{YYYY-MM-DD}.md`.

## Exit Labels

- `dpf_monograph_evidence_program_complete`;
- `dpf_monograph_evidence_program_complete_with_blockers`;
- `dpf_monograph_evidence_program_blocked`.
