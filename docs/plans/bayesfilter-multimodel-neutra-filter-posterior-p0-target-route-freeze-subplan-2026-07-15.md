# P0 Subplan: Target And Route Identity Freeze

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `READY_AFTER_ENTRY_AUDIT`

## Phase Objective

Turn the eleven-cell scope into a machine-checkable registry. For every cell,
freeze the estimand, observations, parameter chart, prior, likelihood semantics,
filter route/settings, source classification, exact same-target comparator,
reference evidence, target signature fields, assumptions, missing work, commands,
and bounded compute. Do not implement or run serious training/HMC in P0.

## Entry Conditions

- The master program and execution runbook exist and agree on the cell matrix.
- The LGSSM reset memo is treated as procedure/harness provenance only.
- Existing dirty work in the repository is inventoried and preserved.
- The current `AGENTS.md` academic governance profile supersedes historical
  launch-token ceremony.

## Cell Scope

All cells: `SVX-SGQF`, `SVX-ZC`, `KSC-UKF`, `PP-SGQF`, `PP-UKF`, `PP-ZC`,
`STR-UKF`, `STR-ZC`, `SIR-SGQF`, `SIR-UKF`, and `SIR-ZC`.

Exact transformed SV and KSC SV must have disjoint target family identifiers.
The structural Zhao-Cui row is `extension_or_invention`. SIR local complete-data
density and UKF scout evidence are not full observed-data posterior admission.

## Required Artifacts

- `p0-target-route-ledger.md` with one row per cell and explicit blocker.
- `target_registry.json` with repository-owned signature inputs and dependency
  hashes; no caller may self-attest a signature.
- `assumption_ledger.json` covering every material prior, chart, observation,
  filter setting, numerical threshold, architecture/optimizer grid, seed policy,
  and equivalence rule.
- `source_anchor_ledger.md` with paper section/equation and author-source
  file/line anchors for every Zhao-Cui reproduction/adaptation.
- A posterior-identity dossier specification requiring the exact constrained
  target decomposition, unconstraining map and full log-absolute-Jacobian,
  bound prior/data/filter settings, independent total-value/total-score
  recomposition, and wrong-substitution negative tests.
- A candidate-family ledger specification with tried, selected, rejected, and
  untried topology/optimizer arms and the legal basis for any cell-level
  rejection.
- `command_manifest.json` with exact P1-P7 commands, environment, device intent,
  timeout, output root, and artifact expectations. Commands unavailable because
  implementation is missing are marked `TO_BE_CREATED_IN_PHASE`, not invented.
- `budget_ledger.json`, `cell_ledger.json`, `artifact_hashes.json`, P0 run
  manifest, and P0 result.
- A refreshed P1 subplan based on the actual adapter inventory.

## Required Checks And Reviews

1. Inspect the exact value/score routes and focused tests, including:
   - exact transformed SV SGQF and Zhao-Cui in
     `bayesfilter/highdim/sv_mixture_cut4.py`;
   - KSC UKF value at line 2062 and score at line 2151 of that module;
   - predator-prey model and SGQF/UKF tests;
   - parameterized SIR model and local complete-data routes;
   - Chapter 18b equations, structural fixture, and identity test.
2. For Zhao-Cui cells, inspect the technical paper/math and local author source,
   then classify each proposed operation as `source_faithful`,
   `fixed_hmc_adaptation`, or `extension_or_invention` with anchors.
3. Trace priors, observations, parameter transformations, filter settings, and
   source dependency closures. Record unknown or conflicting fields as blockers.
4. Verify each comparator is the identical filter-defined posterior, not an
   exact-model, alternate-filter, KSC/exact-SV, or complete-data substitute.
5. Freeze each constrained target decomposition, unconstraining map, complete
   Jacobian term, and an independent recomposition/total-score check that does
   not call the production final target-assembly callable.
6. Freeze the two target-specific candidate-family arms and their optimizer
   ladders: plain dense IAF and one enhanced capacity/topology or other supported
   flow family. Reserve 15 GPU-hours for each arm, 6 GPU-hours for same-target
   plain HMC, and 4 GPU-hours for trusted R0/R1/R1B cell admission,
   cell-specific adapter serialization/artifact emission, and their repairs.
   Common harness/schema/serialization repairs remain in P1's 2-GPU-hour shared
   bucket. A first-arm failure cannot become cell rejection; an unexecuted second
   arm or exhausted mandatory bucket leaves the cell blocked/open.
7. Freeze target-specific comparator estimands, simultaneous uncertainty method,
   equivalence regions, applicability checks, and failure branches before
   serious outputs. If scientific margins cannot be justified, mark the cell
   blocked rather than inventing thresholds.
8. Run JSON/schema validation, duplicate-cell/signature negative tests, explicit
   file existence checks, Markdown link/path checks, and scoped
   `git diff --check`.
9. Perform one material bounded review of the target ledger because target
   identity/source classification affects every downstream claim. Reviewer
   procedural comments are advisory; source or mathematical contradictions are
   material.

## Evidence Contract

| Field | P0 contract |
| --- | --- |
| Question | Is every planned filter-posterior target defined precisely enough that two implementations can be shown to bind the same posterior? |
| Comparator | Current code/docs/tests plus checked paper and author source for Zhao-Cui routes |
| Primary pass | Eleven unique registry rows have complete required fields, honest blockers, disjoint signatures, posterior-recomposition and candidate-family rules, justified gates, and command/budget ownership |
| Vetoes | Target conflation; missing prior/chart/data; caller-stamped identity; unanchored Zhao-Cui faithfulness; complete-data/scout route mislabeled posterior; post-result margin selection |
| Explanatory only | Existing smoke outcomes, approximate-filter gaps, runtime estimates, and legacy readiness labels |
| Not concluded | Implementation admission, filter adequacy, HMC validity, NeuTra quality, or any cell confirmation |

## Default And Assumption Audit

For every material choice record provenance, target-specific justification,
failure mode, earliest diagnostic, and status among owner-selected, reviewed
default, baseline, warm-start hypothesis, convenience choice, or unknown. The
following require explicit entries: data-generating seed/data version, prior
scales, unconstraining chart, filter quadrature/rank/basis settings, fixed
randomness, dtype, affine preconditioner, training topology/grid, learning-rate
grid, batch size, heldout policy, HMC tuning grid, convergence thresholds,
comparator estimands, equivalence margins, and phase budgets.

Cross-model provenance alone is insufficient justification. Unknowns either get
an early diagnostic in the owning phase or block serious execution.

## Repair Triggers

- Missing or conflicting route identity: trace dependencies and patch only the
  ledger/registry or owning future subplan.
- Comparator mismatch: split the cell or bind a true same-target comparator.
- Missing Zhao-Cui anchors: inspect the source; if absent, downgrade to
  `extension_or_invention` or block.
- Unjustified margin/default: derive/review it before output or retain a blocker.
- Schema/report failure: repair in a fresh P0 attempt and rerun focused checks.

## Forbidden Claims And Actions

- No training, long HMC, filter promotion, or target readiness claim.
- No mixing exact transformed SV with KSC evidence.
- No claiming Zhao-Cui source faithfulness from local naming or tests alone.
- No treating SIR local density/UKF scout work as an observed-data posterior.
- No post-result thresholds, magic approval tokens, package changes, or edits to
  unrelated dirty files.

## Handoff Conditions

P1 begins when the registry is schema-valid, all eleven rows are present, every
Zhao-Cui row is anchored or honestly classified/blocked, posterior-identity and
candidate-family dossier schemas are frozen, P1 adapter requirements are
explicit, command and budget ownership is recorded, the P0 result is written,
and the refreshed P1 plan passes its skeptical suitability audit.
Cells may carry explicit target blockers into P1 because P1 is a generic harness
phase; blocked cells cannot proceed to their model phase beyond inventory.

## Stop Conditions

Stop P0 for an irreconcilable target definition, missing required paper/source,
an unresolved comparator ambiguity that changes the estimand, inability to
preserve user work, or exhaustion of the 8 CPU-hour/three-attempt ceiling.
Write a blocker result naming affected cells; independent well-defined cells
remain eligible for P1.

## Compute And Attempt Budget

At most 8 CPU wall-hours, no serious GPU work, and three repair attempts for one
materially identical defect. A tiny trusted GPU import/compile probe is allowed
only if needed to make P1 commands honest and is capped at 10 minutes.

## Skeptical Pre-Execution Audit

The baseline is source/code/target identity rather than existing readiness
labels; proxies cannot promote; target definitions precede numerical results;
and missing routes are recorded rather than hidden. P0 is fit to execute after
rechecking the current dirty-worktree overlap and source availability.
