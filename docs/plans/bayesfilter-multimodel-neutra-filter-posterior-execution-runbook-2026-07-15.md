# Multi-Model NeuTra Filter-Posterior Execution Runbook

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `COMPLETE_WITH_PRECISE_BLOCKERS`

Terminal result:
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-terminal-result-2026-07-16.md`.

Reset memo:
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-reset-memo-2026-07-16.md`.

The first P7 audit corrected the historical P4 `NEUTRA_CONFIRMED` states to
`EVIDENCE_BLOCKED_TUNING_ADMISSION`. The bounded repair then passed fresh
disjoint tuning admission and fresh confirmation for both cells. P7 attempt 02
reverified the resulting terminal matrix. Attempt 01 remains preserved as the
audit that triggered repair; attempt 02 is the active close evidence.

Supervisor/executor: Codex. Claude, when used, is a bounded read-only reviewer
and cannot launch, edit, authorize, or change a scientific boundary.

## Purpose

Execute P0-P7 of
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-master-program-2026-07-15.md`
while preserving exact target identity, cell-local progress, fresh evidence,
bounded repair, and automatic continuation after non-material failures.

This runbook does not pre-authorize package/environment mutation, destructive
operations, public release, external messages, paid compute, private-data
boundary changes, or a material change to a target, method, comparator,
promotion criterion, hardware class, or total campaign budget.

## Control Artifacts

P0 creates these files under a fresh root such as
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/`:

| Artifact | Function |
| --- | --- |
| `cell_ledger.json` | Current cell state and exact evidence paths; append-only transition history |
| `target_registry.json` | Factory-issued target signatures and dependency hashes |
| `assumption_ledger.json` | Per-cell provenance, justification, risk, early diagnostic, and status |
| `command_manifest.json` | Exact commands, environments, device intent, timeouts, and output roots |
| `budget_ledger.json` | Attempt and wall-time ceilings, consumption, and remainder |
| `execution_events.jsonl` | Chronological launch, result, repair, review, and handoff events |
| `artifact_hashes.json` | SHA-256 inventory of claim-bearing outputs |

Never edit an earlier attempt to make a later attempt appear successful. New
attempts use `phase-<p>/<cell-id>/attempt-<nn>-<timestamp>/`. Result notes link
all superseded attempts and say why they were superseded.

## Cell State Transitions

| From | Required gate | To on pass | To on failure |
| --- | --- | --- | --- |
| `UNINVENTORIED` | P0 target/prior/data/route/default inventory | `TARGET_FROZEN` | `TARGET_BLOCKED` |
| `TARGET_FROZEN` | Batched TF value/status and value/score, XLA, focused reference/FD, route vetoes | `VALUE_SCORE_ADMITTED` | `IMPLEMENTATION_BLOCKED` or `FILTER_CANDIDATE_REJECTED` |
| `VALUE_SCORE_ADMITTED` | Independent posterior recomposition dossier, total unconstrained score check, and substitution-negative tests | `POSTERIOR_IDENTITY_ADMITTED` | `TARGET_BLOCKED` or `IMPLEMENTATION_BLOCKED` |
| `POSTERIOR_IDENTITY_ADMITTED` | Same-signature tuned plain-HMC result with valid diagnostics | `COMPARATOR_ADMITTED` | `COMPARATOR_BLOCKED` |
| `COMPARATOR_ADMITTED` | Target-specific recipe/capacity/optimizer screen | `TRAINING_SCREENED` | `RECIPE_REJECTED` or repair |
| `TRAINING_SCREENED` | Fresh selected 5,000-step GPU/XLA run and frozen artifact checks | `TRAINING_ADMITTED` | `RECIPE_REJECTED` or repair |
| `TRAINING_ADMITTED` | Fresh NeuTra tuning, retained warm-up, final samples, diagnostics, agreement | `NEUTRA_CONFIRMED` | `RECIPE_REJECTED`, `SAMPLER_BLOCKED`, or repair |

No state may be inferred from a filename or prose claim. The ledger transition
must bind the target signature and the exact evidence artifact hashes.

## Standard Cell Pipeline

### R0: Target Replay

1. Load the P0 registry row by cell ID.
2. Recompute the target signature from repository-owned fields.
3. Verify prior, observations, parameter chart, filter settings, dtype, and
   dependency closure.
4. Reject caller-stamped identity and all cross-cell transport/comparator reuse.

### R1: Value And Score Admission

1. Run tiny CPU-hidden reference/debug checks with
   `CUDA_VISIBLE_DEVICES=-1` declared before framework import.
2. Run batch-shape, permutation, deterministic-replay, finite/status, and
   value/score consistency checks.
3. Run the target-specific exact/dense/focused filter comparator and same-branch
   score check. Classify reference gaps by their predeclared role.
4. Run a trusted GPU/XLA compile canary with TensorFlow memory growth.
5. Scan the active target/training path for NumPy, `py_function`, host callbacks,
   eager conversions inside compiled work, and Python sample-axis loops.

R1 admits engineering and filter-route validity only. It does not establish a
posterior sampler result.

### R1B: Posterior Identity Admission

1. Write the constrained posterior as prior plus filter log likelihood and name
   every fixed datum/setting.
2. Write the unconstraining map and complete log-absolute-Jacobian contribution.
3. Independently recompose the total unconstrained log density from declared
   pieces without calling the production final target-assembly callable.
4. Check production versus independent total values and total scores over the
   P0-frozen region, including terms whose individual scores cancel.
5. Run negative substitutions for data, prior, filter settings, dtype, chart,
   Jacobian, and signature; every substitution must fail or change identity as
   declared.
6. Preserve the dossier and checks under the cell signature.

Only R1B moves a cell to `POSTERIOR_IDENTITY_ADMITTED`. Plain HMC cannot begin
from `VALUE_SCORE_ADMITTED` alone.

### R2: Same-Target Plain HMC

1. Freeze the tuning grid before observing the confirmation output.
2. Use at least four chains and the exact registered target adapter.
3. Tune a fixed kernel using only tuning draws; record all candidate steps,
   health vetoes, selection rule, and no-selection branch.
4. Retain and separately archive warm-up. Grow warm-up until the recent window
   has modern R-hat `<=1.05` or 10,000 draws per chain are reached.
5. With the fixed selected kernel and a distinct seed, grow retained draws until
   modern R-hat `<=1.01`, bulk ESS `>=1000`, and tail ESS `>=400`, or 10,000
   draws per chain are reached.
6. Record divergences/energy errors, acceptance, target status, and exact
   archives. Acceptance is explanatory, not the tuning objective.

### R3: Target-Specific Training Protocol

1. Audit target scaling, affine chart, architecture/capacity, optimizer, learning
   rate, batch size, schedule, seeds, heldout construction, and compute ladder.
2. Generate external replay/evaluation data with deterministic multicore CPU
   workers when applicable. Record worker count and domain-separated seeds.
3. Run a bounded GPU/XLA recipe screen. Training is batched, requires memory
   growth, and does not use NumPy or Python sample-axis loops.
4. Select only under the frozen rule. Loss and heldout residuals nominate or
   veto; they do not confirm the downstream sampler.
5. Launch one fresh 5,000-step selected training with a seed excluded from the
   recipe screen. Freeze and hash the transport artifact.

If no recipe passes its minimum validity screen, record `RECIPE_REJECTED` for
the attempted family. The cell remains scientifically open while a predeclared
target-specific family or optimizer ladder remains untried within budget. An
enhanced architecture requires a refreshed subplan with its family, optimizer
ladder, selection rule, and budget frozen before outputs, not post hoc grid
expansion. `CELL_CANDIDATE_REJECTED` is legal only when all such predeclared
arms have executed and failed. A continuation veto, budget exhaustion, or
untried arm yields a blocked state and preserves the remaining scientific
question; it is not candidate rejection.

### R4: NeuTra Confirmation

1. Verify frozen transport and target signatures match.
2. Tune a fresh fixed kernel in transported coordinates under the same tuning
   discipline as R2.
3. Use distinct warm-up and retained seeds, at least four batched chains, and
   the shared sequential HMC controller.
4. Retain separate warm-up and retained archives; enforce the modern R-hat,
   ESS, cap, health, and target-status gates from R2.
5. Apply the P0-frozen simultaneous uncertainty/equivalence rule to required
   posterior estimands against the R2 comparator.
6. Record filter-approximation evidence separately from sampler agreement.

Only R4 may move a cell to `NEUTRA_CONFIRMED`.

## Phase Procedure

For P0 through P7, the supervisor performs this exact sequence:

1. **Entry replay:** verify the prior phase result, cell ledger, artifact hashes,
   remaining budget, environment, and subplan entry conditions.
2. **Skeptical audit:** challenge baseline, proxy roles, defaults, stop rules,
   fairness, target identity, stale evidence, environment, and artifact ability
   to answer the question. Patch the subplan before execution if material.
3. **Launch:** allocate a fresh attempt root and append the command/environment/
   target/budget event before running the command.
4. **Check:** run the subplan's focused and phase-wide checks; preserve stdout,
   stderr, exit status, wall time, and partial structured outputs.
5. **Classify:** use the failure taxonomy below. Do not treat a failed candidate
   as a continuation veto.
6. **Repair or decide:** follow the repair algorithm. Independent cells continue
   after cell-local rejection/blockage.
7. **Close:** write the phase result, decision table, inference-status table,
   post-run red-team note, manifest, updated hashes, cell ledger, and budget.
8. **Handoff:** refresh the next subplan from actual evidence and audit its
   consistency, feasibility, artifact coverage, and boundary safety.
9. **Continue:** start the next phase unless a true continuation veto below
   fired. A reviewer timeout or documentation-only objection is recorded and
   does not stop execution.

## Failure Taxonomy And Repair Algorithm

| Class | Examples | Default action |
| --- | --- | --- |
| `HARNESS_INFRASTRUCTURE` | XLA compile, device visibility, multiprocessing, I/O, serialization | Repair locally, focused regression, retry fresh root |
| `TARGET_IDENTITY` | Prior/data/transform/filter/signature mismatch | Quarantine cell; repair registry/adapter; replay from R0 |
| `POSTERIOR_RECOMPOSITION` | Missing/wrong prior, Jacobian, observation, filter setting, total value, or total score | Repair target assembly/dossier; replay R1B before HMC |
| `VALUE_SCORE_IMPLEMENTATION` | Nonfinite, wrong shape, score/FD mismatch, host callback | Repair implementation; replay R1 |
| `FILTER_APPROXIMATION` | Admitted implementation is too inaccurate under frozen filter gate | Record `FILTER_CANDIDATE_REJECTED` or execute predeclared filter repair; never tune NeuTra around it |
| `TUNING` | No fixed kernel passes health/selection rule | Run predeclared tuning repair; do not select by acceptance alone |
| `TRAINING_RECIPE` | Nonfinite loss, invalid artifact, no recipe passes | Repair mechanics or record `RECIPE_REJECTED`; do not reject untried families |
| `SAMPLER` | R-hat/ESS/comparator/health failure | Localize target versus transport versus kernel; retry only a predeclared repair |
| `EVIDENCE_REPORTING` | Missing manifest, archive, seed, hash, or role classification | Repair reporting from preserved raw evidence; rerun only if evidence cannot be reconstructed honestly |
| `CELL_CANDIDATE_REJECTION` | Valid target/harness and every P0-frozen transport family executed and failed its frozen gate | Record tried/selected/rejected ledger, state the bounded candidate scope, and continue independent cells |

Repair algorithm:

1. Preserve the failed attempt and append its classification.
2. Ask whether target, data, method, comparator, criteria, vetoes, hardware,
   privacy boundary, and total phase budget remain unchanged.
3. If yes, write a short repair record: root cause hypothesis, smallest patch,
   focused regression, invalidated rung, attempt/time cost, and rollback rule.
4. Apply the patch in the narrowest scope; run the focused regression.
5. On pass, allocate a new output root and resume from the earliest invalid rung.
6. On the third materially identical failed repair, mark the affected recipe or
   cell blocked and continue other work. Do not convert this ceiling into cell
   rejection while a predeclared candidate family remains untried within budget.
7. If any contract element changes, stop that cell and request direction before
   expanding scope. Do not smuggle a scientific redesign into an infrastructure
   retry.

## Candidate-Family Budget Rule

P0 reserves two arms per cell: plain dense IAF and one target-specific enhanced
family. Each arm has one bounded recipe screen, one selected fresh 5,000-step
training, one NeuTra confirmation, arm-local retries, and a 15-GPU-hour bucket.
A separate 6-GPU-hour bucket funds plain-HMC tuning/confirmation and comparator
repairs once per cell; a separate 4-GPU-hour cell-admission/infrastructure bucket
funds trusted R0/R1/R1B value-score, recomposition, identity, batch, XLA, and
device canaries plus cell-specific adapter serialization/artifact emission and
their repairs. Common harness, schema, or shared serialization/reporting defects
reopen P1, consume only its shared budget, pause downstream cells without state
changes, and never consume a cell bucket. The exact ceiling is 40 GPU-hours per
cell. Skip the enhanced arm after baseline
confirmation; otherwise execute it after baseline `RECIPE_REJECTED` or
downstream rejection. Charge every attempt to one bucket before launch.
Admission-bucket exhaustion yields `TARGET_BLOCKED` or
`IMPLEMENTATION_BLOCKED`; comparator-bucket exhaustion yields
`COMPARATOR_BLOCKED`; an unanswered scientific gate under an exhausted family
arm is budget-blocked. An exhausted mandatory bucket or unexecuted arm never
permits `CELL_CANDIDATE_REJECTED`.

P1 owns a 2-GPU-hour common canary/repair bucket. If a common defect appears
after P1 close, reopen P1 under a fresh attempt, preserve cell states, and charge
only that bucket. Exhaustion before common repair is a program continuation
veto, not `TARGET_BLOCKED` or `IMPLEMENTATION_BLOCKED` for the active cell.

## Review Procedure

Use one bounded material review when it can change a scientific/engineering
decision, especially P0 route/source classification, P5 structural mathematics,
P6 SIR target admission, or P7 claims. The first prompt names one exact path and
one question. If Claude gives no output, use a tiny probe; if alive, narrow or
repair the prompt. After bounded unavailability, record it and continue with a
local skeptical audit. Review never substitutes for tests or source anchors.

## True Continuation Vetoes

Stop the affected scope only for:

- invalid or internally inconsistent target, data, prior, transform, or
  comparator that cannot be repaired without redefining the question;
- shared harness invalidity that contaminates completed evidence;
- corrupted/missing raw evidence required for a claim and not reproducible
  within budget;
- no same-target comparator and no scoped path to construct one;
- unavailable required trusted GPU after an escalated device and framework
  probe;
- exhausted phase or program budget;
- destructive, external, privacy, package/environment, funding, hardware-class,
  or material scientific change needing user authority.

A filter cell's nonconvergence, approximation failure, training failure, or
candidate rejection is not a program continuation veto. Record it and continue.

## Phase Result Template

Every result includes:

```text
Phase and attempt IDs
Git commit and dirty-worktree disclosure
Plan/result paths
Exact commands and environment/conda env
CPU/GPU, XLA, TF32, memory-growth provenance
Target signatures, data version/hash, seeds
Wall time and output paths
Checks and exit statuses
Per-cell state transitions
Failure classifications and repair history
Budget consumed and remaining
Decision table
Inference-status table
Post-run red-team note
Forbidden conclusions
Next subplan path and handoff decision
```

The decision table contains decision, primary criterion, veto status, main
uncertainty, next justified action, and nonclaims. The inference-status table
contains hard veto screen, statistically supported ranking, descriptive-only
differences, default readiness, and next evidence needed.

## Launch Boundary

This document launches planning only. Execute P0 after its skeptical audit.
Serious P2-P6 runs remain blocked until P0 freezes exact commands, equivalence
rules, output roots, and budgets and P1 admits the shared harness. The user's
plain-language instruction to continue this program is sufficient local campaign
authorization once those scientific gates pass; no hash-bound approval phrase
or one-use token is required.
