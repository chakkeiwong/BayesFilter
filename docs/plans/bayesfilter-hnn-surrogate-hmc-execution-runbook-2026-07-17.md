# Corrected Neural-Force HMC Execution Runbook

Date: 2026-07-17

Program ID: `bayesfilter-corrected-neural-force-hmc-20260717`

Status: `REVIEWED_READY_FOR_P0`

Master program:
`docs/plans/bayesfilter-hnn-surrogate-hmc-master-program-2026-07-17.md`.

## Control Artifacts

Use the fresh root
`docs/plans/artifacts/corrected-neural-force-hmc-20260717/` with:

- `program_ledger.json`: phase/cell states and append-only transitions;
- `target_registry.json`: target, chart, data, filter, and dependency identities;
- `force_registry.json`: training, normalization, source, and frozen force hashes;
- `assumption_ledger.json`: default provenance and status;
- `budget_ledger.json`: phase/attempt wall-time consumption;
- `execution_events.jsonl`: launches, failures, repairs, reviews, and closes;
- `artifact_hashes.json`: claim-bearing artifact inventory.

Every launch uses `phase-pN/<cell>/attempt-NN-<timestamp>/`. Never overwrite or
edit a failed attempt to appear successful.

## Standard Phase Procedure

1. Replay the previous phase result, hashes, remaining budget, environment, and
   subplan entry conditions.
2. Perform the skeptical audit required by `AGENTS.md`; patch the subplan before
   execution if a material issue is found.
3. Allocate a fresh attempt root and record the exact command, git commit,
   dirty-worktree disclosure, conda environment, device intent, seeds, target,
   force identity, and budget charge.
4. Run the smallest decisive check before a larger run.
5. Run the subplan checks and preserve stdout, stderr, status, wall time, raw
   tensors, traces, and partial outputs.
6. Classify every failure and decide repair, cell-local block, or continuation.
7. Write the phase result/close record and update ledgers/hashes.
8. Draft or refresh the next subplan and review its consistency, correctness,
   feasibility, defaults, artifact coverage, cost, and target boundary.
9. Continue unless a true continuation veto fired.

## Failure And Repair Table

| Class | Examples | Re-entry |
| --- | --- | --- |
| `HARNESS_INFRASTRUCTURE` | GPU visibility, XLA compile, memory, serialization, archive I/O | repair focused check, retry same rung |
| `TARGET_IDENTITY` | target/chart/data/filter/signature mismatch | quarantine cell, replay P0 target binding |
| `MAP_MECHANICS` | reversal/Jacobian failure, momentum input, asymmetric update, wrong kinetic term | repair P1; invalidate all downstream HNN results |
| `TRAINING_MECHANICS` | nonfinite loss/force, bad normalization, invalid frozen payload | repair P2 or reject recipe |
| `TUNING` | no fixed candidate passes energy/health/admission | execute frozen tuning repair, not acceptance-only selection |
| `SAMPLER` | divergence, status, modern R-hat, ESS, or cap failure | localize force versus kernel versus target; repair affected cell |
| `TRUTH_TAIL` | marginal or severe truth location | one conditional seed for marginal; investigate severe failure |
| `PERFORMANCE` | valid chain but no amortized advantage | record `PERFORMANCE_NOT_DEMONSTRATED`; do not call method invalid |
| `EVIDENCE_REPORTING` | missing hash, manifest, telemetry, archive | reconstruct from raw evidence or rerun if impossible |

Repair sequence: preserve, classify, write repair record, patch narrowly, run
focused regression, retry in a fresh root. Stop the affected scope at its
identical-failure ceiling, but continue independent cells. Changing target,
data, method, criteria, hardware class, environment packages, privacy, or total
budget requires a refreshed program and user direction.

## Phase Close Record

Every close record contains:

- objective and entry conditions;
- exact commands and run manifest;
- checks and exit codes;
- target and force identity hashes;
- engineering, sampler, and scientific ledgers kept separate;
- decision and inference-status tables;
- failure/repair history and remaining budget;
- strongest alternative explanation and evidence that would overturn the
  conclusion;
- explicit nonclaims;
- next subplan path and its review verdict.

## Review Rule

Use one bounded read-only Claude review for the material master/P0 design and
one terminal P8 review. Additional review is reserved for a material theorem,
target-boundary, or evidence-interpretation issue. Start with one exact path and
one question. Claude timeout or formatting-only disagreement is recorded and
does not stop valid local research.

## True Continuation Vetoes

- the shared executed kernel violates Chapter 48 assumptions;
- target identity or deterministic endpoint values cannot be established;
- completed evidence is corrupted and not reproducible within budget;
- trusted GPU remains unavailable after escalated device/framework probes;
- phase or total budget is exhausted;
- a repair requires package/environment mutation, destructive or external
  action, paid/expanded compute, privacy change, new hardware class, or a
  material change to target, method, criteria, or model scope.

Cell-local training, tuning, sampler, truth-tail, and performance failures are
not program vetoes.
