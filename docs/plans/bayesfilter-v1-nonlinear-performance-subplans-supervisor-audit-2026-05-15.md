# Supervisor Audit: BayesFilter V1 Nonlinear Performance Subplans

## Date

2026-05-15

## Scope

Codex is supervising the critical-review loop requested by the user for the
NP0-NP7 subplans under:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

Subplans under review:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-baseline-plan-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-plan-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-plan-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-plan-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-jit-gate-plan-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-plan-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-plan-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-np7-consolidation-default-policy-plan-2026-05-15.md
```

Claude Code is used only as a bounded critical reviewer through:

```bash
bash scripts/claude_worker.sh --cwd /home/chakwong/BayesFilter --name <name> --model sonnet --effort high "<worker prompt>"
```

Claude must not edit files, run tests, run benchmarks, run GPU/CUDA/NVIDIA
probes, or run HMC diagnostics during this planning review.

The loop limit is 5 reviews.

## Codex Pre-Review Audit

Codex inspected:

- the nonlinear performance master program;
- the prior master-program supervisor audit;
- existing plan templates;
- representative V1 and Model B/C phase plans;
- source-map conventions;
- current worktree status.

Pre-review risks:

- subplans weaken the master evidence contract;
- NP2/NP3 omit derivation/proof gates;
- NP4 omits XLA support matrices;
- NP5 omits trusted GPU execution for benchmark commands;
- NP7 mixes engineering, numerical, and performance ledgers;
- subplans authorize execution without phase result artifacts.

Pre-review status:

- Initial NP0-NP7 subplans have been drafted as planning artifacts only.

## Review Log

### Claude Round 1

Verdict: `ACCEPT`.

Material blockers:

- None.

Non-blocking suggestions:

- Align NP0 entry gate with the subplans supervisor-audit filename.
- Add an explicit CUT4 exclusion or point-count cap in NP1 medium rows.
- Define how NP2/NP3 judge material regression for required rows.
- In NP4, distinguish GPU XLA `untested` from `tested_unsupported`.
- In NP7, require a final post-run red-team note.

Codex audit:

- Accepted all suggestions as useful auditability refinements, not blockers.

Revisions made:

- Added the subplans supervisor-audit file to the NP0 entry gate.
- Added a default NP1 CUT4 `point_count <= 512` cap and medium-row exclusion
  unless narrowed or separately approved.
- Added NP2/NP3 material-worsening rules based on median steady-state wall
  time and process RSS.
- Added explicit NP4 XLA status values for supported, unsupported, untested,
  skipped, and not-claimed cells.
- Added an NP7 post-run red-team note requirement.

### Claude Round 2

Verdict: `ACCEPT`.

Material blockers:

- None.

Non-blocking suggestions:

- None.

Codex audit:

- Accepted the convergence result.  The review loop stopped at round 2, within
  the user-requested maximum of 5 loops.

## Final Decision

Decision: `ACCEPT`.

The NP0-NP7 subplans are ready as planning artifacts.  They do not execute
benchmarks, authorize optimization patches, or support performance claims by
themselves.  Each future phase still requires its own phase execution,
evidence contract, skeptical audit, run manifest, result artifact, and
continuation decision.

## Verification

No numerical tests, benchmarks, GPU probes, or HMC diagnostics were run during
this subplan drafting pass.
