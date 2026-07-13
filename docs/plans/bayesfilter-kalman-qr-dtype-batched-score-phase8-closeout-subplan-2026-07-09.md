# Phase 8 Subplan: Closeout

Date: 2026-07-09

## Phase Objective

Close the program with decision tables, inference-status table, manifest,
remaining risks, and explicit nonclaims.

## Entry Conditions Inherited From Previous Phase

- Phase 7 benchmark/correctness result exists or a blocker result explains why
  the program stopped earlier.
- All reached phase artifacts are available.

## Required Artifacts

- Phase 8 closeout result.
- Updated stop handoff.
- Optional reset memo if future agents would otherwise rediscover the dtype or
  batched-score boundary.

## Required Checks, Tests, And Reviews

Run:

```bash
git diff --check -- docs/plans docs/benchmarks bayesfilter/linear tests scripts
```

Claude read-only review is required for the closeout if implementation or
benchmark phases were reached.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What can be concluded from the reached phases, and what remains unsupported? |
| Baseline/comparator | Master program evidence contract and reached phase results. |
| Primary criterion | Closeout includes decision table, inference-status table, run manifest, post-run red-team note, and nonclaims. |
| Veto diagnostics | Unsupported claim, missing artifact path, missing failed-gate explanation, or timing ranked without uncertainty. |
| Explanatory diagnostics | Summary tables and remaining risk inventory. |
| Not concluded | Any claim not directly supported by reached phase evidence. |
| Artifact | Phase 8 closeout result. |

## Forbidden Claims And Actions

- Do not mark the research or engineering direction complete if a blocker fired.
- Do not conflate candidate failure with invalidating the whole direction.
- Do not commit or push unless explicitly requested.

## Exact Next-Phase Handoff Conditions

No next phase.  The closeout must say whether the program is complete, blocked,
or intentionally deferred.

## Stop Conditions

Stop if required result artifacts are missing or if closeout review identifies
unsupported claims that cannot be repaired locally.
