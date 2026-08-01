# Multi-Model NeuTra Filter-Posterior Plan Review Record

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `LOCAL_AUDIT_PASS_AFTER_FIVE_BOUNDED_CLAUDE_ROUNDS`

## Reviewed Scope

- Master program:
  `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-master-program-2026-07-15.md`
- Execution runbook:
  `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-execution-runbook-2026-07-15.md`
- Dedicated subplans P0-P7 with the same program prefix and date.

Claude reviewed only the master program under the repository's one-path bounded
prompt policy. Codex performed the cross-document and local source-context audit.
Claude was advisory and did not edit, launch, or authorize execution.

## Local Skeptical Audit

The initial audit checked wrong baselines, proxy promotion, target conflation,
missing stop rules, unfair comparisons, inherited defaults, stale LGSSM/SIR
context, environment mismatch, artifact sufficiency, structural-identity drift,
Zhao-Cui source boundaries, stochastic overclaim, and whether candidate failures
incorrectly stopped independent cells.

Two local consistency defects were repaired before external review:

- `SAMPLER_BLOCKED` was present in the runbook but absent from the master state
  vocabulary.
- Master aggregate budgets omitted CPU reference allocations present in P2-P6.

## Bounded Claude Review History

| Round | Verdict | Material finding | Visible repair |
| --- | --- | --- | --- |
| 1 | `REVISE` | Same-target samplers could agree on the same wrongly assembled posterior; first dense-IAF recipe failure could be mislabeled cell-level NeuTra rejection. | Added `POSTERIOR_IDENTITY_ADMITTED`, independent posterior/total-score recomposition and substitution-negative tests; split filter, recipe, and cell rejection states with family accounting. |
| 2 | `REVISE` | Multi-family rejection semantics were not funded by the one-recipe-per-cell budget. | Froze two candidate-family arms per cell and reserved one serious screen/training/confirmation path for each. |
| 3 | `REVISE` | Two 15-hour arms exhausted the 30-hour ceiling, leaving no budget for mandatory plain HMC or repairs. | Set 40 GPU-hours per cell: 15 + 15 family arms, 6 plain-HMC/comparator, 4 admission/infrastructure. Updated phase and aggregate budgets. |
| 4 | `REVISE` | R0/R1/R1B trusted GPU admission work had no explicit bucket ownership. | Assigned it to the 4-hour admission bucket and mapped exhaustion to target/implementation blockers. |
| 5 | `REVISE` | Common harness/schema/serialization defects could be mischarged to a cell admission bucket. | Restricted cell buckets to cell-specific adapters/artifacts; common defects reopen P1, consume only its 2-hour shared budget, preserve cell states, and fire a program continuation veto if unresolved. |

The five-round ceiling is exhausted. No sixth Claude review was requested. The
round-5 finding was narrow and mechanically resolvable; the terminal local audit
below verifies its cross-document propagation.

## Terminal Local Audit Contract

The terminal audit requires:

1. all ten program/master/runbook/subplan files plus this record exist;
2. all internal plan paths resolve;
3. P0-P7 each contain objective, entry/scope, artifacts, checks/reviews, evidence
   contract, default audit, repair triggers, forbidden actions/claims, handoff,
   stop conditions, and compute/attempt budget;
4. all eleven cell IDs appear in the master/runbook and their owning subplan;
5. no stale generic `CANDIDATE_REJECTED` label or pre-R1B HMC entry remains;
6. the GPU arithmetic is exact: eleven cells times 40 hours plus P1's 2-hour
   shared bucket equals 442 trusted GPU-hours;
7. every GPU action has one budget owner, and shared P1 defects cannot charge or
   classify a cell;
8. `git diff --check` passes for the complete program file set.

## Review Decision

`PASS_FOR_P0_DESIGN_EXECUTION_ONLY`.

The program is coherent enough to begin P0's inventory and identity-freeze work.
P2-P6 serious GPU training/HMC remains gated on P0 command/default/margin freeze,
P1 harness admission, and each cell's R1/R1B/comparator gates. This record does
not establish that any nonlinear cell is implemented, posterior-correct, trained,
or NeuTra-confirmed.
