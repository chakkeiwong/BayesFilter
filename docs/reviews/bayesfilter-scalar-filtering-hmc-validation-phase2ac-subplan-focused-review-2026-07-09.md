# Phase 2AC Focused Local Review: Sequential Resampling Repair

Date: 2026-07-09
Status: `VERDICT_AGREE_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`

## Scope

Focused local review of the repaired Phase 2AC subplan after Claude review
round 1 found that the runtime command and timeout were missing.

Reviewed artifact:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-subplan-2026-07-09.md`

Claude review artifact:

- `docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase2ac-subplan-claude-review-round1-2026-07-09.md`

## Verdict

`VERDICT: AGREE_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`

## Findings

| Check | Status |
| --- | --- |
| Claude finding addressed | Pass.  The subplan now predeclares a CPU-hidden runtime command with `timeout 600`. |
| Evidence contract | Pass.  The contract asks only whether a boundary-resampling policy repairs the Phase 2AB beta stall enough to nominate independent replication. |
| Baseline | Pass.  The baseline is the final repaired Phase 2AB artifact, not HMC success or posterior correctness. |
| Proxy metrics | Pass.  ESS, max weight, ancestor diversity, and rejuvenation acceptance can nominate replication only. |
| Stop conditions | Pass.  Runtime, artifact, review, claim-boundary, and continuation-veto stops are explicit. |
| Boundary safety | Pass.  The subplan forbids HMC, GPU/XLA, default-policy, product, model-file, and Zhao-Cui source-faithfulness boundaries. |
| Runtime feasibility | Pass.  The command is bounded by `timeout 600`; timeout is a continuation veto, not scientific evidence. |

## Implementation Notes

The implementation should preserve the Phase 2AB target route, seed, particle
count, thresholds, and rejuvenation policy.  The only intended algorithmic
change is to force nonterminal resampling after either:

- a `bisection_largest_minimum_admissible_increment` beta-selection step; or
- a post-temperature ESS ratio within `1.0e-4` of the resampling threshold.

## Nonclaims

This review does not authorize a valid reference claim, HMC-vs-reference
agreement, posterior correctness, HMC readiness, convergence, a zero-divergence
claim, sampler superiority, statistical ranking, GPU/XLA readiness, default
readiness, or Zhao-Cui source faithfulness.
