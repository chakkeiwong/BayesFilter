# Phase 2AD Focused Local Review: Diversity-Preserving Sequential Repair

Date: 2026-07-09
Status: `VERDICT_AGREE_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`

## Scope

Focused local review of the repaired Phase 2AD subplan after Claude review
round 1 returned `VERDICT: REVISE`.

Reviewed artifact:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-subplan-2026-07-09.md`

Claude review artifact:

- `docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase2ad-subplan-claude-review-round1-2026-07-09.md`

## Verdict

`VERDICT: AGREE_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`

## Findings

| Check | Status |
| --- | --- |
| Core repair rule | Pass.  Projected diversity is now defined as realized unique root-ancestor fraction from the systematic-resampling index vector. |
| RNG semantics | Pass.  The resampling draw is consumed before accept/skip, so the fixed-seed path is deterministic. |
| Terminal semantics | Pass.  Terminal resampling is disallowed; ESS/max-weight remain beta-one pre-final-resampling measurements; diversity is measured after the last nonterminal stage. |
| Regression command | Pass.  The Phase 2AC regression file and exact focused pytest command are pinned. |
| Evidence contract | Pass.  The phase can only nominate independent replication and cannot certify a reference or HMC readiness. |
| Boundary safety | Pass.  HMC, GPU/XLA, default-policy, model-file, product, and Zhao-Cui source-faithfulness boundaries remain forbidden. |

## Nonclaims

This review does not authorize a valid reference claim, HMC-vs-reference
agreement, posterior correctness, HMC readiness, convergence, a zero-divergence
claim, sampler superiority, statistical ranking, GPU/XLA readiness, default
readiness, or Zhao-Cui source faithfulness.
