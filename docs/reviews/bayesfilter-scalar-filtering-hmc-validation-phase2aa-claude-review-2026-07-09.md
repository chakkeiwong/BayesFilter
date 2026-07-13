# Phase 2AA Claude Review: Reference Branch Decision

Date: 2026-07-09
Status: `VERDICT_AGREE`

## Scope

Read-only Claude review of the compact Phase 2AA decision plan.  Claude did
not edit files, run commands, authorize runtime, authorize GPU/XLA, or
authorize scientific/default/source-faithfulness claims.

## Compact Review Bundle

- Phase 2W standard-normal SNIS reference failed ESS `22.894679726459746`
  and ESS ratio `0.022358085670370845` below gates.
- Phase 2X shifted-mixture SNIS reference failed ESS `33.4215730897076`
  and ESS ratio `0.01631912748520879` below gates.
- Phase 2Y found affine/proposal-log-density replay passed and
  proposal-family/global-geometry mismatch plausible, with top-weight anchors
  outside the Phase 2S trust region and large quadratic residuals.
- Phase 2Z heavier-tail/anchor/ridge Student-t pilots had finite values but no
  candidate met ESS `>= 256`, ESS ratio `>= 0.05`, and max weight `<= 0.05`.
- Proposed Phase 2AA decision: abandon blind independent SNIS tweaks for this
  target for now, keep Phase 3 GPU/XLA blocked, and draft a reviewed Phase 2AB
  transport/sequential-reference subplan before runtime.

## Claude Verdict

`VERDICT: AGREE`

Claude agreed that the Phase 2AA decision is justified as a narrow engineering
gate decision, provided the result remains scoped to the tested independent
SNIS branch and does not claim SNIS impossibility, target invalidity, HMC
invalidity, posterior correctness, convergence, default readiness, or GPU/XLA
readiness.

## Findings To Preserve

| Finding | Phase 2AA response |
| --- | --- |
| The baseline is acceptable only if stated narrowly. | The result says blind independent SNIS tweaks are abandoned for this target for now, not that all independent references or SNIS are impossible. |
| ESS, ESS ratio, and max weight are reference-validity gates, not scientific proof. | The result uses them only to reject reference candidates and does not promote them to posterior or sampler claims. |
| Phase 3 GPU/XLA should remain blocked. | The result keeps Phase 3 blocked because reference adequacy remains unresolved. |
| Phase 2AB needs explicit comparator, promotion status, continuation veto, and artifacts. | The Phase 2AB subplan states these fields before runtime. |
| Phase 2Y makes proposal-family/global-geometry mismatch plausible, not proven. | The result frames Phase 2AB as a discriminating repair, not as confirmation of a diagnosis. |

## Review Strength

This was a bounded Claude material review of a compact bundle, not a full-file
review of every artifact.  It is stronger than the earlier local substitute
reviews but still does not authorize runtime, HMC readiness, posterior
correctness, default policy, GPU/XLA readiness, or Zhao-Cui source faithfulness.
