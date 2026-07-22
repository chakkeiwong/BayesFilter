# Codex Substitute Review: Phase 8 Result And Phase 9 Subplan, Iteration 1

Date: 2026-07-10

## Scope And Limitation

Fresh local read-only review of the Phase 8 result, Phase 9 subplan, current
nonlinear score CLIs, shared score contract, and LGSSM seed-shard precedent.
This is not independent Claude review. Claude remains policy-blocked as
external repository disclosure, and no workaround was attempted.

## Findings

| Severity | Finding |
| --- | --- |
| Blocking | The first Phase 9 draft required exact GPU commands but named a shared harness that did not yet exist. It therefore could not be reviewed as an executable experiment plan. |
| Blocking | The promotion criterion said "existing row tolerances" without freezing exact FD steps/tolerances or stating whether every seed versus only the aggregate had to pass. This allowed post-result threshold ambiguity. |
| Pass | Phase 8 claims are limited to CPU-hidden wiring evidence and match the four passing post-repair shards. |
| Pass | Sequential wrappers preserve seed order and aggregate log-likelihood/gradient means; default diagnostics do not call historical score routes. |
| Pass | The Phase 9 draft correctly rejects historical fixed-SIR manual-VJP memory as compact evidence and identifies the missing nonlinear XLA/trust/reset-memory harness fields. |

## Required Revision

- Split Gate A implementation authorization from GPU execution authorization.
- Name the exact Gate A files and CPU-hidden checks.
- Require a post-implementation exact GPU command manifest and fresh review.
- Freeze per-row FD step, absolute tolerance, relative tolerance, pass rule,
  and both per-seed plus aggregate requirements before any GPU run.

VERDICT: REVISE

