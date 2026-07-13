# Codex Substitute Review: Phase 6Z Verification Chunk Repair Subplan

Date: 2026-07-09

## Scope

Local read-only substitute review for:

`docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6z-verification-chunk-xla-compile-memory-repair-subplan-2026-07-09.md`

Claude review was attempted through the one-path review gate, but the approval
layer rejected the command as an external data-exfiltration risk. No workaround
was attempted. This substitute review is weaker than Claude review and must not
be described as Claude convergence.

## Review Question

Does the Phase 6Z subplan repair the final verification XLA compile-memory
blocker without introducing manual tuning, non-XLA fallback, runtime
`GradientTape`, weakened verification evidence, target/fixture/mass changes, or
Phase 7 sampling?

## Findings

No material blocker.

The subplan targets the observed Phase 6Y failure boundary:
`verification_start` after a completed/passed trajectory handoff. It does not
treat the trajectory pass as HMC readiness and keeps Phase 7 blocked until
Phase 6 emits a passed final kernel payload/hash.

The key safety point is present: the plan separates
`verification_chunk_max_results` from
`verification_min_retained_results_for_pass`. This prevents a smaller XLA chunk
from silently weakening the R-hat verification gate by allowing early promotion
after too few retained draws.

## Required Implementation Checks

- Validate both new policy fields as positive integers when present.
- Keep defaults backward-compatible for existing small tests.
- For serious deterministic LGSSM runs, use a smaller compile chunk while
  preserving the total verification cap and requiring the full cap before pass.
- Add tests for the early-R-hat pass guard.
- Preserve `jit_compile=true`, CPU-hidden execution, and no runtime
  `GradientTape`.

## Verdict

VERDICT: AGREE
