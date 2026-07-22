# Deterministic LGSSM HMC Phase 7 Plan Substitute Review

Date: 2026-07-11

Reviewed path:
`docs/plans/bayesfilter-deterministic-lgssm-hmc-phase7-repair-and-execution-plan-2026-07-11.md`

## Claude Availability

Codex attempted the required smallest one-path Claude review through
`scripts/claude_worker.sh`. The managed approval boundary rejected the call
before execution because sending the workspace plan to an external reviewer
was classified as unacceptable external disclosure. No plan content was sent.
Codex did not retry, broaden the prompt, or route around the rejection.

Review strength: fresh Codex read-only substitute review, not Claude evidence.

## Findings Resolved Before Verdict

1. The draft specified split-chain R-hat but did not say whether bulk and tail
   ESS used the same split-chain representation. The plan now requires
   split-chain bulk and tail ESS and fixes pooled linear-interpolation q05/q95
   thresholds before splitting.
2. The draft deferred root-seed selection and CPU thread allocation to
   implementation. The plan now pins root seed `(20260711, 701)`, the exact
   per-worker/per-stage/per-check seed formula, and two workers with eight
   TensorFlow intra-op threads each and one inter-op thread each.
3. The draft said to map out of both mass transforms but did not state how the
   nested replay adapter is traversed. The plan now requires the final adapter
   transform followed by the Phase 4/base adapter transform and asserts that
   the terminal adapter is the original LGSSM target.
4. The draft placed the actual-target smoke before the Phase 6 refresh that
   creates its private replay input. The phase order now runs static/unit gates,
   the hash-matching Phase 6 replay refresh, then the actual-target smoke.

## Skeptical Audit

| Risk | Verdict |
| --- | --- |
| Wrong baseline | Controlled by exact config, fixture, XLA, geometry, mass, selected-step, selected-trajectory, public-kernel, and private-loop hash gates. |
| Proxy promotion | Controlled; smoke, acceptance, compile success, and early checks cannot pass Phase 7. |
| Missing stop conditions | Controlled by diagnostic caps, hard vetoes, process/artifact vetoes, and machine wall-time cap. |
| Hidden statistical choice | Controlled after fixing ranks, split/fold transforms, q05/q95, ESS split semantics, burn-in window, retained accumulation, and all-parameter aggregation. |
| Environment mismatch | Controlled by CPU hiding before worker TensorFlow import, spawned workers, fixed thread allocation, version/device capture, and XLA-only execution. |
| Non-executable artifact | Controlled by using BayesFilter's existing private replay API and requiring a deterministic Phase 6 refresh with exact hash equality. |
| Misleading conclusion | Controlled by keeping posterior recovery in Phase 8 and retaining explicit nonclaims. |

## Verdict

The reviewed plan is executable and appropriately fail-closed. Its remaining
conditions are implementation and engineering gates, not unresolved planning
choices. Serious Phase 7 may run only after the stated gates pass.

VERDICT: AGREE
