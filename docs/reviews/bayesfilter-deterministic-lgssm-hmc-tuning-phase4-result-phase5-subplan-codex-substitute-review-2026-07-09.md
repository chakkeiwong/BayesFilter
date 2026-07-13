# Codex Substitute Review: Deterministic LGSSM HMC Tuning Phase 4 Result / Phase 5 Subplan

Date: 2026-07-09

## Scope

This is a local Codex substitute review for the Phase 4 result and Phase 5
subplan. It is not a Claude review. The Claude review gate was attempted with
the bounded review bundle
`docs/reviews/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-result-phase5-subplan-review-bundle-2026-07-09.md`,
but the approval layer rejected the external call as data-exfiltration risk.
No retry or workaround was attempted.

## Paths Reviewed

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-xla-score-gate-result-2026-07-09.md`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase5-geometry-mass-subplan-2026-07-09.md`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json`

## Review Findings

| Check | Finding |
| --- | --- |
| Phase 4 scope | The result limits the claim to XLA value/score compile admissibility. |
| Evidence support | The JSON artifact records `jit_compile=true`, finite value/score, no non-JIT runtime, no runtime autodiff tape, one concrete function, timing, HLO size/hash, and empty vetoes. |
| Boundary safety | The result explicitly avoids HMC convergence, posterior recovery, sampler superiority, production, GPU, and scientific claims. |
| Phase 5 handoff | The Phase 5 subplan inherits Phase 4 and uses existing BayesFilter geometry/mass tools. |
| Manual tuning risk | The Phase 5 subplan forbids manual mass edits and serious HMC. |
| Human approval boundary | The runbook still requires explicit approval for Phase 6 serious HMC, Phase 7 long sampling, Phase 8 serious recovery, and any GPU/CUDA/NeuTra command. |

## Limitations

This review is local and self-review-like. It is weaker than an independent
Claude read-only review and should not be represented as Claude convergence.

## Verdict

`AGREE_WITH_LIMITATION`

The Phase 4 result supports advancing to deterministic Phase 5 geometry/mass
initialization only. It does not support HMC, posterior recovery, production,
GPU, or scientific claims.
