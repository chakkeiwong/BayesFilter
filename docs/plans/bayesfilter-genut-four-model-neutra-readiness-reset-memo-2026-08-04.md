# GenUT Four-Model NeuTra Readiness Reset Memo

Date: 2026-08-04

## Current State

- Serious target-specific NeuTra training is admitted for LGSSM, KSC-SV, and
  predator-prey through
  `bayesfilter.highdim.cubature_genut_neutra_targets.make_admitted_genut_neutra_target`.
- LGSSM must run with TF32 disabled and target signature
  `f33c1f18fe35871d26d5143e8063fb92a88044431e17ff453da93f3f9628aeaa`.
- KSC-SV uses TF32 and target signature
  `53c41570c2fde8ed3693e5044919127efd7a9cb4843a078a97a2be616fae3627`.
- Predator-prey uses TF32 and target signature
  `41b9b4f24d880554e222d36b465b390f4360f1efec8570b2cef6f1b06c790dbe`.
- Austria SIR is blocked.  Its tangent-free endpoint differs from the
  value/score route by `0.001568` relative, above `0.0002`.
- The active Austria stabilization verdict was revised on 2026-08-05:
  pairwise higher-moment correction is materially useful and is the leading
  repair candidate.  At `T=20,N=1008`, it reduced the three score SDs by
  `94.1x`, `71.6x`, and `14.6x` with `16/16` valid rows.  Its historical
  promotion failure records a `1.260` finite-value shift, not failure to
  stabilize the score.
- No model is HMC-ready.  No serious NeuTra training or HMC chain has run.

## Canonical Evidence

- Plan:
  `docs/plans/bayesfilter-genut-four-model-neutra-readiness-plan-2026-08-04.md`
- Result:
  `docs/plans/bayesfilter-genut-four-model-neutra-readiness-result-2026-08-04.md`
- Austria pairwise verdict reassessment:
  `docs/plans/bayesfilter-austria-genut-pairwise-verdict-reassessment-2026-08-05.md`
- Aggregate:
  `docs/benchmarks/artifacts/genut_four_model_neutra_readiness_20260804/aggregate_attempt04/result.json`
- Aggregate manifest:
  `docs/benchmarks/artifacts/genut_four_model_neutra_readiness_20260804/aggregate_attempt04/run_manifest.json`

Historical attempt directories under the same root are preserved.  Do not use
the earlier LGSSM TF32 target, stale tuning paths, or pre-telemetry training
attempts for admission.

## Next Campaign

Write separate target-specific NeuTra training protocols for the three admitted
models.  Each needs an objective/scaling check, architecture and capacity
ladder, optimizer and learning-rate search, budget ladder, multi-seed policy,
heldout target/status criteria, and downstream sequential NeuTra-HMC tuning and
confirmation gates.  LGSSM must keep TF32 disabled throughout target
evaluation.  KSC-SV and predator-prey keep TF32 enabled.

Austria is a separate repair lane.  First localize the tangent-free versus
tangent-carrying value mismatch.  Then port and retune pairwise correction for
the exact batch-native Austria scope.  Pairwise is a required comparator and
the leading stabilization candidate; diagonal-only is the naive baseline and
failure control, not the sole candidate.  Do not train on Austria, substitute a
damped force, or call the current pairwise output an admitted score until its
own same-finite-program, scalar-parity, finite-difference, and replay gates
pass.
