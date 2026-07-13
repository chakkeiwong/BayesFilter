# Scalar Filtering HMC Validation Visible Stop Handoff

Date: 2026-07-09

## Current Status

`PHASE_2AE_CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`

## Current Scope

This visible runbook governs scalar filtering-likelihood HMC validation after
the 2026-07-08 finite-telemetry mechanics closeout.  It begins with native
divergence/trace policy and cannot claim HMC readiness, convergence, posterior
correctness, zero divergences, GPU/XLA readiness, default readiness, or
Zhao-Cui source faithfulness.

## Resume Point

Resume after Phase 2AE:

1. Read the sequential branch closeout ledger entry in
   `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-execution-ledger-2026-07-09.md`.
2. Read Phase 2AB result:
   `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-result-2026-07-09.md`.
3. Read Phase 2AC result:
   `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md`.
4. Read Phase 2AD result:
   `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-result-2026-07-09.md`.
5. Read Phase 2AE result:
   `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md`.
6. Do not proceed to Phase 3 GPU/XLA.  Phase 3 remains blocked because no
   valid reference or HMC-reference agreement handoff exists.
7. Next valid action is either a reviewed materially different reference-method
   design branch or a scalar validation closeout with reference agreement
   unresolved.

## Key Artifacts

- Master program:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`
- Visible runbook:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-gated-execution-runbook-2026-07-09.md`
- Ledger:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-execution-ledger-2026-07-09.md`
- Phase 0 subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-subplan-2026-07-09.md`
- Phase 0 result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-result-2026-07-09.md`
- Phase 1 subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md`
- Phase 1 result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-result-2026-07-09.md`
- Phase 1R subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-subplan-2026-07-09.md`
- Phase 1R result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-result-2026-07-09.md`
- Phase 2 subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-subplan-2026-07-09.md`
- Phase 2 result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-result-2026-07-09.md`
- Phase 2R subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2r-local-reference-localization-subplan-2026-07-09.md`
- Phase 2R result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2r-local-reference-localization-result-2026-07-09.md`
- Phase 2S subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-subplan-2026-07-09.md`
- Phase 2S result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-result-2026-07-09.md`
- Phase 2T subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-subplan-2026-07-09.md`
- Phase 2T result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-result-2026-07-09.md`
- Phase 2U subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-subplan-2026-07-09.md`
- Phase 2U result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-result-2026-07-09.md`
- Phase 2V subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-subplan-2026-07-09.md`
- Phase 2V result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-result-2026-07-09.md`
- Phase 2W subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-subplan-2026-07-09.md`
- Phase 2W result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-result-2026-07-09.md`
- Phase 2X subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-subplan-2026-07-09.md`
- Phase 2X result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-result-2026-07-09.md`
- Phase 2Y subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-subplan-2026-07-09.md`
- Phase 2Y result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-result-2026-07-09.md`
- Phase 2Z subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md`
- Phase 2Z result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-result-2026-07-09.md`
- Phase 2Z local substitute review:
  `docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase2z-subplan-codex-substitute-review-2026-07-09.md`
- Phase 2AA subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-subplan-2026-07-09.md`
- Phase 2AA result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-result-2026-07-09.md`
- Phase 2AB result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-result-2026-07-09.md`
- Phase 2AC result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md`
- Phase 2AD result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-result-2026-07-09.md`
- Phase 2AE result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md`

## Nonclaims

- No HMC readiness.
- No HMC convergence.
- No posterior correctness.
- No zero-divergence claim unless native divergence telemetry is available and
  zero under a reviewed gate.
- No sampler superiority or statistical ranking.
- No GPU/XLA production/default readiness.
- No Zhao-Cui source-faithfulness claim.
