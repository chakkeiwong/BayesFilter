# BayesFilter Scalar Filtering HMC Validation Master Program

Date: 2026-07-09

## Status

`PHASE_2AE_CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`

## Objective

Move the scalar filtering-likelihood HMC lane from finite-telemetry mechanics
smoke toward governed HMC validation.  The program starts with native
divergence/trace policy, then only advances to short-chain validation,
reference agreement, GPU/XLA validation, and final closeout when each prior
gate preserves the evidence boundary.

This program is an `extension_or_invention` lane.  It does not close a
Zhao-Cui source-faithfulness gap.

## Starting Checkpoint

- Prior closeout:
  `docs/plans/bayesfilter-scalar-filtering-geometry-to-hmc-readiness-phase6-closeout-result-2026-07-08.md`
- Pushed commit at launch: `f297b10`
- Prior key artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_mass_handoff_cpu_hidden_2026-07-08.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_short_smoke_cpu_hidden_2026-07-08.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_replicated_diagnostic_cpu_hidden_2026-07-08.json`

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can the scalar filtering-likelihood HMC path produce validation evidence stronger than mechanics smoke without confusing telemetry availability, proxy diagnostics, or posterior claims? |
| Mechanism under test | Existing TensorFlow/TFP fixed-kernel scalar filtering HMC route, followed by modest CPU-hidden short-chain validation, reference agreement, then trusted GPU/XLA reproduction if earlier gates pass. |
| Expected failure mode | Missing native divergence telemetry, nonfinite samples/log probabilities/log accept ratios, acceptance stuck at boundary, unstable short-chain diagnostics, reference disagreement, GPU/XLA route mismatch, or unsupported scientific claim. |
| Promotion criterion | Phase-specific only.  Early phases may pass telemetry policy or short-chain validity screens; only a later reference-agreement phase may support a limited posterior-agreement claim for the scalar target. |
| Promotion veto | Treating unavailable native divergence telemetry as zero divergences, treating log-accept proxies as native divergences, post-hoc threshold changes, missing artifacts, or unsupported convergence/readiness/default/source-faithfulness claims. |
| Continuation veto | Broken scalar target preconditions, inability to preserve telemetry status, invalid or missing artifacts, nonfinite runtime outputs, failed reviewed subplan, or human-required GPU/default-policy boundary. |
| Repair trigger | Trace availability mismatch, large finite log-accept tails outside predeclared screen, boundary acceptance, reference disagreement, or Phase 0 finding that current route cannot support Phase 1 as designed. |
| Explanatory diagnostics | Acceptance, log-accept summaries, target-log-prob summaries, sample summaries, runtime, finite counts, R-hat/ESS if valid for the run length, and native divergence availability status when not available. |
| What must not be concluded | HMC readiness, broad posterior correctness, convergence, sampler superiority, default readiness, package readiness, GPU/XLA production readiness, zero divergences when telemetry is unavailable, or Zhao-Cui source faithfulness. |

## Phase Index

| Phase | Name | Objective | Subplan | Required result |
| --- | --- | --- | --- | --- |
| 0 | Governance and telemetry policy audit | Lock the runbook, review boundary, and verify the current native-divergence/trace contract before any new HMC run. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-result-2026-07-09.md` |
| 1 | CPU-hidden short-chain validation screen | Run a modest scalar fixed-kernel validation screen with predeclared finite, acceptance, and telemetry gates. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-result-2026-07-09.md` |
| 2 | Scalar reference agreement | Build or select a scalar reference and compare HMC draws under a predeclared agreement criterion. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-result-2026-07-09.md` |
| 2S | Geometry centering repair | Build a MAP-local SPD quadratic geometry/reference handoff after Phase 2R selected `outside_geometry_trust_region`. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-result-2026-07-09.md` |
| 2T | MAP-local reference handoff | Validate the Phase 2S MAP-local handoff and draft the next retuned fixed-kernel HMC screen if justified. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-result-2026-07-09.md` |
| 2U | Retuned MAP-local HMC screen | Run a CPU-hidden equal-trajectory-length fixed-kernel screen in the MAP-local `u_new` coordinate and select the first passing candidate for a later reviewed longer screen. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-result-2026-07-09.md` |
| 2V | Longer selected MAP-local screen | Run a longer CPU-hidden finite/acceptance screen for the Phase 2U selected fixed kernel before any GPU/XLA or posterior-agreement phase. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-result-2026-07-09.md` |
| 2W | MAP-local importance reference agreement | Build a fixed independent importance reference in `u_new` and compare Phase 2V HMC moment summaries only if reference validity passes. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-result-2026-07-09.md` |
| 2X | Shifted-mixture reference repair | Repair the failed Phase 2W standard-normal reference with a predeclared shifted-mixture proposal tuned only from Phase 2W pilot diagnostics. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-result-2026-07-09.md` |
| 2Y | Target geometry localization | Diagnose the target/proposal mismatch behind the Phase 2W/2X ESS failures before any further reference proposal attempt. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-result-2026-07-09.md` |
| 2Z | Proposal strategy pilot | Pilot heavier-tail, anchor, or ridge proposal strategies after Phase 2Y localized the Phase 2W/2X failures to proposal-family mismatch rather than artifact replay bugs. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-result-2026-07-09.md` |
| 2AA | Reference branch decision | Decide whether to abandon the current independent SNIS reference branch or move to a reviewed transport/sequential reference route after Phase 2Z nominated no proposal candidate. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-result-2026-07-09.md` |
| 2AB | Transport or sequential reference pilot | Test a CPU-hidden sequential tempering pilot after independent proposal routes failed. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-result-2026-07-09.md` |
| 2AC | Sequential resampling repair | Test whether fallback-boundary resampling repairs the Phase 2AB beta stall. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md` |
| 2AD | Diversity-preserving sequential repair | Test whether projected root-ancestor diversity gating preserves Phase 2AC beta progress and ancestor diversity. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-result-2026-07-09.md` |
| 2AE | Reference-method expansion decision | Decide whether to stop the current sequential branch or open a materially different reference-method design. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md` |
| 3 | Trusted GPU/XLA reproduction | Reproduce accepted scalar validation evidence under trusted GPU/XLA provenance, only after local reference/repair gates pass. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase3-gpu-xla-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase3-gpu-xla-result-2026-07-09.md` |
| 4 | Expansion decision | Decide whether to lift dimension, open Zhao-Cui source-anchor work, or repair covariance/tuning based on evidence. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase4-expansion-decision-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase4-expansion-decision-result-2026-07-09.md` |
| 5 | Closeout | Close the program with decision tables, inference-status table, manifest, and nonclaims. | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase5-closeout-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase5-closeout-result-2026-07-09.md` |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can this scalar fixed-kernel HMC route pass increasingly meaningful validation gates without overstating unsupported scientific conclusions? |
| Baseline/comparator | The 2026-07-08 scalar filtering finite-telemetry closeout and its CPU-hidden artifacts. |
| Primary pass criterion | All phase gates pass with artifacts, review, and nonclaims preserved. |
| Veto diagnostics | Nonfinite runtime outputs, unavailable divergence treated as zero, positive native divergence if available, invalid reference, missing provenance, failed review, changed criteria after seeing results, or unsupported claim. |
| Explanatory diagnostics | Acceptance, log-accept tail summaries, target-log-prob summaries, R-hat/ESS if valid, runtime, sample summaries, and trace availability. |
| Not concluded | HMC readiness, convergence, broad posterior correctness, zero divergences without native telemetry, sampler superiority, default readiness, package readiness, GPU/XLA production readiness, or Zhao-Cui source faithfulness. |
| Artifacts | Master program, visible runbook, execution ledger, stop handoff, per-phase subplans/results, JSON/Markdown benchmark artifacts, bounded review bundles, and ignored `.claude_reviews/` logs. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| CPU-hidden Phase 0/1 execution | Prior scalar closeout and repo GPU policy allowing CPU-only debug/reference checks | Keeps first validation cheap and isolates scientific mechanics from GPU runtime | CPU evidence mistaken for production evidence | Artifacts must record `CUDA_VISIBLE_DEVICES=-1` and nonclaims | reviewed hypothesis |
| Native divergence policy | `bayesfilter/inference/hmc.py` trace status and prior artifacts | Directly addresses the largest residual gap | Missing trace treated as zero divergences | Phase 0 source/artifact/test audit | launch gate |
| Short-chain validation before reference agreement | Prior Phase 5 was tiny finite telemetry only | Smallest next evidence-bearing step | Proxy finite telemetry promoted into posterior correctness | Phase 1 nonclaims and veto table | reviewed hypothesis |
| Reference agreement after short-chain screen | Evidence discipline for posterior claims | Prevents posterior correctness claims from mechanics alone | Invalid reference or unfair comparator | Phase 2 reviewed reference criterion | future gate |
| GPU/XLA after CPU/reference gates | BayesFilter default target is GPU/XLA | Avoids spending trusted GPU time before local scientific gates | GPU pass mistaken for posterior correctness | Phase 3 provenance and nonclaims | future gate |

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the prior scalar finite-telemetry closeout, not HMC success. |
| Proxy metrics promoted | Acceptance/log-accept/finite telemetry cannot establish posterior correctness; reference agreement is a separate later gate. |
| Missing stop conditions | Each subplan has explicit continuation vetoes, review vetoes, and forbidden claims. |
| Unfair comparison | No method ranking occurs.  Phase 2 must define a scalar reference before any agreement interpretation. |
| Hidden assumptions | Numeric thresholds in later phases are hypotheses until refreshed in the phase subplan before execution. |
| Stale context | Phase 0 re-audits source and artifacts at current commit before runtime. |
| Environment mismatch | CPU-hidden artifacts cannot support GPU/XLA/default readiness. |
| Artifact mismatch | Each phase result must name the command, artifact path, diagnostic roles, and nonclaims. |

Audit status: `PASSED_FOR_PHASE_0_REVIEW_AND_TELEMETRY_POLICY_AUDIT`.

## Review Protocol

Claude may be used as a read-only reviewer through
`/home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh`.  If the review
gate cannot obtain a usable Claude verdict, Codex must follow the review guide:
probe, redesign the bundle if Claude is alive, and use a fresh Codex substitute
review only when Claude is unavailable.  Claude cannot authorize runtime,
human, model-file, funding, product-capability, default-policy, or scientific
claim boundaries.
