# Reset Memo: Scalar Filtering HMC Validation

## Date

2026-07-10

## Current Status

`PHASE_2AE_CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`

The scalar filtering-likelihood HMC validation lane has not produced a valid
reference or HMC-reference agreement handoff.  Phase 3 GPU/XLA reproduction is
blocked.

Primary recovery handoff:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-stop-handoff-2026-07-09.md`

Primary decision artifact:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md`

## Context

The work started from scalar MAP-local HMC mechanics and attempted to build a
scalar reference for posterior-agreement validation.  Independent SNIS-style
reference proposals failed earlier.  The latest branch tested a sequential
tempering reference route in the Phase 2S/2U MAP-local `u_new` coordinate.

This reset memo exists so future sessions do not re-run the same local
fallback-resampling repairs or accidentally proceed to GPU/XLA or HMC-readiness
claims without a valid reference.

## Decision / Policy

The current local fallback-resampling sequential-reference branch is blocked.
Continuing requires either:

1. A reviewed materially different reference-method design, such as stronger
   rejuvenation/move design, deterministic transport, learned transport, or a
   different reference family informed by Phase 2Y geometry localization.
2. A closeout that records reference agreement unresolved.

Do not proceed to Phase 3 GPU/XLA from the current evidence state.

## 2026-07-11 Predictive-Equivalence Direction

The owner has selected a new, narrower research question: compare the
posterior-predictive laws induced by ordinary MAP-local HMC and a frozen,
exact-corrected NeuTra-HMC parameterization instead of requiring an independent
four-parameter posterior reference as the only validation route.

The governing design is:

- monograph chapter:
  `docs/chapters/ch28a_neural_network_state_space_model_applications.tex`;
- implementation/test master program:
  `docs/plans/bayesfilter-scalar-ssl-lstm-predictive-equivalence-master-program-2026-07-11.md`.

This does not reopen the blocked local fallback-resampling branch and does not
unblock the old Phase 3 GPU/XLA reproduction. It creates a separate reviewed
program whose possible positive claim is limited to predictive-functional
equivalence for a fixed target, data set, forecast operator, horizon, and
predeclared statistical design. Parameter-posterior agreement, posterior
correctness, identification, SVD-UKF exactness, model adequacy, sampler
superiority, and default readiness remain unresolved.

## Evidence Summary

| Phase | Result | Interpretation |
| --- | --- | --- |
| 2AB | Beta stalled at `0.3419540270406287`; no reference nomination. | Baseline sequential pilot could not complete the bridge. |
| 2AC | Beta reached `1.0`; terminal ESS ratio `0.9912539055044092`; max weight `0.010002188339361427`; unique ancestor fraction fell to `0.21875 < 0.25`. | Forced fallback resampling repaired beta progression but spent too much root-ancestor diversity. |
| 2AD | Unique ancestor fraction stayed `0.4140625`; beta stalled at `0.9712250668187553`. | Diversity-preserving fallback skipping avoided collapse but lost beta completion. |
| 2AE | Decision-only result: current sequential branch blocked. | A materially different reference method is required before more runtime. |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Stop the current local fallback-resampling sequential branch | Passed for decision only | No scientific/default/HMC/GPU promotion is supported | A better reference may need a materially different bridge or move design | Draft a new reviewed reference-method branch, or close out unresolved | No valid reference, HMC-reference agreement, posterior correctness, HMC readiness, convergence, zero-divergence claim, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Current sequential branch blocked for reference nomination. |
| Reference validity | Not established. |
| HMC-reference agreement | Not assessed. |
| Statistically supported ranking | None. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed materially different reference-method design, or closeout with reference agreement unresolved. |

## Important Artifacts

- Master program:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`
- Visible runbook:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-gated-execution-runbook-2026-07-09.md`
- Execution ledger:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-execution-ledger-2026-07-09.md`
- Stop handoff:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-stop-handoff-2026-07-09.md`
- Phase 2AB result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-result-2026-07-09.md`
- Phase 2AC result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md`
- Phase 2AD result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-result-2026-07-09.md`
- Phase 2AE result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md`

## Reviews

- Phase 2AC Claude review found the missing runtime command/timeout; the
  subplan was patched and locally re-reviewed before runtime.
- Phase 2AD Claude review found ambiguous projected-diversity and terminal
  resampling semantics; the subplan was patched and locally re-reviewed before
  runtime.
- Phase 2AE was a no-runtime decision phase with focused local review.

Claude did not authorize runtime, scientific claims, HMC readiness, GPU/XLA
readiness, default readiness, or source-faithfulness claims.

## Verification Already Run

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference.py
python -m py_compile docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_2026_07_09.py
git diff --check
```

Observed:

- `39 passed, 2 warnings`.
- Both Phase 2AC and Phase 2AD harnesses compiled.
- `git diff --check` passed.

## Known Limitations / Cautions

- All scalar reference-branch runs here were CPU-hidden debug/reference
  exceptions with `CUDA_VISIBLE_DEVICES=-1`; they do not support GPU/XLA
  readiness.
- The sequential branch evidence is one-seed diagnostic evidence.  It is enough
  to block this local repair branch, not enough to reject the target or HMC
  mechanics.
- Do not interpret ESS, max weight, acceptance, or ancestor diversity as
  posterior correctness.  In this lane they were only reference-nomination
  diagnostics.
- The worktree is dirty and contains unrelated Kalman QR dtype/batching work
  and other generated artifacts.  Preserve unrelated changes.

## Suggested Next Steps

1. Treat the predictive-equivalence master program as the active next research
   branch and draft its focused Phase 0 governance-and-target-lock subplan.  No
   implementation or runtime is authorized by this reset memo.
2. Review and freeze the Phase 0 target, data, parameter chart, forecast
   semantics, baseline, comparison arm, evidence roles, and artifact contracts
   before beginning Phase 1 implementation.
3. Keep the old parameter-reference Phase 3 GPU/XLA route blocked.  Reopening
   that route still requires a valid reference and HMC-reference agreement
   handoff; it is not unblocked by a future predictive-equivalence result.
4. If the old scalar reference program is revisited instead, first choose
   between closing with reference agreement unresolved and reviewing a
   materially different reference-method branch.  Do not make another
   unreviewed fallback-resampling repair.

## Nonclaims

- No valid independent reference.
- No valid sequential reference.
- No HMC-reference agreement.
- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
