# Phase 0 Subplan: Governance And Native-Divergence Telemetry Policy

Date: 2026-07-09

## Phase Objective

Create and review the validation runbook, then audit the current scalar HMC
trace contract so Phase 1 cannot mistake unavailable native divergence
telemetry for zero divergences or promote log-accept proxies into native
divergence evidence.

## Entry Conditions

- Current branch is `main` at or after `f297b10`.
- Prior scalar filtering geometry-to-HMC runbook is closed with boundaries.
- No untracked non-ignored files are present before execution.
- Claude review gate is attempted for material governance review, or a
  documented Codex substitute review is used only if Claude is unavailable.

## Required Artifacts

- Master program:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`
- Visible runbook:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-gated-execution-runbook-2026-07-09.md`
- Ledger:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-execution-ledger-2026-07-09.md`
- Stop handoff:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-stop-handoff-2026-07-09.md`
- Review bundle:
  `docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase0-governance-review-bundle-2026-07-09.md`
- Phase 0 result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-result-2026-07-09.md`
- Refreshed Phase 1 subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md`

## Required Checks, Tests, And Reviews

Run:

```bash
git diff --check
CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_common_inference_runtime_contracts.py::test_hmc_diagnostics_distinguish_unavailable_from_zero_divergences \
  tests/test_common_inference_runtime_contracts.py::test_hmc_screen_keeps_unavailable_diagnostics_from_passing_as_zero \
  tests/test_common_inference_runtime_contracts.py::test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence \
  tests/test_nonlinear_ssm_phase4_full_chain_hmc.py::test_phase4_tiny_full_chain_hmc_jit_returns_finite_samples_and_metadata \
  tests/test_scalar_ssl_lstm_filtering_hmc_replicated_diagnostic.py::test_aggregate_seed_rows_is_descriptive_only
```

Review:

- Claude read-only review gate for governance artifacts and Phase 0 subplan.
- If Claude is unavailable, a fresh Codex substitute review must inspect the
  same review questions and record weaker review status.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the current scalar HMC route preserve native-divergence availability semantics strongly enough to launch Phase 1 without a zero-divergence claim? |
| Baseline/comparator | Prior scalar Phase 6 closeout and current `bayesfilter/inference/hmc.py` / `hmc_diagnostics.py` semantics. |
| Primary criterion | Source/artifact/test audit confirms unavailable native divergence remains unavailable, not zero; plan artifacts preserve nonclaims and Phase 1 handoff. |
| Veto diagnostics | Missing or failed tests, source route that maps missing divergence to zero, proxy log-accept treated as native divergence, missing review, or unsupported HMC readiness claim. |
| Explanatory diagnostics | Existing trace keys, `divergence_status`, `divergence_count`, log-accept finite/tail summaries, and artifact nonclaims. |
| Not concluded | Zero divergences, convergence, posterior correctness, HMC readiness, sampler superiority, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness. |
| Artifact | Phase 0 result plus refreshed Phase 1 subplan. |

## Forbidden Claims And Actions

- Do not claim zero divergences if native divergence status is
  `not_exposed_by_kernel` or `unavailable`.
- Do not treat log-accept threshold counts as native divergence telemetry.
- Do not run new HMC validation chains in Phase 0.
- Do not change source behavior unless the audit finds a blocker and a repair
  subplan is written.
- Do not use Claude as execution authority.

## Exact Next-Phase Handoff Conditions

Advance to Phase 1 only if:

- governance review is `AGREE` or a documented weaker substitute review is
  accepted by Codex;
- required tests pass;
- Phase 0 result states native-divergence semantics explicitly;
- `screen_hmc_diagnostics` cannot use log-accept threshold counts as native
  zero-divergence evidence when native divergence telemetry is absent;
- Phase 1 subplan is refreshed with the actual Phase 0 finding;
- no unsupported scientific or readiness claim is present.

## Stop Conditions

Stop and write a blocker result if:

- tests fail and the fix is not trivial within the Phase 0 scope;
- current source/artifacts conflate missing divergence with zero divergence;
- Claude and Codex review do not converge after five rounds for the same
  material blocker;
- continuing would require package install, model-file edit, default-policy
  change, destructive git action, or GPU runtime.
