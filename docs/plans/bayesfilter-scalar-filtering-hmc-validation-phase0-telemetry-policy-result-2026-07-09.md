# Phase 0 Result: Governance And Native-Divergence Telemetry Policy

Date: 2026-07-09
Status: `PASSED_AFTER_REPAIR_WITH_CODEX_SUBSTITUTE_REVIEW`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 0 passes after repairing proxy-divergence screening | Passed: focused tests and `git diff --check` passed; `screen_hmc_diagnostics` no longer uses log-accept threshold counts as native zero-divergence evidence | Initial substitute review found a material blocker; repaired and retested | Claude review was blocked by external-transfer policy, so review status is a weaker Codex substitute review | Refresh/review Phase 1 CPU-hidden short-chain subplan with native divergence unavailability preserved as not-zero | No HMC readiness, convergence, posterior correctness, zero divergences, GPU/XLA readiness, default readiness, sampler superiority, or Zhao-Cui source faithfulness |

## Review Record

Claude review gate was attempted with:

```bash
bash /home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh --cwd /home/ubuntu/python/BayesFilter --review-name bayesfilter-scalar-filtering-hmc-validation-phase0-governance --bundle /home/ubuntu/python/BayesFilter/docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase0-governance-review-bundle-2026-07-09.md --model opus --effort max --probe-timeout 90 --timeout-seconds 180 --max-retries 1 --allow-bounded-fallback
```

The escalation reviewer rejected the external Claude call because it would send
private repository planning context to an external service.  No workaround was
used.  A fresh Codex substitute review was used instead and returned
`VERDICT: REVISE`.

The material finding was correct: before repair, `screen_hmc_diagnostics`
allowed log-accept threshold counts to drive the `zero_divergences` check when
native divergence telemetry was absent.  That path was wrong relative to the
Phase 0 target because log-accept threshold counts are not native divergence
telemetry.

## Repair

Patched:

- `bayesfilter/inference/hmc_diagnostics.py`
- `tests/test_common_inference_runtime_contracts.py`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-subplan-2026-07-09.md`

Behavior after repair:

- If native `divergences` is absent, `zero_divergences` is unavailable and
  false.
- Log-accept finite/nonfinite status remains separately screened.
- Log-accept threshold counts cannot establish native zero divergences.

Regression test added:

- `tests/test_common_inference_runtime_contracts.py::test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence`

## Checks

| Check | Status |
| --- | --- |
| `git diff --check` | Passed |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_common_inference_runtime_contracts.py::test_hmc_screen_does_not_use_log_accept_threshold_as_native_divergence tests/test_common_inference_runtime_contracts.py::test_hmc_diagnostics_distinguish_unavailable_from_zero_divergences tests/test_common_inference_runtime_contracts.py::test_hmc_screen_keeps_unavailable_diagnostics_from_passing_as_zero tests/test_common_inference_runtime_contracts.py::test_hmc_screen_classifies_divergence_and_nonfinite_log_accept_as_hard_veto` | Passed: `4 passed, 2 warnings` |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_nonlinear_ssm_phase4_full_chain_hmc.py::test_phase4_tiny_full_chain_hmc_jit_returns_finite_samples_and_metadata` | Passed: `1 passed, 47 warnings` |
| Earlier Phase 0 check `tests/test_scalar_ssl_lstm_filtering_hmc_replicated_diagnostic.py::test_aggregate_seed_rows_is_descriptive_only` | Passed: `1 passed` |

## Telemetry Policy Finding

Current source preserves the required distinction:

- `bayesfilter/inference/hmc.py` reports `native_divergence_status` and
  `divergence_status` as `not_exposed_by_kernel` when standard TFP HMC trace
  lacks a native boolean divergence field.
- `bayesfilter/inference/hmc_diagnostics.py` now refuses to treat log-accept
  threshold counts as native zero-divergence evidence.
- Existing scalar artifacts that say `not_exposed_by_kernel` remain not-zero
  divergence evidence, not zero-divergence evidence.

## Phase 1 Handoff

Phase 1 may proceed only under this rule:

- If native divergence status is `available`, positive native divergence is a
  hard veto and zero native divergence may be recorded.
- If native divergence status is `not_exposed_by_kernel`, `unavailable`, or
  `not_collected`, Phase 1 may still run a non-promoting finite/acceptance
  screen, but must not claim zero divergences or use `zero_divergences` as a
  pass criterion.

The Phase 1 subplan must be refreshed and reviewed with this exact rule before
any new HMC runtime.

## Final Nonclaims

- No HMC readiness.
- No HMC convergence.
- No posterior correctness.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
