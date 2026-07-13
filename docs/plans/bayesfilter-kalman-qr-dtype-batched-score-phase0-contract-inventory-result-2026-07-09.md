# Phase 0 Result: Contract And Dtype Inventory

Date: 2026-07-09

## Status

`PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

## Phase Decision

Phase 0 completed local document hygiene and dtype inventory checks.  The first
Codex substitute review round found repairable wording issues after the Claude
review gate was rejected by approval policy as external disclosure risk.  Those
issues were patched, and Codex substitute review round 2 returned `AGREE`.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Is the master program specific enough to remove hard-coded dtype behavior before building batched analytical score? |
| Baseline/comparator | Current QR files and benchmark harness with hard-coded `tf.float64`; existing FP64 benchmark artifacts. |
| Primary criterion status | `PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`: required governance docs exist, local text checks passed, dtype inventory was recorded, and Codex substitute review round 2 returned `AGREE`. |
| Veto diagnostic status | No source edits, GPU commands, or unsupported FP32/batched-score claims occurred in Phase 0. |
| Main uncertainty | Codex substitute review is weaker than the requested Claude review because external disclosure was not approved. |
| Next justified action | Launch Phase 1 dtype infrastructure under CPU-hidden local checks and focused review if shared helper contracts change. |
| What is not concluded | No dtype cleanup is complete; no FP32 support exists; no batched analytical score exists; no runtime claim is made. |

## Local Checks

### `git diff --check`

Command:

```bash
git diff --check -- docs/plans/bayesfilter-kalman-qr-dtype-batched-score-master-program-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-gated-execution-runbook-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-execution-ledger-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-stop-handoff-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase0-contract-inventory-subplan-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase1-dtype-infrastructure-subplan-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase2-qr-value-dtype-subplan-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase3-analytical-score-dtype-subplan-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase4-benchmark-dtype-controls-subplan-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase5-batched-score-contract-subplan-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase6-batched-score-implementation-subplan-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-subplan-2026-07-09.md docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase8-closeout-subplan-2026-07-09.md docs/reviews/bayesfilter-kalman-qr-dtype-batched-score-governance-review-bundle-2026-07-09.md
```

Result: `PASSED`.

### Dtype Inventory

Command:

```bash
rg -n "tf\\.float64|tf\\.float32|DTYPE = tf\\.float64|tf\\.convert_to_tensor\\([^\\n]*dtype=tf\\.float64" bayesfilter/linear/kalman_qr_tf.py bayesfilter/linear/kalman_qr_derivatives_tf.py bayesfilter/linear/qr_factor_tf.py bayesfilter/linear/experimental_batched_kalman_tf.py scripts/benchmark_kalman_qr_parameter_count_scaling.py tests/test_linear_qr_compact_loglik_tf.py
```

Result: `PASSED_AS_INVENTORY`.  The command found hard-coded dtype sites in the
expected files.  The important current findings are:

- `bayesfilter/linear/kalman_qr_tf.py`: value kernels and helpers coerce
  observations/model tensors, jitter, identities, accumulators, masked row
  weights, and diagnostics to `tf.float64`.
- `bayesfilter/linear/kalman_qr_derivatives_tf.py`: analytical score and
  Hessian helpers coerce inputs, derivative tensors, accumulators, jitter,
  identities, and diagnostics to `tf.float64`.
- `bayesfilter/linear/qr_factor_tf.py`: shared QR/Cholesky/factor derivative
  primitives coerce factors, covariance tensors, derivative stacks, jitter,
  and identities to `tf.float64`.
- `bayesfilter/linear/experimental_batched_kalman_tf.py`: experimental
  batch-native value/score path is also FP64-only today.
- `scripts/benchmark_kalman_qr_parameter_count_scaling.py`: benchmark fixture
  has `DTYPE = tf.float64`.
- `tests/test_linear_qr_compact_loglik_tf.py`: tests are FP64 fixtures today
  and cannot by themselves verify FP32 preservation.

## Review Status

Claude read-only review gate was attempted with:

```bash
bash /home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh --cwd /home/ubuntu/python/BayesFilter --review-name kalman-qr-dtype-batched-score-governance --bundle /home/ubuntu/python/BayesFilter/docs/reviews/bayesfilter-kalman-qr-dtype-batched-score-governance-review-bundle-2026-07-09.md --model opus --effort max --probe-timeout 90 --timeout-seconds 180 --max-retries 1 --allow-bounded-fallback
```

The approval reviewer rejected the command as external disclosure risk.  This
is not a Claude timeout and should not be retried or worked around without
explicit user approval of the disclosure risk.

Required review bundle:

- `docs/reviews/bayesfilter-kalman-qr-dtype-batched-score-governance-review-bundle-2026-07-09.md`

Fresh Codex substitute review round 1 found repairable issues:

- runbook still treated Claude as the active material review path;
- Phase 0 result still said Claude review was pending instead of recording
  approval-policy rejection;
- review protocol did not explicitly classify external-disclosure rejection as
  a substitute-review route.

The governance artifacts were patched.  Fresh Codex substitute review round 2
found no material findings and ended with `VERDICT: AGREE`.

Review status: `AGREE_WEAKER_THAN_CLAUDE_REVIEW`.

## Next-Phase Handoff

Phase 1 is authorized under the weaker Codex substitute review path.  Handoff
conditions satisfied:

- local text checks passed;
- dtype inventory findings are recorded;
- Claude approval-policy rejection is recorded without retry/workaround;
- Codex substitute review round 2 returned `AGREE`;
- Phase 1 subplan remains consistent with Phase 0 findings.

## Nonclaims

- This phase did not fix dtype behavior.
- This phase did not implement FP32 support.
- This phase did not implement batched analytical score.
- This phase did not run GPU commands or benchmarks.
- This phase does not support speed, HMC, posterior, default-readiness, or
  production-readiness claims.
