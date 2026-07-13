# Phase 0 Subplan: Contract And Dtype Inventory

Date: 2026-07-09

## Phase Objective

Lock the dtype-polymorphism and batched-score governance contract, inventory
current hard-coded dtype sites, and review the execution plan before any source
edits.

## Entry Conditions Inherited From Previous Phase

- User requested a governed master program and visible gated execution plan.
- Current worktree contains unrelated dirty/untracked files; this phase must
  only write Kalman QR dtype/batched-score planning and review artifacts.
- Existing QR Kalman parameter-count benchmark artifacts show GPU/XLA runs used
  `tf.float64` tensors while TF32 was globally enabled.
- No source edits for dtype cleanup or batched score have been made in this
  program.

## Required Artifacts

- Master program:
  `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-master-program-2026-07-09.md`
- Visible runbook:
  `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-gated-execution-runbook-2026-07-09.md`
- Ledger:
  `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-execution-ledger-2026-07-09.md`
- Stop handoff:
  `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-stop-handoff-2026-07-09.md`
- Review bundle:
  `docs/reviews/bayesfilter-kalman-qr-dtype-batched-score-governance-review-bundle-2026-07-09.md`
- Phase 0 result:
  `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase0-contract-inventory-result-2026-07-09.md`
- Refreshed Phase 1 subplan:
  `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase1-dtype-infrastructure-subplan-2026-07-09.md`

## Required Checks, Tests, And Reviews

Run local checks:

```bash
git diff --check -- docs/plans/bayesfilter-kalman-qr-dtype-batched-score-* docs/reviews/bayesfilter-kalman-qr-dtype-batched-score-*
rg -n "tf\\.float64|tf\\.float32|DTYPE = tf\\.float64|tf\\.convert_to_tensor\\([^\\n]*dtype=tf\\.float64" bayesfilter/linear/kalman_qr_tf.py bayesfilter/linear/kalman_qr_derivatives_tf.py bayesfilter/linear/qr_factor_tf.py bayesfilter/linear/experimental_batched_kalman_tf.py scripts/benchmark_kalman_qr_parameter_count_scaling.py tests/test_linear_qr_compact_loglik_tf.py
```

Review:

- Claude Opus/max read-only review gate only if separately approved for
  external disclosure.
- The Phase 0 Claude gate attempt was rejected by the approval reviewer as
  external disclosure risk, so this phase uses a fresh Codex substitute review
  and records weaker review status.
- If Claude is later approved and probe succeeds but material review fails,
  reduce the bundle and retry.  If Claude approval is rejected, probe fails, or
  transport fails, use a fresh Codex substitute review and record weaker review
  status.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the master program specific enough to remove hard-coded dtype behavior before building batched analytical score? |
| Baseline/comparator | Current QR files and benchmark harness with hard-coded `tf.float64`; existing FP64 benchmark artifacts. |
| Primary criterion | Required docs exist, local text checks pass, dtype inventory records current hard-coded sites, and the documented Codex substitute review or separately approved Claude review returns `AGREE`. |
| Veto diagnostics | Missing artifact, source edit before inventory, review `REVISE` not repaired, unapproved GPU/runtime command, or unsupported FP32/batched-score claim. |
| Explanatory diagnostics | Count and locations of hard-coded dtype sites; unrelated dirty worktree inventory. |
| Not concluded | No dtype cleanup is complete; no FP32 support exists; no batched analytical score exists; no runtime claim is made. |
| Artifact | Phase 0 result plus refreshed Phase 1 subplan. |

## Forbidden Claims And Actions

- Do not edit QR source, tests, or benchmark harness in Phase 0.
- Do not run GPU commands in Phase 0.
- Do not claim FP32 support from fixture dtype alone.
- Do not claim batch-native analytical score exists.
- Do not stage, revert, or modify unrelated dirty files.
- Do not use Claude as execution authority.

## Exact Next-Phase Handoff Conditions

Advance to Phase 1 only if:

- local text checks pass;
- Codex substitute review converges, or separately approved Claude review
  converges;
- Phase 0 result records dtype inventory findings;
- Phase 1 subplan is refreshed with actual inventory findings;
- no unsupported FP32, GPU, speed, HMC, posterior, or default-readiness claim is
  present.

## Stop Conditions

Stop and write a blocker result if:

- Codex substitute review, or separately approved Claude review, does not
  converge after five rounds for the same material blocker;
- local checks expose a plan artifact flaw that cannot be repaired in scope;
- continuing would require package installation, model-file edit, destructive
  action, GPU runtime, or default-policy change;
- unrelated dirty worktree changes make the intended write set ambiguous.
