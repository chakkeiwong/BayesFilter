# Codex Substitute Re-Review: Phase 9 Gate B Fixed-SIR Repair, Iteration 2

Date: 2026-07-11

## Scope And Limitation

Fresh local read-only re-review after the iteration-1 repair-authorization
provenance blocker was fixed. The review inspected the attempt-1 terminal
artifacts/logs and their archived hashes, the fixed-SIR covariance repair, the
new XLA/eager parity test, the runner's governance manifest and shard validator,
the frozen exact commands, and the repair result. Claude remains policy-blocked
as external repository disclosure. No GPU/CUDA command ran during this review.

This verdict authorizes only rerunning the two exact fixed-SIR Gate B commands:
score-only first, then FD-only if and only if the score shard completes and
passes its hard provenance/device/finite/memory checks. If both fixed-SIR
commands pass their hard screens, the pre-existing Gate A iteration-2
authorization resumes for the remaining eight nonlinear Gate B commands. If
either fixed-SIR command fails, those commands remain blocked pending diagnosis.
Gate C, Gate D, aggregation, and LGSSM remain blocked.

## Iteration-1 Blocker Resolution

| Blocker | Resolution checked |
| --- | --- |
| Post-repair shards named and hashed only the older Gate A authorization. | `GATE_B_REPAIR_REVIEW_PATH` identifies this exact review. It is included in the governance SHA-256 set, emitted in every run manifest, and required by common shard validation. |
| A relabeled or modified repair authorization could escape focused tests. | Adversarial tests mutate the repair-review path and its governance hash independently; both are rejected. Governance and code hashes remain frozen for process lifetime. |
| Provenance hardening could change runtime settings. | The frozen exact-command JSON remains current. No argv, target, seed, shape, device, transport, precision, memory budget, FD step, or FD tolerance changed. |

## Functional Repair Assessment

The repair is correct relative to the fixed-SIR prepared-input contract.
`_build_actual_sir_tensors` evaluates `process_noise_covariance_fn` before XLA
tracing and tiles that fixed covariance over the batch. The removed graph-time
`_dpf_sir_callbacks()` call recovered the same fixed covariance solely to form
its Cholesky factor. Using `transition_covariance[0]` preserves the expected
`[18,18]` factor for the existing process-noise `einsum` and changes neither
the claimed scalar nor the compact score recurrence.

Attempt 1 is correctly classified. The score-only shard produced finite trusted
GPU/XLA output, while FD-only failed during graph extraction before computing a
finite-difference value. It is unsupported to call that a numerical FD failure.
Both live paths must be rerun because the reachable source hash changed, and the
old score/failed-FD evidence has been archived under distinct paths.

## Verification

- Direct repair tests: `2 passed, 76 deselected, 2 warnings in 13.80s`.
- Harness plus fixed-SIR contracts: `97 passed, 2 warnings in 92.99s`.
- Combined harness/cross-model/shared contract before provenance-only repair:
  `151 passed, 2 warnings in 27.58s`.
- LGSSM/fixed-SIR model-specific shard: `53 passed, 2 warnings in 92.11s`.
- Post-provenance focused governance/fixed-SIR suite:
  `34 passed, 45 deselected, 2 warnings in 15.95s`.
- Final review-bound combined harness/cross-model/shared contract:
  `152 passed, 2 warnings in 35.93s`.
- Syntax, exact-command currentness, and `git diff --check` pass.

CPU-hidden XLA checks are engineering evidence only. They do not replace the
trusted GPU retry or establish Gate B, full-row memory, posterior correctness,
HMC readiness, runtime superiority, or scientific validity.

## Residual Risks And Stop Rules

- GPU XLA may expose another device-specific extraction or numerical failure.
- The fixed-SIR score command must run first because its code hash changed. An
  FD process may not consume the archived pre-repair score.
- A failed/nonfinite/wrong-device/wrong-provenance score stops before FD.
- A terminal FD failure or frozen-tolerance mismatch blocks the row and all
  later Gate B commands pending diagnosis.
- A tiny pass remains only Gate B preflight evidence; it cannot admit the row
  or authorize Gate C without the separately required complete Gate B result
  review.

No material blocker remains for the two-command fixed-SIR Gate B retry.

VERDICT: AGREE
