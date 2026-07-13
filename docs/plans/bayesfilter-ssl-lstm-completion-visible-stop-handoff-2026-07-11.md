# BayesFilter SSL-LSTM Completion Visible Stop Handoff

Date: 2026-07-13

Status: `NOT_STOPPED_A3_HARNESS_REPAIR_BEFORE_EVIDENCE`

## Current A3 Checkpoint

This checkpoint supersedes the older A2-current-state wording below while
preserving it as history.

- A2 closure and the terminal trace audit passed; A3 implementation is active
  inside the frozen A3 boundary.
- The independent scalar LGSSM oracle is frozen at source SHA-256
  `74889d699e3575ee163c64d9a67325f0376e161106e9b36fb6b61453c3a5eb43`.
- Predictive statistics are frozen at source SHA-256
  `99ddaa1dcb15e9f3ec7a5a18f96ebd0f656848c40ea76c896b387cace294bc16`
  after repairing forged-result admission, caller-labeled quadratic-MMD
  inference, dynamic scale-floor admission, and roundoff-degenerate MMD
  uncertainty.
- The combined focused suite passed `65/65`; the exact implementation received
  bounded Codex-substitute `VERDICT: AGREE`, explicitly weaker than Claude.
- A3 is not stopped. The active repair target is the generator/verifier pair.
  No evidentiary CPU/GPU command may run until those files consume persisted
  materialized banks and pass an independent bounded harness review.
- No HMC, NeuTra, sampler comparison, A4 calibration, scientific claim,
  package/network action, Git publication, model-file edit, or concurrent
  HMC/Kalman-lane edit is authorized by this checkpoint.

## Current State

- Supervisor/executor: Codex in the current conversation.
- Lane: SSL-LSTM completion only. The concurrent HMC/Kalman lane was not
  edited, staged, committed, reset, restored, or cleaned by this lane.
- Reviewer: Claude is policy-unavailable. Bounded native Codex substitute
  reviews are explicitly weaker than Claude review.
- Current phase: A2 post-result closure; the A3 forecast-oracle/statistics
  subplan is drafted and reviewed.
- Current `HEAD`: `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163`, equal to the A0 anchor; no
  intervening commit path exists.
- A2 focused tests passed `87/87` after finite-admission/provenance and
  terminal-trace parser repair.
- A2 CPU reference status is `CPU_REFERENCE_CONTRACT_PASSED`; artifact SHA-256
  is `8bd1ed508e90674521774f73332e73e2a2f198a057879448dcddc0e30ed35df2`.
- A2 trusted GPU status is `GPU_XLA_CANARY_PASSED`; artifact SHA-256 is
  `0294b06527620336e970bf6a57fd2e0f1a8466502bf47f9595a533d10ca23521`.
- A2 result status is `PASSED_FOR_A3_PLANNING_ONLY`; refreshed exact result
  SHA-256 `dd7ecff91e6549b5abd09d2be4edd88d7fd97ce00b68c357aafbe6b3b6cc0f6f`
  received bounded `VERDICT: AGREE`.
- The prior final checkpoint, post-result ledger, and closure remain stale
  after the trace-parser repair and are not current evidence.
- A3 subplan review converged after repairing dependent-MMD estimand,
  null-degeneracy, joint-alpha, and tiny-chain inference issues.
- Refreshed A3 subplan SHA-256
  `67ee503a15f5e7a81ca2a37e52cc6b60264c1cff89ff5cff1a9fddd3187161c4`
  received bounded `VERDICT: AGREE`.
- A3 implementation is not yet authorized until the A2 post-result closure and
  terminal trace audit complete.

## Resolved Blockers

The A2 failure was a local finite-admission and provenance defect: nonfinite
draws/banks/outputs were not comprehensively rejected, seed metadata could be
misread as cross-backend replay authority, and the A1 adapter signature field
was ambiguous. The repaired implementation rejects invalid values, makes
materialized tensor hashes authoritative, and records the typed A1 adapter
signature. Fresh focused, CPU, GPU, and review gates passed.

The later closure failure was simpler: an unanchored `link(` matcher treated
read-only `readlink(...)` as a mutation. The repaired verifier parses complete
syscalls, rejects ambiguity, and admits only resolved write-open destinations
under the A2 artifact or `/tmp/bayesfilter-a2-*` roots. Its focused 18-test
trace-parser suite and bounded review passed. This was a verifier bug, not a
forecast, CPU/GPU, XLA, target, or model failure.

The user's explicit "fix that and continue" direction authorizes the narrow
`strace -f -qq -yy -s 65535 -e trace=%file` repair for newly regenerated A2
closure traces. It does not broaden the A2 model, source, scientific, Git, or
concurrent-lane boundary.

## Preserved Evidence

| Artifact | SHA-256 |
| --- | --- |
| Protected A1 target | `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667` |
| A2 production forecast module | `0dad54c239de11f105f541527447d167114073ab046c796a813b5c1e867452ed` |
| A2 focused tests | `1812b338ff90633d2fa627642af8ba65425bdaf1c11211f8944d7207ecbded2c` |
| A2 verifier | `d0195063a1686a5332b6788bd1171ffc998370bd3578ceeb64edea240a2511ee` |
| A2 implementation/trace review | `1210e2fcced29448cbcdba7a4ce1dcee93326e3f317e27ec65d45c30364f23fb` |
| A2 CPU artifact | `8bd1ed508e90674521774f73332e73e2a2f198a057879448dcddc0e30ed35df2` |
| A2 GPU artifact | `0294b06527620336e970bf6a57fd2e0f1a8466502bf47f9595a533d10ca23521` |
| A2 result | `dd7ecff91e6549b5abd09d2be4edd88d7fd97ce00b68c357aafbe6b3b6cc0f6f` |
| A2 final checkpoint | Pending hardened traced regeneration |

## Continuation Rule

Generate the A2 post-result write ledger and closure, fresh-process verify the
closure, then audit the exact closure-verification trace. Record
`A2_TERMINAL_WRITE_TRACE_AUDIT_PASSED` and its trace SHA-256. Only then freeze
the A3 boundary and begin the reviewed A3 engineering sequence.

## Nonclaims

The accepted A2 result establishes only bounded terminal-state and forecast API
engineering. The A3 review establishes only plan consistency. Neither is
posterior correctness, HMC/NeuTra readiness, predictive equivalence,
calibration, model adequacy, performance, default/product/release readiness,
sampler ranking, or scientific validity.
