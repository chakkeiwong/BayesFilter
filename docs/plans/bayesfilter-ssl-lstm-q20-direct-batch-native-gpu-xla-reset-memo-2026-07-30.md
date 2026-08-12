# q=20 Direct Batch-Native GPU/XLA Reset Memo

Date: 2026-07-30
Status: `R1_CLOSED_REPAIR_PREFLIGHT_REQUIRED`

## Trusted State

- Result:
  `docs/plans/bayesfilter-ssl-lstm-q20-direct-batch-native-gpu-xla-training-result-2026-07-30.md`.
- Plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-direct-batch-native-gpu-xla-training-plan-2026-07-30.md`.
- Artifact root:
  `docs/plans/artifacts/ssl-lstm-q20-direct-batch-native-gpu-xla-training-2026-07-30/r1/`.
- Full relevant repaired-lane regression suite: `29 passed`.
- GPU mechanics passed for the historical pre-repair GPU-realized target.
- r1 stopped under-budgeted and target-identity-invalid. Do not resume it, issue
  a tuning artifact from it, or reinterpret it as claim-bearing evidence.

## Code State

The lane now has:

- a direct status-bearing q=20 batch target;
- a repository-issued binding and trainer proxy that invokes the bound callable
  and poisons invalid status before optimizer update;
- an isolated GPU/XLA runner with cumulative budget accounting;
- explicit CPU placement for static q-complexity fixture/observation creation;
  and
- v2 target identity policy
  `explicit_cpu_device_hardware_invariant_target_identity_v1`.

The repaired CPU q=20 signature is
`2f7e29d32e45dc309533859c994583db94d82e90ed0a5b8318adef5b9f5f476e`.
GPU parity was not checked because the trusted approval reviewer timed out
twice before process launch.

## Required Next Preflight

1. Run a short trusted GPU target-construction diagnostic with memory growth,
   no filter call, and verify target/adapter signatures and the complete v2
   signature payload are byte-identical to CPU construction.
2. Add per-update or every-five-update progress receipts to a separate timing
   diagnostic so compile, validation, and steady-update time can be separated.
3. Write a new r2 plan with a budget derived from those measurements. Do not
   reuse 100 tuning updates, four arms, two 1,000-step finals, or 18,000 seconds
   merely because they appeared in r1.
4. Preserve disjoint tuning/final/audit partitions and the target-specific
   architecture/optimizer/seed/downstream evidence requirements.
5. Do not launch HMC until a new claim-bearing GPU/XLA training protocol passes
   and a separate HMC plan is explicitly authorized.

## Do Not Conclude

Do not conclude that NeuTra failed, `(32,32)` failed, `lr=2e-4` failed, GPU/XLA
mechanics failed, or the repaired target is GPU-verified. The correct r1 verdict
is: direct mechanics worked, the target identity claim was wrong, and the
training campaign was under-budgeted.
