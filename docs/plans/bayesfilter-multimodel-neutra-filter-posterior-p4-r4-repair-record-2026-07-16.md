# P4 R4 Repair Record

Date: 2026-07-16

## Attempt 01: PP-UKF Pre-Sampling Binding Failure

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/neutra-confirmation/attempt-01`

The attempt failed before transported-target compilation, kernel probes,
warm-up, or retained sampling. No scientific output was generated.

Classification: `HARNESS_ARTIFACT_REPRESENTATION_MISMATCH`.

The frozen manifest stores the training-state digest as
`sha256:599b586c31a4fb9e56a521be92895b539bc9fa0470e215cc2b46cb8dcdb51ced`,
while the admitted training result stores the identical digest without the
`sha256:` prefix. The harness compared those representations literally. The
loaded semantic `transport_hash` and computed `artifact_signature` agreed with
the admitted result; there was no transport, target, tensor, or topology drift.

Repair: normalize only an optional literal `sha256:` prefix after validating an
exact lowercase 64-hex digest. Keep exact comparisons for artifact signature
and transport hash. Add positive bare/prefixed and invalid-input tests.

The scientific target, transport, comparator, kernel grid, seeds, promotion
criteria, vetoes, hardware class, and total budget remain unchanged. Attempt 02
is a contract-preserving retry in a fresh output root.
