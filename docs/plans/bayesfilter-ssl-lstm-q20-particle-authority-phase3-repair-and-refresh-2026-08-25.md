# Phase 3 Repair and Refresh Note

Status: `PASS_GATE_ROLE_LIMITED`

Repair one arm at a time. A failed M1 or M2 remains an auxiliary-candidate
finding and does not invalidate M0. A failed M3 density identity blocks only
its exact-PF-PF label; it may remain a diagnostic proposal if the alternate
quantity is explicitly defined. A failed M4 is an approximate-arm result.

Before Phase 4, refresh with:

- arm-specific role labels and source anchors;
- frozen M0/tuning scope hashes;
- target-level versus proxy diagnostics;
- candidate failures and repairs separately from continuation vetoes;
- which particle banks are eligible for NeuTra training and their provenance;
- a training protocol with target-specific batch, architecture, optimizer,
  validation/audit partitions, seeds, and GPU/XLA receipts.

## Actual result

M1's affine finite-cloud transform matched weighted moments with maximum
covariance residual `2.8041e-10` and all 100 transformed target rows valid. M2's
9-point selected-moment rule matched within its declared `1e-8` ridge and all 9
rows were valid. These are auxiliary contracts only. M3's affine recovery
residual was `1.78e-15`, but the canonical LEDH-PFPF route was not executed and
is therefore not admitted. M4 remains descriptive by definition. Phase 4 is
refreshed to train only on the M0-authoritative candidate bank.

## Mutation-bank revalidation

After Phase 4, the repaired N=100 mutation bank was propagated without
replacing the frozen identity branch. The revalidation artifact is
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase3-mutation-revalidation-attempt1/`.
M1 and M2 again passed their finite moment/status contracts; M3 again passed
only its affine recovery scaffold; and M4 remained approximate. This confirms
that the modular mechanics are not dependent on the identity bank, but it does
not promote the mutation bank or any arm.
