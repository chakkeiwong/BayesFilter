# Phase 15 Repair and Refresh Note

Status: `PASS_HARD_GATES_ROLE_LIMITED_PAIRED_DESCRIPTIVE_REFRESHED_PHASE16`

This phase compares identity and affine coordinates on the same audited bank.
The repair boundary preserves target, rows, weights, mode axis, partitions,
profile, hardware class, and campaign cap.

## Failure classification

| Failure | Classification | Repair |
|---|---|---|
| missing or stale bank/protocol hash | harness/input | stop before training; refresh from Phase 8 receipt |
| GPU memory/XLA/batch receipt failure | infrastructure/harness | repair launch ordering or runner; rerun fresh root |
| target/status/parity failure | implementation | isolate the failing arm and preserve the other arm |
| full-bank metric absent or non-finite | diagnostic harness | repair metric path; do not interpret validation moments |
| large residual in a valid arm | candidate/evidence | retain role-limited result; refresh Phase 16, no HMC |

## Refresh rule

After the completed paired receipt, record exact hashes, wall time, full-bank
metrics, decision and inference-status tables, and the strongest alternative
explanation. Refresh Phase 16 with the measured comparison. A single paired
seed cannot rank stochastic arms; it can only decide whether a larger paired
ladder is justified.

The receipt completed without a harness or numerical veto. It found a strong
arm-by-preconditioner interaction: affine helps `compact_low_lr`, while the
identity route is descriptively better for `compact` and `wider_mid_lr`. The
next phase therefore uses a second paired seed with unchanged controls. It is
not a license to select the lowest-loss or lowest-residual arm as a default.
