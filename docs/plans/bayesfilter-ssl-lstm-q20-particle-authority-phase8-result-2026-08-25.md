# Phase 8 Result: Raw Measure and Ledger Audit

Status: `PASS_MEASURE_AUDIT_METADATA_BOUND`

The independent CPU-hidden auditor checked the metadata-bound N=300 M0 pilot
at
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401`.
It did not rerun the q=20 target and does not prove SMC-U unbiasedness; it
checks whether the stored finite computation is internally consistent.

## Hard audit evidence

All 13 gates passed:

- canonical protocol hash recomputation and explicit `mode_axis=2` binding;
- every stored tensor digest, shape, finite-value, nonnegative-weight, and
  unit-sum check;
- contiguous stage indices, monotone beta schedule ending at one, and finite
  cumulative mass ledger;
- terminal normalized weights reproduced from the final target/proposal
  increment with maximum residual `4.34e-18`;
- particle-level accepted/proposal count identity; and
- weighted negative-mode fraction reproduced from `theta[:, 2]` with zero
  residual.

The final stored log-mass was `-33.7713`, and the cumulative-ledger residual was
exactly zero. These values describe this finite candidate computation only.

The declared legacy negative control (a corrected acceptance receipt without
the new protocol field) was rejected with a structured `MEASURE_AUDIT_FAIL`
for missing `mode_axis`. The first metadata audit attempt also recorded and
repaired an auditor path-resolution harness failure; the second attempt passed.

## Decision and inference tables

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Admit the metadata-bound bank to a role-limited downstream screen | all finite-measure audit gates pass | no ledger/metadata veto | finite-run support and target law remain unproved | rerun NeuTra with the corrected mode axis and aligned weights | no authority, unbiasedness, mode-discovery, posterior, whitening, HMC, or default claim |

| Evidence class | Status |
|---|---|
| Hard veto screen | passed on metadata-bound bank; legacy negative control rejected as required |
| Statistically supported ranking | none |
| Descriptive-only differences | mass, ESS, mode fraction, acceptance, and latent diagnostics |
| Default-readiness | not ready |
| Next evidence needed | corrected mode-stratified NeuTra screen and broader target/mode evidence |

## Red-team note

The audit can pass even if the proposal misses a posterior mode: it validates
the recorded finite measure, not its relationship to the intended q=20 law.
The strongest remaining alternative is therefore support/mode bias in the
seeded chart and local mutation. A failed corrected NeuTra screen would be a
downstream candidate failure, not a proof that the finite ledger is wrong.

The audit artifact is
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase8-attempt2-metadata-bank2401`.
