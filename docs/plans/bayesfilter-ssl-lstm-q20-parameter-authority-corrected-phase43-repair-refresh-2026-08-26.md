# Phase 43 Repair and Refresh

Date: 2026-08-26  
Source result: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase43-result-2026-08-26.md`  
Branch: `bank_a_isolated_outlier_descriptive`  
Next version: `v2.6-larger-n-support-diagnostic`

Phase 43 passed its hard boundary but did not repair the whitening veto. Bank
A was the only clear N=256 outlier across the four frozen arms; banks B and C
were descriptively closer to the old validation comparator. This supports, but
does not prove, a finite-bank/support-draw explanation. It does not justify
changing the target, objective, architecture, or whitening criterion.

## Repair decision

The next smallest discriminating artifact is one independent N=512 theta bank,
evaluated after the exact v2.4 frozen trainer state. Banks A, B, and C remain
untouched contextual audits. The N=512 bank is never used for training,
checkpoint selection, or tuning. The target remains `theta in R^4`, with the
60D UKF state internal only.

The pilot's calibration-only cloud is increased from 64 to 128 rows as a
particle-count-scaled campaign hypothesis. This change is recorded and is not
a promoted default; the claim bank itself is 512 rows. All target/proposal
schedule semantics remain unchanged, so the expected M0/C0 protocol hashes
remain the frozen Phase 28 values, while the particle count and calibration
count are explicitly checked in the new receipt.

## Required gates

1. N=512 pilot has finite theta/status values, exact target signature, measure
   `theta_R4`, and the frozen M0/C0 protocol hashes.
2. Pilot and tensor hashes are distinct from the authority and N=256 banks.
3. One trainer per arm consumes only the old 232-row root-group training split.
4. Each reconstructed state hash equals the v2.4 reference hash.
5. N=512 target/status and transport diagnostics are finite.
6. The report treats all residual and support differences as descriptive and
   does not promote whitening or an objective change.

## Stop and continuation rules

Any hard gate failure gets a fresh output root and a repair classification.
N=512 residuals that remain poor are evidence against an isolated finite-draw
explanation, not an automatic proof of objective failure. Only after a passing
support envelope may a separate objective/capacity plan be written. A repeated
unrepaired infrastructure failure, unavailable target/support, or exhausted
campaign budget is a true continuation veto.
