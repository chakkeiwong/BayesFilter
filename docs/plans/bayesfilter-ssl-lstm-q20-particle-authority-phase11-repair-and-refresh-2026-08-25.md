# Phase 11 Repair and Refresh Note

Status: `REFRESHED_PHASE12_CAPACITY_ARM`

Classify failures as GPU policy, input/hash/mode, split/weight, batch-native
training, target/status, parity, optimizer, and candidate quality. A plateau
in latent moments is a repair trigger or evidence against this training arm,
not a whole-program blocker. Preserve the audited input and do not tune on the
audit partition.

After the trace, write a decision table and inference-status table. If a
longer trace improves materially, refresh a frozen untouched claim-screen
proposal; if it plateaus, refresh a support/representation audit and keep
HMC deferred.

## Executed disposition

The compact 300-update trace passed all hard gates and reached latent
`max|mean|=0.6251` with covariance off-diagonal maximum `0.4779`; this is
still not an IID certificate. Refresh Phase 12 with one high-capacity flow arm
on the same audited input before changing the particle measure.
