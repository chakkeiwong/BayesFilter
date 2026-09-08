# Phase 12 Repair and Refresh Note

Status: `REFRESHED_PHASE13_AFFINE_ORACLE`

Classify failures as GPU policy, input/hash/mode, split/weight, batch-native
training, target/status, parity, capacity/optimizer, and candidate quality.
Preserve the audited input and all unique output roots. A capacity failure is a
candidate repair trigger, not a whole-program blocker.

Record the compact and high-capacity traces, hard-gate receipts, latent
diagnostics, decision table, and inference-status table. Refresh the next phase
toward support/mode evidence or a reviewed final candidate screen; keep HMC
deferred.

## Executed disposition

The high-capacity arm passed all hard gates but did not improve the moment
residual over compact (`max|mean|=1.2591`, off-diagonal `0.4952`). This is not
a whole-direction blocker. Refresh Phase 13 with an exact weighted affine
whitening oracle on the same audited bank.
