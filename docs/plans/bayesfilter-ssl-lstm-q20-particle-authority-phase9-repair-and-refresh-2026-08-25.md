# Phase 9 Repair and Refresh Note

Status: `REFRESHED_PHASE10_TUNING_LADDER`

Classify any failure as GPU launch/policy, input hash/metadata, split/weight
alignment, batch-native training, target/status, transport parity, or
candidate/tuning quality. Preserve every unique output root. A poor whitening
diagnostic is explanatory only and must not be converted into a posterior or
HMC veto.

After the run, record the device/memory policy, mode axis, split, hashes,
parity, audit status, training traces, decision table, and inference-status
table. Refresh the next subplan toward either a component-specific tuning
ladder or a documented support/mode limitation. HMC remains outside this
campaign.

## Executed disposition

The corrected run passed all declared hard gates. Compact was descriptively
selected (`20.2090` versus `21.3424` validation loss), but its latent
`max|mean|=3.5549` and covariance off-diagonal maximum `2.0046` remain far
from an IID Gaussian diagnostic. This is a tuning/representation repair
trigger, not a continuation veto. Refresh Phase 10 with a longer,
target-specific ladder and preserve the same measure/audit input.
