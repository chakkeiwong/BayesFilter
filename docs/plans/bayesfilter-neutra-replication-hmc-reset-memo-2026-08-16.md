# NeuTra Replication And HMC Reset Memo (2026-08-16)

## Current State

The terminal replication/HMC campaign is complete under
`docs/plans/artifacts/neutra-replication-hmc-2026-08-16-r6/`; its artifact
hashes verify and no campaign process remains.

## Scientific Decision

- Gaussian cold `LR=1e-3`: three fresh exact-law training replications passed;
  shared sequential HMC passed warm-up/retained R-hat and ESS gates, and the
  retained draws passed the exact-law screens.
- Banana root-preserving `LR=5e-4`: two fresh seeds passed and one failed a
  coordinate-mean screen. HMC was correctly blocked by the replication veto.

The Gaussian candidate is a valid bounded exact-law control for the complete
training-to-sequential-HMC harness. It is not a universal default or evidence
of HMC superiority. Banana remains a training-replication problem.

## Next Required Work

1. Run a banana-only target-specific repair/replication campaign for the
   remaining coordinate-mean failure, preserving the same three-seed gate.
2. Do not run banana HMC until all three fresh training seeds pass.
3. Do not transfer either control configuration to SSL-LSTM without a new
   SSL-LSTM adapter/group-ownership plan and target-specific training evidence.

## Evidence Classes

Exact-law replication and post-HMC screens are hard gates. HMC acceptance,
loss, ESS, runtime, and standardized discrepancy are descriptive or
explanatory. Native divergence status was unavailable from the TFP HMC kernel;
this is explicitly recorded and is not treated as zero divergences.
