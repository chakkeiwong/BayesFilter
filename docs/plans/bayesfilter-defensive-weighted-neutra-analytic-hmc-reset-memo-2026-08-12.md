# Defensive weighted NeuTra analytic HMC reset memo (2026-08-12)

## Current state

The analytic HMC plan has been executed through a provenance-valid terminal run
and corrected adjudication.

- Frozen transport: width-128, six-stage weighted-NeuTra confirmation replication
  1, selected update 7,000.
- Exact target: normalized unequal-weight two-mode Gaussian mixture in dimension 4.
- Selected fixed kernel: identity mass in `z`, `L=20`,
  `epsilon=0.14091138276334744`, GPU/XLA float64.
- Sequential controller: 2,000 warm-up and 3,000 retained transitions per chain,
  four chains; warm-up excluded.
- Sequential gates passed: max R-hat `1.00551`, min bulk ESS `6,948`, min tail
  ESS `982.8`, no hard veto.
- Analytic responsibility mass: `0.18417`, 99% batch-means interval
  `[0.16566,0.20267]`, containing truth `0.2`.
- Marginal moments: 3/4 mean and 15/16 covariance intervals contain truth. These
  are explanatory diagnostics, not a joint veto.
- Correct terminal status: statistically compatible under declared marginal
  diagnostics, with strong nonclaims.

Canonical result:

`docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-result-2026-08-12.md`

Corrected adjudication:

`docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5-adjudication-v2/adjudication.json`

## Important implementation repairs

- `WeightedDenseIAFTransport` now exposes exact explicit forward-Jacobian
  pullbacks and log-Jacobian scores for serious fixed-transport HMC.
- The frozen weighted checkpoint loader verifies the file hash, semantic state
  hash, protocol, variable shapes, and restored tensor hash.
- The reviewed custom-gradient wrapper returns zero gradients for captured frozen
  variables; this is required by TensorFlow custom-gradient semantics.
- Fixed-transport tuning has explicit `verification_coordinate_system` and an
  explicit persisted `initial_state_bank` consumed by all arms.
- Marginal analytic moment intervals are no longer silently conjoined into an
  uncalibrated joint rejection.

## Invalid or diagnostic-only artifacts

- canary-v1: implementation failure before transitions.
- run-v1: tuner configuration failure before tuning.
- run-v2 and run-v3: valid negative tuning diagnostics at 1,000 verification
  draws, but not terminal runs.
- run-v4: numerically completed but launch-invalid for scientific interpretation
  because its tuning artifact falsely reported all-zero starts.
- run-v5 `result.json`: immutable source artifact whose original binary decision
  is overly stringent; use adjudication-v1 for interpretation.

Do not delete these artifacts; they document the repair chain.

## Next justified action

Replicate the exact frozen run-v5 kernel with independent HMC root seeds while
keeping transport, target, tuning artifact, warm-up/retained policy, and analytic
diagnostics fixed. Predeclare how independent replications are summarized; do not
select seeds by posterior outcome. If independent HMC replications remain
compatible, repeat on fresh frozen weighted-NeuTra transport replications chosen
by a neutral rule.

Do not jump to SSL-LSTM or claim general multimodal correctness from this result.
