# NeuTra Control Repair Reset Memo (2026-08-15)

## Current State

The Gaussian learning-rate and banana target-specific repair campaign is
complete. Artifacts are under
`docs/plans/artifacts/neutra-control-repair-2026-08-15/`; its SHA-256 manifest
verifies and no campaign process remains.

Run provenance: commit `3030d86df9cb00346df82c7c19f015c09c7c6e1f`, TensorFlow
2.20.0, GPU 0, float64, XLA enabled, TF32 disabled, batch size 4,096, 3,000
updates per cell, and memory growth enabled before logical-device
initialization. Wall time was `518.62 s`, below the `3600 s` cap.

## Scientific Decision

The previous control failures were repaired for the declared targets:

- Gaussian: independent exact-law nomination selected cold `LR=1e-3`; both
  confirmation seeds passed.
- Banana: root-preserving permutation with `(32,32)`, scale `0.02`, and
  `LR=5e-4` passed both screening seeds and both confirmation seeds.

These are target-specific viable candidates. They are not promoted defaults,
not statistically ranked winners, and not evidence for HMC or SSL-LSTM.

## Next Required Work

1. Replicate Gaussian cold `LR=1e-3` and banana root-preserving `LR=5e-4` on a
   fresh multi-seed campaign with the same untouched exact-law gate.
2. If replication remains viable, write a separate downstream/HMC plan for
   each frozen transport. That plan must use the repository sequential HMC
   controller and cannot infer posterior validity from proposal moments alone.
3. Do not transfer root-preserving ordering, Gaussian LR, or the repair
   procedure to SSL-LSTM without an SSL-LSTM-specific adapter/group-ownership
   review and a new target-specific plan.

## Evidence Classes

Exact-law screens are hard viability gates. Loss, ESS, ratio SD, runtime,
gradient norms, clipping, and maximum standardized discrepancy are descriptive
or explanatory. Two-seed confirmation is sufficient for this bounded control
repair decision but insufficient for superiority, universal defaults, or
publication-grade ranking.
