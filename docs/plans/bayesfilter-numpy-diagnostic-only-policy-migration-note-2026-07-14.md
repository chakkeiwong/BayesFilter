# BayesFilter NumPy Diagnostic-Only Policy Migration Note

Date: 2026-07-14

## Decision

Owner direction now limits NumPy to explicitly diagnostic or independent-
reference code. TensorFlow/TensorFlow Probability are required for BayesFilter
runtime, algorithms, training, data pipelines, inference, tuning, selection and
admission logic, artifact construction, and executable benchmark kernels.

Tests, comparison fixtures, independent reference solutions, closed-form or
finite-difference checks, and post-run diagnostic inspection may use NumPy when
their non-runtime role is explicit. TensorFlow tensor materialization at a
host-side assertion, diagnostic, or artifact boundary is not permission to do
numerical work in NumPy.

## Immediate Migration

- Replaced the permissive backend wording in `AGENTS.md` and `CLAUDE.md`.
- Removed the direct NumPy dependency from the active target-specific LGSSM
  NeuTra screen, paired-MCSE calculation, held-out aggregation, and JSON helper.
- Added a focused regression preventing NumPy from returning to that protocol
  module.

The replacement statistics use `math.fsum` and the same sample-variance MCSE
definition as the prior implementation: sample variance with denominator
`n - 1`, divided by `n`, followed by the square root.

## Legacy Status

The repository contains pre-existing NumPy imports. Some are eligible tests,
references, fixtures, or diagnostics; others are legacy migration debt. This
policy change does not falsely certify those modules as compliant and does not
authorize their use as precedent. A touched non-diagnostic execution path must
remove its NumPy dependency before promotion.

In particular, `bayesfilter/testing/lgssm_neutra_serious_validation_tf.py`
still uses NumPy for portions of serious HMC sample collection, serialization,
admission statistics, and comparator calculations. Those uses affect runtime
decisions and are not diagnostic-only. The graph-native migration recorded in
`bayesfilter-lgssm-neutra-graph-native-training-migration-result-2026-07-14.md`
isolates the target-specific training command from that module, so training and
the Phase 4 screen may proceed through the strict TensorFlow-only harness.
Delegated Phase 5/6 entry points must not execute until the parent admission
path is migrated to TensorFlow/TFP and focused parity checks pass. Diagnostic
probe/parity code in the parent module may retain NumPy only after its role is
made explicit and isolated from runtime imports.

## Nonclaims

- This note is not a repository-wide NumPy-removal completion certificate.
- It does not reclassify legacy runtime modules as diagnostic.
- It does not change the current NeuTra target, training objective, recipes,
  seeds, compute budget, or downstream HMC evidence contract.
