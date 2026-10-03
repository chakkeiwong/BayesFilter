# Phase 8 C3 diversity-diagnostic repair

Date: 2026-08-31  
Status: `READY_FOR_SKEPTICAL_AUDIT_AND_EXECUTION`

Parent subplan:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-subplan-2026-08-30.md`

## Reason for repair

The C3A runner completed all eight declared training rows and the hard
bridge/checkpoint/reliability screens. Its row records contain beta-one
cross-chart mean distance, but the C3A evidence contract also named covariance
summaries and sign-region occupancy. Those two summaries were not emitted.
This focused repair reconstructs the immutable beta-one checkpoints and fills
that diagnostic gap without retraining, changing a map, or using target draws.

The repair is required before any lineage arm is nominated. It is not a
whitening, mode-discovery, posterior, HMC, or architecture-ranking experiment.

## Evidence contract

| Item | Contract |
|---|---|
| Question | Do the saved pure-continuation and positive-branching beta-one chart pairs have different covariance summaries or declared sign occupancy on fresh disjoint base-Gaussian banks? |
| Target | Frozen q=20 SSL-LSTM proper bridge, target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`; no target evaluation is needed for the summaries, but signatures are rechecked. |
| Inputs | Only the eight immutable beta-one checkpoint JSON files and the completed C3A manifest under `.../c3-lineage-overlap/attempt-01/`. |
| Fresh banks | 256 IID standard-Gaussian latent rows per component, generated with fresh stateless roots `(20260831, 54001)` and `(20260831, 54002)` folded by architecture, arm, root, and component. Banks are disjoint from C3 training/overlap/reliability roots. |
| Primary diagnostic | Per-chart mean, diagonal variance, covariance trace/Frobenius norm, sign fractions for physical coordinate 2 (`>0`, `<0`, `==0`), and pairwise mean/covariance distances. |
| Hard checks | Checkpoint hash/context, target/bridge identity, finite forward values/log determinants, exact bank shape, sign-fraction accounting, and allocator peak below 4 GiB. |
| Interpretation | All differences are descriptive nomination evidence. No ranking, posterior mass, mode count, whitening, or high-dimensional scaling claim is permitted. |
| Artifact | Fresh manifest under `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/diversity-repair-2026-08-31/attempt-01/`. |

## Procedure

1. Verify the C3A manifest status, row hashes, target signature, bridge
   signature, and strict principal-square-root backend.
2. Restore each row's two beta-one checkpoints through the repository checkpoint
   verifier. Reject any stale, mutable, or context-mismatched checkpoint.
3. Generate the fresh component-specific latent banks and evaluate each map in
   TensorFlow float64 on the repository GPU route with memory growth configured
   before device initialization.
4. Compute the summaries with TensorFlow reductions. Materialize values only at
   the JSON artifact boundary; do not import NumPy or evaluate a scalar target
   per row.
5. Recompute row-level cross-chart distances from the fresh banks and compare
   them with the earlier common-bank mean-distance diagnostic descriptively.
6. Write a manifest and a result note. If a hard check fails, preserve the
   failed attempt and repair only the failing artifact/harness boundary.

## Skeptical audit

| Risk | Check and disposition |
|---|---|
| Checkpoint mutation or stale map | Verify the stored checkpoint hash and full expected context before every forward call. |
| Accidental target sampling | The script has no sampler or target-draw path; only checkpoint restoration and map evaluation are allowed. |
| Bank overlap with C3A | Fresh roots and a separate bank identity are bound in the manifest. |
| Sign label misuse | Coordinate and strict inequalities are recorded explicitly; zero is a separate boundary category. |
| Covariance instability | Report raw finite-sample summaries and bank size; treat them as descriptive, not inferential. |
| GPU allocator failure | Require one visible GPU, pre-import memory growth, and the 4 GiB peak cap. |
| Hidden graph-policy violation | Scan the restored runtime route for forbidden row-mapping/pfor tokens and use batch forward calls. |

Audit verdict: `PASS_FOR_BOUNDED_ARTIFACT_ONLY_REPAIR`.

## Budget and stop rules

The repair has a 900 command-wall-second cap and one fresh attempt directory.
It consumes the remaining C3 calibration allocation, not Phase 9 confirmation
budget. Stop on signature mismatch, invalid checkpoint, nonfinite map output,
allocator/memory-growth failure, forbidden route token, or missing required
summary. A successful repair closes the C3 diagnostic gap and permits a
between-phase refresh; it does not authorize HMC.

## Required result interpretation

The result note must include a decision table and inference-status table. It
must state separately whether the C3A hard screen passed, whether the repair
diagnostics were complete, whether any statistical ranking is supported, and
which next subplan is justified. The strongest alternative explanation is
finite-bank variability or chart-scale differences rather than distinct
posterior modes; the repair cannot resolve that question.
