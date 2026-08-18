# V4 Anchored-Orthogonal Ratio Score Result And Reset Memo

Date: 2026-08-14  
Plan: `docs/plans/bayesfilter-sir-anchored-orthogonal-ratio-score-v4-plan-2026-08-14.md`  
Review: `docs/plans/bayesfilter-sir-anchored-orthogonal-ratio-score-v4-plan-review-2026-08-14.md`

## Decision

The full exact Gaussian gate failed. SIR was not run, as required by the
plan. The anchored basis is mathematically valid and the implementation and
execution harness passed its engineering gates, but this run did not produce
an admitted filter-independent score reference for all required cells.

## Exact Run

Artifact: `docs/benchmarks/artifacts/sir_anchored_orthogonal_ratio_score_20260814/exact_full_attempt01/`  
Result: `result.json`  
Manifest: `run_manifest.json`  
Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, GPU `NVIDIA GeForce RTX 5080`, XLA enabled, TF32 disabled, TensorFlow memory growth verified.  
Result status: `FAILED`; `all_reference_cells_admitted=false`; 27 final rows.

## Evidence Contract Status

| Decision item | Status | Evidence |
|---|---|---|
| Hard infrastructure vetoes | Passed | GPU/XLA initialization, memory policy, finite outputs, source and runtime dependency audits, balanced deltas, valid artifacts |
| Exact Gaussian score reference | Failed | 7 of 9 horizon/coordinate cells failed the aggregate exact gate |
| Admitted cells | Descriptive only | `T20_j2` and `T40_j2` passed all current cell gates |
| SIR continuation | Vetoed | Full exact result was not `PASSED` |
| Statistical ranking or method superiority | Not concluded | Three replicates per cell are a bounded diagnostic, not a superiority study |

## Cell Results

The score estimates below are descriptive diagnostics from three final-domain
replicates; they are not promoted references unless the cell status is
`admitted`.

| Cell | Replicate estimates | Mean | SE | Exact Gaussian score | Status |
|---|---:|---:|---:|---:|---|
| `T20_j0` | 16.375, 16.868, 14.968 | 16.071 | 0.569 | 18.900 | failed exact error |
| `T20_j1` | -0.427, -5.811, -5.561 | -3.933 | 1.755 | 0.406 | failed precision |
| `T20_j2` | -11.397, -22.355, -12.885 | -15.546 | 3.432 | -14.198 | admitted |
| `T40_j0` | 7.668, 26.255, 3.309 | 12.410 | 7.036 | 13.727 | failed precision |
| `T40_j1` | -4.134, -8.124, 0.894 | -3.788 | 2.609 | -1.608 | failed precision |
| `T40_j2` | -10.552, -11.209, -20.234 | -13.998 | 3.123 | -14.477 | admitted |
| `T50_j0` | -5.016, 18.156, -3.873 | 3.089 | 7.541 | 5.401 | failed precision |
| `T50_j1` | -5.739, -0.147, -0.898 | -2.261 | 1.752 | -2.279 | failed precision |
| `T50_j2` | -0.238, -12.591, -13.648 | -8.826 | 4.305 | -9.648 | failed precision |

The selected final controls were the anchored linear-quadratic head in every
cell; `l2=1e-5` was selected in eight cells and `l2=0` in `T40_j2`. The
selection was performed on domain 50 and final estimates on disjoint domain
60. The basis diagnostics were `alpha=1.1723166183`, discrete inner product
`0`, derivative anchors `(1,0)`, and condition number `5.80404`.

## Interpretation

This result invalidates neither the simulator nor the classifier identity.
It shows that this V4 candidate, with the frozen perturbation grid, feature
heads, calibration rule, sample budget, and admission thresholds, does not
establish a reliable all-cell exact reference. The dominant cell-level issue
was replicate precision; the selection stage also showed per-delta ECE
failures in 12 of 36 candidate rows. All 27 selected final heads passed their
head-level screens, but only two aggregate cells passed. The two admitted cells do
not authorize extrapolation to SIR or other coordinates.

No SIR artifact was created. No filter, particle, latent-state, Fisher, or
analytical-score route was used.

## Reset For Future Work

- Treat this exact result as terminal for V4. Do not rerun SIR with a relaxed
  gate, altered ECE threshold, or selectively chosen cells.
- Preserve the full artifact and this memo as historical evidence.
- Any next attempt requires a new reviewed plan that states whether it is
  repairing sample/training precision, calibration, or the estimator design.
- Do not reuse V4 controls as defaults; they are scope-specific diagnostics.
- The TensorFlow retracing warning is an efficiency repair opportunity, not
  evidence against the mathematical target.

## Nonclaims

This run does not establish an exact SIR score, filter correctness, algorithm
ranking, statistical superiority, HMC readiness, default readiness, or
scientific validity of the anchored estimator.
