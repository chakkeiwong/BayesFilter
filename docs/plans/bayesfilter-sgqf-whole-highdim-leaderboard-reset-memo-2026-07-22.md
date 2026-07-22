# SGQF Whole High-Dimensional Leaderboard Reset Memo

Date: 2026-07-22

## Current State

Use
`docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/attempt07/sgqf-column/result.json`
as the current SGQF column artifact. It reports
`sgqf_column_complete=true` for six applicable main rows and one excluded
scoped component row.

The whole three-algorithm comparison is not ready. Do not replace the July
historical full leaderboard with the SGQF-only artifact or interpret
`comparison_ready=false` as an SGQF failure.

## Important Corrections

- Zhao--Cui source rows transition before every observation.
- Canonical scalar-SV seed is `81101`, not the stale `81102` written in an
  early Phase 0 draft.
- Actual and KSC SV share reset data but have different likelihoods.
- Fixed SIR J=9, d=18 has no free theta and therefore no score.
- Parameterized SIR is a scoped Zhao--Cui local component and is not applicable
  to SGQF.
- Predator-prey `-103.13789` is an initial-observation-first amended target;
  source-order SGQF is `-102.62270352134469`.
- Generalized SV uses scalar `GeneralizedSVPriorMeanSSM`, raw observations,
  level 3, and a manual score. It is not `NativeGeneralizedSVSSM`.

## Main Code

- `docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py`: applicability
  schema, source-order wiring, repository identities, `--sgqf-only` program.
- `bayesfilter/highdim/source_sv_sgqf_tf.py`: sealed transition-first scalar-SV
  dataset.
- `bayesfilter/highdim/fixed_sir_sgqf_tf.py`: fixed SIR value-only route.
- `bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py`: preserved local
  amended route plus sealed source-order route.
- `bayesfilter/highdim/generalized_sv_sgqf_tf.py`: raw-y Gaussian-projection
  value/manual-score route and dense reference.
- `docs/benchmarks/run_generalized_sv_sgqf_validation.py`: CPU/GPU XLA,
  refinement, FD, identity, and manifest runner.

## Remaining Work

1. If universal GPU/XLA is required, refactor LGSSM/actual-SV/KSC public SGQF
   routes into graph-native kernels and produce route-matched artifacts.
2. Repair or bound the slow generalized-SV Zhao--Cui TT comparator before a
   new full three-way regeneration.
3. Rerun UKF and Zhao--Cui on every reset source-order dataset before claiming
   comparison readiness; old initial-observation-first values are historical.
4. Use stochastic uncertainty evidence before any PF/SMC ranking.

## Concurrent Lane Boundary

Do not revert or absorb concurrent GenUT, transport, structural-UKF, source
route, or Zhao--Cui APF/rung-2 changes. The SGQF lane did not edit those files.

## Reproduction

CPU-only final column:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-mpl \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py \
  --sgqf-only \
  --output /tmp/sgqf-column.json \
  --markdown-output /tmp/sgqf-column.md
```

Generalized-SV trusted GPU validation must be run with GPU access and the CPU
reference artifact, following the current machine GPU/CUDA policy.
