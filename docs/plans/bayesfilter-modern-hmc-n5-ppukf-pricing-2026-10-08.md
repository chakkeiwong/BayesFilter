# N5 additional nonlinear pricing cell: NL-PP-UKF

Date: 8 October 2026  
Status: bounded CPU reference/mechanics pricing diagnostic.

This phase prices one harder actual nonlinear adapter after the LGSSM and
NL-ACC randomized-HMC bridge. It uses the same public `tune_hmc_kernel` route,
typed `HMCTransitionProfile("hmc", "uniform_duration")`, candidate grid,
one-sided qualification policy, four-start trial definition, and independent
trial schedule as the bridge. The target is the current NL-PP-UKF adapter;
its UKF posterior is the numerical target being sampled and is not an exact
posterior for the underlying nonlinear generative model.

## Evidence contract

The primary question is whether this adapter completes the full public
measurement and fresh verification lifecycle with complete typed work pricing.
Pass requires two retained verified candidates, valid finite evidence for all
16 measurement and 16 verification trials per candidate, exact charged versus
expected conservative work, and preserved manifest/result/summary files.
Hard vetoes are target/score or correction failure, nonfinite state/status,
stream or identity mismatch, incomplete trials, missing work charge, stale
artifacts, or a legacy-runner fallback. The declared replicated lower acceptance
bound gates tuning membership. Other acceptance summaries, R-hat, ESS, MCSE and
wall time are explanatory; posterior diagnostics cannot gate or rank tuning candidates.

This phase makes no posterior accuracy, convergence, burn-in, sampler
efficiency, GPU, default-readiness, or scientific-validity claim. The CPU
route hides CUDA and records `use_xla=False` as an explicit reference-only
exception. Device availability is time-specific and must be probed before a
new GPU launch; the historical CPU artifact cannot establish current readiness.

## Bound and pre-mortem

The complete cell is capped at 150,000 typed work units and ten scheduler
units, with no repairs or refinement. The expected maximum is 114,688 units
for the two candidates and both stages. The run could pass while using a
legacy kernel, silently dropping the UKF status payload, or confusing a
numerical UKF target with the generative model; runner identity, target
adapter/source hashes, transition payloads, trial ranges, status policy and
the numerical target semantics are therefore inspected in the result.
Any target-specific failure is a repair trigger for this adapter and does not
reject the modern-HMC research direction.

## Command and output

Run:

```bash
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 \
  TF_NUM_INTEROP_THREADS=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  scripts/run_modern_hmc_bridge.py \
  --targets NL-PP-UKF \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/attempt-05
```

The fresh output root contains the manifest, candidate-set result, immutable
numerical evidence and bridge summary. It is diagnostic evidence only.
