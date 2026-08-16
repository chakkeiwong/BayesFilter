# GenUT Four-Model Leaderboard Rerun Plan

Date: 2026-08-16
Status: prepared for execution

## Research Intent

Rerun the repository's established four-model GenUT comparison under the
current checkout and determine whether the same algorithm remains finite and
well-behaved across LGSSM, KSC-SV, predator-prey, and Austria-SIR.

## Exact Comparator

Use `docs/benchmarks/run_genut_b098_radial2_four_model.py` unchanged as the
canonical ladder:

- `N=1008`;
- LGSSM `T=50`, KSC-SV `T=10`, predator-prey `T=20`, Austria-SIR `T=20`;
- claim seeds `98201..98216`;
- calibration seeds `98401,98402`, with Austria `98301,98302`;
- diagonal, pairwise, coordinate-cap, and dual-cap arms;
- FP32, TF32, XLA, deterministic operations, and verified memory growth;
- finite/program-valid rows, residuals <= `5e-4`, and capped-coordinate bounds
  as hard validity gates.

The previous artifact is
`docs/benchmarks/artifacts/genut_b098_radial2_four_model_20260807/attempt01/`.
The new attempt is written to a fresh dated directory and never overwrites it.

## Similarity Criteria

The rerun is a reproducibility pass if:

1. all four diagonal baselines remain hard-valid;
2. each arm's model-level validity is recorded without aborting unrelated
   models;
3. no new non-finite values, scores, or residual-gate failures appear for an
   arm that was previously hard-valid;
4. per-model value and score seed SD/MCSE remain descriptive and are reported,
   not ranked across models; and
5. the same-program FD diagnostics and all target/event-order hashes remain
   valid.

Cross-model raw value or score equality is not a criterion. Different models,
horizons, and parameterizations have different scales.

## Vetoes and Nonclaims

Stop only for a target/hash mismatch, invalid baseline, corrupted artifact,
GPU/memory-policy failure, or harness serialization failure. An arm-specific
failure is evidence about that arm and must not abort the other models.

This rerun does not establish exact nonlinear likelihood/score correctness,
statistical superiority, posterior correctness, NeuTra/HMC readiness, or a
default change.

## Compute Contract

One fresh four-model campaign plus one localized infrastructure retry, bounded
by 45 GPU minutes. Runtime environment is the repository `tftwogpu` TensorFlow
build with memory growth configured before device initialization.

## Review Decision

The plan passes skeptical review because it uses the existing scope-bound
baseline ladder, common random numbers, explicit validity gates, fresh output,
and descriptive-only interpretation. The prior Austria pairwise failure is not
silently removed; it is a prespecified reproducibility diagnostic.
