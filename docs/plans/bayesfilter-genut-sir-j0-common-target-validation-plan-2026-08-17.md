# GenUT/SIR `j0` Common-Target Validation Plan

Date: 2026-08-17

## Research intent

Determine whether the large Austria-SIR `T=20` `j0` discrepancy is explained
by (a) a simulator-law mismatch, (b) classifier estimation error, (c) GenUT
finite-program/score error, or (d) insufficient matched evidence.

The target is the observed-data score
`d/dtheta log p_theta(y_obs)` for the fixed Austria-SIR observation path,
coordinate `j0`, under the source Zhao-Cui SIR generative law.

## Evidence contract

- Baseline/comparators: the existing observation-only classifier holdout
  bundles and the existing repaired-permutation GenUT claim rows.
- Primary diagnostic: exact source-law/event-order/parameterization agreement,
  including susceptible-clipping event counts.
- Secondary diagnostic: raw per-seed GenUT `j0` distribution versus the
  inherited classifier `j0` holdout distribution.
- Hard vetoes: non-finite output, hash mismatch for the fixed observation,
  parameterization/event-order mismatch, or any failed exact oracle already
  required by the upstream method.
- Explanatory only: seed means, standard deviations, sign counts, and
  classifier-versus-GenUT differences. They cannot establish correctness or
  superiority.
- Nonclaims: no SIR exact-score oracle, no classifier admission, no GenUT
  score correctness, no method ranking, and no HMC/default readiness.

## Pre-run audit and assumptions

The Gaussian classifier campaign failed its exact oracle gate in 8/9 cells;
the GenUT route failed the exact LGSSM Kalman score gate. Those are preserved
as hard upstream veto evidence and are not weakened here. The existing
classifier bundle values are reused only as descriptive holdout evidence.
The SIR simulator and GenUT adapter share the parameterization and RK4 stage,
but the simulator clips susceptible coordinates after process noise while the
GenUT adapter does not; this must be measured rather than assumed inactive.

## Execution

Run `docs/benchmarks/run_genut_sir_j0_common_target_validation_20260817.py` in
the `tftwogpu` environment. It records source hashes, CPU/GPU fixed-path
hashes, FP32/FP64 transition round-off, clipping counts, inherited classifier
values, and the 16 raw repaired-permutation GenUT `j0` rows. The output root
is unique and must not overwrite prior artifacts.

## Stop conditions

Stop without interpretation if any required artifact is missing, hashes do
not match, or any diagnostic produces non-finite values. If clipping is active
at a material rate, classify the discrepancy as a simulator-law mismatch
candidate and do not compare scores as if they shared a target. Otherwise,
the remaining discrepancy is a diagnostic signal against the current GenUT
finite program, with classifier estimation uncertainty still unresolved.

## Planned artifact

`docs/benchmarks/artifacts/genut-sir-j0-common-target-validation-20260817/attempt08/`
containing `result.json`, `result.md`, and the run manifest embedded in the
JSON. Attempts 01--07 are preserved harness-failure/probe-repair evidence;
only attempt08 is used for interpretation. The environment maps
`CUDA_VISIBLE_DEVICES=0` to physical `nvidia-smi` GPU1 (RTX 4080 SUPER), so
attempt06 is the terminal run on the requested GPU. A reset/result memo
records the decision table and nonclaims.
