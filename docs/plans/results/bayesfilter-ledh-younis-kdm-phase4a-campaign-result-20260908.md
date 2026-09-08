# Phase 4A Campaign Result: Small-Cell Pilot

Date: 2026-09-08  
Governing plan: `docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md`  
Research reset: `docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md`

## Decision

The repaired Phase 4A endpoint is an executable, total-derivative
`KDM-FINITE` diagnostic on its declared target. The powered `N=32, T=5`
float32/no-TF32 cell does **not** provide promotion evidence: after selecting
`rho=0.8` on 20 calibration paths, the candidate has a descriptively larger
validation score MSE than the canonical atom endpoint (4.84% worse). The
95% paired bootstrap interval crosses zero, so the ranking is not statistically
supported. The research direction is not rejected; the candidate is rejected
for promotion in this scope and the broad ladder is paused pending a
discriminating diagnostic.

This result is not evidence that the positive-bandwidth finite target equals
the canonical score, that KDM improves score estimation generally, or that the
method applies to a DSGE model.

## Evidence contract

| Item | Declared choice |
|---|---|
| Question | Does full-feedback KDM-FINITE reduce score MSE relative to the actual Contract-E atom endpoint against an independent Kalman score? |
| Baseline | `ATOM-FINITE`, the full Contract-E/GenUT/dual-cap analytical endpoint, on the same streams. |
| Candidate | `KDM-FINITE`, the shared executor with exact linear-Gaussian observation kernelization and full tangent feedback. |
| Oracle | Independent scalar AR(1) Kalman value and analytical score. |
| Primary criterion | Validation paired squared-error bootstrap upper bound `< 0` and relative MSE gain `>= 5%`. |
| Hard vetoes | Nonfinite or invalid output, missing disjoint split, stale/mismatched source or plan identity. |
| Explanatory only | Calibration curves, runtime, first-call compilation, bandwidth, and descriptive error differences. |
| Nonclaims | No canonical-default, HMC, DSGE, Phase 4B, posterior, or general score claim. |
| Artifact | `docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt03-n32t5-powered-f32/` |

## Runs

### Attempt 02: float64 pilot

Command:

```text
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  docs/benchmarks/run_ledh_younis_kdm_phase4a_campaign.py \
  --output docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt02-n32t5-pilot/result.json \
  --particle-counts 32 --horizons 5 --calibration-reps 4 \
  --validation-reps 20 --bootstrap-reps 1000 --max-cells 1 \
  --dtype float64 --tf32-mode disabled --jit-compile true
```

This was a harness pilot, not a powered decision run. It completed with all
finite outputs and selected `rho=0.4`. Validation MSE was `1.2678422` for
KDM-FINITE versus `1.2338021` for ATOM-FINITE; relative gain was `-2.76%`.
The interval `[-0.0716, 0.1458]` crossed zero. The artifact is retained at
`docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt02-n32t5-pilot/result.json`.

### Attempt 03: powered one-cell pilot

Command:

```text
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  docs/benchmarks/run_ledh_younis_kdm_phase4a_campaign.py \
  --output docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt03-n32t5-powered-f32/result.json \
  --particle-counts 32 --horizons 5 --calibration-reps 20 \
  --validation-reps 100 --bootstrap-reps 5000 --max-cells 1 \
  --dtype float32 --tf32-mode disabled --jit-compile true
```

The run completed in `70.06 s` with GPU memory growth, XLA, and TF32
disabled. Calibration selected `rho=0.8`. The validation comparison was:

| Quantity | Value |
|---|---:|
| ATOM-FINITE validation MSE | `3.3366101` |
| KDM-FINITE validation MSE | `3.4980001` |
| Relative MSE gain | `-4.8369%` |
| Paired MSE difference (KDM minus atom) | `0.1613900` |
| Paired bootstrap 95% interval | `[-0.22057, 0.57834]` |
| Paired-difference MCSE | `0.20471` |
| Promotion verdict | `NO_PROMOTION_EVIDENCE` |

All `100` validation paths were finite and the calibration and validation
streams were disjoint. The artifact preserves the path-level rows and
manifest at
`docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt03-n32t5-powered-f32/`.

## Inference status

| Evidence class | Status | Interpretation |
|---|---|---|
| Hard veto screen | Pass | The runner, oracle, disjoint splits, source hashes, memory policy, and endpoint outputs are valid. |
| Statistically supported ranking | No | The bootstrap interval includes zero; the pilot cannot rank KDM-FINITE and ATOM-FINITE. |
| Descriptive difference | KDM worse | KDM-FINITE has 4.84% higher validation MSE in this salient small-particle, short-horizon cell. This is a promotion veto for the cell, not a direction veto. |
| Default readiness | No | The candidate is diagnostic-only and fails the declared promotion criterion. |
| Next evidence needed | Target diagnosis | Compare both routes against the oracle on the same held-out paths, inspect the complete rho curve, and determine whether the baseline's finite-particle error or the positive-bandwidth target is responsible before spending a larger compile budget. |

The canonical atom endpoint is the cheap heuristic adversary in this cell.
The candidate loses descriptively to it, so no claim-bearing promotion is
allowed for this scope even though the uncertainty interval is wide.

## Engineering and environment evidence

The repaired Contract-E covariance carry passed the focused CPU suite (`50`
tests before the oracle additions and `28` tests in the oracle/KDM subset).
The independent scalar LGSSM reference and campaign runner also passed their
focused tests (`28 passed` in the combined reference/covariance/KDM suite).

Fresh complete-endpoint GPU/XLA smoke results after the covariance repair:

- float64/no-TF32: pass; value error `2.66e-15`, score error `8.88e-16`;
- float32/no-TF32: pass; value error `1.91e-6`, score error `4.77e-7`;
- float32/TF32: fail; value error `2.38e-3`, score error `1.88e-3`, above the
  declared `2e-4`/`5e-4` gates.

Therefore TF32 remains vetoed for this diagnostic route. This is a numerical
identity failure, not a score-error result.

The repaired timing probe at `N=128, T=20`, float64, XLA, no TF32 required
`296.10 s` for compile plus first execution, `0.2318 s` per warm execution,
and `31,457,280` peak GPU bytes. The older six-row timing and `0.10` GPU-hour
budget in the historical execution summary describe a pre-repair call chain
and must not be used for campaign planning.

## Post-run red-team

The strongest alternative explanation for the pilot's negative result is not
that KDM is intrinsically harmful: `N=32,T=5` is a noisy small cell, and the
interval is wide. A second explanation is that the selected positive-bandwidth
program changes the finite target in a way that increases bias even while
smoothing a component of variance. The pilot cannot distinguish those causes.

The next run must therefore remain small and diagnostic. It should first
produce a same-stream table for every `rho` against both the Kalman oracle and
ATOM-FINITE, with the calibration choice frozen before holdout evaluation. A
larger cell or broad ladder is not authorized by this result alone. No Phase
4B implementation is authorized until its complete mixture proposal law is
derived.
