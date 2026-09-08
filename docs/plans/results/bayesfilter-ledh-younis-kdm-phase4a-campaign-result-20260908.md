# Phase 4A Campaign Result: Authoritative Small-Cell All-Bandwidth Result

Date: 2026-09-08  
Governing plan: `docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md`  
Research reset: `docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md`

## Decision

The repaired Phase 4A endpoint is an executable, total-derivative
`KDM-FINITE` diagnostic on its declared target. The authoritative
`N=32, T=5` float32/no-TF32 result is Attempt 05, which retains every
prespecified bandwidth on the same 100-path holdout. Calibration selected
`rho=0.8`, whose validation score MSE was 4.84% higher than the canonical
atom endpoint. Its primary paired bootstrap interval crosses zero, so that
selected comparison is not statistically ranked. More strongly, none of the
eight positive bandwidths has a positive point MSE gain on the consumed
holdout, and `rho=1.6` is statistically worse. Phase 4A is rejected for
promotion in this scope. That result does not reject the distinct Phase 4B
resampling mechanism.

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
| Artifact | `docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt05-n32t5-all-rho-f32-current/` |

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

### Attempt 03: initial one-cell decision run

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

The historical directory name contains the word `powered`, but no prospective
variance calculation established that description. It is retained as a path
name only and is not the authoritative all-bandwidth result.

### Attempts 04 and 05: all-bandwidth holdout inspection

Attempt 04 added same-stream validation for every prespecified bandwidth.
Attempt 05 repeated that inspection with current source hashes and retained the
complete result, so Attempt 05 is authoritative. Its selected-bandwidth primary
statistics agree with Attempt 03 up to bootstrap resampling:

| `rho` | Validation MSE | Relative gain versus atom | Paired 95% interval |
|---:|---:|---:|---:|
| `0.025` | `3.336629` | `-0.0006%` | `[-0.000408, 0.000469]` |
| `0.05` | `3.336684` | `-0.0022%` | `[-0.001632, 0.001850]` |
| `0.10` | `3.336941` | `-0.0099%` | `[-0.006254, 0.007533]` |
| `0.20` | `3.338508` | `-0.0569%` | `[-0.024672, 0.031335]` |
| `0.40` | `3.352626` | `-0.4800%` | `[-0.088942, 0.129793]` |
| `0.80` | `3.498000` | `-4.8369%` | `[-0.222624, 0.597347]` |
| `1.20` | `3.893153` | `-16.6799%` | `[-0.156682, 1.361369]` |
| `1.60` | `4.525191` | `-35.6224%` | `[0.189227, 2.399222]` |

The atom MSE is `3.336610`. The selected `rho=0.8` error has mean
`-1.014` and standard deviation `1.580`, compared with atom mean
`-0.868` and standard deviation `1.616`. The small variance
reduction is more than offset by increased bias. Because every bandwidth was
inspected on validation, this holdout is consumed and cannot be used to choose
a new Phase 4A setting.

## Inference status

| Evidence class | Status | Interpretation |
|---|---|---|
| Hard veto screen | Pass | The runner, oracle, disjoint splits, source hashes, memory policy, and endpoint outputs are valid. |
| Statistically supported ranking | No | The bootstrap interval includes zero; the pilot cannot rank KDM-FINITE and ATOM-FINITE. |
| Descriptive difference | Every positive bandwidth is worse by point MSE; `rho=1.6` is statistically worse | The selected KDM-FINITE arm has 4.84% higher validation MSE. This rejects Phase 4A in the cell, not the distinct Phase 4B mechanism. |
| Default readiness | No | The candidate is diagnostic-only and fails the declared promotion criterion. |
| Next evidence needed | Different mechanism on fresh data | The complete Phase 4A rho curve is now inspected. Any further Phase 4A test needs fresh data and a new rationale; the planned next mechanism is Phase 4B full-mixture resampling. |

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

The strongest alternative explanation is that this small `N=32,T=5` cell
does not represent larger-particle or longer-horizon behavior. That uncertainty
does not rescue Phase 4A here: all positive bandwidths lost by point MSE, the
largest bandwidth lost decisively, and the observed bias increase explains why
the selected arm did not benefit from its modest variance reduction.

The same-stream diagnostic requested after Attempt 03 is now complete. A
larger Phase 4A cell would require fresh calibration and validation data, a
prospective power calculation, and its own compile budget. The more
discriminating next question is whether the now-specified Phase 4B
full-mixture resampling mechanism performs differently on fresh paired LGSSM
paths. Its implementation correctness must be established separately before
that scientific campaign.
