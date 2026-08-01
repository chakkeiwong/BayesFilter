# LGSSM NeuTra Gap Closure Reset Memo

Date: 2026-07-15

## Terminal Update

The sequential repair and fresh confirmation are complete. The terminal result
is
`docs/plans/bayesfilter-lgssm-neutra-sequential-hmc-r2-result-2026-07-15.md`.

- Both frozen candidates passed corrected tuning admission after 2,000
  warm-up and 2,000 retained draws per chain. Each failed the retained R-hat
  gate at 1,000 and correctly extended to 2,000.
- `dense_seed1201` passed fresh confirmation with 2,000 warm-up and 4,000
  retained draws per chain: max modern R-hat `1.002149`, min bulk ESS
  `4571.61`, min tail ESS `3976.95`, max tuned plain-HMC disagreement `2.0802`
  combined MCSE, and max truth distance `1.6290` posterior SD.
- `dense_seed1202` was vetoed during fresh warm-up by one energy-error
  divergence (`log_accept_ratio < -1000`) and produced no posterior sample.

Permitted terminal claim: one specific frozen NeuTra candidate works under the
recorded HMC convergence, comparator-agreement, and recovery gates on this
exact favorable 18D LGSSM fixture. Robustness, generality, superiority,
calibration, production readiness, and default readiness remain unproved.

## Reset State

The current LGSSM NeuTra gap-closure campaign completed fresh training and
frozen objective validation for two independent dense-IAF seeds. Its apparent
HMC tuning stop is superseded: the harness used fixed 1,000-transition warm-up
and fixed 1,000-draw sampling rather than the required sequential modern-R-hat
controller. No confirmatory HMC or posterior agreement/recovery claim is valid
yet.

## What Passed

- Phase 0: TensorFlow-only HMC mechanics, import closure, Gaussian XLA gate,
  and historical frozen-target smoke.
- Phase 1: seed1201 5,000-step GPU/XLA training, accepted after a localized
  lazy-runtime-import repair and checkpoint post-validation.
- Phase 2: seed1202 5,000-step GPU/XLA training; both seeds have finite,
  status-valid terminal checkpoints and exact frozen parity.
- Phase 3: both fresh payloads passed GPU/CPU-hidden XLA value/logdet/score
  parity with maximum value difference `1.07e-13`.

## What The Historical Fixed Run Showed

Phase 4's fixed 1,000-draw admission check missed for both candidates after the
predeclared primary and repair grids:

- seed1201: step `0.8`, verification acceptance `0.69275`, max folded R-hat
  `1.01569` at `a22_raw`;
- seed1202: step `0.8`, verification acceptance `0.70425`, max rank R-hat
  `1.01721` at `a42_raw`.

Both had zero target-status failures, zero energy-error divergence screens, and
all finite/movement health. These values require more retained draws under the
correct policy; they are not terminal candidate rejections.

## Governance Repair Record

Two localized harness issues were repaired without changing the scientific
contract:

1. the strict trainer's closure audit was contaminated by an eagerly imported
   NumPy-backed runtime runner; `bayesfilter.runtime` exports are now lazy;
2. the Phase 3 GPU probe configured memory growth after TensorFlow tensors had
   been initialized; device policy now runs before payload loading.

Focused regression tests and post-validation artifacts preserve both repairs.

## Historical Active Continuation

Execute
`docs/plans/bayesfilter-lgssm-neutra-sequential-hmc-repair-plan-2026-07-15.md`.
It keeps the selected step size, leapfrog count, target, and transports fixed,
uses fresh seeds, archives all warm-up, checks recent warm-up modern R-hat at
`1.05`, and extends cumulative retained sampling until modern R-hat `<=1.01`
or 10,000 draws per chain. Historical artifacts remain immutable.

This continuation is now complete; retain the section as the audit trail from
the fixed-budget correction to the terminal result.
