# F0 Plain-HMC Comparator Repair Record

Date: 2026-07-15  
Classification: `LOCAL_TUNING_HARNESS_REPAIR`

## Failed Attempt

`plain-hmc/comparator/result.json` rejected before sequential sampling. The
target, affine mass, and probes through step size `0.4` were finite,
status-valid, mobile, and free of declared energy-error events. Step size `0.8`
had acceptance zero and 256 `log_accept_ratio < -1000` events, so that
configuration is hard-vetoed.

The harness then incorrectly treated one grid configuration veto as a veto of
all healthy configurations. In addition, the grid did not resolve the bracket:
acceptance moved from `0.9336` at `0.2` to `0.3594` at `0.4`, leaving no healthy
point in the predeclared `0.60-0.90` nomination band.

## Repair

Preserve the failed artifact. Run one fresh attempt under
`plain-hmc/comparator-repair-attempt-02` with step sizes
`(0.225, 0.25, 0.275, 0.3, 0.325, 0.35)`. A health veto rejects its own fixed
kernel configuration and remains hard evidence against that configuration; it
does not invalidate different healthy step sizes on the same already-valid
target. Sequential sampling may begin only from a healthy acceptance-band
nominee.

The target, fixture, mass, affine transform, thresholds, seeds classes,
warm-up/retained 10,000 caps, truth-recovery gate, hardware, privacy boundary,
and total F0 budget are unchanged. This is the one localized retry authorized
by the F0 plan.
