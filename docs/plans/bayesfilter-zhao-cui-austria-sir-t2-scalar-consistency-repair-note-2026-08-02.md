# Zhao-Cui Austria SIR T2 Scalar-Consistency Repair Note

Date: 2026-08-02

Status: `FINAL_T2_ATTEMPT_ELIGIBLE_AFTER_DIAGNOSTIC`

## Problem

The packed T2 primal materially replays the admitted T2 density and scalar, but
raw centered differences of gauge-sensitive TT cores do not reproduce the
centered difference of the scalar normalizer when contracted at the parent
core coordinates. The maximum raw mismatch is `0.0030439` for parameter 0.
Changing the contraction base from admitted cores to replay cores does not
repair it. Direct scalar centered differences at `h=5e-5` and `h=1e-4` agree
within `7.55e-5`, so scalar step instability is not the leading explanation.

Evidence:

- `t2-primal-diagnostic-01/result.json` passes material functional/scalar replay.
- `t2-fd-tangent-diagnostic-01/result.json` localizes the raw core/scalar mismatch.
- `t2-fd-tangent-diagnostic-02/result.json` rejects replay-base mismatch as the repair.

## Repair

Retain the raw centered-difference core tangent for shape directions, then add
one radial component to the first TT core so the manual derivative of the same
finite scalar

```text
log(square_mass(cores) + tau) - shift
```

equals its direct `h=5e-5` centered difference. The correction coefficient is

```text
(direct_scalar_derivative - raw_core_contraction)
/ derivative_along_first_core_radial_direction.
```

This is the calibration-scale direction already fixed by the T2 scalar
program. It does not alter the origin value, parent density, optimizer,
prepared clouds, shifts, `tau`, or material replay gate. The artifact must bind
the raw core contraction, direct derivative, radial correction, corrected
manual score, and independent `h=1e-4` scalar check.

Method identifier:
`tensorflow_xla_centered_fd_shape_plus_scalar_radial_projection_h5e5_v1`.

## Evidence Contract

- Corrected manual increment score must match the direct `h=5e-5` increment
  derivative under the existing `3e-4 + 3e-4*abs(score)` gate.
- Corrected manual cumulative score must match the direct `h=5e-5` cumulative
  derivative under the same gate.
- Independent `h=1e-4` cumulative centered differences must match the direct
  `h=5e-5` cumulative derivative under the same gate.
- All four material functional screens, scalar replay, strict T1/T2 loader
  chain, prepared-input identities, tensor hashes, and 6 GiB memory cap remain
  hard gates.
- Raw core FD mismatch and the radial correction are required explanatory
  evidence; they may not be hidden or called exact autodiff.

## Nonclaims

- The raw core centered difference is not an accurate derivative coordinate.
- No exact-autodiff or JVP result is claimed.
- No arbitrary-theta, later-horizon, HMC, source-faithful parameter-estimation,
  or posterior-correctness claim is made.
