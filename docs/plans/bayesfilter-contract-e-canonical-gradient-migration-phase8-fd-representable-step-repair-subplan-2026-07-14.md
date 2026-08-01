# Phase 8 Repair Subplan: Representable Symmetric FD Steps

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `CLOSED_SEVEN_STEP_FD_HEURISTIC_PASSED_FORMAL_CERTIFICATE_UNSUPPORTED`

## Phase Objective

Test the localized hypothesis that the seven-step FD screen was inconclusive
only because its irrational-like cube-root-epsilon base did not generate
bitwise-symmetric float64 endpoints around dyadic fixture coordinates. Freeze a
representable power-of-two approximation to the same numerical scale before
running a new screen. This remains a heuristic-only FD diagnostic.

## Entry Conditions

- The formal callable-error-bound certificate is unconditionally unsupported.
- The first reviewed screen is closed inconclusive: 13/35 exact-symmetric pairs,
  all 13 passing, with 22 pairs invalid only for bitwise step asymmetry.
- Center, source, prepared inputs, charts, branches, one callable, and XLA all
  passed.
- The first screen's ladder and artifacts remain immutable; this is a new plan
  and new output root.
- No target-prefix, `T=10`, GPU, primary-shape, or Kalman comparison runs here.

## Frozen Representable Step Construction

The mathematical cube-root target for float64 is

```text
h_target_j = cbrt(2^-52) * max(1,abs(theta_j)).
```

For this fixture every `abs(theta_j)<1`, so the scale is one for all five
coordinates. Define one dyadic base, before endpoint output, by

```text
e = round(log2(cbrt(2^-52)))
h_base = 2^e
```

where ties round toward the larger step. Numerically `e=-17` and
`h_base=2^-17`. The seven ordered steps are exactly

```text
[8,4,2,1,1/2,1/4,1/8] * h_base
= [2^-14,2^-15,2^-16,2^-17,2^-18,2^-19,2^-20].
```

This is the nearest power-of-two approximation in log scale to the same
cube-root-epsilon target, not a step chosen from the prior FD errors. The center
coordinates and every declared step are recorded as exact binary64 hex values
before execution. For each coordinate and step, the harness must verify

```text
float(theta_j + h) - theta_j == h
theta_j - float(theta_j - h) == h
```

by bitwise float64 equality, where `h` is the declared dyadic step. It must also
verify the two actual steps are bitwise equal. This checks binade alignment and
prevents a symmetrically rounded but different step from silently replacing the
declared ladder. All 35 pairs must pass these nominal-equality checks before the
heuristic is applied.

All other rules remain exactly those of the reviewed reclassification plan:
same fixture, prepared hashes, source closure, one float64 XLA value-and-score
callable, branch/chart identity, diagnostic denominator eligibility,
`abs(score-FD)/abs(FD) <= 0.05*sqrt(5)` at all 35 pairs, no step selection,
and explanatory Richardson diagnostics only.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does a predeclared nearest-dyadic base repair only the representable endpoint asymmetry while retaining same-program FD agreement at every step? |
| Comparator | Closed attempt-2 identities and frozen Phase 5 v2 fixture/callable; not its observed relative errors for selection |
| Pass criterion | all 35 pairs exactly symmetric, finite, branch/chart valid, denominator-eligible, and below the FD-only heuristic threshold |
| Failure | all endpoint predicates valid but any relative error exceeds threshold |
| Inconclusive | any endpoint predicate invalid or denominator ineligible |
| Hard veto | source/prepared/center drift, separate value graph, missing record, overwrite, or altered ladder after output |
| Not concluded | rigorous derivative proof, Kalman equivalence, target-shape FD, numerical/default/HMC/leaderboard readiness |

## Skeptical Plan Audit

Decision: `PASS_FOR_ONE_REPRESENTABLE_STEP_REPAIR_ATTEMPT_ONLY`.

- The repair mechanism follows directly from the observed invalid predicate and
  binary64 representation, not from optimizing relative error.
- The power-of-two base is determined analytically from machine epsilon before
  the new output.
- All 35 pairs must pass; no favorable subset can be selected.
- Forward autodiff zero-ULP equality remains the primary engineering evidence.
- Formal rigorous FD remains unsupported regardless of result.

## Required Artifacts And Budget

- patch the dedicated FD harness to expose a `nearest_dyadic_cuberoot` mode
  without changing its closed default evidence;
- add exact exponent/base/endpoint tests;
- output root
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/fd-representable-repair-attempt1/`;
- CPU-hidden/XLA environment and exact fixture/source/prepared hashes;
- one scientific attempt, `300`-second timeout, no retry without another
  reviewed plan;
- a preflight-only repair is permitted only before any endpoint call and must
  record endpoint evaluation count zero; a defect after the first endpoint
  call is a blocker, not an automatic retry;
- result, focused checks, manifest, and close record.

## Forbidden Claims And Actions

- Do not change threshold, center, fixture, callable, or seven multipliers.
- Do not choose the dyadic exponent from the prior error values.
- Do not call a pass rigorous proof or use it outside FD.
- Do not run target-prefix, `T=10`, GPU, primary shape, HMC, nonlinear,
  leaderboard, release, or integrity work.

## Handoff And Stop Conditions

A pass closes only the same-program FD heuristic screen. Failure or
inconclusiveness is recorded without another automatic ladder. Every result
leaves the owner target numerical and primary statistical design blockers
unchanged. Stop for any identity/endpoint veto, campaign-clock exhaustion, or
material review finding; a pre-output harness defect may be repaired only under
the single-attempt budget only when no endpoint call occurred. A defect after
the first endpoint call is a blocker.

## Frozen Artifact Identity

The prior closed screen is
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/fd-reclassification-attempt2/result.json`
with SHA-256
`5261f5a627b14951f15a39d1e7ef5a8db2916f6e3ce413a25f33a5a74377f1c7`.
The fixture is
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase5-tiny-fixture-freeze-v2-2026-07-14.json`
with SHA-256
`f6b6e2895208d7cd5cba0f57b05d4de7fb0de79e50ba62b7e6c70b06879942f4`.
Expected center objective, score, branch, all eight prepared-input hashes, and
current source closure are exactly those frozen in the reviewed FD
reclassification subplan and attempt-2 artifact; the new harness must reproduce
them before endpoint evaluation.

The exact new result path is
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/fd-representable-repair-attempt1/result.json`.
The writer refuses an existing path. The artifact records command, Git revision,
Python/TensorFlow, logical devices, float64 `jit_compile=True`, one concrete
callable, wall time, source/prepared hashes, every center hex value, every
nominal step hex value, and every actual step hex value.
