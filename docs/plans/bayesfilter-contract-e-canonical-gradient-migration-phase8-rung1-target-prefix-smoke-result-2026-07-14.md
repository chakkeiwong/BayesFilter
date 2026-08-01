# Phase 8 Result: Target-Prefix Canonical Harness Smoke

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `TARGET_PREFIX_WIRING_SMOKE_PASSED_DESCRIPTIVE_ONLY_NUMERICAL_DESIGN_AND_FORMAL_FD_BLOCKED`

## Outcome

The frozen CPU-hidden float64/XLA `T=1,N=4` target-prefix smoke passed every
predeclared wiring hard check. The dataset hash, physical parameter vector,
PHILOX preparation identity, one-canonical-callable identity, two-call exact
serialized equality, chart histories, telemetry schema/shapes/finiteness,
Kalman finiteness, and no-overwrite artifact contract all passed.

The transferred ridge `4`, epsilon `1/2`, scaling `3/4`, two Sinkhorn steps,
and chunks `2` remain `fixture_transfer_harness_smoke_only`. They are not a
target hypothesis, candidate, comparator, or default.

## Descriptive Output

| Quantity | Contract E | Kalman | Difference |
| --- | ---: | ---: | ---: |
| Value | `-6.814269558275509` | `-6.990431703843754` | `0.17616214556824517` |
| Difference / `abs(Kalman)` | | | `0.025200467300378712` |
| `phi1` physical score | `2.2850433504590337` | `3.0911502178854375` | `-0.8061068674264038` |
| `phi2` physical score | `-0.40231616246855983` | `-0.3858525302251643` | `-0.016463632243395532` |
| `phi3` physical score | `0.437618263282293` | `0.21341994107239287` | `0.2241983222099001` |
| `q_scale` physical score | `6.241225724866608` | `6.03822614894438` | `0.2029995759222274` |
| `r_scale` physical score | `15.400519211330016` | `13.651714745073173` | `1.7488044662568427` |

The HMC-coordinate score differences are
`[-0.3882210673525561,-0.011483383489768406,0.19673402773918736,
0.07104985157277977,0.7869620098155794]`. Every difference is explanatory
only. No equivalence margin, confidence interval, same-program FD criterion, or
target-scale decision applies at this rung.

## Diagnostic Interpretation

The chart is valid, minimum quotient mass is `0.8943673515862458`, and the
maximum row-mass residual is `0.1056326484137542`. The ridged covariance
identity residual has Frobenius norm `2.189633899532375e-15`, while the raw
covariance residual has Frobenius norm `2.803870146247005`. Its error relative
to the exact prediction `lambda*(I-AA^T)` is only
`1.2306710804541245e-15`. Mean restoration residual is
`9.71445146547012e-17`.

This is the expected mathematical consequence of the deliberately large
transferred ridge: the code accurately satisfies the ridged identity while the
raw covariance differs materially. It does not identify a target ridge or show
that Contract E is scientifically wrong. The two-step transport residual and
large raw ridge bias are separate mechanisms requiring independently
predeclared numerical diagnostics.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept target-prefix wiring | exact identity, repeatability, chart, schema, finiteness, provenance | All passed | Only `T=1,N=4` | Close smoke; draft independent numerical-design plan | Numerical adequacy |
| Accept transferred settings | Forbidden | Not evaluated | row convergence and raw ridge bias | Do not select from smoke output | Target defaults |
| Accept Kalman equivalence | No margin or replication applies | Not evaluated | finite-N/setting/seed effects | Preserve differences as explanatory | Value/gradient equivalence |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | All predeclared wiring hard checks passed |
| Statistically supported ranking | None; one estimator seed and no candidate comparison |
| Descriptive-only differences | All Contract E/Kalman and telemetry magnitudes |
| Default-readiness | Not established |
| Next evidence needed | Pre-result target numerical hypotheses, formal FD disposition, then separately planned lower rungs |

## Artifacts

- Harness: `docs/benchmarks/emit_contract_e_canonical_lgssm_phase8_target_prefix_smoke.py`
- Tests: `tests/highdim/test_contract_e_phase8_target_prefix_smoke.py`
- Result JSON: `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/rung1-target-prefix-smoke-attempt1/result.json`
- Result SHA-256: `6219eeb937fe8e7ad7814fc4595f29b14ad214499eb7f275901de838cbeba5be`

## Post-Run Red Team

The strongest alternative explanation for the descriptive oracle differences
is the deliberately transferred ridge/transport arm, not a wiring defect. A
different predeclared arm could change the differences, but this smoke cannot
justify which one. The weakest evidence is target numerical adequacy; it was
intentionally not tested. Nothing here establishes `T=10`, GPU, primary-shape,
HMC, admission, leaderboard, release, or integrity readiness.
