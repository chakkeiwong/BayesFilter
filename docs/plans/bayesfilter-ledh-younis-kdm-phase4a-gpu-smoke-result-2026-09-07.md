# Phase 4A GPU/XLA Mechanics Smoke Result

Date: 2026-09-07  
Plan: `docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md`  
Role: bounded engineering gate, not research evidence

## Question and contract

The complete Phase 4A kernelized-observation endpoint must compile and run on
the repository GPU/XLA route, produce finite valid output on a two-step
full-feedback fixture, replay deterministically, and agree with its non-XLA
graph execution within the frozen absolute thresholds.  The calibration
follow-up also checks the explicit atom-identity and observation-map validity
flags, because XLA does not execute TensorFlow assertions:

| quantity | threshold |
|---|---:|
| value | `2e-4` |
| score | `5e-4` |
| post-reset state | `5e-4` |
| posterior weights | `2e-4` |

The smoke does not estimate score MSE, select bandwidth, establish DSGE or HMC
validity, or promote a default. Internal TensorFlow/XLA validity is recorded
separately from graph/XLA parity.

## Environment and artifacts

- Python: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`
- TensorFlow: `2.20.0-dev0+selfbuilt`
- GPU: NVIDIA GeForce RTX 4080 SUPER, compute capability 8.9, selected with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`
- allocator: repository `memory_growth` policy, verified before logical-device
  initialization
- route: `ledh_younis_kdm_integrated_observation_weighting_v1`
- target: `KDM-FINITE`
- fixture: two-dimensional nonlinear, `N=8`, `T=2`, positive diagonal
  bandwidth `0.08`, Contract-E plus diagonal and pairwise dual caps
- source/result artifacts: unique directories below
  `docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260907/`

## Attempts

| attempt | arm | result | interpretation |
|---|---|---|---|
| 01 | initial float32+TF32 | stopped before evidence | float32 identity guard used an absolute `1e-8`; observed one-ulp-scale `2.98e-8`; repaired with a dtype-relative floor, no numerical result accepted |
| 02 | float32+TF32 after guard repair | XLA compiled; parity failed | value `5.07e-3`, score `5.36e-3`, state `2.16e-3`, weights `7.15e-4`; internal route validity finite/true; TF32 parity veto |
| 03 | float32, TF32 disabled | passed | value `0`, score `7.15e-7`, state `1.43e-6`, weights `1.19e-7`; deterministic replay |
| 04 | float64, TF32 disabled | passed | value `2.66e-15`, score `8.88e-16`, state `2.44e-15`, weights `5.55e-16`; deterministic replay |
| 05 | float32+TF32 after validity wiring | failed validity and parity | `valid=false` was observed; this artifact predates the final component fields, so the calibration below is the authoritative localization |

Every failed attempt is preserved. The repair/retry kept the target, method,
data, settings, device class, and bounded smoke scope unchanged.  Calibration
attempt 01 was a harness-only float32/float64 comparison-cast failure;
attempt 02 was repaired by casting the reference before subtraction.  The
corrected calibration is
`docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260907/tf32-calibration-attempt03/result.json`.

The corrected calibration found:

- the float32 no-TF32 graph/XLA baseline passed, with all validity flags true;
- the float64 XLA reference passed, with all validity flags true;
- the TF32 graph/XLA pair first diverged in the traced pre-flow state; and
- the TF32 XLA arm itself failed the explicit observation-map and atom-identity
  checks (`observation_map_value_error_max=7.205e-4`,
  `atom_factor_value_error_max=8.445e-4`), while its finite factor values were
  still emitted.  This is a route-validity failure, not a reason to relax the
  graph/XLA threshold.

## Decision

The full endpoint is graph/XLA-compatible and reproducible in float64 and
float32 without TF32. The TF32 arm is not admitted: the corrected calibration
shows both a reproducible graph/XLA parity error and a failed exact
observation/atom representation check in the XLA arm. The result is a
numerical-validity/promotion veto. It is not evidence that kernelized
observation weighting improves or worsens the statistical score.

The calibration gate has now localized the failure to TF32-sensitive
linear-algebra paths, including the supplied linear observation representation
and the early flow state.  The next action is a code-level repair evaluation:
either make the representation and its validity check use an operationally
identical, non-TF32 path, or explicitly keep this route on the no-TF32/float64
reference arm.  A measured scope-specific error contract may be considered
only after the exact identity checks pass; threshold relaxation alone is not
acceptable.

Only after that gate passes should the bounded LGSSM score-error pilot run. The
pilot must compare `ATOM-FINITE`, Phase 4A `KDM-FINITE`, the exact Kalman score,
and the required heuristic baselines on paired fixed streams, with uncertainty
and a fresh untouched validation partition. No Phase 4B mixture-resampling
implementation is authorized by this result.

## Inference status

| row | status |
|---|---|
| hard veto screen | TF32 arm failed explicit identity validity and graph/XLA parity; no TF32 promotion |
| statistically supported ranking | not applicable; no score-error campaign run |
| descriptive-only differences | TF32 graph/XLA error and timings are mechanics diagnostics only |
| default readiness | not ready for Phase 4A; no canonical/default change |
| next evidence needed | TF32 operation-level repair or a documented no-TF32 route decision, then powered paired LGSSM oracle comparison |
