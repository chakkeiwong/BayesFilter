# GenUT/SIR `j0` Common-Target Validation Result and Reset Memo

Date: 2026-08-17  
Plan: `docs/plans/bayesfilter-genut-sir-j0-common-target-validation-plan-2026-08-17.md`  
Terminal artifact: `docs/benchmarks/artifacts/genut-sir-j0-common-target-validation-20260817/attempt08/`

## Decision

The simulator-law mismatch hypothesis was not supported. The CPU replay of the
fixed Austria-SIR `y1:y20` path matches the frozen source hash
`cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07`. GPU
replay differs from CPU by at most `1.42e-14`. The parameterization and RK4
transition agree; FP32 GenUT versus FP64 simulator transition round-off is
`0.00150` in the probe. Susceptible clipping was inactive in all `163,840`
source-scale transition probes.

The observed `j0` discrepancy therefore remains a GenUT finite-program/score
error candidate, with classifier estimation error unresolved. This is a
diagnostic classification, not a proof that either score is correct.

The terminal run used physical GPU1, RTX 4080 SUPER. This environment maps
that device through `CUDA_VISIBLE_DEVICES=0`; `CUDA_VISIBLE_DEVICES=1` maps to
physical GPU0, RTX 5080. The two terminal runs produced identical diagnostic
values.

## Results

| Source | Mean `j0` | Sample SD | Positive | Negative |
|---|---:|---:|---:|---:|
| Existing classifier holdout bundles, 10 | `50.98` | `24.15` | 10 | 0 |
| Repaired-permutation GenUT seeds, 16 | `-83.26` | `270.47` | 8 | 8 |

GenUT raw values range from approximately `-569.89` to `387.13`; its negative
mean is not a uniformly negative realization. The classifier values are
inherited descriptive holdout outputs, not an admitted exact reference: the
upstream Gaussian classifier exact-oracle campaign failed 8 of 9 cells.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Common simulator target | CPU source hash, event order, parameterization | PASS | FP32/FP64 round-off only | Keep precision distinction explicit | Exact score equality |
| Clipping mismatch | invalid susceptible rate | PASS: 0/163840 | Probe is not a theorem for every cloud | Retain as diagnostic | Clipping can never occur |
| Classifier as oracle | Gaussian exact calibration | HARD VETO | coordinate-specific ratio bias/calibration | Do not promote classifier | Classifier method is invalid |
| GenUT score correctness | exact LGSSM Kalman gate | HARD VETO | nonlinear target has no exact oracle | Repair finite program/score route | GenUT is unusable for every model |
| `j0` ranking | uncertainty-supported comparison | NOT TESTED | only descriptive holdouts/16 GenUT seeds | no ranking claim | repaired permutation is best |

## Inference-status table

| Item | Status |
|---|---|
| Hard veto screen | Upstream classifier and GenUT exact-oracle gates remain failed |
| Statistically supported ranking | None |
| Descriptive-only difference | Classifier holdout `j0` is consistently positive; GenUT is high-variance and sign-balanced |
| Default/HMC readiness | Closed |
| Next evidence needed | A classifier protocol that passes a fresh exact Gaussian oracle and a GenUT route that passes the LGSSM value/score oracle, followed by a truly matched SIR comparison |

## Attempt ledger

- `attempt01`: completed before the clipping probe was corrected; its near-1
  invalid rate came from incorrectly centering states near zero.
- `attempt02`: stopped on a probe dtype error.
- `attempt03`: exposed the need to distinguish GPU bitwise hash drift and was
  superseded after direct replay verified the CPU source hash.
- `attempt04`: computation completed but report serialization used a stale
  field name.
- `attempt05`: precision-aware artifact on the environment's mapped RTX 5080;
  values agree with the terminal run but it was not the requested physical
  GPU.
- `attempt06`: terminal precision-aware artifact on requested physical GPU1,
  RTX 4080 SUPER; values agree with the final run but lacked explicit device
  model fields.
- `attempt07`: stopped on a TensorFlow physical/logical device-details API
  mismatch before artifact creation.
- `attempt08`: final provenance-complete artifact on requested physical GPU1,
  RTX 4080 SUPER; used above.

## Clean restart state

Resume only from the plan, this memo, the terminal `attempt08` artifact, the
existing classifier Gaussian result/reset memo, and the existing GenUT
score-validation-readiness result. Do not treat `attempt01`--`attempt04` as
scientific results. Do not use the SIR online teacher as an exact oracle.

## Nonclaims

No exact SIR observed-data score was established. No classifier admission,
GenUT score correctness, algorithm superiority, default promotion, posterior
correctness, NeuTra, or HMC readiness claim is supported.
