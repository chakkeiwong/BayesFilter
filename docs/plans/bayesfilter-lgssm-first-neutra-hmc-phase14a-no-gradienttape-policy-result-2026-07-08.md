# Phase 14A Result: LGSSM No-GradientTape Target Policy

Date: 2026-07-08

## Scope

This result closes the Phase 14A no-`GradientTape` policy repair slice for the
LGSSM-first NeuTra/HMC program.

The phase did not run NeuTra training, HMC sampling/tuning, external sample
generation, GPU/XLA diagnostics, DSGE/c603, route ranking, default-policy
changes, or scientific/product/readiness claims.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | `PASS_PHASE14A_NO_GRADIENTTAPE_POLICY_REPAIR` |
| Primary criterion status | Passed: the admitted LGSSM generic target score route now uses the analytical QR Kalman score path instead of TensorFlow tape autodiff, and the fixed affine transport wrapper now uses explicit transport score pullbacks. |
| Veto diagnostic status | Passed for the scoped repair: focused tests pass, source guards pass, no training/HMC/sample generation ran. |
| Main uncertainty | Old Phase 10/11/12 artifacts were produced under the previous taped-score target signature and are now historical diagnostics only. |
| Next justified action | Draft a new GPU-training preflight/rerun subplan under the manual-score target signature before regenerating NeuTra artifacts. |
| What is not concluded | HMC convergence, posterior correctness, transport quality, XLA readiness, production readiness, sampler quality, or scientific validity. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the admitted LGSSM target and fixed affine transport mechanics stop using `GradientTape` while preserving finite value/score behavior and analytical parity? |
| Baseline/comparator | Existing analytical QR score route, finite-difference tests, fixed affine transport chain rule tests, and stale Phase 10/11 artifacts as diagnostic history only. |
| Primary criterion | Focused tests pass and source guards show no tape markers in admitted LGSSM target helper or fixed-transport runtime wrapper. |
| Veto diagnostics | Nonfinite values/scores, analytical mismatch, hidden tape use in admitted route, silent promotion of stale artifacts, HMC/training/sample execution. |
| Explanatory diagnostics | Target signature, adapter signature, source scan, focused pytest. |
| Artifact | This result note plus changed source/tests. |

## Implementation Summary

- `bayesfilter/testing/lgssm_generic_target_adapter_tf.py`
  - changed LGSSM score source metadata from taped QR likelihood to analytical
    QR Kalman score;
  - replaced the taped score helper with
    `QRStaticLGSSMTarget.analytic_score_hessian`.
- `bayesfilter/inference/neutra_artifacts.py`
  - added explicit affine transport score pullbacks and zero logdet-score
    methods.
- `bayesfilter/inference/batched_value_score.py`
  - removed tape-based transport differentiation from
    `FixedTransportValueScoreAdapter`;
  - now requires `pullback_score`, `pullback_score_batch`,
    `log_abs_det_jacobian_score`, and
    `log_abs_det_jacobian_score_batch` for fixed transports.
- Tests now guard no-tape admitted routes and fail closed when fixed transports
  lack explicit pullbacks.

## Signature Impact

The no-tape policy intentionally changes the LGSSM target/adapter signatures:

| Item | Old taped-signature artifact | Current no-tape signature |
| --- | --- | --- |
| Target | `290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb` | `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038` |
| Adapter | `0a48b43d2750cad5b452708f7619a1119a259231d8955769809460f256575a97` | `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900` |

Consequence: existing Phase 10 GPU training, Phase 11 frozen affine payload,
and Phase 12 CPU sample-boundary artifacts tied to the old signatures must not
be promoted as current no-tape target evidence. They remain historical
diagnostics only unless regenerated under the current signatures.

## Local Checks

- `python -m py_compile bayesfilter/inference/batched_value_score.py bayesfilter/inference/neutra_artifacts.py bayesfilter/testing/lgssm_generic_target_adapter_tf.py tests/test_batched_value_score.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_gpu_affine_payload_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_batched_value_score.py tests/test_neutra_artifact_loader.py tests/test_lgssm_fixed_transport_mechanics_tf.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_gpu_affine_payload_tf.py tests/test_neutra_cpu_sample_boundary.py tests/test_neutra_xla_repair_tf.py -q`: passed, `59 passed, 2 warnings`.
- `rg -n "GradientTape|batch_jacobian|tape\\." bayesfilter/inference/batched_value_score.py bayesfilter/testing/lgssm_generic_target_adapter_tf.py`: no matches.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_batched_value_score.py::test_fixed_transport_value_score_adapter_uses_no_gradient_tape tests/test_lgssm_generic_target_adapter_tf.py::test_lgssm_generic_target_admitted_score_route_uses_no_gradient_tape -q`: passed, `2 passed, 2 warnings`.
- `git diff --check` for changed Phase 14A files: passed.

CPU-hidden tests are support checks only. They are not GPU/XLA evidence and not
readiness evidence.

## Stale Artifact Guard

`tests/test_neutra_gpu_affine_payload_tf.py` now records that the old Phase 10
GPU artifact is stale after the no-tape target policy. The packaging path must
reject it with a target-signature mismatch instead of silently repackaging it
as current evidence.

## Nonclaims

- No NeuTra training was run.
- No HMC sampling or tuning was run.
- No external sample generation was run.
- No GPU/XLA diagnostic was run.
- No DSGE/c603 target was used.
- No XLA readiness, HMC readiness, posterior correctness, sampler quality,
  production readiness, default-policy change, or scientific validity is
  claimed.

## Next Handoff

The next phase should be a reviewed GPU preflight/rerun plan for the current
manual-score LGSSM target signatures. It should not reuse the stale Phase 10/11
payloads for promotion.
