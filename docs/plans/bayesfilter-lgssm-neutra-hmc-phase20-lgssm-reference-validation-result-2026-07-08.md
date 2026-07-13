# Phase 20 Result: LGSSM Reference HMC Validation

Date: 2026-07-08

## Scope

Phase 20 ran bounded CPU-hidden multicore fixed-transport LGSSM NeuTra HMC
chains with `jit_compile=True` and compared retained parameter-coordinate
sample summaries against a deterministic 2D quadrature reference posterior over
the exact LGSSM Kalman likelihood target.

This is a narrow fixture-local validation. It is not a sampler superiority,
production readiness, default readiness, nonlinear SSM, DSGE/c603, broad
NeuTra, or scientific validity claim.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | `PASS_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION` |
| Primary criterion | Passed |
| Veto diagnostics | No veto fired |
| Main uncertainty | Short chains, R-hat/ESS unavailable, covariance residual passed by a narrow margin |
| Next justified action | Continue to Phase 21 readiness classification |
| Not concluded | Sampler superiority, optimal tuning, production/default readiness, nonlinear SSM validity, DSGE/c603 validity, broad HMC validity, or scientific validity |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | Not recorded in artifact; dirty worktree present |
| Command | `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_lgssm_reference_hmc_validation_tf ... --jit-compile true` |
| Environment | local `tf-gpu` Python environment |
| CPU/GPU status | Chain/sample generation CPU-hidden with `CUDA_VISIBLE_DEVICES=-1` |
| Seed | `20260707` |
| Workers/chains | 2 workers, 4 chains |
| Retained samples | 256 total |
| Burn-in/results/leapfrog/step | 64 burn-in, 64 retained per chain, 4 leapfrog steps, step size 0.05 |
| Wall time | 125.59709774400108 seconds |
| Artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase20_lgssm_reference_hmc_validation_seed20260707.json` |
| Artifact file SHA-256 | `094962f3fd8dbd5002ef5d92e42e23ae34cc52dc234ad836d78fe4edd768e188` |
| Stable artifact hash | `sha256:07404fd92a4b5e69449088c8852392241fdbbc9cb61eea91ceb4b2b235f6d553` |

## Evidence

Phase 20 preserved the inherited signatures:

- target signature: `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038`
- source adapter signature: `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900`
- Phase 18 fixed-transport adapter signature: `db6b58a7adc8190f5ed2e48e42482956d32faf02bdf10a7104659a2bd86722c9`
- Phase 20 scoped base adapter signature: `733b68d9ee7b549c8930e2f4bd88bfafbae21b0343421e28ca3fd7c9715ab224`
- Phase 20 scoped fixed-transport adapter signature: `4b93044d685a40a5fef432741c7b7fb22beed937a02155f631fd1a03bcebe8af`
- transport hash: `f1780d9eb8ae0f6d5e6865da6dbb3d1d1a22c4c2e5c89beb60c1f887c5f48fc7`

Reference posterior:

- reference type: deterministic 2D quadrature;
- reference hash: `sha256:22ba481cd336adc4fdc6194c6d09bf01ee62ee1db85598bb524cc7bb47b1b614`;
- reference mean: `[0.04746681088897498, -1.4328721077927253]`;
- reference covariance:
  `[[0.7761135526402296, -0.013481312922316373], [-0.013481312922316368, 0.4323110025371811]]`.

Sample summaries:

- sample mean: `[-0.0999019666751822, -1.6683776461258473]`;
- sample covariance:
  `[[0.12633981112782028, 0.06064836388459259], [0.06064836388459259, 0.23699868278630182]]`;
- mean max absolute residual: `0.23550553833312193` versus tolerance `0.35`;
- covariance max absolute residual: `0.6497737415124094` versus tolerance `0.65`;
- worker acceptance rates: `[1.0, 1.0]`.

The covariance residual passed by only about `0.0002262584875906`. This is a
fixture-local pass with tight margin, not broad HMC convergence evidence.

## Veto Screen

| Diagnostic | Status |
| --- | --- |
| CPU-hidden chain generation | Passed, artifact records `cuda_visible_devices="-1"` |
| `jit_compile=true` | Passed |
| `jit_compile=false` runtime | Not run |
| Hidden training | Not run |
| GPU sample generation | Not run |
| Worker return codes | Passed, both workers returned 0 |
| Finite samples and target values | Passed |
| Reference posterior present | Passed |
| Posterior residuals | Passed, with tight covariance margin |
| R-hat/ESS | Unavailable in this bounded helper; not used as promotion evidence |

## Repair Note

The first Phase 20 attempt was interrupted after it remained active without
writing an artifact. The likely issue was procedural: the parent process
computed the TensorFlow quadrature reference before forking HMC workers. The
helper was repaired to use spawned worker processes and compute the reference
posterior outside the parent process. The repaired run used the same
`jit_compile=True` policy and did not run a non-JIT fallback.

## Local Checks

- `python -m py_compile bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py tests/test_neutra_lgssm_reference_hmc_validation_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_lgssm_reference_hmc_validation_tf.py tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py -q`: passed, `19 passed, 2 warnings`.
- Phase 20 source scan for `GradientTape`, `batch_jacobian`, `tape.`, and `jit_compile=False`: passed.
- `python -m json.tool` on the Phase 20 JSON: passed.
- Phase 20 JSON field validation: passed with `failed=[]`.
- `git diff --check` on Phase 20 helper/tests/subplan: passed.

## Handoff

Phase 21 may proceed. It must classify this evidence using exactly one of:

- `LGSSM_REFERENCE_HMC_READY`;
- `BLOCKED_FOR_REPAIR`;
- `INSUFFICIENT_EVIDENCE_NO_PROMOTION`.

Any Phase 21 readiness claim must remain local to this LGSSM fixture, these
Phase 17-20 artifacts, CPU-hidden sample generation, and `jit_compile=True`.
