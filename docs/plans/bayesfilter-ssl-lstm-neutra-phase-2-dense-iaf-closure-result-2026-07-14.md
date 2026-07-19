# SSL-LSTM NeuTra Phase 2 Dense-IAF Closure Result

Date: 2026-07-14

Status: `PHASE_2_DENSE_IAF_MATHEMATICAL_CLOSURE_PASSED`

## Outcome

The frozen dense-IAF transport now provides explicit scalar and batch forward,
diagnostic inverse, score pullback, and log-Jacobian-score operations for all
existing schema components: dense autoregressive IAF, mixing linear, diagonal
and dense affine, and nested composition. The implementation uses manual
component VJPs and reverse-order composition; no production score method uses
`GradientTape`.

The persisted payload schema and hash semantics did not change. Existing legacy
payloads continue to load and replay.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept Phase 2 closure | `44/44` final CPU-hidden focused tests plus trusted GPU/XLA canary passed | No direction, transpose, inverse, logdet, composition, finiteness, serialization, integration, or XLA veto | Tests cover bounded known fixtures, not arbitrary learned weights or transport quality | Implement Phase 3 reverse-KL trainer and its analytic controls | NeuTra quality, HMC readiness, posterior correctness, mixing, performance, or scientific validity |

## Mathematical And Engineering Evidence

| Surface | Evidence |
| --- | --- |
| Autoregressive VJP | Manual masked-network reverse pass matches debug autodiff for `tanh`, `elu`, and `relu` fixtures |
| Autoregressive inverse | Coordinate-sequential inverse roundtrips center, shell, and tail rows |
| Logdet score | Manual score matches debug autodiff and directional finite differences |
| Transpose convention | Non-symmetric mixing and dense-affine matrices match independent row-vector formulas |
| Composition | Nested/two-stage forward order, reverse cotangent order, inverse, and batch permutation checks pass |
| Identity/affine | Identity and signed diagonal-affine maps pass exact forward/inverse/pullback checks |
| Fail-closed behavior | Shape mismatch, singular matrices, nonfinite tensors, hash tampering, wrong target, wrong masks, and unsupported components reject |
| Reload compatibility | Reloaded payload reproduces every closure operation exactly; legacy import suite passes |
| Target integration | `FixedTransportValueScoreAdapter` explicit transformed score matches debug autodiff of a known Gaussian transformed value |

Debug autodiff is an implementation test oracle only. The Gaussian fixture is a
known separate problem and is not an SSL-LSTM posterior reference.

## Checks And Runtime Manifest

| Check | Result |
| --- | --- |
| Entry-boundary target/forecast suite | `117 passed` in `1853.35s`; began before Phase 2 edits and is not new-method validation |
| First focused closure run | `35 passed` |
| Integration/legacy focused run | `40 passed` |
| Final focused run | `44 passed` in `8.97s` |
| Compile and whitespace checks | Passed |
| Initial non-trusted GPU attempt | CUDA device unavailable; sandbox evidence only; no artifact written |
| Trusted GPU/XLA canary r1 | Passed; retained as preliminary receipt |
| Trusted GPU/XLA canary r2 | `PHASE_2_DENSE_IAF_GPU_XLA_CANARY_PASSED`; final authority |

Final canary receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-2/dense-iaf-gpu-xla-canary-r2.json`,
SHA-256 `884b26194575d35f3e689ba2032ee378d2494be066485dee91d7b3acf9294ffc`.

The r2 manifest records Git commit
`3d353253dc93a102722e00cbca8803a1b3fce7fa`, dirty worktree, conda `tfgpu`,
Python `3.13.13`, TensorFlow `2.20.0`, two visible RTX 4080 SUPER devices,
output on `GPU:0`, `float64`, TF32 enabled, `jit_compile=True`, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`. Wall time was
`2.7333136070519686s`; fixed deterministic tensors required no random seed.

Maximum eager/XLA residual was `1.3322676295501878e-15` in the inverse, versus
tolerance `1e-11`; all other recorded eager/XLA residuals were zero. The same
value was the maximum forward/inverse roundtrip residual. Outputs were finite
and GPU-resident, and TensorFlow logged actual XLA cluster compilation.

Final source bindings:

| Path | SHA-256 |
| --- | --- |
| `bayesfilter/inference/neutra_artifacts.py` | `757b030ee6357f560cfb3e4ef4cad4e1ec9179cb429b2f7e2ff9b825ce48e391` |
| `tests/test_dense_iaf_neutra_artifact_loader.py` | `463d7a7dfe092f8f98be1e77770c440097cd114e7d742afe972a251031200e6b` |
| GPU canary runner | `6d41bad60ba810cfb7126a1b72d6391e73c0ac5a1054eab55ec40479bb4357ea` |

## Focused Review And Red Team

The focused mathematical review checked the two distinct row-vector transpose
rules, the `s_max*tanh(a/s_max)` derivative, activation derivatives, direct and
indirect autoregressive cotangents, coordinate-sequential inversion, and the
reverse recurrence for composed logdet scores. Missing identity, ELU/ReLU,
reload, and singular-matrix fixtures were added; all checks and the source-bound
GPU receipt were rerun.

Strongest alternative explanation: the fixtures may miss a pathology at an
extreme learned parameter configuration. Phase 3 and Phase 4 must therefore
retain saturation, finite-update, validation-support, and frozen-payload replay
gates. A later learned failure would reject that candidate or trigger a repair;
it would not retroactively make these fixture identities false.

## Handoff

Phase 3 may build a BayesFilter-owned GPU/XLA reverse-KL trainer against this
closed topology. It must prove objective sign/reduction, manual target-score
VJP authority, batch invariance, checkpoint/resume replay, and finite compiled
updates before any material SSL-LSTM training.
