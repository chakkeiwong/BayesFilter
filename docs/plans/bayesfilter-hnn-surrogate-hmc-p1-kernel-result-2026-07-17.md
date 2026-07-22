# P1 Result: Corrected Neural-Force HMC Kernel

Date: 2026-07-17

Decision: `PASS_P1_KERNEL_MECHANICS_ADMITTED`

## Outcome

The standalone TensorFlow implementation in
`bayesfilter/inference/neural_force_hmc.py` executes a batched frozen
position-only kick--drift--kick proposal with a terminal momentum flip.  It
uses one cached current true potential, evaluates one new deterministic true
endpoint potential, retains both endpoint momenta, and applies

`delta_h = U(q_proposed) + K(p_proposed) - U(q_current) - K(p_initial)`.

The public binding rejects momentum-dependent forces, direct learned state
updates, asymmetric schedules, mutable forces, stochastic endpoint targets,
and transformed-coordinate targets that do not attest the complete chart
log-Jacobian.  A declared `+inf` support boundary is an ordinary rejection;
nonfinite force values, `NaN` target values, `-inf` target values, and
indeterminate energy differences fail closed.

The active leapfrog and sample recursions use `tf.while_loop`; there is no
NumPy, `py_function`, `numpy_function`, Python sample loop, or Python leapfrog
loop in the kernel.

## Checks

- CPU/reference: `15 passed in 7.69s`.
- Trusted GPU/XLA: `2 passed in 6.67s` on NVIDIA GeForce RTX 4080 SUPER.
- TensorFlow: 2.19.1; XLA compilation was observed; TF32 was enabled.
- GPU allocator: repository memory-growth policy reported full preallocation
  disabled and all physical devices configured for memory growth.
- Biased-force Gaussian fixture, 8 chains and 8,000 retained draws after 2,000
  retained warm-up draws: mean `-0.0019179`, variance `0.9957524`, acceptance
  `0.8042656`.
- Source SHA-256:
  - kernel: `dffe1f9be87949742b9a6af3aa7dc2ebef1e2c1baf9ec923d6f869017462910d`;
  - CPU tests: `52a8e714f587d38344ae75bbfc2f0c55c3266185039f7aff1c1f67c92205969b`;
  - GPU tests: `584c26cc38f4128db5789ea07eb35abf82c02ef8b8c352bf0aa2e2d3c0c4d699`.

The Gaussian fixture is a tested reference case.  It is not evidence that an
arbitrary learned force mixes well on a filtering posterior.

## Repair History

| Attempt | Classification | Finding | Repair | Result |
| --- | --- | --- | --- | --- |
| CPU 1 | test/API harness | guard rejection wording differed; a test closure exposed a defaulted second parameter | clarified error and used a true one-argument closure; graph-compiled long chain | CPU 2 passed |
| GPU 1 | harness infrastructure | pytest collection saw no GPU | trusted direct probe confirmed GPU | no kernel executed |
| GPU 2 | harness infrastructure | identical pytest failure repeated | inspected `tests/conftest.py`; found default `CUDA_VISIBLE_DEVICES=-1` | no kernel executed |
| GPU 3 | focused harness repair | launched with existing `BAYESFILTER_TEST_DEVICE_SCOPE=visible` opt-in | none | both XLA canaries passed |

The GPU failure did not invalidate CUDA, TensorFlow, the target, or the kernel.
It was the intended repository test isolation policy applied without its GPU
opt-in.

## Decision Table

| Field | Status |
| --- | --- |
| P1 primary criterion | passed |
| Map reversal/involution | passed |
| Unit Jacobian fixture | passed |
| Full joint endpoint energy | passed |
| Endpoint cache/call count | passed: one new endpoint call |
| Transformed target boundary | passed, omission rejected |
| Undefined numerical execution | fail-closed tests passed |
| GPU/XLA route | passed with memory growth |
| Main uncertainty | useful target-specific scalar-force training is not yet established |
| Next justified action | P2 scalar-force training and campaign harness |
| Not concluded | no filtering-model HNN validity, convergence, performance, superiority, or default readiness |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | clear for P1 tested mechanics |
| Statistically supported ranking | none |
| Descriptive-only differences | acceptance and moment estimates from one analytic fixture |
| Default readiness | false |
| Next evidence | target-specific training, frozen reload, disjoint tuning/archive, Tier A dry runs |

## Post-Run Red Team

The strongest alternative explanation for the Gaussian moment pass is that the
fixture is unusually easy and eight parallel chains still share one fixed
configuration.  This does not weaken the exact MH argument, but it limits the
empirical claim to this fixture.  A later model result can still fail through
poor training, poor tuning, or low mixing.

The weakest P1 boundary is callable introspection: it prevents an explicit
second argument but cannot prove that an arbitrary Python closure has no hidden
mutable state.  Serious P2 artifacts therefore freeze weights and identity,
and retained sampling must load that frozen artifact rather than an arbitrary
mutable training object.

## P2 Handoff Review

The P2 audit found that BayesFilter already has typed NeuTra target identities,
frozen transport loaders, separate sample archives, and rank-normalized
split/folded R-hat plus bulk/tail ESS.  P2 will reuse those contracts.  The
missing component is a target-specific scalar residual-potential trainer and
frozen force loader bound to the admitted P1 kernel.

The P2 plan retains loss/heldout checks as nomination or veto diagnostics only;
only corrected downstream chains may establish sampler validity.  Every
trusted pytest command must set `BAYESFILTER_TEST_DEVICE_SCOPE=visible`.
No true continuation veto fired.

Status: `CONTINUE_TO_P2`.

