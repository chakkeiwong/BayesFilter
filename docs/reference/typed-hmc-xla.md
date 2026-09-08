# Typed proposal-field HMC with XLA

The public `tune_hmc_kernel` route accepts `use_xla=True` in
`TensorFlowHMCKernelTuningConfig`. This applies XLA to the enclosing stable-
signature tuning graph, including windowed metric updates, dual averaging,
candidate selection and fresh verification. The retained archive runner uses
the same compilation mode, returns numerical tensors, and writes its eleven
archive tensors outside the graph. The transition mathematics is unchanged.

The historical `use_xla=False` constructor default remains for compatibility;
it is not an XLA-qualified default. Explicitly set `True` for consumers that
require full-chain XLA. `xla_mode=requested_full_chain_compilation` and
`xla_qualification=requires_external_qualification` distinguish a compilation
request from evidence that a particular target compiles and preserves its
values. Serialization and reload preserve these fields. Continuation rejects
a changed mode; an older archive with no mode is compatible only with a
non-XLA tuning result.

## Scoped qualification

`tests/test_tensorflow_hmc_tuning_xla_gpu.py` is an opt-in trusted single-GPU
regression. It checks actual enclosing XLA execution and HLO, deterministic
repeat, synthetic nonexact position-only field semantics, failed-candidate
handoff refusal, artifact reload, retained replay and predecessor continuation.
It requires `BAYESFILTER_TEST_DEVICE_SCOPE=visible`, exactly one selected
`CUDA_VISIBLE_DEVICES=0` or `1`, and `TF_FORCE_GPU_ALLOW_GROWTH=true` before
TensorFlow import. It verifies growth before logical-device initialization.

On September 8, 2026, the clean candidate based on `d2124d425b0ea0ae0e3e5f4246bd6b03ff8a2170`
passes this regression under TensorFlow 2.20.0 on GPU1, with native-operator
SHA-256 `2feb93135f2359e60aca446f52f8468b104199b4d398d4fe6057030df20f23d6`.
The preserving record is in the BGS consumer repository at
`docs/experiments/bgs/2026-09-07-master-program-repair/qualification-01/`.
This establishes synthetic compilation/replay behavior only. Canonical BGS
qualification remains separate; its endpoint code initially exposes unsupported
eigenvalue/determinant operations and an unregistered CUDA custom call.

XLA may discard TensorFlow assertion operators. A target must not rely solely
on those assertions for invalid-proposal classification; review its returned
status/finite-value semantics and native-op behavior. Passing compilation,
acceptance or this tiny regression does not establish posterior convergence,
exact-score correctness, sampler superiority or scientific admission.
