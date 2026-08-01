# P5 R1B Dataset-Device Repair Record

Date: 2026-07-16

GPU attempt 01 failed at the frozen-dataset hash preflight before compiling the
posterior or issuing an identity. TensorFlow stateless random generation was
executed under the active GPU device and did not reproduce the CPU-owned
serialized dataset hashes. Classification:
`HARNESS_DEVICE_DEPENDENT_DATASET_REPLAY`.

The repair wraps frozen dataset replay in `tf.device("/CPU:0")`. This enforces
the BayesFilter policy that external NeuTra datasets are CPU-generated and
makes the data identity independent of the later target-computation device.
Model equations, TensorFlow RNG, seed, horizon, truth, dtype, expected hashes,
posterior, audit points, thresholds, and hardware class remain unchanged.

Because the typed identity binds repository source dependency closure, CPU
attempt 01 is superseded after this source repair even though its mathematical
and numerical checks passed. It remains preserved as historical evidence but
must not be used for later training or HMC. CPU attempt 02 must issue the
post-repair identity in a fresh root; GPU attempt 02 must reproduce that exact
signature.
