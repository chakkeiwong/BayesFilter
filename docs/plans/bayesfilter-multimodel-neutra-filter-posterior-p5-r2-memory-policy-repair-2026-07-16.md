# P5 R2 Memory-Policy Repair Record

Date: 2026-07-16

Plain-HMC attempt 01 failed before identity replay, target compilation, probes,
or sampling. TensorFlow-dependent campaign modules were imported before the
runner called the explicit GPU memory-growth verifier, so the runtime was
already initialized. `TF_FORCE_GPU_ALLOW_GROWTH=true` prevented full
preallocation, but the verifier correctly failed closed because it could no
longer prove programmatic configuration order.

Classification: `HARNESS_GPU_MEMORY_POLICY_IMPORT_ORDER`.

The repair imports TensorFlow, immediately configures and verifies memory
growth, then imports TensorFlow Probability and campaign/target modules. The
target, identity, HMC grid, seeds, diagnostics, caps, hardware class, and budget
are unchanged. Attempt 02 must use a fresh root.
