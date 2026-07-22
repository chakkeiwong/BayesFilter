# P2 Repair Record

Date: 2026-07-15

## SVX-SGQF Target Admission Attempt 01

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p2/SVX-SGQF/target-admission/attempt-01-20260715T102616Z/`

Classification: `INFRASTRUCTURE_GPU_MEMORY_POLICY_ORDER`.

The launcher imported TensorFlow Probability and BayesFilter target modules
before programmatically configuring memory growth. Those imports materialized
TensorFlow constants and initialized the logical GPU. The environment variable
`TF_FORCE_GPU_ALLOW_GROWTH=true` did enable allocator growth, but the repository
helper correctly failed because it could no longer verify configuration before
initialization. No dataset replay, level evaluation, identity issuance, HMC, or
training ran.

Repair: import TensorFlow, immediately call and verify
`configure_tensorflow_gpu_memory_growth`, enable the recorded TF32 flag, and
only then import TFP and BayesFilter modules. The target, levels, thresholds,
data, hardware, and budget remain unchanged. Retry in a fresh attempt-02 root.

## SVX-SGQF Target Admission Attempt 02

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p2/SVX-SGQF/target-admission/attempt-02-20260715T102856Z/`

Classification: `INFRASTRUCTURE_DATA_GENERATION_DEVICE_PLACEMENT`.

Memory-growth verification passed, but the frozen dataset replay hash failed
before any candidate level executed. The original seed-81101 trajectory was
generated on CPU. With a visible GPU, the stateful TensorFlow generator and
dependent recurrence defaulted to GPU, whose random-number stream did not
bitwise match the preserved CPU artifact.

Repair: explicitly pin the entire dependent trajectory generator, TensorArrays,
random draws, and recurrence to `/CPU:0`. This follows the repository policy
that external dataset generation is a separate CPU lane. The preserved data
hash remains the authority; do not accept the GPU-generated substitute. Retry
in a fresh attempt-03 root after visible-GPU hash replay passes.

## SVX-SGQF Target Admission Attempt 03

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p2/SVX-SGQF/target-admission/attempt-03-20260715T103435Z/`

Classification: `INFRASTRUCTURE_AUDIT_POINT_DTYPE`.

The visible-GPU frozen-data replay and memory-growth checks passed. The launcher
then failed before any SGQF level executed because the transformed-truth audit
point inherited TensorFlow Probability's `float32` distribution default while
the fixed audit matrix was `float64`. TensorFlow correctly rejected the mixed
dtype concatenation. No level result, identity, HMC, or training was produced.

Repair: construct the standard-normal distribution, truth probabilities, and
fixed audit matrix explicitly in `float64`, and regression-test both the dtype
and the recovered physical truth point. The target, data, ladder, thresholds,
hardware class, and remaining campaign budget are unchanged. Retry in a fresh
attempt-04 root after the focused CPU regression passes.
