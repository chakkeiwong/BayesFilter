# Model B confirmation: host diagnostic repair

Attempt B-probe-01 failed after the compiled filter returned, in an eager GPU
`tf.cast` on its reset-fallback diagnostic. CUDA reported INVALID_HANDLE.
This is a harness/device-compatibility failure, not evidence against the filter
or the model. The numerical filter code and controls are unchanged.

Repair: copy the existing validity, step-count and fallback arrays to the host
and aggregate them with NumPy, alongside the JSON serialization that already
requires host data. Keep all scientific kernels compiled on GPU. The focused
controller now accepts the existing source-migration note mechanism. Generic
migration preserves spent CPU/GPU budget and preparation receipts, invalidates
dependent results, and reruns references and GPU checks; no old result is
deleted. The controller and worker hashes change; target numerical code does not.

Review: this is observability-only bookkeeping. No score, likelihood, proposal,
data, sampling criterion or budget changes. Re-run the focused controller
regressions and the actual production GPU probe before training. A successful
probe must still have all steps, finite validity and exact repeatability.
