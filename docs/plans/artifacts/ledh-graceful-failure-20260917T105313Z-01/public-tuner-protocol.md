# Public HMC binding and telemetry diagnostic

Question: can the real LG1 finite target reach the registered public tuner,
with endpoint support/numerical/force outcomes preserved through every attempted
adaptation and verification transition and artifact serialization?

The source audit found that the TensorFlow tuner is explicitly diagnostic-only,
rejects XLA, and discarded the kernel's already computed endpoint statuses.
Preserve these statuses without changing acceptance, candidate selection or
admission. Retain mass/step adaptation, failed candidates and repair rounds;
skip unexecuted rounds. Older diagnostic artifacts may have no trace: do not
manufacture historical telemetry. No canonical tuning authority is created.

Use deliberate CPU-only float64 graph execution, a documented small reference
exception to the GPU/XLA default. Run the real LG1 fixture with N16,T10 on
calibration replicate 0, initial unconstrained coordinate zero, scale 0.1;
one mass transition, one step adaptation, one verification, L=1, step=0.01.
These are mechanics-only convenience settings; acceptance bands [0,1] test
wiring, never tuning, convergence, approximation or posterior correctness.
Limit the public binding diagnostic to 300 seconds; tests to 300 seconds.
Stop on nonfinite accepted state, wrong target identity, missing trace,
serialization mismatch, or a diagnostic result falsely admitting retention.

Validate trace preservation with synthetic typed status in focused tests,
including an attempted repair and a skipped candidate. Run the existing
dispatcher tests for compatibility. Artifacts: public-tuner-cpu-01 and
tuner-telemetry-tests-cpu-01 (fresh suffix on infrastructure retries).
The real LG1 accuracy attempts are exhausted; this independent diagnostic
does not add a candidate or consume heldout data. HMC/posterior admission
remains unready regardless of diagnostic success.
