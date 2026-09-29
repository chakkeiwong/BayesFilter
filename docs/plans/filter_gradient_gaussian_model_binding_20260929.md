# Gaussian analytical model binding inside XLA

Prepared follow-up after the serialized DZ5 constant-sinking diagnostic closes.
Adaptive iAPF remains deferred. This bounded repair addresses the ordinary
Gaussian Kalman/UKF endpoint and its Gaussian oracle: `adapters.py:41,79`
currently calls `parameterized_model` eagerly and passes twelve model/tangent
arrays to an otherwise compiled recursive filter. Compiling the recurrence
alone leaves parameter transforms and initial-law sensitivities outside XLA.

Add one cached, fixed-signature theta/observation owner that calls the existing
`gaussian_tf.parameterized_model` and `make_gaussian_kernel` authorities inside
`tf.function(jit_compile=True)` by default. Keep the same filter equations,
initial derivatives, precision, validity checks, public result schema and
non-JIT diagnostic option. Wire non-iAPF Gaussian oracle and Kalman/UKF calls to
this owner. Preserve the deferred iAPF path. Optional KDM pilot controls and
common eager random-cloud preparation remain separately recorded work, not
completion claims from this unit.

Baseline: exact source from commit `ea5319db9`, including the original eager
model preparation and already compiled recurrence. Freeze source hashes in
test evidence. Inputs remain actual tensor operands. Use dimensions1/3,
observation dimensions1/2, short horizons, both Kalman/UKF, FP64/FP32 and changed
theta/observations. Check all five outputs, covariance positivity and replay;
retain one trace and enclosing HLO without pfor or host callbacks. FP64 complete
outputs must satisfy the existing score-study1e-9 absolute/relative comparison;
FP32 uses1e-5 absolute/relative. Independently validate FP64 analytical scores
with two five-point finite-difference steps and1e-7 absolute/relative bounds.
Do not count this diagnostic derivative as the runtime analytical score.

Qualify both actual public endpoints with live dataset seeds and the same
physical generated observations, comparing every non-runtime field with the
archived original adapter. Only timing/memory/device-placement provenance is
excluded from numerical comparison. Exercise changed parameters and exact
replay, default JIT, explicit graph reference and current host error behavior.
Tests may use NumPy as an independent diagnostic comparator only.

After numerical CPU/GPU checks, run a descriptive fresh-process CPU cost screen
against the original endpoint with synchronized calls, source/environment/input
identity, compilation separately from warm execution, and sampled RSS plus
allocator data where available. If GPU hardware remains uncontended, run the
same two descriptive cost arms there under the runner's cost preflight. A single
process per arm cannot supply statistical ranking or cost acceptance, or explain
the separate native/compiler-residency findings.

Reserve at most14 serialized workers,3000 CPU/1800 GPU process-seconds, within
the unchanged56/52-hour campaign cap. Numerical groups are
`gaussian_binding_{float64,float32}_{cpu,gpu}`,
`gaussian_binding_public_{cpu,gpu}`, two CPU cost arms, two conditional uncontended GPU cost arms and one combined
readback; remaining slots cover bounded harness repairs. Use the stable campaign runner
and fresh raw output directories. GPU checks require trusted placement and
verified memory growth under the user's display-device rule. Activate this
allocation only after the current unit closes. No simultaneous numerical jobs.

Veto numerical/status/seed drift, retracing on same-shape operands, callbacks,
missing evidence or source mismatch. A failure triggers localization under the
same remaining allocation, never tolerance relaxation or a scalar fallback.
Preserve and charge failures. No training/HMC, canonical LEDH claim, package
change, live MacroFinance edit, expanded allowance or main merge.

Skeptical primary-agent review: enclosing model transforms may change floating
fusion and their derivative tensors. Direct comparison to the eager model plus
the original compiled recurrence, independent finite differences and actual
public callers address that risk. Data generation is unchanged and cannot be
claimed repaired by this unit. Small well-conditioned cases do not establish
arbitrary conditioning or full repository compliance. No independent reviewer
is used.
