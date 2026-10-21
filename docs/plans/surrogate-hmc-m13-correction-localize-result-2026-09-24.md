# M13 correction arithmetic result — 2026-09-24

The exact-input replay reproduces controls-07: two of nine nominal rows fail
correction at time zero under FP32/TF32. Turning TF32 off does not remove the
failures. Lifting every realized FP32 input and coefficient to FP64 produces
finite values, forces and filtering moments for all nine rows. This supports
arithmetic sensitivity, not posterior validity or an FP64 default.

Evidence: `artifacts/surrogate-hmc-closure-20260922-m13-controls-10/`.
The master executed `--stage closure --closure-stages m13-correction-localize`
with that output directory using the tftwogpu interpreter. Its review passed
33 tests. The GPU/XLA worker verified memory growth and saved the unchanged
canonical source hashes, saved inputs, seed1703/partition6 binding, all statuses
and all correction diagnostics. Scope is N256/T20, calibration attempt3 only.

| Arithmetic | Healthy nominal rows | Failure |
| --- | --- | --- |
| FP32/TF32 default | 7/9 | First two, correction at time0 |
| FP32 without TF32 | 7/9 | Same first two, correction at time0 |
| Exact FP64 lift | 9/9 | None in these rows |

The first two rows have filtering ESS approximately one. Their last-coordinate
fourth moments are nonfinite in both FP32 modes; the FP64 results are
2.90635e19 and 1.38539e19, within FP32's representable range. The shared moment
routine computes an unweighted fourth power before multiplying by the weight.
This can overflow although the weighted result is finite. A direct primitive
regression and an exact-input replay after reordering multiplication will test
whether that mechanism explains these failures. FP64 has different upstream
roundoff, so its success alone does not prove the precise operation responsible.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Diagnostic complete | Three bound-input arms and complete diagnostics | All failures typed; saved source/input checks pass | Root operation and geometry after repair | Test algebraically unchanged weighted multiplication order | Control selection, endpoint or HMC admission |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | FP32 correction invalidity supported for two rows |
| Statistically supported ranking | None; arithmetic localization only |
| Descriptive differences | Values, scores and moments differ across precision |
| Default-readiness | FP32 remains owner default; no new promotion |
| Next evidence needed | Primitive value/JVP parity, healthy-case non-harm, saved-input replay |

Controls-09 failed before evaluation because JSON converted seed keys to
strings; its review passed32 tests. The normalized serialization check was
regressed and retried in controls-10. Charges are 11.846571 CPU/5.506622 GPU
seconds for09 and143.155233 CPU/94.135296 GPU seconds for10. Cumulative campaign
usage:45177.565653292 CPU/35204.231887523 GPU seconds. Controls pool remaining:
197.444047 CPU/2403.442064 GPU seconds; global remaining8822.434347 CPU/
191595.768112 GPU seconds. Worker wall time94.135296 seconds; precise receipts
and both source snapshots are preserved.

Post-run red team: extreme weighted-cloud collapse remains even if overflow is
repaired. Representable moments may still be impossible for an equal-weight
cloud to match. This is a proposal-force robustness investigation, not evidence
that a finite likelihood matches the model posterior. A repair that changes
moment targets, adds clipping, or changes controls requires a separate explicit
numerical assessment; it cannot be hidden in arithmetic reordering.
