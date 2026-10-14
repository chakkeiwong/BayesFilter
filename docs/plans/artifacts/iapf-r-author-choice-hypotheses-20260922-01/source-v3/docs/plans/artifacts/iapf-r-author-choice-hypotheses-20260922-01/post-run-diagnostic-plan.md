# Bounded failure diagnosis and passive observability repair

The source-v1 campaign remains frozen and runs to its planned decision. The
new dimension-80 weighted fits fail the numerical rank check. Inspect the saved
failing particle clouds and target values to measure effective regression sample
size, concentration, weighted design rank and condition number. The comparator
is the unweighted design on exactly the same saved cloud. These are explanatory
diagnostics, not new candidates, selection criteria or proof about author code.
Preserve file hashes and the actual R command in a separate diagnostic attempt.
Missing source data or a disagreement with a stored diagnostic stops that
diagnostic; it does not invalidate unrelated campaign results.

Code inspection also found that learning-pass floor quantiles were computed but
discarded by the controller when a later fit failed. Retain these existing
values in controller returns and fit errors. This changes neither RNG consumption
nor mathematical operations used in fitting, filtering or stopping. Test both the
failed consumer and exact numerical parity with the preserved core before use.
Keep the original source-v1 campaign and its limitations explicit; richer fields
in a later source version cannot retroactively certify missing diagnostics.

Skeptical review: the saved cloud comparison avoids regenerating easier data;
effective sample size is an explanation, not an accuracy or rank certificate;
the actual weighted QR rank supplies the numerical rank evidence. No solver,
weight exponent, model, tolerance or promotion condition changes. PASS for this
bounded diagnostic and additive output repair. This is a self-review.

Use at most three additional CPU launches and 600 aggregate seconds within the
existing 100-launch/7200-second budget and original deadline. Wait for the two
campaign workers to finish before launching the diagnostic or focused tests.
Record results in diagnostic-failures-v2 and tests-v2; no prior output is replaced.

## Final-review repair: diagnostic loss must not reject a different objective

The saved-cloud audit found one affected result: floor_probe-d20-j2-wlog1-
peak4-sample, replica3. A shared check rejected a weighted-log fit because the
explanatory Equation15 density residual underflowed. That is wrong relative to
F3's declared target and the plan's diagnostic roles. Keep the underflow rejection
for F1/F2, where that loss is optimized; retain it as a reported diagnostic for
QR/weighted-log fits. Their rank, concavity and finite Gaussian validity checks
remain mandatory. No optimizer setting or scientific acceptance criterion changes.

Before retrying, use the saved failure as an executable regression: the corrected
function must return the same previously computed finite positive-variance fit,
with the density-underflow flag preserved. Repeat the existing full focused suite.
Then rerun only the affected full filter with identical data/RNG seed, floor,
controller and numerical settings, in a fresh source-v3 attempt. Preserve the old
failure and explicitly supersede that one record for final aggregation. Recompute
all calibration screens; if selection changes, continue the existing conditional
phases within the unchanged campaign budget. No new candidate is introduced.

Self-review: PASS. This restores the original evidence contract rather than
relaxing a failed scientific threshold. Reserve at most three further launches
and600 seconds for tests, retry and revised report within the original limits.
The revised report must distinguish completed records from all scheduled probes;
timeouts are resource-censored evidence, not numerical failures. Preserve the
original report and selections, and write versioned replacements.
