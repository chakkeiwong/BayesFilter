# M13 likelihood-gap localization — 2026-10-04

Question: why does the correctly differentiated finite LEDH likelihood have
such different slopes from the frozen importance acceptance likelihood at the
eight measured M13 points? This is a bounded explanatory continuation of the
score measurement, within its remaining 1,678.16-second allocation. No HMC,
control selection, algorithm change, or default promotion is planned.

Use the same data1704/T20, 4,096 frozen particles, canonical Contract-E reset
and selected corrections, rounded model inputs, and frozen post-sync source
snapshot as `surrogate-hmc-m13-score-before-after-20261004-01`. Run one GPU/XLA
FP64 diagnostic, with verified memory growth and a 600-second timeout. FP64 is
an explicit reference exception that isolates approximation from FP32 rounding.
Output: `artifacts/surrogate-hmc-m13-likelihood-gap-localization-20261004-01/`.

The consumer call chain is M13 target -> canonical batch consumer ->
`canonical_value_and_analytical_score`. Invoke that same final function and
single-cloud model adapter with trace outputs, reducing those outputs to
per-observation likelihood increments, score increments, particle ESS, maximum
weights, prior importance normalizers, and covariance/flow displacement scales.
Check value/score parity against the saved full consumer FP64 results before
interpreting the trace. This is observability, not a replacement implementation.

Compare the first observation with independent two-dimensional Gaussian
quadrature at 512 and 1,024 intervals and central differences h=3e-4,1e-4.
This integrates the first-step position marginal under the same model; it is
not an oracle for the full 20-observation score. Report grid and step changes
and the existing quadrature tail bound. All eight fixed points have bearing
standard deviations inside the quadrature's declared support.

Evidence contract and intent: the candidate mechanism is proposal/weight
degeneration when the UKF covariance greatly exceeds the process covariance,
possibly compounded by later reset errors. Primary diagnostic criterion is
localization of likelihood/score error and weight concentration at the same
points; it cannot promote a sampler. Nonfinite/status failures, missing source
or input parity, or timeout stop interpretation. Poor ESS, covariance mismatch,
and a normalizer far from one explain a failed approximation, not the rejection
of LEDH or surrogate HMC as research directions. No stochastic ranking, full
model oracle claim, causal attribution to reset alone, or repair claim follows.

Default audit: points, data, particles, controls and source are frozen from the
completed comparison (baseline). FP64 and trace outputs are diagnostic choices;
graph changes may affect rounding, checked by parity. Quadrature grids and FD
steps are refinement hypotheses, checked against one another. The eight points
are visited-state probes, not an independent posterior sample. The cheap
adversaries are the identity-flow prior normalization (exact integral one),
the independent first-step marginal quadrature, and the already saved fixed
importance acceptance reference. Evaluate conditionally at each point, with
special attention to low process noise; no pooled average hides the tail.

Skeptical audit passed: the weight code includes transition/proposal density
ratios and the flow Jacobian, so a poor flow is not by itself proof of biased
weights. The proposed trace tests whether finite integration is actually poor.
The independent T1 comparison can separate a first-step problem from later
reset accumulation. Uncertain full-T20 reference derivatives remain uncertain.
An accurate own-program derivative cannot certify the intended model score.
Preserve command, provenance, complete log, wall time and result/decision tables.

Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/plans/artifacts/surrogate-hmc-m13-likelihood-gap-localization-20261004-01/diagnose.py`.
The script supervises one identical command with `--worker` and enforces the
600-second limit. It imports the completed diagnostic's frozen source snapshot.
