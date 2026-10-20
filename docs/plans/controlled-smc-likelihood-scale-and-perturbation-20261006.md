# Likelihood scale and perturbation-calibration interpretation, 6 October 2026

This note preserves the earlier conditional-refit calculation. The completed
[full-recursion perturbation experiment](controlled-smc-perturbation-result-20261006.md)
now provides the primary evidence for autonomous source/port differences.

The earlier 187–207 log-likelihood differences were end-of-observation-series
differences pooled across all refinement stages, including early stages.
They were not differences at the last available refinement in each case.
At the six last captured stages, the actual likelihood-estimate differences
are much smaller:

| TF fit | Signed likelihood difference range | Median absolute likelihood difference | Largest absolute log-likelihood difference |
|---|---:|---:|---:|
| Raw normal equations | -2.3973% to +1.0589% | 0.3441% | 0.024265 |
| Centered QR | -1.8542% to +1.0993% | 0.3388% | 0.018716 |

For a source log estimate ell_A and port log estimate ell_T, the signed
percentage is 100 * expm1(ell_T - ell_A). Common multiplicative likelihood
constants cancel. This is not 100*(ell_T-ell_A)/ell_A, whose denominator
depends on additive log-density normalization.

Each result is one TF refit from a saved author cloud followed by a forward
pass through the full observation series with matched random draws. It is
not an independently evolved TF adaptive run. Cases use three forcing values,
two seeds each and different source-selected refinement counts. Six case
comparisons and dependent intermediate stages do not establish a solver
ranking or statistical non-inferiority. Early-stage disagreements remain
relevant because they could change an independently evolved trajectory.

## Owner-proposed random-draw perturbation

The proposed 1e-8 perturbation was subsequently executed as a source-sensitivity
calibration: see [the result](controlled-smc-perturbation-result-20261006.md).
The discussion below preserves the pre-run interpretation; it is not an acceptance threshold.
With observations, parameters and settings fixed, define
S_A(u,delta) = abs(ell_A(u+delta)-ell_A(u)) and
D(u) = abs(ell_T(u)-ell_A(u)).
Compare D with a predeclared summary of repeated source perturbations,
separately by regime and refinement stage. Also measure the TF response
S_T to the same perturbations. These are sensitivity changes, not errors
relative to an unavailable exact likelihood.

Use multiple independent, symmetric perturbation directions for the saved
normal and resampling-uniform draws; keep uniforms strictly in their valid
domain. Perturb the complete fit/forward sequence, with the same data.
A final-forward-only perturbation cannot calibrate fitting instability.
Separate normal-only, uniform-only and combined perturbations to distinguish
smooth sensitivity from resampling changes. Track failures, stopping counts
and the first ancestry divergence. An intrinsic source failure remains a
reference outcome; an additional port failure must be investigated.

The owner-suggested 1e-8 magnitude is the primary hypothesis. A small
magnitude ladder around it can check whether it measures a stable local
response or a resampling threshold. The comparison statistic, perturbation
count, uncertainty treatment and margins must be frozen using calibration
cases before testing new cases. Do not choose the largest observed
perturbation as an automatically permissive acceptance threshold.

This experiment measures sensitivity to random input, not every error
introduced by matrix arithmetic. Agreement within the source sensitivity
range can support a conditioning-aware deterministic comparison. Downstream
no-worse qualification still needs repeated-run evidence for systematic
differences and additional failures. Source arithmetic need not be bitwise
copied; a large source sensitivity envelope cannot by itself certify a port.

## Evidence and disposition

The standard-library calculation checked every source logZ binary against its
saved hash. That original percentage calculation required no filtering, GPU
work, tuning or threshold selection; the subsequent perturbation runs are
reported separately in the linked result.
Derived data, every per-case percentage, input hashes and calculation code are
under artifacts/controlled-smc-kdm-replication-20260928/likelihood-scale-20261006-01/.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Correct likelihood-scale reporting | Percentages computed from exp(delta logZ), with stages separated | Source hashes verified; no output overwritten | Conditional refits do not test autonomous recursion | Use these figures only when describing the conditional refits | Full TF replication accepted |
| Retain 1e-8 perturbation as calibration hypothesis | Subsequently evaluated in the linked 6 October result | No acceptance margin changed | Perturbation direction, resampling and sample count affect the envelope | Use the completed experiment to localize the unexplained shared offset | Differences under a single observed maximum are acceptable |

| Inference status | Finding |
|---|---|
| Hard veto screen | Prior 32 fits/forwards were valid; strict equality diagnostic failed |
| Statistically supported ranking | None |
| Descriptive-only differences | All percentages in the table |
| Default readiness | Not established |
| Next evidence needed | Localization of the shared offset, then untouched downstream no-worse comparisons |

Post-run review: the terminal-stage summary could conceal early-stage
differences that affect autonomous refinement. Keep both summaries. Small
terminal percentages alone cannot close that gap, while early-stage maxima
alone misdescribe the last-stage comparison.
