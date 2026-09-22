# A10: robust guide and independent TT coordinates

Status: complete. The exposed numerical failures are repaired, but the current
TT candidates are not promoted: all lose to a cheap proposal in at least one
observation regime, and one four-dimensional reference remains insufficiently
precise. These are downstream filtering comparisons, not an SGQF fit-loss
criterion. They motivate further TT fitting work rather than rejection of TT.
The full protection reduces scalar MSE by 19.0% under the predeclared
exploratory paired analysis; the four-dimensional difference is descriptively
unfavorable and has no admitted interval.
Governing [A10 plan](observation-aware-tt-master-amendment-10-robust-guide-20260916.md),
[master program](observation-aware-tt-repair-complete-program-20260913.md).
The owner authorized mathematical documentation, MathDevMCP audit, a reviewed
plan and execution. This work is an optional extension for exact-importance-
corrected filtering; it does not establish a Zhao--Cui source-faithfulness,
default, production or HMC claim.

## What was repaired

The exposed covariance collapse was real. In A09 d4-s04, essentially all
quadrature weight landed on one node. The resulting covariance was around
1e-31, but its condition number could look harmless because the SPD test
scaled its tolerance by that same tiny covariance. The TT's internal
defensive Gaussian shared the collapsed coordinates and therefore did not
supply an independent physical scale.

The optional implementation now checks guide moments in predictive coordinates,
records signed cancellation and independent scale margins, and compares
successive quadrature rules. Unresolved signed rules trigger positive
Gauss--Hermite integration around the unique SV posterior mode. If that
calculation also remains unresolved, the guide explicitly falls back to its
predictive Gaussian. These are finite numerical resolution checks, not an
accuracy theorem.

The stable-coordinate arm fits TT in charts whose covariance follows
R_t = A R_(t-1) A' + Q, independently of guide posterior covariance.
The full arm adds q_epsilon = (1-epsilon) q_TT + epsilon f in physical space
and evaluates both component densities at the actual selected draw.
The consumer still uses the original SV transition and likelihood divided by
the complete proposal density. At the initial time the proposal is an
analytic Gaussian; each later time has a separate TT fit.

The first confirmation exposed another implementation defect: a least-squares
initial amplitude scale of -0.3106276392630666 was rejected solely for its
sign. A signed amplitude is valid when the represented density uses its
square. The corrected endpoint retains the signed minimizer, checks finite
positive training norm and finite nonzero scale, and leaves positive-scale
arithmetic unchanged. A zero scale remains rejected. See the
[derivation, review and retry record](artifacts/observation-tt-robust-guide-20260916-01/amplitude-sign-repair.md).

## Mathematical and executable evidence

Section 17 of the [manuscript](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex)
contains nine propositions and proofs: the self-scaled SPD counterexample,
signed cancellation, the strictly convex SV mode, positive reweighting,
independent covariance charts, exact mixture importance weighting, a finite
conditional second-moment bound, local mixture smoothness, and signed amplitude
scaling. The density bound is q_epsilon >= epsilon f. It does **not** imply
a minimum realized ESS, uniformly accurate quadrature, or a total derivative
of fitting, adaptive guide selection, resampling and filtering.

[MathDevMCP review](artifacts/observation-tt-robust-guide-20260916-01/math-review.md)
records the actual tool calls, checked targets, corrections and abstentions.
The first equation audit covered eleven equations; the additional audit
selected the amplitude equation. Derivation coverage was incomplete and the
formalization backend did not issue a proof certificate. The direct
proposition/proof review is executor self-review, not independent approval.
The amplitude statement now explicitly requires finite training values and
a finite positive squared-amplitude normalizer. Its Gaussian-reference
defense is homogeneous when the additive mass scales with the TT mass;
global sign invariance holds even for a fixed additive mass.

All 38 focused tests pass in tests-05.log, including actual consumer wiring,
both mixture branches, density/normalizer identities, guide guard behavior,
healthy Gaussian no-fire checks, sign equivalence, seed separation and failure
retention in calibration. The wrong test paths in tests-04.log ran no tests;
they were corrected rather than counted as a pass.

Smoke-03 completed all protected arms on both healthy fixtures and three
exposed cases, with zero inverse-CDF bracket failures and invalid consumer
steps. It reproduced the exact formerly rejected negative scale. On the
earlier exposed collapsed case, smoke-02 changed the minimum chart covariance
from 3.72e-32 in the baseline to .27047 with the repaired guide and .80365
with independent charts. Smokes have no filtering-accuracy reference.

## Frozen comparison and attempt history

Calibration uses three independent T20 sequences in each dimension, N512 and
four particle repetitions. L1 candidates are 1e-5 and .001; external mixture
weights are .05, .2 and .5. The entire calibration-02 selection ledger equals
calibration-01, including reported errors. Guide L1 is 1e-5 in d1 and .001 in
d4; stable L1 is .001 in both; epsilon=.05 in both. All calibration candidates
selected for confirmation pass their calibration admission checks.

Confirmation-01 is preserved: all 24 references passed, but the stable and full
arms were missing on d4-s05 because of the amplitude assertion. It was a
localized fitting failure, not an invalid target or evidence against the
mixture identity. The correction was reviewed before using the remaining
attempt slots. Confirmation-02 uses fresh sequence blocks 48--71, disjoint
from calibration 0--5 and earlier confirmation 24--47; there is no tuning on
the exposed confirmation data.

Filtering error is the mean squared difference from the independent reference
filter mean, normalized coordinatewise by prior variance and averaged over
time and four repetitions. Scalar references compare 801/1201-point grids.
Four-dimensional references compare four repetitions per level along
N=32768,65536,131072, stopping at the first accepted successive-level check.
The predeclared paired sequence bootstrap uses 9999 resamples and Bonferroni
intervals across six dimension/TT contrasts. The non-harm contrast is
MSE_candidate - 1.1 MSE_baseline. Its margin and small-sample intervals are
explicit diagnostic hypotheses, not universal equivalence guarantees.

Numerical runs use escalated RTX 5080 access, verified memory growth,
TensorFlow float64 and XLA numerical kernels. Host setup, CPU guide-rule
eigendecomposition and TT-SVD are recorded exceptions; the full driver is
not claimed to be one XLA graph. Each attempt retains its command, source
hashes, environment, seeds, wall time and logs in
`../benchmarks/artifacts/observation_tt_robust_guide_20260916/`.
The earlier failed attempt's source snapshot remains separate from the fresh
confirmation snapshot. No source, criterion or selected control is changed
during confirmation.

## Fresh confirmation results

All eight methods complete all 24 sequences. All TT arms have zero nonfinite
consumer steps, inverse-CDF bracket failures and log-evidence vetoes. All
scalar references pass. Four-dimensional d4-s10 fails reference precision even
at N=131072: maximum normalized mean MCSE .0236217 exceeds .02; log-evidence
MCSE .0353303 is below .10. Its result is retained and excluded from admitted
accuracy comparisons for **every** method. This limits accuracy evidence; it
is not a TT crash. The stationary-prior comparator also has one log-evidence
veto. No missing case is treated as an improvement.

| d | TT arm | Reference-valid MSE | Mean ESS / 512 | Minimum ESS | Maximum particle weight |
|---|---|---:|---:|---:|---:|
| 1 | A09 baseline | .00187975 | 383.29 | 62.46 | .11931 |
| 1 | Repaired guide | .00175544 | 383.61 | 143.32 | .03838 |
| 1 | Independent charts | .00165335 | 375.14 | 164.18 | .05382 |
| 1 | Charts + physical mixture | .00152322 | 374.61 | 114.69 | .08249 |
| 4 | A09 baseline | .00241227 | 325.01 | 46.18 | .11344 |
| 4 | Repaired guide | .00239518 | 323.52 | 45.74 | .12242 |
| 4 | Independent charts | .00311999 | 304.27 | 2.62 | .61774 |
| 4 | Charts + physical mixture | .00274654 | 303.89 | 14.06 | .26337 |

MSE uses 12 scalar and 11 four-dimensional reference-valid sequences. ESS and
weight diagnostics use all 12 numerically completed sequences per dimension;
their validity does not depend on the independent reference. These continuous
aggregates and extreme values are descriptive unless an interval is stated.
In d4 the full mixture's mean error is 13.86% above A09 on the matched valid
cases; independent charts alone are 29.34% above it. The guide-only difference
is -0.71%. None establishes a d4 ranking. In particular, the physical mixture
does not guarantee a high observed ESS: a minimum of 14.06 remains possible.

The scalar full-minus-baseline MSE interval is
[-.0007837013, -.0000048193]; it excludes zero under the predeclared exploratory
sequence bootstrap. Guide-only and stable-chart difference intervals include
zero. All three scalar repairs pass the declared 10% non-harm comparison;
their non-harm upper endpoints are -.0001587146, -.0001450550 and -.0001699382.
No interval is issued for the d4 comparisons because only 11/12 references
are valid. No ranking among the repaired TT arms is established.

The regime checks explain why the scalar result is not sufficient for
promotion. All three scalar candidates have larger observed error than the
transition proposal on ordinary observations (normalized absolute observations
between .5 and 2). The full arm's conditional mean excess is .00025103.
In d4, guide-only loses to the SGQF joint on near-zero observations
(excess .00004129); independent charts lose to it in all three regimes; the
full mixture loses near zero and in the ordinary regime (excess .00039037
and .00022084). These are predeclared **descriptive promotion vetoes**, not
statistically established heuristic rankings or continuation vetoes.

For scale, on the 11 matched d4 valid cases the transition, SGQF marginal and
SGQF joint errors are .00621658, .00662136 and .00256813. The corresponding
mean ESS values over all completed cases are 181.43, 197.77 and 313.45. The
TT candidates remain useful research candidates under exact importance
correction; beating a Gaussian fit loss is not their objective.

Guide diagnostics retain all unresolved operations. Out of 240 updates per
dimension, d1 uses 233 accepted signed rules and 7 positive Laplace repairs;
d4 uses 160 accepted signed rules, 77 positive repairs and 3 explicit predictive
fallbacks (s01 t7, s03 t2, s09 t7). A resolved numerical comparison is not a
posterior-accuracy certificate. All 1920 full-mixture particle steps respect
the checked density lower bound; its minimum log margin is 1.67e-11. The
observed physical-component draw fraction is .04968 for epsilon=.05.

Observed d4 guide-plus-build time is 18.97 s for baseline, 19.48 s for the
repaired guide, and 21.34 s for independent charts/full defense. Particle time
per N512/T20 repetition is 1.32, 1.10, 1.10 and 1.24 s respectively. The SGQF
joint is 2.62 s to prepare and .24 s per repetition. Full defense reuses the
stable TT fit; its standalone total includes that fit. These measurements
include compilation in a fixed execution order and are not a randomized speed
ranking. The [complete comparison](artifacts/observation-tt-robust-guide-20260916-01/attempt-confirmation-02/comparison.md)
also preserves all-case MSEs, all eight methods and scalar timings.

## Decision and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept localized amplitude repair | Exact positive-branch parity; negative case completes | Zero/nonfinite checks retained | Finite test coverage | Keep corrected endpoint | Improved TT approximation from the sign fix |
| Mechanism repair passes its bounded checks | Exposed collapse rejected; independent charts and actual mixture valid | No TT numerical/likelihood failures in fresh run | Rare failures and unresolved guide fallbacks | Retain optional protected implementation | Universal stability or an ESS floor |
| Scalar full arm is promising | 19.0% lower mean error, exploratory paired interval excludes zero | Ordinary-observation heuristic loss blocks promotion | Twelve sequences, regime heterogeneity | Test mechanism/fit on new calibration and confirmation data under an amendment | Default or unconditional superiority |
| d4 quality unresolved | Guide-only error close to baseline; broad-chart arms descriptively worse | One imprecise reference; conditional heuristic losses | Fixed rank/degree capacity and reference error | First repair reference precision and investigate broader-chart fit capacity | Rejection of TT or the exact mixture identity |

| Inference status | Conclusion |
|---|---|
| Hard veto screen | Zero TT implementation/CDF/nonfinite/log-evidence failures; one reference-precision veto; one stationary-prior log-evidence veto |
| Statistically supported ranking | Only scalar full versus A09 is supported under the declared exploratory bootstrap; no repaired-arm or d4 ranking |
| Descriptive-only differences | ESS, minima, maximum weights, timings, d4 errors and all conditional heuristic differences |
| Default-readiness | Not established; current candidates fail the predeclared conditional comparison screen |
| Next evidence needed | More precise d4 references, scoped calibration of TT capacity in stable coordinates, fresh multi-sequence confirmation and eventual total-derivative validation |

Engineering correctness, numerical validity and scientific performance are
separate here. The math and endpoint tests establish the stated identities;
the smoke establishes that the exposed pathologies no longer break the
protected consumer; fresh filtering determines the bounded performance result.
The strongest alternative explanation for d4 degradation is insufficient
representation or fitting capacity on the broader charts, rather than a
defect in the mixture identity. That explanation is **not checked** by a
capacity ablation. An adequately calibrated capacity study could overturn it.
The weakest statistical evidence is the small sequence count and one reference
failure. The three guide fallbacks also prevent a universal guide-resolution
claim. Continued TT repair is justified, but must enter through the next
reviewed master amendment; no further sweep is launched ad hoc.

## Execution record and closeout

| Attempt | Status | Wall seconds | Interpretation |
|---|---|---:|---|
| smoke-01 | Failed | 150 reserved | One-repetition MCSE/reporting failure; exact time unavailable |
| smoke-02 | Complete | 169.254 | Healthy and exposed covariance cases |
| calibration-01 | Complete | 581.310 | Frozen original selection |
| confirmation-01 | Complete, candidate failure retained | 1211.580 | Stable/full missing one negative-scale fit |
| smoke-03 | Complete | 236.078 | Exact negative-scale replay succeeds |
| calibration-02 | Complete | 551.951 | Entire selection ledger unchanged |
| confirmation-02 | Complete | 1291.996 | Fresh final evidence reported above |

Numerical time is 4042.169 measured seconds plus the 150-second failed-smoke
reservation: 4192.169 s, below the 18000-second numerical cap. All three smoke,
two calibration and two confirmation slots are consumed. This is a subset
of elapsed A10 work and is not charged twice. Source hashes match the archived
run snapshots, and calibration/confirmation sequence blocks are disjoint; see
[terminal checks](artifacts/observation-tt-robust-guide-20260916-01/attempt-confirmation-02/terminal-checks.json).
The [terminal review](artifacts/observation-tt-robust-guide-20260916-01/result-review.md)
accepts this completed campaign record without promoting a filtering default.
The final 64-page [PDF](artifacts/observation-tt-robust-guide-20260916-01/latex/observation-aware-tt-a10.pdf)
builds cleanly; the new rendered pages were visually inspected. The protected
manuscript comparison has one 610-line insertion and no removed or replaced
baseline lines. MathDevMCP coverage remains partial, as documented above.

The sole [H11 ledger](artifacts/observation-tt-continuation-24h-20260915-01/budget.json)
closed A10 at 2026-09-16T18:24:33Z. It charges 20193 elapsed seconds plus
120 seconds for closeout, totaling 20313 seconds within the 28800-second
active ceiling. All numerical time and tool/approval waits are included once.
Prior reservations of 19602.194975 seconds remain unchanged.
The conservative available balance is **20707.892609 seconds
(5 hours 45 minutes)**. The master is current; the next amendment
must address reference precision and broader-chart fitting before further
research runs.
