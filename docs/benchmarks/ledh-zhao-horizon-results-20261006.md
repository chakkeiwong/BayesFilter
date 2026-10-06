# Marginal LEDH versus ancestor LEDH and original Zhao–Cui references

The marginal weight change alone does not repair the present untuned SIR d18
filter. Even a bootstrap filter with the same 1008 particles is much closer to
the independent high-particle and author-TT likelihood calculations. Predator–
prey likelihood differences are small; the four designs do not establish a
ranking. Both LEDH arms return finite likelihoods and analytical scores at all
requested horizons 10, 20, 40, 50.

This remains an interim result while the SIR rank40 reference and its later
quadratic checks run. The rank20 SIR T50 score repair is complete. All requested
LEDH likelihoods and score coordinates are in [the full numerical tables](../plans/artifacts/ledh-zhao-horizons-20261006-01/committed-evidence-03/completed-comparison.md)
and [CSV](../plans/artifacts/ledh-zhao-horizons-20261006-01/committed-evidence-03/report-009/values-and-scores.csv).
The [plan](../plans/ledh-zhao-horizon-comparison-20261006.md) fixes one length-50
dataset per model, exact prefixes, four paired LEDH designs, N=1008 and GPU
FP64/XLA. Inherited controls are held fixed to isolate the weighting change;
this is not scope-specific tuning or TF32 production validation.

## Actual log likelihoods

The following means use four LEDH designs. The author column uses three
rank20 fits for predator–prey and one rank20 fit for SIR; the separate rank40
SIR check is retained in the full tables and never pooled with rank20.

| Model | T | Ancestor LEDH | Marginal LEDH | Author TT importance calculation | Bootstrap N=131072 |
|---|---:|---:|---:|---:|---:|
| Predator–prey |10| -47.370765 | -47.375320 | -47.404361 | -47.401235 |
| Predator–prey |20| -93.905989 | -93.896008 | -93.905788 | -93.905077 |
| Predator–prey |40| -193.316953 | -193.315618 | -193.302270 | -193.300220 |
| Predator–prey |50| -240.277206 | -240.275854 | -240.236455 | -240.240141 |
| SIR d18 |10| -600.498501 | -579.659131 | -345.662228 | -345.691163 |
| SIR d18 |20| -942.713047 | -921.054190 | -687.653457 | -687.676933 |
| SIR d18 |40| -1602.371361 | -1580.725006 | -1347.240553 | -1347.265592 |
| SIR d18 |50| -1924.199574 | -1902.553778 | -1669.093917 | -1669.110558 |

Full tables report standard errors and all individual designs. At SIR T50,
the LEDH standard errors are 36.623 and 25.898, respectively; the bootstrap
reference SE is 0.0511. The author rank20 conditional path SE is 0.00494, which
excludes fitting error and finite-path bias. At T10 the bootstrap with only 1008
particles gives -346.0071 ±0.3480, compared with -600.4985 ±37.0875 and
-579.6591 ±25.8836 for the two LEDH arms. The early discrepancy persists across
later prefixes. This suggests investigating early cloud/flow/reset behavior;
it does not identify a cause.

For example, the T50 predator–prey score in physical coordinates(r,K,a,s,u,v)
is(13.93236,3.470885,0.069236,-10.31052,-12.91046,15.67558) for ancestor
weights, and(13.83001,3.472606,0.068796,-10.21140,-12.86069,15.61188) for
marginal weights. The author quadratic estimate is
(13.92233,3.481381,0.068990,-10.27191,-12.87451,15.62893), while the high-
particle bootstrap/Fisher estimate is
(14.50089,3.480804,0.069806,-10.27772,-13.03798,15.84019). Agreement of
likelihoods does not remove this reference-score disagreement. Neither
reference is established as an oracle.

At SIR T50, the three score components are derivatives with respect to
(log kappa scale, log nu scale, log observation-noise scale). The means are
(-1624.905, 745.927, 606.639) for ancestor LEDH and
(999.186, -265.064, 286.511) for marginal LEDH. Their between-design standard
errors are approximately (1509, 642, 227) and (1792, 851, 578), respectively.
The completed rank20 author quadratic calculation gives
(108.593, -50.440, -30.385), with conditional path standard errors
(5.157, 2.207, 0.042). The large bootstrap calculation gives
(87.551, -37.113, -30.268), with standard errors (13.7, 3.06, 0.205).
Thus the SIR score discrepancy remains substantial under both weight policies;
the reference disagreement also requires further rank and sampling checks.
The quadratic heldout residual RMS is 1.07e-6 at the smaller predeclared radius;
this checks the local regression fit, not reference bias.

## What was traced and repaired

The immutable author source is pinned to commit
80034dccb99eb1d86284a1839b4a12067d13b9da. The call chain follows the author's
TT fitting and backward inverse-Rosenblatt sampling, with fixed current-model
callbacks explicitly classified as an extension. Source anchors are Zhao–Cui,
JMLR 2024, equation 26 and Algorithm 4, and local
models/full_sol.m:139–206. The sampler's proposal density is the cumulative
path density in proposal_history[:,0]. The author code's returned lml is
mean(log weight), whereas a marginal log likelihood estimate is
logsumexp(log weight)-log N. Both quantities are preserved separately. This
mismatch in the returned diagnostic does not establish that the paper's ESS
experiments are wrong. The nonlinear source route drops density constants;
it is excluded from absolute-likelihood comparisons.

Repairs preserve the full saved-data hash and verify each exact observation
prefix and target. Completed smoothing checkpoints can be recovered after a
later timeout only after validating the completion row, source fingerprint,
path shapes and proposal densities. The incremental checkpoint hook restores
the fitter RNG. Existing matched hook-on/off tests give exact equality for
samples, proposal densities, raw weights, fitter ESS and rank in both models.
The checkpoint writer now accepts both author return formats: the nonlinear
route has no legacy log-evidence field, so it records an explicit unavailable
flag and NaN. Both formats were executed through the generated Octave writer.

Long-horizon quadratic runs exceeded their estimated runtime. Their completed
radii are retained; only missing predeclared radii are retried with the same
paths, random design, 128 training points, 64 heldout points and uncertainty
calculation. Reports group by fitted proposal so a retry cannot inflate the
number of independent fits. Separate job ledgers are combined in the total
172800-second budget, including failures and derivative checks.

The earlier explicit paper/released-driver campaign is documented in
[the publication replication result](zhao-cui-publication-replication-status-20261004.md).
It does not reproduce every published ESS level. Original datasets are
unavailable, and the paper and released drivers differ in dynamics and other
settings. The current matched-target comparison must not be presented as an
exact replication of published numerical values.

## Verification and interpretation

47 focused tests pass. The T50 SIR rank20 quadratic repair also completed with finite heldout diagnostics. The one-component path preserves LGSSM/KSC values and
scores to numerical precision; the multiple-component M13 path matches the
other branch. These engineering checks are recorded with the shared correction
commit b3f4ca646. A smaller N144 finite-difference diagnostic checks the recursive
score of the same finite LEDH program. PP agrees at roughly 2.2e-9 normalized
error. A coarse SIR check was step-sensitive; the subsequent GPU FP64/XLA
check gives maximum normalized errors 7.44e-6 and 6.04e-5 over its fine step
ladder, with exact trace value/coordinate-zero score parity. A moment-safety
branch changes under a small perturbation, so local agreement does not prove
global smoothness. These tests do not establish agreement with the statistical
likelihood score. The monograph compiles to 614 pages, and the three pages of
new equations and evidence were visually inspected.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain shared optional weighting implementation | Parity and mechanics checks pass | No unresolved stable derivative mismatch at checked points | Global branch smoothness | Finish reference comparison | Default or HMC readiness |
| SIR configuration cannot be promoted | Large likelihood discrepancy persists | Same-N bootstrap is much closer | Cause and reference-score accuracy | Investigate early filter behavior after comparison | Marginal weighting is universally ineffective |
| PP remains a diagnostic candidate | Small value differences | References disagree on some scores | Reference bias and only four designs | Additional independent reference validation | Superiority |
| Continue source reference checks | Rank20 values and all requested rank20 scores complete; rank40 partial | Runtime underestimation repaired | Between-fit/rank error | Finish within existing budget | Exact paper-number replication |

| Inference status | Evidence |
|---|---|
| Hard veto screen | All requested LEDH evaluations finite; SIR fails the declared heuristic promotion check. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Marginal versus ancestor means, SE and component errors on one dataset. |
| Default-readiness | Not established; controls are untuned for these scopes. |
| Next evidence needed | Complete rank40/score checks, independent datasets and scope-specific tuning before admission. |

Post-run review: the strongest alternative explanation is error in the common
untuned flow/reset/correction, rather than the importance-weight formula. A
matched independent reference near the LEDH values would overturn the present
SIR likelihood finding. The weakest score evidence is conditional sampling
uncertainty without established reference bias or rank convergence.
